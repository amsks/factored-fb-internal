"""Spec 23 P0 -- thin dm_control eval adapter for ExoRL `point_mass_maze`,
parallel to (never editing) the vendored OGBench eval loop
(`third_party/ogbench/impls/utils/evaluation.py`). Does not import
`humenv`/`metamotivo` -- see spec 23 section 4.2 / section 7.

Obs layout (VERIFIED against real downloaded data AND a real env
instantiation, 2026-08-11 -- see task-report.md): 4-dim
`[x, y, vx, vy]` = position (dims 0:2) then velocity (dims 2:4), matching
the block the spec's hit_spec `0:2:<eps>` assumes. Episode length 1000
(20s / 0.02s control timestep, action_repeat=1, matching the ExoRL
collector and the recorded episode files).

Goal-success definition (P0, spec 23 section 4.4): success = the point
mass ever comes within `GOAL_EPS` of the goal (x, y) at any point in the
episode ("any-time proximity"). `GOAL_EPS = 0.03` is the `target_size` used
by the published FB zero-shot-RL maze protocol's `MazeMultiGoal` reward
(20 goals, 5 per room x 4 rooms) -- see `MAZE_20_GOALS` / `GOAL_EPS` below
for the exact source. Note: the published protocol itself scores maze
goals with a *dense* tolerance-shaped reward summed over the episode, not
a hard binary threshold; our binary "success" is a derived metric that
reuses the SAME full-credit radius as that dense reward's `bounds=(0,
target_size)`, not a value copied from a stated binary-success number
(the upstream repo does not report one). This is stated explicitly here
per spec 23 section 4.4's requirement to source or justify epsilon.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from exorl_dmc_tasks import point_mass_maze as _pmm  # noqa: E402

OBS_DIM = 4
ACTION_DIM = 2
EPISODE_LEN = 1000  # 20s / 0.02s control timestep, action_repeat=1

# Source: url_benchmark/goals.py, class MazeMultiGoal, in
# https://github.com/facebookresearch/controllable_agent
# @ 5a9950b07f1edcb4bddd04f5819379541793d975 (2023-01-03) -- the official
# reference implementation for "Does Zero-Shot Reinforcement Learning
# Exist?" (arXiv 2209.14935), which spec 23 section 8 cites for the
# "maze (20 goals)" claim. Verified 2026-08-11 by cloning that commit and
# reading goals.py directly (not from a paper PDF, which failed to extract
# cleanly -- see task-report.md). These are DATA (goal coordinates + the
# tolerance radius), not code, reproduced here as constants.
MAZE_20_GOALS = np.array(
    [
        [-0.15, 0.15], [-0.22, 0.22], [-0.08, 0.08], [-0.22, 0.08], [-0.08, 0.22],   # room 1: top left
        [0.15, 0.15], [0.22, 0.22], [0.08, 0.08], [0.22, 0.08], [0.08, 0.22],        # room 2: top right
        [-0.15, -0.15], [-0.22, -0.22], [-0.08, -0.08], [-0.22, -0.08], [-0.08, -0.22],  # room 3: bottom left
        [0.15, -0.15], [0.22, -0.22], [0.08, -0.08], [0.22, -0.08], [0.08, -0.22],       # room 4: bottom right
    ],
    dtype=np.float64,
)
assert MAZE_20_GOALS.shape == (20, 2)
GOAL_EPS = 0.03  # MazeMultiGoal.from_goal's `target_size`


class PointMassMazeEnv:
    """Minimal gym-like wrapper around the vendored dm_control task.

    reset() -> obs (float32, shape (4,))
    step(action) -> (obs, terminated, truncated, info)  [no reward -- P0
        computes goal proximity directly from obs, per spec 23 4.4; a
        future zero-shot reward readout consumes `info['position']`]
    """

    def __init__(self, task: str = "reach_top_left", seed: int | None = None):
        if not hasattr(_pmm, task):
            raise ValueError(f"unknown point_mass_maze task {task!r}")
        self._make = getattr(_pmm, task)
        self._seed = seed
        self._env = self._make(random=seed)
        self._t = 0

    @staticmethod
    def _flatten(obs_dict) -> np.ndarray:
        # dm_control flat_observation=True concatenates OrderedDict values
        # in insertion order; MultiTaskPointMassMaze.get_observation emits
        # position then velocity (verified against the real env, see
        # task-report.md), so this is NOT dependent on dict iteration order
        # matching flat_observation's wrapper internals -- we do the
        # concatenation ourselves, explicitly, in the verified order.
        return np.concatenate(
            [np.asarray(obs_dict["position"]), np.asarray(obs_dict["velocity"])]
        ).astype(np.float32)

    def reset(self) -> np.ndarray:
        self._env = self._make(random=self._seed)
        ts = self._env.reset()
        self._t = 0
        return self._flatten(ts.observation)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ts = self._env.step(action)
        self._t += 1
        obs = self._flatten(ts.observation)
        truncated = bool(ts.last())
        info = {"position": obs[:2].copy(), "velocity": obs[2:].copy(), "t": self._t}
        return obs, False, truncated, info

    @property
    def action_spec(self):
        return self._env.action_spec()


def random_policy(action_dim: int = ACTION_DIM, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()

    def _policy(_obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

    return _policy


@dataclass
class RolloutResult:
    observations: np.ndarray  # (T+1, obs_dim)
    actions: np.ndarray  # (T, action_dim)
    distances: np.ndarray  # (T+1,) distance to goal at each obs
    min_distance: float
    success_any_time: bool
    success_final: bool
    length: int


def rollout(
    env: PointMassMazeEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    goal_xy: np.ndarray,
    eps: float = GOAL_EPS,
    horizon: int = EPISODE_LEN,
) -> RolloutResult:
    """Roll out `policy` for up to `horizon` steps and score goal proximity
    against `goal_xy` under the P0 success definition (section 4.4):
    success_any_time = ever within eps; success_final = within eps at the
    last step."""
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    obs = env.reset()
    obs_list = [obs]
    act_list = []
    dist_list = [float(np.linalg.norm(obs[:2] - goal_xy))]
    for _ in range(horizon):
        a = policy(obs)
        obs, terminated, truncated, _info = env.step(a)
        obs_list.append(obs)
        act_list.append(a)
        dist_list.append(float(np.linalg.norm(obs[:2] - goal_xy)))
        if terminated or truncated:
            break
    distances = np.array(dist_list, dtype=np.float64)
    return RolloutResult(
        observations=np.stack(obs_list, axis=0),
        actions=np.stack(act_list, axis=0) if act_list else np.zeros((0, ACTION_DIM), dtype=np.float32),
        distances=distances,
        min_distance=float(distances.min()),
        success_any_time=bool((distances <= eps).any()),
        success_final=bool(distances[-1] <= eps),
        length=len(act_list),
    )
