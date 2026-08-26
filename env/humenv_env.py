"""env/humenv_env.py — the HumEnv (Meta Motivo SMPL humanoid) eval env.

Deliberately much smaller than env/exorl_env.py, because HumEnv needs no loader here: the
offline data is Meta Motivo's released replay buffer, which scripts/convert_humenv_buffer.py
rewrites into the `buffer/episode_*.npz` layout `env/unified.py::load_unified` already reads.
The domain yaml points `data_dir` at that tree, so this module only has to build the env.

WHAT THIS ENV DOES *NOT* SCORE — read before interpreting any eval.csv row from a hum- run
-------------------------------------------------------------------------------------------
HumEnv's goal-reaching benchmark is scored by `humenv.bench.GoalEvaluation`, not by a reward
the env returns. Two consequences, both deliberate and neither papered over:

  * `task=None` (the setting GoalEvaluation itself uses), so **every step returns reward
    0.0**. A hum- run's `episode_return` column is therefore a constant 0 and means nothing.
    Do not read it as performance.
  * No `success` key is emitted. Same rule as ExoRLWrapper: HumEnv has no binary success
    flag of its own, and fabricating a 0.0 would read as "the policy failed" in eval.csv
    rather than "this domain has no such metric". The adapter's `saw_success` logic then
    correctly omits the column.

The real metric is `humenv.bench.utils.metrics.get_episode_goal_stats`: L2 over the pose
block `obs[:214]` against the goal pose, `success` = inside radius 2.0 at any point,
`proximity` = time-average of a shaped indicator ramping linearly from 1.0 at d=2.0 to 0.0 at
d=4.0. Scoring a checkpoint that way needs the agent wrapped behind `act(obs, z)` +
`goal_inference(goal)`, and the `humenv[bench]` extra (torch, pot) which lives in the
separate humenv venv -- so it is a post-hoc pass over checkpoints, the same shape as
scripts/reeval_checkpoint.py, not something this training-time env can do.

Set EVAL_INTERVAL high on hum- training runs. The per-eval cost buys nothing.

`state_init="Fall"` matches GoalEvaluation's own env_kwargs: the humanoid starts from a
fallen state and has to get up into the goal pose. Training-time rollouts use the same
initialisation so that a checkpoint's behaviour here and under the bench are comparable.
"""
import os

import numpy as np

# HumEnv's episode length: 30 Hz decisions, 300 steps. Confirmed against the released
# buffer -- 16,600 of its 16,650 episodes are exactly 300 steps (the other 50 are 400).
EPISODE_LEN = 300

OBS_DIM = 358
ACT_DIM = 69
# The pose block the benchmark's distance is defined over (metrics.py::distance_smpl uses
# `next_obs[..., :214]`). Carried here so a domain yaml's measure_hit_spec and the metric
# cannot silently disagree about which columns "the pose" means.
POSE_BLOCK = (0, 214)


class HumEnvWrapper:
    """Presents HumEnv through the same 5-tuple gymnasium API the unified adapter drives,
    with observations cast to float32 so they match the converted offline data exactly
    (HumEnv's own space is float64; the npz tree is float32)."""

    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return np.asarray(obs, np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self._env.step(np.asarray(action, np.float64))
        return np.asarray(obs, np.float32), float(r), bool(term), bool(trunc), info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):          # unwrapped, physics, etc.
        return getattr(self._env, name)


def make_humenv_env(task, seed=0, state_init="Fall"):
    """-> (env, horizon) for a `hum-` task slug. Only one domain exists (the SMPL humanoid),
    so `task` is validated rather than dispatched on."""
    if task != "hum-humanoid":
        raise ValueError(f"unknown HumEnv task {task!r} (known: 'hum-humanoid')")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium
    from humenv import make_humenv

    env, mp_info = make_humenv(
        num_envs=1,
        vectorization_mode="sync",
        wrappers=[gymnasium.wrappers.FlattenObservation],
        state_init=state_init,
    )
    if mp_info is not None and "manager" in mp_info:
        # num_envs=1 takes the non-vectorised path, so this should not fire; shut the
        # manager down rather than leak it if a future humenv changes that.
        mp_info["manager"].shutdown()
    return HumEnvWrapper(env), EPISODE_LEN
