"""evals/phase_probe.py — test where FB fails in a cube episode.

M1: classify normal rollouts into reach/secure/success phases.
M2: counterfactual start states (S0 baseline / S1 skip-reach /
S2 pre-grasped) to isolate the grasp bottleneck.

See docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md.
Pure logic here is unit-tested; MuJoCo physics is validated by a
documented Linux acceptance run. No training/eval code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import numpy as np

PHASE_ORDER = ["none", "reached", "secured", "success"]
N_SETTLE = 5  # MuJoCo steps to clamp the cube in S2 (module constant; YAGNI)


@dataclass
class Thresholds:
    eps_reach: float = 0.06   # m, hand-to-cube distance for "reached"
    delta_lift: float = 0.03  # m, cube height above table for "lifted"
    k_steps: int = 5          # consecutive steps the lift must hold
    tau_grip: float = 0.5     # gripper_opening > tau => "closed"


def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify_phases(signals: Dict[str, Any], thr: Thresholds) -> Dict[str, Any]:
    """Classify ONE episode's per-step signals into phase milestones.

    signals: eff[T,3], cube[T,3], grip[T], goal[3], success(bool),
             table_z(float).
    Returns the milestones, monotone furthest_phase/fail_phase, and
    diagnostics. Handles T == 0.
    """
    eff = np.asarray(signals["eff"], dtype=np.float64).reshape(-1, 3)
    cube = np.asarray(signals["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(signals["grip"], dtype=np.float64).reshape(-1)
    goal = np.asarray(signals["goal"], dtype=np.float64).reshape(3)
    table_z = float(signals["table_z"])
    success = bool(signals["success"])
    T = eff.shape[0]

    if T == 0:
        return {
            "reached": success, "secured": success, "success": success,
            "reached_raw": False, "secured_raw": False,
            "furthest_phase": "success" if success else "none",
            "fail_phase": "none" if success else "reach",
            "min_eff_cube_dist": float("nan"),
            "max_cube_lift": float("nan"),
            "final_cube_goal_dist": float("nan"),
            "length": 0,
        }

    eff_cube_dist = np.linalg.norm(eff - cube, axis=1)
    eff_cube_xy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - table_z

    reached_raw = bool(eff_cube_dist.min() < thr.eps_reach)
    secured_mask = (
        (lift > thr.delta_lift)
        & (grip > thr.tau_grip)
        & (eff_cube_xy < thr.eps_reach)
    )
    secured_raw = bool(_max_true_run(secured_mask) >= thr.k_steps)

    # Monotone ladder: success => secured => reached.
    secured = secured_raw or success
    reached = reached_raw or secured

    if success:
        furthest, fail = "success", "none"
    elif secured:
        furthest, fail = "secured", "transport"
    elif reached:
        furthest, fail = "reached", "grasp"
    else:
        furthest, fail = "none", "reach"

    return {
        "reached": reached, "secured": secured, "success": success,
        "reached_raw": reached_raw, "secured_raw": secured_raw,
        "furthest_phase": furthest, "fail_phase": fail,
        "min_eff_cube_dist": float(eff_cube_dist.min()),
        "max_cube_lift": float(lift.max()),
        "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal)),
        "length": int(T),
    }


def step_signals(info: Dict[str, Any]) -> Dict[str, Any]:
    """Per-step signals from an OGBench manipspace `info` dict."""
    return {
        "eff": np.asarray(info["proprio/effector_pos"], dtype=np.float64),
        "cube": np.asarray(info["privileged/block_0_pos"], dtype=np.float64),
        "grip": float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]),
        "success": bool(info.get("success", False)),
    }


def _target_block(env) -> int:
    return int(getattr(env.unwrapped, "_target_block", 0) or 0)


def episode_goal(env) -> np.ndarray:
    tb = _target_block(env)
    return np.asarray(
        env.unwrapped.cur_task_info["goal_xyzs"][tb], dtype=np.float64
    )


def episode_table_z(env) -> float:
    tb = _target_block(env)
    return float(env.unwrapped.cur_task_info["init_xyzs"][tb][2])


def ensure_manip_env(env) -> None:
    """Raise a clear error if this is not an OGBench cube manip env."""
    u = getattr(env, "unwrapped", None)
    if u is None or not all(
        hasattr(u, a) for a in ("_data", "_pinch_site_id", "cur_task_info")
    ):
        raise RuntimeError(
            "phase_probe is cube-only: env.unwrapped lacks manipspace "
            "handles (_data/_pinch_site_id/cur_task_info)."
        )


SCENARIOS = ["S0", "S1", "S2"]


def apply_scenario(env, scenario: str) -> None:
    """Mutate post-reset MuJoCo state for a counterfactual start.

    S0: no-op. S1: cube under the hand at table height, gripper open.
    S2: cube at the effector, gripper closed, settle N_SETTLE steps so
    the fingers physically clamp it (no weld). The task goal/target
    mocap is never touched.
    """
    if scenario == "S0":
        return

    u = env.unwrapped
    model, data = u._model, u._data
    eff = np.asarray(data.site_xpos[u._pinch_site_id], dtype=np.float64).copy()
    joint = data.joint("object_joint_0")

    if scenario == "S1":
        table_z = episode_table_z(env)
        joint.qpos[0] = eff[0]
        joint.qpos[1] = eff[1]
        joint.qpos[2] = table_z
        mujoco.mj_forward(model, data)
        return

    if scenario == "S2":
        joint.qpos[:3] = eff
        mujoco.mj_forward(model, data)
        data.ctrl[u._gripper_actuator_ids] = 255.0
        for _ in range(N_SETTLE):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return

    raise ValueError(f"unknown scenario {scenario!r} (expected one of {SCENARIOS})")


def _initial_obs(env):
    """Initial observation for the post-reset (post-scenario) state.

    The caller already did exactly one env.reset() this episode; calling
    env.reset() again would re-randomize the task. OGBench manip envs
    expose compute_observation() on the unwrapped env, so use that;
    fall back to a fresh reset's observation only if it is unavailable.
    """
    u = env.unwrapped
    if hasattr(u, "compute_observation"):
        return np.asarray(u.compute_observation(), dtype=np.float32)
    obs, _ = env.reset()
    return np.asarray(obs, dtype=np.float32)


def rollout_with_phase_signals(
    env, agent, z, n_episodes: int, thr: Thresholds, scenario: str = "S0",
) -> List[Dict[str, Any]]:
    """Roll out n_episodes under `scenario`, recording per-step signals.

    Each episode: one env.reset() -> apply_scenario -> run policy.
    `success` is OR-accumulated from per-step info (same convention as
    evals/ogbench.py). Returns one signals dict per episode suitable for
    classify_phases.
    """
    import torch

    ensure_manip_env(env)
    episodes: List[Dict[str, Any]] = []
    for _ in range(n_episodes):
        _, info = env.reset()
        apply_scenario(env, scenario)
        goal = episode_goal(env)
        table_z = episode_table_z(env)

        effs, cubes, grips = [], [], []
        s0 = step_signals(info)
        effs.append(s0["eff"]); cubes.append(s0["cube"]); grips.append(s0["grip"])
        ep_success = bool(s0["success"])

        observation = _initial_obs(env)
        while True:
            a = agent.act(
                obs=torch.tensor(np.asarray(observation), device=agent.device,
                                 dtype=torch.float32)[None],
                z=z,
            ).cpu().numpy()[0]
            observation, _, terminated, truncated, info = env.step(a)
            s = step_signals(info)
            effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])
            ep_success = ep_success or s["success"]
            if terminated or truncated:
                break

        episodes.append({
            "eff": np.asarray(effs, dtype=np.float64),
            "cube": np.asarray(cubes, dtype=np.float64),
            "grip": np.asarray(grips, dtype=np.float64),
            "goal": goal,
            "success": bool(ep_success),
            "table_z": table_z,
            "length": len(effs),
        })
    return episodes
