"""Restore a recorded dataset state (qpos, qvel) into a live OGBench cube env and check the restore.

obs[18] (contact force) needs mj_rnePostConstraint after mj_forward. Gripper action +1 closes.
Settling steps the env with zero arm action so the effector holds its pose.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from scripts.probes.cube_bottleneck.phases import PhaseThresholds

# qpos slice of the cube's free joint
CUBE_QPOS = slice(14, 17)
# Observation columns
CONTACT_IDX = 18
CUBE_OBS = slice(19, 22)
OBS_XYZ_SCALER = 10.0
GRIPPER_SCALER = 3.0

# Two settle steps restore the contact reading; more steps only add drift
N_SETTLE = 2
G2_DRIFT_M = 0.04  # max cube drift after restore, m
G2_CONTACT_MIN = 0.5  # min contact reading for a held state


def make_env(env_name: str, seed: int = 0):
    """Make an OGBench env without loading the dataset."""
    import ogbench

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env.reset(seed=seed, options=dict(task_id=1, render_goal=False))
    return env


def ensure_cube_env(env) -> None:
    u = getattr(env, "unwrapped", None)
    missing = [
        a
        for a in (
            "_data",
            "_model",
            "_pinch_site_id",
            "cur_task_info",
            "_gripper_actuator_ids",
            "set_state",
            "compute_observation",
        )
        if u is None or not hasattr(u, a)
    ]
    if missing:
        raise RuntimeError(
            f"cube_bottleneck needs an OGBench cube manip env; env.unwrapped lacks {missing}"
        )


def forward(env) -> None:
    """Run mj_forward and mj_rnePostConstraint so contact forces are filled."""
    import mujoco

    u = env.unwrapped
    mujoco.mj_forward(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)


def observation(env) -> np.ndarray:
    return np.asarray(env.unwrapped.compute_observation(), np.float32)


def set_state(env, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Put the sim at (qpos, qvel) and return the resulting observation."""
    u = env.unwrapped
    u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    forward(env)
    return observation(env)


def settle(env, n_settle: int = N_SETTLE, a_grip: float = 1.0) -> np.ndarray:
    """Hold pose for `n_settle` env steps with gripper action `a_grip` (+1 closes)."""
    act = np.zeros(env.action_space.shape, np.float32)
    act[-1] = a_grip
    ob = None
    for _ in range(n_settle):
        ob, *_ = env.step(act)
    forward(env)
    return observation(env) if ob is None else np.asarray(ob, np.float32)


def restore(
    env,
    task_id: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    n_settle: int = N_SETTLE,
    a_grip: float = 1.0,
) -> np.ndarray:
    """Reset to `task_id`, set state, settle; return the observation."""
    env.reset(options=dict(task_id=task_id, render_goal=False))
    set_state(env, qpos, qvel)
    if n_settle > 0:
        settle(env, n_settle=n_settle, a_grip=a_grip)
    return observation(env)


# Restore fidelity check


@dataclass
class G2Result:
    held: bool  # lifted and gripped after one zero-action step
    in_contact: bool  # obs[18] > 0.5 before the step
    drift_m: float  # cube displacement from the recorded qpos
    lift: float  # cube height above the task's table_z
    grip: float  # gripper_opening in [0, 1]
    contact: float  # obs[18]
    passes: bool

    def as_row(self) -> Dict[str, Any]:
        return dict(
            held=self.held,
            in_contact=self.in_contact,
            drift_m=self.drift_m,
            lift=self.lift,
            grip=self.grip,
            contact=self.contact,
            passes=self.passes,
        )


def g2_check(
    env, qpos_ref: np.ndarray, *, thr: Optional[PhaseThresholds] = None
) -> G2Result:
    """Check a restored env: contact > 0.5 now, then held and drift < 0.04 m after one zero-action step."""
    thr = thr or PhaseThresholds()
    u = env.unwrapped

    # Read contact before the step; a zero action relaxes the grip
    contact = float(observation(env)[CONTACT_IDX])
    in_contact = bool(contact > G2_CONTACT_MIN)

    env.step(np.zeros(env.action_space.shape, np.float32))
    forward(env)

    ob = observation(env)
    cube = np.asarray(u._data.qpos[CUBE_QPOS], np.float64)
    drift = float(np.linalg.norm(cube - np.asarray(qpos_ref, np.float64)[CUBE_QPOS]))
    table_z = float(
        u.cur_task_info["init_xyzs"][int(getattr(u, "_target_block", 0) or 0)][2]
    )
    lift = float(cube[2] - table_z)
    grip = float(ob[17] / GRIPPER_SCALER)
    held = bool(lift > thr.delta_lift and grip > thr.tau_grip)
    return G2Result(
        held=held,
        in_contact=in_contact,
        drift_m=drift,
        lift=lift,
        grip=grip,
        contact=contact,
        passes=bool(held and in_contact and drift < G2_DRIFT_M),
    )


# Gripper convention check


def determine_gripper_convention(
    env, rows: Sequence[Dict[str, np.ndarray]], task_id: int = 1
) -> Dict[str, Any]:
    """Restore held states under raw ctrl 0/255 and action +1/-1, and report which holds the cube."""
    import mujoco

    out = {}
    for label, kind, value in (
        ("action_close(+1)", "action", +1.0),
        ("action_open(-1)", "action", -1.0),
        ("ctrl_255", "ctrl", 255.0),
        ("ctrl_0", "ctrl", 0.0),
    ):
        res = []
        for r in rows:
            env.reset(options=dict(task_id=task_id, render_goal=False))
            set_state(env, r["qpos"], r["qvel"])
            u = env.unwrapped
            if kind == "action":
                settle(env, a_grip=value)
            else:
                u._data.ctrl[u._gripper_actuator_ids] = value
                for _ in range(N_SETTLE):
                    mujoco.mj_step(u._model, u._data)
                forward(env)
            res.append(g2_check(env, r["qpos"]))
        out[label] = dict(
            held_rate=float(np.mean([x.held for x in res])),
            pass_rate=float(np.mean([x.passes for x in res])),
            median_drift_m=float(np.median([x.drift_m for x in res])),
            median_grip=float(np.median([x.grip for x in res])),
            median_contact=float(np.median([x.contact for x in res])),
        )
    return out
