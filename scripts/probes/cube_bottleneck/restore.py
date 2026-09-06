"""Restore a recorded dataset state into a live OGBench cube env, and gate the restore.

Spec section 2.2. Three facts this module pins down, each established from the ogbench
source in third_party and re-checked empirically by `determine_gripper_convention`:

1. **The observation is a pure function of (qpos, qvel).**
   `cube_env.compute_observation` reads only `_data` (qpos, qvel, site_xpos, cfrc_ext); the
   `prev_qpos`/`prev_qvel` that `compute_ob_info` also carries never enter the 28-dim state
   vector. So set_state + forward reconstructs the observation exactly.

2. **There is no cross-step IK integrator.**
   `manipspace_env.set_control` recomputes `effector_pos`, `effector_yaw` and
   `gripper_opening` from `self._data` on every call, assigns `_target_effector_pose`, and
   consumes it inside that same call (`T_wa = self._target_effector_pose @ self._T_pa`).
   Nothing carries it across steps. `[torch] scripts/atlas/prefix_replay.py`'s docstring
   ("the env holds an IK-target integrator outside mjData") does not hold for this build's
   control path, which is why snapshot/restore is available to us where prefix-replay was
   the only option there. Spec step 3 is therefore a no-op; `settle()` still verifies it by
   construction, since a stale hidden target would show up as G2 drift.

3. **`mj_forward` alone is not enough.**
   `obs[18]` is `||cfrc_ext(right_pad)|| / 50` -- the constraint forces are not filled by
   `mj_forward`, so the contact channel reads stale/zero without `mj_rnePostConstraint`.
   `[torch] scripts/atlas/gates.py:18-19` restores with exactly this pair; we match it. A
   restore that skipped it would silently report "no contact" at every held state.

**Gripper convention.** `ctrl[gripper] = 255 * target_gripper_opening` and
`gripper_opening = qpos[right_driver_joint] / 0.8` clipped to [0, 1], so the channel counts
DRIVER travel: it rises as the Robotiq closes. `255` is the closed extreme, which is what
`[torch] evals/phase_probe.py` S2 clamps to and what `phase_probe`'s `tau_grip = 0.5`
("closed") reads. `[torch] scripts/value/counterfactual_value_probe.py:165`'s
`ctrl = 0.0 # establish the grasp` commands the gripper fully OPEN and is inverted.
`determine_gripper_convention` re-derives this from the running env rather than asserting it.

**Settling uses env actions, not raw ctrl.** After `set_state` the arm actuators still hold
the ctrl targets the *reset* pose left behind, so raw `mj_step` settling yanks the arm from
the restored pose back toward home -- drift that has nothing to do with the grasp. Stepping
the env with action `[0, 0, 0, 0, a_grip]` instead routes through `set_control`, whose
positional target is `effector_pos + 0` (i.e. hold where you are) while `a_grip` drives the
gripper. Same physics, same controller the policy will use, no stale-target artefact.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from scripts.probes.cube_bottleneck.phases import PhaseThresholds

# qpos slice of the target cube's free joint (matches [torch] scripts/atlas/config.py).
CUBE_QPOS = slice(14, 17)
# obs columns (cube_env.compute_observation, state branch): see module docstring of
# [torch] scripts/atlas/oracle.py.
CONTACT_IDX = 18
CUBE_OBS = slice(19, 22)
OBS_XYZ_SCALER = 10.0
GRIPPER_SCALER = 3.0

# Spec section 2.2 step 4 says "N_SETTLE = 5". MEASURED on the cube-single held bank
# (64 rows), the contact channel is fully rebuilt after TWO steps and nothing improves
# after that, while drift grows linearly with every extra step:
#
#   settle   contact_p50   drift_p95   G2 pass
#        0        0.163      0.0046      1.000   <- contact contradicts a held cube
#        2        0.964      0.0088      1.000   <- recorded contact is 0.963
#        5        0.963      0.0149      1.000
#       20        0.962      0.0466      0.797   <- drift crosses 0.04
#       40        0.963      0.0886      0.000
#
# So 2 is the setting that buys the whole benefit at the least cost, and it is what H1 uses.
# `phases.N_SETTLE` stays 5: that is the vendored constant of the S2 clamp, and H2 ports S2
# VERBATIM, so the two must not be unified.
N_SETTLE = 2
G2_DRIFT_M = 0.04      # [torch] scripts/atlas/gates.py dataset_state_restore_drift
G2_CONTACT_MIN = 0.5   # the bank's own contact criterion; see g2_check


def make_env(env_name: str, seed: int = 0):
    """A bare OGBench env, without pulling the 1M-transition dataset along.

    `ogbench.make_env_and_datasets` loads the npz; the probes that only need physics
    (restore gates, counterfactual rollouts) do not want that cost per worker.
    """
    import ogbench

    # `env_only=True` is ogbench's own switch for exactly this: it maps the DATASET name
    # ("cube-single-play-v0") to the ENV id ("cube-single-v0") with the same code the
    # training path uses, and skips the npz load. Reimplementing that mapping here would be
    # one more place for the two to drift apart.
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env.reset(seed=seed, options=dict(task_id=1, render_goal=False))
    return env


def ensure_cube_env(env) -> None:
    u = getattr(env, "unwrapped", None)
    missing = [a for a in ("_data", "_model", "_pinch_site_id", "cur_task_info",
                           "_gripper_actuator_ids", "set_state", "compute_observation")
               if u is None or not hasattr(u, a)]
    if missing:
        raise RuntimeError(
            f"cube_bottleneck needs an OGBench cube manip env; env.unwrapped lacks {missing}")


def forward(env) -> None:
    """`mj_forward` + `mj_rnePostConstraint`: the pair that makes obs[18] real."""
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
    """Hold pose for `n_settle` env steps, commanding the gripper with `a_grip`.

    `a_grip = +1` closes (see module docstring). The positional channels are zero, which
    `set_control` reads as "target = current effector pose".
    """
    act = np.zeros(env.action_space.shape, np.float32)
    act[-1] = a_grip
    ob = None
    for _ in range(n_settle):
        ob, *_ = env.step(act)
    forward(env)
    return observation(env) if ob is None else np.asarray(ob, np.float32)


def restore(env, task_id: int, qpos: np.ndarray, qvel: np.ndarray,
            *, n_settle: int = N_SETTLE, a_grip: float = 1.0) -> np.ndarray:
    """The full spec-2.2 recipe. Returns the observation the policy takes over from."""
    env.reset(options=dict(task_id=task_id, render_goal=False))
    set_state(env, qpos, qvel)
    if n_settle > 0:
        settle(env, n_settle=n_settle, a_grip=a_grip)
    return observation(env)


# ── G2: restore fidelity ────────────────────────────────────────────────────────────────

@dataclass
class G2Result:
    held: bool           # phase label is `held` (lifted + gripped) after one zero-action step
    in_contact: bool     # obs[18] at HAND-OVER > 0.5 -- latent-drop guard, see g2_check
    drift_m: float       # cube displacement from the recorded qpos
    lift: float          # cube height above the task's table_z
    grip: float          # gripper_opening in [0, 1]
    contact: float       # obs[18]
    passes: bool

    def as_row(self) -> Dict[str, Any]:
        return dict(held=self.held, in_contact=self.in_contact, drift_m=self.drift_m,
                    lift=self.lift, grip=self.grip, contact=self.contact, passes=self.passes)


def g2_check(env, qpos_ref: np.ndarray, *, thr: Optional[PhaseThresholds] = None) -> G2Result:
    """Spec gate G2, evaluated on the ALREADY-restored env.

    One further zero-action step (the spec's "one further step with the zero action"), then:
    the phase label must be `held` -- `transport_mask`'s two conditions, lift > delta_lift
    and grip > tau_grip -- and the cube must sit within 0.04 m of where it was recorded.

    **Contact is checked too, beyond the spec.** `transport_mask` is pure geometry (lift and
    driver travel), so a restore whose fingers have gone slack still passes it for the few
    steps the cube takes to fall. That is not hypothetical: commanding the gripper with raw
    `ctrl = 0` (the value `[torch] scripts/value/counterfactual_value_probe.py:165` uses to
    "establish the grasp") scores held = 1.00 while contact collapses to 0.01 -- a latent
    drop that would have entered the bank silently. Requiring obs[18] > 0.5, the same
    criterion the bank selected on, closes that hole.
    """
    thr = thr or PhaseThresholds()
    u = env.unwrapped

    # Contact is read at HAND-OVER -- the state the policy is actually given -- not after
    # the probe step below. A zero action re-commands the gripper to its CURRENT opening
    # (`target_gripper_opening = gripper_opening + 0`), i.e. ctrl ~ 140 rather than the 255
    # that was squeezing, so the measured contact force relaxes from 0.96 to ~0.29 within
    # one step without anything being wrong. Judging the restore on the post-step value
    # would reject every correctly restored row (it rejected 64/64).
    contact = float(observation(env)[CONTACT_IDX])
    in_contact = bool(contact > G2_CONTACT_MIN)

    env.step(np.zeros(env.action_space.shape, np.float32))
    forward(env)

    ob = observation(env)
    cube = np.asarray(u._data.qpos[CUBE_QPOS], np.float64)
    drift = float(np.linalg.norm(cube - np.asarray(qpos_ref, np.float64)[CUBE_QPOS]))
    table_z = float(u.cur_task_info["init_xyzs"][int(getattr(u, "_target_block", 0) or 0)][2])
    lift = float(cube[2] - table_z)
    grip = float(ob[17] / GRIPPER_SCALER)
    held = bool(lift > thr.delta_lift and grip > thr.tau_grip)
    return G2Result(held=held, in_contact=in_contact, drift_m=drift, lift=lift, grip=grip,
                    contact=contact,
                    passes=bool(held and in_contact and drift < G2_DRIFT_M))


# ── the empirical gripper determination the spec asks for ───────────────────────────────

def determine_gripper_convention(env, rows: Sequence[Dict[str, np.ndarray]],
                                 task_id: int = 1) -> Dict[str, Any]:
    """Run the restore under both gripper conventions and report which one holds the cube.

    `phase_probe` closes with raw `ctrl = 255`; `counterfactual_value_probe` "establishes the
    grasp" with raw `ctrl = 0.0`. The spec forbids guessing, so both raw values and both
    action-channel signs are run on the same recorded held states and scored by G2.
    """
    import mujoco

    out = {}
    for label, kind, value in (("action_close(+1)", "action", +1.0),
                               ("action_open(-1)", "action", -1.0),
                               ("ctrl_255", "ctrl", 255.0),
                               ("ctrl_0", "ctrl", 0.0)):
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
