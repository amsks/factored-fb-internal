"""Physically-grounded candidate fields for the action-grounded mode decode.

For a state s (qpos,qvel) and action a:
  cube_delta(s,a)   one-step cube displacement dx_cube in R^3 (restore qpos/qvel, step a).
                    qpos/qvel restore is cube-accurate for ONE step (<~1mm), so no prefix
                    replay is needed and the internal IK-target residual is negligible here.
  T_dir(s,a,g)      = dx_cube(s,a) . d_g(s)   (goal-directed transport)
  T_grasp(s,a)      = contact gained after the step (obs[18])   (grasp acquisition)

Cube xyz: qpos[14:17] (physical); in an OBSERVATION it is obs[19:22]/10 + [.425,0,0].
Run under the jax venv OR .venv (pure mujoco/numpy; no jax needed here).
"""
import mujoco
import numpy as np

CUBE_QPOS = slice(14, 17)
CONTACT_IDX = 18
_XYZ_CENTER = np.array([0.425, 0.0, 0.0], np.float32)


def obs_cube_xyz(obs):
    return np.asarray(obs, np.float32)[19:22] / 10.0 + _XYZ_CENTER


def restore_qposvel(env, qpos, qvel):
    env.reset(seed=0, options=dict(task_id=1, render_goal=False))
    u = env.unwrapped
    u._data.qpos[:] = qpos
    u._data.qvel[:] = qvel
    mujoco.mj_forward(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)


def cube_delta_batch(env, qpos, qvel, actions):
    """dx_cube for every action from the SAME state. actions [n,5] -> [n,3].
    Restores qpos/qvel before each action (single step, cube-accurate)."""
    u = env.unwrapped
    c0 = qpos[CUBE_QPOS].copy()
    out = np.zeros((len(actions), 3), np.float32)
    contact = np.zeros(len(actions), np.float32)
    for i, a in enumerate(actions):
        restore_qposvel(env, qpos, qvel)
        ob, *_ = env.step(np.asarray(a, np.float32))
        out[i] = u._data.qpos[CUBE_QPOS] - c0
        contact[i] = ob[CONTACT_IDX]
    return out, contact


def goal_dir(state_cube_xyz, goal_obs, eps=1e-6):
    g = obs_cube_xyz(goal_obs)
    d = g - np.asarray(state_cube_xyz, np.float32)
    return d / (np.linalg.norm(d) + eps), float(np.linalg.norm(d))


def T_dir(dx_cube, dg):
    """[n,3] . [3] -> [n]  goal-directed transport score per action."""
    return dx_cube @ dg
