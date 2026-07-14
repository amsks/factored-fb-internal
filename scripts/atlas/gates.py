"""Decision gates for the value-function atlas.

- ``dataset_state_restore_drift``: split-oracle feasibility. Dataset rows cannot be
  prefix-replayed (their generating reset is unknown), so this decides whether C1/C2 use
  dataset states (approximate qpos/qvel restore reproduces the recorded dynamics to < 4cm
  in cube-xyz) or fall back to mu-rollout states.
- Phase 0A/0B (exploitation branch) are added in a later task.
"""
import mujoco
import numpy as np

CUBE = slice(14, 17)


def _restore_qposvel(env, qpos, qvel):
    # reset first so the TimeLimit wrapper allows step(); then overwrite mjData.
    env.reset(seed=0, options=dict(task_id=1, render_goal=False))
    u = env.unwrapped
    u._data.qpos[:] = qpos
    u._data.qvel[:] = qvel
    mujoco.mj_forward(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)


def dataset_state_restore_drift(env, dataset, idx, horizon, grasp_actions=None):
    """Restore a DATASET row via qpos/qvel, replay the dataset's OWN recorded actions,
    and compare the resulting trajectory against the dataset's RECORDED continuation.

    The divergence is real (the IK-target integrator ``_target_effector_pose`` is not in
    the dataset, so qpos/qvel alone cannot reconstruct it). ``grasp_actions`` overrides the
    replayed actions (used by tests); default replays ``dataset['actions'][idx:]``.

    Returns cube-xyz drift (vs recorded qpos[14:17]) and full-obs drift (vs recorded
    observations), plus passes = cube_xyz_drift_max < 0.04 m.
    """
    u = env.unwrapped
    acts = (grasp_actions if grasp_actions is not None
            else dataset['actions'][idx:idx + horizon])
    rec_qpos = dataset['qpos']
    rec_obs = dataset['observations']

    _restore_qposvel(env, dataset['qpos'][idx], dataset['qvel'][idx])
    cube_drift = 0.0
    obs_drift = 0.0
    for t, a in enumerate(acts[:horizon]):
        o, *_ = env.step(np.asarray(a, np.float32))
        j = idx + t + 1
        if j >= len(rec_qpos):
            break
        cube_drift = max(cube_drift,
                         float(np.linalg.norm(u._data.qpos[CUBE] - rec_qpos[j][CUBE])))
        obs_drift = max(obs_drift, float(np.abs(o - rec_obs[j]).max()))
    return dict(cube_xyz_drift_max=cube_drift, full_state_drift_max=obs_drift,
                threshold_m=0.04, passes=cube_drift < 0.04)
