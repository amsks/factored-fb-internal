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


def _whiten(mu_cloud, lam, eps):
    cov = np.cov(np.asarray(mu_cloud, np.float64).T)
    cov = (1 - lam) * cov + lam * np.diag(np.diag(cov)) + eps * np.eye(cov.shape[0])
    return np.linalg.cholesky(np.linalg.inv(cov))


def whitened_dist(a, mu_cloud, lam, eps):
    """Relative whitened nearest-neighbour distance of `a` to the mu-reference cloud,
    normalized by the cloud's own leave-one-out median spread (spec section 3)."""
    W = _whiten(mu_cloud, lam, eps)

    def nn(x):
        return float(np.min(np.linalg.norm((mu_cloud - x) @ W, axis=1)))

    loo = np.median([
        float(np.min(np.linalg.norm((np.delete(mu_cloud, i, 0) - mu_cloud[i]) @ W, axis=1)))
        for i in range(len(mu_cloud))
    ])
    return nn(a) / (loo + 1e-12)


def _actor_action(critic, obs, goal):
    import jax
    sa = critic.agent.sample_actions(
        observations=np.asarray(obs)[None], goals=np.asarray(goal)[None],
        seed=jax.random.PRNGKey(0), temperature=0.0)
    return np.clip(np.asarray(sa)[0], -1, 1)


def phase_0a(critics, banks, phases=(0, 2), n_states=8):
    """Exploitation branch gate. For matched (state, eval-goal), compare how far each
    agent's chosen action sits from the mu-reference cloud. branch_alive = the bilinear
    DDPG+BC action departs the reference MORE than the MLP DDPG+BC action does.

    `critics`: dict name -> critic object (has .agent.sample_actions).
    """
    from scripts.atlas import config as C
    goals = banks['goals_eval']
    key = {'ddpgbc_bil': 'bilinear_ddpgbc', 'ddpgbc_mlp': 'mlp_ddpgbc',
           'flow_bil': 'bilinear_flow', 'flow_mlp': 'mlp_flow'}
    acc = {k: [] for k in key}
    for p in phases:
        for k in range(n_states):
            obs = banks['state_obs'][p, k]
            cloud = banks['A_mu'][p, k]
            for gi in range(len(goals)):
                for slot, cname in key.items():
                    a = _actor_action(critics[cname], obs, goals[gi])
                    acc[slot].append(whitened_dist(a, cloud, C.COV_SHRINK_LAMBDA, C.COV_EPS))
    med = {k: float(np.median(v)) for k, v in acc.items()}
    return dict(d_ddpgbc_bil=med['ddpgbc_bil'], d_ddpgbc_mlp=med['ddpgbc_mlp'],
                d_flow_bil=med['flow_bil'], d_flow_mlp=med['flow_mlp'],
                branch_alive=med['ddpgbc_bil'] > med['ddpgbc_mlp'])
