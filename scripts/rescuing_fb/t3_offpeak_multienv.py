"""T3 — off-support peak x domain (spec: PAPER/rescuing-fb/03-probes-spec.md).

Tests the EXTRACTION boundary condition (01-synthesis §2.4): the bilinear+DDPGBC critic's
off-support Q-maximum (known on cube: co-trained DDPG+BC action beats every behavior-reference
action ~79% of the time, median +2.14) should be absent-or-benign on antmaze, because antmaze's
dataset action coverage is broad while cube's scripted play is narrow.

Unlike the Finding-1 instrument (scripts/atlas/off_reference_test.py), which used mu-flow
samples as the reference cloud, this probe needs NO BC flow (DDPG+BC cells have no vector
field): the reference cloud B_ref(s) = the dataset action at s + the actions of the 64
nearest-neighbor dataset states in whitened (per-dim standardized) obs space.

Per domain (512 random dataset states x the domain's 5 eval-task goals; goal = full obs):
  1. peak     : median[Q(s,a_DDPG,g) - max_{a in B_ref(s)} Q(s,a,g)] and frac[> max].
  2. departure: whitened NN distance of a_DDPG from B_ref(s) (gates.whitened_dist statistic:
                shrunk-cov whitening of the cloud, normalized by its leave-one-out median
                spread — dimensionless, cross-domain comparable), plus a raw per-dim version.
  3. coverage (dataset-only, no model): median NN-distance of 256 uniform actions in [-1,1]^A
                to the conditional action set (the 65 B_ref actions), and to a size-matched
                marginal (65 random dataset actions). Reported raw and /sqrt(A).

Sanity gate: the cube cell must roughly replicate 79%/+2.14 (different reference cloud, so
qualitative agreement only); cube frac < 0.4 => B_ref too generous, flag loudly.

Run: JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/t3_offpeak_multienv.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

SEED = 0
N_STATES = 512
K_NN = 64
N_UNIFORM = 256
COV_SHRINK_LAMBDA = 0.1   # = scripts/atlas/config.py values used by phase_0a
COV_EPS = 1e-4

_B = os.path.expanduser('~/bridge_ckpt_backup/gciql_outputs/factored-fb')
DOMAINS = {
    'cube': ('cube-single-play-v0',
             f'{_B}/GCIQL-State-Flow-Bilinear-d50-ddpgbc/sd000_20260617_221451', 500000),
    'antmaze': ('antmaze-medium-navigate-v0',
                f'{_B}/GCIQL-AntMaze-Medium-Bilinear-d50-DDPGBC/sd000_20260625_193818', 500000),
    'scene': ('scene-play-v0',
              f'{_B}/GCIQL-Scene-Bilinear-d50-DDPGBC/scene_state_gciql_bilinear_d50_ddpgbc_s0',
              500000),
}


def load_agent_dims(ckpt_dir, step, obs_dim, act_dim):
    """jax_load.load_agent with domain-correct example dims (that one hardcodes cube 28/5;
    restore replaces all params from the pkl, but action_dim/actor shapes come from create)."""
    from scripts.atlas.jax_load import _AGENTS, _read_agent_cfg, setup_jax_path
    setup_jax_path()
    import importlib

    import jax.numpy as jnp
    from utils.flax_utils import restore_agent
    ag = _read_agent_cfg(ckpt_dir)
    name = ag.get('agent_name', 'gciql_flow')
    modname, clsname = _AGENTS[name]
    mod = importlib.import_module(modname)
    cls = getattr(mod, clsname)
    cfg = dict(mod.get_config())
    cfg.update({k: v for k, v in ag.items() if k in cfg})
    cfg['encoder'] = None
    if 'frame_stack' in cfg:
        cfg['frame_stack'] = None
    agent = cls.create(0, jnp.zeros((1, obs_dim), jnp.float32),
                       jnp.zeros((1, act_dim), jnp.float32), cfg)
    return restore_agent(agent, ckpt_dir, step)


def knn_indices(pool_w, query_idx, k, chunk=200_000):
    """k nearest pool rows (excluding self) for each query (queries ARE pool rows).
    pool_w: [N,D] whitened obs. Returns [nq,k] pool indices (unordered within top-k)."""
    Q = pool_w[query_idx]                                     # [nq, D]
    nq = len(query_idx)
    best_d = np.full((nq, k), np.inf, np.float32)
    best_i = np.zeros((nq, k), np.int64)
    q_sq = (Q ** 2).sum(1)[:, None]
    for start in range(0, len(pool_w), chunk):
        block = pool_w[start:start + chunk]
        d2 = q_sq - 2.0 * (Q @ block.T) + (block ** 2).sum(1)[None, :]   # [nq, B]
        # exclude self
        in_blk = (query_idx >= start) & (query_idx < start + block.shape[0])
        rows = np.nonzero(in_blk)[0]
        d2[rows, query_idx[rows] - start] = np.inf
        kk = min(k, d2.shape[1])
        top = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        cand_d = np.take_along_axis(d2, top, 1).astype(np.float32)
        cand_i = (top + start).astype(np.int64)
        all_d = np.concatenate([best_d, cand_d], 1)
        all_i = np.concatenate([best_i, cand_i], 1)
        sel = np.argpartition(all_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(all_d, sel, 1)
        best_i = np.take_along_axis(all_i, sel, 1)
    return best_i


def whitened_nn_dist(a, cloud):
    """gates.whitened_dist without the mujoco import: relative whitened NN distance of `a`
    to the action cloud, normalized by the cloud's leave-one-out median spread."""
    cov = np.cov(np.asarray(cloud, np.float64).T)
    cov = ((1 - COV_SHRINK_LAMBDA) * cov + COV_SHRINK_LAMBDA * np.diag(np.diag(cov))
           + COV_EPS * np.eye(cov.shape[0]))
    W = np.linalg.cholesky(np.linalg.inv(cov))
    cw = cloud @ W
    loo = np.median([
        np.min(np.linalg.norm(np.delete(cw, i, 0) - cw[i], axis=1))
        for i in range(len(cw))])
    return float(np.min(np.linalg.norm(cw - a @ W, axis=1))) / (loo + 1e-12)


def run_domain(name, env_name, ckpt_dir, step):
    import jax
    import jax.numpy as jnp
    import ogbench
    rng = np.random.default_rng(SEED)

    env, train_ds, _ = ogbench.make_env_and_datasets(env_name)
    obs_all = np.asarray(train_ds['observations'], np.float32)    # [N, od]
    act_all = np.asarray(train_ds['actions'], np.float32)         # [N, A]
    N, od = obs_all.shape
    A = act_all.shape[1]
    print(f'[{name}] dataset {obs_all.shape} actions {act_all.shape}', flush=True)

    # eval-task goals = full obs, as in training
    goals = []
    for ti in range(1, 6):
        _, info = env.reset(seed=1000 * ti, options=dict(task_id=ti))
        goals.append(np.asarray(info['goal'], np.float32))
    goals = np.stack(goals)                                       # [5, od]
    assert goals.shape[1] == od, goals.shape

    # whitened obs space (per-dim standardization) + 64-NN reference clouds
    mu, sd = obs_all.mean(0), obs_all.std(0)
    pool_w = (obs_all - mu) / np.maximum(sd, 1e-6)
    q_idx = rng.choice(N, size=N_STATES, replace=False)
    nn_idx = knn_indices(pool_w, q_idx, K_NN)                     # [512, 64]
    # B_ref(s) = dataset action at s + the 64 NN states' actions -> [512, 65, A]
    bref = np.concatenate([act_all[q_idx][:, None], act_all[nn_idx]], axis=1)
    S, R = N_STATES, K_NN + 1
    obs_q = obs_all[q_idx]                                        # [512, od]

    agent = load_agent_dims(ckpt_dir, step, od, A)

    def q_min(obs, goal_rep, acts, bs=16384):
        out = []
        for i in range(0, len(obs), bs):
            q1, q2 = agent.network.select('critic')(
                jnp.asarray(obs[i:i + bs]), jnp.asarray(goal_rep[i:i + bs]),
                jnp.asarray(acts[i:i + bs]))
            out.append(np.asarray(jnp.minimum(q1, q2)))
        return np.concatenate(out)

    obs_rep = np.repeat(obs_q[:, None], R, 1).reshape(-1, od)     # [512*65, od]
    bref_flat = bref.reshape(-1, A)

    peaks = np.zeros((5, S), np.float32)
    dep_w = np.zeros((5, S), np.float32)
    dep_raw = np.zeros((5, S), np.float32)
    for gi in range(5):
        g = goals[gi]
        a_ddpg = np.clip(np.asarray(agent.sample_actions(
            observations=obs_q, goals=np.repeat(g[None], S, 0),
            seed=jax.random.PRNGKey(SEED + gi), temperature=0.0)), -1, 1)  # [512, A]
        q_ddpg = q_min(obs_q, np.repeat(g[None], S, 0), a_ddpg)            # [512]
        q_ref = q_min(obs_rep, np.repeat(g[None], S * R, 0),
                      bref_flat).reshape(S, R)                              # [512, 65]
        peaks[gi] = q_ddpg - q_ref.max(1)
        for si in range(S):
            dep_w[gi, si] = whitened_nn_dist(a_ddpg[si], bref[si])
            dep_raw[gi, si] = np.min(np.linalg.norm(bref[si] - a_ddpg[si], axis=1)) / np.sqrt(A)
        print(f'[{name}] goal {gi + 1}: peak_med={np.median(peaks[gi]):+.3f} '
              f'frac>0={(peaks[gi] > 0).mean():.3f} dep_w_med={np.median(dep_w[gi]):.2f}',
              flush=True)

    # dataset-only action coverage: uniform actions vs conditional / marginal action sets
    unif = rng.uniform(-1, 1, size=(S, N_UNIFORM, A)).astype(np.float32)
    marg = act_all[rng.integers(0, N, size=(S, R))]               # size-matched marginal
    d_cond = np.zeros((S, N_UNIFORM), np.float32)
    d_marg = np.zeros((S, N_UNIFORM), np.float32)
    for si in range(S):
        d_cond[si] = np.linalg.norm(unif[si][:, None] - bref[si][None], axis=2).min(1)
        d_marg[si] = np.linalg.norm(unif[si][:, None] - marg[si][None], axis=2).min(1)

    pk = peaks.ravel()
    res = dict(
        env=env_name, ckpt=ckpt_dir, step=step, n_states=S, k_nn=K_NN, n_pairs=int(pk.size),
        action_dim=A, obs_dim=od,
        peak_median=float(np.median(pk)),
        peak_frac_positive=float((pk > 0).mean()),
        peak_q25=float(np.quantile(pk, 0.25)), peak_q75=float(np.quantile(pk, 0.75)),
        peak_per_goal_median=[float(np.median(peaks[g])) for g in range(5)],
        peak_per_goal_frac=[float((peaks[g] > 0).mean()) for g in range(5)],
        departure_whitened_median=float(np.median(dep_w)),
        departure_whitened_q75=float(np.quantile(dep_w, 0.75)),
        departure_raw_perdim_median=float(np.median(dep_raw)),
        coverage_cond_nn_median=float(np.median(d_cond)),
        coverage_cond_nn_median_perdim=float(np.median(d_cond)) / float(np.sqrt(A)),
        coverage_marg_nn_median=float(np.median(d_marg)),
        coverage_marg_nn_median_perdim=float(np.median(d_marg)) / float(np.sqrt(A)),
    )
    print(f"[{name}] PEAK median={res['peak_median']:+.3f} frac>0={res['peak_frac_positive']:.3f}"
          f" | dep_w={res['departure_whitened_median']:.2f}"
          f" | cov_cond/dim={res['coverage_cond_nn_median_perdim']:.3f}"
          f" cov_marg/dim={res['coverage_marg_nn_median_perdim']:.3f}", flush=True)
    return res


def main():
    out = {'spec': 'T3 off-support peak x domain (03-probes-spec.md)',
           'config': dict(seed=SEED, n_states=N_STATES, k_nn=K_NN, n_uniform=N_UNIFORM,
                          whitening='per-dim standardized obs; departure = gates.whitened_dist '
                                    '(shrunk-cov, LOO-normalized) on the 65-action B_ref cloud'),
           'domains': {}}
    order = ['cube', 'antmaze', 'scene']
    for name in order:
        env_name, ckpt, step = DOMAINS[name]
        out['domains'][name] = run_domain(name, env_name, ckpt, step)
        if name == 'cube':
            frac = out['domains']['cube']['peak_frac_positive']
            gate = frac >= 0.4
            out['cube_sanity_gate'] = dict(frac=frac, passes=bool(gate),
                                           note='expect qualitative replication of 79%/+2.14 '
                                                '(reference cloud differs: dataset-NN, not mu-flow)')
            print(f'[gate] cube frac>0 = {frac:.3f} -> {"PASS" if gate else "FAIL: B_ref suspect"}',
                  flush=True)
            if not gate:
                print('[gate] cube sanity FAILED — continuing but antmaze read is NOT trusted '
                      'until B_ref construction is vetted', flush=True)
    c, a = out['domains']['cube'], out['domains']['antmaze']
    out['coverage_ratio_cube_over_antmaze_perdim'] = (
        c['coverage_cond_nn_median_perdim'] / a['coverage_cond_nn_median_perdim'])
    p = REPO / 'analysis/rescuing-fb/t3_offpeak_multienv.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'[t3] -> {p}', flush=True)


if __name__ == '__main__':
    main()
