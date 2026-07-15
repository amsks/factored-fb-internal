"""Frozen, deterministic banks. Byte-identical across all critics. Runs under the jax venv.

State bank: P0 = pinned reset states; P1..P3 = phase-labelled states rolled forward from mu
(all prefix-replayable). Dataset-start states for the C1/C2 factorial are added separately
by the factorial code (Task 10) per the split-oracle decision (analysis/atlas/DECISIONS.md).
Phase labels are policy-independent (cube-effector geometry + obs[18]); the generating prefix
is recorded so any state can be exactly reconstructed.
"""
import numpy as np
from scipy.stats import qmc

from scripts.atlas import config as C
from scripts.atlas.prefix_replay import make_env


def _phase_label(obs, qpos):
    contact = obs[C.CONTACT_IDX] > 0.5
    lifted = qpos[C.CUBE_SLICE][2] > 0.05
    if contact and lifted:
        return 3          # P3 holding/transport
    if contact and not lifted:
        return 2          # P2 at-contact, cube on table
    return 1              # P1 pre-contact


def _eval_goals(env):
    """The 5 canonical task goal observations, exactly as agents were evaluated:
    info['goal'] from reset(task_id=t) (evaluation.py:73-74)."""
    gs = []
    for tid in range(1, 6):
        _, info = env.reset(seed=0, options=dict(task_id=tid, render_goal=False))
        gs.append(np.asarray(info['goal'], np.float32))
    return np.stack(gs)


def build_banks(mu, dataset, out_path):
    rng = np.random.RandomState(C.SEED)
    env = make_env(task_id=1)
    K = C.K_STATES_PER_PHASE
    states = {p: [] for p in range(4)}     # phase -> [(obs, qpos, qvel, prefix)]

    ep = 0
    while min(len(states[p]) for p in range(4)) < K:
        ob, _ = env.reset(seed=ep, options=dict(task_id=1, render_goal=False))
        u = env.unwrapped
        if len(states[0]) < K:              # P0 = the reset state itself
            states[0].append((ob.copy(), u._data.qpos.copy(), u._data.qvel.copy(),
                              np.zeros((0, 5), np.float32)))
        prefix = []
        a = mu.sample(ob[None], n=1, seed=ep)[0, 0]
        for t in range(60):
            ob, *_ = env.step(a)
            prefix.append(a.copy())
            ph = _phase_label(ob, u._data.qpos)
            if len(states[ph]) < K:
                states[ph].append((ob.copy(), u._data.qpos.copy(), u._data.qvel.copy(),
                                   np.stack(prefix)))
            a = mu.sample(ob[None], n=1, seed=ep * 1000 + t + 1)[0, 0]
        ep += 1
        if ep > 5000:
            raise RuntimeError(f'could not fill phase banks; got '
                               f'{ {p: len(states[p]) for p in range(4)} }')

    obs = np.zeros((4, K, 28), np.float32)
    qp = np.zeros((4, K, 21), np.float32)
    qv = np.zeros((4, K, 20), np.float32)
    maxpref = max(len(s[3]) for p in range(4) for s in states[p][:K])
    pref = np.zeros((4, K, maxpref, 5), np.float32)
    preflen = np.zeros((4, K), np.int32)
    for p in range(4):
        for k, (o, q, v, pr) in enumerate(states[p][:K]):
            obs[p, k] = o
            qp[p, k] = q
            qv[p, k] = v
            pref[p, k, :len(pr)] = pr
            preflen[p, k] = len(pr)

    flat = obs.reshape(-1, 28)
    A_mu = mu.sample(flat, n=C.N_MU, seed=C.SEED).reshape(4, K, C.N_MU, 5)
    mu_center = mu.sample(flat, n=C.N_MU, seed=C.SEED + 1).reshape(4, K, C.N_MU, 5)
    A_pert = np.clip(A_mu[:, :, :C.N_PERTURB, :]
                     + rng.normal(0, C.PERTURB_SIGMA, (4, K, C.N_PERTURB, 5)),
                     -1, 1).astype(np.float32)
    A_glob = (qmc.Sobol(d=5, seed=C.SEED).random(C.N_GLOBAL) * 2 - 1).astype(np.float32)

    goals_eval = _eval_goals(env)
    gi = rng.choice(len(dataset['observations']), C.N_GRID, replace=False)
    goals_grid = dataset['observations'][gi].astype(np.float32)
    # Per-state REACHABLE hindsight goals: roll mu forward from each state and take the
    # observation a few horizons ahead. Reachable by construction (mu just did it), so the
    # oracle can discriminate first-actions toward them. Shape [4, K, N_HG, 28].
    goals_hind_ps = _reachable_hindsight(mu, env, obs, pref, preflen)

    np.savez(out_path, state_obs=obs, state_qpos=qp, state_qvel=qv,
             state_phase=np.repeat(np.arange(4), K),
             state_prefix_actions=pref, state_prefix_len=preflen,
             A_mu=A_mu, A_perturb=A_pert, A_global=A_glob, mu_center_samples=mu_center,
             goals_eval=goals_eval, goals_hindsight_ps=goals_hind_ps, goals_grid=goals_grid)


def _reachable_hindsight(mu, env, obs, pref, preflen, horizons=(15, 30, 45), n_alt=2):
    """For each state, roll mu forward from it and snapshot the observation at a few
    horizons -> reachable hindsight goals. n_alt independent rollouts per horizon give
    variety. Returns [4, K, len(horizons)*n_alt, 28]."""
    P, K = obs.shape[:2]
    NH = len(horizons) * n_alt
    out = np.zeros((P, K, NH, 28), np.float32)
    for p in range(P):
        for k in range(K):
            # reconstruct the state by prefix-replay, then roll mu forward
            L = int(preflen[p, k])
            gi = 0
            for alt in range(n_alt):
                env.reset(seed=k % 5000, options=dict(task_id=1, render_goal=False))
                for a in pref[p, k, :L]:
                    env.step(a)
                ob = obs[p, k]
                snaps = {}
                a = mu.sample(ob[None], n=1, seed=1000 * (p * K + k) + 17 * alt)[0, 0]
                for t in range(max(horizons)):
                    ob, *_ = env.step(a)
                    if (t + 1) in horizons:
                        snaps[t + 1] = ob.copy()
                    a = mu.sample(ob[None], n=1, seed=1000 * (p * K + k) + 17 * alt + t + 1)[0, 0]
                for h in horizons:
                    out[p, k, gi] = snaps.get(h, ob)
                    gi += 1
    return out


def load_banks(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
