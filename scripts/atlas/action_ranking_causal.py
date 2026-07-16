"""Does the critic know which ACTION is better? (causal, calibration-free)

For each fixed (s,g): shared action bank a_1..a_N ~ mu(.|s). Evaluate both critics' centered
advantages A_bil(s,a_i,g), A_mlp(s,a_i,g). Execute each first action in the simulator (exact
via prefix-replay), then continue EVERY branch with the SAME frozen continuation policy (mu,
goal-agnostic BC flow). Record a graded realized outcome Y_i, and within each state compute
    rho_m = Spearman( A_m(s,.,g), Y_i ).
This asks whether the critic ranks ACTIONS by realized value, uncontaminated by state-value
calibration (Finding 5: bilinear's absolute state value is anti-calibrated).

Outcome Y (graded, to avoid the sparse-success floor that broke the old oracle):
  Y_progress = d_init - min_t ||cube_t - goal||     (transport progress toward the goal cube)
  Y_1step    = d_init - ||cube after first action|| (continuation-free anchor)
  Y_success  = 1[min_t dist <= 4cm]

PRE-REGISTERED (matches primaryA / oodgrid factorial):
  cell            expected
  C1 ID/ID        rho_MLP >= rho_bil
  C4 OOD/ID       rho_MLP ~= rho_bil   (tie)
  C2 ID/OOD       rho_bil  > rho_MLP
  C3 OOD/OOD eval rho_bil  > rho_MLP   (the benchmark; largest gap)

Run under /dev/shm/.venv-jax/bin/python.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.atlas import config as C                                       # noqa: E402
from scripts.atlas.build_atlas import center_and_standardize               # noqa: E402
from scripts.atlas.prefix_replay import PrefixState, replay_to, make_env    # noqa: E402
from scripts.atlas.oracle import goal_cube_xyz                             # noqa: E402

CELLS = {'C1': (2, 'hindsight'), 'C2': (2, 'eval'),
         'C4': (0, 'hindsight'), 'C3': (0, 'eval')}
PREREG = {'C1': 'rho_MLP>=rho_bil', 'C4': 'rho_MLP~=rho_bil',
          'C2': 'rho_bil>rho_MLP', 'C3': 'rho_bil>rho_MLP'}
_QPOS_CUBE = slice(14, 17)


def _ps(banks, p, k):
    L = int(banks['state_prefix_len'][p, k])
    return PrefixState(seed=k % 5000, task_id=1, prefix=banks['state_prefix_actions'][p, k, :L])


def _cell_goals(banks, phase, k, source, n_eval):
    if source == 'eval':
        return banks['goals_eval'][:n_eval]
    if source == 'hindsight':
        return banks['goals_hindsight'][k][None]         # one ID goal per state
    raise ValueError(source)


def _atlas_col(crit, obs, acts, mu_center, goal):
    Q = crit.Q(obs[None], acts[None], goal[None])[0]            # [n_a, 1]
    Qmu = crit.Q(obs[None], mu_center[None], goal[None])[0]     # [N_MU, 1]
    return center_and_standardize(Q, Qmu)[:, 0]


def action_outcomes(pool, cont_fn, ps, acts, goal_obs, H, gamma=0.99):
    """Branch each first action, continue every branch under the SHARED frozen continuation
    cont_fn(obs[na,28], goal_obs) -> actions[na,5]. Time axis: first action = t=0, continuation
    = t=1..H. Returns (Y_return, Y_progress, Y_1step, Y_success, d_init, cf_success) where
    Y_return = gamma^{t_hit} (discounted goal-hit return, 0 if never hit -- the value-aligned,
    time-graded outcome), Y_progress = d_init - min_t dist, Y_1step = d_init - dist after first
    action (continuation-free anchor)."""
    na = len(acts)
    envs = pool[:na]
    goal_xyz = goal_cube_xyz(goal_obs)
    replay_to(envs[0], ps)
    cube0 = np.asarray(envs[0].unwrapped._data.qpos[_QPOS_CUBE], np.float64).copy()
    d_init = float(np.linalg.norm(cube0 - goal_xyz))
    obs = np.zeros((na, 28), np.float32)
    dmin = np.full(na, d_init, np.float64)
    d_1 = np.zeros(na)
    thit = np.full(na, -1, int)                                 # first hit timestep, -1 = none
    for j in range(na):
        replay_to(envs[j], ps)
        o, *_ = envs[j].step(np.asarray(acts[j], np.float32))
        obs[j] = np.asarray(o, np.float32)
        d = float(np.linalg.norm(np.asarray(envs[j].unwrapped._data.qpos[_QPOS_CUBE]) - goal_xyz))
        d_1[j] = d
        dmin[j] = min(dmin[j], d)
        if d <= 0.04:
            thit[j] = 0
    for t in range(1, H + 1):
        a = cont_fn(obs, goal_obs)                              # [na,5] shared continuation
        for j in range(na):
            if thit[j] >= 0:
                continue                                       # freeze after first hit
            o, *_ = envs[j].step(np.asarray(a[j], np.float32))
            obs[j] = np.asarray(o, np.float32)
            d = float(np.linalg.norm(np.asarray(envs[j].unwrapped._data.qpos[_QPOS_CUBE]) - goal_xyz))
            dmin[j] = min(dmin[j], d)
            if d <= 0.04:
                thit[j] = t
    Yret = np.where(thit >= 0, gamma ** np.maximum(thit, 0), 0.0)
    Ysucc = (thit >= 0).astype(float)
    return Yret, d_init - dmin, d_init - d_1, Ysucc, d_init, float(Ysucc.mean())


def _sp(a, y):
    if np.std(y) < 1e-9 or np.std(a) < 1e-9:
        return np.nan
    r = spearmanr(a, y).correlation
    return float(r) if np.isfinite(r) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-states', type=int, default=8)
    ap.add_argument('--n-act', type=int, default=16)
    ap.add_argument('--n-eval-goals', type=int, default=5)
    ap.add_argument('--H', type=int, default=150)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--cont', default='mlp_flow',
                    choices=['mu', 'bilinear_flow', 'mlp_flow'],
                    help="shared frozen continuation policy. 'mu'=goal-agnostic BC (floors); "
                         "a flow actor = goal-conditioned deployed-style (default mlp_flow, the "
                         "conservative choice for the bilinear-wins prediction).")
    ap.add_argument('--cells', default='C1,C2,C4,C3')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    from scripts.atlas.critics import load_critic
    from scripts.atlas.jax_load import CRITICS, load_agent
    from scripts.atlas.mu_flow import MuFlow

    banks = np.load(REPO / 'analysis/atlas/banks.npz')
    crits = {'bilinear_flow': load_critic('bilinear_flow'), 'mlp_flow': load_critic('mlp_flow')}

    ck, st = CRITICS[args.cont if args.cont != 'mu' else 'bilinear_flow']
    cont_agent = load_agent(ck, st)
    if args.cont == 'mu':
        _mu = MuFlow(cont_agent)
        def cont_fn(obs, goal_obs):
            return _mu.sample(obs, n=1, seed=0)[:, 0, :]
    else:
        import jax
        _seedbox = {'k': jax.random.PRNGKey(C.SEED)}
        def cont_fn(obs, goal_obs):
            _seedbox['k'], sk = jax.random.split(_seedbox['k'])
            g = np.broadcast_to(np.asarray(goal_obs, np.float32)[None], (len(obs), 28))
            return np.clip(np.asarray(cont_agent.sample_actions(
                observations=obs, goals=g, seed=sk, temperature=0.0)), -1, 1)

    pool = [make_env(task_id=1) for _ in range(args.n_act)]
    for e in pool:
        e.reset(options=dict(task_id=1))

    rng = np.random.RandomState(C.SEED)
    results = {}
    for cell in args.cells.split(','):
        phase, source = CELLS[cell]
        rho = {'bilinear_flow': {'ret': [], 'prog': [], '1step': []},
               'mlp_flow': {'ret': [], 'prog': [], '1step': []}}
        sr = []
        for k in range(args.n_states):
            obs = banks['state_obs'][phase, k]
            A_mu = banks['A_mu'][phase, k]
            mu_center = banks['mu_center_samples'][phase, k]
            acts = A_mu[rng.choice(len(A_mu), args.n_act, replace=False)]
            ps = _ps(banks, phase, k)
            for g in _cell_goals(banks, phase, k, source, args.n_eval_goals):
                Yret, Yp, Y1, Ys, _d0, _srg = action_outcomes(
                    pool, cont_fn, ps, acts, g, args.H, gamma=args.gamma)
                sr.append(_srg)
                for name, crit in crits.items():
                    A = _atlas_col(crit, obs, acts, mu_center, g)
                    rho[name]['ret'].append(_sp(A, Yret))
                    rho[name]['prog'].append(_sp(A, Yp))
                    rho[name]['1step'].append(_sp(A, Y1))

        def m(name, key):
            v = np.array(rho[name][key], float)
            return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
        rb, rm = m('bilinear_flow', 'ret'), m('mlp_flow', 'ret')   # PRIMARY = discounted return
        verdict = ('tie' if abs(rb - rm) <= C.EQUIV_MARGIN_DELTA
                   else 'bilinear' if rb > rm else 'mlp')
        results[cell] = dict(
            prereg=PREREG[cell], verdict=verdict,
            rho_bil_ret=rb, rho_mlp_ret=rm,
            rho_bil_prog=m('bilinear_flow', 'prog'), rho_mlp_prog=m('mlp_flow', 'prog'),
            rho_bil_1step=m('bilinear_flow', '1step'), rho_mlp_1step=m('mlp_flow', '1step'),
            cf_success=float(np.mean(sr)),
            n_pairs=len(rho['bilinear_flow']['ret']),
            n_ret_defined=int(np.isfinite(np.array(rho['bilinear_flow']['ret'], float)).sum()))
        print(f"  {cell} [{PREREG[cell]:>16}]  rho_bil {rb:+.3f}  rho_mlp {rm:+.3f}  -> {verdict:>8}"
              f"   (prog bil {results[cell]['rho_bil_prog']:+.3f} mlp {results[cell]['rho_mlp_prog']:+.3f}; "
              f"succ {results[cell]['cf_success']:.2f}; ret-def {results[cell]['n_ret_defined']}/{results[cell]['n_pairs']})",
              flush=True)

    out = REPO / f"analysis/atlas/action_ranking_causal{args.out_tag}.json"
    out.write_text(json.dumps({'cont': args.cont, 'H': args.H, 'n_act': args.n_act,
                               'n_states': args.n_states, 'cells': results}, indent=2))
    print(f"\n[action-ranking-causal] -> {out}", flush=True)


if __name__ == '__main__':
    main()
