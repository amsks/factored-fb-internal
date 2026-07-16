"""Advantage-geometry vs value-geometry decomposition of the in-hand probe.

Hypothesis (user, 2026-07-15): bilinear GCIQL learns an ADVANTAGE geometry, not a value
geometry. Decompose Q_m(s,a,g) = U_m(s,g) + A_m(s,a,g), U_m = E_{a~mu}[Q_m], A_m the centered
(atlas) operator. Two halves:

  H1  "MLP fits return magnitude": the in-hand STATE ranking lives in U. Predict
      AUC(U_mlp) ~ 0.81, AUC(U_bil) ~ 0.39, and AUC(A@pi) ~ 0.5 for both (advantage does not
      rank states). Needs the in-hand success labels -> one batched rollout pass per critic.

  H2-free "bilinear organizes policy improvement": on a SHARED action bank per state, do the
      two critics' ADVANTAGE geometries agree (per-state Spearman over actions) even though
      their baselines U diverge across states? Pure forward passes, no rollout.

Reductions match the atlas: Q = min(q1,q2). Run under /dev/shm/.venv-jax/bin/python.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.value.counterfactual_value_probe import sample_grasps            # noqa: E402
from scripts.value.counterfactual_value_probe_jax import _auc, _establish_grasp  # noqa: E402

N_TASKS = 5


def _spearman_rows(X, Y):
    """Per-row Spearman between X[S,M] and Y[S,M] (rank-correlation over the M axis)."""
    def rank(A):
        order = np.argsort(np.argsort(A, axis=1), axis=1).astype(float)
        return order
    rx, ry = rank(X), rank(Y)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    num = (rx * ry).sum(1)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1)) + 1e-12
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bil-dir', required=True)
    ap.add_argument('--bil-step', type=int, default=500000)
    ap.add_argument('--mlp-dir', required=True)
    ap.add_argument('--mlp-step', type=int, default=1000000)
    ap.add_argument('--domain', default='cube-single-play-v0')
    ap.add_argument('--data-path', default='datasets')
    ap.add_argument('--n-states', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--mu-k', type=int, default=16)
    ap.add_argument('--pool', type=int, default=16)
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    import mujoco
    import jax
    import jax.numpy as jnp
    import ogbench
    from scripts.value.repsep_extract_jax import _load_gciql
    from scripts.atlas.mu_flow import MuFlow

    obs_g, phys_g = sample_grasps(args.data_path, args.domain, args.n_states)
    n = len(obs_g)
    agents = {}
    for name, d, st in (('bil', args.bil_dir, args.bil_step), ('mlp', args.mlp_dir, args.mlp_step)):
        ag, _cfg, env_name = _load_gciql(d, st)
        agents[name] = ag
        agents[name + '_env'] = env_name

    def V(ag, O, G):
        v = jnp.asarray(ag.network.select('value')(O, G))
        return np.asarray(v.min(0) if v.ndim > 1 else v)

    def Q(ag, O, G, A):
        q = jnp.asarray(ag.network.select('critic')(O, G, A))
        return np.asarray(q.min(0) if q.ndim > 1 else q)      # conservative min, atlas convention

    def Qbank(ag, O, G, Abank):
        """O[S,28], G[S,28], Abank[S,M,5] -> Q[S,M] (min over ensemble)."""
        S, M = Abank.shape[0], Abank.shape[1]
        Orep = np.repeat(O[:, None], M, 1).reshape(S * M, -1)
        Grep = np.repeat(G[:, None], M, 1).reshape(S * M, -1)
        q = Q(ag, Orep, Grep, Abank.reshape(S * M, 5))
        return np.array(q.reshape(S, M))      # writable copy

    mus = {'bil': MuFlow(agents['bil']), 'mlp': MuFlow(agents['mlp'])}

    # task goals from a throwaway env
    env0 = ogbench.make_env_and_datasets(agents['bil_env'], env_only=True)
    task_goal = {}
    for ti in range(1, N_TASKS + 1):
        _o, info = env0.reset(options=dict(task_id=ti))
        task_goal[ti] = np.asarray(info['goal'], np.float32)

    # ---------- readouts + shared-bank advantage geometry (no rollout) ----------
    per = {ti: {} for ti in range(1, N_TASKS + 1)}
    geom = {ti: {} for ti in range(1, N_TASKS + 1)}
    for ti in range(1, N_TASKS + 1):
        G = np.broadcast_to(task_goal[ti][None], obs_g.shape).astype(np.float32)
        # own-mu baseline U_m = E_{mu_m} Q_m ; value head V_m ; policy action pi_m
        rd = {}
        shared = []
        for name in ('bil', 'mlp'):
            ag = agents[name]
            Amu = mus[name].sample(obs_g, n=args.mu_k, seed=0)             # [S,K,5] own mu
            shared.append(mus[name].sample(obs_g, n=args.mu_k, seed=1))     # shared-bank half
            U = Qbank(ag, obs_g, G, Amu).mean(1)                           # [S]
            Vh = V(ag, obs_g, G)                                           # [S]
            api = np.clip(np.asarray(ag.sample_actions(
                observations=obs_g, goals=G, seed=jax.random.PRNGKey(ti), temperature=0.0)), -1, 1).astype(np.float32)
            Qpi = Q(ag, obs_g, G, api)                                     # [S]
            rd[name] = dict(U=U, V=Vh, Qpi=Qpi, AdvPi=Qpi - U)
        per[ti] = rd
        # shared bank = union of both mu proposals (critic-independent set of actions)
        B = np.concatenate(shared, axis=1).astype(np.float32)             # [S, 2K, 5]
        A_bil = Qbank(agents['bil'], obs_g, G, B); A_bil = A_bil - A_bil.mean(1, keepdims=True)
        A_mlp = Qbank(agents['mlp'], obs_g, G, B); A_mlp = A_mlp - A_mlp.mean(1, keepdims=True)
        sp = _spearman_rows(A_bil, A_mlp)                                  # [S] within-state adv agreement
        geom[ti] = dict(adv_agree_mean=float(np.mean(sp)), adv_agree_med=float(np.median(sp)),
                        U_bil=rd['bil']['U'], U_mlp=rd['mlp']['U'])
        print(f"  task{ti}: within-state adv agreement Spearman(A_bil,A_mlp) "
              f"mean {np.mean(sp):+.3f} med {np.median(sp):+.3f}", flush=True)

    # across-state baseline agreement (do U's diverge?) pooled over goals
    Ub = np.concatenate([geom[ti]['U_bil'] for ti in per]); Um = np.concatenate([geom[ti]['U_mlp'] for ti in per])
    base_agree = float(np.corrcoef(np.argsort(np.argsort(Ub)), np.argsort(np.argsort(Um)))[0, 1])
    print(f"  across-state baseline agreement Spearman(U_bil,U_mlp) = {base_agree:+.3f}", flush=True)

    # ---------- rollout for in-hand success labels, per critic ----------
    pool = [ogbench.make_env_and_datasets(agents['bil_env'], env_only=True) for _ in range(args.pool)]
    for e in pool:
        e.reset(options=dict(task_id=1))

    def rollout_labels(ag):
        rng = jax.random.PRNGKey(0)
        sr = {}
        labels = {}
        for ti in range(1, N_TASKS + 1):
            goal = task_goal[ti]
            psucc = np.zeros(n, float)
            for c in range(0, n, len(pool)):
                idx = list(range(c, min(c + len(pool), n)))
                envs = pool[:len(idx)]
                for env, j in zip(envs, idx):
                    env.reset(options=dict(task_id=ti))
                    _establish_grasp(env.unwrapped, phys_g[j], mujoco)
                ob = np.stack([np.asarray(e.unwrapped.compute_observation(), np.float32) for e in envs])
                gb = np.broadcast_to(goal[None], (len(idx), goal.shape[0])).astype(np.float32)
                done = np.zeros(len(idx), bool); hit = np.zeros(len(idx), bool)
                for _t in range(args.max_steps):
                    rng, k = jax.random.split(rng)
                    a = np.clip(np.asarray(ag.sample_actions(
                        observations=ob, goals=gb, seed=k, temperature=0.0)), -1, 1)
                    for jj, e in enumerate(envs):
                        if done[jj]:
                            continue
                        ob_, _, term, trunc, info = e.step(a[jj])
                        ob[jj] = np.asarray(ob_, np.float32)
                        if info.get('success', False):
                            hit[jj] = True
                        if term or trunc:
                            done[jj] = True
                    if done.all():
                        break
                for jj, j in enumerate(idx):
                    psucc[j] = float(hit[jj])
            labels[ti] = (psucc >= 0.5).astype(int)
            sr[ti] = float(psucc.mean())
        return labels, sr

    out = {'n_states': n, 'baseline_agree_spearman': base_agree,
           'adv_agree_per_task': {ti: geom[ti]['adv_agree_mean'] for ti in per},
           'adv_agree_mean': float(np.mean([geom[ti]['adv_agree_mean'] for ti in per])),
           'auc': {}}
    print("\n  === H1: AUC of each readout vs in-hand success ===", flush=True)
    for name in ('mlp', 'bil'):
        labels, sr = rollout_labels(agents[name])
        aucs = {k: [] for k in ('U', 'V', 'Qpi', 'AdvPi')}
        for ti in range(1, N_TASKS + 1):
            y = labels[ti].astype(bool)
            for k in aucs:
                aucs[k].append(_auc(y, per[ti][name][k]))
        summ = {k: float(np.nanmean(np.array(v, float))) for k, v in aucs.items()}
        out['auc'][name] = dict(summ, cf_success=float(np.mean(list(sr.values()))),
                                per_task={k: [None if x != x else round(x, 3) for x in v] for k, v in aucs.items()})
        print(f"  {name}: U {summ['U']:.3f}  V {summ['V']:.3f}  Qpi {summ['Qpi']:.3f}  "
              f"AdvPi {summ['AdvPi']:.3f}   (cf success {np.mean(list(sr.values())):.2f})", flush=True)

    outp = REPO / f"analysis/value/repsep/value_decomp{args.out_tag}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[value-decomp] -> {outp}", flush=True)


if __name__ == '__main__':
    main()
