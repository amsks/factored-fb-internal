"""Batched-rollout version of the RLBrew counterfactual value probe. Numerically matches
scripts/value/counterfactual_value_probe_jax.py (same sample_grasps, _establish_grasp, value
definitions, _auc) but rolls the 200 grasps per task in lockstep across an env pool with ONE
policy call per timestep -- ~1hr sequential -> ~minutes.

Value: gciql -> V(s,g)=value(s,g);  gciql_flow_bilinear -> Q(s,pi(s,g),g) at the grasp.
Success: any info['success'] over the rollout (matches the paper's `s = s or ...`).
Run under the jax venv.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['gciql', 'gciql_flow_bilinear'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--step', type=int, default=500000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--domain', default='cube-single-play-v0')
    ap.add_argument('--data-path', default='datasets')
    ap.add_argument('--n-states', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--pool', type=int, default=40)
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    import mujoco
    import jax
    import jax.numpy as jnp
    import ogbench
    from scripts.value.repsep_extract_jax import _load_gciql

    obs_g, phys_g = sample_grasps(args.data_path, args.domain, args.n_states)
    agent, _cfg, env_name = _load_gciql(args.run_dir, args.step)

    @jax.jit
    def value(O, G):
        v = jnp.asarray(agent.network.select('value')(O, G))
        return v.mean(0) if v.ndim > 1 else v

    @jax.jit
    def qcrit(O, G, A):
        v = jnp.asarray(agent.network.select('critic')(O, G, A))
        return v.mean(0) if v.ndim > 1 else v

    pool = [ogbench.make_env_and_datasets(env_name, env_only=True) for _ in range(args.pool)]
    for e in pool:
        e.reset(options=dict(task_id=1))
    task_goal = {}
    for ti in range(1, N_TASKS + 1):
        _o, info = pool[0].reset(options=dict(task_id=ti))
        task_goal[ti] = np.asarray(info['goal'], np.float32)

    rng = jax.random.PRNGKey(args.seed)
    per_task_auc, per_task_sr, records = [], [], []
    n = len(obs_g)
    for ti in range(1, N_TASKS + 1):
        goal = task_goal[ti]
        G = np.broadcast_to(goal[None], obs_g.shape).astype(np.float32)
        if args.method == 'gciql':
            val = np.asarray(value(obs_g, G))
        else:
            rng, k = jax.random.split(rng)
            A = np.clip(np.asarray(agent.sample_actions(
                observations=obs_g, goals=G, seed=k, temperature=0.0)), -1, 1).astype(np.float32)
            val = np.asarray(qcrit(obs_g, G, A))

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
            for t in range(args.max_steps):
                rng, k = jax.random.split(rng)
                a = np.clip(np.asarray(agent.sample_actions(
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
        lbl = (psucc >= 0.5).astype(int)
        auc = _auc(lbl.astype(bool), val)
        per_task_auc.append(auc); per_task_sr.append(float(psucc.mean()))
        for j in range(n):
            records.append((ti, float(psucc[j]), float(val[j])))
        print(f"  task{ti}: cf success {psucc.mean():.2f}  value-AUC {auc if auc==auc else float('nan'):.3f}", flush=True)

    per_task_auc = np.array(per_task_auc, float)
    out = REPO / f"analysis/value/repsep/cf_value_{args.method}{args.out_tag}.json"
    out.write_text(json.dumps({
        'method': args.method, 'n_states': n, 'batched': True,
        'cf_value_auc_mean': float(np.nanmean(per_task_auc)),
        'cf_value_auc_per_task': per_task_auc.tolist(),
        'cf_success_rate': per_task_sr}, indent=2))
    print(f"[cf-batched:{args.method}] value-AUC mean {np.nanmean(per_task_auc):.3f} "
          f"(cf success {np.mean(per_task_sr):.2f}) -> {out}", flush=True)


if __name__ == '__main__':
    main()
