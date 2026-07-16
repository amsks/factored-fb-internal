"""Exp 1: behavior-sample reranking eval (no actor training, no sim branching).

Deploy, per critic m, the policy: at every env step sample a_1..a_N ~ mu(.|s) (one SHARED frozen
BC flow -> the same candidate set for every critic), execute a* = argmax_i Q_m(s,a_i,g). Roll out
on the 5 eval tasks from reset() (OOD start, H=200, success = cube within 4cm; the C3 protocol)
and report success. Anchors: mu-floor (N=1, the behavior policy) and the trained flow actors
(bilinear-flow C3 0.947, MLP-flow 0.747, from the oodgrid).

  bilinear rerank wins -> the FINAL bilinear critic ranks supported actions better.
  MLP rerank wins/ties -> the bilinear advantage is actor-training dynamics, not final ordering.

Run under /dev/shm/.venv-jax/bin/python.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

N_TASKS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=32, help='mu candidates per step (rerank N)')
    ap.add_argument('--episodes', type=int, default=25, help='episodes per task')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--mu', default='bilinear_flow', choices=['bilinear_flow', 'mlp_flow'],
                    help='which BC flow supplies the SHARED candidate set (both critics rerank it)')
    ap.add_argument('--bil-step', type=int, default=None, help='override bilinear critic step')
    ap.add_argument('--mlp-step', type=int, default=None, help='override MLP critic step')
    ap.add_argument('--bil-dir', default=None, help='override bilinear critic run-dir')
    ap.add_argument('--mlp-dir', default=None, help='override MLP critic run-dir')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import ogbench
    from scripts.atlas.critics import load_critic, _JaxGCIQL
    from scripts.atlas.jax_load import CRITICS, load_agent
    from scripts.atlas.mu_flow import MuFlow

    def _crit(name, step, rdir):
        if step is None and rdir is None:
            return load_critic(name)
        d = rdir or CRITICS[name][0]
        return _JaxGCIQL(load_agent(d, step if step is not None else CRITICS[name][1]))
    crits = {'bilinear_flow': _crit('bilinear_flow', args.bil_step, args.bil_dir),
             'mlp_flow': _crit('mlp_flow', args.mlp_step, args.mlp_dir)}
    mu = MuFlow(load_agent(*CRITICS[args.mu]))

    E = args.episodes
    pool = [ogbench.make_env_and_datasets('cube-single-play-v0', env_only=True) for _ in range(E)]

    def q_min(crit, obs, acts, goal):
        """obs[E,28], acts[E,N,5], goal[28] -> Q[E,N] (min ensemble)."""
        return crit.Q(obs, acts, goal[None])[:, :, 0]

    # policies: 'floor' = execute mu sample (N=1); each critic = argmax over N mu candidates
    policies = ['floor', 'bilinear_flow', 'mlp_flow']
    succ = {p: {} for p in policies}
    for pol in policies:
        N = 1 if pol == 'floor' else args.n_samples
        rng = jax.random.PRNGKey(0)
        for ti in range(1, N_TASKS + 1):
            obs = np.zeros((E, 28), np.float32)
            goal = None
            for j, e in enumerate(pool):
                o, info = e.reset(seed=1000 * ti + j, options=dict(task_id=ti))
                obs[j] = np.asarray(o, np.float32)
                goal = np.asarray(info['goal'], np.float32)
            hit = np.zeros(E, bool); done = np.zeros(E, bool)
            for t in range(args.max_steps):
                cand = mu.sample(obs, n=N, seed=(ti * 100003 + t))        # [E,N,5] shared
                if pol == 'floor':
                    a = cand[:, 0, :]
                else:
                    Q = np.asarray(q_min(crits[pol], obs, cand, goal))    # [E,N]
                    a = cand[np.arange(E), Q.argmax(1)]                   # [E,5]
                for j, e in enumerate(pool):
                    if done[j]:
                        continue
                    o, _, term, trunc, info = e.step(np.clip(a[j], -1, 1))
                    obs[j] = np.asarray(o, np.float32)
                    if info.get('success', False):
                        hit[j] = True
                    if term or trunc:
                        done[j] = True
                if done.all():
                    break
            succ[pol][ti] = float(hit.mean())
        overall = float(np.mean(list(succ[pol].values())))
        succ[pol]['overall'] = overall
        print(f"  {pol:14} N={N:<3} overall {overall:.3f}   per-task "
              f"{[round(succ[pol][t], 2) for t in range(1, N_TASKS + 1)]}", flush=True)

    out = REPO / f"analysis/atlas/rerank_eval{args.out_tag}.json"
    out.write_text(json.dumps({'n_samples': args.n_samples, 'episodes': E,
                               'mu_source': args.mu, 'success': succ}, indent=2))
    print(f"\n[rerank-eval] -> {out}", flush=True)


if __name__ == '__main__':
    main()
