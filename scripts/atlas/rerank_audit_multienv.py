"""Reranking audit across environments, as a function of candidate count N (Exp 1 generalized).

Per critic m, deploy the policy: each env step sample N candidates a_i ~ mu(.|s) from ONE shared
frozen BC flow (the bilinear agent's actor_vf), execute argmax_i Q_m(s,a_i,g). Sweep
N in {1,2,4,8,16,32,64,128} and compare, per environment:
    mu-floor (N=1) | bilinear rerank | MLP rerank | trained actors (each agent's own policy).

Dimension-agnostic (cube 28/5, antmaze 29/8, scene 40/5); everything inferred from the env/agents
(the cube-only helpers in critics.py/mu_flow.py hardcode 28/5, so this is self-contained).

Interpretation:
  strong gain with larger N  -> candidate COVERAGE is the bottleneck.
  early saturation at low val -> RANKING / planning is the bottleneck.
  trained actor >> reranking  -> continuous action modification / training history matters.

Run under /dev/shm/.venv-jax/bin/python.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

N_TASKS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--bil-dir', required=True)
    ap.add_argument('--mlp-dir', required=True)
    ap.add_argument('--bil-step', type=int, default=500000)
    ap.add_argument('--mlp-step', type=int, default=500000)
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--max-steps', type=int, default=1000)
    ap.add_argument('--n-list', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--flow-steps', type=int, default=10)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import ogbench
    from scripts.atlas.jax_load import load_agent

    N_LIST = [int(x) for x in args.n_list.split(',')]
    bil = load_agent(args.bil_dir, args.bil_step)
    mlp = load_agent(args.mlp_dir, args.mlp_step)
    agents = {'bilinear': bil, 'mlp': mlp}

    def mu_sample(obs, n, seed):
        """Shared candidate set from the bilinear agent's BC flow (goal-agnostic actor_vf)."""
        obs = jnp.asarray(obs)
        S, od = obs.shape
        ad = args.action_dim
        a = jax.random.normal(jax.random.PRNGKey(seed), (S, n, ad))
        orep = jnp.repeat(obs[:, None], n, axis=1).reshape(S * n, od)
        for i in range(args.flow_steps):
            t = jnp.ones((S * n, 1)) * (i / args.flow_steps)
            v = bil.network.select('actor_vf')(orep, a.reshape(-1, ad), t)
            a = a + v.reshape(S, n, ad) / args.flow_steps
        return np.clip(np.asarray(a), -1, 1)

    def q_min(agent, obs, acts, goal):
        """obs[S,od], acts[S,N,ad], goal[od] -> Q[S,N] (min ensemble)."""
        S, N, ad = acts.shape
        od = obs.shape[-1]
        orep = np.repeat(obs[:, None], N, axis=1).reshape(S * N, od)
        grep = np.repeat(np.asarray(goal)[None], S * N, axis=0)
        out = agent.network.select('critic')(jnp.asarray(orep), jnp.asarray(grep),
                                              jnp.asarray(acts.reshape(S * N, ad)))
        q = jnp.minimum(out[0], out[1]) if isinstance(out, (tuple, list)) else \
            (jnp.asarray(out).min(0) if jnp.asarray(out).ndim > 1 else jnp.asarray(out))
        return np.asarray(q).reshape(S, N)

    E = args.episodes
    pool = [ogbench.make_env_and_datasets(args.env, env_only=True) for _ in range(E)]
    args.action_dim = pool[0].action_space.shape[0]

    def rollout(policy_fn):
        """policy_fn(obs[E,od], goal[od], rng) -> actions[E,ad]. Returns overall success."""
        rng = jax.random.PRNGKey(0)
        srs = []
        for ti in range(1, N_TASKS + 1):
            obs, goal = None, None
            obs = np.zeros((E, pool[0].observation_space.shape[0]), np.float32)
            for j, e in enumerate(pool):
                o, info = e.reset(seed=1000 * ti + j, options=dict(task_id=ti))
                obs[j] = np.asarray(o, np.float32)
                goal = np.asarray(info['goal'], np.float32)
            hit = np.zeros(E, bool); done = np.zeros(E, bool)
            for _t in range(args.max_steps):
                rng, k = jax.random.split(rng)
                a = policy_fn(obs, goal, k)
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
            srs.append(float(hit.mean()))
        return float(np.mean(srs)), srs

    def rerank_policy(agent, N):
        def f(obs, goal, rng):
            cand = mu_sample(obs, N, int(jax.random.randint(rng, (), 0, 2**30)))
            if N == 1:
                return cand[:, 0, :]
            Q = q_min(agent, obs, cand, goal)
            return cand[np.arange(len(obs)), Q.argmax(1)]
        return f

    def trained_policy(agent):
        def f(obs, goal, rng):
            g = np.broadcast_to(np.asarray(goal)[None], (len(obs), goal.shape[0])).astype(np.float32)
            return np.clip(np.asarray(agent.sample_actions(
                observations=obs.astype(np.float32), goals=g, seed=rng, temperature=0.0)), -1, 1)
        return f

    res = {'env': args.env, 'episodes': E, 'n_list': N_LIST, 'sweep': {}, 'anchors': {}}
    # floor (N=1) once
    fov, _ = rollout(rerank_policy(bil, 1)); res['anchors']['floor'] = fov
    print(f"  [{args.env}] floor {fov:.3f}", flush=True)
    # trained actors once (non-fatal: a checkpoint whose actor net doesn't reconstruct under
    # load_agent -- seen on antmaze -- must not sink the reranking sweep, which is the point)
    for name, ag in agents.items():
        try:
            v, _ = rollout(trained_policy(ag)); res['anchors'][f'trained_{name}'] = v
            print(f"  [{args.env}] trained_{name} {v:.3f}", flush=True)
        except Exception as e:
            res['anchors'][f'trained_{name}'] = None
            print(f"  [{args.env}] trained_{name} FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
    # N-sweep for both rerankers
    for N in N_LIST:
        row = {}
        for name, ag in agents.items():
            v, _ = rollout(rerank_policy(ag, N)); row[name] = v
        res['sweep'][N] = row
        print(f"  [{args.env}] N={N:<4} bil {row['bilinear']:.3f}  mlp {row['mlp']:.3f}", flush=True)

    out = REPO / f"analysis/atlas/rerank_audit_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[rerank-audit:{args.env}] -> {out}", flush=True)


if __name__ == '__main__':
    main()
