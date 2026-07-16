"""T1: goal-smoothed reranking (pre-registered in PAPER/rescuing-fb/03-probes-spec.md).

Tests the ADVANTAGE-POOLING mechanism (01-synthesis §2.2): the MLP critic's per-goal
action-advantage is noise because it never pools across goals; the bilinear head pools by
construction. If pooling is the mechanism, hand-pooling the MLP at inference —

    Qbar_K(s,a,g) = (1/K) sum_j Q(s,a,g_j),  g_j = g with cube-xyz dims obs[19:22]
                    perturbed by eps_j ~ N(0, sigma^2 I_3)

— should lift its rerank success materially (0.57 -> >=0.70 at K=32, sigma=4cm) while the
bilinear stays ~0.95 (Qbar ≈ Q for a rank-d linear functional under a local goal cloud).

Protocol = Exp 1 / C3 (scripts/atlas/rerank_eval.py): at every env step sample N=32
candidates from ONE shared frozen BC flow (the bilinear agent's goal-agnostic actor_vf),
execute argmax_a Qbar(s, a, g_eval). env.reset() starts, eval-task goals, H=200, success =
cube within 4 cm. Batched rollouts: E envs in lockstep, ONE critic call per step with all
K goal variants in the same forward pass.

Unit check (verified vs cube_env.compute_observation): obs[19:22] = (cube_xyz - center)*10
=> 1 obs unit = 10 cm; sigma 2 cm = 0.2 obs units, 4 cm = 0.4 obs units.

Arms: {mlp, bilinear} x [ K=1 (Exp-1 replication) ; K in {8,32} x sigma in {2cm,4cm} ;
control K=32 with g_j = random dataset observations (destroys goal specificity — must hurt) ].

Sanity gate: the K=1 arms must approximately reproduce the known rerank numbers
(bilinear ~0.95, MLP ~0.57-0.59) BEFORE the smoothed arms run.

Run under /dev/shm/.venv-jax/bin/python with CUDA_VISIBLE_DEVICES=0,
XLA_PYTHON_CLIENT_PREALLOCATE=false.
Output: analysis/rescuing-fb/t1_goal_smooth_rerank.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

N_TASKS = 5
CUBE_SL = slice(19, 22)          # cube-xyz dims of a 28-dim cube observation
OBS_PER_METER = 10.0             # 1 obs unit = 10 cm (cube_env: xyz_scaler = 10)


def make_scorer(agent):
    """Jitted Qbar: obs [E,28], cand [E,N,5], goals [K,28] -> mean-over-K min-ensemble Q [E,N].
    All K goal variants ride in one forward pass (rows = E*N*K)."""
    import jax
    import jax.numpy as jnp
    crit = agent.network.select('critic')

    @jax.jit
    def score(obs, cand, goals):
        E, N, A = cand.shape
        K, D = goals.shape
        obs_b = jnp.broadcast_to(obs[:, None, None, :], (E, N, K, D)).reshape(-1, D)
        act_b = jnp.broadcast_to(cand[:, :, None, :], (E, N, K, A)).reshape(-1, A)
        goal_b = jnp.broadcast_to(goals[None, None, :, :], (E, N, K, D)).reshape(-1, D)
        q1, q2 = crit(obs_b, goal_b, act_b)
        return jnp.minimum(q1, q2).reshape(E, N, K).mean(-1)

    return score


def build_goal_cloud(goal, K, sigma_obs, mode, ds_obs, seed):
    """[K,28] goal variants. mode='perturb': cube-xyz jitter N(0, sigma^2). mode='random':
    K random dataset observations (control). K=1 => the exact goal (Exp-1 replication)."""
    if K == 1:
        return goal[None].copy()
    rng = np.random.default_rng(seed)
    if mode == 'random':
        idx = rng.choice(len(ds_obs), size=K, replace=False)
        return np.asarray(ds_obs[idx], np.float32)
    g = np.repeat(goal[None], K, axis=0)
    g[:, CUBE_SL] += rng.normal(0.0, sigma_obs, size=(K, 3)).astype(np.float32)
    return g


def run_arm(pool, mu, scorer, arm, ds_obs, n_samples, max_steps):
    """One arm over 5 tasks x len(pool) episodes. Candidate seeds match rerank_eval.py
    (ti*100003 + t) so K=1 replicates Exp 1 up to batch-size RNG."""
    E = len(pool)
    succ = {}
    for ti in range(1, N_TASKS + 1):
        t0 = time.time()
        obs = np.zeros((E, 28), np.float32)
        goal = None
        for j, e in enumerate(pool):
            o, info = e.reset(seed=1000 * ti + j, options=dict(task_id=ti))
            obs[j] = np.asarray(o, np.float32)
            goal = np.asarray(info['goal'], np.float32)
        goals = build_goal_cloud(goal, arm['K'], arm['sigma_obs'], arm['mode'], ds_obs,
                                 seed=90000 + 17 * ti + 3 * arm['K'] +
                                      int(1000 * arm['sigma_obs']))
        hit = np.zeros(E, bool)
        done = np.zeros(E, bool)
        for t in range(max_steps):
            cand = mu.sample(obs, n=n_samples, seed=(ti * 100003 + t))   # [E,N,5] shared flow
            Q = np.asarray(scorer(obs, cand, goals))                     # [E,N]
            a = cand[np.arange(E), Q.argmax(1)]
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
        succ[str(ti)] = float(hit.mean())
        print(f"    task {ti}: {succ[str(ti)]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    succ['overall'] = float(np.mean([succ[str(t)] for t in range(1, N_TASKS + 1)]))
    return succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50, help='episodes per task (= batched envs)')
    ap.add_argument('--n-samples', type=int, default=32, help='mu candidates per step')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--skip-gate', action='store_true', help='do not stop on K=1 gate failure')
    ap.add_argument('--smoke', action='store_true', help='tiny shapes, 2 arms, no gate')
    ap.add_argument('--out', default=str(REPO / 'analysis/rescuing-fb/t1_goal_smooth_rerank.json'))
    args = ap.parse_args()

    import ogbench
    from scripts.atlas.critics import load_critic
    from scripts.atlas.jax_load import CRITICS, load_agent
    from scripts.atlas.mu_flow import MuFlow

    if args.smoke:
        args.episodes, args.max_steps = 4, 25

    print(f"[t1] loading critics + mu flow (E={args.episodes}, N={args.n_samples}, "
          f"H={args.max_steps})", flush=True)
    crits = {'bilinear': load_critic('bilinear_flow'),      # d50 flow @500k
             'mlp': load_critic('mlp_flow')}                # MLP flow @1M
    scorers = {k: make_scorer(c.agent) for k, c in crits.items()}
    mu = MuFlow(load_agent(*CRITICS['bilinear_flow']))      # shared candidate flow (as Exp 1)

    print("[t1] loading dataset observations (random-goal control)", flush=True)
    _, train_ds, _ = ogbench.make_env_and_datasets('cube-single-play-v0')
    ds_obs = np.asarray(train_ds['observations'], np.float32)
    print(f"[t1] dataset obs: {ds_obs.shape}", flush=True)

    pool = [ogbench.make_env_and_datasets('cube-single-play-v0', env_only=True)
            for _ in range(args.episodes)]

    # sigma in obs units: 2 cm = 0.2, 4 cm = 0.4 (verified: obs[19:22] = (xyz-c)*10)
    sigmas = [(0.02, 0.2), (0.04, 0.4)]
    arms = []
    for c in ('bilinear', 'mlp'):
        arms.append(dict(critic=c, K=1, sigma_obs=0.0, sigma_m=0.0, mode='perturb'))
    for c in ('bilinear', 'mlp'):
        for K in (8, 32):
            for sm, so in sigmas:
                arms.append(dict(critic=c, K=K, sigma_obs=so, sigma_m=sm, mode='perturb'))
        arms.append(dict(critic=c, K=32, sigma_obs=0.0, sigma_m=0.0, mode='random'))
    if args.smoke:
        arms = [arms[0], dict(critic='mlp', K=8, sigma_obs=0.4, sigma_m=0.04, mode='perturb'),
                dict(critic='mlp', K=32, sigma_obs=0.0, sigma_m=0.0, mode='random')]

    def arm_key(a):
        if a['mode'] == 'random':
            return f"{a['critic']}|K{a['K']}|randomgoal"
        if a['K'] == 1:
            return f"{a['critic']}|K1"
        return f"{a['critic']}|K{a['K']}|sig{a['sigma_m'] * 100:g}cm"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = dict(n_samples=args.n_samples, episodes=args.episodes, max_steps=args.max_steps,
                   mu_source='bilinear_flow', obs_units_per_meter=OBS_PER_METER,
                   cube_dims='obs[19:22]', arms={})

    for arm in arms:
        key = arm_key(arm)
        print(f"[t1] arm {key}  (K={arm['K']}, sigma={arm['sigma_obs']} obs units, "
              f"mode={arm['mode']})", flush=True)
        succ = run_arm(pool, mu, scorers[arm['critic']], arm, ds_obs,
                       args.n_samples, args.max_steps)
        results['arms'][key] = dict(critic=arm['critic'], K=arm['K'],
                                    sigma_m=arm['sigma_m'], sigma_obs=arm['sigma_obs'],
                                    mode=arm['mode'], n_episodes=args.episodes,
                                    per_task={t: succ[str(t)] for t in range(1, N_TASKS + 1)},
                                    mean_success=succ['overall'])
        print(f"[t1] arm {key} overall {succ['overall']:.3f}", flush=True)
        out_path.write_text(json.dumps(results, indent=2))   # incremental save

        # ---- sanity gate: after both K=1 arms, require ~Exp-1 replication ----
        if not args.smoke and key == 'mlp|K1':
            bil = results['arms']['bilinear|K1']['mean_success']
            mlp = results['arms']['mlp|K1']['mean_success']
            ok = (bil >= 0.88) and (0.45 <= mlp <= 0.70)
            print(f"[t1] GATE K=1: bilinear {bil:.3f} (expect ~0.95), mlp {mlp:.3f} "
                  f"(expect 0.57-0.59) -> {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok and not args.skip_gate:
                print('[t1] gate FAILED — stopping before the smoothed sweep '
                      '(rerun with --skip-gate to override).', flush=True)
                sys.exit(2)

    print(f"[t1] done -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
