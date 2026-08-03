"""P2 goal-approach monotonicity on EVAL-TASK rollouts (phenomenon.tex Pred. mech.3).

The deferred probe from docs/specs/2026-07-31-bilinear-gciql-metric-probe.md §P2, run on
eval-pair (s_t, g_task) sequences instead of val-set pairs. The mechanism claim it tests:
a rank-d critic generalizes to eval (s, g) combinations never co-observed in play data
(matrix completion), so the monolithic critic's value ranking should degrade MORE from
val-pairs (P1, where gciql beats biql 0.63 vs 0.54) to eval-task rollouts than the
bilinear's — an interaction effect, not a main effect.

Two modes, so critics can be scored on each other's rollouts (2x2: the rollout-generating
policy is a state-distribution confound, crossing it controls for that):

  collect — build env + restore a policy, roll out the OGBench eval protocol
            (task_id 1..N, temperature 0), store per-episode obs sequences + goal + success:
    python -m eval.rollout_probe --mode collect --critic iql_bilinear --actor flowbc \
        --domain cube-single --override value_latent_dim=50 --seed 0 \
        --restore_path <run_dir> --restore_epoch 500000 --episodes 10 --out rollouts_biql_s0.npz

  score   — env-free (like eval.metric_probe): restore a critic, compute V(s_t, g) along
            every stored episode, report per-episode Spearman rho(V, t) (positive = value
            rises as the policy approaches the goal; rank-invariant to the steps/selfsub
            transforms, which are monotone in V) and the adjacent-step bump rate
            P[V_{t+1} < V_t]:
    python -m eval.rollout_probe --mode score --critic gciql --actor flowbc \
        --domain cube-single --seed 0 --restore_path <run_dir> --restore_epoch 500000 \
        --rollouts rollouts_biql_s0.npz --out p2_gciql_on_biql_s0
"""
import argparse
import json
import os

import numpy as np

from eval.metric_probe import spearman


def collect(args):
    import jax
    from env import make_config, get_adapter
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.flax_utils import restore_agent

    cfg = make_config(args.critic, args.actor, args.domain, args.override)
    adapter = get_adapter(cfg)
    env, train_ds, _ = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    import jax.numpy as jnp
    agent = Agent.create(args.critic, args.actor, args.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    n_tasks = len(env.unwrapped.task_infos)
    obs_seqs, goals, task_ids, successes = [], [], [], []
    key = jax.random.PRNGKey(args.seed * 1000 + 17)
    for tid in range(1, n_tasks + 1):
        for _ep in range(args.episodes):
            observation, info = env.reset(options=dict(task_id=tid, render_goal=False))
            goal = np.asarray(info["goal"], np.float32)
            seq = [np.asarray(observation, np.float32)]
            done, success = False, False
            while not done:
                key, sub = jax.random.split(key)
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(observation, np.float32), goals=goal,
                    seed=sub, temperature=0))
                observation, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                seq.append(np.asarray(observation, np.float32))
                done = bool(term) or bool(trunc)
            success = bool(info.get("success", 0))
            obs_seqs.append(np.stack(seq))
            goals.append(goal)
            task_ids.append(tid)
            successes.append(success)
            print(f"task {tid} ep {_ep}: T={len(seq)} success={success}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # A plain np.array(..., dtype=object) on equal-length episodes builds an (N,T,D) object
    # array of scalars instead of a 1-D array of (T,D) blocks; fill explicitly.
    seq_arr = np.empty(len(obs_seqs), dtype=object)
    for i, s in enumerate(obs_seqs):
        seq_arr[i] = s
    np.savez_compressed(
        args.out,
        obs_seqs=seq_arr,
        goals=np.stack(goals), task_ids=np.array(task_ids),
        successes=np.array(successes),
        source=f"{args.critic}_{args.actor}_s{args.seed}_e{args.restore_epoch}")
    print(f"wrote {args.out}: {len(obs_seqs)} episodes, "
          f"success rate {np.mean(successes):.3f}")


def score(args):
    import jax.numpy as jnp
    from env import make_config
    from impls.agents import Agent
    from impls.utils.flax_utils import restore_agent

    z = np.load(args.rollouts, allow_pickle=True)
    obs_seqs, goals = z["obs_seqs"], z["goals"]
    task_ids, successes = z["task_ids"], z["successes"]

    cfg = make_config(args.critic, args.actor, args.domain, args.override)
    ex_obs = jnp.asarray(np.asarray(obs_seqs[0])[:1])
    ex_act = jnp.zeros((1, int(cfg.get("probe_action_dim", 5))), jnp.float32)
    agent = Agent.create(args.critic, args.actor, args.seed, ex_obs, ex_act, cfg)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    params = {g: agent.net[g].params for g in agent.net}

    rhos, bumps = [], []
    for seq, g in zip(obs_seqs, goals):
        S = np.asarray(seq, np.float32)
        G = np.broadcast_to(g, S.shape)
        V = np.asarray(agent.critic.v_fn(params, jnp.asarray(S), {"goal": jnp.asarray(G)}),
                       np.float64)
        rhos.append(spearman(V, np.arange(len(V))))
        bumps.append(float((np.diff(V) < 0).mean()))
    rhos, bumps = np.array(rhos), np.array(bumps)
    ok = successes.astype(bool)

    report = {
        "scoring_critic": args.critic, "scoring_seed": args.seed,
        "scoring_epoch": args.restore_epoch,
        "rollout_source": str(z["source"]),
        "n_episodes": int(len(rhos)),
        "rollout_success_rate": float(ok.mean()),
        "p2_rho_median": float(np.median(rhos)),
        "p2_rho_mean": float(rhos.mean()),
        "p2_bump_rate_mean": float(bumps.mean()),
        "p2_rho_median_success": float(np.median(rhos[ok])) if ok.any() else float("nan"),
        "p2_rho_median_fail": float(np.median(rhos[~ok])) if (~ok).any() else float("nan"),
        "p2_bump_rate_success": float(bumps[ok].mean()) if ok.any() else float("nan"),
        "per_task_rho_median": {int(t): float(np.median(rhos[task_ids == t]))
                                for t in np.unique(task_ids)},
        "per_episode": {"rho": rhos.tolist(), "bump": bumps.tolist(),
                        "task_id": task_ids.tolist(), "success": successes.tolist()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_episode"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("collect", "score"), required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--actor", default="flowbc")
    ap.add_argument("--domain", default="cube-single")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore_path", required=True)
    ap.add_argument("--restore_epoch", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=10, help="collect: episodes per task_id")
    ap.add_argument("--rollouts", default="", help="score: the collect-mode npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "collect":
        collect(args)
    else:
        assert args.rollouts, "--mode score needs --rollouts"
        score(args)


if __name__ == "__main__":
    main()
