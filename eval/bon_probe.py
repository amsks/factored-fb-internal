"""Best-of-N re-ranking probe — the critic as a pure action selector (no actor training).

The decisive isolation of the guidance mechanism (phenomenon.tex Ph.V): freeze ONE
proposal distribution pi_0 (a pure-BC flow actor from a bcflow run), and at every env step
draw N actions from it, score them with a TRAINED critic's Q(s, a, g), execute the argmax.
No Q-gradient ever touches a policy, so nothing about actor optimization, exploitation, or
BC regularization can confound the result: success as a function of N measures exactly the
quality of the critic's action ranking on the proposal's support, on the eval state
distribution. N=1 executes the raw proposal (the BC floor; a built-in control that must
reproduce the bcflow campaign numbers).

Pre-registered predictions (same protocol on cube AND Metaworld):
  - bilinear GCIQL's success rises monotonically with N;
  - monolithic GCIQL rises less and flattens or degrades at large N (its ranking is only
    coherent near its own policy's states — P2 2x2);
  - the biql-gciql gap widens with N. If the same curves appear on both benchmarks, the
    completion phenomenon holds in both.

    python -m eval.bon_probe --domain cube-single \
        --proposal_critic qrl --proposal_actor bcflow \
        --proposal_override "qrl_components=8 qrl_dim_per_component=8" \
        --proposal_path <bcflow_run_dir> --proposal_epoch 500000 \
        --critic iql_bilinear --actor flowbc --critic_override "value_latent_dim=50" \
        --critic_path <flowbc_run_dir> --critic_epoch 500000 \
        --n_list 1,8,64 --episodes 10 --seed 0 --out bon_biql_s0.json
"""
import argparse
import json
import os

import numpy as np


def build_agent(critic, actor, domain, override, seed, path, epoch, ex_obs, ex_act):
    import jax.numpy as jnp
    from env import make_config
    from impls.agents import Agent
    from impls.utils.flax_utils import restore_agent

    # make_config takes a LIST of dotlist entries; a bare string would be iterated
    # character-by-character by OmegaConf and silently dropped.
    cfg = make_config(critic, actor, domain, override.split() if isinstance(override, str) else override)
    agent = Agent.create(critic, actor, seed, jnp.asarray(ex_obs), jnp.asarray(ex_act), cfg)
    return restore_agent(agent, path, epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--proposal_critic", required=True)
    ap.add_argument("--proposal_actor", default="bcflow")
    ap.add_argument("--proposal_override", default="")
    ap.add_argument("--proposal_path", required=True)
    ap.add_argument("--proposal_epoch", type=int, required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--actor", default="flowbc")
    ap.add_argument("--critic_override", default="")
    ap.add_argument("--critic_path", required=True)
    ap.add_argument("--critic_epoch", type=int, required=True)
    ap.add_argument("--goal_mode", default="fixed", choices=["fixed", "synth"],
                    help="unified (MW) only: 'fixed' = the campaign's dataset goal frame; "
                         "'synth' = per-episode goal built from the episode's own first obs "
                         "with hand+object slots set to the episode's true target (repairs "
                         "the Ph.II context mismatch; obs layout verified: obj=4:7, "
                         "prev copies at +18, target=36:39, and at success frames obj≈target)")
    ap.add_argument("--n_list", default="1,8,64")
    ap.add_argument("--episodes", type=int, default=10,
                    help="episodes per task (ogbench) or total (unified) per N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from env import make_config, get_adapter
    from impls.utils.datasets import make_dataset

    # Env + example batch come from the PROPOSAL's config (same domain for both agents).
    pcfg = make_config(args.proposal_critic, args.proposal_actor, args.domain,
                      args.proposal_override.split())
    adapter = get_adapter(pcfg)
    env, train_ds, _ = adapter.make_env_and_datasets(pcfg)
    ds = make_dataset(train_ds, pcfg)
    ex = ds.sample(1)
    ex_obs, ex_act = np.asarray(ex["observations"]), np.asarray(ex["actions"])

    proposal = build_agent(args.proposal_critic, args.proposal_actor, args.domain,
                           args.proposal_override, args.seed,
                           args.proposal_path, args.proposal_epoch, ex_obs, ex_act)
    scorer = build_agent(args.critic, args.actor, args.domain, args.critic_override,
                         args.seed, args.critic_path, args.critic_epoch, ex_obs, ex_act)
    sparams = {g: scorer.net[g].params for g in scorer.net}

    def q_of(obs_t, acts, goal_t):
        q = scorer.critic.q_fn(sparams, obs_t, {"goal": goal_t}, acts)
        return q[0] if isinstance(q, tuple) else q
    q_of = jax.jit(q_of)

    is_ogbench = adapter.name == "ogbench"
    if not is_ogbench:
        from env.unified import seeded_reset
        goal_fixed = adapter._goal    # the campaign's eval goal: highest-reward next_obs

    def run_episode(N, ep_idx, task_id, key):
        if is_ogbench:
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = np.asarray(info["goal"], np.float32)
            horizon = 10**9               # env terminates/truncates itself
        else:
            obs, info = seeded_reset(env, pcfg["task"], ep_idx)
            if args.goal_mode == "synth":
                # Per-episode goal: this episode's own obs with hand+object (and their
                # prev-frame copies) moved to the episode's true target. Context dims
                # 36:39 stay the episode's own -> the (s, g) pair is context-consistent,
                # unlike the fixed dataset frame. The construction now lives in
                # env/unified.py (shared with evaluate()'s eval_goal_mode=synth) so the
                # probe and the trained-policy eval cannot drift apart; it is bit-identical
                # to the slot list this probe verified on Metaworld.
                from env.unified import synth_goal_frame
                goal = synth_goal_frame(pcfg["task"], obs)
            else:
                goal = goal_fixed
            mask_dims = pcfg.get("goal_mask_dims")
            if mask_dims:
                goal = np.array(goal, copy=True)
                goal[list(mask_dims)] = 0.0
            horizon = getattr(adapter, "_horizon", None) or int(pcfg.get("horizon", 100))
        goal_t = jnp.asarray(np.broadcast_to(goal, (N, goal.shape[-1])))
        success = bool(np.any(np.asarray(info.get("success", 0), dtype=float)))
        steps = 0
        while steps < horizon:
            key, sub = jax.random.split(key)
            obs_t = jnp.asarray(np.broadcast_to(np.asarray(obs, np.float32),
                                                (N, obs.shape[-1])))
            acts = proposal.sample_actions(observations=obs_t, goals=goal_t,
                                           seed=sub, temperature=0)
            if N > 1:
                a = np.asarray(acts)[int(np.argmax(np.asarray(q_of(obs_t, acts, goal_t))))]
            else:
                a = np.asarray(acts)[0]
            obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
            success = success or bool(np.any(np.asarray(info.get("success", 0), dtype=float)))
            steps += 1
            if bool(np.any(np.asarray(term))) or bool(np.any(np.asarray(trunc))):
                break
        return success

    results = {}
    key = jax.random.PRNGKey(args.seed * 1000 + 29)
    for N in [int(x) for x in args.n_list.split(",")]:
        succ = []
        if is_ogbench:
            n_tasks = len(env.unwrapped.task_infos)
            for tid in range(1, n_tasks + 1):
                for ep in range(args.episodes):
                    key, sub = jax.random.split(key)
                    succ.append(run_episode(N, ep, tid, sub))
        else:
            for ep in range(args.episodes):
                key, sub = jax.random.split(key)
                succ.append(run_episode(N, ep, None, sub))
        results[N] = float(np.mean(succ))
        print(f"N={N}: success {results[N]:.3f} over {len(succ)} episodes", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "domain": args.domain, "critic": args.critic,
            "critic_path": args.critic_path, "critic_epoch": args.critic_epoch,
            "proposal": f"{args.proposal_critic}_{args.proposal_actor}",
            "proposal_path": args.proposal_path, "seed": args.seed,
            "episodes_per_point": args.episodes, "goal_mode": args.goal_mode,
            "success_by_n": results,
        }, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
