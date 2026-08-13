#!/usr/bin/env python
"""scripts/rescuing_fb/t9_advantage_rank.py -- T9: rank of the TRUE advantage
(PAPER/ogbench-sota/implementations.md item 6).

Layer 0 of the bilinear-mechanism claim says the win is a LOW-RANK story: a rank-d
bilinear head with d >~ 28 is sufficient because the success-relevant advantage
A*(s,a,g) is itself low rank over the (s,a) x g grid. This probe measures that rank
directly, from the SIMULATOR, with an ACTOR-INDEPENDENT action set -- so the number
is the rank of the TRUE advantage, not the rank of any particular actor's Q.

Method (reuses the RLBrew cf-probe machinery):
  * probe states: sample_grasps(...) -- ~200 in-hand grasps from the play data, the
    same set the counterfactual-value probe uses (value's training distribution).
  * action set at state s (ACTOR-INDEPENDENT): the dataset-NN action cloud, exactly
    as t3_offpeak_multienv.py builds it -- the dataset action at s plus the actions of
    the K_NN nearest-neighbour dataset states in whitened (per-dim standardized) obs
    space. No policy, no BC flow -> the ranks below are not an artifact of an actor.
  * A*(s,a,g) via K-action counterfactual rollouts: establish the grasp (set_state +
    hold the gripper closed, as in _establish_grasp), execute the candidate FIRST
    action a, then roll a fixed continuation policy pi_ref to the goal g; the
    counterfactual value is the success rate over `cont_seeds` continuations. The
    advantage centers out the state-goal baseline:
        cfQ(s,a,g)  = P[reach g | a_0=a, then a_t~pi_ref]
        Vbar(s,g)   = mean_a cfQ(s,a,g)            (baseline over the SAME action set)
        A*(s,a,g)   = cfQ(s,a,g) - Vbar(s,g)
    pi_ref is the SAME for every (a,g), so it does not create (s,a)-structure; it only
    supplies a common continuation. Documented as a design choice.
  * assemble the [(s,a)] x [g] matrix of A* and report its epsilon-rank spectrum:
    singular values, epsilon-rank at several epsilon (relative-to-sigma_max AND
    energy-cutoff conventions), and the participation ratio PR = (sum s^2)^2 / sum s^4.

Pre-registered prediction: epsilon-rank(A*) <= ~28 at epsilon = the success-relevant
resolution (matches the d>=28 flat band from the bilinear-d sweep). A high-rank result
would break Layer 0's sufficiency story for the *advantage* (and imply the win is even
more Layer-1 / extraction dependent). Either way it calibrates the d floor.

CPU-HEAVY. Run under the jax venv, cores capped (shared box):
  taskset -c 0-47 env JAX_PLATFORMS=cpu MUJOCO_GL=egl OMP_NUM_THREADS=4 \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/t9_advantage_rank.py \\
    --continuation-run-dir <bilinear-d50 run dir> --step 500000 \\
    --n-states 200 --k-nn 64 --n-goals 5 --cont-seeds 1 --max-steps 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]


def whiten(obs: np.ndarray) -> np.ndarray:
    """Per-dim standardization (matches t3's whitened obs space)."""
    mu = obs.mean(0); sd = obs.std(0) + 1e-6
    return (obs - mu) / sd


def knn_actions(probe_w, pool_w, pool_actions, k, chunk=200_000):
    """For each probe (whitened obs), the actions of its k nearest dataset states.
    Returns [n_probe, k, act_dim]. Actor-independent (dataset only)."""
    nq = len(probe_w)
    best_d = np.full((nq, k), np.inf, np.float32)
    best_i = np.zeros((nq, k), np.int64)
    q_sq = (probe_w ** 2).sum(1)[:, None]
    for start in range(0, len(pool_w), chunk):
        block = pool_w[start:start + chunk]
        d2 = q_sq - 2.0 * (probe_w @ block.T) + (block ** 2).sum(1)[None, :]
        kk = min(k, d2.shape[1])
        top = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        cand_d = np.take_along_axis(d2, top, 1).astype(np.float32)
        cand_i = (top + start).astype(np.int64)
        all_d = np.concatenate([best_d, cand_d], 1)
        all_i = np.concatenate([best_i, cand_i], 1)
        sel = np.argpartition(all_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(all_d, sel, 1)
        best_i = np.take_along_axis(all_i, sel, 1)
    return pool_actions[best_i]                       # [n_probe, k, act_dim]


def eps_rank_spectrum(M: np.ndarray) -> dict:
    """SVD-based rank diagnostics of the (s,a) x g advantage matrix."""
    # center columns (advantage is already state-goal centered per row-group; also
    # de-mean each goal column so a constant goal-offset does not inflate rank-1).
    Mc = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    s = np.asarray(s, np.float64)
    smax = float(s[0]) if len(s) and s[0] > 0 else 1.0
    energy = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-12)
    def erank_rel(eps):   # # singular values with sigma_i/sigma_max > eps
        return int((s / smax > eps).sum())
    def erank_energy(eps):  # smallest k capturing >= 1-eps of the energy
        return int(np.searchsorted(energy, 1.0 - eps) + 1)
    pr = float((s ** 2).sum() ** 2 / max(((s ** 4).sum()), 1e-12))   # participation ratio
    return {
        "singular_values": s.tolist(),
        "sigma_max": smax,
        "eps_rank_relative": {str(e): erank_rel(e) for e in (0.1, 0.05, 0.02, 0.01)},
        "eps_rank_energy": {str(e): erank_energy(e) for e in (0.1, 0.05, 0.02, 0.01)},
        "participation_ratio": pr,
        "shape": list(M.shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--continuation-run-dir", default="",
                    help="run dir of the continuation policy pi_ref (e.g. bilinear-d50). "
                         "Empty + --continuation hold/zero for the machinery smoke.")
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--continuation", choices=["agent", "hold", "zero"], default="agent")
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--k-nn", type=int, default=64, help="actor-independent action cloud size")
    ap.add_argument("--n-goals", type=int, default=5, help="goal columns (>5 adds sampled dataset-state goals)")
    ap.add_argument("--cont-seeds", type=int, default=1, help="continuations averaged per (s,a,g)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--pool-cap", type=int, default=200_000, help="dataset rows used for the NN pool")
    ap.add_argument("--out", default=str(REPO / "analysis/rescuing-fb/t9_advantage_rank.json"))
    ap.add_argument("--fig", default=str(REPO / "analysis/rescuing-fb/t9_advantage_rank.png"))
    args = ap.parse_args()

    import mujoco
    import jax
    import ogbench
    from scripts.value.counterfactual_value_probe import sample_grasps
    from scripts.value.counterfactual_value_probe_jax import _establish_grasp

    # ---- probe states + goals ----
    obs_g, phys_g = sample_grasps(args.data_path, args.domain, args.n_states)
    obs_g = np.asarray(obs_g, np.float32); phys_g = np.asarray(phys_g)
    n_states = len(obs_g)

    # dataset pool for the actor-independent NN action cloud.
    _e0, ds, _ = ogbench.make_env_and_datasets(EVAL_TASKS[0])
    pool_obs = np.asarray(ds["observations"], np.float32)[: args.pool_cap]
    pool_act = np.asarray(ds["actions"], np.float32)[: args.pool_cap]
    mu = pool_obs.mean(0); sd = pool_obs.std(0) + 1e-6
    pool_w = (pool_obs - mu) / sd
    probe_w = (obs_g - mu) / sd
    cloud = knn_actions(probe_w, pool_w, pool_act, args.k_nn)     # [n_states, k_nn, A]
    act_dim = cloud.shape[-1]
    print(f"[t9] {n_states} probe states x {args.k_nn}-action cloud (dim {act_dim})", flush=True)

    # goals: the 5 singletask goal obs, + optional sampled dataset-state goals.
    goal_obs = []
    for task in EVAL_TASKS:
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs.append(np.asarray(info["goal"], np.float32))
        if hasattr(env, "close"):
            env.close()
    goal_obs = goal_obs[: min(args.n_goals, 5)]
    if args.n_goals > 5:
        extra = np.random.default_rng(0).choice(len(pool_obs), args.n_goals - 5, replace=False)
        goal_obs += [pool_obs[i] for i in extra]
    goal_obs = np.stack(goal_obs)                                 # [n_goals, obs]
    n_goals = len(goal_obs)

    # ---- continuation policy pi_ref ----
    cont_agent = None
    if args.continuation == "agent":
        assert args.continuation_run_dir, "--continuation agent needs --continuation-run-dir"
        from scripts.value.repsep_extract_jax import _load_gciql
        cont_agent, _cfg, _env_name = _load_gciql(args.continuation_run_dir, args.step)

    def pi_ref(obs_np, goal_np, rng):
        if args.continuation == "zero":
            return np.zeros(act_dim, np.float32)
        if args.continuation == "hold":
            return None                                          # caller holds a_0
        a = cont_agent.sample_actions(observations=np.asarray(obs_np, np.float32),
                                      goals=np.asarray(goal_np, np.float32),
                                      seed=rng, temperature=0.0)
        return np.clip(np.asarray(a), -1, 1)

    env = ogbench.make_env_and_datasets(args.domain, env_only=True)
    u = env.unwrapped

    def rollout(qpos, goal_task_obs, a0, seed):
        """Establish grasp, execute a0, continue with pi_ref; success = any info['success']."""
        _o = _establish_grasp(u, qpos, mujoco)
        obs = _o
        rng = jax.random.PRNGKey(seed)
        succ = False
        for t in range(args.max_steps):
            if t == 0 and a0 is not None:
                a = a0
            else:
                rng, sub = jax.random.split(rng)
                a = pi_ref(obs, goal_task_obs, sub)
                if a is None:            # 'hold' continuation: repeat a0
                    a = a0
            ob_, _r, term, trunc, info = env.step(np.clip(a, -1, 1))
            obs = np.asarray(ob_, np.float32)
            succ = succ or bool(info.get("success", False))
            if term or trunc:
                break
        return float(succ)

    # ---- assemble A*[(s,a), g] ----
    import time
    t0 = time.time()
    cfQ = np.zeros((n_states, args.k_nn, n_goals), np.float32)
    Vbar = np.zeros((n_states, n_goals), np.float32)
    for i in range(n_states):
        for gj in range(n_goals):
            # baseline: continuation from step 0 (no counterfactual a0)
            vb = np.mean([rollout(phys_g[i], goal_obs[gj], None, seed=1000 * gj + s)
                          for s in range(args.cont_seeds)]) if args.continuation == "agent" else 0.0
            Vbar[i, gj] = vb
            for ai in range(args.k_nn):
                cfQ[i, ai, gj] = np.mean([
                    rollout(phys_g[i], goal_obs[gj], cloud[i, ai], seed=7 * ai + 1000 * gj + s)
                    for s in range(args.cont_seeds)])
        if (i + 1) % 10 == 0:
            print(f"[t9] {i+1}/{n_states} states ({time.time()-t0:.0f}s)", flush=True)

    # advantage: center each (state,goal) column over the SAME action set.
    A = cfQ - cfQ.mean(axis=1, keepdims=True)                    # [n_states, k_nn, n_goals]
    M = A.reshape(n_states * args.k_nn, n_goals)                 # rows=(s,a), cols=g
    spec = eps_rank_spectrum(M)
    result = {
        "meta": {"n_states": n_states, "k_nn": args.k_nn, "n_goals": n_goals,
                 "cont_seeds": args.cont_seeds, "max_steps": args.max_steps,
                 "continuation": args.continuation, "run_dir": args.continuation_run_dir,
                 "action_set": "dataset-NN cloud (actor-independent)",
                 "spec": "PAPER/ogbench-sota/implementations.md item 6"},
        "cfQ_mean": float(cfQ.mean()), "Vbar_mean": float(Vbar.mean()),
        "spectrum": spec,
        "prediction": "eps-rank(A*) <= ~28 at the success-relevant resolution (d>=28 flat band)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = np.asarray(spec["singular_values"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.arange(1, len(s) + 1), s / spec["sigma_max"], "o-", ms=4)
        ax.axvline(28, color="crimson", ls="--", lw=1, label="d=28 flat band")
        ax.set_xlabel("index"); ax.set_ylabel(r"$\sigma_i/\sigma_{max}$")
        ax.set_title(f"T9 true-advantage spectrum  (PR={spec['participation_ratio']:.1f})")
        ax.legend(); fig.tight_layout(); fig.savefig(args.fig, dpi=120)
        print(f"[t9] fig -> {args.fig}", flush=True)
    except Exception as e:
        print(f"[t9] figure skipped: {e}", flush=True)

    er = spec["eps_rank_relative"]
    print(f"[t9] DONE ({time.time()-t0:.0f}s): eps-rank(rel) 0.05->{er['0.05']} 0.01->{er['0.01']} "
          f"PR={spec['participation_ratio']:.1f} shape={spec['shape']} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
