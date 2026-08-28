"""Zero-retraining probe: how much of rldp_density's eval performance is lost in the
goal->z eval interface?

Runs the standard 5-task x N-episode OGBench eval TWICE on one already-trained checkpoint:

  A. Control -- the shipped goal-embedding interface. The agent is evaluated exactly as
     impls/main.py evaluates it: the evaluator hands sample_actions the env's goal frame,
     and the critic's map_eval_cond hook re-derives z = project_z(B(goal)). Should sit
     ~at the logged 1M eval numbers (sanity anchor).

  B. Kernel-reward inference -- the critic's OWN reward->z readout (infer_cond,
     z proportional to mean_j r_j * B(s'_j)), fed the object-coordinate kernel of the
     task's goal evaluated at a fixed seeded pool of dataset next states:
         r_j = exp(-0.5 * ||(cube_xyz(s'_j) - cube_xyz(goal)) / sigma||^2),
     cube_xyz = obs[19:22], sigma = 0.4. The env goal frame is then NOT supplied at
     rollout time (a goal handed to sample_actions would be merged over cond_eval and
     re-encoded by map_eval_cond, silently overwriting the inferred z -- the exact
     failure mode impls/main.py::InferredCondAgent documents).

If B ~= A the eval interface is not where the gap lives (training-time damage);
if B >> A part of the gap is pure eval-interface loss.

No library code is touched: everything below drives the existing restore / eval APIs.

Launch (GPU 7, per the probe spec):
  CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      /dev/shm/.venv-jax/bin/python scripts/probe_eval_interface.py
"""
import os

# Device/EGL pin MUST precede any mujoco/ogbench import (same discipline as impls/main.py).
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp
import numpy as np

from env import get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent


class DropGoalAgent:
    """EVAL-ONLY view that rolls the stored cond_eval and refuses the env's goal frame.

    Same mechanism as impls/main.py::InferredCondAgent (which is the --eval_on_inferred_cond
    route): Agent.sample_actions merges any supplied goal over cond_eval and map_eval_cond
    then re-derives z from it, so the ONLY way to evaluate on an inferred z is to never
    forward a goal. Local copy so this probe does not import impls.main (absl flag module).
    """

    ROLLS_INFERRED_COND = True

    def __init__(self, agent):
        self.agent = agent
        self.critic = agent.critic
        self.cond_eval = agent.cond_eval
        self.goals_dropped = 0

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if goals is not None:
            self.goals_dropped += 1
        return self.agent.sample_actions(observations, goals=None, seed=seed,
                                         temperature=temperature)


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_arm(adapter, env, cfg, tasks, episodes, agent_for_task, label):
    """One full 5-task eval sweep. Reseeds the env RNG and the global numpy RNG first so
    both arms score the same episode draws (evaluate() seeds its actor RNG off np.random,
    and the env's reset randomization runs off the env's own np_random)."""
    np.random.seed(0)
    env.reset(seed=0)
    per_task = {}
    for tid in tasks:
        agent_t = agent_for_task(tid)
        row, _ = adapter.evaluate(agent_t, env, tid, cfg, episodes, 0)
        per_task[tid] = float(row["success"])
        dropped = getattr(agent_t, "goals_dropped", None)
        extra = f"  (goal frames dropped: {dropped})" if dropped is not None else ""
        print(f"[{label}] task {tid}: success {per_task[tid]:.3f}{extra}", flush=True)
    overall = float(np.mean(list(per_task.values())))
    print(f"[{label}] OVERALL: {overall:.4f}", flush=True)
    return per_task, overall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=os.path.join(
        _REPO, "exp", "cube-single-play-v0_rldp_density_flowbc_seed_0"))
    p.add_argument("--epoch", type=int, default=1_000_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--pool_size", type=int, default=10_000)
    p.add_argument("--pool_seed", type=int, default=12345)
    p.add_argument("--sigma", type=float, default=0.4)
    p.add_argument("--cube_slice", default="19:22",
                   help="obs columns holding the cube xyz (cube-single: block pos)")
    p.add_argument("--out", default=os.path.join(
        _REPO, "results", "probe_eval_interface_rldp_density_s0.json"))
    args = p.parse_args()

    ca, cb = (int(x) for x in args.cube_slice.split(":"))

    with open(os.path.join(args.run_dir, "config.json")) as f:
        run_meta = json.load(f)
    cfg = run_meta["config"]
    critic_name, actor_name, seed = run_meta["critic"], run_meta["actor"], run_meta["seed"]
    print(f"[probe] run: {args.run_dir}\n[probe] critic={critic_name} actor={actor_name} "
          f"seed={seed} epoch={args.epoch}", flush=True)

    # --- env + dataset, the same path impls/main.py uses ------------------------------
    np.random.seed(seed)
    adapter = get_adapter(cfg)
    env, train_ds, _val_ds = adapter.make_env_and_datasets(cfg)
    base_ds = train_ds                      # FBDataset (compact); pool rows come from here
    wrapped_ds = make_dataset(train_ds, cfg)
    ex = wrapped_ds.sample(1)

    # --- build + restore the agent ----------------------------------------------------
    agent = Agent.create(critic_name, actor_name, seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, args.run_dir, args.epoch)
    bnorm = float(jnp.linalg.norm(
        jnp.concatenate([jnp.ravel(x) for x in
                         jax.tree_util.tree_leaves(agent.net["backward"].params)])))
    print(f"[probe] restored; backward-tower param norm {bnorm:.3f}", flush=True)

    # --- fixed seeded pool of dataset next states (the marginal infer_cond declares) --
    rng = np.random.default_rng(args.pool_seed)
    if hasattr(base_ds, "valid_idxs"):
        idxs = rng.choice(base_ds.valid_idxs, size=args.pool_size, replace=False)
    else:
        idxs = rng.choice(base_ds.size, size=args.pool_size, replace=False)
    pool_batch = base_ds.sample(len(idxs), idxs=np.sort(idxs))
    pool_next = np.asarray(pool_batch["next_observations"], np.float32)
    pool_next_j = jnp.asarray(pool_next)
    print(f"[probe] pool: {pool_next.shape[0]} next_observations (seed {args.pool_seed})",
          flush=True)

    # --- per-task conditioning vectors: A (goal embedding) and B (kernel-reward) ------
    tasks = adapter.eval_tasks(env, cfg)
    assert tasks and tasks[0] is not None, (
        "probe expects the per-task_id OGBench eval (eval_aggregate off), matching the "
        "logged per-task eval.csv rows")
    z_A, z_B, cosines, kernel_stats = {}, {}, {}, {}
    for tid in tasks:
        _obs, info = env.reset(options=dict(task_id=tid, render_goal=False))
        goal = np.asarray(info["goal"], np.float32)
        # A-z: exactly what the shipped interface computes inside sample_actions.
        z_A[tid] = np.asarray(
            agent.critic.map_eval_cond(agent.net, {"goal": jnp.asarray(goal)})["z"])
        # B-z: the critic's own reward->z readout on the object-coordinate kernel.
        d = (pool_next[:, ca:cb] - goal[ca:cb]) / args.sigma
        r = np.exp(-0.5 * (d ** 2).sum(-1)).astype(np.float32)
        z_B[tid] = np.asarray(
            agent.critic.infer_cond(agent.net, pool_next_j, jnp.asarray(r))["z"])
        cosines[tid] = _cos(z_A[tid], z_B[tid])
        kernel_stats[tid] = dict(
            r_max=float(r.max()), r_mean=float(r.mean()),
            r_ess=float(r.sum() ** 2 / ((r ** 2).sum() + 1e-12)),
            goal_cube_xyz=[float(x) for x in goal[ca:cb]])
        print(f"[probe] task {tid}: cos(z_A, z_B) = {cosines[tid]:+.4f}  "
              f"kernel r_max {kernel_stats[tid]['r_max']:.3f} "
              f"ess {kernel_stats[tid]['r_ess']:.0f}", flush=True)

    # --- arm A: the shipped goal-embedding interface, agent used as-is ----------------
    a_tasks, a_overall = run_arm(adapter, env, cfg, tasks, args.episodes,
                                 lambda tid: agent, "A/goal-embed")

    # --- arm B: kernel-reward-inferred z, env goal refused ----------------------------
    b_tasks, b_overall = run_arm(
        adapter, env, cfg, tasks, args.episodes,
        lambda tid: DropGoalAgent(agent.replace(
            cond_eval={"z": jnp.asarray(z_B[tid])})),
        "B/kernel-infer")

    # --- report -----------------------------------------------------------------------
    print("\n================ PROBE SUMMARY ================")
    print(f"checkpoint: {args.run_dir}/params_{args.epoch}.pkl")
    print(f"{'task':>4}  {'A goal-embed':>12}  {'B kernel-z':>10}  {'cos(zA,zB)':>10}")
    for tid in tasks:
        print(f"{tid:>4}  {a_tasks[tid]:>12.3f}  {b_tasks[tid]:>10.3f}  {cosines[tid]:>+10.4f}")
    print(f"{'ALL':>4}  {a_overall:>12.4f}  {b_overall:>10.4f}")
    print("===============================================", flush=True)

    out = dict(
        run_dir=args.run_dir, epoch=args.epoch,
        checkpoint=os.path.join(args.run_dir, f"params_{args.epoch}.pkl"),
        episodes_per_task=args.episodes, pool_size=args.pool_size,
        pool_seed=args.pool_seed, sigma=args.sigma, cube_slice=args.cube_slice,
        overall=dict(A_goal_embed=a_overall, B_kernel_infer=b_overall),
        per_task={str(t): dict(A=a_tasks[t], B=b_tasks[t], cos_zA_zB=cosines[t],
                               **kernel_stats[t]) for t in tasks},
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
