#!/usr/bin/env python
"""scripts/probe_hiql_goalrep.py — H1 probe: does bilinear training degrade HIQL's goal_rep?

Hypothesis H1 (HIQL-bilinear brainstorm, 2026-07-07): in stock HIQL the value loss is the
SOLE gradient source for goal_rep; the bilinear head (_BilinearRepValue) routes that gradient
through phi(s)·psi(goal_rep([s;g]))/sqrt(d), which may starve/warp the rep that the low actor
consumes and the high actor generates into. rep_dim=10 and LengthNormalize'd in all runs, so
raw rank is bounded at 10 either way — the probe measures what the rep ENCODES, not just its
spectrum:

  1. spectrum      — eigenvalues + participation ratio of the rep covariance (10 dims)
  2. decodability  — ridge-regression R² (train/test split): rep -> goal cube xyz, and
                     rep -> relative cube displacement (goal cube − current cube)
  3. geometry      — Spearman corr between rep-space distances and goal-cube-xyz distances
                     across goals at FIXED state (the space the high actor must generate in)

Pairs are sampled at three horizons: subgoal-scale (t+10), in-trajectory future (uniform),
and random cross-trajectory — mirroring how goal_rep is queried (low adv / value goals).

Runs per-checkpoint on CPU (JAX_PLATFORMS=cpu; training GPUs stay untouched):
    JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/probe_hiql_goalrep.py \
        --checkpoint-dir /dev/shm/gciql_outputs/factored-fb/HIQL-State-Native-MLP-sg10/cube_single_state_hiql_native_mlp_sg10_s0 \
        --checkpoint-step 500000 --seed 0 --value-arch mlp --out analysis/hiql_goalrep/native_mlp_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "wandb_mode_shim"))

import numpy as np

DATA_DIR = Path.home() / ".ogbench" / "data"
GOAL_SLICES = {  # coordinates a subgoal rep should encode, per env family
    "cube-single-play-v0": "14:17",   # privileged cube xyz (matches eval_zcond / infer_ablation)
    "antmaze-medium-navigate-v0": "0:2",  # ant global xy (verified: dims 0:2 span the maze extent)
}


def load_hiql(checkpoint_dir: Path, step: int, seed: int, value_arch: str,
              value_latent_dim: int, obs_dim: int, action_dim: int,
              agent_kind: str = "hiql"):
    import jax.numpy as jnp
    from utils.flax_utils import restore_agent

    if agent_kind == "hiql_flow":  # flow-actor ckpts have *_vf networks in the tree
        from hiql_flow import HIQLFlowAgent as HIQLAgent, get_config
    else:
        from agents.hiql import HIQLAgent, get_config

    cfg = dict(get_config())
    if agent_kind == "hiql_flow":
        cfg["flow_levels"] = "both"
    cfg["value_arch"] = value_arch
    cfg["value_latent_dim"] = value_latent_dim
    cfg["encoder"] = None
    cfg["frame_stack"] = None
    ex_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_act = jnp.zeros((1, action_dim), dtype=jnp.float32)
    agent = HIQLAgent.create(seed, ex_obs, ex_act, cfg)
    return restore_agent(agent, str(checkpoint_dir), step)


def load_pairs(rng: np.random.Generator, n: int, dataset_path: Path, horizon_steps: int):
    """(s, g) pairs at three horizons from an OGBench dataset.

    OGBench datasets are fixed-length trajectories flagged by terminals;
    sample within-trajectory indices the way HGCDataset does (no crossing).
    """
    data = np.load(dataset_path)
    obs = data["observations"].astype(np.float32)
    terminals = data["terminals"]
    ends = np.where(terminals == 1)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    traj = np.stack([starts, ends], 1)  # inclusive [start, end]

    def sample(horizon: str):
        rows = traj[rng.integers(0, len(traj), n)]
        t = np.array([rng.integers(s, e) for s, e in rows])
        if horizon == "subgoal":
            g = np.minimum(t + horizon_steps, rows[:, 1])
        elif horizon == "future":
            g = np.array([rng.integers(ti + 1, e + 1) for ti, e in zip(t, rows[:, 1])])
        else:  # random cross-trajectory
            g = rng.integers(0, len(obs), n)
        return obs[t], obs[g]

    return {h: sample(h) for h in ("subgoal", "future", "random")}, obs


def participation_ratio(reps: np.ndarray) -> tuple[float, list[float]]:
    lam = np.linalg.eigvalsh(np.cov(reps.T))[::-1]
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12)), lam.tolist()


def ridge_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> float:
    """Held-out R² of ridge y~x (50/50 split), averaged over target dims."""
    n = len(x)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    xm, ym = x[tr].mean(0), y[tr].mean(0)
    xtr, ytr = x[tr] - xm, y[tr] - ym
    w = np.linalg.solve(xtr.T @ xtr + alpha * n * np.eye(x.shape[1]), xtr.T @ ytr)
    pred = (x[te] - xm) @ w + ym
    ss_res = np.square(y[te] - pred).sum(0)
    ss_tot = np.square(y[te] - y[te].mean(0)).sum(0) + 1e-12
    return float((1 - ss_res / ss_tot).mean())


def spearman_dist_corr(rng, rep_fn, obs: np.ndarray, goal_slice: slice,
                       n_states: int = 20, n_goals: int = 256) -> float:
    """Per fixed state s: Spearman(pairwise-ish) via rank corr of d(rep(s,gi), rep(s,gj))
    vs d(goal_coord(gi), goal_coord(gj)) over goal pairs; averaged across states."""
    from scipy.stats import spearmanr
    cors = []
    for _ in range(n_states):
        s = obs[rng.integers(0, len(obs))]
        goals = obs[rng.integers(0, len(obs), n_goals)]
        reps = rep_fn(np.repeat(s[None], n_goals, 0), goals)
        gx = goals[:, goal_slice]
        iu = np.triu_indices(n_goals, 1)
        dr = np.linalg.norm(reps[iu[0]] - reps[iu[1]], axis=-1)
        dg = np.linalg.norm(gx[iu[0]] - gx[iu[1]], axis=-1)
        cors.append(spearmanr(dr, dg).statistic)
    return float(np.mean(cors))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-arch", choices=("mlp", "bilinear"), required=True)
    ap.add_argument("--value-latent-dim", type=int, default=50)
    ap.add_argument("--agent", choices=("hiql", "hiql_flow"), default="hiql")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--dataset-name", default="cube-single-play-v0")
    ap.add_argument("--goal-slice", default=None,
                    help="obs slice 'a:b' of the goal coordinate; default from GOAL_SLICES")
    ap.add_argument("--action-dim", type=int, default=5)
    ap.add_argument("--subgoal-horizon", type=int, default=10,
                    help="steps ahead for the 'subgoal' horizon (10 cube sg10, 25 antmaze)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gs = args.goal_slice or GOAL_SLICES[args.dataset_name]
    lo, hi = (int(x) for x in gs.split(":"))
    goal_slice = slice(lo, hi)
    rng = np.random.default_rng(args.seed)
    pairs, obs = load_pairs(rng, args.pairs, DATA_DIR / f"{args.dataset_name}.npz", args.subgoal_horizon)
    obs_dim = obs.shape[1]

    agent = load_hiql(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                      args.seed, args.value_arch, args.value_latent_dim,
                      obs_dim, action_dim=args.action_dim, agent_kind=args.agent)
    print(f"[goalrep] restored {args.value_arch} s{args.seed} @{args.checkpoint_step}")

    def rep_fn(s, g, chunk=8192):
        import jax.numpy as jnp
        outs = []
        for i in range(0, len(s), chunk):
            x = jnp.concatenate([jnp.asarray(s[i:i + chunk]), jnp.asarray(g[i:i + chunk])], -1)
            outs.append(np.asarray(agent.network.select("goal_rep")(x)))
        return np.concatenate(outs)

    result = {"arch": args.value_arch, "seed": args.seed, "ckpt": str(args.checkpoint_dir),
              "step": args.checkpoint_step, "dataset": args.dataset_name, "goal_slice": gs,
              "horizons": {}}
    for h, (s, g) in pairs.items():
        reps = rep_fn(s, g)
        pr, spectrum = participation_ratio(reps)
        r2_goal = ridge_r2(reps, g[:, goal_slice])
        r2_rel = ridge_r2(reps, g[:, goal_slice] - s[:, goal_slice])
        result["horizons"][h] = {"participation_ratio": pr, "spectrum": spectrum,
                                 "r2_goal_cube_xyz": r2_goal, "r2_rel_cube_disp": r2_rel}
        print(f"  {h:8s} PR={pr:.2f}  R2(goal xyz)={r2_goal:.3f}  R2(rel disp)={r2_rel:.3f}")

    result["spearman_repdist_vs_goalxyz"] = spearman_dist_corr(rng, rep_fn, obs, goal_slice)
    print(f"  geometry Spearman(rep dist, goal-xyz dist) = {result['spearman_repdist_vs_goalxyz']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[goalrep] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
