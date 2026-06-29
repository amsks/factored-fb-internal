#!/usr/bin/env python
"""scripts/q_landscape_gciql.py — is the GCIQL rank-d bilinear critic's Q landscape
over ACTIONS multimodal, and does that multimodality concentrate after grasp?

Hypothesis (why flow > gaussian for the bilinear critic): at a fixed (s, g), the
map a -> Q(s,a,g) has several separated high-Q basins. A DDPG+BC gaussian actor,
pulled toward multimodal data, averages the modes and lands in the low-Q valley
between them; the flow actor can sit on a single mode. The decisive evidence is
ACTION-space multimodality, sharpest at/after grasp.

What this produces (per seed, under <out>/):
  1. q_slice_<phase>.png   — 2-D Q heatmaps over action-dim pairs at a representative
       state of each phase (approach / contact / grasp-transport), with the flow
       actor's K samples (scatter) and their MEAN (the gaussian proxy) overlaid.
       Local-maxima count = number of Q basins.
  2. q_gap_by_phase.png    — distribution of the "averaging gap"
       Q(best-of-K flow samples) - Q(mean flow sample), per phase. Large gap =>
       a unimodal mean is penalised => flow needed.
  3. energy_vs_dist.png    — V(s,g) (the action-free value = -energy) vs cube-goal
       distance, coloured by phase. The potential the policy descends.
  4. energy_field.png       — V binned over cube (x,y) for a fixed goal: the energy field.
  5. phi_pca_phase.png      — PCA of value-net phi(s) coloured by phase (representation
       structure; complementary, NOT the action-multimodality test).
  6. summary.json           — basin counts, per-phase gap stats.

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/q_landscape_gciql.py --checkpoint-dir <dir> --seed 0 \
       --value-latent-dim 50 --out-dir analysis/q_landscape/d50_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "scripts", REPO / "third_party" / "ogbench" / "impls",
           REPO / "tools" / "wandb_mode_shim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]
# phase thresholds (mirror evals/phase_probe.Thresholds)
EPS_REACH = 0.06   # m, hand-cube distance for contact
DELTA_LIFT = 0.03  # m, cube lifted above table => grasped
ACTION_DIM = 5
PHASES = ["approach", "contact", "grasp"]
PHASE_COLOR = {"approach": "#1f77b4", "contact": "#ff7f0e", "grasp": "#2ca02c"}


def label_phase(eff, cube, table_z):
    """Per-state phase from effector/cube geometry."""
    d = np.linalg.norm(eff - cube)
    dxy = np.linalg.norm(eff[:2] - cube[:2])
    lift = cube[2] - table_z
    if lift > DELTA_LIFT and dxy < EPS_REACH:
        return "grasp"                 # cube held/lifted (post-grasp / transport)
    if d < EPS_REACH:
        return "contact"               # hand at cube, not yet lifted
    return "approach"                  # pre-grasp


def collect_states(agent, env, n_episodes, max_steps=200):
    """Roll the flow policy; per step keep obs, eff, cube, grip, phase, dists."""
    import jax
    from evals.phase_probe import step_signals, episode_table_z, episode_goal
    rec = {k: [] for k in ("obs", "eff", "cube", "grip", "phase",
                           "cube_goal_dist", "goal_obs", "goal_xyz")}
    rng = jax.random.PRNGKey(0)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        table_z = episode_table_z(env)
        goal_obs = np.asarray(info.get("goal"), np.float32)
        goal_xyz = np.asarray(episode_goal(env), np.float64).reshape(3)
        for _t in range(max_steps):
            rng, sub = jax.random.split(rng)
            a = np.asarray(agent.sample_actions(observations=np.asarray(obs, np.float32),
                                                goals=goal_obs, seed=sub, temperature=0.0))
            obs, _r, term, trunc, info = env.step(np.clip(a, -1, 1))
            s = step_signals(info)
            eff = np.asarray(s["eff"], np.float64); cube = np.asarray(s["cube"], np.float64)
            rec["obs"].append(np.asarray(obs, np.float32))
            rec["eff"].append(eff); rec["cube"].append(cube)
            rec["grip"].append(float(np.asarray(s["grip"]).reshape(-1)[0]))
            rec["phase"].append(label_phase(eff, cube, table_z))
            rec["cube_goal_dist"].append(float(np.linalg.norm(cube - goal_xyz)))
            rec["goal_obs"].append(goal_obs); rec["goal_xyz"].append(goal_xyz)
            if term or trunc:
                break
    out = {k: np.asarray(v) for k, v in rec.items()}
    return out


def q_batch(agent, obs, goal, actions):
    """min(q1,q2) over a batch of actions at fixed (obs, goal). actions [M,5]."""
    import jax.numpy as jnp
    M = actions.shape[0]
    O = np.repeat(np.asarray(obs, np.float32)[None], M, 0)
    G = np.repeat(np.asarray(goal, np.float32)[None], M, 0)
    out = np.asarray(agent.network.select("critic")(
        jnp.asarray(O), jnp.asarray(G), jnp.asarray(actions, np.float32)))
    return out.min(0) if out.ndim == 2 else out


def v_batch(agent, obs, goal):
    """V(s,g) for batched obs (action-free value net). obs [N,28], goal [N,28]."""
    import jax.numpy as jnp
    return np.asarray(agent.network.select("value")(
        jnp.asarray(obs, np.float32), jnp.asarray(goal, np.float32))).reshape(-1)


def phi_batch(agent, obs, goal):
    import jax.numpy as jnp
    _v, phi, _psi = agent.network.select("value")(
        jnp.asarray(obs, np.float32), jnp.asarray(goal, np.float32), info=True)
    return np.asarray(phi)


def flow_samples(agent, obs, goal, k, base_seed=1234):
    """K stochastic flow-actor action samples at (obs, goal)."""
    import jax
    A = []
    for i in range(k):
        a = agent.sample_actions(observations=np.asarray(obs, np.float32),
                                 goals=np.asarray(goal, np.float32),
                                 seed=jax.random.PRNGKey(base_seed + i), temperature=1.0)
        A.append(np.asarray(a, np.float32))
    return np.stack(A)


def count_local_maxima(Z, rel_thresh=0.5):
    """# of strict local maxima on a 2-D grid above max - rel_thresh*range (8-neighbour)."""
    cut = Z.max() - rel_thresh * (Z.max() - Z.min() + 1e-9)
    cnt = 0
    H, W = Z.shape
    for i in range(H):
        for j in range(W):
            if Z[i, j] < cut:
                continue
            nb = Z[max(0, i-1):i+2, max(0, j-1):j+2]
            if Z[i, j] >= nb.max() - 1e-9:
                cnt += 1
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-latent-dim", type=int, default=50)
    ap.add_argument("--task", default="cube-single-play-singletask-task1-v0")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--ksamples", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ogbench
    from phase_probe_gciql import load_agent

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    agent = load_agent(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                       args.seed, value_arch="bilinear", value_latent_dim=args.value_latent_dim)
    env = ogbench.make_env_and_datasets(args.task, env_only=True)
    print(f"[q_landscape] collecting states ({args.episodes} eps)…")
    D = collect_states(agent, env, args.episodes)
    n = len(D["phase"])
    print(f"[q_landscape] {n} states  phase counts: "
          + ", ".join(f"{p}={int((D['phase']==p).sum())}" for p in PHASES))

    summary = {"seed": args.seed, "d": args.value_latent_dim, "task": args.task,
               "n_states": n, "phase_counts": {p: int((D["phase"] == p).sum()) for p in PHASES}}

    # ---- per-state averaging gap: Q(best of K flow) - Q(mean flow) -------------
    gaps = {p: [] for p in PHASES}
    rng = np.random.default_rng(0)
    for p in PHASES:
        idx = np.where(D["phase"] == p)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(40, len(idx)), replace=False)
        for i in pick:
            A = flow_samples(agent, D["obs"][i], D["goal_obs"][i], args.ksamples)
            qA = q_batch(agent, D["obs"][i], D["goal_obs"][i], A)
            qmean = q_batch(agent, D["obs"][i], D["goal_obs"][i], A.mean(0)[None])[0]
            gaps[p].append(float(qA.max() - qmean))
    summary["gap_by_phase"] = {p: {"mean": float(np.mean(g)) if g else None,
                                   "median": float(np.median(g)) if g else None,
                                   "n": len(g)} for p, g in gaps.items()}

    # ---- 2-D Q heatmaps for a representative (highest-gap) state per phase ------
    DIM_PAIRS = [(0, 1), (2, 4)]
    lin = np.linspace(-1, 1, args.grid)
    GX, GY = np.meshgrid(lin, lin)
    basin_counts = {}
    for p in PHASES:
        idx = np.where(D["phase"] == p)[0]
        if len(idx) == 0:
            continue
        # representative = the state with the largest averaging gap (where it matters)
        pick = rng.choice(idx, size=min(40, len(idx)), replace=False)
        gvals = []
        for i in pick:
            A = flow_samples(agent, D["obs"][i], D["goal_obs"][i], 32)
            qA = q_batch(agent, D["obs"][i], D["goal_obs"][i], A)
            qmean = q_batch(agent, D["obs"][i], D["goal_obs"][i], A.mean(0)[None])[0]
            gvals.append(qA.max() - qmean)
        s_i = pick[int(np.argmax(gvals))]
        obs_i, goal_i = D["obs"][s_i], D["goal_obs"][s_i]
        a_flow = flow_samples(agent, obs_i, goal_i, args.ksamples)  # [K,5]
        a_mean = a_flow.mean(0)

        fig, axes = plt.subplots(1, len(DIM_PAIRS), figsize=(5.2 * len(DIM_PAIRS), 4.6))
        if len(DIM_PAIRS) == 1:
            axes = [axes]
        nb = []
        for ax, (di, dj) in zip(axes, DIM_PAIRS):
            base = a_mean.copy()
            grid_actions = np.repeat(base[None], args.grid * args.grid, 0)
            grid_actions[:, di] = GX.ravel(); grid_actions[:, dj] = GY.ravel()
            Z = q_batch(agent, obs_i, goal_i, grid_actions).reshape(args.grid, args.grid)
            nb.append(count_local_maxima(Z))
            im = ax.contourf(GX, GY, Z, levels=24, cmap="viridis")
            ax.scatter(a_flow[:, di], a_flow[:, dj], s=10, c="white",
                       edgecolors="k", lw=0.3, alpha=0.7, label="flow samples")
            ax.scatter([a_mean[di]], [a_mean[dj]], s=160, marker="*", c="red",
                       edgecolors="k", label="mean (gaussian proxy)")
            ax.set_xlabel(f"action[{di}]"); ax.set_ylabel(f"action[{dj}]")
            ax.set_title(f"Q  dims({di},{dj})  basins={nb[-1]}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].legend(loc="upper right", fontsize=7)
        fig.suptitle(f"{p}: Q(s,a,g) over actions  (d={args.value_latent_dim}, gap={max(gvals):.1f})")
        fig.tight_layout()
        fig.savefig(out / f"q_slice_{p}.png", dpi=130); plt.close(fig)
        basin_counts[p] = nb
    summary["basins_by_phase"] = basin_counts

    # ---- gap-by-phase boxplot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [gaps[p] for p in PHASES if gaps[p]]
    labs = [p for p in PHASES if gaps[p]]
    ax.boxplot(data, labels=labs, showmeans=True)
    ax.set_ylabel("Q(best-of-K flow) − Q(mean flow)")
    ax.set_title(f"Action-averaging gap by phase (d={args.value_latent_dim})")
    fig.tight_layout(); fig.savefig(out / "q_gap_by_phase.png", dpi=130); plt.close(fig)

    # ---- energy (V = -energy) vs cube-goal distance ---------------------------
    V = v_batch(agent, D["obs"], D["goal_obs"])
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for p in PHASES:
        m = D["phase"] == p
        if m.any():
            ax.scatter(D["cube_goal_dist"][m], V[m], s=8, alpha=0.5,
                       c=PHASE_COLOR[p], label=p)
    ax.set_xlabel("cube → goal distance (m)"); ax.set_ylabel("V(s,g)  (= −energy)")
    ax.set_title(f"Value/energy vs distance-to-goal (d={args.value_latent_dim})")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "energy_vs_dist.png", dpi=130); plt.close(fig)

    # ---- energy field over cube (x,y) -----------------------------------------
    cube_xy = D["cube"][:, :2]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sc = ax.scatter(cube_xy[:, 0], cube_xy[:, 1], c=V, s=14, cmap="magma")
    gx = D["goal_xyz"][0]
    ax.scatter([gx[0]], [gx[1]], marker="*", s=240, c="cyan", edgecolors="k", label="goal")
    held = D["phase"] == "grasp"
    if held.any():
        ax.scatter(cube_xy[held, 0], cube_xy[held, 1], s=4, c="lime", label="grasp")
    ax.set_xlabel("cube x"); ax.set_ylabel("cube y"); ax.legend(fontsize=7)
    ax.set_title("Energy field: V over cube position")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="V(s,g)")
    fig.tight_layout(); fig.savefig(out / "energy_field.png", dpi=130); plt.close(fig)

    # ---- phi(s) PCA coloured by phase (complementary rep-structure view) -------
    phi = phi_batch(agent, D["obs"], D["goal_obs"])
    phc = phi - phi.mean(0)
    U, S, Vt = np.linalg.svd(phc, full_matrices=False)
    pcs = phc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for p in PHASES:
        m = D["phase"] == p
        if m.any():
            ax.scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.5, c=PHASE_COLOR[p], label=p)
    ev = (S[:2] ** 2 / (S ** 2).sum())
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"phi(s) PCA by phase (d={args.value_latent_dim})")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "phi_pca_phase.png", dpi=130); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[q_landscape] gap_by_phase:",
          {p: round(summary["gap_by_phase"][p]["median"] or 0, 2) for p in PHASES})
    print("[q_landscape] basins_by_phase:", basin_counts)
    print(f"[q_landscape] wrote figures + summary.json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
