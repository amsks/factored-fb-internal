"""Fig: the critic's belief drifting away from reality, frame by frame, during a carry.

A play segment in which the arm picks the cube up and carries it. Six recorded frames.

    row 1   the rendered dataset frames -- recorded states, identical for both critics
    row 2   FB's predicted future cube position, as a blob on the table
    row 3   Density-FB's, same frames, same goal

Nothing here is a rollout and nothing is a policy: the critic reads the DATASET's own
observation and the DATASET's own action at each frame, so there is no behaviour to argue
about. The only thing that differs between rows 2 and 3 is the estimator.

Each blob is normalised to its OWN maximum. That is deliberate: Density-FB's measure is
sharper (N_eff ~31 vs ~335) and this figure is not about sharpness, it is about POSITION --
a blob in the wrong place is wrong at any temperature. Per-panel normalisation removes the
sharpness advantage from the comparison entirely.

Markers: the true cube position, and the cube position DECODED from the critic's backward
embedding by a linear probe fit on dataset rows. The decoded marker is the representation
speaking for itself -- if it lags the true cube once the carry starts, the cube factor is
not being tracked.

DEVIATION from the brief, stated because it matters: the task cannot be given to both
methods as reward samples. FB has `infer_cond` (z = sum r*B), but these Density-FB
checkpoints were trained goal-conditioned (`measure_density_z: false`) and the config notes
a checkpoint trained with it off cannot be restored with it on. So both critics get the
IDENTICAL goal observations instead -- same rows, same draws, same everything else.
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, MASS_CMAP, task_goals
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ROOT = Path("analysis/cube_bottleneck/cube-single-play-v0")

# TOP-DOWN VIEW. The env's x is DEPTH (workspace bounds 0.25-0.60 m, ~35 cm) and y is
# LATERAL (-0.35 to 0.35 m, ~70 cm), so a plot with x across the page draws the table
# tall-and-narrow when it is physically wide-and-shallow. `world_to_plot` puts lateral
# across and depth up -- the standard overhead view, matching the rendered frames -- and
# every position on these panels goes through it so the blobs and the markers cannot
# disagree about the frame.
def world_to_plot(xy):
    xy = np.asarray(xy)
    return np.stack([xy[..., 1], xy[..., 0]], axis=-1)
CUBE_OBS, EFF_OBS = slice(19, 22), slice(12, 15)
ORDER = ["fb_ref", "density_fb_state"]


def b_embed(arm, obs):
    """The backward tower's embedding of a batch of observations, per critic."""
    import jax.numpy as jnp
    c, net = arm.agent.critic, arm.agent.net
    x = jnp.asarray(obs)
    if arm.critic.startswith("density_fb"):
        return np.asarray(c._B(net["density_b"].params, c._b_query(x)))
    return np.asarray(c._B(net["backward"].params, x))


def fit_probe(arm, obs_pool, target, ridge=1e-3):
    """Least-squares probe B(s) -> target, with a held-out R^2 so the marker is honest."""
    E = b_embed(arm, obs_pool)
    E = np.concatenate([E, np.ones((len(E), 1), np.float32)], 1)
    n = len(E) * 4 // 5
    A_ = E[:n].T @ E[:n] + ridge * np.eye(E.shape[1])
    W = np.linalg.solve(A_, E[:n].T @ target[:n])
    pred = E[n:] @ W
    ss = ((target[n:] - pred) ** 2).sum(0) / ((target[n:] - target[n:].mean(0)) ** 2).sum(0)
    return W, float(1 - ss.mean())


def predicted_cube_mass(arm, obs_row, act_row, goals, pool_obs):
    from scripts.probes.cube_bottleneck.fig_reach_map import batched_mass
    p, neg = batched_mass(arm, obs_row, act_row, goals, pool_obs)
    return p.mean(0), neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=65090, help="dataset row of the carry")
    ap.add_argument("--end", type=int, default=65130)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--probe-rows", type=int, default=20000)
    ap.add_argument("--bins", type=int, default=18)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = BK.load_raw(args.env)
    rows = np.linspace(args.start, args.end - 1, args.frames).astype(int)
    obs_f = np.asarray(raw["observations"][rows], np.float32)
    act_f = np.asarray(raw["actions"][rows], np.float32)
    true_xy = world_to_plot(obs_f[:, CUBE_OBS][:, :2] * CM)
    lifted = (np.asarray(raw["qpos"])[rows][:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH) & \
             (obs_f[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH)
    print(f"frames {rows.tolist()}  lifted {lifted.astype(int).tolist()}")

    pool_obs = dict(np.load(ROOT / "pool.npz"))["obs"]
    pool_xy = world_to_plot(pool_obs[:, CUBE_OBS][:, :2] * CM)
    lo, hi = np.percentile(pool_xy, [1, 99], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    # ---- render the recorded frames (physics only; the critic never sees these) ----
    import ogbench
    env = ogbench.make_env_and_datasets(args.env, env_only=True, render_mode="rgb_array")
    env.reset(seed=0, options=dict(task_id=args.task, render_goal=False))
    imgs = []
    for r in rows:
        R.set_state(env, np.asarray(raw["qpos"][r]), np.asarray(raw["qvel"][r]))
        R.forward(env)
        imgs.append(np.asarray(env.render()))
    print(f"rendered {len(imgs)} frames {imgs[0].shape}")

    probe_idx = np.random.default_rng(0).choice(len(raw["observations"]),
                                                size=args.probe_rows, replace=False)
    obs_p = np.asarray(raw["observations"][probe_idx], np.float32)
    tgt_cube = world_to_plot(obs_p[:, CUBE_OBS][:, :2] * CM)

    grids, dec, stats, goal_xy = {}, {}, {}, None
    for a in args.critics:
        arm = A.load_arm(Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}")
        goals = task_goals(arm, args.task, args.goal_draws, 20260904)
        if goal_xy is None:
            goal_xy = world_to_plot(np.mean([g[CUBE_OBS][:2] for g in goals], axis=0) * CM)
        W, r2 = fit_probe(arm, obs_p, tgt_cube)
        E = b_embed(arm, obs_f)
        dec[a] = np.concatenate([E, np.ones((len(E), 1), np.float32)], 1) @ W
        gs, negs = [], []
        for i in range(len(rows)):
            p, neg = predicted_cube_mass(arm, obs_f[i], act_f[i], goals, pool_obs)
            h, _, _ = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1], bins=[xe, ye], weights=p)
            gs.append(h), negs.append(neg)
        grids[a] = gs
        err = np.linalg.norm(dec[a] - true_xy, axis=1)
        stats[a] = (r2, float(np.mean(negs)))
        print(f"  {a:20s} probe R2 {r2:+.3f} | decode error per frame (cm) "
              f"{np.round(err, 1).tolist()}")

    # ---- draw ----
    nrow, ncol = 1 + len(args.critics), len(rows)
    # The heat panels are now ~59x25 cm with aspect="equal", i.e. 2.4:1 wide, so they need
    # far less height than the square render row; without the ratio they float in whitespace.
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.62 + 0.80 * len(args.critics)),
                             gridspec_kw=dict(height_ratios=[1.9] + [1.0] * len(args.critics)))
    fig.patch.set_facecolor(SURFACE)
    for c in range(ncol):
        ax = axes[0][c]
        ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d6d5d1")
        ax.set_title(f"frame {c + 1}" + ("  (carrying)" if lifted[c] else ""),
                     fontsize=8, color=INK if lifted[c] else INK2, pad=4)
    # Row labels sit HORIZONTALLY in the left margin: rotated labels are taller than these
    # short wide panels and ran into each other.
    axes[0][0].text(-0.10, 0.5, "recorded\nplay data", transform=axes[0][0].transAxes,
                    ha="right", va="center", fontsize=9, color=INK2)

    for r, a in enumerate(args.critics, start=1):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            z = grids[a][c]
            # per-panel normalisation: this figure is about position, not sharpness
            ax.imshow((z / max(z.max(), 1e-12)).T, origin="lower", extent=ext,
                      aspect="equal", cmap=MASS_CMAP, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.plot(goal_xy[0], goal_xy[1], marker="*", ms=11, color="#ffd166",
                    mec="#1a1a1a", mew=0.8, zorder=6)
            ax.plot(true_xy[c, 0], true_xy[c, 1], marker="o", ms=7, mfc="none",
                    mec="#0b0b0b", mew=1.6, zorder=6)
            ax.plot(dec[a][c, 0], dec[a][c, 1], marker="x", ms=7, color="#1b6ff5",
                    mew=1.8, zorder=6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#d6d5d1")
        axes[r][0].text(-0.10, 0.5, ARM_LABEL[a].replace(" (", "\n("),
                        transform=axes[r][0].transAxes, ha="right", va="center",
                        fontsize=9.5, color=ARM_COLOR[a], fontweight="bold")

    fig.tight_layout()
    fig.suptitle("The critic's predicted cube position during a carry", fontsize=12,
                 color=INK, y=1.12)
    fig.text(0.5, 1.035, "Overhead view of the table (across = lateral, up = depth).    "
             "o = cube now    x = cube decoded from the backward embedding    * = goal.    "
             "Each blob normalised to its own maximum.",
             fontsize=8.5, color=INK2, ha="center")
    out = Path(args.out) if args.out else ROOT / "figures" / "fig_carry_strip"
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
