"""Fig: under each method's own policy, where does the cube actually end up?

Apples-to-apples by construction: both panels are ROLLOUTS, measured the same way. No
predicted measure enters, so Density-FB's cube-block reward kernel gives it no advantage in
the measurement -- the only thing being compared is the behaviour each method learned.

Discounted occupancy of the cube, sum_t gamma^t 1[cube at u], from pre-grasp (P1) starts,
each arm rolled under ITS OWN actor (truth_occupancy.py; the policy-conditioning rule from
the 2026-09-04 audit). Same colour scale, same bins, same states, same episodes.
"""
import argparse, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.probes.cube_bottleneck.fig_outcome_map import CM, MASS_CMAP
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ROOT = Path("analysis/cube_bottleneck/cube-single-play-v0")
ORDER = ["fb_ref", "density_fb_state"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--bins", type=int, default=22)
    ap.add_argument("--from-frac", type=float, default=0.5,
                    help="use only the last (1-f) of each rollout. The cube DWELLS at its "
                         "start until it is grasped, so early steps paint one speck per "
                         "start state and swamp the panel; the question is where it ends "
                         "up, not where it began.")
    ap.add_argument("--smooth", type=int, default=1,
                    help="3x3 box passes; 32 discrete starts otherwise read as grain.")
    ap.add_argument("--bare", action="store_true")
    args = ap.parse_args()

    pool = dict(np.load(ROOT / "pool.npz"))["obs"][:, 19:21] * CM
    lo, hi = np.percentile(pool, [1, 99], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    panels, stats, goal = {}, {}, None
    for a in args.critics:
        fs = sorted(glob.glob(str(ROOT / f"truth/truth_P1_{a}_s*.npz")))
        assert fs, f"no truth rollouts for {a}"
        acc, occs, seeds = np.zeros((args.bins, args.bins)), [], []
        for f in fs:
            d = dict(np.load(f, allow_pickle=True))
            XY, w, g = d["cube_xy"] * CM, d["weights"], d["goal_cube_xy"] * CM
            goal = g if goal is None else goal
            # every (state, step, episode) sample, weighted by gamma^t and averaged
            S, T, E, _ = XY.shape
            t0 = int(args.from_frac * T)
            ww = w[t0:] / w[t0:].sum()
            XYl = XY[:, t0:]
            wt = np.broadcast_to(ww[None, :, None], (S, T - t0, E)).reshape(-1) / (S * E)
            h, _, _ = np.histogram2d(XYl[..., 0].reshape(-1), XYl[..., 1].reshape(-1),
                                     bins=[xe, ye], weights=wt)
            acc += h
            occs.append(float((w[None, :, None] *
                               (np.linalg.norm(XY - g, axis=-1) < 0.4 * CM)).sum(1).mean()))
            seeds.append(int(d["seed"]))
        z = acc / len(fs)
        for _ in range(args.smooth):
            z = sum(np.roll(np.roll(z, dx, 0), dy, 1)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)) / 9.0
        panels[a] = z
        stats[a] = (float(np.mean(occs)), seeds)
        print(f"  {a:20s} seeds {seeds} occupancy at goal {np.mean(occs):.4f}")

    vmax = float(max(p.max() for p in panels.values()))
    cols = [a for a in args.critics if a in panels]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.5 * len(cols) + 1.1, 3.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, a in zip(axes, cols):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(panels[a].T, origin="lower", extent=ext, aspect="equal",
                       cmap=MASS_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.plot(goal[0], goal[1], marker="*", ms=16, color="#ffd166", mec="#1a1a1a",
                mew=1.0, zorder=5)
        occ, seeds = stats[a]
        if not args.bare:
            ax.text(0.04, 0.02, f"at goal {occ:.2f}", transform=ax.transAxes, fontsize=8,
                    color=INK, va="bottom", ha="left",
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.82, pad=1.5))
        ax.set_title(ARM_LABEL[a], fontsize=10.5, color=ARM_COLOR[a], fontweight="bold",
                     pad=6)
        ax.set_xlabel("table x (cm)", fontsize=8, color=INK2)
        for sp in ax.spines.values():
            sp.set_color("#d6d5d1")
        ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    axes[0].set_ylabel("table y (cm)", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    cb.outline.set_edgecolor("#d6d5d1")
    if not args.bare:
        cb.set_label("discounted occupancy of the cube", fontsize=8, color=INK2)
        fig.suptitle("Where the cube actually ends up", fontsize=11.5, color=INK, y=1.07)
        fig.text(0.5, 1.005, "Rollouts under each method's own policy from pre-grasp "
                 "starts.  * = the goal.", fontsize=8.5, color=INK2, ha="center")
    stem = "fig_truth_map" + ("_bare" if args.bare else "")
    out = ROOT / "figures" / stem
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
