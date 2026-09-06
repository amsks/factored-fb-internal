"""Fig: how sharp is the predicted successor distribution, seen in the workspace?

The measured contrast is concentration: effective support over the frozen 1024-row pool is
1024 (uniform) -> ~325 (FB) -> ~30 (Density-FB). This figure shows that qualitatively, in
the coordinate where the sharpness actually lives -- the CUBE's position on the table.

Three panels, one colour scale:

    dataset prior   uniform weights over the pool: the concentration the DATA already has
    FB              its signed measure, clipped and renormalised (the clipped share is
                    printed -- roughly half the entries)
    Density-FB      its softmax weights, untouched

The prior panel is load-bearing, not decoration. Its omission is what made the earlier
gripper-relative reach map look like a result: 74% of the pool already sits within 1cm of
the cube, so concentration there was free and both methods scored a lift of ~1.0 over the
prior (docs: fig_reach_map.py). Read every panel against the prior, never against a flat
map.

Starts are PRE-GRASP states (P1: cube on the table, untouched, arm away), so the cube has
not moved yet and the prediction is about what the policy will do, not about where it
already is. The goal is fixed across panels and marked.

What this figure claims: sharpness. What it does NOT claim: correctness. Density-FB's mass
sits near the goal, and its prediction is conditioned on that goal, so an echoing critic
would look identical here -- that hole is closed by the unreachable-goal test or by rollout
truth, not by this picture.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, MASS_CMAP, task_goals
from scripts.probes.cube_bottleneck.fig_reach_map import CUBE_OBS, batched_mass
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ANALYSIS = Path("analysis/cube_bottleneck")
ORDER = ["fb_ref", "density_fb_state"]
NEAR_CM = 4.0          # the benchmark's own success radius on the cube


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--phase", default="P1", choices=["P1", "P2", "P3"])
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--goal-seed", type=int, default=20260904)
    ap.add_argument("--bins", type=int, default=22)
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = ANALYSIS / args.env
    banks = dict(np.load(root / "phase_banks.npz"))
    obs_b, act_b = banks[f"{args.phase}_obs"], banks[f"{args.phase}_action"]
    pool_obs = dict(np.load(root / "pool.npz"))["obs"]
    cube_xy = pool_obs[:, CUBE_OBS][:, :2] * CM        # absolute table coords, cm

    lo, hi = np.percentile(cube_xy, [1, 99], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    def to_grid(w):
        g, _, _ = np.histogram2d(cube_xy[:, 0], cube_xy[:, 1], bins=[xe, ye], weights=w)
        return g

    panels, summary, goal_xy = {}, {}, None
    for a in args.critics:
        acc, neffs, negs, nears = None, [], [], []
        for s in args.seeds:
            run = Path("exp") / f"{args.env}_{a}_flowbc_seed_{s}"
            if not (run / "params_500000.pkl").exists():
                continue
            arm = A.load_arm(run)
            goals = task_goals(arm, args.task, args.goal_draws, args.goal_seed)
            gxy = np.mean([g[CUBE_OBS][:2] for g in goals], axis=0) * CM
            if goal_xy is None:
                goal_xy = gxy
                near = np.linalg.norm(cube_xy - goal_xy, axis=1) < NEAR_CM
            for i in range(len(obs_b)):
                p, neg = batched_mass(arm, obs_b[i], act_b[i], goals, pool_obs)
                pm = p.mean(0)
                acc = pm if acc is None else acc + pm
                nz = pm[pm > 0]
                neffs.append(float(np.exp(-(nz * np.log(nz)).sum())))
                negs.append(neg)
                nears.append(float(pm[near].sum()))
            print(f"  [{a} s{s}] {len(obs_b)} {args.phase} states", flush=True)
        assert acc is not None, f"no checkpoints for {a}"
        panels[a] = to_grid(acc / len(neffs))
        summary[a] = (float(np.mean(neffs)), float(np.mean(negs)), float(np.mean(nears)))

    prior = np.full(len(pool_obs), 1.0 / len(pool_obs))
    panels["prior"] = to_grid(prior)
    summary["prior"] = (float(np.exp(-(prior * np.log(prior)).sum())), 0.0,
                        float(prior[near].sum()))

    cols = ["prior"] + [a for a in args.critics if a in panels]
    titles = {"prior": "dataset prior"}
    titles.update({a: ARM_LABEL[a] for a in args.critics})
    colours = {"prior": INK2}
    colours.update({a: ARM_COLOR[a] for a in args.critics})

    vmax = float(max(panels[c].max() for c in cols))
    fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols) + 1.1, 3.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, c in zip(axes, cols):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(panels[c].T, origin="lower", extent=ext, aspect="equal",
                       cmap=MASS_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.plot(goal_xy[0], goal_xy[1], marker="*", ms=15, color="#ffd166", mec="#1a1a1a",
                mew=1.0, zorder=5)
        n_eff, neg, m_near = summary[c]
        if not args.bare:
            lbl = f"N_eff {n_eff:.0f}\nat goal {m_near:.2f}"
            if neg:
                lbl += f"\n{100 * neg:.0f}% neg. clipped"
            ax.text(0.04, 0.02, lbl, transform=ax.transAxes, fontsize=7.5, color=INK,
                    va="bottom", ha="left",
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.82, pad=1.6))
        ax.set_title(titles[c], fontsize=10, color=colours[c],
                     fontweight="bold" if c != "prior" else "normal", pad=6)
        ax.set_xlabel("table x (cm)", fontsize=8, color=INK2)
        for sp in ax.spines.values():
            sp.set_color("#d6d5d1")
        ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    axes[0].set_ylabel("table y (cm)", fontsize=8, color=INK2)

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    cb.outline.set_edgecolor("#d6d5d1")
    if not args.bare:
        cb.set_label("predicted probability the cube ends up here", fontsize=8, color=INK2)
        fig.suptitle("Where will the cube end up?", fontsize=11.5, color=INK, y=1.10)
        fig.text(0.5, 1.035, "Each method's predicted distribution, against the "
                 "concentration the data already has.", fontsize=8.5, color=INK2,
                 ha="center")
        fig.text(0.5, 0.995, f"* = the goal.  Pre-grasp ({args.phase}) starts, averaged "
                 f"over {len(obs_b)} states x {len(args.seeds)} seeds.",
                 fontsize=8.5, color=INK2, ha="center")

    stem = f"fig_sharpness_map_{args.phase}" + ("_bare" if args.bare else "")
    out = Path(args.out) if args.out else root / "figures" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    for c in cols:
        n_eff, neg, m_near = summary[c]
        print(f"  {c:18s} N_eff {n_eff:7.1f} | mass within {NEAR_CM:.0f}cm of goal "
              f"{m_near:.4f} (lift vs prior {m_near / max(summary['prior'][2], 1e-9):.2f}x) | "
              f"negative {100 * neg:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
