"""Fig: the value's energy funnel, per state factor.

Reads `energy_pool_*.npz` (energy_landscape.py --mode pool) and draws, for each state
factor, how the critic's value rises as that factor moves away from the goal.

Reading the panels. Every point is a REAL dataset transition where the cube is held, so the
grasp is already made and all that remains is transport. Rows are binned by how far the
factor is from the goal, and each bin's mean value is expressed in units of that critic's
OWN value spread -- a scale-free axis, since FB's bilinear score and Density-FB's kernel
expectation are numerically incomparable. Energy = -value, so a funnel toward the goal
reads as a curve rising left-to-right, and a critic that does not resolve the factor is a
flat line at 0.

The finding is the FB line, and it is FLAT -- on BOTH factors. Measured funnel depth:

    FB            cube .052 SD    gripper .033 SD
    Density-FB    cube .284 SD    gripper .000 SD

Once the cube is in the gripper, FB's value stops tracking the goal at all: it is within
noise of a critic that has never been told what the task is. Density-FB, on the same
states, puts .28 SD between a cube at the goal and a cube 15 cm away.

What this figure does NOT show, and must not be captioned as showing: a fidelity GRADIENT
with effective controllability. That framing predicts FB does well on the arm (~1.0) and
badly on the cube (.52); what is measured is FB flat on both. The result this reproduces
is the audited one -- post-acquisition goal-blindness (docs/notes/2026-09-04-density-fb-
central-perspective.md) -- and the controllability numbers in the panel titles are context
for which factor the task rewards, not a demonstrated axis.

Two further honesty notes for the caption:
  - Density-FB's cube curve is NOT monotone: it crests near 14 cm and falls again, so the
    funnel is local to the goal, not a global potential.
  - Density-FB's goal kernel is defined on the object block (`measure_hit_spec 19:22`), so
    its cube/gripper asymmetry is partly by construction. The FB line carries the claim.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette and arm identity are figures.py's, unchanged: an arm keeps its hue in every panel
# of every figure in this paper.
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ANALYSIS = Path("analysis/cube_bottleneck")
ORDER = ["fb_ref", "density_fb_state"]

# Distance bins in the observation's scaled frame (1 unit = 10 cm), so the first bin is the
# 4 cm kernel radius the task's own success criterion uses.
EDGES = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0])
CM = 10.0  # scaled unit -> cm

FACTORS = (("cube", "state_cube_xy", "goal_cube_xy", "cube position", 0.52),
           ("effector", "state_eff_xy", "goal_eff_xy", "gripper position", 1.0))
MIN_ROWS = 20


def profiles(root: Path, arm: str, seeds) -> dict:
    """[cell, bin] standardized ENERGY per factor; a cell is one (seed, task)."""
    out = {f[0]: [] for f in FACTORS}
    for s in seeds:
        f = root / f"energy_pool_{arm}_s{s}.npz"
        if not f.exists():
            continue
        d = dict(np.load(f))
        q, held = d["q"], d["held"].astype(bool)
        for name, key, gkey, _lab, _c in FACTORS:
            for ti in range(len(d["task_id"])):
                dist = np.linalg.norm(d[key] - d[gkey][ti], axis=1)
                qq = q[ti]
                # standardize on the SAME rows the bins are drawn from, so the unit is
                # "spread of this critic's value across held states", not across phases.
                mu, sd = qq[held].mean(), qq[held].std()
                row = []
                for a, b in zip(EDGES[:-1], EDGES[1:]):
                    m = held & (dist >= a) & (dist < b)
                    row.append(-(qq[m].mean() - mu) / sd if m.sum() > MIN_ROWS else np.nan)
                out[name].append(row)
    return {k: np.asarray(v, float) for k, v in out.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cube-single-play-v0")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = ANALYSIS / args.env / "energy"
    data = {arm: profiles(root, arm, args.seeds) for arm in ORDER}
    arms = [a for a in ORDER if any(len(v) for v in data[a].values())]
    assert arms, f"no energy_pool_*.npz in {root}"

    centers = (EDGES[:-1] + EDGES[1:]) / 2 * CM
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (name, _k, _g, label, ctrl) in zip(axes, FACTORS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6d5d1")
        ax.tick_params(colors=INK2, labelsize=8, length=3)
        ax.grid(True, axis="y", color="#eceae6", lw=0.8)
        ax.set_axisbelow(True)
        # "no funnel" is a real reference value here, not decoration.
        ax.axhline(0.0, color="#b9b7b2", lw=1.0, ls=(0, (4, 3)), zorder=1)

        for arm in arms:
            P = data[arm][name]
            if not len(P):
                continue
            mu = np.nanmean(P, axis=0)
            n = np.sum(~np.isnan(P), axis=0)
            se = np.nanstd(P, axis=0) / np.maximum(np.sqrt(np.maximum(n, 1)), 1)
            ok = n > 1
            c = ARM_COLOR[arm]
            ax.fill_between(centers[ok], (mu - se)[ok], (mu + se)[ok], color=c,
                            alpha=0.16, lw=0, zorder=2)
            ax.plot(centers[ok], mu[ok], color=c, lw=2.0, zorder=3,
                    solid_capstyle="round")
            ax.plot(centers[ok], mu[ok], "o", ms=4.5, color=c, mec=SURFACE, mew=1.2,
                    zorder=4)
        ax.set_title(f"{label}\neffective controllability {ctrl:.2f}", fontsize=9,
                     color=INK, pad=8)
        ax.set_xlabel("distance from the goal (cm)", fontsize=8.5, color=INK2)

    axes[0].set_ylabel("energy  (critic's own SD)", fontsize=8.5, color=INK2)

    # Direct labels, so identity never rests on colour alone (two series, so the legend is
    # the labels themselves).
    # FB's curve ends at the top of the pair and Density-FB's at the bottom, so one label
    # goes above its line and the other below -- no leader lines, no collision.
    for arm, dy in zip(arms, (10, -16)):
        P = data[arm]["cube"]
        mu = np.nanmean(P, axis=0)
        j = int(np.max(np.where(~np.isnan(mu))[0]))
        axes[0].annotate(ARM_LABEL[arm], (centers[j], mu[j]), xytext=(-2, dy),
                         textcoords="offset points", color=ARM_COLOR[arm], fontsize=8.5,
                         fontweight="bold", ha="right")

    fig.suptitle("Does the value fall toward the goal?  Held states, cube-single, "
                 "3 seeds x 5 tasks", fontsize=9.5, color=INK, y=1.005)
    fig.tight_layout()
    out = Path(args.out) if args.out else ANALYSIS / args.env / "figures" / "fig_energy_funnel"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")

    for arm in arms:
        for name, *_ in FACTORS:
            mu = np.nanmean(data[arm][name], axis=0)
            k = int(np.nanargmax(mu))
            # End-to-end AND the peak: Density-FB's cube curve is not monotone (it crests
            # near 14 cm and falls again), so a max-based "depth" would flatter it.
            print(f"  {arm:18s} {name:9s} energy {mu[0]:+.3f} @{centers[0]:.0f}cm -> "
                  f"{mu[-1]:+.3f} @{centers[-1]:.0f}cm (end-to-end {mu[-1]-mu[0]:+.3f}); "
                  f"peak {mu[k]:+.3f} @{centers[k]:.0f}cm (depth {mu[k]-mu[0]:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
