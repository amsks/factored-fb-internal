"""Fig: the value over the WORKSPACE, one column per task goal.

Reads `energy_pool_*.npz` (energy_landscape.py --mode pool). Each panel is the table seen
from above. Every real dataset transition where the cube is HELD is dropped into the bin
its cube sits in, and the bin is coloured by the critic's value there -- so the map is the
critic's own answer to "if the cube were here, how good would that be?", asked only at
places the data actually visited. Nothing is extrapolated and no state is fabricated.

Energy = -value in units of that critic's own spread across held states, which makes FB's
bilinear score and Density-FB's kernel expectation comparable and lets ONE colour scale
serve every panel. Dark = low energy = the critic likes it there. The star is the goal.

What to look for: across the three columns the goal MOVES. Density-FB's dark basin moves
with it. FB's map barely changes -- the same faint structure sits in the same place
whatever the task is, which is what post-acquisition goal-blindness looks like laid out on
the table. Each panel carries its own energy range, so the flatness is a number and not
just a pale rectangle.

Caveat kept from fig_energy.py: Density-FB's goal kernel is defined on the object block
(`measure_hit_spec 19:22`), so its cube sensitivity is partly by construction. The FB row
is the finding.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ANALYSIS = Path("analysis/cube_bottleneck")
ORDER = ["fb_ref", "density_fb_state"]
CM = 10.0          # scaled observation unit -> cm
MIN_COUNT = 12     # bins thinner than this are left blank rather than drawn from noise

# Diverging red <-> black about a pale neutral. The residual has a true zero (the method's
# own across-goal average), which is what licenses a diverging ramp.
#
# RED = the critic LIKES the cube here more than it does averaged over the goals; BLACK =
# less; pale = the goal changed nothing here. Note this means red is LOW energy: the colour
# axis is deliberately inverted relative to the raw energy number, so that the salient
# colour marks the wanted region. The colourbar keeps its natural direction (energy
# increasing upward), so red sits at its NEGATIVE end -- read it as "red = wanted".
#
# Both poles carry lightness as well as hue, so the split survives greyscale, print and CVD.
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "energy_rk", ["#ff2a00", "#d1341c", "#f0ede7", "#4a4a4a", "#000000"])


def _smooth(z: np.ndarray, passes: int = 1) -> np.ndarray:
    """NaN-aware 3x3 box smooth of ONE 2-D map: empty bins neither contribute nor get
    filled in. Strictly 2-D on purpose -- rolling a stacked [task, x, y] array here would
    smooth across the task axis and silently average the goals into each other."""
    assert z.ndim == 2, f"_smooth takes one 2-D map, got shape {z.shape}"
    for _ in range(passes):
        v = np.where(np.isnan(z), 0.0, z)
        w = (~np.isnan(z)).astype(float)
        acc = np.zeros_like(v)
        cnt = np.zeros_like(w)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                acc += np.roll(np.roll(v, dx, 0), dy, 1)
                cnt += np.roll(np.roll(w, dx, 0), dy, 1)
        z = np.where(np.isnan(z), np.nan, np.divide(acc, np.maximum(cnt, 1e-9)))
    return z


def maps(root: Path, arm: str, seeds, tasks, xe, ye):
    """[task] mean standardized-energy map, averaged over seeds."""
    acc, cnt = None, None
    for s in seeds:
        f = root / f"energy_pool_{arm}_s{s}.npz"
        if not f.exists():
            continue
        d = dict(np.load(f))
        q, held = d["q"], d["held"].astype(bool)
        xy, tid = d["state_cube_xy"], list(d["task_id"])
        if acc is None:
            acc = np.zeros((len(tasks), len(xe) - 1, len(ye) - 1))
            cnt = np.zeros_like(acc)
        for k, t in enumerate(tasks):
            ti = tid.index(t)
            qq = q[ti]
            e = -(qq - qq[held].mean()) / qq[held].std()
            n, _, _ = np.histogram2d(xy[held, 0], xy[held, 1], bins=[xe, ye])
            ssum, _, _ = np.histogram2d(xy[held, 0], xy[held, 1], bins=[xe, ye],
                                        weights=e[held])
            m = n >= MIN_COUNT
            acc[k] += np.where(m, ssum / np.maximum(n, 1), 0.0)
            cnt[k] += m
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cube-single-play-v0")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tasks", nargs="+", type=int, default=[3, 4, 5],
                   help="the three most widely separated goals, so the star visibly moves")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--residual", action="store_true",
                   help="plot each goal's map MINUS the method's across-goal mean, i.e. "
                        "the goal-dependent part alone. The raw maps carry goal-INdependent "
                        "structure (arm pose, coverage) of the same magnitude as the effect, "
                        "which buries it; subtracting the mean cancels exactly that.")
    p.add_argument("--bare", action="store_true",
                   help="strip the prose -- no title, subtitle or per-panel annotation. "
                        "Structural labels (method row, task column, axes) stay: without "
                        "them the panels cannot be told apart. The caption carries the rest.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = ANALYSIS / args.env / "energy"
    ref = dict(np.load(root / f"energy_pool_{ORDER[0]}_s{args.seeds[0]}.npz"))
    held = ref["held"].astype(bool)
    xy = ref["state_cube_xy"]
    lo, hi = np.percentile(xy[held], [1, 99], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    goals = {int(t): ref["goal_cube_xy"][list(ref["task_id"]).index(t)] for t in args.tasks}

    arms = [a for a in ORDER if (root / f"energy_pool_{a}_s{args.seeds[0]}.npz").exists()]
    M = {a: np.stack([_smooth(m) for m in maps(root, a, args.seeds, args.tasks, xe, ye)])
         for a in arms}
    # The figure's actual claim, as a number on the figure: how much the map MOVES when the
    # goal moves. A goal-blind critic redraws nothing.
    churn = {a: float(np.mean([np.nanmean(np.abs(M[a][i] - M[a][j]))
                               for i, j in ((0, 1), (0, 2), (1, 2))])) for a in arms}

    if args.residual:
        # Churn is a difference between goals, so it is unchanged by this subtraction --
        # the number quoted in the subtitle still describes what is drawn.
        M = {a: M[a] - np.nanmean(M[a], axis=0, keepdims=True) for a in arms}

    # ONE scale for every panel: the units are each critic's own SD, so they are already
    # commensurate, and a shared scale is what makes FB's row read as flat rather than as
    # a differently-stretched version of the same picture.
    allv = np.concatenate([m[~np.isnan(m)].ravel() for m in M.values()])
    v = float(np.percentile(np.abs(allv), 99))
    ext = [xe[0] * CM, xe[-1] * CM, ye[0] * CM, ye[-1] * CM]

    fig, axes = plt.subplots(len(arms), len(args.tasks),
                             figsize=(2.35 * len(args.tasks) + 1.0, 2.5 * len(arms)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(SURFACE)

    for r, arm in enumerate(arms):
        for c, t in enumerate(args.tasks):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            z = M[arm][c]
            # No contour lines. They were iso-energy lines of THIS SAME field -- a second
            # encoding of the data already in the colour, drawn on a smoothed, binned,
            # noisy surface where their wiggles were binning artefacts rather than
            # structure. They added texture that read as detail and wasn't.
            im = ax.imshow(z.T, origin="lower", extent=ext, aspect="equal",
                           cmap=ENERGY_CMAP, vmin=-v, vmax=v, interpolation="bilinear")
            g = goals[t] * CM
            ax.plot(g[0], g[1], marker="*", ms=17, color="#ffd166", mec="#1a1a1a",
                    mew=1.0, zorder=5)
            if not args.bare:
                # Panels are narrow; the unit is on the colourbar, so the annotation is
                # just the number -- a longer string overflows into the next column.
                rng = np.nanmax(z) - np.nanmin(z)
                ax.text(0.04, 0.02, f"range {rng:.2f}", transform=ax.transAxes,
                        fontsize=7, color=INK2, va="bottom", ha="left",
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.75, pad=1.4))
            for s in ax.spines.values():
                s.set_color("#d6d5d1")
            ax.tick_params(colors=INK2, labelsize=7.5, length=3)
            if r == 0:
                ax.set_title(f"task {t}", fontsize=9, color=INK, pad=6)
            if c == 0:
                ax.set_ylabel("table y (cm)", fontsize=8, color=INK2)
                # Method identity sits outboard of the axis label, in the arm's own hue.
                # The panels are narrow, so anything else in this margin collides: the
                # churn number goes INSIDE the panel instead.
                ax.text(-0.72, 0.5, ARM_LABEL[arm], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10.5, fontweight="bold",
                        color=ARM_COLOR[arm])
                # (the churn number is the headline, so it goes in the subtitle -- these
                # panels are too narrow to hold a sentence without overflowing)
            if r == len(arms) - 1:
                ax.set_xlabel("table x (cm)", fontsize=8, color=INK2)

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    if not args.bare:
        cb.set_label("energy  (critic's own SD)", fontsize=8, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    cb.outline.set_edgecolor("#d6d5d1")

    if not args.bare:
        fig.suptitle("What does the goal change about where the critic wants the cube?"
                     if args.residual else "Where does the critic want the cube?",
                     fontsize=11.5, color=INK, y=1.055)
        fig.text(0.5, 1.005,
                 ("Each goal's map minus the method's average over goals, so everything not "
                  "caused by the goal cancels.  * = the goal." if args.residual else
                  "Dark = the critic prefers the cube here.  * = the goal.")
                 + "  Held states, cube-single, 3 seeds.",
                 fontsize=8.5, color=INK2, ha="center")
        if len(arms) == 2:
            a0, a1 = arms
            ratio = churn[a1] / max(churn[a0], 1e-9)
            fig.text(0.5, 0.962,
                     f"When the goal moves, {ARM_LABEL[a0]}'s map moves {churn[a0]:.3f} SD; "
                     f"{ARM_LABEL[a1]}'s moves {churn[a1]:.3f} SD  ({ratio:.1f}x).",
                     fontsize=9, color=INK, ha="center")

    stem = "fig_energy_map_residual" if args.residual else "fig_energy_map"
    if args.bare:
        stem += "_bare"
    out = Path(args.out) if args.out else ANALYSIS / args.env / "figures" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png   (shared scale +-{v:.2f} SD)")

    # Does the basin follow the goal? Distance from the map's minimum to the goal, and how
    # much the map changes between goals -- the two numbers the picture is claiming.
    for arm in arms:
        ds, flat = [], []
        for c, t in enumerate(args.tasks):
            z = M[arm][c]
            i, j = np.unravel_index(np.nanargmin(z), z.shape)
            mn = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
            ds.append(np.linalg.norm(mn - goals[t]) * CM)
            flat.append(np.nanmax(z) - np.nanmin(z))
        pair = [float(np.nanmean(np.abs(M[arm][a] - M[arm][b])))
                for a, b in ((0, 1), (0, 2), (1, 2))]
        print(f"  {arm:18s} min-to-goal {np.round(ds, 1)} cm (mean {np.mean(ds):.1f}) | "
              f"range {np.round(flat, 2)} SD | mean |map difference| between goals "
              f"{np.mean(pair):.3f} SD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
