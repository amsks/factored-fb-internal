"""Figures for reports/2026-07-23-successor-measures.md.

Three figures, regenerable from logged artifacts:
  1_playdata.png    -- what play data looks like: cube xy traces (dwell/revisit) +
                       the distribution of discounted revisit mass R at a goal ball.
  2_curves.png      -- success vs steps: reaching TD / density TD on frozen psi /
                       visitation TD, against the GC-IQL bilinear anchor.
  3_extraction.png  -- aggregate extraction routes incl. the GC-IQL bilinear baseline.

Palette: dataviz reference instance, light mode, categorical slots in fixed order.
"""
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e6e5e2"

OUT = Path(__file__).parent
E6 = Path("/home/adityamohan/exp_spec06/factored-fb")
E7 = Path("/home/adityamohan/exp_spec07/factored-fb")
GAMMA, EPS = 0.99, 0.4  # obs units; 1 unit = 10 cm => 4 cm ball

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10,
})


def eval_series(run_dir):
    rows = list(csv.DictReader(open(run_dir / "eval.csv")))
    c = [k for k in rows[0] if "success" in k and "overall" in k][0]
    return (np.array([int(r["step"]) for r in rows]),
            np.array([float(r[c]) for r in rows]))


def mean_series(run_dirs):
    per = [eval_series(d) for d in run_dirs]
    steps = per[0][0]
    ys = np.stack([np.interp(steps, s, y) for s, y in per])
    return steps, ys.mean(0), ys


# ---------------------------------------------------------------- fig 1: play data
def fig_playdata():
    import ogbench
    _env, ds, _ = ogbench.make_env_and_datasets("cube-single-play-v0")
    cube = np.asarray(ds["observations"], np.float32)[:, 19:22]
    term = np.asarray(ds["terminals"], bool)
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # (a) three trajectory traces, sequential shading light->dark = time
    rng = np.random.default_rng(0)
    trajs = rng.choice(len(starts), 3, replace=False)
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    for k, ti in enumerate(trajs):
        seg = cube[starts[ti]:ends[ti] + 1, :2]
        pts = seg.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cmap = LinearSegmentedColormap.from_list("t", ["#cfe0f5", BLUE])
        lc = LineCollection(segs, cmap=cmap, linewidth=1.4,
                            array=np.linspace(0, 1, len(segs)))
        ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_aspect("equal")
    ax1.set_xlabel("cube x (obs units)")
    ax1.set_ylabel("cube y (obs units)")
    ax1.set_title("Cube traces, 3 play trajectories\n(light→dark = time)",
                  loc="left", color=INK)

    # (b) distribution of discounted revisit mass R after first arrival
    R = []
    n_traj = min(len(starts), 4000)
    idxs = rng.choice(len(starts), n_traj, replace=False)
    for ti in idxs:
        s0, s1 = starts[ti], ends[ti] + 1
        seg = cube[s0:s1]
        if len(seg) < 10:
            continue
        gi = rng.integers(5, len(seg))
        g = seg[gi]
        d = np.linalg.norm(seg - g[None], axis=-1)
        hit = np.flatnonzero(d <= EPS)
        if len(hit) == 0:
            continue
        t0 = hit[0]
        after = d[t0 + 1:] <= EPS
        if len(after) == 0:
            continue
        R.append(float((GAMMA ** np.arange(len(after)) * after).sum()))
    R = np.array(R)
    ax2.hist(R, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax2.axvline(R.mean(), color=ORANGE, linewidth=2)
    ax2.annotate(f"mean = {R.mean():.1f} discounted visits",
                 xy=(R.mean(), 1), xycoords=("data", "axes fraction"),
                 xytext=(6, -14), textcoords="offset points", color=INK)
    ax2.set_xlabel("discounted revisits of the goal ball after first arrival "
                   "(within-trajectory)")
    ax2.set_ylabel("count")
    ax2.set_title("Occupancy magnitude is mostly revisit mass\n"
                  "(discounted revisits after first arrival, $\\gamma$=0.99)",
                  loc="left", color=INK)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(OUT / "1_playdata.png", dpi=180)
    plt.close(fig)
    print(f"[fig1] n={len(R)} mean R={R.mean():.2f}")


# ---------------------------------------------------------------- fig 2: curves
def fig_curves():
    fp = [E6 / "GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50" / f"m_firstpassage_noortho_s{s}"
          for s in (0, 1, 2)]
    vis = [E7 / "GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50" / f"vis_s{s}" for s in (0, 1)]
    occ = E7 / "GCIQL-State-Flow-Measure-Occ-Phase2-FrozenPsi-d50" / "occfz_s0"

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axhline(0.904, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("GC-IQL bilinear anchor (0.904)", xy=(0.02, 0.904),
                xycoords=("axes fraction", "data"), ha="left",
                xytext=(0, -11), textcoords="offset points", color=INK2, fontsize=8)

    s, m, ys = mean_series(fp)
    ax.plot(s, m, color=BLUE, linewidth=2, label="phase 1: reaching TD (3 seeds)")
    ax.fill_between(s, ys.min(0), ys.max(0), color=BLUE, alpha=0.15, linewidth=0)

    s, m, ys = mean_series(vis)
    ax.plot(s, m, color=AQUA, linewidth=2, label="phase 2: visitation TD on frozen $\\psi$ (2 seeds)")
    ax.fill_between(s, ys.min(0), ys.max(0), color=AQUA, alpha=0.15, linewidth=0)

    s, y = eval_series(occ)
    ax.plot(s, y, color=ORANGE, linewidth=2, marker="o", markersize=5,
            label="density TD on the SAME frozen $\\psi$ (diverges; killed @100k)")

    ax.set_xlabel("gradient steps")
    ax.set_ylabel("overall success")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 500_000)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v/1000)}k")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Same representation, same data — the estimator decides", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "2_curves.png", dpi=180)
    plt.close(fig)
    print("[fig2] done")


# ---------------------------------------------------------------- fig 3: extraction
def fig_extraction():
    rows = [  # (label, per-seed values)
        ("GC-IQL bilinear baseline\n(goal-conditioned, not zero-shot)", [0.904]),
        ("phase 1 reaching value, goal-conditioned", [0.956, 0.964, 0.980]),
        ("phase 2 visitation, goal-conditioned", [0.952, 0.928]),
        ("zero-shot single goal, retrieval decode", [0.888, 0.960, 0.968]),
        ("zero-shot 2-goal mixture, GPI", [0.96, 0.98, 0.95]),
        ("direct-z single goal (separated form)", [0.90, 0.92]),
        ("direct-z 2-goal mixture (separated form)", [0.87, 0.89]),
        ("direct-z dense reward (separated form)", [0.90, 0.13]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ypos = np.arange(len(rows))[::-1]
    for y, (label, vals) in zip(ypos, rows):
        v = float(np.mean(vals))
        color = INK2 if "baseline" in label else BLUE
        ax.barh(y, v, height=0.55, color=color, alpha=0.9 if color == BLUE else 0.45)
        ax.scatter(vals, [y] * len(vals), color=INK, s=14, zorder=3)
        ax.annotate(f"{v:.2f}", xy=(v, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("overall success (dots = individual seeds)")
    ax.set_title("What the learned measures support today", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "3_extraction.png", dpi=180)
    plt.close(fig)
    print("[fig3] done")


if __name__ == "__main__":
    fig_curves()
    fig_extraction()
    fig_playdata()
    print("all figures written to", OUT)
