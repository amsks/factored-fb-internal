"""Two standalone figures for the paper, no in-figure explanation.

fig_mdp      the nine-state graph
fig_values   the value of each action at the decision state, for the exact
             successor measure and for the two critics

Both are written as PDF and PNG. Values come from analysis/didactic_graph/bars;
no training happens here.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import graph_bottleneck_td as gb

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
})

INK, EDGE = "#2b2b2b", "#9a9a9a"
ACCENT, CUTFILL, GOALFILL = "#c1440e", "#f6e2d3", "#dfe8f2"
TRUE, FBC, DENS = "#4a4a4a", "#1f6fb2", "#b25fa6"
GAMMA = 0.8
POS = {0: (0.0, 0.95), 1: (0.95, 1.62), 2: (0.95, 0.28), 3: (1.9, 0.95),
       4: (3.05, 0.95),
       5: (4.2, 0.95), 6: (5.15, 1.62), 7: (5.15, 0.28), 8: (6.1, 0.95)}
RAD = 0.235


def draw_mdp(out_base):
    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    for a, (t0, t1) in enumerate(gb.EDGES):
        for b in (t0, t1):
            if a == b:
                ax.add_patch(FancyArrowPatch(
                    (POS[a][0] - .085, POS[a][1] + .22), (POS[a][0] + .085, POS[a][1] + .22),
                    arrowstyle="-|>", mutation_scale=8, lw=0.9, color=EDGE,
                    connectionstyle="arc3,rad=3.2", zorder=2))
                continue
            decision = (a, b) == (gb.DECISION, gb.GOAL)
            (x0, y0), (x1, y1) = POS[a], POS[b]
            dx, dy = x1 - x0, y1 - y0
            L = np.hypot(dx, dy)
            ax.add_patch(FancyArrowPatch(
                (x0 + RAD * dx / L, y0 + RAD * dy / L),
                (x1 - (RAD + .03) * dx / L, y1 - (RAD + .03) * dy / L),
                arrowstyle="-|>", mutation_scale=9,
                lw=1.9 if decision else 0.9, color=ACCENT if decision else EDGE,
                connectionstyle="arc3,rad=0.17", zorder=3 if decision else 2))
    for s, (x, y) in POS.items():
        fc = CUTFILL if s == gb.CUT else (GOALFILL if s == gb.GOAL else "white")
        ec = ACCENT if s == gb.CUT else (INK if s == gb.GOAL else EDGE)
        ax.add_patch(Circle((x, y), RAD, fc=fc, ec=ec,
                            lw=1.6 if s in (gb.CUT, gb.GOAL) else 1.0, zorder=4))
        ax.text(x, y, str(s), ha="center", va="center", fontsize=9.5, color=INK, zorder=5)
    ax.set_xlim(-.42, 6.52); ax.set_ylim(-.02, 2.12); ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.pdf / .png")


def draw_values(rows, out_base):
    truth = np.array(rows[0]["q_decision_true"]) / (1 - GAMMA)
    def v(method):
        q = np.array([r["q_decision"] for r in rows if r["method"] == method]) / (1 - GAMMA)
        return q.mean(0), q.std(0)
    groups = [("Exact", truth, np.zeros(2), TRUE),
              ("FB", *v("fb"), FBC), ("Density-FB", *v("density"), DENS)]

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    w, x = 0.33, np.arange(3)
    ebar = dict(ecolor=INK, capsize=2, error_kw={"lw": 0.8})
    for i, (lab, vals, sd, col) in enumerate(groups):
        ax.bar(x[i] - w / 2, vals[0], w, yerr=sd[0], color=col, edgecolor=INK, lw=0.5,
               zorder=3, **ebar)
        ax.bar(x[i] + w / 2, vals[1], w, yerr=sd[1], color=col, edgecolor=INK, lw=0.5,
               zorder=3, alpha=0.32, **ebar)
        for dx, val, e in ((-w / 2, vals[0], sd[0]), (w / 2, vals[1], sd[1])):
            ax.text(x[i] + dx, val + e + .07, f"{val:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.bar(0, 0, 0, color=EDGE, edgecolor=INK, lw=0.5, label="to the goal")
    ax.bar(0, 0, 0, color=EDGE, edgecolor=INK, lw=0.5, alpha=0.32, label="away")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("value at state 7", fontsize=9)
    ax.set_ylim(0, 2.75)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=.25, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
              handlelength=1.1, columnspacing=1.0, borderpad=0.1)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.pdf / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("analysis/didactic_graph/bars"))
    ap.add_argument("--out", type=Path, default=Path("PAPER/ICLR/didactic_new/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    draw_mdp(a.out / "fig_mdp")
    draw_values(json.load(open(a.run / "summary.json")), a.out / "fig_values")
