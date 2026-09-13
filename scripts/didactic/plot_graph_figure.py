"""The didactic figure: nine states, one crossing, and what the critic loses.

A  the graph. Every path from the first group to the goal passes state 4.
B  the value difference between the two actions at state 7, learned divided by
   exact, against the rank of the critic.
C  the same quantity as the crossing is weakened: `leak` is how often a
   transition ignores the edges and lands in the other group directly, so leak 0
   is a single crossing and larger values mix the two groups.
"""
import argparse, collections, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import graph_bottleneck_td as gb

FBC, DENS, HL, GREY = "#0173b2", "#cc78bc", "#d55e00", "#8a8a8a"
POS = {0: (0.0, 0.95), 1: (0.95, 1.6), 2: (0.95, 0.3), 3: (1.9, 0.95),
       4: (3.1, 0.95),
       5: (4.3, 0.95), 6: (5.25, 1.6), 7: (5.25, 0.3), 8: (6.2, 0.95)}


def load(root, sub="main"):
    g = collections.defaultdict(list)
    for r in json.load(open(root / sub / "summary.json")):
        g[(r["method"], r["d"], r["leak"])].append(r)
    return g


GAMMA = 0.8


def val(g, method, d, leak, key="gap_near"):
    """Value the critic puts on reaching the goal minus on not, at state 7.

    The stored gap is action 1 minus action 0, and at state 7 action 0 is the one
    that reaches the goal, so the sign is flipped here to make larger better."""
    return -float(np.mean([r[key] for r in g[(method, d, leak)]])) / (1 - GAMMA)


def panel_graph(ax):
    for a, (t0, t1) in enumerate(gb.EDGES):
        for b in (t0, t1):
            if a == b:
                ax.add_patch(FancyArrowPatch((POS[a][0] - .13, POS[a][1] + .21),
                                             (POS[a][0] + .13, POS[a][1] + .21),
                                             arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                             color=GREY, connectionstyle="arc3,rad=2.2",
                                             zorder=2))
                continue
            dec = (a, b) == (gb.DECISION, gb.GOAL)
            x0, y0 = POS[a]; x1, y1 = POS[b]
            dx, dy = x1 - x0, y1 - y0
            L = np.hypot(dx, dy)
            ax.add_patch(FancyArrowPatch(
                (x0 + .23 * dx / L, y0 + .23 * dy / L), (x1 - .25 * dx / L, y1 - .25 * dy / L),
                arrowstyle="-|>", mutation_scale=11, lw=2.2 if dec else 1.1,
                color=HL if dec else GREY, connectionstyle="arc3,rad=0.16", zorder=2))
    for s, (x, y) in POS.items():
        fc = "#ffe9d6" if s == gb.CUT else ("#eef4fa" if s == gb.GOAL else "white")
        ec = HL if s == gb.CUT else ("black" if s == gb.GOAL else GREY)
        ax.add_patch(Circle((x, y), .22, fc=fc, ec=ec, lw=2.0 if s in (gb.CUT, gb.GOAL) else 1.2,
                            zorder=3))
        ax.text(x, y, str(s), ha="center", va="center", fontsize=10, zorder=4)
    ax.annotate("every path to the goal\npasses through state 4",
                xy=(POS[gb.CUT][0], POS[gb.CUT][1] + .24), xytext=(3.1, 2.35),
                fontsize=8.5, color=HL, ha="center",
                arrowprops=dict(arrowstyle="->", color=HL, lw=1.2))
    ax.annotate("the decision:\nreach the goal or not",
                xy=(5.75, .58), xytext=(5.6, -.55), fontsize=8.5, color=HL, ha="center",
                arrowprops=dict(arrowstyle="->", color=HL, lw=1.2))
    ax.text(POS[gb.GOAL][0], POS[gb.GOAL][1] + .36, "goal", ha="center", fontsize=8.5)
    ax.set_xlim(-.5, 6.8); ax.set_ylim(-.95, 2.75); ax.axis("off")
    ax.set_title("A  nine states, one crossing", fontsize=10.5, loc="left")


def panel_rank(ax, g, ranks):
    true = val(g, "fb", ranks[0], 0, "gap_near_true")
    ax.axhline(true, color="black", ls="--", lw=1.2)
    ax.text(ranks[-1], true - .05, "the true value difference", ha="right",
            va="top", fontsize=8.5)
    ax.axhline(0, color="black", lw=.8)
    for meth, col, lab in [("density", DENS, "Density-FB"), ("fb", FBC, "FB")]:
        ax.plot(ranks, [val(g, meth, d, 0) for d in ranks], "-o", color=col, lw=2.2, ms=5,
                label=lab)
    ax.set_xticks(ranks); ax.set_xlabel("rank $d$", fontsize=9.5)
    ax.set_ylabel("value of reaching the goal\nminus value of not, at state 7", fontsize=9.5)
    ax.grid(True, axis="y", alpha=.3); ax.legend(fontsize=9, loc="center left")
    ax.set_title("B  what each critic thinks the goal is worth", fontsize=10.5, loc="left")


def panel_leak(ax, g, leaks, d=3):
    ax.plot(leaks, [val(g, "fb", d, k, "gap_near_true") for k in leaks], "--",
            color="black", lw=1.2, label="the truth")
    ax.axhline(0, color="black", lw=.8)
    for meth, col, lab in [("density", DENS, "Density-FB"), ("fb", FBC, "FB")]:
        ax.plot(leaks, [val(g, meth, d, k) for k in leaks], "-o", color=col, lw=2.2, ms=5,
                label=lab)
    ax.set_xlabel("how often a transition skips the crossing", fontsize=9.5)
    ax.set_ylabel("value of reaching the goal\nminus value of not, at state 7", fontsize=9.5)
    ax.grid(True, axis="y", alpha=.3)
    ax.legend(fontsize=9)
    ax.set_title("C  the crossing is what breaks it (rank 3)", fontsize=10.5, loc="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/didactic_graph"))
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--leaks", nargs="+", type=float, default=[0, .02, .05, .1, .2])
    ap.add_argument("--out", type=Path,
                    default=Path("PAPER/ICLR/didactic_new/figures/fig_graph.png"))
    a = ap.parse_args()
    g = load(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14.6, 3.8),
                           gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    panel_graph(ax[0]); panel_rank(ax[1], g, a.ranks); panel_leak(ax[2], g, a.leaks)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
