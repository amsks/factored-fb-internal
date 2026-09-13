"""The single didactic figure: FB loses the decision after the bottleneck.

Panels: the MDP, the action gap at each decision along the route at rank 2, and
the gap against rank. Reads analysis/didactic_family; no training here.
"""
import argparse, collections, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import bottleneck_family_td as bf

FB, DENS, BAD, GREY = "#0173b2", "#cc78bc", "#d55e00", "#949494"
SLAB = ["$L_0$", "$L_1$", "$b$", "$R_0$", "$R_1$"]
ROUTE = ["$L_0$\nfar", "$L_1$\napproach", "$b$\ncross", "$R_0$\ntransport"]


def load(root, sub):
    g = collections.defaultdict(list)
    for r in json.load(open(root / sub / "summary.json")):
        g[(r["method"], r["d"], r["R"])].append(r)
    return g


def route(g, method, d=2, R=2):
    a = np.array([r["state_gap_ratios"] for r in g[(method, d, R)]])[:, :4]
    return a.mean(0), a.min(0), a.max(0)


def panel_mdp(ax, fb_route):
    for s in range(5):
        ec = "black" if s == 4 else (FB if fb_route[s] > 0 else BAD)
        ax.add_patch(Circle((s, 0), .3, fc="white", ec=ec, lw=2.2 if s != 4 else 1.2, zorder=3))
        ax.text(s, 0, SLAB[s], ha="center", va="center", fontsize=11, zorder=4)
    ax.text(4, .46, "goal", ha="center", fontsize=8.5)
    for a, b, lab in [(0, 1, "move"), (1, 2, "reach"), (2, 3, "cross"), (3, 4, "place")]:
        ax.add_patch(FancyArrowPatch((a + .32, .1), (b - .32, .1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.6, color="black", zorder=2))
        ax.text((a + b) / 2, .22, lab, ha="center", fontsize=8.5)
    for a, b, lab in [(0, 0, "wait"), (1, 0, "back"), (2, 1, "slip"), (3, 3, "wait")]:
        st = dict(arrowstyle="-|>", mutation_scale=10, lw=1.2, color=GREY, zorder=2)
        if a == b:
            ax.add_patch(FancyArrowPatch((a - .22, -.22), (a + .22, -.22),
                                         connectionstyle="arc3,rad=1.9", **st))
            ax.text(a, -.92, lab, ha="center", fontsize=8, color=GREY)
        else:
            ax.add_patch(FancyArrowPatch((a - .2, -.22), (b + .2, -.22),
                                         connectionstyle="arc3,rad=0.4", **st))
            ax.text((a + b) / 2, -.82, lab, ha="center", fontsize=8, color=GREY)
    ax.annotate("FB picks\nthe wrong action", xy=(3, .34), xytext=(3.2, 1.12), fontsize=9,
                color=BAD, ha="center", arrowprops=dict(arrowstyle="->", color=BAD, lw=1.3))
    for x0, x1, lab in [(-.35, 1.35, "approach"), (1.6, 2.4, "acquire control"),
                        (2.6, 4.35, "transport")]:
        ax.plot([x0, x1], [-1.25, -1.25], color="black", lw=.9)
        ax.text((x0 + x1) / 2, -1.52, lab, ha="center", fontsize=9)
    ax.set_xlim(-.65, 4.65); ax.set_ylim(-1.85, 1.5); ax.axis("off")
    ax.set_title("A  five-state bottleneck", fontsize=10.5, loc="left")


def panel_route(ax, g):
    x = np.arange(4)
    ax.axhspan(-.6, 0, color=BAD, alpha=.10, lw=0)
    ax.axhline(1, color="black", ls="--", lw=1); ax.axhline(0, color="black", lw=.8)
    for method, col, lab in [("density", DENS, "Density-FB"), ("fb", FB, "FB")]:
        m, lo, hi = route(g, method)
        ax.plot(x, m, "-o", color=col, lw=2.2, ms=5, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=.18, lw=0)
    ax.text(.04, .045, "wrong action", transform=ax.transAxes, fontsize=8.5, color=BAD)
    ax.set_xticks(x); ax.set_xticklabels(ROUTE, fontsize=9)
    ax.set_ylabel("learned gap $\\div$ exact gap", fontsize=9.5)
    ax.set_ylim(-.6, 1.6); ax.grid(True, axis="y", alpha=.3); ax.legend(fontsize=9)
    ax.set_title("B  action gap at each decision (rank 2)", fontsize=10.5, loc="left")


def panel_rank(ax, g, ranks, rooms):
    ax.axhspan(-.6, 0, color=BAD, alpha=.10, lw=0)
    ax.axhline(1, color="black", ls="--", lw=1); ax.axhline(0, color="black", lw=.8)
    for method, col, lab in [("density", DENS, "Density-FB"), ("fb", FB, "FB")]:
        Y = np.array([[np.mean([r["gap_ratio_near"] for r in g[(method, d, R)]])
                       for d in ranks] for R in rooms])
        ax.plot(ranks, Y.mean(0), "-o", color=col, lw=2.2, ms=5, label=lab)
        ax.fill_between(ranks, Y.min(0), Y.max(0), color=col, alpha=.18, lw=0)
    ax.set_xticks(ranks); ax.set_xlabel("rank $d$", fontsize=9.5)
    ax.set_ylabel("gap $\\div$ exact gap at transport", fontsize=9.5)
    ax.set_ylim(-.6, 1.6); ax.grid(True, axis="y", alpha=.3); ax.legend(fontsize=9)
    ax.set_title("C  same decision against rank ($n$ = 5 to 9)", fontsize=10.5, loc="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/didactic_family"))
    ap.add_argument("--sub", default="main3")
    ap.add_argument("--out", type=Path,
                    default=Path("PAPER/ICLR/didactic_new/figures/fig_didactic.png"))
    a = ap.parse_args()
    g = load(a.root, a.sub)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.7),
                           gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    panel_mdp(ax[0], route(g, "fb")[0])
    panel_route(ax[1], g)
    panel_rank(ax[2], g, [2, 3, 4, 6], [2, 3, 4, 5, 6])
    fig.tight_layout(w_pad=2.0)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
