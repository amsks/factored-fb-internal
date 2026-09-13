"""The five-second version: the graph, and what each critic thinks the two
actions at the decision state are worth.

Reads analysis/didactic_graph/bars (rank 3, one crossing). No training here.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import graph_bottleneck_td as gb
from plot_graph_figure import panel_graph

FBC, DENS, HL = "#0173b2", "#cc78bc", "#d55e00"
GAMMA = 0.8


def values(rows, method):
    q = np.array([r["q_decision"] for r in rows if r["method"] == method])
    return q.mean(0) / (1 - GAMMA)          # [reach the goal, go back]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("analysis/didactic_graph/bars"))
    ap.add_argument("--out", type=Path,
                    default=Path("PAPER/ICLR/didactic_new/figures/fig_simple.png"))
    a = ap.parse_args()
    rows = json.load(open(a.run / "summary.json"))
    truth = np.array(rows[0]["q_decision_true"]) / (1 - GAMMA)
    bars = [("the truth", truth, "#444444"),
            ("FB", values(rows, "fb"), FBC),
            ("Density-FB", values(rows, "density"), DENS)]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.3, 1.0]})
    panel_graph(ax[0])
    ax[0].set_title("", loc="left")

    x = np.arange(3)
    w = 0.34
    for i, (lab, v, col) in enumerate(bars):
        ax[1].bar(x[i] - w / 2, v[0], w, color=col, edgecolor="black", lw=.4)
        ax[1].bar(x[i] + w / 2, v[1], w, color=col, edgecolor="black", lw=.4, alpha=.35)
        for dx, val in ((-w / 2, v[0]), (w / 2, v[1])):
            ax[1].text(x[i] + dx, val + .06, f"{val:.2f}", ha="center", fontsize=9)
    ax[1].bar(0, 0, 0, color="#888888", label="reach the goal")
    ax[1].bar(0, 0, 0, color="#888888", alpha=.35, label="go back")
    ax[1].annotate("FB says both\nare the same", xy=(1, values(rows, "fb").max() + .12),
                   xytext=(1, 2.45), ha="center", fontsize=10.5, color=HL,
                   arrowprops=dict(arrowstyle="->", color=HL, lw=1.4))
    ax[1].set_xticks(x); ax[1].set_xticklabels([b[0] for b in bars], fontsize=10)
    ax[1].set_ylabel("value at state 7", fontsize=10)
    ax[1].set_ylim(0, 2.95)
    ax[1].legend(fontsize=9, loc="upper right")
    ax[1].grid(True, axis="y", alpha=.3)
    ax[1].set_title("what the two actions at state 7 are worth", fontsize=11, loc="left")
    fig.tight_layout(w_pad=2.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
