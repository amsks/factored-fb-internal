"""The bottleneck MDP diagram for Section 4 (failure.tex), drawn to match the lemma.

(a) The nine-state process of graph_bottleneck_td.py at leak 0. Solid arrows are the first
    action, dashed the second. The cut (orange) is the only way from the first group into the
    second and nothing returns. The decision state (green outline) is the control bottleneck the
    experiments measure: its first action a+ enters the goal (filled green), its second a- goes
    back into the group.
(b) The two successor densities the lemma compares, m+ = m(s7, a+) and m- = m(s7, a-), relative
    to the uniform reference rho = 1/9, over the nine states, and their difference Delta m.
    Every downstream reward reads the action gap through <Delta m, r>_rho, so a fit that drops
    the part of Delta m outside its feature span loses exactly that inner product.

    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python scripts/didactic/plot_lemma_mdp.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_bottleneck_td as gb          # noqa: E402
import bottleneck_family_td as bf         # noqa: E402

REPO = HERE.parent.parent
OUT = REPO / "PAPER" / "ICLR" / "figures" / "didactic"
GAMMA = 0.8
INK, GREY, GREEN, ORANGE, APLUS, AMINUS = "#2a2a2a", "#9a9a9a", "#029E73", "#DE8F05", "#029E73", "#8a8a8a"
POS = {0: (0.0, 0.0), 1: (0.9, 0.7), 2: (0.9, -0.7), 3: (1.8, 0.0),
       4: (2.9, 0.0),
       5: (4.0, 0.0), 6: (4.9, 0.7), 7: (4.9, -0.7), 8: (5.8, 0.0)}


def exact_rows():
    n = len(gb.EDGES)
    p = gb.build_graph(0.0).reshape(-1, n)                  # (2n, n), row 2s + a
    t = bf.exact(p, np.full((n, 2), 0.5), GAMMA)            # (1-gamma) M, rows sum to 1
    m = t / (1 - GAMMA) * n                                 # density ratio w.r.t. rho = 1/n
    return m


def panel_graph(ax):
    n = len(gb.EDGES)
    for s in range(n):
        for a in (0, 1):
            t = int(gb.EDGES[s, a])
            if s == gb.DECISION:
                color, lw, style = (APLUS if a == 0 else AMINUS), 1.6, ("-" if a == 0 else (0, (3, 2)))
            else:
                color, lw, style = (INK if a == 0 else GREY), 0.8, ("-" if a == 0 else (0, (3, 2)))
            if t == s:
                c = np.array(POS[s]); ax.add_patch(Circle(c + (0.0, -0.3), 0.11, fill=False, edgecolor=color,
                                                          linewidth=lw, linestyle=style, zorder=1))
                continue
            ax.add_patch(FancyArrowPatch(POS[s], POS[t], connectionstyle="arc3,rad=0.2", arrowstyle="-|>",
                                         mutation_scale=7, linewidth=lw, linestyle=style, color=color,
                                         shrinkA=8, shrinkB=8, zorder=1, alpha=0.95))
    for s in range(n):
        if s == gb.GOAL:
            face, edge, txt = GREEN, GREEN, "white"
        elif s == gb.DECISION:
            face, edge, txt = "white", GREEN, INK
        elif s == gb.CUT:
            face, edge, txt = ORANGE, ORANGE, "white"
        else:
            face, edge, txt = "white", GREY, INK
        ax.add_patch(Circle(POS[s], 0.19, facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2))
        ax.text(*POS[s], str(s), ha="center", va="center", fontsize=7, color=txt, zorder=3)
    ax.text(5.62, -0.52, "$a_+$", color=APLUS, fontsize=7.5, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(5.28, 0.08, "$a_-$", color="#6a6a6a", fontsize=7.5, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(2.9, 0.42, "cut", color=ORANGE, fontsize=6.5, ha="center", va="center")
    ax.text(4.9, -1.12, "decision", color=GREEN, fontsize=6.5, ha="center", va="center")
    ax.text(5.8, 0.42, "goal", color=GREEN, fontsize=6.5, ha="center", va="center")
    ax.set_xlim(-0.45, 6.25); ax.set_ylim(-1.3, 1.05)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], color=INK, linewidth=0.9, label="first action"),
                       Line2D([0], [0], color=GREY, linewidth=0.9, linestyle=(0, (3, 2)), label="second action")],
              loc="lower left", fontsize=6.2, frameon=False, handlelength=1.6, bbox_to_anchor=(-0.04, -0.06))


def panel_rows(ax, m):
    n = m.shape[1]
    i0, i1 = 2 * gb.DECISION, 2 * gb.DECISION + 1
    x = np.arange(n); w = 0.38
    ax.bar(x - w / 2, m[i0], w, color=APLUS, label="$m_+ = m(s_7, a_+,\\cdot)$", zorder=2)
    ax.bar(x + w / 2, m[i1], w, color=AMINUS, label="$m_- = m(s_7, a_-,\\cdot)$", zorder=2)
    ax.plot(x, m[i0] - m[i1], color=INK, marker="o", markersize=3, linewidth=0.9, label="$\\Delta m$", zorder=3)
    ax.axhline(0, color=GREY, linewidth=0.6, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in range(n)], fontsize=7)
    ax.set_xlabel("next state $s^+$", fontsize=7.5)
    ax.set_ylabel("density w.r.t. $\\rho=1/9$", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left", handlelength=1.4)
    gap = float((m[i0] - m[i1]) @ np.eye(n)[gb.GOAL]) / n
    ax.set_title(f"successor densities at the decision state; gap for the goal reward = {gap:.2f}",
                 fontsize=7, color=INK, loc="left")


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "text.color": INK,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    m = exact_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.2), gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    panel_graph(axes[0]); panel_rows(axes[1], m)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_lemma_mdp.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig_lemma_mdp.pdf")
    i0, i1 = 2 * gb.DECISION, 2 * gb.DECISION + 1
    print("m+ ", np.round(m[i0], 3)); print("m- ", np.round(m[i1], 3)); print("dm ", np.round(m[i0] - m[i1], 3))


if __name__ == "__main__":
    main()
