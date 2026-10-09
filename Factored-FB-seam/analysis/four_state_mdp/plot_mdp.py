"""The four-state control-bottleneck diagram for Section 4 (failure lemma).

(a) the base process: state 0 branches (1/2 to the bottleneck state 1, 1/4 to each of the absorbing
    states 2 and 3); at the bottleneck, a_+ moves to the rewarded state 2 and a_- to state 3.
(b) the discounted next-state probabilities at the bottleneck: m_+ = M(1, a_+, .) is a point mass at
    state 2, m_- = M(1, a_-, .) a point mass at state 3, and their difference Delta m.
The reward is 1 at state 2, so a_+ has the higher return; a rank-one successor fit collapses the two
rows to one and loses the whole difference (Lemma 1).  For feature dimension d the proof adds d-1
disconnected components; those do not touch this diagram.

    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/four_state_mdp/plot_mdp.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, RegularPolygon
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mdp as fsm     # noqa: E402

OUT = REPO / "PAPER" / "ICLR" / "figures" / "four-state-mdp"
INK, GREY, GREEN, BLUE, ORANGE = "#2a2a2a", "#9a9a9a", "#029E73", "#0173B2", "#DE8F05"
POS = {0: (0.0, 0.0), 1: (1.15, 0.0), 2: (2.35, 0.72), 3: (2.35, -0.72)}
FS_SINGLE = (4.6, 3.4)          # matches analysis/four_state_mdp/run.py FS_SINGLE
FS_WIDE = (9.2, 3.4)            # combined two-panel diagram


def panel_graph(ax):
    def arrow(s, t, color, lw, style, rad, alpha=0.95):
        ax.add_patch(FancyArrowPatch(POS[s], POS[t], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=7, linewidth=lw, linestyle=style, color=color,
                                     shrinkA=8, shrinkB=8, zorder=1, alpha=alpha))

    # state 0: stochastic branch, same under both actions (grey), near-straight so nothing crosses
    arrow(0, 1, GREY, 1.0, "-", 0.0)
    arrow(0, 2, GREY, 1.0, "-", 0.05)
    arrow(0, 3, GREY, 1.0, "-", -0.05)
    pbox = dict(boxstyle="round,pad=0.04", facecolor="white", edgecolor="none", alpha=0.95)
    ax.text(0.78, 0.0, "$\\frac{1}{2}$", color="#6a6a6a", fontsize=12, ha="center", va="center", bbox=pbox, zorder=4)
    ax.text(0.82, 0.256, "$\\frac{1}{4}$", color="#6a6a6a", fontsize=12, ha="center", va="center", bbox=pbox, zorder=4)
    ax.text(0.82, -0.256, "$\\frac{1}{4}$", color="#6a6a6a", fontsize=12, ha="center", va="center", bbox=pbox, zorder=4)
    # bottleneck actions
    arrow(1, 2, GREEN, 1.8, "-", 0.16)
    arrow(1, 3, BLUE, 1.8, (0, (3, 2)), -0.16)
    # nodes; absorbing states 2 and 3 get a double ring (standard terminal-state convention)
    styles = {0: ("white", GREY, INK), 1: (ORANGE, ORANGE, "white"),
              2: ("white", GREEN, INK), 3: ("white", BLUE, INK)}
    for s, (face, edge, txt) in styles.items():
        if s in (2, 3):
            ax.add_patch(Circle(POS[s], 0.255, facecolor="none", edgecolor=edge, linewidth=1.2, zorder=2))
        ax.add_patch(Circle(POS[s], 0.2, facecolor=face, edgecolor=edge, linewidth=1.5, zorder=2))
        ax.text(*POS[s], str(s), ha="center", va="center", fontsize=12, color=txt, zorder=3)
    # reward star on state 2
    ax.add_patch(RegularPolygon((POS[2][0] + 0.44, POS[2][1]), numVertices=5, radius=0.1,
                                orientation=0.0, facecolor=GREEN, edgecolor="none", zorder=3))
    ax.text(POS[2][0] + 0.62, POS[2][1], "$r=1$", color=GREEN, fontsize=12, ha="left", va="center")
    ax.text(POS[1][0], -0.5, "bottleneck", color=ORANGE, fontsize=12, ha="center", va="center")
    ax.text(POS[1][0] + 0.52, 0.56, "$a_+$", color=GREEN, fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(POS[1][0] + 0.58, -0.62, "$a_-$", color=BLUE, fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(POS[0][0], 0.42, "entry", color="#6a6a6a", fontsize=12, ha="center")
    ax.text(POS[2][0] + 0.02, 1.14, "absorbing", color="#6a6a6a", fontsize=11, ha="center")
    ax.set_xlim(-0.45, 3.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_rows(ax, t, b):
    n = t.shape[1]
    x = np.arange(n)
    w = 0.38
    ax.bar(x - w / 2, t[2 * b], w, color=GREEN, label="after $a_+$:  $m_+(s^+)$", zorder=2)
    ax.bar(x + w / 2, t[2 * b + 1], w, color=BLUE, label="after $a_-$:  $m_-(s^+)$", zorder=2)
    ax.plot(x, t[2 * b] - t[2 * b + 1], color=INK, marker="o", markersize=3.2, linewidth=0.9,
            label="difference $\\Delta m$", zorder=3)
    ax.axhline(0, color=GREY, linewidth=0.6, zorder=1)
    ax.axvspan(1.5, 2.5, color=GREEN, alpha=0.08, zorder=0)   # rewarded state 2
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(n)], fontsize=12.5)
    ax.set_xlabel("next state $s^+$", fontsize=16)
    ax.set_ylabel("discounted probability of $s^+$", fontsize=13.5)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.set_ylim(-1.2, 1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.4)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "text.color": INK,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    m = fsm.make_mdp(0)
    t = m["t_exact"]
    b = m["decision"]

    fig, ax = plt.subplots(figsize=FS_SINGLE)
    panel_graph(ax)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_four_state_graph.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FS_SINGLE)
    panel_rows(ax, t, b)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_four_state_densities.{ext}", dpi=300)
    plt.close(fig)

    # combined single-file diagram: the graph and the bottleneck densities side by side
    fig, axes = plt.subplots(1, 2, figsize=FS_WIDE, gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_graph(axes[0])
    panel_rows(axes[1], t, b)
    fig.tight_layout(pad=0.5, w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_four_state_mdp.{ext}", dpi=300)
    plt.close(fig)

    print("wrote", OUT / "fig_four_state_graph.pdf", ",", OUT / "fig_four_state_densities.pdf",
          "and", OUT / "fig_four_state_mdp.pdf")
    print("m+ ", np.round(t[2 * b], 3))
    print("m- ", np.round(t[2 * b + 1], 3))
    print("dm ", np.round(t[2 * b] - t[2 * b + 1], 3), " gap =", m["gap"])

    # the single connected shared-tail MDP used by the fixed-rank ablation, drawn whole
    sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
    sys.path.insert(0, str(REPO / "scripts" / "didactic"))
    import design_single_bottleneck as dsb          # noqa: E402
    import plot_single_bottleneck_mdp as psb         # noqa: E402
    edges, bb, A, B, T = dsb.build(entry=1, cluster=2, tail=6, merge=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    psb.panel_graph(ax, edges, bb, A, B, T)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_connected_graph.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_connected_graph.pdf", f"(connected MDP, n={len(edges)})")


if __name__ == "__main__":
    main()
