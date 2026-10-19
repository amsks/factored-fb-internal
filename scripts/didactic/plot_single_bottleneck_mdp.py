"""Single-bottleneck MDP diagram and its successor measures at the bottleneck.

JAX_PLATFORMS=cpu python scripts/didactic/plot_single_bottleneck_mdp.py
"""

import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
import os

import design_single_bottleneck as ds  # noqa: E402

OUT = pathlib.Path(os.environ.get("SB_GRAPH_OUT", REPO / "figures" / "didactic"))
INK, GREY, GREEN, ORANGE, BLUE = "#2a2a2a", "#9a9a9a", "#029E73", "#DE8F05", "#0173B2"
ENTRY, TAIL = (
    (int(sys.argv[1]) if len(sys.argv) > 1 else 2),
    (int(sys.argv[2]) if len(sys.argv) > 2 else 4),
)
CLUSTER = int(sys.argv[3]) if len(sys.argv) > 3 else 2
PASSTHROUGH = (
    os.environ.get("SB_PASSTHROUGH", "0") == "1"
)  # clusters step straight into the tail
TAIL_REWARD = os.environ.get("SB_TAIL_REWARD")  # tail reward label, optional
MUTUAL = TAIL == 0  # no tail: clusters leak into each other


def layout(b, A, B, T):
    pos = {}
    for i in range(b):
        pos[i] = (0.72 * i, 0.0)
    pos[b] = (0.72 * b + 0.15, 0.0)
    xa = pos[b][0] + 0.95
    for i, s in enumerate(A):
        pos[s] = (xa + 0.78 * i, 0.72)
    for i, s in enumerate(B):
        pos[s] = (xa + 0.78 * i, -0.72)
    cx = xa + 0.78 * (len(A) - 1) + 1.2
    for i, s in enumerate(T):
        ang = np.pi - 2 * np.pi * i / len(T)
        pos[s] = (cx + 0.6 * np.cos(ang), 0.62 * np.sin(ang))
    return pos


def panel_graph(ax, edges, b, A, B, T):
    n = len(edges)
    pos = layout(b, A, B, T)
    FS = 1.3 if PASSTHROUGH else 1.0
    for s in range(n):
        for a in (0, 1):
            t = int(edges[s, a])
            if s == b:
                color, lw = (GREEN if a == 0 else BLUE), 1.6
            else:
                color, lw = (INK if a == 0 else GREY), 0.75
            style = "-" if a == 0 else (0, (3, 2))
            if t == s:
                c = np.array(pos[s])
                ax.add_patch(
                    Circle(
                        c + (0, 0.3),
                        0.1,
                        fill=False,
                        edgecolor=color,
                        linewidth=lw,
                        linestyle=style,
                        zorder=1,
                    )
                )
                continue
            ax.add_patch(
                FancyArrowPatch(
                    pos[s],
                    pos[t],
                    connectionstyle="arc3,rad=0.22",
                    arrowstyle="-|>",
                    mutation_scale=6.5,
                    linewidth=lw,
                    linestyle=style,
                    color=color,
                    shrinkA=7,
                    shrinkB=7,
                    zorder=1,
                    alpha=0.95,
                )
            )
    for s in range(n):
        if s == b:
            face, edge, txt = ORANGE, ORANGE, "white"
        elif s in A:
            face, edge, txt = "white", GREEN, INK
        elif s in B:
            face, edge, txt = "white", BLUE, INK
        else:
            face, edge, txt = "white", GREY, INK
        ax.add_patch(
            Circle(pos[s], 0.2, facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2)
        )
        ax.text(
            *pos[s],
            str(s),
            ha="center",
            va="center",
            fontsize=7.5 * FS,
            color=txt,
            zorder=3,
        )
    compact = PASSTHROUGH
    if compact:
        # Labels beside their nodes
        ax.text(
            pos[b][0],
            pos[b][1] + 0.3,
            "bottleneck",
            color=ORANGE,
            fontsize=7 * FS,
            ha="center",
            va="bottom",
        )
        ax.text(
            pos[A[0]][0] - 0.27,
            pos[A[0]][1],
            "A (reward 1)",
            color=GREEN,
            fontsize=7 * FS,
            ha="right",
            va="center",
        )
        ax.text(
            pos[B[0]][0] - 0.27,
            pos[B[0]][1] - 0.12,
            "B (reward 0)",
            color=BLUE,
            fontsize=7 * FS,
            ha="right",
            va="center",
        )
    else:
        ax.text(
            pos[b][0] - 0.1,
            -0.5,
            "bottleneck",
            color=ORANGE,
            fontsize=7 * FS,
            ha="center",
            va="center",
        )
        word = "cluster " if len(A) > 1 else ""
        ax.text(
            pos[A[0]][0] + 0.39 * (len(A) - 1),
            1.02 if len(A) > 1 else 1.2,
            f"{word}A (reward 1)",
            color=GREEN,
            fontsize=7 * FS,
            ha="center",
        )
        ax.text(
            pos[B[0]][0] + 0.39 * (len(B) - 1),
            -1.1,
            f"{word}B (reward 0)",
            color=BLUE,
            fontsize=7 * FS,
            ha="center",
        )
    cx = np.mean([pos[s][0] for s in T]) if T else pos[A[-1]][0]
    if T:
        ax.text(
            cx,
            0.86 if compact else 0.98,
            "shared tail" + (f" (reward {TAIL_REWARD})" if TAIL_REWARD else ""),
            color="#6a6a6a",
            fontsize=7 * FS,
            ha="center",
        )
    ax.text(
        pos[b][0] + 0.52,
        0.48,
        "$a_+$",
        color=GREEN,
        fontsize=7.5 * FS,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9
        ),
    )
    ax.text(
        pos[b][0] + (0.3 if PASSTHROUGH else 0.62),
        -0.42 if PASSTHROUGH else -0.6,
        "$a_-$",
        color=BLUE,
        fontsize=7.5 * FS,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9
        ),
    )
    ax.text(
        0.72 * (b - 1) / 2,
        -0.42 if PASSTHROUGH else 0.45,
        "entry",
        color="#6a6a6a",
        fontsize=7 * FS,
        ha="center",
    )
    ax.set_xlim(-0.35, cx + 0.9)
    ax.set_ylim(*((-0.95, 1.0) if compact else (-1.3, 1.2 if len(A) > 1 else 1.35)))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color=INK, linewidth=0.9, label="first action"),
            Line2D(
                [0],
                [0],
                color=GREY,
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                label="second action",
            ),
        ],
        loc="upper left" if compact else "lower left",
        fontsize=6.5 * FS,
        frameon=False,
        handlelength=1.6,
        bbox_to_anchor=(-0.02, 1.02) if compact else (-0.02, -0.06),
    )


def panel_rows(ax, t, b, A, B, T):
    """Plot the normalized successor measure t = (1-gamma) M of both actions at b."""
    n = t.shape[1]
    x = np.arange(n)
    w = 0.38
    ax.bar(
        x - w / 2, t[2 * b], w, color=GREEN, label="after $a_+$:  $m_+(s^+)$", zorder=2
    )
    ax.bar(
        x + w / 2,
        t[2 * b + 1],
        w,
        color=BLUE,
        label="after $a_-$:  $m_-(s^+)$",
        zorder=2,
    )
    ax.plot(
        x,
        t[2 * b] - t[2 * b + 1],
        color=INK,
        marker="o",
        markersize=2.8,
        linewidth=0.9,
        label="difference $\\Delta m$",
        zorder=3,
    )
    ax.axhline(0, color=GREY, linewidth=0.6, zorder=1)
    for s_, col in [(A, GREEN), (B, BLUE)]:
        ax.axvspan(s_[0] - 0.5, s_[-1] + 0.5, color=col, alpha=0.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(n)], fontsize=7)
    ax.set_xlabel("next state $s^+$", fontsize=7.5)
    ax.set_ylabel("discounted probability of $s^+$", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right", handlelength=1.4)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "text.color": INK,
            "pdf.fonttype": 42,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    if PASSTHROUGH:
        edges, b, A, B, T = ds.build_passthrough(entry=ENTRY, tail=TAIL)
    else:
        edges, b, A, B, T = ds.build(
            entry=ENTRY, cluster=CLUSTER, tail=TAIL, merge=not MUTUAL, mutual=MUTUAL
        )
    m, t = ds.exact_ratio(edges)
    stem = "fig_single_bottleneck" + ("_notail" if MUTUAL else "")
    fig, ax = plt.subplots(figsize=(3.6 if MUTUAL else 3.9, 2.5))
    panel_graph(ax, edges, b, A, B, T)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}_graph.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    panel_rows(ax, t, b, A, B, T)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}_densities.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        "wrote",
        OUT / f"{stem}_graph.pdf",
        "and",
        OUT / f"{stem}_densities.pdf",
        f"(entry {ENTRY}, clusters {CLUSTER}+{CLUSTER}, tail {TAIL})",
    )
    print("edges:", edges.tolist())
    i0, i1 = 2 * b, 2 * b + 1
    print("m+ ", np.round(t[i0], 3))
    print("m- ", np.round(t[i1], 3))
    print("dm ", np.round(t[i0] - t[i1], 3))


if __name__ == "__main__":
    main()
