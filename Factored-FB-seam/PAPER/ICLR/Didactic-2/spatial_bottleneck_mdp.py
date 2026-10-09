#!/usr/bin/env python3
"""Draw the minimal deterministic spatial bottleneck MDP."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/didactic2-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

HERE = Path(__file__).resolve().parent
INK, MUTED, GREEN, RED, BLUE = "#233342", "#536371", "#148263", "#C13B44", "#356AA5"

def edge(ax, p, q, label, color=INK, rad=0.0, offset=(0.0, 0.0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, lw=2.0,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=18, shrinkB=18))
    x = (p[0] + q[0]) / 2 + offset[0]
    y = (p[1] + q[1]) / 2 + offset[1]
    ax.text(x, y, label, ha="center", va="center", fontsize=11, color=color,
            bbox=dict(fc="white", ec="none", pad=1.5))

def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    ax.set_xlim(-1.2, 11.2); ax.set_ylim(-1.25, 4.5); ax.axis("off")
    pos = {"L0": (1.2, 1.8), "L1": (3.2, 1.8), "b": (6.2, 1.8),
           "R0": (9.0, 2.8), "R1": (9.0, 0.8)}
    for key in ("L0", "L1"):
        ax.add_patch(Circle(pos[key], .36, fc="white", ec=BLUE, lw=2.2))
    ax.add_patch(Circle(pos["b"], .43, fc="#FFF2D8", ec=RED, lw=3.0))
    for key in ("R0", "R1"):
        ax.add_patch(Circle(pos[key], .36, fc="white", ec=GREEN, lw=2.2))
    for key, label in (("L0", "$L_0$"), ("L1", "$L_1$"), ("b", "$b$"),
                       ("R0", "$R_0$"), ("R1", "$R_1$")):
        ax.text(*pos[key], label, ha="center", va="center", fontsize=16,
                color=RED if key == "b" else INK, weight="bold" if key == "b" else "normal")
    edge(ax, pos["L0"], pos["L1"], "move")
    edge(ax, pos["L1"], pos["L0"], "", rad=.22)
    edge(ax, pos["L1"], pos["b"], "approach", color=MUTED, offset=(0, .32))
    edge(ax, pos["b"], pos["R0"], "cross", color=GREEN, offset=(0, .28))
    edge(ax, pos["b"], pos["L1"], "fail", color=RED, rad=.45, offset=(0, -.34))
    edge(ax, pos["b"], pos["b"], "wait", color=RED, rad=.75, offset=(.82, .22))
    edge(ax, pos["R0"], pos["R1"], "move", offset=(.32, 0))
    edge(ax, pos["R1"], pos["R0"], "", rad=.22)
    ax.scatter([pos["L0"][0]], [pos["L0"][1] - .62], s=25, color=BLUE)
    ax.text(pos["L0"][0], pos["L0"][1] - .86, "start", ha="center", fontsize=11, color=BLUE)
    ax.text(pos["b"][0], 3.75, "single bottleneck state", ha="center", fontsize=15,
            weight="bold", color=RED)
    ax.text(pos["b"][0], 3.38, "cross unlocks the right-hand states; fail/wait do not",
            ha="center", fontsize=11.5, color=INK)
    ax.text(5.15, -0.82, "Five states, deterministic transitions, explicit actions",
            ha="center", fontsize=11, color=MUTED)
    fig.suptitle("Minimal spatial bottleneck MDP", fontsize=19, weight="bold", color=INK, y=.98)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"spatial_bottleneck_mdp.{ext}", dpi=240, facecolor="white",
                    bbox_inches="tight", pad_inches=.08)
    plt.close(fig)

if __name__ == "__main__":
    main()
