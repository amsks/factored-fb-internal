#!/usr/bin/env python3
"""Draw only the deterministic MDP for the spherical FB certificate.

The graph geometry comes from ../deterministic_path/draw.py. No unrestricted-FB
value panel or numeric optimum is imported. See README.md and caption.tex for
the separate constrained exact-target theorem and its scope.
"""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/deterministic-path-spherical-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
HERE = Path(__file__).resolve().parent
INK = "#233342"
MUTED = "#536371"
GREEN = "#148263"
LIGHT_GREEN = "#E3F3EB"

def export(fig, name):
    for extension in ("png", "pdf", "svg"):
        output = HERE / f"{name}.{extension}"
        fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.08)
        print(output, flush=True)
    plt.close(fig)

def draw_mdp():
    fig, ax = plt.subplots(figsize=(12, 3.3), facecolor="white")
    fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.17)
    ax.set(xlim=(-1.95, 10.95), ylim=(-1.43, 1.1), aspect="equal")
    ax.axis("off")
    positions = np.arange(4) * 3.0
    radius, arrow_y = .57, .24
    side = np.sqrt(radius ** 2 - arrow_y ** 2) + .045

    for x in positions[:-1]:
        for start, end, y, label in (
            (x + side, x + 3 - side, arrow_y, "Right"),
            (x + 3 - side, x + side, -arrow_y, "Left"),
        ):
            ax.add_patch(FancyArrowPatch(
                (start, y), (end, y), arrowstyle="-|>", mutation_scale=19,
                linewidth=1.9, color=INK, shrinkA=0, shrinkB=0, zorder=1,
            ))
            ax.text(x + 1.5, y + (.19 if y > 0 else -.19), label,
                    ha="center", va="center", fontsize=17, color=INK)

    for x, direction, label in ((0, -1, "Left"), (9, 1, "Right")):
        path = MplPath(
            [(x + direction * .42, .42), (x + direction * 1.74, 1.19),
             (x + direction * 1.74, -1.19), (x + direction * .42, -.42)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=19,
                                    linewidth=1.9, color=INK, zorder=1))
        ax.text(x + direction * 1.04, .92, label, ha="center", va="center",
                fontsize=17, color=INK)

    for state, x in enumerate(positions):
        is_goal = state == 1
        ax.add_patch(Circle((x, 0), radius, linewidth=2.2,
                            edgecolor=GREEN if is_goal else INK,
                            facecolor=LIGHT_GREEN if is_goal else "white", zorder=3))
        ax.text(x, 0, rf"$s_{state}$", ha="center", va="center", fontsize=27,
                color=GREEN if is_goal else INK, zorder=4)
        if state in (0, 1):
            ax.text(x, -.94, "Goal" if is_goal else "Start", ha="center",
                    va="center", fontsize=20, weight="bold",
                    color=GREEN if is_goal else INK)

    fig.text(.5, .057, "Deterministic transitions; reward +1 on entering Goal.",
             ha="center", va="center", fontsize=16, color=MUTED)
    export(fig, "mainmdp")

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    draw_mdp()
