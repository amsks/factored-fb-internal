#!/usr/bin/env python3
"""Draw only one initial configuration of the verified one-decision MDP."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/objective-counterexamples-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INK, MUTED, GOAL = "#182E40", "#657482", "#087D72"
ACTION_COLORS = ("#3B69A0", "#8B5B88")


def connect(ax, start, end, start_radius, end_radius, color, width=1.8):
    start, end = np.asarray(start, float), np.asarray(end, float)
    unit = (end - start) / np.linalg.norm(end - start)
    ax.add_patch(FancyArrowPatch(
        start + (start_radius + .04) * unit,
        end - (end_radius + .055) * unit,
        arrowstyle="-|>", mutation_scale=16, linewidth=width,
        color=color, zorder=1,
    ))


def node(ax, position, radius, label, sublabel=None, color=INK, fill="white"):
    ax.add_patch(Circle(position, radius, facecolor=fill, edgecolor=color,
                        linewidth=2, zorder=3))
    text_color = "white" if fill == GOAL else color
    x, y = position
    ax.text(x, y + (.13 if sublabel else 0), label, ha="center", va="center",
            fontsize=18, color=text_color, weight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - .20, sublabel, ha="center", va="center", fontsize=12.5,
                color=text_color, zorder=4)


def main():
    source = subprocess.run(
        [sys.executable, str(ROOT / "scripts/didactic/verify_objective_counterexample.py")],
        capture_output=True, text=True, check=True,
    )
    target = np.asarray(json.loads(source.stdout)["target_probabilities"][:2])
    np.testing.assert_allclose(target, [
        [.3275, .1725, .2825, .2175],
        [.2675, .2325, .3225, .1775],
    ], rtol=0, atol=1e-12)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    fig.text(.5, .929, "A simple one-decision MDP", ha="center", fontsize=26,
             weight="bold", color=INK)
    fig.text(.5, .871, "Choose A or B; then sample an outcome.", ha="center",
             fontsize=16, color=MUTED)
    ax = fig.add_axes([.035, .19, .93, .64])
    ax.set_xlim(-.15, 12.85)
    ax.set_ylim(-2.83, 2.83)
    ax.set_aspect("equal")
    ax.axis("off")

    start = (.55, 0)
    actions = ((2.90, 2.22), (2.90, -2.22))
    outcomes = [(x, 0) for x in (4.90, 6.80, 9.05, 12.00)]
    for index, action in enumerate(actions):
        color = ACTION_COLORS[index]
        connect(ax, start, action, .57, .46, INK, width=2)
        for j, outcome in enumerate(outcomes):
            connect(ax, action, outcome, .46, .57, color)
            # A horizontal row of labels keeps the four fan edges readable.
            label = .43 * np.asarray(action) + .57 * np.asarray(outcome)
            ax.text(*label, f"{100 * target[index, j]:.2f}%", fontsize=14,
                    color=color, ha="center", va="center", zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.6})
        node(ax, action, .46, "AB"[index], color=color)

    node(ax, start, .57, "Start", "x = +1")
    for j, outcome in enumerate(outcomes):
        is_goal = j == 2
        node(ax, outcome, .57, ("O1", "O2", "Goal", "O4")[j],
             f"r = {int(is_goal)}", color=GOAL if is_goal else INK,
             fill=GOAL if is_goal else "#F5F7FA")

    fig.text(.5, .116,
             "One initial configuration shown; the second has different probabilities.",
             ha="center", fontsize=13, color=MUTED)
    fig.text(.5, .063, "One decision. The episode ends at the sampled outcome.",
             ha="center", fontsize=13, color=MUTED)
    for suffix in ("png", "pdf", "svg"):
        path = HERE / f"simple_mdp.{suffix}"
        fig.savefig(path, dpi=190, facecolor="white")
        print(path)
    plt.close(fig)
    print("All eight transition probabilities match the verified construction.")


if __name__ == "__main__":
    main()
