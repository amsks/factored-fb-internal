"""Draw the proposed three-state MDP; no fitted models or experimental results.

Run with a Python environment containing numpy and matplotlib:
    MPLCONFIGDIR=/tmp/didactic-mpl python draw_mdp.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
import numpy as np


HERE = Path(__file__).resolve().parent
STATES = ("Left", "Centre", "Right")
COLORS = ("#0072B2", "#D55E00", "#008875")
INK, MUTED, EDGE = "#172B3A", "#526674", "#718390"
POSITIONS = np.array([[-1.05, -0.42], [0.0, 1.13], [1.05, -0.42]])
RADIUS = 0.34
BETA = 0.9
COMMAND = np.full((3, 3), 0.1)
np.fill_diagonal(COMMAND, 0.8)
# Axes: action, current state, next state.
TRANSITIONS = np.stack([
    BETA * np.eye(3) + (1 - BETA) * np.tile(COMMAND[a], (3, 1))
    for a in range(3)
])


def arrow(ax, start, end, color, width=1.65, **kwargs):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, zorder=2, **kwargs,
    )
    ax.add_patch(patch)


def loop(ax, state, probability, color):
    centre = POSITIONS[state]
    direction = centre - POSITIONS.mean(axis=0)
    direction /= np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    start = centre + RADIUS * (0.65 * direction + 0.76 * perp)
    end = centre + RADIUS * (0.65 * direction - 0.76 * perp)
    control1 = centre + 1.12 * direction + 0.58 * perp
    control2 = centre + 1.12 * direction - 0.58 * perp
    path = MplPath([start, control1, control2, end],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=1.65, zorder=2,
    ))
    label = centre + 1.02 * direction
    ax.text(*label, f"{probability:.0%}", color=color, fontsize=12,
            ha="center", va="center", fontweight="semibold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=5)


def panel(ax, action):
    accent = COLORS[action]
    for source in range(3):
        for dest in range(3):
            probability = TRANSITIONS[action, source, dest]
            color = accent if dest == action else EDGE
            if source == dest:
                loop(ax, source, probability, color)
                continue
            origin, target = POSITIONS[source], POSITIONS[dest]
            delta = target - origin
            length = np.linalg.norm(delta)
            unit = delta / length
            start = origin + (RADIUS + 0.02) * unit
            end = target - (RADIUS + 0.02) * unit
            arrow(ax, start, end, color,
                  width=2.15 if dest == action else 1.4,
                  connectionstyle="arc3,rad=0.20")
            # Arc3 bends towards the right-hand normal of the directed edge.
            right = np.array([unit[1], -unit[0]])
            label = (origin + target) / 2 + 0.225 * right
            ax.text(*label, f"{probability:.0%}", ha="center", va="center",
                    fontsize=11.5, color=color, fontweight="semibold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
    for i, centre in enumerate(POSITIONS):
        ax.add_patch(Circle(
            centre, RADIUS, facecolor=accent if i == action else "#F0F4F7",
            edgecolor=accent if i == action else "#9DAEB9", linewidth=1.7, zorder=4,
        ))
        ax.text(*centre, STATES[i], color="white" if i == action else INK,
                fontsize=12.5, ha="center", va="center", fontweight="semibold", zorder=6)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.75, 2.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Action: place {STATES[action].lower()}",
                 fontsize=16, color=accent, fontweight="semibold", pad=8)


def main():
    assert np.all(TRANSITIONS >= 0)
    np.testing.assert_allclose(TRANSITIONS.sum(axis=-1), 1)
    for action in range(3):
        for state in range(3):
            expected = np.full(3, 0.01)
            expected[action] = 0.08
            expected[state] += 0.9
            np.testing.assert_allclose(TRANSITIONS[action, state], expected)
    behavior = TRANSITIONS.mean(axis=0)
    np.testing.assert_allclose(np.full(3, 1 / 3) @ behavior, np.full(3, 1 / 3))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14.7, 7.2), facecolor="white")
    fig.text(0.5, 0.94, "A three-location placement MDP", ha="center",
             fontsize=24, color=INK, fontweight="semibold")
    fig.text(0.5, 0.885,
             "State = object location. All three placement actions are available in every state.",
             ha="center", fontsize=12.5, color=MUTED)
    for action in range(3):
        ax = fig.add_axes([0.018 + action * 0.326, 0.255, 0.31, 0.535])
        panel(ax, action)
    fig.text(0.5, 0.233,
             "Each panel is the same three states under a different action. Arrow labels are total one-step probabilities.",
             ha="center", fontsize=11.5, color=MUTED)
    box = FancyBboxPatch(
        (0.045, 0.052), 0.91, 0.132, transform=fig.transFigure,
        boxstyle="round,pad=0.014,rounding_size=0.012",
        linewidth=0, facecolor="#F0F4F7", zorder=0,
    )
    fig.add_artist(box)
    fig.text(0.065, 0.143, "The transition rule", fontsize=12.5,
             color=INK, fontweight="semibold")
    fig.text(0.065, 0.103,
             "90%: keep the current location.   10%: attempt placement — 80% intended location, 10% each other location.",
             fontsize=11.5, color=INK)
    fig.text(0.065, 0.066,
             "A command to the current location stays there with probability 98%; a command elsewhere stays put with probability 91%.",
             fontsize=11, color=MUTED)
    for suffix in ("png", "pdf", "svg"):
        target = HERE / f"placement_mdp.{suffix}"
        fig.savefig(target, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        print(target)
    plt.close(fig)
    data = {
        "states": list(STATES), "actions": [f"place {s.lower()}" for s in STATES],
        "axes": ["action", "current_state", "next_state"],
        "transition_probabilities": TRANSITIONS.round(12).tolist(),
        "uniform_behavior_stationary_distribution": [1 / 3] * 3,
        "expected_consecutive_steps_at_a_state_under_uniform_behavior": 15,
        "description": "Proposed MDP only; no training or model-performance claims.",
    }
    (HERE / "transitions.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Verified all transition rows and uniform-policy stationary distribution.")


if __name__ == "__main__":
    main()
