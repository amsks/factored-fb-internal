#!/usr/bin/env python3
"""Draw two verified finite-capacity, one-decision MDP counterexamples.

Run from any directory with Python, NumPy, SciPy, and Matplotlib installed.
The existing numerical verification scripts are the source of every plotted
transition, prediction, action gap, and objective certificate. No training or
new numerical optimization of a density model is performed here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/objective-counterexamples-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INK = "#182E40"
MUTED = "#5C6D7B"
EDGE = "#DAE1E7"
PANEL = "#F7F9FC"
FB = "#B75526"
DENSITY = "#087D72"
GOAL = "#087D72"
GOAL_PALE = "#E4F3ED"
OUTCOME_FILL = ("#EAF0F8", "#F0EDF6", GOAL_PALE, "#EDF1F4")
OUTCOME_INK = ("#42658C", "#786690", GOAL, "#647889")


def rounded(ax, x, y, width, height, *, fill="white", edge=EDGE, radius=.09, lw=1):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, width=1.5, **kwargs):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        color=color, linewidth=width, shrinkA=0, shrinkB=0, **kwargs,
    ))


def plain_axes(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def mdp_panel(fig, rect, target, context):
    """Action arrows enter conditional probability strips below four terminals."""
    ax = plain_axes(fig, rect, (0, 7.3), (0, 3.65))
    rounded(ax, .02, .02, 7.26, 3.61, fill=PANEL, radius=.15)
    sign = "+1" if context == 0 else "−1"
    ax.text(.25, 3.30, f"Initial configuration {context + 1}",
            fontsize=15, weight="bold", color=INK, va="center")
    ax.text(7.01, 3.30, f"x = {sign}", fontsize=12, color=MUTED,
            ha="right", va="center")

    # Both diagrams display the same four terminal states; the transition
    # probabilities depend on the initial configuration and chosen action.
    columns = np.arange(4) * 1.035 + 3.10
    row_y = (1.73, .87)
    table_left, table_right = 2.60, 6.72
    for j, x in enumerate(columns):
        rounded(ax, x - .475, .44, .95, 1.80,
                fill=OUTCOME_FILL[j], edge="none", radius=.06)
        arrow(ax, (x, 2.20), (x, 2.40), color=OUTCOME_INK[j], width=1.0)
        rounded(ax, x - .475, 2.41, .95, .62,
                fill=GOAL if j == 2 else "white",
                edge=GOAL if j == 2 else EDGE, radius=.15)
        color = "white" if j == 2 else INK
        label = ("O1", "O2", "Goal", "O4")[j]
        ax.text(x, 2.82, label, ha="center", va="center", color=color,
                fontsize=13, weight="bold")
        ax.text(x, 2.57, f"r = {int(j == 2)}", ha="center", va="center",
                color=color if j == 2 else MUTED, fontsize=10.5)
    ax.plot([table_left, table_right], [1.29, 1.29], color="white", lw=2)

    start = (.65, 1.30)
    ax.add_patch(Circle(start, .33, facecolor="white", edgecolor="#92A4B4", lw=1.5))
    ax.text(*start, "Start", ha="center", va="center", fontsize=11.5,
            color=INK, weight="bold")
    for action_index, y in enumerate(row_y):
        label = "A" if action_index == 0 else "B"
        action_sign = "+1" if action_index == 0 else "−1"
        arrow(ax, (.93, 1.45 if action_index == 0 else 1.15), (1.52, y))
        rounded(ax, 1.52, y - .31, .73, .62, edge="#92A4B4", radius=.11)
        ax.text(1.885, y + .10, label, ha="center", va="center", color=INK,
                fontsize=14, weight="bold")
        ax.text(1.885, y - .14, f"({action_sign})", ha="center", va="center",
                fontsize=9.5, color=MUTED)
        arrow(ax, (2.25, y), (2.59, y), width=1.4)
        row = 2 * context + action_index
        for j, x in enumerate(columns):
            ax.text(x, y, f"{100 * target[row, j]:.2f}%", ha="center",
                    va="center", color=GOAL if j == 2 else INK,
                    fontsize=12.8, weight="bold" if j == 2 else "normal")
    ax.text(4.65, .20, "Choose an action → sample one terminal outcome",
            ha="center", va="center", fontsize=10.2, color=MUTED)
    return ax


def gap_string(gap):
    if abs(gap) < .0001:
        return rf"$\Delta = {gap / 1e-5:.4f}\!\times\!10^{{-5}}$".replace("-", "−", 1)
    return f"Δ = {gap:+.4f}".replace("-", "−")


def chosen_actions(prediction):
    goal = prediction[:, 2].reshape(2, 2)
    return np.argmax(goal, axis=1)


def actual_success(target, prediction):
    choices = chosen_actions(prediction)
    return float(target[2 * np.arange(2) + choices, 2].mean())


def value_table(fig, matrices, loss_only):
    target, fb, density = matrices
    ax = plain_axes(fig, [.04, .275, .92, .242], (0, 15), (0, 2.85))
    ax.text(0, 2.69, "What each value function chooses", fontsize=16,
            weight="bold", color=INK)
    ax.text(15, 2.69, "Goal value = probability of reaching Goal", fontsize=11.5,
            ha="right", color=MUTED)
    centers = (6.0, 10.0)
    for context, x in enumerate(centers):
        ax.text(x, 2.31, f"Configuration {context + 1}", fontsize=12,
                color=INK, ha="center", weight="bold")
        ax.text(x, 2.06, "value(A)  /  value(B)", fontsize=10.5,
                color=MUTED, ha="center")
    ax.text(13.55, 2.31, "Actual success", ha="center", fontsize=12,
            color=INK, weight="bold")
    ax.text(13.55, 2.06, "under true transitions", fontsize=10.5,
            color=MUTED, ha="center")
    ax.plot([0, 15], [1.88, 1.88], color=EDGE, lw=1)
    rows = (
        (target, "True values", "Optimal action under the MDP", INK),
        (fb, "Optimal FB values", "Unique global squared-loss prediction", FB),
        (density, "Explicit density model", "Feasible construction; no optimum computed", DENSITY),
    )
    for index, (prediction, title, subtitle, color) in enumerate(rows):
        y = 1.53 - .64 * index
        rounded(ax, 0, y - .26, 15, .58,
                fill=("#F4F6F9", "#FBF0E9", "#EBF5F1")[index], edge="none", radius=.06)
        ax.text(.16, y + .11, title, fontsize=13.1, weight="bold", color=color, va="center")
        ax.text(.16, y - .12, subtitle, fontsize=9.7, color=MUTED, va="center")
        choices = chosen_actions(prediction)
        for context, x in enumerate(centers):
            a, b = prediction[2 * context:2 * context + 2, 2]
            precision = 7 if loss_only and index == 1 else (5 if index == 2 and not loss_only else 4)
            ax.text(x, y + .11, f"{a:.{precision}f}  /  {b:.{precision}f}",
                    fontsize=12.4, ha="center", va="center", color=INK)
            selected = "AB"[choices[context]]
            ax.text(x - .15, y - .14, gap_string(a - b), fontsize=10.8,
                    ha="right", va="center", color=color)
            ax.text(x + .01, y - .14, f"→ choose {selected}", fontsize=11.0,
                    ha="left", va="center", color=color, weight="bold")
        success = actual_success(target, prediction)
        success_precision = 1 if loss_only else 0
        ax.text(13.55, y + .05, f"{success:.{success_precision}%}", fontsize=20,
                color=color, ha="center", va="center", weight="bold")
        ax.text(13.55, y - .18, "average of the two starts", ha="center",
                color=MUTED, fontsize=8.9, va="center")
    return ax


def certificate(fig, feasible_kl, bounds, action, loss_only):
    ax = plain_axes(fig, [.04, .109, .92, .132], (0, 15), (0, 1.52))
    rounded(ax, 0, .015, 15, 1.495, fill="#F1F7F5", edge="#CADFD6", radius=.12)
    ax.text(.20, 1.18, f"Certificate: every global cross-entropy optimum chooses {action} in both configurations",
            color=DENSITY, fontsize=13.8, weight="bold")
    ax.text(.25, .72, "Feasible mean KL", fontsize=11.3, color=MUTED, va="center")
    ax.text(3.40, .72, f"{feasible_kl:.9f}", fontsize=15.5, color=DENSITY,
            va="center", ha="center", weight="bold")
    ax.text(4.94, .72, "<", fontsize=21, color=INK, ha="center", va="center")
    ax.text(6.52, .72, f"{min(bounds):.9f}", fontsize=15.5, color=INK,
            va="center", ha="center", weight="bold")
    ax.text(8.13, .72, "KL lower bound for any wrong or tied decision",
            fontsize=11.6, color=MUTED, va="center")
    ax.text(.25, .27,
            "Goal / non-Goal coarsening gives the bound; a global optimum cannot have more KL than a feasible model.",
            fontsize=10.9, color=MUTED, va="center")
    return ax


def make_figure(data, loss_only=False):
    if loss_only:
        target = np.asarray(data["P"])
        fb = np.asarray(data["Qls"])
        density = np.asarray(data["Qgood"])
        feasible_kl = data["KL_good"]
        bounds = data["wrong_order_KL_bounds"]
        name = "loss_only_counterexample"
        title = "The loss alone can reverse the chosen action"
        subtitle = "Shared stochastic rank-two probability class · Independent uniform reference · Unrestricted backward features"
        scope = "Reference matters: the natural marginal is (0.49, 0.49, 0.01, 0.01); using it restores the correct FB ordering."
    else:
        target = np.asarray(data["target_probabilities"])
        fb = np.asarray(data["least_squares_optimum"]["probabilities"])
        density = np.asarray(data["categorical_comparator"]["probabilities"])
        feasible_kl = data["categorical_comparator"]["mean_kl_regret"]
        bounds = data["wrong_or_tied_decision_kl_bounds"]
        name = "natural_reference_counterexample"
        title = "A squared-loss optimum can prefer the wrong action"
        subtitle = "Two-dimensional features · Natural uniform reference · Spherical backward features · Different probability classes"
        scope = "Model geometry also changes: squared loss on the softmax class succeeds too. This figure does not isolate cross-entropy."
    matrices = (target, fb, density)
    true_actions = chosen_actions(target)
    assert true_actions[0] == true_actions[1]
    assert np.all(chosen_actions(fb) != true_actions)
    assert np.all(chosen_actions(density) == true_actions)
    assert feasible_kl < min(bounds)
    fig = plt.figure(figsize=(16, 11.2), facecolor="white")
    fig.text(.04, .955, title, fontsize=25.5, color=INK, weight="bold")
    fig.text(.04, .919, subtitle, fontsize=12.1, color=MUTED)
    fig.text(.04, .880,
             "ONE-DECISION MDP", fontsize=10.5, color=DENSITY, weight="bold")
    fig.text(.206, .880,
             "Two equally likely starts · Two actions each · Four terminal outcomes · γ = 0 · Reward on arrival",
             fontsize=11.5, color=INK)
    for context in range(2):
        mdp_panel(fig, [.04 + .47 * context, .543, .45, .315], target, context)
    fig.text(.04, .525,
             "The same four terminal states appear in both panels. Training samples all four start–action pairs equally.",
             fontsize=10.1, color=MUTED)
    value_table(fig, matrices, loss_only)
    fig.text(.04, .253, "Δ = value(A) − value(B). Actions maximize the displayed value; success is always evaluated in the true MDP.",
             fontsize=10.5, color=MUTED)
    certificate(fig, feasible_kl, bounds, "AB"[true_actions[0]], loss_only)
    fig.text(.04, .081, scope, fontsize=11.3, color=INK)
    fig.text(.04, .051,
             "A two-feature FB model can preserve the exact action differences for every reward.",
             fontsize=11.3, color=INK, weight="bold")
    fig.text(.04, .023,
             "Finite-capacity population construction · Exact reward integration · No learned actor or reward-inference evaluation",
             fontsize=10.2, color=MUTED)
    for ext in ("png", "pdf", "svg"):
        path = HERE / f"{name}.{ext}"
        fig.savefig(path, dpi=190, facecolor="white")
        print(path)
    plt.close(fig)
    return {
        "figure": name,
        "goal_column": 2,
        "action_labels": {"A": 1, "B": -1},
        "goal_values": {key: value[:, 2].reshape(2, 2).tolist()
                        for key, value in zip(("truth", "FB_optimum", "explicit_density_model"), matrices)},
        "goal_gaps_A_minus_B": {key: (value[[0, 2], 2] - value[[1, 3], 2]).tolist()
                                for key, value in zip(("truth", "FB_optimum", "explicit_density_model"), matrices)},
        "true_success": {key: actual_success(target, value)
                         for key, value in zip(("truth", "FB_optimum", "explicit_density_model"), matrices)},
        "feasible_kl": feasible_kl,
        "any_wrong_or_tied_decision_kl_lower_bound": min(bounds),
        "per_context_wrong_or_tied_decision_kl_lower_bounds": bounds,
    }


def verified_output(filename):
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "didactic" / filename)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(process.stdout)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
    })
    objective = verified_output("verify_objective_counterexample.py")
    loss_only_records = verified_output("verify_loss_only_counterexample.py")
    loss_only = next(record for record in loss_only_records if record["name"] == "strict")
    rendered = [make_figure(objective), make_figure(loss_only, loss_only=True)]
    report = {
        "description": "Outputs from the existing verification scripts plus values shown in the figures.",
        "natural_reference_source": objective,
        "loss_only_strict_source": loss_only,
        "rendered": rendered,
    }
    (HERE / "verification_data.json").write_text(json.dumps(report, indent=2) + "\n")
    for record in rendered:
        print(json.dumps(record, indent=2))
    print("Both source verification scripts passed; action choices and KL certificates checked.")


if __name__ == "__main__":
    main()
