#!/usr/bin/env python3
"""Render the deterministic path and its exact-successor value comparison."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deterministic-path-mpl")

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
LEFT_COLOR = "#61768A"
GAMMA = 4 / 5


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
    export(fig, "mdp")


def verification_data():
    destinations = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    immediate = np.eye(4)[destinations.reshape(-1)]
    weights = np.full(8, 1 / 8)
    rho = immediate.T @ weights
    continuation = immediate.reshape(4, 2, 4).mean(axis=1)
    target = (1 - GAMMA) * np.linalg.inv(np.eye(4) - GAMMA * continuation)
    np.testing.assert_allclose(target, np.array(
        [[43, 22, 12, 8], [22, 33, 18, 12], [12, 18, 33, 22], [8, 12, 22, 43]]
    ) / 85, atol=2e-15, rtol=0)
    np.testing.assert_allclose(rho, .25, atol=1e-15, rtol=0)

    # These are the unique two leading eigendirections of the exact target.
    _, basis = np.linalg.eigh(target)
    projector = basis[:, -2:] @ basis[:, -2:].T
    fb = target @ projector
    weighted = np.sqrt(weights[:, None]) * target[destinations.reshape(-1)] / np.sqrt(rho)
    u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    full_svd = ((u[:, :2] * singular[:2]) @ vh[:2]) * np.sqrt(rho) / np.sqrt(weights[:, None])
    np.testing.assert_allclose(full_svd, fb[destinations.reshape(-1)], atol=2e-15, rtol=0)
    assert singular[1] > singular[2]
    assert fb.min() > 0
    np.testing.assert_allclose(fb.sum(axis=1), 1, atol=2e-15, rtol=0)

    # Rational, explicitly feasible rank-two logit factors; no optimizer is run.
    backward = np.array([[-1, 1], [-1 / 2, -1], [1 / 2, -1], [1, 1]])
    forward = np.array([[-4 / 5, 1 / 10], [-2 / 5, -23 / 100],
                        [2 / 5, -23 / 100], [4 / 5, 1 / 10]])
    logits = forward @ backward.T
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    density = np.exp(log_prob)
    candidate_kl = np.mean(np.sum(target * (np.log(target) - log_prob), axis=1))

    def bernoulli_kl(p, q):
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    # Duplicate exact-target rows have identical predictions at every global CE
    # optimum, so their two 1/8 weights combine to 1/4. See README.md.
    wrong_bounds = {}
    for state in (0, 2):
        pair = target[destinations[state], 1]
        midpoint = pair.mean()
        wrong_bounds[state] = float(sum(bernoulli_kl(p, midpoint) for p in pair) / 4)
    assert candidate_kl < min(wrong_bounds.values())
    assert np.linalg.matrix_rank(target) == 4
    assert np.linalg.matrix_rank(np.log(target) - np.log(target).mean(axis=1, keepdims=True)) == 3
    control = np.repeat(((1 - target[:, 1]) / 3)[:, None], 4, axis=1)
    control[:, 1] = target[:, 1]
    assert np.linalg.matrix_rank(control) == 2
    np.testing.assert_allclose(control.sum(axis=1), 1, atol=2e-15, rtol=0)
    assert control.min() > 0
    assert fb[1, 1] < fb[0, 1]
    assert density[1, 1] > density[0, 1]
    assert density[1, 1] > density[3, 1]
    canonical_b = basis[:, -2:] / np.sqrt(rho[:, None])

    return {
        "gamma": GAMMA, "dimension": 2, "goal_state": 1, "start_state": 0,
        "action_order": ["Left", "Right"], "destinations": destinations.tolist(),
        "state_action_weights": weights.tolist(), "reference": rho.tolist(),
        "continuation_transition": continuation.tolist(), "exact_normalized_successor": target.tolist(),
        "fb_exact_target_squared_optimum": fb.tolist(),
        "density_feasible_forward": forward.tolist(), "density_feasible_backward": backward.tolist(),
        "density_feasible_probability": density.tolist(), "density_feasible_kl": float(candidate_kl),
        "wrong_order_kl_bounds_collapsed_rows": wrong_bounds,
        "control_sufficient_rank_two_probability": control.tolist(),
        "fb_backward_norms": np.linalg.norm(canonical_b, axis=1).tolist(),
        "weighted_singular_values": singular.tolist(),
        "start_normalized_values_left_right": {
            "exact": target[:2, 1].tolist(), "fb": fb[:2, 1].tolist(),
            "density_feasible": density[:2, 1].tolist(),
        },
        "optimal_start_return": 1 / (1 - GAMMA ** 2),
        "fb_greedy_start_return": 0,
        "scope": "Exact-successor approximation; unrestricted backward norms; no global TD optimality claim.",
    }


def draw_values(data):
    fig, ax = plt.subplots(figsize=(10.4, 5.4), facecolor="white")
    fig.subplots_adjust(left=.1, right=.985, bottom=.22, top=.77)
    values = np.array(list(data["start_normalized_values_left_right"].values()))
    positions = np.arange(3)
    for action, color, offset, label in (
        (0, LEFT_COLOR, -.17, "Left (stay)"), (1, GREEN, .17, "Right (enter Goal)"),
    ):
        ax.bar(positions + offset, values[:, action], width=.28, color=color,
               label=label, zorder=3)
        for index, value in enumerate(values[:, action]):
            ax.text(index + offset, value + .009, f"{value:.4f}", ha="center",
                    va="bottom", fontsize=14, color=INK)
    ax.set_ylim(0, .45)
    ax.set_yticks(np.arange(0, .41, .1))
    ax.set_ylabel("Normalized goal occupancy", fontsize=14, color=INK, labelpad=12)
    ax.set_xticks(positions, ["Exact successor", "FB: best squared fit", "Density: feasible fit"])
    ax.tick_params(axis="x", labelsize=14, length=0, pad=12, colors=INK)
    ax.tick_params(axis="y", labelsize=12, length=0, colors=MUTED)
    ax.grid(axis="y", color="#E2E7EB", linewidth=.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCD5DC")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2,
              frameon=False, fontsize=14, handlelength=1.3, columnspacing=2.1)
    for index, pair in enumerate(values):
        winner = "Right" if pair[1] > pair[0] else "Left"
        ax.text(index, -.165, f"Chooses {winner}", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=13,
                weight="bold", color=GREEN if winner == "Right" else INK)
    fig.text(.5, .947, "Action values at Start", ha="center", va="center",
             fontsize=23, weight="bold", color=INK)
    fig.text(.5, .045,
             r"Exact-successor approximation: $\gamma=0.8$, $d=2$; density shows an explicit feasible model.",
             ha="center", va="center", fontsize=12, color=MUTED)
    export(fig, "values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    draw_mdp()
    if not args.graph_only:
        data = verification_data()
        draw_values(data)
        (HERE / "verification_data.json").write_text(json.dumps(data, indent=2) + "\n")
        print("Verified transitions, natural reference, squared optimum, feasible density, and KL certificate.")


if __name__ == "__main__":
    main()
