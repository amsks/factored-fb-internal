#!/usr/bin/env python
"""scripts/plot_rlbrew_figures.py — generate the cross-method figures for
PAPER/rlbrew/rlbrew.tex.

Emits into PAPER/rlbrew/figures/:
  f1_composition.png        — 4-method 5-bucket outcome stacked bars (Tab. cross_method)
  f1_funnel.png             — 4-method reach -> secure -> success funnel
  f3_value_vs_dist.png      — value vs cube-to-goal distance by phase (copied)

The outcome composition and funnel are drawn from PAPER-CANONICAL numbers
(`OUTCOMES`, matching rlbrew.tex tab:cross_method_template and
tab:fb_state_outcomes, 10 seeds) so the figures and tables never disagree. The
funnel rates are derived from the same distribution: with mutually exclusive
first-failure labels, reached = 100 - approach, secured = reached - grasp, and
success is the success share.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

# Paper-canonical S0 outcome distribution (% episodes, mean across 10 seeds).
# Must match rlbrew.tex tab:cross_method_template.
OUTCOMES = {
    "FB":    {"success": 49.4, "approach": 0.0, "grasp": 9.8,  "maintain": 23.8, "transport": 17.0},
    "GCIQL": {"success": 69.2, "approach": 0.2, "grasp": 12.2, "maintain": 9.2,  "transport": 9.2},
    "CRL":   {"success": 23.2, "approach": 0.4, "grasp": 23.8, "maintain": 28.6, "transport": 24.0},
    "RLDP":  {"success": 25.4, "approach": 0.2, "grasp": 9.6,  "maintain": 44.0, "transport": 20.8},
}
METHODS = ["FB", "GCIQL", "CRL", "RLDP"]
METHOD_COLORS = {"FB": "#d62728", "GCIQL": "#1f77b4", "CRL": "#2ca02c",
                 "RLDP": "#9467bd"}
OUTCOME_COLORS = {
    "success":   "#2ca02c",
    "approach":  "#9467bd",
    "grasp":     "#ff7f0e",
    "maintain":  "#d62728",
    "transport": "#1f77b4",
}


def plot_composition(out: Path) -> None:
    """5-bucket outcome stacked bar, one column per method."""
    buckets = ["success", "approach", "grasp", "maintain", "transport"]
    labels = ["Success", "Approach fail", "Grasp fail",
              "Maintain fail", "Transport fail"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(METHODS))
    bottom = np.zeros(len(METHODS))
    for b, lab in zip(buckets, labels):
        h = np.array([OUTCOMES[m][b] for m in METHODS])
        ax.bar(x, h, bottom=bottom, color=OUTCOME_COLORS[b], label=lab,
               edgecolor="black", linewidth=0.4)
        for xi, (hi, bi) in enumerate(zip(h, bottom)):
            if hi >= 3:
                ax.text(xi, bi + hi / 2, f"{hi:.1f}", ha="center", va="center",
                        fontsize=8.5, color="white" if hi >= 8 else "black",
                        fontweight="bold")
        bottom += h
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=11, fontweight="bold")
    ax.set_ylabel("percent of episodes (S0, mean across seeds)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Outcome composition: where each URL method spends its failures",
                 fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.0), fontsize=8.5,
              framealpha=0.95)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_funnel(out: Path) -> None:
    """reach -> secure -> success funnel, grouped by method.

    Derived from the mutually-exclusive first-failure distribution:
      reached = 100 - approach;  secured = reached - grasp;  success.
    """
    stages = ["reached", "secured", "success"]
    rates = {}
    for m in METHODS:
        o = OUTCOMES[m]
        reached = 100 - o["approach"]
        secured = reached - o["grasp"]
        rates[m] = [reached, secured, o["success"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(stages))
    width = 0.2
    for i, m in enumerate(METHODS):
        off = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(x + off, rates[m], width, color=METHOD_COLORS[m], label=m,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Reached", "Secured", "Success"], fontsize=10)
    ax.set_ylabel("percent of episodes (S0)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="black", linewidth=0.4, linestyle=":")
    ax.set_title("Phase funnel: methods reach and secure, then collapse at transport",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def copy_existing() -> None:
    """Copy the value-vs-distance figure into figures/."""
    pairs = [
        (ANALYSIS / "training_value_multiseed/aggregate_rldp/value_vs_dist.png",
         OUT / "f3_value_vs_dist.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[plot] copied {src.name} -> {dst.name}")
        else:
            print(f"[plot] WARN missing {src}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_composition(OUT / "f1_composition.png")
    plot_funnel(OUT / "f1_funnel.png")
    copy_existing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
