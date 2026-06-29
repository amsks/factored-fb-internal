#!/usr/bin/env python
"""scripts/plot_funnel_comparison.py — side-by-side S0 phase funnel across methods.

Reads each method's aggregate_per_episode.parquet (produced by
phase_probe_aggregate.py) and draws the manipulation funnel
Reached -> Secured -> Success as one descending line per method. The
FB-family lines collapse at the Secured->Success step (the post-control-
acquisition / transport failure); GCIQL-bilinear (G3) stays flat — it
delivers what it grasps.

Per-method mean is over episodes; the error band is +/- 1 std ACROSS SEEDS
(each seed's own stage rate), so it reflects seed variability, not
within-seed episode noise.

Run: /dev/shm/.venv-jax/bin/python scripts/plot_funnel_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

# (label, aggregate_per_episode.parquet, color). Order: failing methods first.
METHODS = [
    ("RLDP",          "analysis/profiles/rldp_phase/aggregate/aggregate_per_episode.parquet",      "#d62728"),
    ("PSM",           "analysis/profiles/psm_phase/aggregate/aggregate_per_episode.parquet",       "#ff7f0e"),
    ("PSM-o1000",     "analysis/profiles/psm_o1000_phase/aggregate/aggregate_per_episode.parquet", "#9467bd"),
    ("GCIQL-bilinear (G3)", "analysis/phase_probe_gciql_bilinear/aggregate/aggregate_per_episode.parquet", "#2ca02c"),
]
STAGES = ["reached", "secured", "success"]
STAGE_LABELS = ["Reached\n(approach)", "Secured\n(grasp+lift)", "Success\n(transport)"]
SCENARIO = "S0"


def stage_stats(parquet: Path):
    """Return (mean[3], std_across_seeds[3]) for the S0 funnel stages."""
    df = pd.read_parquet(parquet)
    df = df[df["scenario"] == SCENARIO]
    # per-seed stage rate, then mean/std across seeds
    per_seed = df.groupby("seed")[STAGES].mean()
    return per_seed.mean().to_numpy() * 100.0, per_seed.std(ddof=0).to_numpy() * 100.0, len(df)


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(STAGES))
    for label, rel, color in METHODS:
        p = REPO / rel
        if not p.exists():
            print(f"[warn] missing {p}, skipping {label}")
            continue
        mean, std, n = stage_stats(p)
        ax.plot(x, mean, "-o", color=color, lw=2.4, ms=8,
                label=f"{label}  ({mean[2]:.0f}% succ)")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        print(f"{label:24s} reached={mean[0]:5.1f} secured={mean[1]:5.1f} "
              f"success={mean[2]:5.1f}  (n={n})")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylabel("% of episodes reaching stage")
    ax.set_ylim(0, 103)
    ax.set_title("Cube-single manipulation funnel (S0 baseline)\n"
                 "FB-family collapses at transport; bilinear GCIQL does not")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()

    outs = [REPO / "analysis/phase_probe_gciql_bilinear/aggregate/funnel_comparison.png",
            REPO / "PAPER/rlbrew/figures/fig_funnel_comparison.png"]
    for o in outs:
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=160)
        print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
