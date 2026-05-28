#!/usr/bin/env python
"""scripts/plot_rlbrew_figures.py — generate 3-way comparison figures for
PAPER/rlbrew/rlbrew.tex.

Emits 4 figures into PAPER/rlbrew/figures/:
  F1: curves_aggregate.png + curves_pertask.png   (copied from fb_gciql_crl)
  F2: funnel_3way.png        — per-task reach/secure/success rates per method
  F3: failure_composition.png — 5-bucket stacked-bar per method (Table 7 vis)
  F4: intervention_3way.png   — S0/S1/S2 reach/secure/success for FB and CRL

Reads each method's native data format:
  - FB:    analysis/phase_probe/aggregate/aggregate_per_episode.parquet
  - GCIQL: analysis/gciql_profile/aggregate/funnel.parquet (no 5-bucket)
  - CRL:   analysis/phase_probe_crl/aggregate/aggregate_per_episode.parquet

For F3, GCIQL's 5-bucket numbers come from rlbrew.tex Table 7 (paper-canonical)
since the GCIQL aggregator doesn't compute final_cube_lift.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

# Paper-canonical numbers for the FB and GCIQL methods (from rlbrew.tex
# Table 7). CRL is computed from data. The figures must match the table
# the reader cross-references against.
TABLE7 = {
    "FB":    {"success": 51.1, "approach": 0.0, "grasp": 7.7,  "maintain": 23.7, "transport": 17.4},
    "GCIQL": {"success": 69.2, "approach": 0.2, "grasp": 12.2, "maintain": 9.2,  "transport": 9.2},
    # CRL filled in from data below
}
METHOD_COLORS = {"FB": "#1f77b4", "GCIQL": "#2ca02c", "CRL": "#d62728"}
OUTCOME_COLORS = {
    "success":   "#2ca02c",
    "approach":  "#9467bd",
    "grasp":     "#ff7f0e",
    "maintain":  "#d62728",
    "transport": "#1f77b4",
}
TASK_LABELS = ["task1", "task2", "task3", "task4", "task5"]


def _short_task(t: str) -> str:
    if "singletask-task" in t:
        return "task" + t.split("task")[-1].rstrip("-v0")[0]
    return t


def _iqm(rates: np.ndarray) -> float:
    """Interquartile mean of a 1-D array (trim top+bottom 25%)."""
    rates = np.sort(np.asarray(rates, dtype=float))
    lo, hi = int(np.ceil(0.25 * len(rates))), int(np.floor(0.75 * len(rates)))
    return float(rates[lo:hi + 1].mean()) if hi >= lo else float(rates.mean())


def _iqm_per_task(per_seed_task: pd.DataFrame, value_col: str) -> pd.Series:
    """Per-task IQM across seeds of a per-(seed, task) rate frame."""
    return per_seed_task.groupby("task_short")[value_col].apply(
        lambda x: _iqm(x.values))


def funnel_rates_fb_or_crl(parquet: Path, scenario: str = "S0") -> pd.DataFrame:
    """Per-task IQM-across-seeds of reach/secure/success rates."""
    df = pd.read_parquet(parquet)
    df = df[df["scenario"] == scenario].copy()
    df["task_short"] = df["task"].map(_short_task)
    per_seed_task = (df.groupby(["seed", "task_short"])
                       .agg(reach=("reached", "mean"),
                            secure=("secured", "mean"),
                            success=("success", "mean"))
                       .reset_index())
    out = pd.DataFrame({
        "task_short": sorted(per_seed_task["task_short"].unique()),
    })
    out["reach"]   = out["task_short"].map(_iqm_per_task(per_seed_task, "reach"))   * 100
    out["secure"]  = out["task_short"].map(_iqm_per_task(per_seed_task, "secure"))  * 100
    out["success"] = out["task_short"].map(_iqm_per_task(per_seed_task, "success")) * 100
    return out


def funnel_rates_gciql() -> pd.DataFrame:
    """Per-task IQM-across-seeds of reach/secure/success rates (GCIQL has its
    own schema: encodes furthest_phase rather than booleans)."""
    df = pd.read_parquet(ANALYSIS / "gciql_profile/aggregate/funnel.parquet")
    df = df.copy()
    df["reach"]    = df["furthest_phase"].isin({"grasp", "transport", "success"})
    df["secure"]   = df["furthest_phase"].isin({"transport", "success"})
    df["success_"] = df["furthest_phase"].eq("success")
    df["task_short"] = df["task"]
    per_seed_task = (df.groupby(["seed", "task_short"])
                       .agg(reach=("reach", "mean"),
                            secure=("secure", "mean"),
                            success=("success_", "mean"))
                       .reset_index())
    out = pd.DataFrame({
        "task_short": sorted(per_seed_task["task_short"].unique()),
    })
    out["reach"]   = out["task_short"].map(_iqm_per_task(per_seed_task, "reach"))   * 100
    out["secure"]  = out["task_short"].map(_iqm_per_task(per_seed_task, "secure"))  * 100
    out["success"] = out["task_short"].map(_iqm_per_task(per_seed_task, "success")) * 100
    return out


def crl_5bucket_from_parquet() -> dict[str, float]:
    """Compute CRL's 5-bucket outcome distribution from its per_episode parquet."""
    df = pd.read_parquet(ANALYSIS / "phase_probe_crl/aggregate/aggregate_per_episode.parquet")
    df = df[df["scenario"] == "S0"].copy()
    DELTA_LIFT = 0.03

    def bucket(r):
        if r["success"]:
            return "success"
        if r["fail_phase"] == "reach":
            return "approach"
        if r["fail_phase"] == "grasp":
            return "grasp"
        if r["final_cube_lift"] < DELTA_LIFT:
            return "maintain"
        return "transport"

    df["bucket"] = df.apply(bucket, axis=1)
    pct = (df["bucket"].value_counts(normalize=True) * 100).to_dict()
    return {k: pct.get(k, 0.0) for k in ["success", "approach", "grasp",
                                          "maintain", "transport"]}


def intervention_rates(parquet: Path) -> pd.DataFrame:
    """Per-scenario aggregate reach/secure/success rates (M2 study)."""
    df = pd.read_parquet(parquet)
    grouped = df.groupby("scenario").agg(
        reach=("reached", "mean"),
        secure=("secured", "mean"),
        success=("success", "mean"),
    )
    return (grouped * 100).reindex(["S0", "S1", "S2"])


# ---------------------------------------------------------------------------
# Plot makers
# ---------------------------------------------------------------------------


def plot_funnel(fb: pd.DataFrame, gciql: pd.DataFrame, crl: pd.DataFrame,
                out: Path, include_crl: bool = True) -> None:
    """Per-(method, task) plot reach/secure/success as a 3-segment bar."""
    if include_crl:
        method_data = [("FB", fb), ("GCIQL", gciql), ("CRL", crl)]
        figsize = (11, 3.4)
    else:
        method_data = [("FB", fb), ("GCIQL", gciql)]
        figsize = (8, 3.4)
    fig, axes = plt.subplots(1, len(method_data), figsize=figsize, sharey=True)
    if len(method_data) == 1:
        axes = [axes]
    x = np.arange(len(TASK_LABELS))
    width = 0.25

    for ax, (name, data) in zip(axes, method_data):
        data = data.set_index("task_short").reindex(TASK_LABELS)
        r = data["reach"].values
        s = data["secure"].values
        ok = data["success"].values
        ax.bar(x - width, r, width, color="#cccccc", label="Reached", edgecolor="black", linewidth=0.4)
        ax.bar(x, s, width, color="#888888", label="Secured", edgecolor="black", linewidth=0.4)
        ax.bar(x + width, ok, width, color=METHOD_COLORS[name],
               label="Success", edgecolor="black", linewidth=0.4)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_LABELS, fontsize=8)
        ax.set_ylim(0, 105)
        ax.axhline(100, color="black", linewidth=0.4, linestyle=":")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    axes[0].set_ylabel("% episodes", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Phase funnel: reach → secure → success by method and task",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_composition(crl_5bucket: dict[str, float], out: Path) -> None:
    """Stacked bar of 5-bucket outcomes (success / approach / grasp / maintain
    / transport), one column per method. Visual of rlbrew Table 7."""
    methods = ["FB", "GCIQL", "CRL"]
    table = dict(TABLE7)
    table["CRL"] = crl_5bucket
    buckets = ["success", "approach", "grasp", "maintain", "transport"]
    labels = ["Success", "Approach fail", "Grasp fail",
              "Maintain fail", "Transport fail"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for b, lab in zip(buckets, labels):
        heights = [table[m][b] for m in methods]
        ax.bar(x, heights, bottom=bottom, color=OUTCOME_COLORS[b],
               label=lab, edgecolor="black", linewidth=0.4)
        # annotate non-trivial segments
        for xi, (h, bot) in enumerate(zip(heights, bottom)):
            if h >= 3:
                ax.text(xi, bot + h / 2, f"{h:.1f}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if h >= 8 else "black", fontweight="bold")
        bottom += np.asarray(heights)

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11, fontweight="bold")
    ax.set_ylabel("% episodes (S0, mean across seeds)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Outcome composition: where each URL method spends its failures",
                 fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.0),
              fontsize=8.5, framealpha=0.95)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_intervention(fb_int: pd.DataFrame, crl_int: pd.DataFrame,
                       out: Path) -> None:
    """M2 intervention: S0/S1/S2 reach/secure/success for FB and CRL."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    scen = ["S0", "S1", "S2"]
    scen_lab = ["S0 baseline", "S1 control", "S2 pre-grasped"]
    x = np.arange(len(scen))
    width = 0.25

    for ax, (name, data) in zip(axes, [("FB", fb_int), ("CRL", crl_int)]):
        data = data.reindex(scen)
        r = data["reach"].values
        s = data["secure"].values
        ok = data["success"].values
        ax.bar(x - width, r, width, color="#cccccc", label="Reached", edgecolor="black", linewidth=0.4)
        ax.bar(x, s, width, color="#888888", label="Secured", edgecolor="black", linewidth=0.4)
        ax.bar(x + width, ok, width, color=METHOD_COLORS[name],
               label="Success", edgecolor="black", linewidth=0.4)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(scen_lab, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        for xi, (h, lab) in enumerate(zip(ok, scen_lab)):
            ax.text(xi + width, h + 2, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8.5)

    axes[0].set_ylabel("% episodes", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Pre-grasp intervention (M2): S2 hurts both methods",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def copy_existing_curves() -> None:
    """Copy 2-way (FB+GCIQL) curves for §6 and 3-way (+CRL) curves for §7."""
    pairs = [
        (ANALYSIS / "fb_gciql/curve_aggregate.png",
         OUT / "f1_curves_aggregate.png"),
        (ANALYSIS / "fb_gciql/curves_pertask.png",
         OUT / "f1_curves_pertask.png"),
        (ANALYSIS / "fb_gciql_crl/curve_aggregate.png",
         OUT / "f3a_curves_aggregate_3way.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[plot] copied {src.name} -> {dst.name}")
        else:
            print(f"[plot] WARN missing {src}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # F1: existing IQM training curves
    copy_existing_curves()

    # Compute per-method funnel rates (S0)
    fb = funnel_rates_fb_or_crl(
        ANALYSIS / "phase_probe/aggregate/aggregate_per_episode.parquet")
    crl = funnel_rates_fb_or_crl(
        ANALYSIS / "phase_probe_crl/aggregate/aggregate_per_episode.parquet")
    gciql = funnel_rates_gciql()
    plot_funnel(fb, gciql, crl, OUT / "f2_funnel_fbgciql.png", include_crl=False)
    plot_funnel(fb, gciql, crl, OUT / "f2b_funnel_3way.png", include_crl=True)

    # F3: outcome composition (5-bucket)
    crl_5 = crl_5bucket_from_parquet()
    plot_composition(crl_5, OUT / "f3_failure_composition.png")

    # F4: intervention study (FB + CRL only — GCIQL didn't run M2)
    fb_int  = intervention_rates(
        ANALYSIS / "phase_probe/aggregate/aggregate_per_episode.parquet")
    crl_int = intervention_rates(
        ANALYSIS / "phase_probe_crl/aggregate/aggregate_per_episode.parquet")
    plot_intervention(fb_int, crl_int, OUT / "f4_intervention.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
