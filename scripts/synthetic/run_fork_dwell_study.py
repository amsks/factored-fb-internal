"""Spec 26 R0a study driver: fork-and-dwell grid + GATE P1 evaluation.

Grid: T_dwell {5,20,44,100} x duplication {1,2,5,10,20} x 5 arms x seeds.
Writes one JSON per (T_dwell, dup) cell plus a summary JSON with per-arm
slopes of success and action-gap ratio vs the duplication factor, and the
pre-registered GATE P1 read (normalized arms materially flatter in
duplication than unnormalized arms) plus the CE-vs-MSE secondary read.

Usage:
  .venv/bin/python scripts/synthetic/run_fork_dwell_study.py \
      --out results/spec26_r0a [--seeds 5] [--steps 5000] [--quick]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.synthetic.fork_dwell import (  # noqa: E402
    ALL_ARMS, NORMALIZED_ARMS, UNNORMALIZED_ARMS, run_cell,
)

T_DWELLS = [5, 20, 44, 100]
DUPS = [1, 2, 5, 10, 20]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    return float((xc @ (y - y.mean())) / (xc @ xc))


def summarize(cells: list[dict]) -> dict:
    """Per-(T_dwell, arm): mean metrics per dup + slopes vs dup factor."""
    summary = {"per_T": {}, "gate_P1": {}, "ce_vs_mse": {}}
    for T in sorted({c["T_dwell"] for c in cells}):
        per_arm = {}
        for arm in ALL_ARMS:
            rows = {}
            for dup in sorted({c["dup"] for c in cells if c["T_dwell"] == T}):
                ms = [c["arms"][arm] for c in cells
                      if c["T_dwell"] == T and c["dup"] == dup]
                rows[dup] = {
                    k: float(np.mean([m[k] for m in ms]))
                    for k in ("success", "action_gap", "action_gap_ratio",
                              "critic_scale_mean_abs", "mass_ratio",
                              "entropy_mean", "fixed_point_error", "gt_l1")
                }
                rows[dup]["action_gap_true"] = float(
                    np.mean([m["action_gap_true"] for m in ms]))
            dups = sorted(rows)
            per_arm[arm] = {
                "by_dup": {str(d): rows[d] for d in dups},
                "slope_success_vs_dup": _slope(dups, [rows[d]["success"] for d in dups]),
                "slope_gap_ratio_vs_dup": _slope(dups, [rows[d]["action_gap_ratio"] for d in dups]),
                "slope_gap_vs_dup": _slope(dups, [rows[d]["action_gap"] for d in dups]),
                "slope_scale_vs_dup": _slope(dups, [rows[d]["critic_scale_mean_abs"] for d in dups]),
            }
        summary["per_T"][str(T)] = per_arm

    # GATE P1: normalized arms flatter in dup than unnormalized (success + gap)
    def group_absslopes(arms, key):
        return {T: {a: abs(summary["per_T"][T][a][key]) for a in arms}
                for T in summary["per_T"]}

    gate = {}
    for key, label in (("slope_success_vs_dup", "success"),
                       ("slope_gap_ratio_vs_dup", "gap_ratio")):
        norm = group_absslopes(NORMALIZED_ARMS, key)
        unnorm = group_absslopes(UNNORMALIZED_ARMS, key)
        gate[label] = {
            "normalized_abs_slopes": norm,
            "unnormalized_abs_slopes": unnorm,
            "mean_abs_slope_normalized": float(np.mean(
                [v for T in norm for v in norm[T].values()])),
            "mean_abs_slope_unnormalized": float(np.mean(
                [v for T in unnorm for v in unnorm[T].values()])),
        }
    summary["gate_P1"] = gate
    summary["ce_vs_mse"] = {
        T: {m: summary["per_T"][T][a]
            for m, a in (("ce", "density_ce"), ("mse", "density_mse"))}
        for T in summary["per_T"]
    }
    return summary


def _run_one(T, dup, seed, kwargs):
    return run_cell(T, dup, seed, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/spec26_r0a")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes over (T, dup, seed) cells")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke grid (2 cells, 1 seed, short training)")
    args = ap.parse_args()

    t_dwells, dups, seeds, steps, episodes = (
        T_DWELLS, DUPS, args.seeds, args.steps, args.episodes)
    if args.quick:
        t_dwells, dups, seeds, steps, episodes = [5], [1, 20], 1, 600, 50

    os.makedirs(args.out, exist_ok=True)
    jobs = [(T, dup, seed, dict(steps=steps, n_episodes=episodes))
            for T in t_dwells for dup in dups for seed in range(seeds)]
    t0 = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            cells = pool.starmap(_run_one, jobs)
    else:
        cells = [_run_one(*j) for j in jobs]
    for T in t_dwells:
        for dup in dups:
            cell_seeds = [c for c in cells
                          if c["T_dwell"] == T and c["dup"] == dup]
            path = os.path.join(args.out, f"cell_T{T}_dup{dup}.json")
            with open(path, "w") as f:
                json.dump({"T_dwell": T, "dup": dup, "seeds": cell_seeds}, f, indent=1)
            done = [c["arms"]["fb_unnorm"]["success"] for c in cell_seeds]
            print(f"[{time.time()-t0:7.1f}s] T={T:>3} dup={dup:>2} done "
                  f"(fb_unnorm success {np.mean(done):.2f})", flush=True)

    summary = summarize(cells)
    summary["config"] = {"t_dwells": t_dwells, "dups": dups, "seeds": seeds,
                         "steps": steps, "episodes": episodes}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== GATE P1 ===")
    for label in ("success", "gap_ratio"):
        g = summary["gate_P1"][label]
        print(f"{label}: mean|slope| normalized={g['mean_abs_slope_normalized']:.4f} "
              f"unnormalized={g['mean_abs_slope_unnormalized']:.4f}")
    print(f"summary -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
