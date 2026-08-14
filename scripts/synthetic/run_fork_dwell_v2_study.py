"""Spec 26 R0a-v2 study driver: weighted fork-and-dwell grid + GATE P1-v2.

Grid: column {d2, d4, feat_full, tab} x T_dwell {5,20,44,100} x weight factor
{1,2,5,10,20} x 5 arms x seeds.  One JSON per (column, T, weight) plus a
summary carrying, per column and per logged step (1k/3k/10k/30k), the slope of
each metric in log2(weight factor) for the normalized and unnormalized arm
groups, and the pre-registered GATE P1-v2 read.

GATE P1-v2 (pre-registered, both halves required):
  A. Under constraint (columns d2/d4, fixed 10k budget) the normalized arms are
     materially flatter in the weight factor than the unnormalized arms:
     |slope|_norm <= 0.5 * |slope|_unnorm, with the unnormalized slope itself
     materially non-zero (its 95% CI excludes 0).
  B. In the converged tabular control column the separation VANISHES: the
     control's |slope|_unnorm - |slope|_norm gap is <= 0.25x the constrained
     columns' gap, and neither group's slope in the control differs from 0.
Primary metric: `margin_ratio` (the aliasing-hardened decision score).
Secondary: `success`, `action_gap_ratio`.  Slopes are taken in log2(weight)
because the factors {1,2,5,10,20} are geometric; raw-factor slopes are also
reported.  Units for the CIs are the (T_dwell, seed) pairs.

Usage:
  .venv/bin/python scripts/synthetic/run_fork_dwell_v2_study.py \
      --out results/spec26_r0a_v2 [--seeds 3] [--jobs 48] [--quick]
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

from scripts.synthetic.fork_dwell_v2 import (  # noqa: E402
    ALL_ARMS, CHECKPOINTS, COLUMNS, CONSTRAINED_COLUMNS, CONTROL_COLUMN,
    NORMALIZED_ARMS, T_DWELLS, UNNORMALIZED_ARMS, WEIGHTS, run_cell,
)

METRICS = ("margin_ratio", "success", "action_gap_ratio", "critic_scale_mean_abs",
           "mass_ratio", "entropy_mean", "gt_l1", "fixed_point_error",
           "alias_confusion")
PRIMARY_METRIC = "margin_ratio"
GATE_METRICS = ("margin_ratio", "success", "action_gap_ratio")
FLAT_RATIO = 0.5      # half A: normalized at most this fraction of unnormalized
CONTROL_RATIO = 0.25  # half B: control gap at most this fraction of the constrained gap


def _slope(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)) or np.allclose(x, x[0]):
        return float("nan")
    xc = x - x.mean()
    return float((xc @ (y - y.mean())) / (xc @ xc))


def _mean_ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size < 2:
        return mean, float("nan"), float("nan")
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    # 95% t-CI; t table for the small sample sizes we use
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}.get(
        v.size - 1, 1.96 + 2.0 / max(v.size - 1, 1))
    return mean, mean - tcrit * se, mean + tcrit * se


def _cell_index(cells):
    """(column, T, weight, seed) -> cell."""
    return {(c["column"], c["T_dwell"], c["weight"], c["seed"]): c for c in cells}


def summarize(cells: list[dict]) -> dict:
    idx = _cell_index(cells)
    columns = sorted({c["column"] for c in cells}, key=list(COLUMNS).index)
    t_dwells = sorted({c["T_dwell"] for c in cells})
    weights = sorted({c["weight"] for c in cells})
    seeds = sorted({c["seed"] for c in cells})

    out = {"per_column": {}, "gate_P1_v2": {}, "ce_vs_mse": {}}
    for col in columns:
        steps = sorted({int(s) for c in cells if c["column"] == col
                        for s in c["arms"][ALL_ARMS[0]]["by_step"]})
        col_out = {"steps": steps, "by_step": {}}
        for step in steps:
            def read(arm, T, wt, seed):
                c = idx.get((col, T, wt, seed))
                if c is None:
                    return None
                return c["arms"][arm]["by_step"].get(str(step))

            per_arm = {}
            for arm in ALL_ARMS:
                # level means per weight factor
                levels = {}
                for wt in weights:
                    ms = [read(arm, T, wt, sd) for T in t_dwells for sd in seeds]
                    ms = [m for m in ms if m is not None]
                    levels[str(wt)] = {k: float(np.nanmean([m[k] for m in ms]))
                                       for k in METRICS}
                # per-(T, seed) slopes in log2(weight) and in raw weight
                slopes_log, slopes_raw = {k: [] for k in METRICS}, {k: [] for k in METRICS}
                for T in t_dwells:
                    for sd in seeds:
                        ms = [read(arm, T, wt, sd) for wt in weights]
                        if any(m is None for m in ms):
                            continue
                        for k in METRICS:
                            ys = [m[k] for m in ms]
                            slopes_log[k].append(_slope(np.log2(weights), ys))
                            slopes_raw[k].append(_slope(weights, ys))
                per_arm[arm] = {
                    "levels": levels,
                    "slope_log2w": {k: dict(zip(("mean", "lo", "hi"),
                                                _mean_ci(slopes_log[k])))
                                    for k in METRICS},
                    "slope_raw_w": {k: dict(zip(("mean", "lo", "hi"),
                                                _mean_ci(slopes_raw[k])))
                                    for k in METRICS},
                    "_slopes_log": {k: slopes_log[k] for k in GATE_METRICS},
                }

            groups = {}
            for gname, arms in (("normalized", NORMALIZED_ARMS),
                                ("unnormalized", UNNORMALIZED_ARMS)):
                g = {}
                for k in GATE_METRICS:
                    # unit = (T, seed): average the arms in the group, then CI
                    stacked = np.array([per_arm[a]["_slopes_log"][k] for a in arms],
                                       dtype=float)
                    unit = np.nanmean(stacked, axis=0) if stacked.size else np.array([])
                    mean, lo, hi = _mean_ci(unit)
                    g[k] = {"mean": mean, "lo": lo, "hi": hi, "abs_mean": abs(mean),
                            "n_units": int(np.sum(np.isfinite(unit)))}
                groups[gname] = g
            for arm in ALL_ARMS:
                per_arm[arm].pop("_slopes_log")
            col_out["by_step"][str(step)] = {"per_arm": per_arm, "groups": groups}
        out["per_column"][col] = col_out

    # ---- GATE P1-v2 -------------------------------------------------------
    def final_groups(col):
        steps = out["per_column"][col]["steps"]
        return out["per_column"][col]["by_step"][str(steps[-1])]["groups"]

    gate = {"flat_ratio_threshold": FLAT_RATIO,
            "control_ratio_threshold": CONTROL_RATIO,
            "primary_metric": PRIMARY_METRIC, "per_metric": {}}
    for k in GATE_METRICS:
        con = {}
        for col in CONSTRAINED_COLUMNS:
            if col not in out["per_column"]:
                continue
            g = final_groups(col)
            n, u = g["normalized"][k], g["unnormalized"][k]
            ratio = n["abs_mean"] / u["abs_mean"] if u["abs_mean"] > 0 else float("nan")
            con[col] = {
                "abs_slope_normalized": n["abs_mean"],
                "abs_slope_unnormalized": u["abs_mean"],
                "slope_normalized": n["mean"], "slope_norm_ci": [n["lo"], n["hi"]],
                "slope_unnormalized": u["mean"], "slope_unnorm_ci": [u["lo"], u["hi"]],
                "flat_ratio": ratio,
                "gap": u["abs_mean"] - n["abs_mean"],
                "unnorm_slope_nonzero": bool(np.isfinite(u["lo"]) and (u["lo"] > 0 or u["hi"] < 0)),
                "passes": bool(np.isfinite(ratio) and ratio <= FLAT_RATIO
                               and np.isfinite(u["lo"]) and (u["lo"] > 0 or u["hi"] < 0)),
            }
        ctrl = None
        if CONTROL_COLUMN in out["per_column"]:
            g = final_groups(CONTROL_COLUMN)
            n, u = g["normalized"][k], g["unnormalized"][k]
            gap = u["abs_mean"] - n["abs_mean"]
            con_gap = max([v["gap"] for v in con.values()] or [float("nan")])
            both_zero = bool(
                np.isfinite(n["lo"]) and n["lo"] <= 0 <= n["hi"]
                and np.isfinite(u["lo"]) and u["lo"] <= 0 <= u["hi"])
            ctrl = {
                "abs_slope_normalized": n["abs_mean"],
                "abs_slope_unnormalized": u["abs_mean"],
                "slope_norm_ci": [n["lo"], n["hi"]],
                "slope_unnorm_ci": [u["lo"], u["hi"]],
                "gap": gap, "constrained_max_gap": con_gap,
                "gap_ratio_vs_constrained": (gap / con_gap) if con_gap and np.isfinite(con_gap)
                and con_gap > 0 else float("nan"),
                "both_group_slopes_contain_zero": both_zero,
                "passes": bool(np.isfinite(con_gap) and con_gap > 0
                               and gap <= CONTROL_RATIO * con_gap),
                "passes_strict": bool(np.isfinite(con_gap) and con_gap > 0
                                      and gap <= CONTROL_RATIO * con_gap and both_zero),
            }
        half_a = any(v["passes"] for v in con.values())
        half_b = bool(ctrl and ctrl["passes"])
        gate["per_metric"][k] = {
            "half_A_constrained": con, "half_A_passes": half_a,
            "half_B_control": ctrl, "half_B_passes": half_b,
            "verdict": "PASS" if (half_a and half_b) else "FAIL",
        }
    gate["verdict"] = gate["per_metric"][PRIMARY_METRIC]["verdict"]
    out["gate_P1_v2"] = gate

    out["ce_vs_mse"] = {
        col: {str(s): {m: out["per_column"][col]["by_step"][str(s)]["per_arm"][a]
                       ["slope_log2w"][PRIMARY_METRIC]
                       for m, a in (("ce", "density_ce"), ("mse", "density_mse"))}
              for s in out["per_column"][col]["steps"]}
        for col in columns
    }
    return out


def _run_one(column, T, weight, seed, kwargs):
    return run_cell(T, weight, seed, column, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/spec26_r0a_v2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--columns", nargs="*", default=list(COLUMNS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke grid (1 T, 2 weights, 1 seed, short budget)")
    args = ap.parse_args()

    t_dwells, weights, seeds = list(T_DWELLS), list(WEIGHTS), args.seeds
    episodes, steps_override, ckpts = args.episodes, None, CHECKPOINTS
    if args.quick:
        t_dwells, weights, seeds = [5], [1, 20], 1
        episodes, steps_override, ckpts = 60, 400, (100, 200)

    os.makedirs(args.out, exist_ok=True)
    kw = dict(n_episodes=episodes, checkpoints=ckpts)
    if steps_override is not None:
        kw["steps"] = steps_override
    jobs = [(col, T, wt, sd, kw) for col in args.columns
            for T in t_dwells for wt in weights for sd in range(seeds)]
    t0 = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            cells = pool.starmap(_run_one, jobs)
    else:
        cells = [_run_one(*j) for j in jobs]
    print(f"[{time.time()-t0:7.1f}s] {len(cells)} cells done", flush=True)

    for col in args.columns:
        for T in t_dwells:
            for wt in weights:
                sel = [c for c in cells if c["column"] == col
                       and c["T_dwell"] == T and c["weight"] == wt]
                path = os.path.join(args.out, f"cell_{col}_T{T}_w{wt}.json")
                with open(path, "w") as f:
                    json.dump({"column": col, "T_dwell": T, "weight": wt,
                               "seeds": sel}, f, indent=1)

    summary = summarize(cells)
    summary["config"] = {"columns": args.columns, "t_dwells": t_dwells,
                         "weights": weights, "seeds": seeds, "episodes": episodes,
                         "column_specs": {c: COLUMNS[c] for c in args.columns},
                         "steps_override": steps_override}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== GATE P1-v2 ===")
    for k in GATE_METRICS:
        g = summary["gate_P1_v2"]["per_metric"][k]
        star = " (PRIMARY)" if k == PRIMARY_METRIC else ""
        print(f"\n[{k}]{star} verdict={g['verdict']}")
        for col, v in g["half_A_constrained"].items():
            print(f"  A {col:<9} |slope| norm={v['abs_slope_normalized']:.4f} "
                  f"unnorm={v['abs_slope_unnormalized']:.4f} "
                  f"ratio={v['flat_ratio']:.3f} unnorm!=0={v['unnorm_slope_nonzero']} "
                  f"-> {'pass' if v['passes'] else 'fail'}")
        c = g["half_B_control"]
        if c:
            print(f"  B {CONTROL_COLUMN:<9} |slope| norm={c['abs_slope_normalized']:.4f} "
                  f"unnorm={c['abs_slope_unnormalized']:.4f} gap={c['gap']:.4f} "
                  f"vs constrained gap {c['constrained_max_gap']:.4f} "
                  f"(ratio {c['gap_ratio_vs_constrained']:.3f}, "
                  f"both-CIs-contain-0={c['both_group_slopes_contain_zero']}) "
                  f"-> {'pass' if c['passes'] else 'fail'}"
                  f" (strict: {'pass' if c['passes_strict'] else 'fail'})")
    print(f"\nOVERALL GATE P1-v2 ({PRIMARY_METRIC}): "
          f"{summary['gate_P1_v2']['verdict']}")
    print(f"summary -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
