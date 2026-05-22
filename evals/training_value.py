"""evals/training_value.py — pure helpers for the training-data value
analysis (numpy/pandas only; no torch/jax)."""

from __future__ import annotations

import numpy as np

from evals._profile_core import _spearman


def region_labels(grip: np.ndarray, lift: np.ndarray, thr) -> np.ndarray:
    """Per-state phase: 'transport' if lifted+gripped, else 'grasp' if
    gripped, else 'reach'. Mirrors evals._profile_core._region (lift->grasp)."""
    grip = np.asarray(grip, np.float64).reshape(-1)
    lift = np.asarray(lift, np.float64).reshape(-1)
    transport = (lift > thr.delta_lift) & (grip > thr.tau_grip)
    return np.where(transport, "transport",
                    np.where(grip > thr.tau_grip, "grasp", "reach"))


def cube_to_goal_dist(cube_xyz: np.ndarray, goal_xyz: np.ndarray) -> np.ndarray:
    """Euclidean cube-to-goal distance per row."""
    cube = np.asarray(cube_xyz, np.float64).reshape(-1, 3)
    goal = np.asarray(goal_xyz, np.float64).reshape(3)
    return np.linalg.norm(cube - goal, axis=1)


def phase_spearman_table(df, value_col: str = "V", group_cols=("task",)):
    """Per (region) Spearman rho(value, -d), aggregated mean/std over the
    `group_cols` combinations (default per-task; pass ('pair','task') for a
    multi-seed run). `df` has columns region, d, <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    rows = []
    for region, g in df.groupby("region"):
        vals = []
        for _, gg in g.groupby(group_cols):
            vals.append(_spearman(gg[value_col].to_numpy(),
                                  -gg["d"].to_numpy()))
        vals = np.array(vals, dtype=float)
        rows.append({"region": region,
                     "rho_mean": float(np.nanmean(vals)),
                     "rho_std": float(np.nanstd(vals)),
                     "n_groups": int(np.isfinite(vals).sum())})
    return pd.DataFrame(rows)
