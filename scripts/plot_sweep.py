"""scripts/plot_sweep.py — replot wandb cache (canonical schema only).

Reads analysis/wandb_data/*.parquet + _meta.json, groups by
(domain, ortho_coef, lr_b), and renders 4 plot types per domain.
Metric key schema is imported from scripts.wandb_pull (single source).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wandb_pull import EVAL_AGG_KEYS, TRAIN_KEYS, per_task_eval_keys  # noqa: E402

Cell = Tuple[str, str, str]  # (domain, ortho_coef, lr_b)


def _fmt_lr(lr: float) -> str:
    return f"{float(lr):.0e}".replace("e-0", "e-")


def hp_cell(config: dict) -> Cell:
    return (
        str(config.get("domain")),
        str(int(config.get("ortho_coef"))),
        _fmt_lr(config.get("lr_b")),
    )


def load_cache(in_dir: Path | str) -> Tuple[List[dict], Dict[str, pd.DataFrame]]:
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / "_meta.json").read_text())
    histories: Dict[str, pd.DataFrame] = {}
    for m in meta:
        p = in_dir / f"{m['id']}.parquet"
        if p.exists():
            histories[m["id"]] = pd.read_parquet(p)
    return meta, histories


def aggregate(
    histories: Dict[str, pd.DataFrame],
    meta: List[dict],
    metric_key: str,
) -> Dict[Cell, pd.DataFrame]:
    """For each HP cell, align runs on `_step` and compute mean/std/n.

    Pure function — no matplotlib, no IO. Runs lacking `metric_key` are
    skipped for that metric.
    """
    by_cell: Dict[Cell, List[pd.DataFrame]] = defaultdict(list)
    for m in meta:
        h = histories.get(m["id"])
        if h is None or metric_key not in h.columns or "_step" not in h.columns:
            continue
        sub = h[["_step", metric_key]].dropna()
        if sub.empty:
            continue
        sub = sub.rename(columns={"_step": "step", metric_key: "value"})
        by_cell[hp_cell(m["config"])].append(sub)

    out: Dict[Cell, pd.DataFrame] = {}
    for cell, frames in by_cell.items():
        cat = pd.concat(frames, ignore_index=True)
        g = cat.groupby("step")["value"]
        out[cell] = pd.DataFrame({
            "step": list(g.groups.keys()),
            "mean": g.mean().values,
            "std": g.std(ddof=0).fillna(0.0).values,
            "n": g.count().values,
        })
    return out
