"""scripts/wandb_pull.py — pull canonical-schema run histories from wandb.

Targets EXACTLY the metric names the current code logs (train.py +
evals/ogbench.py). No back-compat remapping. See
docs/superpowers/specs/2026-05-17-cube-analysis-pipeline-design.md §4.3.

Usage:
    python scripts/wandb_pull.py --project amsks/factored-fb \
        --groups cube-single-video-replay-ortho1000-lrb1e-4 --hours 96
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.ogbench import ALL_TASKS

# Aggregate eval metrics: train.py:228 logs evaluator.run() with prefix
# "eval/reward/"; evals/ogbench.py:100-102 emits eval/{success,reward,reward#std}
EVAL_AGG_KEYS = [
    "eval/reward/eval/success",
    "eval/reward/eval/reward",
    "eval/reward/eval/reward#std",
]

# Train metrics: train.py:247 logs agent.update() with prefix "train/";
# agents/fb/agent.py + flow_bc/agent.py return a FLAT dict (not nested loss/).
TRAIN_KEYS = [
    "train/fb_loss",
    "train/orth_loss",
    "train/actor_loss",
    "train/bc_flow_loss",
    "train/q",
    "train/M1",
    "train/B_norm",
    "train/z_norm",
]


def per_task_eval_keys(domain: str) -> List[str]:
    """eval/reward/<task>/{success,reward} for every task of the domain.

    evals/ogbench.py:85-86 emits "<task>/success" and "<task>/reward";
    train.py:228 adds the "eval/reward/" prefix.
    """
    keys: List[str] = []
    for task in ALL_TASKS.get(domain, []):
        keys.append(f"eval/reward/{task}/success")
        keys.append(f"eval/reward/{task}/reward")
    return keys


def metric_keys(domain: str) -> List[str]:
    """Full canonical key list for a domain (single source of truth)."""
    return EVAL_AGG_KEYS + per_task_eval_keys(domain) + TRAIN_KEYS


def _created(run) -> datetime:
    c = run.created_at
    if isinstance(c, str):
        return datetime.fromisoformat(c.replace("Z", "+00:00"))
    return c


def filter_runs(
    runs: Iterable,
    *,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    now: Optional[datetime] = None,
) -> List:
    """Pure run filter: group in groups, any tag in tags, within `hours`,
    config.domain in domains. Empty/None filter means 'do not filter on it'."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    kept = []
    for r in runs:
        if groups and r.group not in groups:
            continue
        if tags and not (set(r.tags or []) & set(tags)):
            continue
        if _created(r) < cutoff:
            continue
        if domains and r.config.get("domain") not in domains:
            continue
        kept.append(r)
    return kept
