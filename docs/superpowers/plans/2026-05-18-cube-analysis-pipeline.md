# Cube Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible wandb pull→plot pipeline (locked to the current canonical metric schema) and a single-checkpoint re-run pipeline (trajectory logs, z/task-conditioning probes, video) for analyzing cube agent behavior.

**Architecture:** Thin scripts under `scripts/` plus one importable, unit-tested helper module `evals/analysis.py`. The wandb component (Phase A) is fully shippable before the checkpoint component (Phase B). No training/eval code paths are modified — all additions are additive. Canonical metric keys are defined once in `scripts/wandb_pull.py` and imported by `scripts/plot_sweep.py`.

**Tech Stack:** Python 3.11, `wandb` API, `pandas`/`pyarrow` (parquet cache), `matplotlib` (Agg), `torch`/`omegaconf`/`hydra` (existing), `pytest` + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-05-17-cube-analysis-pipeline-design.md`

**Test command:** `.venv/bin/python -m pytest <path> -v`

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/wandb_pull.py` | Canonical metric-key constants + helpers; pull runs from wandb API → `analysis/wandb_data/<id>.parquet` + `_meta.json` |
| `scripts/plot_sweep.py` | Load parquet cache → 4 plot types per domain in `analysis/plots/` (imports key constants from `wandb_pull`) |
| `evals/analysis.py` | Importable helpers: `load_cfg`, `build_env_and_agent`, `load_checkpoint`, `rollout_with_trajectory`, `save_trajectories`, `z_probe_cross_task`, `z_interp` |
| `scripts/analyze_checkpoint.py` | CLI gluing `evals/analysis.py` for one checkpoint |
| `tests/test_wandb_pull.py` | Unit tests: key constants, run filtering, schema check, pull (mocked Api) |
| `tests/test_plot_sweep.py` | Unit tests: cache load, HP-cell grouping, aggregation, plot file creation |
| `tests/test_analysis.py` | Unit tests: checkpoint round-trip, trajectory shapes, probe matrix shapes |
| `requirements.txt`, `pyproject.toml` | Add `pandas`, `pyarrow`, `matplotlib` deps |

---

# PHASE A — wandb replot pipeline (shippable on its own)

## Task A0: Add analysis dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml:10-21` (the `dependencies` list)

- [ ] **Step 1: Add deps to requirements.txt**

Append to `requirements.txt` (after the `imageio>=2.31` line):

```
# Analysis (wandb pull + plotting)
pandas>=2.2
pyarrow>=15.0
matplotlib>=3.8
```

- [ ] **Step 2: Add deps to pyproject.toml**

In `pyproject.toml`, the `[project].dependencies` list, add three entries after `"tqdm",`:

```toml
    "tqdm",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "matplotlib>=3.8",
```

- [ ] **Step 3: Verify imports resolve**

Run: `.venv/bin/python -c "import pandas, pyarrow, matplotlib; print('deps ok')"`
Expected: `deps ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "deps: add pandas/pyarrow/matplotlib for analysis pipeline"
```

---

## Task A1: Canonical metric keys + run filtering (`scripts/wandb_pull.py`)

**Files:**
- Create: `scripts/wandb_pull.py`
- Test: `tests/test_wandb_pull.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_wandb_pull.py`:

```python
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import wandb_pull as wp


def test_metric_keys_cube_includes_canonical_eval_and_train():
    keys = wp.metric_keys("cube-single-play-v0")
    assert "eval/reward/eval/success" in keys
    assert "eval/reward/eval/reward" in keys
    assert "eval/reward/cube-single-play-singletask-task1-v0/success" in keys
    assert "eval/reward/cube-single-play-singletask-task5-v0/success" in keys
    assert "train/fb_loss" in keys
    assert "train/orth_loss" in keys
    assert "train/actor_loss" in keys
    assert "train/bc_flow_loss" in keys
    # old nested style must NOT appear
    assert "train/loss/fb" not in keys


def _run(group, tags, created, domain):
    return SimpleNamespace(
        id="x", group=group, tags=tags,
        created_at=created.isoformat().replace("+00:00", "Z"),
        config={"domain": domain, "ortho_coef": 1000, "lr_b": 1e-4, "seed": 1},
    )


def test_filter_runs_by_group_tag_hours_domain():
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    recent = _run("g1", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    old = _run("g1", ["sweep"], now - timedelta(hours=99), "cube-single-play-v0")
    wrong_group = _run("g2", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    wrong_domain = _run("g1", ["sweep"], now - timedelta(hours=1), "foo-v0")
    kept = wp.filter_runs(
        [recent, old, wrong_group, wrong_domain],
        groups=["g1"], tags=["sweep"], hours=36,
        domains=["cube-single-play-v0"], now=now,
    )
    assert [r.id for r in kept] == ["x"]
    assert kept[0] is recent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.wandb_pull'`

- [ ] **Step 3: Create `scripts/wandb_pull.py` with constants + filter**

Create `scripts/wandb_pull.py`:

```python
"""scripts/wandb_pull.py — pull canonical-schema run histories from wandb.

Targets EXACTLY the metric names the current code logs (train.py +
evals/ogbench.py). No back-compat remapping. See
docs/superpowers/specs/2026-05-17-cube-analysis-pipeline-design.md §4.3.

Usage:
    python scripts/wandb_pull.py --project amsks/factored-fb \\
        --groups cube-single-video-replay-ortho1000-lrb1e-4 --hours 96
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

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
    """Pure run filter: group ∈ groups, any tag ∈ tags, within `hours`,
    config.domain ∈ domains. Empty/None filter means 'do not filter on it'."""
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
```

- [ ] **Step 4: Ensure `scripts/` is an importable package**

Tests and `plot_sweep.py` do `from scripts import wandb_pull`. Create an
empty `scripts/__init__.py` if it does not already exist:

Run: `test -f scripts/__init__.py || : > scripts/__init__.py`
Expected: file `scripts/__init__.py` exists (empty).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/wandb_pull.py tests/test_wandb_pull.py
git commit -m "feat(wandb_pull): canonical metric keys + run filtering"
```

---

## Task A2: Schema check + pull to parquet/_meta (`scripts/wandb_pull.py`)

**Files:**
- Modify: `scripts/wandb_pull.py`
- Test: `tests/test_wandb_pull.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_wandb_pull.py`:

```python
import pandas as pd


def test_has_canonical_schema_true_when_eval_agg_present():
    df_ok = pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.1]})
    df_bad = pd.DataFrame({"_step": [0], "eval/cube-x/success": [0.0]})
    assert wp.has_canonical_schema(df_ok) is True
    assert wp.has_canonical_schema(df_bad) is False


class _FakeRun:
    def __init__(self, rid, df, domain="cube-single-play-v0"):
        self.id = rid
        self.name = f"run-{rid}"
        self.group = "g1"
        self.tags = ["sweep"]
        self.state = "finished"
        self.created_at = "2026-05-18T00:00:00Z"
        self.config = {"domain": domain, "ortho_coef": 1000, "lr_b": 1e-4, "seed": 1}
        self.summary = {"eval/reward/eval/success": 0.5}
        self._df = df

    def history(self, keys=None, samples=None, pandas=True):
        return self._df


class _FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, project, order=None, per_page=None):
        return list(self._runs)


def test_pull_writes_parquet_and_meta(tmp_path):
    good = _FakeRun("aaa", pd.DataFrame(
        {"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.4],
         "train/fb_loss": [9.0, 3.0]}))
    bad = _FakeRun("bbb", pd.DataFrame({"_step": [0], "eval/old/success": [0.0]}))
    n = wp.pull(
        entity=None, project="amsks/factored-fb",
        groups=["g1"], tags=None, hours=999, domains=["cube-single-play-v0"],
        out_dir=tmp_path, api=_FakeApi([good, bad]),
    )
    assert n == 1  # only the canonical-schema run is written
    assert (tmp_path / "aaa.parquet").exists()
    assert not (tmp_path / "bbb.parquet").exists()
    meta = json.loads((tmp_path / "_meta.json").read_text())
    assert [m["id"] for m in meta] == ["aaa"]
    assert meta[0]["config"]["seed"] == 1
    back = pd.read_parquet(tmp_path / "aaa.parquet")
    assert "eval/reward/eval/success" in back.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'has_canonical_schema'`

- [ ] **Step 3: Add `has_canonical_schema` + `pull` to `scripts/wandb_pull.py`**

Append to `scripts/wandb_pull.py`:

```python
def has_canonical_schema(history: "pd.DataFrame") -> bool:
    """Canonical iff the primary aggregate eval key is present.

    `eval/reward/eval/success` only exists under the current logging
    prefix; old-schema runs use un-prefixed `eval/<task>/success`. This
    one key is therefore the exact new-vs-old discriminator (§4.3).
    """
    return EVAL_AGG_KEYS[0] in history.columns


def pull(
    *,
    entity: Optional[str],
    project: str,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    out_dir: Path,
    api=None,
) -> int:
    """Pull canonical-schema runs into out_dir. Returns #runs written.

    Per-run try/except: one failure warns and is skipped, never aborts.
    """
    import pandas as pd

    if api is None:
        import wandb
        api = wandb.Api()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = list(api.runs(project, order="-created_at", per_page=300))
    kept = filter_runs(runs, groups=groups, tags=tags, hours=hours, domains=domains)

    meta = []
    written = 0
    for r in kept:
        domain = r.config.get("domain")
        try:
            hist = r.history(keys=metric_keys(domain), samples=2000, pandas=True)
        except Exception as e:  # noqa: BLE001 - resilience over precision
            print(f"  [warn] history fetch failed for {r.id}: {e}")
            continue
        if hist is None or hist.empty:
            print(f"  [warn] empty history for {r.id}, skipping")
            continue
        if not has_canonical_schema(hist):
            print(f"  [warn] {r.id} ({getattr(r, 'name', '?')}) lacks canonical "
                  f"eval keys — old schema, skipping")
            continue
        hist.to_parquet(out_dir / f"{r.id}.parquet")
        meta.append({
            "id": r.id,
            "name": getattr(r, "name", r.id),
            "state": getattr(r, "state", None),
            "group": r.group,
            "config": dict(r.config),
            "summary": dict(getattr(r, "summary", {}) or {}),
        })
        written += 1

    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[wandb_pull] wrote {written} run(s) to {out_dir}")
    return written
```

Add `import pandas as pd` is intentionally inside `pull` for the type hint string `"pd.DataFrame"` — add a module-level `from typing import TYPE_CHECKING` block at the top imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull.py -v`
Expected: PASS (4 passed — 2 from A1 + 2 from A2)

- [ ] **Step 5: Commit**

```bash
git add scripts/wandb_pull.py tests/test_wandb_pull.py
git commit -m "feat(wandb_pull): schema check + pull to parquet/_meta"
```

---

## Task A3: `wandb_pull` CLI entrypoint

**Files:**
- Modify: `scripts/wandb_pull.py`

- [ ] **Step 1: Add `main()` + `__main__` guard**

Append to `scripts/wandb_pull.py`:

```python
def _csv(value: Optional[str]) -> Optional[List[str]]:
    return [x for x in value.split(",") if x] if value else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="amsks/factored-fb")
    ap.add_argument("--groups", default=None, help="comma-separated")
    ap.add_argument("--tags", default=None, help="comma-separated")
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--domains", default=None, help="comma-separated")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "wandb_data"))
    args = ap.parse_args()
    pull(
        entity=args.entity,
        project=args.project,
        groups=_csv(args.groups),
        tags=_csv(args.tags),
        hours=args.hours,
        domains=_csv(args.domains),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses (no network)**

Run: `.venv/bin/python scripts/wandb_pull.py --help`
Expected: argparse help text listing `--entity --project --groups --tags --hours --domains --out`

- [ ] **Step 3: Commit**

```bash
git add scripts/wandb_pull.py
git commit -m "feat(wandb_pull): CLI entrypoint"
```

---

## Task A4: Cache load + HP-cell grouping + aggregation (`scripts/plot_sweep.py`)

**Files:**
- Create: `scripts/plot_sweep.py`
- Test: `tests/test_plot_sweep.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_plot_sweep.py`:

```python
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_sweep as ps


def _cache(tmp_path):
    # two seeds, same HP cell -> aggregate should average them
    pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.4]}
                 ).to_parquet(tmp_path / "r1.parquet")
    pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.6]}
                 ).to_parquet(tmp_path / "r2.parquet")
    meta = [
        {"id": "r1", "name": "s1", "group": "g", "state": "finished",
         "config": {"domain": "cube-single-play-v0", "ortho_coef": 1000,
                    "lr_b": 1e-4, "seed": 1}, "summary": {}},
        {"id": "r2", "name": "s2", "group": "g", "state": "finished",
         "config": {"domain": "cube-single-play-v0", "ortho_coef": 1000,
                    "lr_b": 1e-4, "seed": 2}, "summary": {}},
    ]
    (tmp_path / "_meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_load_cache_returns_meta_and_histories(tmp_path):
    _cache(tmp_path)
    meta, hist = ps.load_cache(tmp_path)
    assert {m["id"] for m in meta} == {"r1", "r2"}
    assert set(hist) == {"r1", "r2"}
    assert "eval/reward/eval/success" in hist["r1"].columns


def test_hp_cell():
    assert ps.hp_cell({"domain": "cube-single-play-v0", "ortho_coef": 1000,
                        "lr_b": 1e-4}) == ("cube-single-play-v0", "1000", "1e-4")


def test_aggregate_mean_std_across_seeds(tmp_path):
    _cache(tmp_path)
    meta, hist = ps.load_cache(tmp_path)
    agg = ps.aggregate(hist, meta, "eval/reward/eval/success")
    cell = ("cube-single-play-v0", "1000", "1e-4")
    assert cell in agg
    df = agg[cell].sort_values("step").reset_index(drop=True)
    assert list(df["step"]) == [0, 100]
    assert df.loc[1, "mean"] == 0.5          # (0.4 + 0.6) / 2
    assert df.loc[1, "n"] == 2
    assert abs(df.loc[1, "std"] - 0.1) < 1e-9  # population std
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_plot_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.plot_sweep'`

- [ ] **Step 3: Create `scripts/plot_sweep.py` core**

Create `scripts/plot_sweep.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_plot_sweep.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_sweep.py tests/test_plot_sweep.py
git commit -m "feat(plot_sweep): cache load + HP grouping + aggregation"
```

---

## Task A5: Plot rendering (4 plot types + summary)

**Files:**
- Modify: `scripts/plot_sweep.py`
- Test: `tests/test_plot_sweep.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_plot_sweep.py`:

```python
def test_render_all_writes_four_outputs(tmp_path):
    _cache(tmp_path)
    out = tmp_path / "plots"
    meta, hist = ps.load_cache(tmp_path)
    ps.render_all(meta, hist, out, domains=["cube-single-play-v0"])
    assert (out / "sweep-cube-single-play-v0__eval.png").exists()
    assert (out / "sweep-cube-single-play-v0__eval_pertask.png").exists()
    assert (out / "sweep-cube-single-play-v0__train.png").exists()
    assert (out / "sweep-cube-single-play-v0__final.png").exists()
    assert (out / "final_summary.md").exists()
    assert "cube-single-play-v0" in (out / "final_summary.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plot_sweep.py::test_render_all_writes_four_outputs -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'render_all'`

- [ ] **Step 3: Add rendering functions to `scripts/plot_sweep.py`**

Append to `scripts/plot_sweep.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _label(cell: Cell) -> str:
    return f"ortho={cell[1]}, lr_b={cell[2]}"


def _plot_metric(ax, agg: Dict[Cell, pd.DataFrame], domain: str, title: str):
    plotted = False
    for cell, df in sorted(agg.items()):
        if cell[0] != domain:
            continue
        d = df.sort_values("step")
        ax.plot(d["step"], d["mean"], label=_label(cell))
        ax.fill_between(d["step"], d["mean"] - d["std"], d["mean"] + d["std"],
                        alpha=0.2)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("step")
    if plotted:
        ax.legend(fontsize=7)
    return plotted


def render_all(
    meta: List[dict],
    histories: Dict[str, pd.DataFrame],
    out_dir: Path | str,
    domains: List[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# Final eval success per HP cell", ""]

    for domain in domains:
        safe = domain

        # 1. eval success + reward
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        _plot_metric(axes[0], aggregate(histories, meta, "eval/reward/eval/success"),
                     domain, "eval/success")
        _plot_metric(axes[1], aggregate(histories, meta, "eval/reward/eval/reward"),
                     domain, "eval/reward")
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__eval.png", dpi=120)
        plt.close(fig)

        # 2. per-task success
        task_keys = [k for k in per_task_eval_keys(domain) if k.endswith("/success")]
        ncols = max(1, len(task_keys))
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
        for ax, key in zip(axes[0], task_keys):
            _plot_metric(ax, aggregate(histories, meta, key), domain,
                         key.split("/")[-2])
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__eval_pertask.png", dpi=120)
        plt.close(fig)

        # 3. train losses
        loss_keys = ["train/fb_loss", "train/orth_loss", "train/actor_loss",
                     "train/bc_flow_loss"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
        for ax, key in zip(axes[0], loss_keys):
            _plot_metric(ax, aggregate(histories, meta, key), domain, key)
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__train.png", dpi=120)
        plt.close(fig)

        # 4. final-performance bar + summary md
        agg = aggregate(histories, meta, "eval/reward/eval/success")
        cells = sorted(c for c in agg if c[0] == domain)
        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(cells)), 4))
        summary_lines.append(f"## {domain}")
        if cells:
            means, stds, labels = [], [], []
            for c in cells:
                last = agg[c].sort_values("step").iloc[-1]
                means.append(last["mean"])
                stds.append(last["std"])
                labels.append(_label(c))
                summary_lines.append(
                    f"- {_label(c)}: {last['mean']*100:.2f} ± "
                    f"{last['std']*100:.2f} (n={int(last['n'])}, "
                    f"@step {int(last['step'])})")
            ax.bar(range(len(cells)), means, yerr=stds)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        else:
            summary_lines.append("- (no data)")
        ax.set_title(f"{domain} final eval/success")
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__final.png", dpi=120)
        plt.close(fig)
        summary_lines.append("")

    (out_dir / "final_summary.md").write_text("\n".join(summary_lines))
    print(f"[plot_sweep] wrote plots to {out_dir}")
```

- [ ] **Step 4: Run full test file to verify pass**

Run: `.venv/bin/python -m pytest tests/test_plot_sweep.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_sweep.py tests/test_plot_sweep.py
git commit -m "feat(plot_sweep): render eval/pertask/train/final plots + summary"
```

---

## Task A6: `plot_sweep` CLI entrypoint

**Files:**
- Modify: `scripts/plot_sweep.py`

- [ ] **Step 1: Add `main()` + guard**

Append to `scripts/plot_sweep.py`:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir",
                    default=str(REPO_ROOT / "analysis" / "wandb_data"))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "plots"))
    ap.add_argument("--domains", default="cube-single-play-v0,antmaze-medium-navigate-v0",
                    help="comma-separated")
    args = ap.parse_args()
    meta, hist = load_cache(args.in_dir)
    domains = [d for d in args.domains.split(",") if d]
    render_all(meta, hist, args.out, domains)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `.venv/bin/python scripts/plot_sweep.py --help`
Expected: argparse help with `--in --out --domains`

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_sweep.py
git commit -m "feat(plot_sweep): CLI entrypoint"
```

- [ ] **Step 4: Phase A manual acceptance (run by user, documented here)**

Run (requires `wandb login`):
```
.venv/bin/python scripts/wandb_pull.py --project amsks/factored-fb \
  --groups cube-single-video-replay-ortho1000-lrb1e-4 --hours 200 \
  --domains cube-single-play-v0
.venv/bin/python scripts/plot_sweep.py --domains cube-single-play-v0
```
Expected: parquet files refreshed in `analysis/wandb_data/`, 4 PNGs +
`final_summary.md` in `analysis/plots/`. **Phase A is shippable here.**

---

# PHASE B — checkpoint re-run / behavior pipeline

## Task B1: cfg load + agent build + checkpoint load (`evals/analysis.py`)

**Files:**
- Create: `evals/analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import analysis as A
from train import make_agent


def _cfg(domain="cube_single"):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"),
                               version_base="1.3"):
        return compose(config_name="train",
                       overrides=[f"domain={domain}", "device=cpu",
                                  "batch_size=16"])


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    a1 = make_agent(cfg, obs_space, action_dim=4)
    ckpt = tmp_path / "step_10.pt"
    torch.save(a1.state_dict(), ckpt)

    a2 = make_agent(cfg, obs_space, action_dim=4)
    A.load_checkpoint(a2, ckpt, map_location="cpu")
    # a parameter tensor should now match
    p1 = next(iter(a1.state_dict()["model"].values()))
    p2 = next(iter(a2.state_dict()["model"].values()))
    assert torch.allclose(p1, p2)


def test_load_checkpoint_clear_error_on_mismatch(tmp_path):
    junk = tmp_path / "junk.pt"
    torch.save({"model": {"not_a_real_key": torch.zeros(1)}}, junk)
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        A.load_checkpoint(agent, junk, map_location="cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.analysis'`

- [ ] **Step 3: Create `evals/analysis.py` with loaders**

Create `evals/analysis.py`:

```python
"""evals/analysis.py — importable helpers for checkpoint re-run analysis.

Reuses train.make_agent + envs/evals; never modifies training paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf


def load_cfg(config_path: str | Path, device: str = "cpu"):
    """Load a run's .hydra/config.yaml and force the device."""
    cfg = OmegaConf.load(str(config_path))
    cfg.device = device
    return cfg


def build_env_and_agent(cfg):
    """Build (env, agent, obs_space, action_dim) from a loaded cfg."""
    from envs.ogbench import create_ogbench_env
    from train import make_agent

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    agent = make_agent(cfg, obs_space, action_dim)
    return env, agent, obs_space, action_dim


def load_checkpoint(agent, ckpt_path: str | Path, map_location: str = "cpu") -> None:
    """Load a torch state_dict checkpoint into agent with a clear error
    message when the checkpoint does not match the agent variant."""
    state = torch.load(str(ckpt_path), map_location=map_location)
    try:
        agent.load_state_dict(state)
    except (KeyError, RuntimeError) as e:
        raise RuntimeError(
            f"checkpoint does not match agent (cfg.agent build). "
            f"Loading {ckpt_path} failed: {e}"
        ) from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): cfg load + agent build + checkpoint load"
```

---

## Task B2: trajectory rollout + save (`evals/analysis.py`)

**Files:**
- Modify: `evals/analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis.py`:

```python
class _StubEnv:
    """Episode of fixed length 3, then done."""
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(15, dtype=np.float32), {"success": False}

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return (np.full(15, self.t, dtype=np.float32), 1.0, done, False,
                {"success": done})

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class _StubAgent:
    device = "cpu"

    def act(self, obs=None, z=None):
        return torch.zeros((1, 4))


def test_rollout_with_trajectory_shapes():
    res = A.rollout_with_trajectory(_StubEnv(), _StubAgent(), num_episodes=2,
                                    z=torch.zeros(4), record=False)
    assert res["observations"].shape == (2, 4, 15)  # T_max = 3 steps + reset
    assert res["actions"].shape == (2, 4, 4)
    assert list(res["lengths"]) == [4, 4]
    assert res["success"].shape == (2,)
    assert bool(res["success"][0]) is True


def test_save_trajectories_writes_npz_and_summary(tmp_path):
    res = A.rollout_with_trajectory(_StubEnv(), _StubAgent(), num_episodes=2,
                                    z=torch.zeros(4), record=False)
    A.save_trajectories(res, task="taskX", out_dir=tmp_path)
    assert (tmp_path / "trajectories" / "taskX.npz").exists()
    assert (tmp_path / "trajectories" / "trajectory_summary.parquet").exists()
    import pandas as pd
    summ = pd.read_parquet(tmp_path / "trajectories" / "trajectory_summary.parquet")
    assert len(summ) == 2
    assert set(["task", "episode", "return", "length", "success"]).issubset(
        summ.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -k trajectory -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'rollout_with_trajectory'`

- [ ] **Step 3: Add functions to `evals/analysis.py`**

Append to `evals/analysis.py`:

```python
def rollout_with_trajectory(
    env: Any,
    agent: Any,
    num_episodes: int,
    z: torch.Tensor,
    record: bool = False,
) -> Dict[str, Any]:
    """Roll out `num_episodes` episodes, collecting per-step arrays.

    Returns padded arrays [n_episodes, T_max, ...] plus lengths[n_episodes].
    Mirrors the env protocol used by envs/rollout.py.
    """
    episodes = []
    frames_all = [] if record else None
    for _ in range(num_episodes):
        obs, info = env.reset()
        ep = {"observations": [obs], "actions": [], "rewards": [],
              "terminated": [], "truncated": [], "success": False}
        ep_frames = [env.render()] if record else None
        while True:
            a = agent.act(
                obs=torch.tensor(np.asarray(obs), device=agent.device,
                                 dtype=torch.float32)[None],
                z=z,
            ).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(a)
            ep["actions"].append(a)
            ep["rewards"].append(float(reward))
            ep["terminated"].append(bool(terminated))
            ep["truncated"].append(bool(truncated))
            ep["observations"].append(obs)
            if info.get("success", False):
                ep["success"] = True
            if record:
                ep_frames.append(env.render())
            if terminated or truncated:
                break
        episodes.append(ep)
        if record:
            frames_all.append(ep_frames)

    lengths = np.array([len(e["observations"]) for e in episodes], dtype=np.int64)
    t_max = int(lengths.max())
    obs_dim = np.asarray(episodes[0]["observations"][0]).shape[-1]
    act_dim = np.asarray(episodes[0]["actions"][0]).shape[-1]

    n = len(episodes)
    observations = np.zeros((n, t_max, obs_dim), dtype=np.float32)
    actions = np.zeros((n, t_max, act_dim), dtype=np.float32)
    rewards = np.zeros((n, t_max), dtype=np.float32)
    terminated = np.zeros((n, t_max), dtype=bool)
    truncated = np.zeros((n, t_max), dtype=bool)
    for i, e in enumerate(episodes):
        L = len(e["observations"])
        observations[i, :L] = np.asarray(e["observations"], dtype=np.float32)
        na = len(e["actions"])
        actions[i, :na] = np.asarray(e["actions"], dtype=np.float32)
        rewards[i, :na] = np.asarray(e["rewards"], dtype=np.float32)
        terminated[i, :na] = np.asarray(e["terminated"], dtype=bool)
        truncated[i, :na] = np.asarray(e["truncated"], dtype=bool)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "lengths": lengths,
        "success": np.array([e["success"] for e in episodes], dtype=bool),
        "z": z.detach().cpu().numpy(),
        "frames": frames_all,
    }


def save_trajectories(result: Dict[str, Any], task: str,
                      out_dir: str | Path) -> None:
    """Write trajectories/<task>.npz and append trajectory_summary.parquet."""
    import pandas as pd

    traj_dir = Path(out_dir) / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_dir / f"{task}.npz",
        observations=result["observations"],
        actions=result["actions"],
        rewards=result["rewards"],
        terminated=result["terminated"],
        truncated=result["truncated"],
        lengths=result["lengths"],
        success=result["success"],
        z=result["z"],
    )
    rows = []
    for i in range(len(result["lengths"])):
        L = int(result["lengths"][i])
        rows.append({
            "task": task,
            "episode": i,
            "return": float(result["rewards"][i, :max(L - 1, 0)].sum()),
            "length": L,
            "success": bool(result["success"][i]),
        })
    new = pd.DataFrame(rows)
    summ_path = traj_dir / "trajectory_summary.parquet"
    if summ_path.exists():
        new = pd.concat([pd.read_parquet(summ_path), new], ignore_index=True)
    new.to_parquet(summ_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): trajectory rollout + npz/summary save"
```

---

## Task B3: z probes (`evals/analysis.py`)

**Files:**
- Modify: `evals/analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis.py`:

```python
def test_z_probe_cross_task_matrix_shape():
    tasks = ["t1", "t2", "t3"]

    def infer_z(task):
        return torch.zeros(4)

    def make_env(task):
        return _StubEnv()

    df = A.z_probe_cross_task(tasks, _StubAgent(), infer_z, make_env,
                              n_episodes=2)
    assert set(df.columns) == {"env_task", "z_task", "success", "reward"}
    assert len(df) == 9  # 3x3
    assert set(df["env_task"]) == set(tasks)


def test_z_interp_sweeps_alpha():
    def make_env(task):
        return _StubEnv()

    df = A.z_interp(torch.zeros(4), torch.ones(4), _StubAgent(),
                    make_env("t2"), n_alpha=5, n_episodes=1)
    assert len(df) == 5
    assert list(df["alpha"]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert {"alpha", "success", "reward"}.issubset(df.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "z_probe or z_interp" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'z_probe_cross_task'`

- [ ] **Step 3: Add probe functions to `evals/analysis.py`**

Append to `evals/analysis.py`:

```python
def _episode_stats(env, agent, z, n_episodes):
    res = rollout_with_trajectory(env, agent, n_episodes, z, record=False)
    success = float(np.mean(res["success"]))
    reward = float(np.mean([
        res["rewards"][i, : max(int(res["lengths"][i]) - 1, 0)].sum()
        for i in range(len(res["lengths"]))
    ]))
    return success, reward


def z_probe_cross_task(tasks, agent, infer_z, make_env, n_episodes: int = 10):
    """Roll out in env_task conditioned on z_task's z for all pairs.

    `infer_z(task) -> Tensor`, `make_env(task) -> env`. Returns a tidy
    DataFrame (env_task, z_task, success, reward) — a |tasks|^2 matrix.
    """
    import pandas as pd

    zs = {t: infer_z(t) for t in tasks}
    rows = []
    for env_task in tasks:
        env = make_env(env_task)
        try:
            for z_task in tasks:
                s, r = _episode_stats(env, agent, zs[z_task], n_episodes)
                rows.append({"env_task": env_task, "z_task": z_task,
                             "success": s, "reward": r})
        finally:
            if hasattr(env, "close"):
                env.close()
    return pd.DataFrame(rows)


def z_interp(z_a, z_b, agent, env, n_alpha: int = 11, n_episodes: int = 10):
    """Sweep z = (1-α)·z_a + α·z_b for α in linspace(0,1,n_alpha) in `env`."""
    import pandas as pd

    rows = []
    for alpha in np.linspace(0.0, 1.0, n_alpha):
        z = (1.0 - float(alpha)) * z_a + float(alpha) * z_b
        s, r = _episode_stats(env, agent, z, n_episodes)
        rows.append({"alpha": round(float(alpha), 10), "success": s,
                     "reward": r})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): cross-task z matrix + z interpolation probes"
```

---

## Task B4: `scripts/analyze_checkpoint.py` CLI

**Files:**
- Create: `scripts/analyze_checkpoint.py`

- [ ] **Step 1: Create the CLI**

Create `scripts/analyze_checkpoint.py`:

```python
"""scripts/analyze_checkpoint.py — re-run ONE checkpoint for behavior analysis.

Loads a run's .hydra/config.yaml + a checkpoint .pt, then writes trajectory
logs, z/task-conditioning probes, and (optionally) videos. Caller scripts any
looping over checkpoints.

Usage:
    python scripts/analyze_checkpoint.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/checkpoint_runs/s3_final --what traj,probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default=None, help="comma-separated; default=all")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--what", default="traj,probe",
                    help="comma-separated subset of traj,probe,video")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mujoco-gl", default=None,
                    help="osmesa|glfw|egl; sets MUJOCO_GL before env import")
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import numpy as np
    import torch

    from evals.analysis import (
        build_env_and_agent, load_cfg, load_checkpoint,
        rollout_with_trajectory, save_trajectories, z_interp,
        z_probe_cross_task,
    )
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    what = set(x for x in args.what.split(",") if x)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))

    # Reuse the evaluator only for its z-inference machinery.
    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions,
    )
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False,
    )

    def infer_z(task):
        z, _ = evaluator._infer_z(task)
        return z

    summary = {"config": args.config, "checkpoint": args.checkpoint,
               "tasks": tasks, "per_task": {}}

    if "traj" in what or "video" in what:
        for task in tasks:
            tenv, _ = create_ogbench_env(task, seed=cfg.seed,
                                         obs_type=cfg.obs_type)
            try:
                z = infer_z(task)
                res = rollout_with_trajectory(
                    tenv, agent, args.n_episodes, z,
                    record=("video" in what))
            finally:
                tenv.close()
            if "traj" in what:
                save_trajectories(res, task=task, out_dir=out)
            if "video" in what and res["frames"]:
                try:
                    _save_videos(out, task, res["frames"])
                except Exception as e:  # noqa: BLE001 - render is best-effort
                    print(f"  [warn] video write failed for {task}: {e} "
                          f"(traj/probe unaffected)")
            summary["per_task"][task] = {
                "success": float(np.mean(res["success"])),
            }

    if "probe" in what:
        def make_env(task):
            e, _ = create_ogbench_env(task, seed=cfg.seed,
                                      obs_type=cfg.obs_type)
            return e
        probe_dir = out / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        df = z_probe_cross_task(tasks, agent, infer_z, make_env,
                                n_episodes=args.n_episodes)
        df.to_parquet(probe_dir / "cross_task.parquet")
        _heatmap(df, probe_dir / "cross_task.png")

        # z interpolation between the first two tasks' inferred z
        if len(tasks) >= 2:
            ienv = make_env(tasks[1])
            try:
                idf = z_interp(infer_z(tasks[0]), infer_z(tasks[1]), agent,
                               ienv, n_alpha=11, n_episodes=args.n_episodes)
            finally:
                if hasattr(ienv, "close"):
                    ienv.close()
            idf.to_parquet(probe_dir / "z_interp.parquet")
            _lineplot(idf, probe_dir / "z_interp.png",
                      title=f"z: {tasks[0]} -> {tasks[1]} (env={tasks[1]})")

    (out / "metrics.json").write_text(json.dumps(summary, indent=2,
                                                 default=str))
    print(f"[analyze_checkpoint] done -> {out}")


def _save_videos(out: Path, task: str, frames) -> None:
    import imageio
    import numpy as np
    vid_dir = out / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for k, ep in enumerate(frames):
        imageio.mimsave(vid_dir / f"{task}_ep{k}.gif",
                        np.stack(ep), format="GIF", fps=30, loop=0)


def _heatmap(df, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="env_task", columns="z_task", values="success")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("z task")
    ax.set_ylabel("env task")
    fig.colorbar(im, ax=ax, label="success")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lineplot(df, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["alpha"], df["success"], marker="o", label="success")
    ax.plot(df["alpha"], df["reward"], marker="s", label="reward")
    ax.set_xlabel("alpha (z_a -> z_b)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `.venv/bin/python scripts/analyze_checkpoint.py --help`
Expected: argparse help listing `--config --checkpoint --out --device --tasks --n-episodes --what --seed --mujoco-gl`

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_checkpoint.py
git commit -m "feat(analyze_checkpoint): single-checkpoint behavior CLI"
```

- [ ] **Step 4: Phase B manual acceptance (run by user, documented here)**

Run (needs the OGBench cube dataset available at `cfg.data_path`):
```
.venv/bin/python scripts/analyze_checkpoint.py \
  --config RESULTS/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3/.hydra/config.yaml \
  --checkpoint RESULTS/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3/checkpoints/final.pt \
  --out analysis/checkpoint_runs/s3_final --what traj,probe --n-episodes 5
```
Expected: `analysis/checkpoint_runs/s3_final/` with `trajectories/*.npz`,
`trajectories/trajectory_summary.parquet`, `probes/cross_task.parquet` +
`.png`, `probes/z_interp.parquet` + `.png`, and `metrics.json`. (Video on
macOS may require `--mujoco-gl osmesa`; if rendering fails, traj/probe still
complete.)

---

## Final verification

- [ ] Run full suite: `.venv/bin/python -m pytest tests/ -v` → all green
- [ ] Confirm no training/eval code paths were modified:
  `git diff --stat main -- train.py evals/ogbench.py envs/rollout.py agents/`
  → empty
```
