# Pixel Results Paper Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, expandable section to the cube-failure paper reporting FB+FlowBC performance on `cube-single` from pixel observations (IQM learning curves + final success table + per-task filmstrips), with a reserved one-flag slot for GCIQL-pixel.

**Architecture:** Reuse the existing wandb-pull → rliable-IQM → paper pipeline. Add an `--obs-type` filter to `scripts/wandb_pull.py` to separate pixel runs from state runs (same domain, empty group/tags). Refactor `scripts/fb_gciql_curves.py`'s `render()` to be method-generic, then add a thin `scripts/fb_pixel_curves.py` driver and a `scripts/pixel_filmstrips.py` GIF→PNG tool. Edit `PAPER/fb-cube-failure/fb_cube_failure.tex`.

**Tech Stack:** Python 3 (`.venv`), pandas, numpy, rliable (with arch8 monkeypatch), matplotlib, wandb, imageio, Pillow; LaTeX.

**Spec:** `docs/superpowers/specs/2026-05-20-pixel-results-paper-extension-design.md`

---

## Conventions & ground rules (read first)

- **Always run Python via `.venv/bin/python`** (rliable needs the arch8 shim in the venv). Run tests via `.venv/bin/python -m pytest`.
- **`scripts/`, `tests/`, `docs/` are git-tracked → commit them.**
- **`PAPER/`, `analysis/`, `RESULTS/` are untracked and deliberately kept out of git.** DO NOT `git add` them. Generate artifacts in place and leave them untracked. `RESULTS/` contains multi-hundred-MB checkpoints — never stage it.
- `scripts/` is a package (`scripts/__init__.py` exists), so `from scripts.X import Y` works after inserting `REPO_ROOT` on `sys.path`.
- The 5 FB pixel runs log canonical keys identical to state runs; the only disambiguator is `config.obs_type == "pixels"`.
- New paper figures: write the PNG into `PAPER/fb-cube-failure/figures/` and reference it as `Cube-Failure/figures/<name>.png` in the `.tex` (match the existing — build-specific — convention; do not "fix" it).

---

## File structure

- `scripts/wandb_pull.py` (modify) — add `obs_type` filter param + `--obs-type` CLI.
- `scripts/fb_gciql_curves.py` (modify) — `render(methods, out_dir, *, title_prefix, n_seeds_note)` generic over methods; `main()` behavior unchanged.
- `scripts/fb_pixel_curves.py` (create) — FB-pixel driver, optional `--gciql` slot.
- `scripts/pixel_filmstrips.py` (create) — eval-GIF → per-task filmstrip PNGs.
- `tests/test_wandb_pull_obs_type.py` (create).
- `tests/test_fb_gciql_curves.py` (modify) — add a generic-`render` test.
- `tests/test_fb_pixel_curves.py` (create).
- `tests/test_pixel_filmstrips.py` (create).
- `PAPER/fb-cube-failure/fb_cube_failure.tex` (modify) — new `sec:pixels` section (NOT committed).
- `PAPER/fb-cube-failure/figures/pixel_*.png`, `PAPER/fb-cube-failure/tables/pixel_iqm_table.tex`, `analysis/fb_pixel/*` (generated, NOT committed).

---

## Task 1: `--obs-type` filter in `wandb_pull.py`

**Files:**
- Modify: `scripts/wandb_pull.py` (`filter_runs`, `pull`, `main`)
- Test: `tests/test_wandb_pull_obs_type.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wandb_pull_obs_type.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import wandb_pull as wp


class FakeRun:
    def __init__(self, rid, obs_type):
        self.id = rid
        self.group = None
        self.tags = []
        self.created_at = datetime.now(timezone.utc)
        self.config = {"domain": "cube-single-play-v0", "obs_type": obs_type}


def test_obs_type_filter_keeps_only_matching():
    runs = [FakeRun("pix", "pixels"), FakeRun("st", "state")]
    kept = wp.filter_runs(runs, groups=None, tags=None, hours=96,
                          domains=None, obs_type="pixels")
    assert [r.id for r in kept] == ["pix"]


def test_obs_type_none_keeps_all():
    runs = [FakeRun("pix", "pixels"), FakeRun("st", "state")]
    kept = wp.filter_runs(runs, groups=None, tags=None, hours=96,
                          domains=None, obs_type=None)
    assert {r.id for r in kept} == {"pix", "st"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull_obs_type.py -v`
Expected: FAIL — `TypeError: filter_runs() got an unexpected keyword argument 'obs_type'`.

- [ ] **Step 3: Add `obs_type` to `filter_runs`**

In `scripts/wandb_pull.py`, change the `filter_runs` signature to add `obs_type` (keyword), and add the filter in the loop. New signature + new check:

```python
def filter_runs(
    runs: Iterable,
    *,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    obs_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> List:
    """Pure run filter: group in groups, any tag in tags, within `hours`,
    config.domain in domains, config.obs_type == obs_type. Empty/None filter
    means 'do not filter on it'."""
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
        if obs_type and r.config.get("obs_type") != obs_type:
            continue
        kept.append(r)
    return kept
```

- [ ] **Step 4: Thread `obs_type` through `pull` and `main`**

In `pull(...)`, add `obs_type: Optional[str] = None` to the signature (place it after `domains`) and pass it into the `filter_runs(...)` call:

```python
def pull(
    *,
    entity: Optional[str],
    project: str,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    out_dir: Path,
    obs_type: Optional[str] = None,
    api=None,
) -> int:
```

and:

```python
    kept = filter_runs(runs, groups=groups, tags=tags, hours=hours,
                       domains=domains, obs_type=obs_type)
```

In `main()`, add the CLI arg and pass it:

```python
    ap.add_argument("--obs-type", default=None,
                    help="filter on config.obs_type, e.g. pixels or state")
```

and in the `pull(...)` call add `obs_type=args.obs_type,`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wandb_pull_obs_type.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/wandb_pull.py tests/test_wandb_pull_obs_type.py
git commit -m "feat(wandb_pull): --obs-type filter to separate pixel from state runs"
```

---

## Task 2: Make `fb_gciql_curves.render()` method-generic

**Files:**
- Modify: `scripts/fb_gciql_curves.py` (`render`, `main`)
- Test: `tests/test_fb_gciql_curves.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fb_gciql_curves.py`:

```python
def test_render_generic_single_method(tmp_path):
    # Three seeds, FB-only; render must emit all four artifacts with one row.
    fb = {}
    for s in (0, 1, 2):
        df = pd.DataFrame({"step": [0, 500_000, 1_000_000],
                           "overall": [0.0, 0.3, 0.5]})
        for t in fc.TASKS:
            df[t] = [0.0, 0.25, 0.45]
        fb[s] = df
    out = tmp_path / "out"
    fc.render({"FB": fb}, out, title_prefix="FB (pixels)",
              n_seeds_note="n=3 seeds")
    for f in ("curves_pertask.png", "curve_aggregate.png",
              "iqm_table.tex", "iqm_table.md"):
        assert (out / f).exists()
    tex = (out / "iqm_table.tex").read_text()
    assert tex.count(r" \\") == 2  # header row + one FB data row
    assert "GCIQL" not in tex
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fb_gciql_curves.py::test_render_generic_single_method -v`
Expected: FAIL — `render()` currently has signature `render(fb, gc, out_dir)` → `TypeError`.

- [ ] **Step 3: Replace `render` with the method-generic version**

In `scripts/fb_gciql_curves.py`, replace the entire `render(...)` function with:

```python
def render(methods: Dict[str, Dict[int, pd.DataFrame]], out_dir: Path,
           *, title_prefix: str = "FB vs GCIQL",
           n_seeds_note: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(methods)

    def _inputs(cols: List[str]) -> Dict[str, np.ndarray]:
        return {m: interp_tensor(methods[m], cols) for m in names}

    # Per-task sample-efficiency curves (IQM across seeds per task).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), TASKS):
        pt, ci = iqm_curve(_inputs([t]))
        _plot_curve(ax, pt, ci, TASK_LABELS[t])
        ax.set_ylabel("IQM success")
    pt_all, ci_all = iqm_curve(_inputs(TASKS))
    _plot_curve(axes.ravel()[5], pt_all, ci_all, "aggregate (all 5 tasks)")
    axes.ravel()[5].set_ylabel("IQM success")
    axes.ravel()[5].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{title_prefix} — cube-single eval success (IQM, 95% CI)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_pertask.png", dpi=130)
    plt.close(fig)

    # Standalone aggregate curve.
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve(ax, pt_all, ci_all,
                f"{title_prefix} — aggregate across 5 tasks")
    ax.set_ylabel("IQM success (95% stratified bootstrap CI)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_aggregate.png", dpi=130)
    plt.close(fig)

    # Final IQM table (last eval per seed).
    cols = TASKS + ["overall"]
    final = {m: final_scores(methods[m], cols) for m in names}
    pt, ci = iqm_table_estimates(final)
    note = f" {n_seeds_note}" if n_seeds_note else ""
    hdr = ["Method"] + [TASK_LABELS[t] for t in TASKS] + ["overall"]
    md = [f"# {title_prefix} — final eval success (IQM, 95% CI)", "",
          "Final eval per seed (failed seeds use their last checkpoint;"
          f"{note}). IQM with 95% stratified-bootstrap CI.", "",
          "| " + " | ".join(hdr) + " |",
          "| " + " | ".join([":---"] * len(hdr)) + " |"]
    for method in names:
        row = [method]
        for k in range(len(cols)):
            lo, hi = ci[method][0][k], ci[method][1][k]
            row.append(f"{pt[method][k]*100:.1f} "
                       f"[{lo*100:.1f}, {hi*100:.1f}]")
        md.append("| " + " | ".join(row) + " |")
    (out_dir / "iqm_table.md").write_text("\n".join(md) + "\n")

    # LaTeX booktabs version for the paper.
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & ".join(["Method"] + [t for t in TASKS] + ["Overall"])
           + r" \\", r"\midrule"]
    for method in names:
        cells = [method]
        for k in range(len(cols)):
            lo, hi = ci[method][0][k] * 100, ci[method][1][k] * 100
            cells.append(f"{pt[method][k]*100:.1f} "
                         f"\\scriptsize[{lo:.1f}, {hi:.1f}]")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "iqm_table.tex").write_text("\n".join(tex) + "\n")

    print(f"[curves] wrote curves + table to {out_dir}")
    print("\n".join(md))
```

- [ ] **Step 4: Update `main()` to call the generic `render`**

In `scripts/fb_gciql_curves.py`, replace the last line of `main()`:

```python
    render(fb, gc, Path(args.out))
```

with:

```python
    render({"FB": fb, "GCIQL": gc}, Path(args.out))
```

- [ ] **Step 5: Run the full curves test file to verify no regression**

Run: `.venv/bin/python -m pytest tests/test_fb_gciql_curves.py -v`
Expected: PASS (all prior tests + `test_render_generic_single_method`). The header line counts: `iqm_table.tex` has the column header row ending in ` \\` plus one FB data row ending in ` \\` → `tex.count(" \\\\") == 2`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fb_gciql_curves.py tests/test_fb_gciql_curves.py
git commit -m "refactor(fb_gciql_curves): method-generic render() for reuse"
```

---

## Task 3: `fb_pixel_curves.py` driver

**Files:**
- Create: `scripts/fb_pixel_curves.py`
- Test: `tests/test_fb_pixel_curves.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fb_pixel_curves.py`:

```python
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import fb_gciql_curves as fc
from scripts import fb_pixel_curves as fp


def _write_fb_cache(cache_dir: Path, seeds):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in seeds:
        rid = f"run{s}"
        cols = {"_step": [0, 500_000, 1_000_000],
                fc.FB_OVERALL: [0.0, 0.3, 0.5]}
        for t in fc.TASKS:
            cols[fc._fb_task_key(t)] = [0.0, 0.25, 0.45]
        pd.DataFrame(cols).to_parquet(cache_dir / f"{rid}.parquet")
        meta.append({"id": rid, "config": {"seed": s}})
    (cache_dir / "_meta.json").write_text(json.dumps(meta))


def test_fb_pixel_curves_outputs(tmp_path, monkeypatch):
    cache = tmp_path / "wandb_data_fbpixel"
    _write_fb_cache(cache, seeds=[0, 1, 2, 3, 4])
    out = tmp_path / "fb_pixel"
    monkeypatch.setattr(sys, "argv",
                        ["fb_pixel_curves.py", "--fb", str(cache),
                         "--out", str(out)])
    fp.main()
    for f in ("curves_pertask.png", "curve_aggregate.png",
              "iqm_table.tex", "iqm_table.md"):
        assert (out / f).exists()
    tex = (out / "iqm_table.tex").read_text()
    assert "FB" in tex
    assert "GCIQL" not in tex  # FB-only until GCIQL pixel data is supplied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fb_pixel_curves.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fb_pixel_curves'`.

- [ ] **Step 3: Create `scripts/fb_pixel_curves.py`**

```python
"""scripts/fb_pixel_curves.py — FB (and optionally GCIQL) pixel eval curves.

Loads FB pixel run histories from a wandb_pull cache and produces per-task +
aggregate IQM success curves (95% stratified-bootstrap CI) and a final IQM
success table, reusing the method-generic machinery in fb_gciql_curves.

GCIQL pixel runs are not available yet; pass --gciql <eval.csv root> to add
them when they exist (the curves + table then include both methods).

Usage:
    python scripts/fb_pixel_curves.py \
        --fb analysis/wandb_data_fbpixel \
        --out analysis/fb_pixel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fb_gciql_curves import load_fb, load_gciql, render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fb", default=str(REPO_ROOT / "analysis"
                                        / "wandb_data_fbpixel"))
    ap.add_argument("--gciql", default=None,
                    help="optional GCIQL pixel eval.csv root (reserved slot)")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "fb_pixel"))
    args = ap.parse_args()

    methods = {"FB": load_fb(args.fb)}
    if args.gciql:
        methods["GCIQL"] = load_gciql(args.gciql)

    print(f"[fb_pixel_curves] methods={list(methods)} "
          f"FB seeds={sorted(methods['FB'])}")
    render(methods, Path(args.out), title_prefix="FB (pixels)",
           n_seeds_note="n=5 seeds")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fb_pixel_curves.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/fb_pixel_curves.py tests/test_fb_pixel_curves.py
git commit -m "feat(fb_pixel_curves): FB-pixel IQM curves + table with reserved GCIQL slot"
```

---

## Task 4: `pixel_filmstrips.py` GIF→PNG tool

**Files:**
- Create: `scripts/pixel_filmstrips.py`
- Test: `tests/test_pixel_filmstrips.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_filmstrips.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import pixel_filmstrips as pf


def test_sample_frames_count_and_endpoints():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    out = pf.sample_frames(frames, 4)
    assert len(out) == 4
    assert int(out[0][0, 0, 0]) == 0
    assert int(out[-1][0, 0, 0]) == 9


def test_make_filmstrip_width():
    frames = [np.zeros((8, 6, 3), dtype=np.uint8) for _ in range(3)]
    strip = pf.make_filmstrip(frames, pad=2)
    assert strip.height == 8
    assert strip.width == 6 * 3 + 2 * 2  # 22


def test_build_writes_five_filmstrips(tmp_path):
    import imageio.v2 as imageio
    root = tmp_path / "fb-pixel-results"
    run = root / "2026-05-20_00-00-00__cube-single-play-v0__fb_flowbc__s0"
    vid = run / "eval_videos" / "step_1000000"
    vid.mkdir(parents=True)
    frames = [np.full((16, 16, 3), 30 * i, dtype=np.uint8) for i in range(8)]
    for t in range(1, 6):
        imageio.mimsave(
            str(vid / f"cube-single-play-singletask-task{t}-v0.gif"), frames)
    out = tmp_path / "figures"
    written = pf.build(root, seed=0, step="step_1000000", n_frames=6,
                       out_dir=out)
    assert len(written) == 5
    for p in written:
        assert p.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pixel_filmstrips.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pixel_filmstrips'`.

- [ ] **Step 3: Create `scripts/pixel_filmstrips.py`**

```python
"""scripts/pixel_filmstrips.py — static filmstrips from FB pixel eval GIFs.

For each cube-single task, sample evenly-spaced frames from a representative
seed's eval GIF at a chosen checkpoint and composite a horizontal filmstrip PNG
for the paper (a PDF cannot embed animation).

Usage:
    python scripts/pixel_filmstrips.py \
        --results-root RESULTS/fb-pixel-results \
        --seed 0 --step step_1000000 --frames 6 \
        --out PAPER/fb-cube-failure/figures
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

TASKS = [1, 2, 3, 4, 5]


def sample_frames(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """n evenly-spaced frames, inclusive of the first and last."""
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def make_filmstrip(frames: List[np.ndarray], pad: int = 2) -> Image.Image:
    """Composite frames into one horizontal strip with white separators."""
    imgs = [Image.fromarray(np.asarray(f)).convert("RGB") for f in frames]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    strip = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.width + pad
    return strip


def find_run_dir(results_root: Path, seed: int) -> Path:
    matches = sorted(glob.glob(str(results_root / f"*__s{seed}")))
    if not matches:
        raise FileNotFoundError(
            f"no run dir matching *__s{seed} under {results_root}")
    return Path(matches[0])


def build(results_root: Path, seed: int, step: str, n_frames: int,
          out_dir: Path) -> List[Path]:
    import imageio.v2 as imageio

    run = find_run_dir(results_root, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for t in TASKS:
        gif = (run / "eval_videos" / step
               / f"cube-single-play-singletask-task{t}-v0.gif")
        frames = imageio.mimread(str(gif))
        strip = make_filmstrip(sample_frames(frames, n_frames))
        dest = out_dir / f"pixel_filmstrip_task{t}.png"
        strip.save(dest)
        written.append(dest)
        print(f"[pixel_filmstrips] wrote {dest} ({len(frames)} src frames)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="RESULTS/fb-pixel-results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", default="step_1000000")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default="PAPER/fb-cube-failure/figures")
    args = ap.parse_args()
    build(Path(args.results_root), args.seed, args.step, args.frames,
          Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pixel_filmstrips.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pixel_filmstrips.py tests/test_pixel_filmstrips.py
git commit -m "feat(pixel_filmstrips): per-task filmstrips from FB pixel eval GIFs"
```

---

## Task 5: Generate real artifacts (pull + curves + filmstrips)

This task produces data/figures from real runs. It is NOT TDD and its outputs are
NOT committed (untracked `analysis/`, `PAPER/`). If the wandb pull yields fewer
than 5 FB pixel seeds, STOP and report — do not proceed with partial/fabricated
data.

**Files:**
- Generate: `analysis/wandb_data_fbpixel/*.parquet` + `_meta.json`
- Generate: `analysis/fb_pixel/{curves_pertask.png, curve_aggregate.png, iqm_table.tex, iqm_table.md}`
- Generate: `PAPER/fb-cube-failure/figures/pixel_filmstrip_task{1..5}.png`
- Generate: `PAPER/fb-cube-failure/figures/pixel_curves_pertask.png`
- Generate: `PAPER/fb-cube-failure/tables/pixel_iqm_table.tex`

- [ ] **Step 1: Pull FB pixel runs from wandb**

Run:
```bash
.venv/bin/python scripts/wandb_pull.py --project amsks/factored-fb \
    --obs-type pixels --domains cube-single-play-v0 --hours 168 \
    --out analysis/wandb_data_fbpixel
```
Expected: `[wandb_pull] wrote 5 run(s) to analysis/wandb_data_fbpixel`.

- [ ] **Step 2: Verify exactly 5 seeds landed**

Run:
```bash
.venv/bin/python -c "import json; m=json.load(open('analysis/wandb_data_fbpixel/_meta.json')); print(len(m), sorted(r['config'].get('seed') for r in m)); print({r['config'].get('obs_type') for r in m})"
```
Expected: `5 [0, 1, 2, 3, 4]` and `{'pixels'}`. If not 5 pixel seeds, STOP and report (check `--hours`, project, that the runs finished and logged the canonical `eval/reward/eval/success` key).

- [ ] **Step 3: Generate the pixel curves + IQM table**

Run:
```bash
.venv/bin/python scripts/fb_pixel_curves.py \
    --fb analysis/wandb_data_fbpixel --out analysis/fb_pixel
```
Expected: prints `[fb_pixel_curves] methods=['FB'] FB seeds=[0, 1, 2, 3, 4]`, the markdown table, and `[curves] wrote curves + table to analysis/fb_pixel`. Confirm the 4 files exist in `analysis/fb_pixel/`.

- [ ] **Step 4: Generate the per-task filmstrips**

Run:
```bash
.venv/bin/python scripts/pixel_filmstrips.py \
    --results-root RESULTS/fb-pixel-results --seed 0 \
    --step step_1000000 --frames 6 \
    --out PAPER/fb-cube-failure/figures
```
Expected: 5 lines `[pixel_filmstrips] wrote .../pixel_filmstrip_task{N}.png`. Confirm all 5 PNGs exist.

- [ ] **Step 5: Copy curve PNG + table into the paper tree**

Run:
```bash
cp analysis/fb_pixel/curves_pertask.png PAPER/fb-cube-failure/figures/pixel_curves_pertask.png
cp analysis/fb_pixel/iqm_table.tex PAPER/fb-cube-failure/tables/pixel_iqm_table.tex
ls -la PAPER/fb-cube-failure/figures/pixel_*.png PAPER/fb-cube-failure/tables/pixel_iqm_table.tex
```
Expected: `pixel_curves_pertask.png`, `pixel_filmstrip_task{1..5}.png`, and `tables/pixel_iqm_table.tex` all present.

- [ ] **Step 6: Record the actual aggregate FB-pixel IQM for the prose**

Run:
```bash
cat analysis/fb_pixel/iqm_table.md
```
Note the FB **overall** IQM value — Task 6 uses it to write the intro/takeaway prose so the text matches the real numbers. (No commit; outputs are untracked.)

---

## Task 6: Add the pixel section to the paper

**Files:**
- Modify: `PAPER/fb-cube-failure/fb_cube_failure.tex` (NOT committed)

- [ ] **Step 1: Insert the new section before Discussion**

In `PAPER/fb-cube-failure/fb_cube_failure.tex`, immediately BEFORE the line
`\section{Discussion and Next Steps}` (around line 449), insert:

```latex
\FloatBarrier
\section{Does the failure persist under pixel observations?}
\label{sec:pixels}

The diagnosis so far uses low-dimensional state observations. A natural question
is whether the transport-resolution failure is an artifact of the state
featurization or a property of the FB representation itself. We therefore re-run
FB+FlowBC on the same five \texttt{cube-single} tasks, but from $64\times64$ RGB
observations encoded by a DrQ convolutional encoder with a frame stack of three
and random-shift augmentation. The offline data, tasks, and evaluation protocol
are otherwise identical to the state-based setting. GCIQL-pixel runs are not yet
available; this section reports FB and is structured so the GCIQL comparison and
the deeper representation diagnostics drop in once those runs complete.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_curves_pertask.png}
    \caption{Per-task and aggregate learning curves for FB+FlowBC from pixel
    observations. Curves report interquartile mean (IQM) success over five seeds
    with 95\% stratified-bootstrap confidence intervals. GCIQL-pixel curves will
    be overlaid here for the same comparison made in
    \Cref{fig:learning-curves}.}
    \label{fig:pixel-learning-curves}
\end{figure}

\begin{table}[H]
\centering
\small
\caption{Final evaluation success on the five \texttt{cube-single} tasks from
pixel observations (FB+FlowBC, IQM over five seeds with 95\% CI). The GCIQL-pixel
row will be added once those runs complete; the comparison mirrors
\Cref{tab:fb-gciql-iqm} in the state setting.}
\label{tab:pixel-iqm}
\vspace{0.25em}
% The GCIQL-pixel row appears automatically once GCIQL pixel eval.csv data is
% passed to scripts/fb_pixel_curves.py via --gciql; re-run it, then re-copy
% analysis/fb_pixel/iqm_table.tex to tables/pixel_iqm_table.tex.
\input{tables/pixel_iqm_table.tex}
\end{table}

Figures~\ref{fig:pixel-filmstrip-task1}--\ref{fig:pixel-filmstrip-task5} show
representative closed-loop rollouts (seed~0, final checkpoint) as frame
filmstrips, one per task; time runs left to right within each strip.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_filmstrip_task1.png}
    \caption{Representative pixel rollout, task1 (horizontal).}
    \label{fig:pixel-filmstrip-task1}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_filmstrip_task2.png}
    \caption{Representative pixel rollout, task2 (vertical1).}
    \label{fig:pixel-filmstrip-task2}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_filmstrip_task3.png}
    \caption{Representative pixel rollout, task3 (vertical2).}
    \label{fig:pixel-filmstrip-task3}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_filmstrip_task4.png}
    \caption{Representative pixel rollout, task4 (diagonal1).}
    \label{fig:pixel-filmstrip-task4}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\linewidth]{Cube-Failure/figures/pixel_filmstrip_task5.png}
    \caption{Representative pixel rollout, task5 (diagonal2).}
    \label{fig:pixel-filmstrip-task5}
\end{figure}

\takeaway{From pixel observations, FB+FlowBC shows the same qualitative profile
as the state setting on \texttt{cube-single}. The GCIQL-pixel comparison and the
phase, coverage, value-gradient, and backward-map diagnostics are deferred until
the visual GCIQL runs are available, at which point this section expands into the
full diagnostic chain.}
```

- [ ] **Step 2: Align the prose with the real numbers**

Using the FB **overall** IQM noted in Task 5 Step 6, edit the intro paragraph
and `\takeaway` so any quantitative claim matches the table. For example, if the
FB pixel aggregate is materially lower than the state aggregate (0.500), add a
sentence like: "Aggregate success is X\%, comparable to / below the state
result, indicating the limitation is not removed by the visual encoder." Do NOT
assert a direction the numbers don't support. If the numbers are not yet
trustworthy, keep the prose qualitative.

- [ ] **Step 3: Sanity-check the section's figure/table references resolve**

Run:
```bash
cd PAPER/fb-cube-failure
for f in figures/pixel_curves_pertask.png figures/pixel_filmstrip_task1.png \
         figures/pixel_filmstrip_task2.png figures/pixel_filmstrip_task3.png \
         figures/pixel_filmstrip_task4.png figures/pixel_filmstrip_task5.png \
         tables/pixel_iqm_table.tex; do
  test -f "$f" && echo "OK  $f" || echo "MISSING  $f"
done
cd /Users/adityamohan/git/Austin/Factored-FB
```
Expected: 7 `OK` lines, no `MISSING`.

- [ ] **Step 4: Verify LaTeX environments balance in the new section**

Run:
```bash
.venv/bin/python -c "
t=open('PAPER/fb-cube-failure/fb_cube_failure.tex').read()
import re
for env in ('figure','table','tabular','subfigure'):
    o=len(re.findall(r'\\\\begin{'+env+'}',t)); c=len(re.findall(r'\\\\end{'+env+'}',t))
    print(env,o,c,'OK' if o==c else 'MISMATCH')
"
```
Expected: every row prints `OK`. (No commit — paper is untracked.)

---

## Task 7: Final verification

**Files:** none modified (verification + plan/spec doc commit only).

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass (new: obs_type filter, generic render, fb_pixel_curves, pixel_filmstrips; plus existing).

- [ ] **Step 2: Confirm only intended paths are staged-able**

Run: `git status --short`
Expected: tracked changes limited to `scripts/` and `tests/`. `PAPER/`, `analysis/`, `RESULTS/` remain untracked (`??`) and are NOT to be added.

- [ ] **Step 3: Commit the plan document**

```bash
git add docs/superpowers/plans/2026-05-20-pixel-results-paper-extension.md
git commit -m "docs(pixel-paper): implementation plan — FB pixel results section"
```

- [ ] **Step 4: Report to the user**

Summarize: scripts/tests added and committed; FB pixel curves + table + 5 filmstrips generated under `analysis/fb_pixel/` and `PAPER/fb-cube-failure/`; new `sec:pixels` added to the paper (left untracked per repo convention). Note the FB pixel aggregate IQM and remind that GCIQL-pixel slots in via `fb_pixel_curves.py --gciql <root>` + re-copy of the table.

---

## Self-review notes

- **Spec coverage:** obs_type pull (Task 1), method-generic render + GCIQL slot (Task 2, 3), FB-pixel curves/table (Task 3, 5), filmstrips per task (Task 4, 5), dedicated section before Discussion (Task 6), n=5 caption note (render `n_seeds_note`, Task 6 caption), figure-path convention (Task 5/6), commit boundaries excluding PAPER/analysis/RESULTS (ground rules, Task 7). All covered.
- **GCIQL slot:** reserved via `--gciql` (no throwaway code) + `\input` table that regenerates with both rows; paper comment documents the one-step refresh. Refines the spec's "commented row" to a cleaner regenerate-and-reinput mechanism (same intent: FB now, GCIQL later).
- **Table embedding:** uses `\input{tables/pixel_iqm_table.tex}` (deterministic, reproducible) rather than manual inline paste — a minor, transparent refinement of the spec's inline preference; the table still renders inline in the PDF.
- **Type/name consistency:** `render(methods, out_dir, *, title_prefix, n_seeds_note)` used identically in Task 2 (def + main), Task 3 (call); `sample_frames`/`make_filmstrip`/`find_run_dir`/`build` signatures consistent between Task 4 impl and tests; `load_fb`/`load_gciql` imported from `fb_gciql_curves` match existing definitions.
