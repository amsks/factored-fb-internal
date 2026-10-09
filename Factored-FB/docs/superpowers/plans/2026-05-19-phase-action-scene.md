# Phase-Conditioned Object-Centric Value & Action Fields — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render 10 phase-stratified figures (task × method) showing FB vs GCIQL extracted value over cube xy, with greedy-policy cube-motion arrows, goal star, and decoded dataset-support contours.

**Architecture:** Additively persist `t` + `region` per step into `value_steps.parquet` from both recorders (reusing `evals._profile_core._region`), add a pure `evals/dataset_support.py` (decode offline cube xy + KDE, fail-loud verification), and a pure `_phase_action_fields` seam + `_phase_action_scene()` renderer in `gciql_profile_aggregate.py`. One-directional flow: recorders persist → aggregate renders. No model loading in the renderer.

**Tech Stack:** Python, numpy, pandas, scipy (gaussian_kde), matplotlib (Agg), pytest. Re-profile uses the JAX venv (GCIQL) and torch host (FB).

Spec: `docs/superpowers/specs/2026-05-19-phase-action-scene-design.md`

---

### Task 1: Persist `t` + `region` in the GCIQL recorder

**Files:**
- Modify: `scripts/gciql_profile.py:39-97` (`episodes_to_frames`)
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gciql_profile.py`:

```python
def test_episodes_to_frames_emits_t_and_region():
    from scripts import gciql_profile as gp
    from evals.phase_probe import Thresholds
    import numpy as np
    T = 6
    ep = {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.linspace(1.0, 0.1, T),
        "V": np.linspace(0.0, 1.0, T),
        "grip": np.array([0.0, 0.0, 0.9, 0.9, 0.9, 0.9]),
        "cube": np.column_stack([np.linspace(0, 0.3, T),
                                 np.zeros(T),
                                 np.array([0.02, 0.02, 0.02,
                                           0.10, 0.10, 0.10])]),
        "eff": np.zeros((T, 3), np.float64),
        "goal": np.array([0.3, 0.0, 0.02]),
        "table_z": 0.02,
        "success": True, "task": "task1", "episode": 0,
    }
    _, _, _, vs = gp.episodes_to_frames([ep], Thresholds(),
                                        np.zeros((2, 4), np.float32))
    assert list(vs["t"]) == list(range(T))
    assert set(vs["region"]) <= {"reach", "lift", "transport"}
    # grip<=0.5 -> reach; grip>0.5 & not lifted -> lift;
    # lifted (z-table_z>0.03) & gripped -> transport
    assert list(vs["region"]) == ["reach", "reach", "lift",
                                  "transport", "transport", "transport"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py::test_episodes_to_frames_emits_t_and_region -v`
Expected: FAIL with `KeyError: 'region'` (or `'t'`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/gciql_profile.py`, add the import near the existing one (line 27-28 area):

```python
from evals._profile_core import _region, _spearman, probe_coverage, transport_mask
```

(extend the existing `from evals._profile_core import ...` line to include `_region`.)

In `episodes_to_frames`, replace the per-step `vs_rows` loop (currently lines ~85-91):

```python
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        for dd, vv, mm, cxy in zip(d, V, tm, cube_xy):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "outcome": outcome, "d": float(dd),
                            "V": float(vv), "transport": bool(mm),
                            "cube_x": float(cxy[0]),
                            "cube_y": float(cxy[1])})
```

with:

```python
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        ep_for_region = dict(ep)
        ep_for_region["transport_mask"] = tm
        region = _region(ep_for_region, thr)
        for ti, (dd, vv, mm, cxy, rg) in enumerate(
                zip(d, V, tm, cube_xy, region)):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "t": int(ti), "outcome": outcome,
                            "region": str(rg), "d": float(dd),
                            "V": float(vv), "transport": bool(mm),
                            "cube_x": float(cxy[0]),
                            "cube_y": float(cxy[1])})
```

Update the trailing `pd.DataFrame(vs_rows, columns=[...])` to:

```python
    vs = pd.DataFrame(vs_rows, columns=["task", "episode", "t", "outcome",
                                        "region", "d", "V", "transport",
                                        "cube_x", "cube_y"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -v`
Expected: PASS (new test + all existing `test_gciql_profile.py` tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gciql_profile.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): persist per-step t + region in value_steps"
```

---

### Task 2: Persist `t` + `region` in the FB recorder

**Files:**
- Modify: `evals/representation_profile.py:102-134` (`probe_value_landscape`)
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_representation_profile.py` (match the file's existing import style; if it imports `from evals import representation_profile as rp`, reuse that):

```python
def test_probe_value_landscape_emits_t_and_region(monkeypatch):
    import numpy as np
    from evals import representation_profile as rp
    from evals.phase_probe import Thresholds

    T = 5
    ep = {
        "obs": np.zeros((T, 3), np.float32),
        "d": np.linspace(1.0, 0.1, T),
        "transport_mask": np.array([False, False, False, True, True]),
        "grip": np.array([0.0, 0.0, 0.9, 0.9, 0.9]),
        "cube": np.column_stack([np.linspace(0, 0.2, T), np.zeros(T),
                                 np.array([0.02, 0.02, 0.02,
                                           0.10, 0.10])]),
        "table_z": 0.02,
        "outcome": "success",
    }
    # v_values is torch/model-dependent; stub it to a deterministic ramp.
    monkeypatch.setattr(rp, "v_values",
                         lambda *a, **k: np.linspace(0.0, 1.0, T))
    _, per_step = rp.probe_value_landscape(None, None, [ep], None,
                                           thr=Thresholds())
    assert list(per_step["t"]) == list(range(T))
    assert list(per_step["region"]) == ["reach", "reach", "lift",
                                        "transport", "transport"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py::test_probe_value_landscape_emits_t_and_region -v`
Expected: FAIL — `probe_value_landscape()` got an unexpected keyword `thr` (signature lacks it) or `KeyError: 'region'`.

- [ ] **Step 3: Write minimal implementation**

In `evals/representation_profile.py`, `_region` is already imported (line 18). Add `Thresholds` import near the top imports:

```python
from evals.phase_probe import Thresholds
```

Change the signature (line 102) to accept an optional `thr`:

```python
def probe_value_landscape(model, agent, episodes, z, thr=None):
```

Immediately inside the function body (before the loop), add:

```python
    thr = thr or Thresholds()
```

Replace the per-step loop (lines 116-122):

```python
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        for dd, vv, mm, cxy in zip(d, V, m, cube_xy):
            per_step_rows.append({"episode": i, "outcome": ep["outcome"],
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm),
                                  "cube_x": float(cxy[0]),
                                  "cube_y": float(cxy[1])})
```

with:

```python
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        region = _region(ep, thr)
        for ti, (dd, vv, mm, cxy, rg) in enumerate(
                zip(d, V, m, cube_xy, region)):
            per_step_rows.append({"episode": i, "t": int(ti),
                                  "outcome": ep["outcome"],
                                  "region": str(rg),
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm),
                                  "cube_x": float(cxy[0]),
                                  "cube_y": float(cxy[1])})
```

Note `_region(ep, thr)` requires `ep["transport_mask"]` and `ep["grip"]`, both already present on FB episode dicts (line 115 uses `ep["transport_mask"]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (new test + existing tests; if an existing test calls `probe_value_landscape` positionally it still works since `thr` is keyword-defaulted).

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(representation_profile): persist per-step t + region in value_steps"
```

---

### Task 3: `evals/dataset_support.py` — decode offline cube xy + KDE

**Files:**
- Create: `evals/dataset_support.py`
- Test: `tests/test_dataset_support.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_support.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import dataset_support as ds


def _make_buffer(tmp_path, cube_xy, n_ep=3):
    """Write npz episodes whose physics[:, 7:9] == cube_xy, plus
    decoy observation columns that do NOT match cube_xy."""
    buf = tmp_path / "buffer"
    buf.mkdir()
    T = len(cube_xy)
    for e in range(n_ep):
        phys = np.zeros((T, 21), np.float32)
        phys[:, 7:9] = cube_xy
        obs = np.random.RandomState(e).randn(T, 28).astype(np.float32)
        np.savez(buf / f"episode_{e:06d}_{T}.npz",
                 observation=obs, physics=phys,
                 action=np.zeros((T, 5), np.float32),
                 reward=np.zeros((T, 1), np.float32),
                 discount=np.ones((T, 1), np.float32))
    return buf


def test_decode_finds_matching_slice(tmp_path):
    ref = np.random.RandomState(0).uniform(-0.3, 0.5, size=(200, 2))
    buf = _make_buffer(tmp_path, ref[:50])
    xy = ds.dataset_cube_xy(buf, ref_cube_xy=ref, n_files=10)
    assert xy.shape[1] == 2
    assert xy[:, 0].min() >= ref[:, 0].min() - 0.2
    assert xy[:, 0].max() <= ref[:, 0].max() + 0.2


def test_decode_raises_when_no_slice_matches(tmp_path):
    ref = np.full((100, 2), 999.0)          # nothing in buffer matches
    buf = _make_buffer(tmp_path, np.zeros((50, 2)))
    with pytest.raises(ValueError, match="no cube-xy slice"):
        ds.dataset_cube_xy(buf, ref_cube_xy=ref, n_files=10)


def test_support_kde_shape_and_norm():
    rng = np.random.RandomState(1)
    xy = rng.normal(0, 0.1, size=(500, 2))
    gx = np.linspace(-0.5, 0.5, 20)
    gy = np.linspace(-0.5, 0.5, 15)
    k = ds.support_kde(xy, gx, gy)
    assert k.shape == (15, 20)
    assert 0.0 <= k.min() and abs(k.max() - 1.0) < 1e-9


def test_support_kde_histogram_fallback_on_degenerate():
    xy = np.tile([0.1, 0.2], (50, 1))       # singular covariance
    gx = np.linspace(0, 0.3, 8)
    gy = np.linspace(0, 0.3, 8)
    k = ds.support_kde(xy, gx, gy)
    assert k.shape == (8, 8)
    assert k.max() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dataset_support.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.dataset_support'`.

- [ ] **Step 3: Write minimal implementation**

Create `evals/dataset_support.py`:

```python
"""evals/dataset_support.py — decode offline-buffer cube xy + KDE.

Pure (numpy/scipy only). The OGBench cube buffer stores raw
`observation[T,28]` / `physics[T,21]` with no labeled cube field, so
the cube-xy slice is identified by matching candidate 2-column windows
against the cube xy already recorded in value_steps from rollouts
(same physical quantity, same workspace). Fail loud if none matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _hist2d_unit(xy: np.ndarray, bins: int = 24) -> np.ndarray:
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    n = np.linalg.norm(h.ravel())
    return h.ravel() / n if n > 0 else h.ravel()


def _match_score(cand: np.ndarray, ref: np.ndarray) -> float:
    """Range-overlap × 2D-histogram cosine similarity in [0, 1]."""
    def overlap(a, b):
        lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
        span = max(a.max(), b.max()) - min(a.min(), b.min())
        return max(0.0, hi - lo) / span if span > 1e-12 else 0.0

    ro = 0.5 * (overlap(cand[:, 0], ref[:, 0])
                + overlap(cand[:, 1], ref[:, 1]))
    cos = float(np.dot(_hist2d_unit(cand), _hist2d_unit(ref)))
    return ro * cos


def _load_buffer_arrays(buffer_dir, n_files: int):
    files = sorted(Path(buffer_dir).glob("episode_*.npz"))[:n_files]
    if not files:
        raise ValueError(f"no episode_*.npz under {buffer_dir}")
    phys, obs = [], []
    for f in files:
        with np.load(f) as z:
            if "physics" in z:
                phys.append(np.asarray(z["physics"], np.float64))
            key = ("observation" if "observation" in z
                   else "observations" if "observations" in z else "obs")
            obs.append(np.asarray(z[key], np.float64))
    return (np.concatenate(phys, 0) if phys else None,
            np.concatenate(obs, 0))


def dataset_cube_xy(buffer_dir, ref_cube_xy: np.ndarray,
                    n_files: int = 200,
                    min_score: float = 0.5) -> np.ndarray:
    """Return decoded offline cube xy `[N,2]`.

    Scans consecutive 2-column windows of `physics` then `observation`,
    scoring each against `ref_cube_xy`; returns the best window if its
    score >= `min_score`, else raises ValueError.
    """
    ref = np.asarray(ref_cube_xy, np.float64).reshape(-1, 2)
    phys, obs = _load_buffer_arrays(buffer_dir, n_files)

    best, best_s = None, -1.0
    for src in (phys, obs):
        if src is None:
            continue
        for j in range(src.shape[1] - 1):
            cand = src[:, j:j + 2]
            if np.ptp(cand[:, 0]) < 1e-9 and np.ptp(cand[:, 1]) < 1e-9:
                continue
            s = _match_score(cand, ref)
            if s > best_s:
                best, best_s = cand, s
    if best is None or best_s < min_score:
        raise ValueError(
            f"no cube-xy slice in buffer matched rollout cube xy "
            f"(best score {best_s:.3f} < {min_score})")
    return np.ascontiguousarray(best)


def support_kde(xy: np.ndarray, grid_x: np.ndarray,
                grid_y: np.ndarray) -> np.ndarray:
    """Gaussian-KDE density on the (grid_y, grid_x) mesh, normalized to
    [0,1]. Falls back to a normalized 2D histogram if KDE is singular.
    Returns `[len(grid_y), len(grid_x)]`."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.vstack([gx.ravel(), gy.ravel()])
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(xy.T)
        z = kde(pts).reshape(gx.shape)
    except Exception:
        h, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1],
            bins=[len(grid_x), len(grid_y)],
            range=[[grid_x.min(), grid_x.max()],
                   [grid_y.min(), grid_y.max()]])
        z = h.T
    m = z.max()
    return z / m if m > 0 else z
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dataset_support.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add evals/dataset_support.py tests/test_dataset_support.py
git commit -m "feat(dataset_support): decode offline cube xy + KDE with fail-loud verify"
```

---

### Task 4: Pure seam `_phase_action_fields` in the aggregate

**Files:**
- Modify: `scripts/gciql_profile_aggregate.py` (add function near `_bin_vz_grid`)
- Test: `tests/test_phase_action_scene.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase_action_scene.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile_aggregate as ga


def test_phase_action_fields_basic():
    # one episode: cube moves +x by 0.1 each step in 'transport'
    df = pd.DataFrame({
        "episode": [0, 0, 0],
        "t": [0, 1, 2],
        "cube_x": [0.0, 0.1, 0.2],
        "cube_y": [0.0, 0.0, 0.0],
        "Vz": [-1.0, 0.0, 1.0],
    })
    grid = (np.linspace(-0.1, 0.3, 9), np.linspace(-0.2, 0.2, 9))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert vz.shape == (8, 8)            # 40x40 default? here ny-1,nx-1
    assert np.nanmax(vz) > np.nanmin(vz)
    # net displacement is +x, ~0 y
    us, vs = uv
    assert np.nanmean(us[np.isfinite(us)]) > 0
    assert abs(np.nanmean(vs[np.isfinite(vs)])) < 1e-6
    assert counts.sum() >= 1


def test_phase_action_fields_empty():
    df = pd.DataFrame(columns=["episode", "t", "cube_x", "cube_y", "Vz"])
    grid = (np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert np.isnan(vz).all()
    assert counts.sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_phase_action_scene.py -v`
Expected: FAIL — `AttributeError: module 'scripts.gciql_profile_aggregate' has no attribute '_phase_action_fields'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/gciql_profile_aggregate.py`, add after `_bin_vz_grid` (after line ~143):

```python
def _phase_action_fields(df, grid, n_arrow: int = 12, n_min: int = 5):
    """Pure: (Vz_grid, (U,V), counts) for one panel's rows.

    `df` has columns episode,t,cube_x,cube_y,Vz (already filtered to one
    task/method/outcome/region). `grid` = (x_edges, y_edges). Vz_grid is
    the per-cell mean of Vz on the fine grid (NaN where empty). (U,V) is
    a coarse `n_arrow x n_arrow` mean of per-step cube displacement
    (NaN where < n_min samples). counts is the coarse sample count.
    """
    xe, ye = grid
    nx, ny = len(xe) - 1, len(ye) - 1
    vz = np.full((ny, nx), np.nan)
    if not df.empty:
        gx = np.clip(np.digitize(df["cube_x"], xe) - 1, 0, nx - 1)
        gy = np.clip(np.digitize(df["cube_y"], ye) - 1, 0, ny - 1)
        tmp = pd.DataFrame({"gx": gx, "gy": gy,
                            "Vz": df["Vz"].to_numpy()})
        for (i, j), s in tmp.groupby(["gx", "gy"])["Vz"]:
            vz[j, i] = float(s.mean())

    axe = np.linspace(xe[0], xe[-1], n_arrow + 1)
    aye = np.linspace(ye[0], ye[-1], n_arrow + 1)
    U = np.full((n_arrow, n_arrow), np.nan)
    Vv = np.full((n_arrow, n_arrow), np.nan)
    counts = np.zeros((n_arrow, n_arrow), int)
    if not df.empty:
        d = df.sort_values(["episode", "t"])
        dx = d.groupby("episode")["cube_x"].diff().shift(-1)
        dy = d.groupby("episode")["cube_y"].diff().shift(-1)
        ok = dx.notna() & dy.notna()
        cx, cy = d["cube_x"][ok], d["cube_y"][ok]
        ai = np.clip(np.digitize(cx, axe) - 1, 0, n_arrow - 1)
        aj = np.clip(np.digitize(cy, aye) - 1, 0, n_arrow - 1)
        agg = pd.DataFrame({"ai": ai, "aj": aj,
                            "dx": dx[ok].to_numpy(),
                            "dy": dy[ok].to_numpy()})
        for (i, j), s in agg.groupby(["ai", "aj"]):
            counts[j, i] = len(s)
            if len(s) >= n_min:
                U[j, i] = float(s["dx"].mean())
                Vv[j, i] = float(s["dy"].mean())
    return vz, (U, Vv), counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_phase_action_scene.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gciql_profile_aggregate.py tests/test_phase_action_scene.py
git commit -m "feat(gciql_aggregate): pure _phase_action_fields seam"
```

---

### Task 5: Renderer `_phase_action_scene()` + wire into aggregate

**Files:**
- Modify: `scripts/gciql_profile_aggregate.py` (add `_phase_action_scene`, call it in `_comparison`)
- Test: `tests/test_phase_action_scene.py` (smoke)

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_phase_action_scene.py`:

```python
def test_phase_action_scene_smoke(tmp_path, monkeypatch):
    # minimal value_steps for FB & GCIQL, 1 task, both outcomes/regions
    def _vs(seed_root, method_seed):
        d = Path(seed_root) / "s1_final"
        d.mkdir(parents=True)
        rows = []
        for oc in ("success", "transport_fail"):
            for rg in ("reach", "lift", "transport"):
                for t in range(4):
                    rows.append(dict(task="task1", episode=0, t=t,
                                     outcome=oc, region=rg, d=0.5,
                                     V=float(t), transport=(rg ==
                                     "transport"),
                                     cube_x=0.1 + 0.05 * t,
                                     cube_y=0.0))
        pd.DataFrame(rows).to_parquet(d / "value_steps.parquet")

    fb_root = tmp_path / "fb"
    gc_root = tmp_path / "gc"
    _vs(fb_root, 1)
    _vs(gc_root, 1)
    monkeypatch.setattr(ga, "_decode_support_xy",
                        lambda *a, **k: np.random.RandomState(0)
                        .uniform(0, 0.4, size=(200, 2)))
    out = tmp_path / "phase_scene"
    ga._phase_action_scene(out, str(gc_root), str(fb_root),
                           tasks=["task1"])
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["task1__FB.png", "task1__GCIQL.png"]
    assert (out / "_index.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_phase_action_scene.py::test_phase_action_scene_smoke -v`
Expected: FAIL — `AttributeError: ... has no attribute '_phase_action_scene'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/gciql_profile_aggregate.py` add a thin support-decode wrapper and the renderer. Add near the other imports inside a function-local scope is fine; here add module-level helpers after `_phase_action_fields`:

```python
def _decode_support_xy(gciql_root, fb_seed_root):
    """Decode dataset cube xy once, cached to
    analysis/dataset_support_cube_xy.npy."""
    from evals.dataset_support import dataset_cube_xy
    cache = REPO_ROOT / "analysis" / "dataset_support_cube_xy.npy"
    if cache.exists():
        return np.load(cache)
    # ref = rollout cube xy already in value_steps (any method)
    ref = None
    for rt in (gciql_root, fb_seed_root):
        if rt is None:
            continue
        fr = [pd.read_parquet(p) for p in
              sorted(Path(rt).glob("s*_final/value_steps.parquet"))]
        if fr:
            cat = pd.concat(fr, ignore_index=True)
            ref = cat[["cube_x", "cube_y"]].to_numpy()
            break
    if ref is None:
        raise ValueError("no value_steps to derive ref cube xy")
    buf = REPO_ROOT / "datasets" / "cube-single-play-v0" / "buffer"
    xy = dataset_cube_xy(buf, ref_cube_xy=ref)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, xy)
    return xy


REGIONS = [("reach", "approach"), ("lift", "contact"),
           ("transport", "transport")]


def _phase_action_scene(out_dir, gciql_root, fb_seed_root,
                         tasks=None) -> None:
    """10 figures (task x method); rows {success,transport_fail},
    cols {approach,contact,transport}. Pure data + matplotlib; no
    model loading. Skips a method with no value_steps (warns)."""
    import json

    import matplotlib.image as mpimg
    from evals.scene_calib import world_to_px

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _seed_concat(rt):
        if rt is None:
            return None
        fr = [pd.read_parquet(p) for p in
              sorted(Path(rt).glob("s*_final/value_steps.parquet"))]
        return pd.concat(fr, ignore_index=True) if fr else None

    src = {"FB": _seed_concat(fb_seed_root),
           "GCIQL": _seed_concat(gciql_root)}
    src = {k: v for k, v in src.items() if v is not None
           and "region" in v.columns}
    if not src:
        print("  [warn] _phase_action_scene: no value_steps w/ region")
        return
    tasks = tasks or sorted(
        set().union(*[set(v["task"].unique()) for v in src.values()]))

    sx = REPO_ROOT / "analysis" / "scene"
    scene_png, calib = sx / "topdown.png", sx / "calib.json"
    have_scene = scene_png.exists() and calib.exists()
    if have_scene:
        cal = json.loads(calib.read_text())
        ws = cal["workspace"]
        bg = mpimg.imread(str(scene_png))

    support_xy = _decode_support_xy(gciql_root, fb_seed_root)
    written = []
    for task in tasks:
        for method, vs in src.items():
            sub_all = vs[vs.task == task]
            if sub_all.empty:
                continue
            mu, sd = sub_all["V"].mean(), sub_all["V"].std(ddof=0)
            sd = sd if sd > 1e-9 else 1.0
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for r, oc in enumerate(("success", "transport_fail")):
                for c, (rg, lbl) in enumerate(REGIONS):
                    ax = axes[r, c]
                    d = sub_all[(sub_all.outcome == oc)
                                & (sub_all.region == rg)].copy()
                    d["Vz"] = (d["V"] - mu) / sd
                    if have_scene:
                        ax.imshow(bg, extent=[0, cal["img_w"],
                                              cal["img_h"], 0])
                        ax.set_xlim(0, cal["img_w"])
                        ax.set_ylim(cal["img_h"], 0)
                        xe = np.linspace(ws["xmin"], ws["xmax"], 41)
                        ye = np.linspace(ws["ymin"], ws["ymax"], 41)
                    else:
                        xe = np.linspace(-0.5, 0.5, 41)
                        ye = np.linspace(-0.5, 0.5, 41)
                    ax.axis("off")
                    ax.set_title(f"{oc} — {lbl} (n={len(d)})",
                                 fontsize=9)
                    if len(d) < 5:
                        continue
                    vz, (U, Vv), _ = _phase_action_fields(
                        d, (xe, ye))
                    if have_scene:
                        p00 = world_to_px(ws["xmin"], ws["ymin"], cal)
                        p11 = world_to_px(ws["xmax"], ws["ymax"], cal)
                        ext = [p00[0], p11[0], p00[1], p11[1]]
                    else:
                        ext = [xe[0], xe[-1], ye[0], ye[-1]]
                    ax.imshow(np.ma.masked_invalid(vz),
                              origin="lower", extent=ext,
                              cmap="viridis", alpha=0.55,
                              aspect="auto")
                    kx = 0.5 * (xe[:-1] + xe[1:])
                    ky = 0.5 * (ye[:-1] + ye[1:])
                    kz = __import__("evals.dataset_support",
                                    fromlist=["support_kde"]
                                    ).support_kde(support_xy, kx, ky)
                    ax.contour(
                        np.linspace(ext[0], ext[1], len(kx)),
                        np.linspace(ext[2], ext[3], len(ky)),
                        kz, levels=[0.2, 0.5, 0.8],
                        colors="grey", linewidths=0.6, alpha=0.7)
                    ac = np.linspace(ext[0], ext[1], U.shape[1])
                    ar = np.linspace(ext[2], ext[3], U.shape[0])
                    AX, AY = np.meshgrid(ac, ar)
                    ax.quiver(AX, AY, U, -Vv, color="red",
                              scale=2.0, width=0.004)
            fig.suptitle(f"{task} — {method}: phase value/action "
                         f"fields", fontsize=13)
            fig.tight_layout()
            fp = out_dir / f"{task}__{method}.png"
            fig.savefig(fp, dpi=120)
            plt.close(fig)
            written.append(fp.name)
    (out_dir / "_index.md").write_text(
        "# phase_scene\n\n"
        + "\n".join(f"- {n}" for n in sorted(written)) + "\n")
    print(f"[phase_action_scene] wrote {len(written)} figs -> {out_dir}")
```

Then wire it into `_comparison` next to the existing `_value_scene` call (after line ~388 where `_value_scene(...)` is invoked):

```python
    _phase_action_scene(cdir / "phase_scene", gciql_root, fb_seed_root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_phase_action_scene.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (66 prior + the new tests; the pre-existing matplotlib boxplot warning is unrelated).

- [ ] **Step 6: Commit**

```bash
git add scripts/gciql_profile_aggregate.py tests/test_phase_action_scene.py
git commit -m "feat(gciql_aggregate): phase-conditioned value/action scene (10 figs)"
```

---

### Task 6: Re-profile both methods + end-to-end verification

**Files:** none (operational); produces refreshed `value_steps.parquet` and the 10 figures.

- [ ] **Step 1: Re-profile GCIQL (JAX venv)**

For each of the 10 GCIQL seed dirs under
`results/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0XX_*`,
run the profiler so `value_steps.parquet` gains `t`/`region` (use the
JAX venv python the project uses for `gciql_profile.py`; defaults
`Thresholds()` — do not pass non-default `eps_reach/delta_lift/tau_grip`):

```bash
for d in results/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd*; do
  <jax-venv-python> scripts/gciql_profile.py --run-dir "$d" \
    --step 1000000 --out "analysis/gciql_profile/$(basename $d)_final"
done
```

Expected: each output dir has `value_steps.parquet`; verify columns:

```bash
.venv/bin/python -c "import pandas as pd,glob; \
df=pd.read_parquet(sorted(glob.glob('analysis/gciql_profile/sd*_final/value_steps.parquet'))[0]); \
print(sorted(df.columns)); print(df['region'].value_counts())"
```
Expected: columns include `region`, `t`; region values ⊆ {reach,lift,transport}.

- [ ] **Step 2: Re-profile FB (torch host)**

Re-run `scripts/representation_profile.py` for the FB `s*_final` seed
dirs that feed `fb_seed_root` (same invocation the repo already uses for
that script), confirming refreshed `value_steps.parquet` has `t`/`region`.

```bash
.venv/bin/python -c "import pandas as pd,glob; \
fs=sorted(glob.glob('<fb_seed_root>/s*_final/value_steps.parquet')); \
df=pd.read_parquet(fs[0]); print(sorted(df.columns))"
```
Expected: columns include `region`, `t`.

- [ ] **Step 3: Run the aggregate end-to-end**

Run the existing aggregate entrypoint that calls `_comparison`
(`scripts/gciql_profile_aggregate.py` `main`/`aggregate` with the same
`--root` / `fb_seed_root` it already uses). Confirm:

```bash
ls analysis/gciql_profile/aggregate/comparison/phase_scene/*.png | wc -l
```
Expected: `10` (task1..5 × {FB,GCIQL}); plus `_index.md`.

- [ ] **Step 4: Raw spot-check one panel**

```bash
.venv/bin/python - <<'PY'
import pandas as pd, glob, numpy as np
fs=sorted(glob.glob('analysis/gciql_profile/sd*_final/value_steps.parquet'))
df=pd.concat([pd.read_parquet(f) for f in fs])
d=df[(df.task=='task1')&(df.outcome=='transport_fail')&(df.region=='transport')]
print('rows', len(d), 'mean cube dx per step:',
      d.sort_values(['episode','t']).groupby('episode')['cube_x']
       .diff().mean())
PY
```
Expected: non-empty; printed mean dx is finite — sanity that arrows have data.

- [ ] **Step 5: Commit any regenerated tracked artifacts**

Only code/specs/plans are tracked (analysis/ is untracked per repo
convention). If `git status` shows only untracked `analysis/` outputs,
there is nothing to commit here; record completion in the task tracker.

```bash
git status --short
```
Expected: no tracked changes (figures live under untracked `analysis/`).

---

### Task 7: Paper wiring (explicit, optional — only on user go-ahead)

**Files:**
- Copy: chosen subset → `PAPER/fb-cube-failure/figures/`
- Modify: `PAPER/fb-cube-failure/fb_cube_failure.tex`

- [ ] **Step 1: Select the diagnostic subset**

With the user, pick the most diagnostic figures (recommend the
transport-phase FB vs GCIQL contrast for 1–2 representative tasks).
Do not embed all 10 (too heavy for the note).

- [ ] **Step 2: Copy and embed**

```bash
cp analysis/gciql_profile/aggregate/comparison/phase_scene/<chosen>.png \
   PAPER/fb-cube-failure/figures/
```

Add a `figure` block under the existing
`\subsection{Characterizing FB's Losing Strategy}` referencing the
copied files with a caption derived from the spec's "Final Figure
Caption". Keep `\label{fig:phase_action_scene}`.

- [ ] **Step 3: Sanity-check references**

```bash
cd PAPER/fb-cube-failure && \
grep -o 'figures/[A-Za-z0-9_]*\.png' fb_cube_failure.tex | sort -u | \
while read f; do test -f "$f" || echo "MISSING $f"; done
```
Expected: no `MISSING` lines. (No LaTeX toolchain on host — cannot compile; verify by inspection as in prior paper edits.)

- [ ] **Step 4: (Per repo convention) PAPER/ is untracked**

`PAPER/` is not version-controlled in this repo; no commit. Inform the
user the figures/tex were updated on disk.

---

## Self-Review

**Spec coverage:**
- Recorder change (`t`,`region`, Thresholds() pinned) → Tasks 1, 2 ✓
- `dataset_support.py` decode + fail-loud verify + KDE/hist fallback → Task 3 ✓
- Pure `_phase_action_fields` seam → Task 4 ✓
- `_phase_action_scene` renderer (per task×method, 2×3, z-score per
  method per task, star/contour/quiver, sparse handling, schematic bg) →
  Task 5 ✓
- Decoded-xy cached to `analysis/dataset_support_cube_xy.npy` → Task 5
  `_decode_support_xy` ✓
- Re-profile both methods + verification + raw spot-check → Task 6 ✓
- Paper wiring as separate explicit step → Task 7 ✓
- Testing files (`test_dataset_support`, extend `test_gciql_profile`,
  `test_phase_action_scene`, full suite green) → Tasks 1,3,4,5 ✓

**Placeholder scan:** Task 6 uses `<jax-venv-python>` / `<fb_seed_root>`
deliberately — these are environment paths the executor must supply from
the repo's existing profiling invocation (the JAX venv path and the FB
seed root are not in the spec and vary by host); flagged, not silent
"TODO". All code steps contain full code.

**Type consistency:** `_phase_action_fields(df, grid, n_arrow, n_min) ->
(vz, (U,V), counts)` is defined in Task 4 and consumed with the same
signature in Task 5. `dataset_cube_xy(buffer_dir, ref_cube_xy, n_files,
min_score)` and `support_kde(xy, grid_x, grid_y)` defined in Task 3,
consumed in Task 5. `_region(ep, thr)` (real name) used in Tasks 1–2.
`value_steps` schema (`task,episode,t,outcome,region,d,V,transport,
cube_x,cube_y`) consistent across Tasks 1,2,5,6.

One open item the executor must resolve from the environment (not a plan
gap): the exact JAX-venv interpreter and FB `fb_seed_root` path used by
the project's existing profiling runs (Task 6 Steps 1–2).
