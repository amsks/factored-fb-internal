# FB-vs-GCIQL Value-Over-Scene Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce four FB/GCIQL × success/transport-fail value heatmaps composited over one shared top-down render of the cube scene, plus a 2×2 montage, and hand-insert them into the paper.

**Architecture:** Add cube (x,y) to per-step `value_steps`; render the scene once with a controlled top-down camera whose world→pixel mapping is a fixed analytic affine (we define the affine and aim the camera to match it; schematic fallback if offscreen render fails on macOS); bin per-method z-scored V over visited cube positions and alpha-composite onto the background.

**Tech Stack:** Python, numpy, pandas, matplotlib (Agg); torch `.venv` (FB), `.venv-jax-cpu` (GCIQL + vendored OGBench render); pytest.

Spec: `docs/superpowers/specs/2026-05-19-fb-gciql-value-scene-overlays-design.md`

---

## Verified codebase facts

- `evals/representation_profile.py` `probe_value_landscape` builds
  per-step rows in a loop (lines 116-119):
  `for dd, vv, mm in zip(d, V, m): per_step_rows.append({"episode": i,
  "outcome": ep["outcome"], "d": float(dd), "V": float(vv),
  "transport": bool(mm)})`. `ep["cube"]` is `[T,3]` float (set by
  `rollout_for_profile`). `obs.shape[0]` may be 0 → `continue` before
  the loop, so cube length matches `d/V/m`.
- `scripts/gciql_profile.py` `episodes_to_frames` builds per-step rows:
  `for dd, vv, mm in zip(d, V, tm): vs_rows.append({"task": ep["task"],
  "episode": ep["episode"], "outcome": outcome, "d": float(dd),
  "V": float(vv), "transport": bool(mm)})`. `ep["cube"]` is `[T,3]`.
  `vs` is built with explicit `columns=[...]`.
- `value_steps` schema (both): `task, episode, outcome, d, V,
  transport`.
- `scripts/gciql_profile_aggregate.py`: `_zscore_v(df)` adds `Vz`;
  `_comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
  g_verd, gciql_root=None, fb_seed_root=None)`; `_seed_concat` helper
  inside it loads `s*_final/<name>.parquet`. Module imports `np`,
  `pd`, `plt`, `Path`.
- `aggregate(root, out, fb_aggregate=None, fb_seed_root=None)` calls
  `_comparison(..., gciql_root=root, fb_seed_root=fb_seed_root)`.
- Tasks per cube: task1..task5 with `init_xyzs`/`goal_xyzs` from
  `env.unwrapped.cur_task_info`; workspace x∈[0.30,0.55],
  y∈[−0.25,0.25] (cube z≈0.02).
- `.venv-jax-cpu` runs `scripts/gciql_profile.py`; FB uses `.venv`
  with `MUJOCO_GL=glfw`. Sweeps: FB ~2.1 min, GCIQL ~5.6 min.
- FB sweep cmd:
  `.venv/bin/python scripts/representation_profile.py --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt --out analysis/representation_profile/sN_final --data-path datasets --device cpu --mujoco-gl glfw --n-episodes 10`
  looped over `results/Factored-FB-cube-run/*__fb_flowbc__s{3,4,5,6,7,8,10}`.
- GCIQL sweep cmd:
  `.venv-jax-cpu/bin/python scripts/gciql_profile.py --run-dir <sd> --step 1000000 --out analysis/gciql_profile/sN_final --tasks 1,2,3,4,5 --n-episodes 10 --dataset-path datasets/cube-single-play-v0`
  looped over `RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0*_*`.

---

## Task 1: cube (x,y) in `value_steps`

**Files:**
- Modify: `evals/representation_profile.py`
- Modify: `scripts/gciql_profile.py`
- Test: `tests/test_gciql_profile.py`, `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing schema assertions**

In `tests/test_gciql_profile.py`, in
`test_episodes_to_frames_t1_t4_funnel`, change:

```python
    assert set(vs.columns) == {"task", "episode", "outcome", "d", "V",
                               "transport"}
```

to:

```python
    assert set(vs.columns) == {"task", "episode", "outcome", "d", "V",
                               "transport", "cube_x", "cube_y"}
    # cube_x/cube_y equal the source cube positions
    assert vs["cube_x"].iloc[0] == eps[0]["cube"][0][0]
    assert vs["cube_y"].iloc[0] == eps[0]["cube"][0][1]
```

In `tests/test_representation_profile.py`, find the test that calls
`probe_value_landscape` and unpacks `per_ep, per_step` (named
`test_value_landscape_rho_success_vs_fail`) and after the existing
`assert set(per_step.columns) >= {"outcome", "d", "V"}` line add:

```python
    assert {"cube_x", "cube_y"} <= set(per_step.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k episodes_to_frames tests/test_representation_profile.py -k value_landscape_rho -q`
Expected: FAIL — `cube_x`/`cube_y` not in columns / KeyError.

- [ ] **Step 3: Emit cube_x/cube_y from FB `probe_value_landscape`**

In `evals/representation_profile.py`, replace this block (lines
114-119):

```python
        d = np.asarray(ep["d"], np.float64)
        m = np.asarray(ep["transport_mask"], bool)
        for dd, vv, mm in zip(d, V, m):
            per_step_rows.append({"episode": i, "outcome": ep["outcome"],
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm)})
```

with:

```python
        d = np.asarray(ep["d"], np.float64)
        m = np.asarray(ep["transport_mask"], bool)
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        for dd, vv, mm, cxy in zip(d, V, m, cube_xy):
            per_step_rows.append({"episode": i, "outcome": ep["outcome"],
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm),
                                  "cube_x": float(cxy[0]),
                                  "cube_y": float(cxy[1])})
```

- [ ] **Step 4: Emit cube_x/cube_y from GCIQL `episodes_to_frames`**

In `scripts/gciql_profile.py`, replace this block:

```python
        for dd, vv, mm in zip(d, V, tm):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "outcome": outcome, "d": float(dd),
                            "V": float(vv), "transport": bool(mm)})
```

with:

```python
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        for dd, vv, mm, cxy in zip(d, V, tm, cube_xy):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "outcome": outcome, "d": float(dd),
                            "V": float(vv), "transport": bool(mm),
                            "cube_x": float(cxy[0]),
                            "cube_y": float(cxy[1])})
```

and update the explicit `columns=` list in the same function from:

```python
    vs = pd.DataFrame(vs_rows, columns=["task", "episode", "outcome",
                                        "d", "V", "transport"])
```

to:

```python
    vs = pd.DataFrame(vs_rows, columns=["task", "episode", "outcome",
                                        "d", "V", "transport",
                                        "cube_x", "cube_y"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py tests/test_representation_profile.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/representation_profile.py scripts/gciql_profile.py tests/test_gciql_profile.py tests/test_representation_profile.py
git commit -m "feat(profile): persist per-step cube_x/cube_y in value_steps"
```

---

## Task 2: world→pixel calibration + scene render

**Files:**
- Create: `evals/scene_calib.py`
- Create: `scripts/render_cube_scene.py`
- Test: `tests/test_scene_calib.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_calib.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.scene_calib import make_calib, world_to_px


def test_calib_maps_workspace_corners_to_pixel_box():
    c = make_calib(xmin=0.30, xmax=0.55, ymin=-0.25, ymax=0.25,
                    img_w=600, img_h=600, margin=50)
    # x-min/x-max -> horizontal pixel box [margin, img_w-margin]
    px0, _ = world_to_px(0.30, 0.0, c)
    px1, _ = world_to_px(0.55, 0.0, c)
    assert round(px0) == 50 and round(px1) == 550
    # world y maps into the vertical box, inverted (image y grows down)
    _, py0 = world_to_px(0.40, -0.25, c)
    _, py1 = world_to_px(0.40, 0.25, c)
    assert round(py0) == 550 and round(py1) == 50
    # center maps to image center
    pcx, pcy = world_to_px(0.425, 0.0, c)
    assert round(pcx) == 300 and round(pcy) == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scene_calib.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.scene_calib'`.

- [ ] **Step 3: Create `evals/scene_calib.py`**

```python
"""evals/scene_calib.py — fixed analytic world(x,y)->pixel mapping for
the top-down cube scene. We DEFINE this affine (workspace box ->
inset pixel box) and aim the MuJoCo camera to match it; this keeps the
overlay alignment exact and render-independent. torch/jax/mujoco-free."""

from __future__ import annotations

from typing import Dict


def make_calib(xmin: float, xmax: float, ymin: float, ymax: float,
                img_w: int, img_h: int, margin: int = 50) -> Dict:
    """Workspace [xmin,xmax]x[ymin,ymax] (metres) -> pixel box inset by
    `margin`. World +x -> +px (right); world +y -> -py (up), since
    image y grows downward."""
    ax = (img_w - 2 * margin) / (xmax - xmin)
    bx = margin - ax * xmin
    ay = -(img_h - 2 * margin) / (ymax - ymin)
    by = (img_h - margin) - ay * ymin
    return {"ax": ax, "bx": bx, "ay": ay, "by": by,
            "img_w": int(img_w), "img_h": int(img_h),
            "workspace": {"xmin": xmin, "xmax": xmax,
                          "ymin": ymin, "ymax": ymax},
            "margin": int(margin)}


def world_to_px(world_x: float, world_y: float, calib: Dict):
    """(x,y) metres -> (px,py) float pixels under `calib`."""
    px = calib["ax"] * world_x + calib["bx"]
    py = calib["ay"] * world_y + calib["by"]
    return px, py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scene_calib.py -q`
Expected: PASS.

- [ ] **Step 5: Create `scripts/render_cube_scene.py`**

```python
"""scripts/render_cube_scene.py — one top-down background of the cube
scene + calib.json. Tries an offscreen MuJoCo render; on failure
(macOS offscreen GL) writes a schematic top-down canvas instead. The
world->pixel affine is fixed by evals.scene_calib (camera is aimed to
match it), so downstream overlays align either way.

Run in .venv-jax-cpu (vendored OGBench cube env has render())."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evals.scene_calib import make_calib, world_to_px  # noqa: E402

XMIN, XMAX, YMIN, YMAX = 0.30, 0.55, -0.25, 0.25
IMG_W = IMG_H = 600
MARGIN = 50


def _try_mujoco_topdown(out_png: Path) -> bool:
    """Render a top-down RGB frame via the vendored OGBench cube env.
    Returns True on success, False if offscreen GL is unavailable."""
    try:
        ogb = REPO_ROOT / "third_party" / "ogbench" / "impls"
        if str(ogb) not in sys.path:
            sys.path.insert(0, str(ogb))
        import ogbench  # noqa: F401
        import mujoco

        env = ogbench.make_env_and_datasets("cube-single-play-v0",
                                             env_only=True)
        env.reset(options=dict(task_id=1))
        m = env.unwrapped.model
        d = env.unwrapped.data
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.5 * (XMIN + XMAX), 0.5 * (YMIN + YMAX), 0.05]
        cam.elevation = -89.9
        cam.azimuth = 90.0
        cam.distance = 1.2
        ren = mujoco.Renderer(m, IMG_H, IMG_W)
        ren.update_scene(d, camera=cam)
        img = ren.render()
        ren.close()
        env.close()
        plt.imsave(str(out_png), np.asarray(img))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [render_cube_scene] mujoco offscreen failed: {e}")
        return False


def _schematic_topdown(out_png: Path) -> None:
    """Fallback: schematic top-down table-plane canvas (clearly not
    photoreal). Table workspace rectangle + the five task goal markers."""
    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((MARGIN, MARGIN),
                               IMG_W - 2 * MARGIN, IMG_H - 2 * MARGIN,
                               facecolor="#d9d2c5", edgecolor="#555"))
    ax.text(IMG_W / 2, 24, "SCHEMATIC (offscreen render unavailable)",
            ha="center", fontsize=8, color="#555")
    fig.savefig(str(out_png))
    plt.close(fig)


def main() -> None:
    out = REPO_ROOT / "analysis" / "scene"
    out.mkdir(parents=True, exist_ok=True)
    png = out / "topdown.png"
    photoreal = _try_mujoco_topdown(png)
    if not photoreal:
        _schematic_topdown(png)
    calib = make_calib(XMIN, XMAX, YMIN, YMAX, IMG_W, IMG_H, MARGIN)
    calib["photoreal"] = bool(photoreal)
    (out / "calib.json").write_text(json.dumps(calib, indent=2))

    # smoke validation: known task1 init/goal must land in-image
    init_xy = (0.425, 0.10)
    goal_xy = (0.425, -0.10)
    for name, (wx, wy) in (("init", init_xy), ("goal", goal_xy)):
        px, py = world_to_px(wx, wy, calib)
        ok = 0 <= px <= IMG_W and 0 <= py <= IMG_H
        print(f"  [render_cube_scene] {name} ({wx},{wy}) -> "
              f"({px:.1f},{py:.1f}) in-image={ok}")
    print(f"[render_cube_scene] photoreal={photoreal} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add evals/scene_calib.py scripts/render_cube_scene.py tests/test_scene_calib.py
git commit -m "feat(scene): world->pixel calib + top-down render (schematic fallback)"
```

---

## Task 3: value-over-scene overlay producer

**Files:**
- Modify: `scripts/gciql_profile_aggregate.py`
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Write failing tests for the pure seams**

Append to `tests/test_gciql_profile.py`:

```python
def test_bin_vz_grid_masks_empty_cells():
    import numpy as np
    from scripts.gciql_profile_aggregate import _bin_vz_grid
    df = pd.DataFrame({
        "cube_x": [0.31, 0.31, 0.54],
        "cube_y": [-0.24, -0.24, 0.24],
        "Vz": [1.0, 3.0, -2.0],
    })
    grid, xe, ye = _bin_vz_grid(df, xmin=0.30, xmax=0.55,
                                ymin=-0.25, ymax=0.25, nbins=5)
    assert grid.shape == (5, 5)
    # the (0.31,-0.24) cell holds mean(1,3)=2.0; far corner holds -2.0
    assert np.isfinite(grid).sum() == 2
    assert np.nanmin(grid) == -2.0 and np.nanmax(grid) == 2.0


def test_medoid_episode_picks_closest_to_mean_path():
    import numpy as np
    from scripts.gciql_profile_aggregate import _medoid_episode
    # ep 0 & 1 near y=0; ep 2 far at y=1 -> medoid is 0 or 1, not 2
    rows = []
    for ep, y in ((0, 0.0), (1, 0.01), (2, 1.0)):
        for t in range(4):
            rows.append({"episode": ep, "cube_x": 0.4 + 0.01 * t,
                         "cube_y": y})
    df = pd.DataFrame(rows)
    assert _medoid_episode(df) in (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k "bin_vz_grid or medoid_episode" -q`
Expected: FAIL — `ImportError: cannot import name '_bin_vz_grid'`.

- [ ] **Step 3: Add the pure helpers + `_value_scene` to the aggregate**

In `scripts/gciql_profile_aggregate.py`, immediately after the
`_zscore_v` function definition, add:

```python
def _bin_vz_grid(df, xmin, xmax, ymin, ymax, nbins):
    """2D mean of df['Vz'] over an (nbins x nbins) cube-(x,y) grid;
    cells with no samples are NaN. Returns (grid[ny,nx], x_edges,
    y_edges). grid[j,i] = mean Vz for x-bin i, y-bin j."""
    xe = np.linspace(xmin, xmax, nbins + 1)
    ye = np.linspace(ymin, ymax, nbins + 1)
    gx = np.clip(np.digitize(df["cube_x"], xe) - 1, 0, nbins - 1)
    gy = np.clip(np.digitize(df["cube_y"], ye) - 1, 0, nbins - 1)
    grid = np.full((nbins, nbins), np.nan)
    tmp = pd.DataFrame({"gx": gx, "gy": gy, "Vz": df["Vz"].to_numpy()})
    for (i, j), s in tmp.groupby(["gx", "gy"])["Vz"]:
        grid[j, i] = float(s.mean())
    return grid, xe, ye


def _medoid_episode(df) -> int:
    """Episode id whose per-step (cube_x,cube_y) path is closest (mean
    L2) to the across-episode mean path (truncated to the shortest)."""
    eps = sorted(df["episode"].unique())
    paths = {e: df[df.episode == e][["cube_x", "cube_y"]].to_numpy()
             for e in eps}
    if not paths:
        return -1
    L = min(len(p) for p in paths.values())
    if L == 0:
        return eps[0]
    stack = np.stack([paths[e][:L] for e in eps], axis=0)  # [E,L,2]
    mean_path = stack.mean(axis=0)                          # [L,2]
    dists = [float(np.linalg.norm(stack[k] - mean_path, axis=1).mean())
             for k in range(len(eps))]
    return int(eps[int(np.argmin(dists))])


def _value_scene(cdir, gciql_root, fb_seed_root, scene_png, calib):
    """Four {FB,GCIQL}x{success,transport_fail} translucent V heatmaps
    over the shared top-down scene + medoid trajectory + a 2x2 montage.
    Skipped (with a warning) if inputs are missing."""
    import json

    import matplotlib.image as mpimg
    from evals.scene_calib import world_to_px

    if not (Path(scene_png).exists() and Path(calib).exists()):
        print("  [warn] _value_scene skipped (no scene/calib)")
        return
    cal = json.loads(Path(calib).read_text())
    ws = cal["workspace"]
    bg = mpimg.imread(str(scene_png))

    def _seed_concat(rt, name):
        if rt is None:
            return None
        fr = [pd.read_parquet(p)
              for p in sorted(Path(rt).glob(f"s*_final/{name}.parquet"))]
        return pd.concat(fr, ignore_index=True) if fr else None

    sources = {"FB": _seed_concat(fb_seed_root, "value_steps"),
               "GCIQL": _seed_concat(gciql_root, "value_steps")}
    panels = []
    for method, vs in sources.items():
        if vs is None or "cube_x" not in vs.columns:
            print(f"  [warn] _value_scene: no value_steps for {method}")
            continue
        vsz = _zscore_v(vs)
        for oc in ("success", "transport_fail"):
            sub = vsz[(vsz.outcome == oc) & (vsz.transport)]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(bg, extent=[0, cal["img_w"], cal["img_h"], 0])
            ax.set_xlim(0, cal["img_w"]); ax.set_ylim(cal["img_h"], 0)
            ax.axis("off")
            title = f"{method} — {oc}"
            if not sub.empty:
                grid, xe, ye = _bin_vz_grid(
                    sub, ws["xmin"], ws["xmax"], ws["ymin"], ws["ymax"],
                    nbins=40)
                p00 = world_to_px(ws["xmin"], ws["ymin"], cal)
                p11 = world_to_px(ws["xmax"], ws["ymax"], cal)
                ax.imshow(np.ma.masked_invalid(grid), origin="lower",
                          extent=[p00[0], p11[0], p00[1], p11[1]],
                          cmap="viridis", alpha=0.55, aspect="auto")
                med = _medoid_episode(sub)
                mp = sub[sub.episode == med]
                pts = [world_to_px(x, y, cal) for x, y in
                       zip(mp["cube_x"], mp["cube_y"])]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "-", color="red", lw=2.0,
                            label="medoid path")
                    ax.scatter([xs[0]], [ys[0]], c="white",
                               edgecolors="k", s=40, zorder=5)
                ax.legend(loc="lower right", fontsize=7)
            else:
                title += " (no transport steps)"
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            fp = cdir / f"cmp_scene_{method}_{oc}.png"
            fig.savefig(fp, dpi=120)
            plt.close(fig)
            panels.append(fp)

    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for axx, fp in zip(axes.ravel(), panels):
            axx.imshow(mpimg.imread(str(fp)))
            axx.axis("off")
        for axx in list(axes.ravel())[len(panels):]:
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(cdir / "cmp_value_scene.png", dpi=120)
        plt.close(fig)
```

(`np`, `pd`, `plt`, `Path` already imported at module top.)

- [ ] **Step 4: Call `_value_scene` from `_comparison`**

In `scripts/gciql_profile_aggregate.py`, at the very end of the
`_comparison` function body (after the
`(cdir / "comparison.md").write_text(...)` call), add:

```python
    _value_scene(cdir, gciql_root, fb_seed_root,
                 REPO_ROOT / "analysis" / "scene" / "topdown.png",
                 REPO_ROOT / "analysis" / "scene" / "calib.json")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (pure-seam tests green; `_value_scene` is exercised in
Task 4's run, not unit-tested).

- [ ] **Step 6: Commit**

```bash
git add scripts/gciql_profile_aggregate.py tests/test_gciql_profile.py
git commit -m "feat(gciql_aggregate): value-over-scene overlays + montage"
```

---

## Task 4: Full verify, re-run, render, paper, report

**Files:** modify `PAPER/fb-cube-failure/fb_cube_failure.tex` (hand).

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, zero failures.

- [ ] **Step 2: Re-run both sweeps (emit cube_x/cube_y)**

```bash
rm -rf analysis/representation_profile/s*_final
for s in 3 4 5 6 7 8 10; do
  RUN=$(ls -d results/Factored-FB-cube-run/*__fb_flowbc__s${s})
  .venv/bin/python scripts/representation_profile.py \
    --config "$RUN/.hydra/config.yaml" \
    --checkpoint "$RUN/checkpoints/final.pt" \
    --out analysis/representation_profile/s${s}_final \
    --data-path datasets --device cpu --mujoco-gl glfw --n-episodes 10
done
rm -rf analysis/gciql_profile/s*_final
i=0
for RUN in RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0*_*; do
  i=$((i+1))
  .venv-jax-cpu/bin/python scripts/gciql_profile.py \
    --run-dir "$RUN" --step 1000000 \
    --out analysis/gciql_profile/s${i}_final \
    --tasks 1,2,3,4,5 --n-episodes 10 \
    --dataset-path datasets/cube-single-play-v0
done
```

Verify: `.venv/bin/python -c "import pandas as pd,glob;
print('cube_x' in pd.read_parquet(sorted(glob.glob('analysis/representation_profile/s*_final/value_steps.parquet'))[0]).columns and 'cube_x' in pd.read_parquet(sorted(glob.glob('analysis/gciql_profile/s*_final/value_steps.parquet'))[0]).columns)"`
Expected: `True`.

- [ ] **Step 3: Render the scene + aggregate**

```bash
.venv-jax-cpu/bin/python scripts/render_cube_scene.py
.venv/bin/python scripts/gciql_profile_aggregate.py
```

Expected: `analysis/scene/{topdown.png,calib.json}` (note the printed
`photoreal=True|False` and the in-image init/goal check);
`analysis/gciql_profile/aggregate/comparison/` now also has
`cmp_scene_FB_success.png`, `cmp_scene_FB_fail.png`,
`cmp_scene_GCIQL_success.png`, `cmp_scene_GCIQL_fail.png`,
`cmp_value_scene.png`.

- [ ] **Step 4: Copy figures + hand-insert into the paper**

```bash
cp analysis/gciql_profile/aggregate/comparison/cmp_scene_*.png \
   analysis/gciql_profile/aggregate/comparison/cmp_value_scene.png \
   PAPER/fb-cube-failure/figures/
```

Then hand-edit `PAPER/fb-cube-failure/fb_cube_failure.tex`: in the
`\subsection{Baseline Comparison: FB vs. GCIQL}` (after the existing
"What the comparison shows" itemize), insert a `figure` using the
established `\minipage` idiom embedding `figures/cmp_value_scene.png`
(full-width single `\includegraphics[width=\linewidth]`), followed by a
`\paragraph{Characterizing FB's losing strategy.}` whose sentences are
written from the actually-produced overlays + these numbers (read at
report time, not invented): the medoid transport-fail vs success cube
end-points per method, and whether FB's heatmap shows a goal-directed
basin. Do **not** run `build_paper_fragment.py`.

- [ ] **Step 5: Report**

Read the four `cmp_scene_*.png` (and `analysis/scene/calib.json`
`photoreal` flag) and report to the user: render path used (photoreal
vs schematic), what FB's losing overlay shows vs GCIQL's, and the
characterized losing strategy. Do not commit `analysis/`/`PAPER/`
artifacts unless asked.

---

## Self-review

- **Spec coverage:** Component A → Task 1 (cube_x/cube_y both
  producers + schema tests). Component B → Task 2 (`scene_calib`
  pure affine + `render_cube_scene` with mujoco attempt + schematic
  fallback + smoke validation of known points). Component C → Task 3
  (`_bin_vz_grid` mask, `_medoid_episode`, `_value_scene` wired into
  `_comparison`, per-method `_zscore_v`, masked empty cells, montage).
  Component D → Task 4 Step 4 (hand-insert, no autogeneration).
  Execution/risk → Task 4 (re-runs, render path reported, fallback in
  Task 2 code).
- **Placeholder scan:** none — all code complete; render fallback is
  fully implemented (not "handle later"); paper prose is explicitly
  data-derived at report time with the exact quantities named.
- **Type consistency:** `value_steps` schema `+cube_x,cube_y`
  identical in FB (Task 1 Step 3), GCIQL (Step 4), the schema tests
  (Step 1), and the consumers `_bin_vz_grid`/`_value_scene` (Task 3).
  `make_calib`/`world_to_px` signatures defined Task 2 Step 3, used
  identically in `render_cube_scene` (Step 5) and `_value_scene`
  (Task 3 Step 3). `_zscore_v` reused (adds `Vz`, consumed by
  `_bin_vz_grid`). `calib` JSON keys (`ax,bx,ay,by,img_w,img_h,
  workspace{xmin..},margin,photoreal`) consistent across producer and
  consumers. `_value_scene(cdir, gciql_root, fb_seed_root, scene_png,
  calib)` call args match its definition.
