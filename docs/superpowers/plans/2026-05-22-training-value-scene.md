# Training-Value Phase × Outcome Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-data renderer that reproduces the eval `phase_scene` figure on the training states — per task per method, a value-saliency heatmap + cube-flow arrows over the top-down scene, gridded by phase × outcome (success_bound / fail_bound), plus a dataset-support panel.

**Architecture:** New `.venv` script `scripts/training_value_scene.py` (no model loading). Value heatmap from the already-scored multiseed parquets (`p*/fb_values`, `p*/gciql_values`); cube-flow arrows + support from the play buffer; styling/helpers reused from `scripts/gciql_profile_aggregate.py` and `evals.dataset_support`. One new pure helper `flow_step_labels` in `evals/training_value.py` (TDD).

**Tech Stack:** Python 3 (`.venv`), numpy, pandas, matplotlib, scipy (KDE).

**Spec:** `docs/superpowers/specs/2026-05-22-training-value-scene-design.md`

---

## Conventions & ground rules

- Runs under `.venv/bin/python` ([[project-venvs]]). Pure data — no torch/jax.
- Commit only `evals/`, `scripts/`, `tests/`, `docs/`. `analysis/` stays untracked ([[project-untracked-artifacts]]). No Claude attribution in commits ([[feedback-commit-style]]).
- Outcome label identical to the parquets: success_bound iff cube within 4 cm of the 3-D goal within the next 200 steps.

## File structure

- `evals/training_value.py` (modify) — add pure `flow_step_labels`.
- `tests/test_training_value.py` (modify) — unit test for it.
- `scripts/training_value_scene.py` (create) — flow frame + value frame + render + main.

---

## Task 1: Pure per-step flow labeler (`evals/training_value.py`)

**Files:**
- Modify: `evals/training_value.py`
- Test: `tests/test_training_value.py`

- [ ] **Step 1: Write the failing test**

Add to the import line in `tests/test_training_value.py`:

```python
from evals.training_value import (region_labels, cube_to_goal_dist,
                                   phase_spearman_table, horizon_reach_label,
                                   outcome_spearman_table, value_discrimination,
                                   flow_step_labels)
```

Append this test:

```python
def test_flow_step_labels_bounded_reach():
    # one episode, cube approaches goal g at step 3 then leaves
    cube = np.array([[0.5, 0.0, 0.0],
                     [0.4, 0.0, 0.0],
                     [0.31, 0.0, 0.0],
                     [0.30, 0.0, 0.0],   # within 0.04 of g here
                     [0.6, 0.0, 0.0]])
    g = np.array([[0.30, 0.0, 0.0]])     # one goal
    out = flow_step_labels(cube, g, horizon=2, thresh=0.04)  # [T, n_goals]
    # step0 tail[0:2]={0.5,0.4} -> no; step2 tail{0.31,0.30}->0.30 hits -> yes
    assert out.shape == (5, 1)
    assert out[2, 0] and out[3, 0]
    assert not out[0, 0] and not out[1, 0]
    assert not out[4, 0]                 # last step, tail {0.6}, no reach
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_training_value.py::test_flow_step_labels_bounded_reach -v`
Expected: FAIL — `ImportError: cannot import name 'flow_step_labels'`.

- [ ] **Step 3: Implement the helper**

Append to `evals/training_value.py`:

```python
def flow_step_labels(cube_traj: np.ndarray, goals_xyz: np.ndarray,
                     horizon: int, thresh: float) -> np.ndarray:
    """[T, n_goals] bool success-bound per step per goal for ONE episode:
    does the cube reach within `thresh` of each goal within `horizon` future
    steps. `cube_traj` = [T,3] cube xyz; `goals_xyz` = [n_goals,3]."""
    cube_traj = np.asarray(cube_traj, np.float64).reshape(-1, 3)
    goals_xyz = np.asarray(goals_xyz, np.float64).reshape(-1, 3)
    T, G = len(cube_traj), len(goals_xyz)
    out = np.zeros((T, G), dtype=bool)
    for gi in range(G):
        d = np.linalg.norm(cube_traj - goals_xyz[gi], axis=1)  # [T]
        for t in range(T):
            out[t, gi] = horizon_reach_label(d[t:], thresh, horizon)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/training_value.py tests/test_training_value.py
git commit -m "feat(training_value): pure per-step flow_step_labels (bounded reach per goal)"
```

---

## Task 2: Scene renderer script (`scripts/training_value_scene.py`)

**Files:**
- Create: `scripts/training_value_scene.py`

- [ ] **Step 1: Create the script**

Create `scripts/training_value_scene.py` with the content below. It reuses
`_bin_vz_grid`, `_phase_action_fields`, `_nan_gaussian` from
`scripts.gciql_profile_aggregate` and `support_kde` from `evals.dataset_support`;
mirrors the eval `_phase_action_scene` panel styling.

```python
"""scripts/training_value_scene.py — eval phase_scene analog on TRAINING
states. Per task per method (FB policy, GCIQL): value-saliency heatmap (from the
scored multiseed parquets) + cube-flow arrows (from the play buffer) over the
top-down scene, gridded by phase x outcome, + a dataset-support panel. Pure data
+ matplotlib; no model loading. Run under .venv.

Usage:
    .venv/bin/python scripts/training_value_scene.py \
        --root analysis/training_value_multiseed --data-path datasets \
        --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import region_labels, flow_step_labels
from evals.phase_probe import Thresholds
from evals.dataset_support import support_kde
from scripts.gciql_profile_aggregate import (_bin_vz_grid, _phase_action_fields,
                                             _nan_gaussian)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.image as mpimg         # noqa: E402
from matplotlib.cm import get_cmap       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

CUBE_SLICE = slice(14, 17)
GRIP_QPOS_IDX = 6
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"
REGIONS = [("reach", "reach"), ("grasp", "grasp"), ("transport", "transport")]
OUTCOMES = ["success_bound", "fail_bound"]
NBINS, NARROW, LOW_N = 16, 10, 30


def _pool(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for pdir in sorted(d for d in root.glob("p*") if d.is_dir()):
        f = pdir / name
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit(f"no {name} under {root}/p*/")
    return pd.concat(frames, ignore_index=True)


def _value_frame(root: Path) -> pd.DataFrame:
    fb = _pool(root, "fb_values.parquet")[
        ["task", "region", "outcome", "cube_x", "cube_y", "V_policy"]
    ].rename(columns={"V_policy": "V"})
    fb["method"] = "FB policy"
    gq = _pool(root, "gciql_values.parquet")[
        ["task", "region", "outcome", "cube_x", "cube_y", "V"]
    ].copy()
    gq["method"] = "GCIQL"
    return pd.concat([fb, gq], ignore_index=True)


def _table_z(seed, obs_type) -> float:
    from envs.ogbench import create_ogbench_env
    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=seed,
                               obs_type=obs_type)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    tz = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    e0.close()
    return tz


def _flow_frame(data_path, goals_xyz, thr, table_z, horizon, thresh,
                n_eps, seed=0):
    """Long flow DataFrame (task, outcome, region, episode, t, cube_x, cube_y)
    + raw play cube-xy [N,2] for support."""
    files = sorted(glob.glob(str(Path(data_path) / "cube-single-play-v0"
                                  / "buffer" / "episode_*.npz")))
    rng = np.random.default_rng(seed)
    if len(files) > n_eps:
        files = [files[i] for i in sorted(rng.choice(len(files), n_eps,
                                                     replace=False))]
    rows, sup = [], []
    for ei, f in enumerate(files):
        phys = np.asarray(np.load(f)["physics"], np.float32)
        cube = phys[:, CUBE_SLICE].astype(np.float64)
        grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
        region = region_labels(grip, phys[:, 16] - table_z, thr)  # [T]
        oc = flow_step_labels(cube, goals_xyz, horizon, thresh)    # [T, 5]
        sup.append(cube[:, :2])
        T = len(cube)
        for ti in range(goals_xyz.shape[0]):
            df = pd.DataFrame({
                "task": f"task{ti + 1}",
                "outcome": np.where(oc[:, ti], "success_bound", "fail_bound"),
                "region": region, "episode": ei, "t": np.arange(T),
                "cube_x": cube[:, 0], "cube_y": cube[:, 1]})
            rows.append(df)
    return pd.concat(rows, ignore_index=True), np.concatenate(sup)


def _extent(vdf_task):
    xs, ys = vdf_task["cube_x"], vdf_task["cube_y"]
    xlo, xhi = np.percentile(xs, [1, 99])
    ylo, yhi = np.percentile(ys, [1, 99])
    px, py = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
    xlo, xhi, ylo, yhi = xlo - px, xhi + px, ylo - py, yhi + py
    xr, yr = xhi - xlo, yhi - ylo
    if xr < yr:
        pad = 0.5 * (yr - xr); xlo, xhi = xlo - pad, xhi + pad
    elif yr < xr:
        pad = 0.5 * (xr - yr); ylo, yhi = ylo - pad, yhi + pad
    return xlo, xhi, ylo, yhi


def _render(task, method, vdf, fdf, support_xy, anchors, bg_img, out_dir):
    vt = vdf[(vdf.task == task) & (vdf.method == method)]
    if vt.empty:
        return None
    ft = fdf[fdf.task == task]
    xlo, xhi, ylo, yhi = _extent(vt)
    xe = np.linspace(xlo, xhi, NBINS + 1)
    ye = np.linspace(ylo, yhi, NBINS + 1)
    ext = [xlo, xhi, ylo, yhi]
    anc = anchors.get(task, {})
    goal_xy = tuple(anc["goal"]) if "goal" in anc else None
    kx, ky = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    kz = support_kde(support_xy, kx, ky)

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1],
                          wspace=0.06, hspace=0.10)
    axes = np.empty((2, 3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            axes[rr, cc] = fig.add_subplot(gs[rr, cc])
    ax_data = fig.add_subplot(gs[0, 3])

    ac = 0.5 * (np.linspace(xlo, xhi, NARROW + 1)[:-1]
                + np.linspace(xlo, xhi, NARROW + 1)[1:])
    ar = 0.5 * (np.linspace(ylo, yhi, NARROW + 1)[:-1]
                + np.linspace(ylo, yhi, NARROW + 1)[1:])
    AX, AY = np.meshgrid(ac, ar)

    for r, oc in enumerate(OUTCOMES):
        for c, (rg, lbl) in enumerate(REGIONS):
            ax = axes[r, c]
            dv = vt[(vt.outcome == oc) & (vt.region == rg)]
            n = len(dv)
            low = n < LOW_N
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if bg_img is not None:
                ax.imshow(bg_img, extent=ext, aspect="auto", zorder=0,
                          interpolation="bilinear")
            else:
                ax.set_facecolor("0.92")
            ax.set_title(f"{oc} — {lbl} (n={n}{', low' if low else ''})",
                         fontsize=9, color="grey" if low else "black")
            if n >= 5:
                ax.scatter([dv["cube_x"].mean()], [dv["cube_y"].mean()],
                           marker="s", s=170, c="white", edgecolors="black",
                           linewidths=1.4, zorder=6)
            if goal_xy is not None:
                ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=900,
                           c="white", alpha=0.55, zorder=6)
                ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=520,
                           c="gold", edgecolors="black", linewidths=1.6,
                           zorder=7)
            if n < 5:
                continue
            a = 0.35 if low else 1.0
            grid, _, _ = _bin_vz_grid(dv.assign(Vz=dv["V"]),
                                      xlo, xhi, ylo, yhi, NBINS)
            vz = _nan_gaussian(grid, sigma=1.0)
            fin = np.isfinite(vz)
            if fin.any():
                lo, hi = np.percentile(vz[fin], [2, 98])
                if hi - lo < 1e-9:
                    hi = lo + 1.0
                norm = np.clip((vz - lo) / (hi - lo), 0.0, 1.0)
                rgba = get_cmap("Reds")(norm)
                rgba[..., 3] = np.where(fin, norm * (0.55 * a), 0.0)
                ax.imshow(rgba, origin="lower", extent=ext, zorder=2,
                          aspect="auto", interpolation="bilinear")
            df_arrow = ft[(ft.outcome == oc) & (ft.region == rg)]
            if not df_arrow.empty:
                _, (U, Vv), _ = _phase_action_fields(
                    df_arrow.assign(Vz=0.0), (xe, ye), n_arrow=NARROW, n_min=5)
                mag = np.hypot(U, Vv)
                ok = np.isfinite(mag) & (mag > 0)
                if ok.any():
                    un = np.where(ok, U / np.maximum(mag, 1e-9), np.nan)
                    vn = np.where(ok, Vv / np.maximum(mag, 1e-9), np.nan)
                    ax.quiver(AX[ok], AY[ok], un[ok], vn[ok], color="white",
                              edgecolor="black", linewidth=0.5, alpha=a,
                              scale=NARROW * 1.1, width=0.007, headwidth=4,
                              zorder=4, pivot="mid")

    ax_data.set_xlim(xlo, xhi); ax_data.set_ylim(ylo, yhi)
    ax_data.set_aspect("equal"); ax_data.set_xticks([]); ax_data.set_yticks([])
    if bg_img is not None:
        ax_data.imshow(bg_img, extent=ext, aspect="auto", zorder=0,
                       interpolation="bilinear")
    else:
        ax_data.set_facecolor("0.92")
    in_ext = ((support_xy[:, 0] >= xlo) & (support_xy[:, 0] <= xhi)
              & (support_xy[:, 1] >= ylo) & (support_xy[:, 1] <= yhi))
    sup_clip = support_xy[in_ext]
    n_sub = min(8000, len(sup_clip))
    if n_sub:
        idx = np.random.default_rng(0).choice(len(sup_clip), n_sub,
                                              replace=False)
        ax_data.scatter(sup_clip[idx, 0], sup_clip[idx, 1], s=2.0,
                        c="#ffd86e", alpha=0.18, zorder=1, linewidths=0)
    ax_data.contourf(kx, ky, kz, levels=[0.10, 0.30, 0.60, 1.01],
                     colors=["#f0c0a0", "#e89060", "#d05050"], alpha=0.45,
                     zorder=2)
    ax_data.contour(kx, ky, kz, levels=[0.15, 0.6], colors="white",
                    linewidths=1.2, alpha=0.9, zorder=3)
    if goal_xy is not None:
        ax_data.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=900,
                        c="white", alpha=0.55, zorder=6)
        ax_data.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=520,
                        c="gold", edgecolors="black", linewidths=1.6, zorder=7)
    ax_data.set_title("offline dataset support\n(cube xy in the play data)",
                      fontsize=9)

    handles = [
        Line2D([0], [0], marker="s", color="white", markeredgecolor="black",
               markersize=10, linestyle="", label="cube (mean this phase)"),
        Line2D([0], [0], marker="*", color="gold", markeredgecolor="black",
               markersize=15, linestyle="", label="goal (task target)"),
        Line2D([0], [0], color="white", marker=r"$\rightarrow$",
               markeredgecolor="black", markersize=12, linestyle="",
               label="cube flow (data)")]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7,
                      framealpha=0.85)
    fig.suptitle(f"{task} — {method}: phase x outcome value (saliency) "
                 "& cube-flow fields  [training states]", fontsize=13)
    fig.tight_layout()
    fp = out_dir / f"{task}__{method}.png"
    fig.savefig(fp, dpi=125)
    plt.close(fig)
    return fp.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/training_value_multiseed")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", nargs="*",
                    default=[f"task{i}" for i in range(1, 6)])
    ap.add_argument("--methods", nargs="*", default=["FB policy", "GCIQL"])
    ap.add_argument("--n-eps", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-type", default="states")
    ap.add_argument("--mujoco-gl", default=None)
    args = ap.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "aggregate" / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)

    sx = REPO_ROOT / "analysis" / "scene"
    cal = json.loads((sx / "calib.json").read_text())
    bg_img = (mpimg.imread(str(sx / "topdown.png"))
              if (sx / "topdown.png").exists() and cal.get("photoreal")
              else None)
    anchors = (json.loads((sx / "task_anchors.json").read_text())
               if (sx / "task_anchors.json").exists() else {})

    npz = next(iter(sorted((root).glob("p*/training_states.npz"))), None)
    if npz is None:
        raise SystemExit(f"no p*/training_states.npz under {root}")
    goals_xyz = np.asarray(np.load(npz)["goals"], np.float64)  # [5,3]

    vdf = _value_frame(root)
    fdf, support_xy = _flow_frame(args.data_path, goals_xyz, Thresholds(),
                                  _table_z(args.seed, args.obs_type),
                                  args.horizon, args.thresh, args.n_eps,
                                  seed=args.seed)
    written = []
    for task in args.tasks:
        for method in args.methods:
            name = _render(task, method, vdf, fdf, support_xy, anchors,
                           bg_img, out_dir)
            if name:
                written.append(name)
    (out_dir / "_index.md").write_text(
        "# training_value scene (phase x outcome)\n\n"
        + "\n".join(f"- {n}" for n in sorted(written)) + "\n")
    print(f"[training_value_scene] wrote {len(written)} figs -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse-check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/training_value_scene.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 3: Smoke (reduced — task1 only, few episodes)**

```bash
MUJOCO_GL=glfw .venv/bin/python scripts/training_value_scene.py \
  --root analysis/training_value_multiseed --data-path datasets \
  --mujoco-gl glfw --tasks task1 --n-eps 40 --out /tmp/tv_scene_smoke 2>&1 | tail -4
ls /tmp/tv_scene_smoke
```
Expected: `[training_value_scene] wrote 2 figs`; `/tmp/tv_scene_smoke/` has `task1__FB policy.png`, `task1__GCIQL.png`, `_index.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/training_value_scene.py
git commit -m "feat(training_value): phase x outcome value/action scene renderer (eval analog, pure data)"
```

---

## Task 3: Real run + verification

**Files:** none modified (run + verify).

- [ ] **Step 1: Full render (all 5 tasks x 2 methods)**

```bash
MUJOCO_GL=glfw .venv/bin/python scripts/training_value_scene.py \
  --root analysis/training_value_multiseed --data-path datasets \
  --mujoco-gl glfw --n-eps 300
ls analysis/training_value_multiseed/aggregate/scene/
```
Expected: `[training_value_scene] wrote 10 figs`; 10 `task{1..5}__{FB policy,GCIQL}.png` + `_index.md`.

- [ ] **Step 2: Visual check**

Open `analysis/training_value_multiseed/aggregate/scene/task1__FB policy.png` and confirm: 2 rows (success_bound/fail_bound) × 3 phase panels with red value saliency + white flow arrows over the scene, goal star, and the support panel in the top-right.

- [ ] **Step 3: Full test suite**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -q`
Expected: 7 passed.
Run: `git status --short` — only `analysis/` untracked changes remain.

- [ ] **Step 4: Commit the plan doc**

```bash
git add docs/superpowers/plans/2026-05-22-training-value-scene.md
git commit -m "docs(training-value): implementation plan — phase x outcome value/action scene"
```

---

## Self-review notes

- **Spec coverage:** pure-data renderer (Task 2, no model load); value heatmap from pooled parquets (`_value_frame` + `_bin_vz_grid`); arrows + support from buffer (`_flow_frame` + `_phase_action_fields`); grid rows=outcome × cols=phase + support panel (`_render`); FB policy + GCIQL; scene bg/calib/anchors reuse; EEF dropped (documented); per-step flow label TDD (Task 1). All covered.
- **Placeholder scan:** none — full script + commands.
- **Type/name consistency:** `flow_step_labels(cube_traj, goals_xyz, horizon, thresh) -> [T,n_goals]` defined Task 1, used in `_flow_frame`; reused helpers called with their real signatures (`_bin_vz_grid(df, xmin, xmax, ymin, ymax, nbins)`, `_phase_action_fields(df_with_episode_t_Vz, (xe,ye), n_arrow, n_min)`, `_nan_gaussian(grid, sigma)`, `support_kde(xy, gx, gy)`); value-frame columns (`task, method, outcome, region, cube_x, cube_y, V`) and flow columns (`task, outcome, region, episode, t, cube_x, cube_y`) consistent between producers and `_render`.
- **Open risk:** filenames contain a space (`task1__FB policy.png`) — matches the `method` label; acceptable (quoted in shell). If undesirable, slugify in a follow-up.
```