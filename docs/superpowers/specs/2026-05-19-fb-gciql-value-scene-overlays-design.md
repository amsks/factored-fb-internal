# Spec: FB-vs-GCIQL value-over-scene overlays

**Date:** 2026-05-19
**Status:** approved (design), pending implementation plan

## Problem

The paper compares FB and GCIQL value gradients with bar/box/curve
plots, but does not *spatially* show, on the cube scene, where the value
function helps or fails — i.e. what FB's losing transport strategy looks
like versus GCIQL's winning one. We want a value heatmap composited over
a top-down render of the cube workspace, for winning vs losing episodes,
FB vs GCIQL, to characterize FB's losing strategies.

## Goals

- Four overlays — {FB, GCIQL} × {success, transport_fail} — each a
  translucent V heatmap over a single shared top-down render of the
  `cube-single-play-v0` scene, plus 1–2 representative cube trajectories,
  and a 2×2 montage.
- Data-driven: the heatmap is binned mean V over cube (x,y) positions
  the policy *actually visited* in the transport region (no synthetic
  off-manifold states, no interpolation across unvisited cells).
- Per-method V z-scoring (FB V≈+2000 vs GCIQL V≈−60: a shared absolute
  scale is meaningless).
- Hand-inserted into the user's reformatted paper with grounded prose;
  no regeneration of the user-owned `.tex`.

## Non-goals

- Rendering in both env stacks: the scene is identical for FB and
  GCIQL; render **once**, reuse the background for all overlays.
- Counterfactual cube-position sweep / true value landscape (user chose
  data-driven visited states).
- Per-frame video; per-episode galleries. One static representative
  top-down background frame only.
- Autogenerating the paper (`build_paper_fragment.py` is not used here).

## Verified facts

- Cube xyz is `physics[:,14:17]`; task is planar (cube z≈0.02 const).
  Task `init_xyzs`/`goal_xyzs` known per task from
  `env.unwrapped.cur_task_info` / `task_infos` (e.g. task1
  init `[0.425,0.1,0.02]`, goal `[0.425,-0.1,0.02]`); workspace x∈~[0.3,
  0.55], y∈~[-0.25,0.25].
- `value_steps.parquet` (both FB and GCIQL) currently:
  `task, episode, outcome, d, V, transport`. No cube xy.
- FB per-step rows are built in `evals/representation_profile.py`
  `probe_value_landscape` (`per_step_rows.append({... "d","V",
  "transport" ...})`); GCIQL in `scripts/gciql_profile.py`
  `episodes_to_frames` (`vs_rows.append({... "d","V","transport"})`).
  Both have the per-step cube array in scope at that point
  (`ep["cube"]` / `cube`).
- FB sweep ≈ 2.1 min (7 seeds), GCIQL sweep ≈ 5.6 min (10 seeds),
  `.venv-jax-cpu` works for GCIQL; FB uses `.venv` + `MUJOCO_GL=glfw`.
- `mujoco` 3.7.0 importable in `.venv`. `MUJOCO_GL=glfw` worked for
  stepping on this macOS box earlier; `egl` failed (Linux-only).
- The comparison plots live in `scripts/gciql_profile_aggregate.py`
  `_comparison(...)`, output dir `.../comparison/`; figures are
  hand-copied into `PAPER/fb-cube-failure/figures/` and referenced from
  the user's `.tex`.

## Design

### Component A — persist cube (x,y) per step

- `evals/representation_profile.py`: in the `per_step_rows.append`
  dict add `"cube_x": float(cube_xy[0]), "cube_y": float(cube_xy[1])`
  using the per-step cube already iterated for that frame.
- `scripts/gciql_profile.py` `episodes_to_frames`: in the `vs_rows`
  loop add `"cube_x"`, `"cube_y"` from `ep["cube"]` (shape [T,3]) for
  the matching step.
- `value_steps` schema becomes
  `task, episode, outcome, d, V, transport, cube_x, cube_y`.
- Re-run both sweeps to materialize the new column.
- Tests assert the new schema and that cube_x/cube_y equal the input
  cube positions.

### Component B — top-down scene render + calibration

`scripts/render_cube_scene.py` (run in `.venv-jax-cpu`, vendored
OGBench cube env which has working `render()`):

- Build the cube env, reset task1, set a **fixed top-down camera**:
  free camera at `(cx, cy, H)` looking straight down (elevation −90°,
  azimuth fixed), `H` and `fovy` constants we choose so the table
  workspace fills the frame.
- Render one RGB frame `analysis/scene/topdown.png` (e.g. 600×600).
- Compute the analytic world→pixel map for the controlled camera
  (orthographic-equivalent affine from camera height/fovy/img size):
  `px = ax * world_x + bx`, `py = ay * world_y + by`. Persist to
  `analysis/scene/calib.json` (`ax,bx,ay,by,img_w,img_h,workspace`).
- Validation (smoke, not unit): project task `init_xyzs`/`goal_xyzs`
  with the affine; assert each lands inside the image and near the
  visually-expected spot (printed for human check in the smoke).
- Pure, unit-testable seam: `world_to_px(xy, calib)` (no mujoco) —
  tested with a synthetic calib (known points → known pixels).

### Component C — overlay producer

In `scripts/gciql_profile_aggregate.py`, add `_value_scene(cdir,
gciql_root, fb_seed_root, scene_png, calib)` called from `_comparison`
when `scene_png`/`calib` exist (else skipped with a warning):

- For each (method, df) ∈ {FB: f_vs, GCIQL: g_vs}:
  - `Vz = _zscore_v(df)` (reuse existing helper).
  - For each outcome ∈ {success, transport_fail}:
    - Restrict to `transport` rows; 2D-bin `Vz` over a fixed
      workspace grid (e.g. 40×40 over x∈[0.30,0.55], y∈[−0.25,0.25]);
      `nanmean` per cell; **mask cells with <1 sample**.
    - Composite: load `topdown.png`; for each populated cell, map its
      world centre via `world_to_px` and paint a translucent colormap
      patch (alpha≈0.55); mark the goal (task-mean) and start; overlay
      the medoid success and medoid transport-fail cube (x,y) polyline
      (medoid = episode whose path is closest to the per-group mean
      path, by mean step distance).
    - Save `cmp_scene_<METHOD>_<outcome>.png`.
  - Assemble 2×2 montage `cmp_value_scene.png`.
- Per-method colorbar (z-scored units); never a shared FB/GCIQL scale.
- Determinism: fixed grid, fixed seed for any sampling.

### Component D — paper

Hand-edit `PAPER/fb-cube-failure/fb_cube_failure.tex` (user-owned):
add the 2×2 `cmp_value_scene.png` (or four panels in the existing
minipage idiom) under the FB-vs-GCIQL subsection, with a grounded
"what we see" paragraph characterizing FB's losing strategy
(numbers/positions read from the produced data, not invented).

### Testing (TDD)

- `value_steps` schema test (Component A) — FB and GCIQL produce
  `cube_x,cube_y` equal to the source cube positions.
- `world_to_px` pure unit test (Component B) — synthetic calib, known
  world points → expected pixels; out-of-workspace handled.
- Binning/mask/z-score aggregation pure test (Component C) — synthetic
  `value_steps` (two methods, known cube xy + V) → expected masked grid
  shape, empty-cell mask, per-method standardisation; medoid selection
  on a tiny set returns the closest-to-mean episode.
- Render + composite is integration-smoked (run here), not unit-tested.
- Full `pytest tests/ -q` green.

### Execution

1. TDD Components A–C; `pytest tests/ -q` green.
2. Re-run FB sweep (`.venv`, `MUJOCO_GL=glfw`, ~2 min) and GCIQL sweep
   (`.venv-jax-cpu`, ~6 min) to emit cube_x/cube_y.
3. `scripts/render_cube_scene.py` → `analysis/scene/{topdown.png,
   calib.json}` (smoke-validate projected known points).
4. `scripts/gciql_profile_aggregate.py` → 4 `cmp_scene_*.png` +
   `cmp_value_scene.png`.
5. Hand-insert into the paper with grounded prose; report what the
   overlays show (FB losing strategy vs GCIQL).

## Risks / mitigations

- **Offscreen MuJoCo on macOS** (primary risk): `mujoco.Renderer`
  offscreen may fail on darwin (`egl` failed; `glfw` worked for
  stepping). Mitigation: attempt render in `.venv-jax-cpu` vendored
  env (and `.venv` env as fallback); **if both fail**, Component B
  emits a schematic top-down background instead (table rectangle, goal
  markers drawn to scale in world coords) and `calib.json` for that
  synthetic canvas — the heatmap + trajectory analysis still ships,
  explicitly labelled "schematic (render unavailable on this host)".
  Either way Components A/C and all tests are unaffected.
- **Calibration mismatch** → heatmap misaligned with the scene:
  controlled fixed camera with analytic affine; smoke validates by
  projecting known `init_xyzs`/`goal_xyzs`.
- **Sparsity** of visited cells: mask, never interpolate (no fabricated
  value where the policy never went).
- **Scale**: per-method z-score + per-method colorbar only.

## Acceptance

- `pytest tests/ -q` green (existing + new schema/affine/aggregation
  tests).
- After re-runs + render + aggregate:
  `analysis/gciql_profile/aggregate/comparison/` contains
  `cmp_scene_FB_success.png`, `cmp_scene_FB_fail.png`,
  `cmp_scene_GCIQL_success.png`, `cmp_scene_GCIQL_fail.png`,
  `cmp_value_scene.png`; `analysis/scene/{topdown.png,calib.json}`
  exist; projected known points validated in the render smoke.
- Paper updated by hand with the montage + grounded prose; the
  render path used (photoreal vs schematic fallback) is reported.
