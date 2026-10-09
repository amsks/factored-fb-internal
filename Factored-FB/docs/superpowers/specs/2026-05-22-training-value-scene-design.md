# Design: Phase × outcome value/action scene on training states (eval phase_scene analog)

Date: 2026-05-22
Status: approved (brainstorming)
Branch: `feat/cube-analysis-pipeline`

## Goal

Reproduce the eval `phase_scene` figure (`gciql_profile/aggregate/comparison/
phase_scene/task{n}__{method}.png`, built by `_phase_action_scene` in
`scripts/gciql_profile_aggregate.py`) on the **training states**, split by the
bounded-reach outcome label. Per task per method: a value-saliency heatmap +
cube-flow arrows over the top-down scene, gridded by phase × outcome, plus an
offline-dataset-support panel.

Scope: **state observations only**; methods **FB policy** + **GCIQL** (matching
the eval figure).

## Key insight: pure-data renderer, no model scoring

The eval plot derives the value heatmap and the arrows from one rollout
dataframe. On training data we **decouple** them so nothing needs re-scoring:

- **Value heatmap** ← the already-scored multiseed parquets: `p*/fb_values.parquet`
  (`V_policy`) and `p*/gciql_values.parquet` (`V`), pooled over the 6 seeds.
  They already carry `cube_x, cube_y, region, outcome`.
- **Flow arrows + dataset support** ← computed directly from the play buffer
  (`datasets/cube-single-play-v0`): per-step cube xy, cube displacement
  `cube[t+1]−cube[t]`, per-step phase, and per-(step,task) success-bound label.
  No value needed.

So the whole thing is **pure data + matplotlib**, like `_phase_action_scene`
itself.

## Context / reuse

- Reused pure helpers (import from `scripts.gciql_profile_aggregate`):
  `_bin_vz_grid(df, xmin, xmax, ymin, ymax, nbins)` (mean-Vz grid; needs only
  `cube_x, cube_y, Vz`), `_phase_action_fields(df, (xe,ye), n_arrow, n_min)`
  (its `(U,V)` arrow output; needs `episode, t, cube_x, cube_y, Vz`),
  `_nan_gaussian(grid, sigma)`.
- `support_kde(xy, gx, gy)` from `evals.dataset_support`.
- Scene assets: `analysis/scene/{topdown.png, calib.json, task_anchors.json}`
  (calib `photoreal=true`; anchors give per-task `cube_init`, `goal`,
  `eef_init`).
- Phase + label helpers: `evals.training_value.region_labels`,
  `horizon_reach_label`; `evals.phase_probe.Thresholds`. Grip/lift from physics
  exactly as the FB pass (`grip = clip(physics[:,6]/0.8,0,1)`,
  `lift = physics[:,16] − table_z`; cube = `physics[:,14:17]`).
- Goals (3-D, for the outcome label) from a `p*/training_states.npz` `goals`
  key (so flow labels match the parquet labels). `table_z` from the same npz is
  not stored; recompute via one task env, or reuse the FB-pass constant — see
  plan (read from a task env once).

## Chosen approach

A **new script** `scripts/training_value_scene.py` (pure data, `.venv`). No
model loading; no jax. Rejected: extending `training_value_aggregate.py` (its
job is tables/curves over the random sample; this needs full-trajectory flow and
the scene background — different substrate, heavier render).

### Data flow

```
analysis/training_value_multiseed/p*/{fb_values,gciql_values}.parquet  (value heatmap, pooled)
analysis/training_value_multiseed/p0/training_states.npz                (goals 3-D)
datasets/cube-single-play-v0/buffer/episode_*.npz                       (flow + support)
analysis/scene/{topdown.png,calib.json,task_anchors.json}               (background + markers)
  -> training_value_scene.py (.venv)
       -> analysis/training_value_multiseed/aggregate/scene/{task}__{method}.png  (x10)
       -> .../scene/_index.md
```

### Components (one script)

1. **`_flow_frame(files, goals_xyz, thr, table_z, horizon, thresh, n_eps)`** —
   subsample `n_eps` episodes (default 300); per step emit `episode, t, cube_x,
   cube_y, region`, and `outcome_task{1..5}` (success_bound via
   `horizon_reach_label` on that episode's own cube tail vs each 3-D goal).
   Returns a long DataFrame (`task, outcome, region, episode, t, cube_x,
   cube_y`) for the arrows, and the raw play cube-xy `[N,2]` for support.

2. **`_value_frame(root)`** — pool `p*/fb_values.parquet` → method "FB policy"
   (`V_policy`→`V`); pool `p*/gciql_values.parquet` → method "GCIQL". Long
   columns `task, method, outcome, region, cube_x, cube_y, V`.

3. **`_render(task, method, vdf, fdf, support_xy, anchors, bg, calib, out)`** —
   2×4 gridspec: rows `success_bound`/`fail_bound`, cols reach/grasp/transport;
   `gs[0,3]` = support panel (`gs[1,3]` blank). Per phase panel:
   - heatmap: `_bin_vz_grid(panel_vdf.assign(Vz=V), …, NBINS)` →
     `_nan_gaussian` → per-panel percentile-[2,98] Reds saliency overlay
     (alpha ≤ 0.55), `imshow` over the scene bg (mirrors eval lines 502–524).
   - arrows: `_phase_action_fields(panel_fdf.assign(Vz=0.0), (xe,ye), n_arrow=
     NARROW)` → unit-normalized white `quiver` (mirrors eval 525–544).
   - markers: per-panel mean cube (white square) + goal star (anchor). **No EEF
     marker** (not available offline).
   - support panel: play cube-xy scatter + `support_kde` contourf/contour +
     anchor markers (mirrors eval 545–601).
   Shared world extent = 1–99th pct of the task's value-frame cube xy,
   square-padded (eval 400–416). Constants `NBINS=16, NARROW=10, LOW_N=30`.

4. **`main`** — `--root analysis/training_value_multiseed`,
   `--data-path datasets`, `--out <root>/aggregate/scene`,
   `--tasks task1..task5`, `--n-eps 300`, `--horizon 200`, `--thresh 0.04`,
   `--mujoco-gl` (for the one env used to read `table_z`).

## Testing

- Unit (`.venv`): write the per-step flow labeler as a pure helper
  `flow_step_labels(cube_traj, goals_xyz, region, horizon, thresh)` and test it
  on a hand-built 1-episode cube trajectory — assert `outcome_task*` matches a
  hand-computed bounded reach. The phase + reach primitives it composes
  (`region_labels`, `horizon_reach_label`) are already unit-tested.
- Smoke: run the renderer on the real multiseed root; assert 10 PNGs +
  `_index.md`; visually inspect `task1__FB.png` (heatmap + arrows + scene +
  support render; success vs fail rows present).

## Risks & mitigations

- **No EEF offline:** dropped (documented). Cube + goal markers retained.
- **Sparse panels** (e.g., grasp×success_bound): `_bin_vz_grid` returns NaN
  cells (smoothed/transparent); arrow cells with `< n_min` are skipped; panels
  with `< LOW_N` states drawn faded with an `n=…, low` title (eval behavior).
- **Goal/`table_z` provenance:** goals from the npz (match parquet labels);
  `table_z` read once from a task env (same as the FB pass) so the flow phase
  labels match the value-frame phase labels.
- **Heatmap vs arrows from different samples:** intentional — heatmap = learned
  value (scored states), arrows = data dynamics (full trajectories). Noted in
  the figure caption / index.

## Out of scope

- Re-scoring V on whole episodes (unnecessary — reuse the parquets).
- FB-data value variant (FB policy + GCIQL only; trivial to add later).
- Pixel observations.

## Commit boundary

Commit `scripts/training_value_scene.py`, any new test, and this spec/plan.
`analysis/` stays untracked ([[project-untracked-artifacts]]). Runs under
`.venv` ([[project-venvs]]).
