# Phase-Conditioned Object-Centric Value & Action Fields — Design

Date: 2026-05-19
Status: approved (brainstorming) — pending spec review

## Goal

Visualize, for FB vs GCIQL on `cube-single-play-v0`, the extracted
goal-conditioned value landscape over cube `(x, y)` position, **stratified
by robot–cube interaction phase**, with overlaid greedy-policy-induced cube
motion (arrows), the task goal (star), and dataset support (contours).

This is the manipulation analogue of a navigation value map: a cube `(x,y)`
heatmap is too collapsed because the same cube position spans approach,
contact, and transport regimes — so the visualization is conditioned on the
manipulation phase.

## Decisions (locked during brainstorming)

1. **Computation basis:** reuse the already-recorded greedy-policy rollout
   trajectories (`value_steps.parquet`, recorded at temperature 0 for both
   methods). No model re-evaluation, no k-NN-over-dataset, no fresh sim
   rollouts for arrows.
2. **Phase scheme:** reuse the existing per-step region label from
   `evals/_profile_core.region_label` — `reach`/`lift`/`transport` — mapped
   to Approach / Contact–Pre-grasp / Carry–Transport. Same labels the
   paper's T4 coverage table uses. No second taxonomy.
3. **Dataset support:** decode true offline-buffer cube xy (faithful), with
   a verification gate against recorded rollout cube xy.
4. **Goal handling:** per task (no pooling). One goal star per panel.
5. **Outcome handling:** keep the success vs transport_fail split.
6. **Packaging:** 10 figures, one per (task × method); each figure is
   2 rows {success, transport_fail} × 3 cols {approach, contact, transport}.
7. **Value normalization:** V z-scored per method, per task.
8. **Approach:** A — extend the recorders to persist two columns, add a new
   pure dataset-support module, render in `gciql_profile_aggregate.py`.

## Architecture

Three isolated, independently testable units; one-directional data flow
(recorders persist → aggregate renders). No model loading in the renderer,
so it runs on the analysis host like the current `_value_scene`.

```
 gciql_profile.py (JAX venv) ─┐
 representation_profile (torch)┴─→ value_steps.parquet  (+ new cols: t, region)
 datasets/.../buffer ─→ evals/dataset_support.py ─→ decoded cube xy + KDE
                                                          │
 gciql_profile_aggregate.py :: _phase_action_scene()  ◄───┘
        └─→ analysis/gciql_profile/aggregate/phase_scene/{task}__{method}.png  (×10)
```

- `evals/_profile_core.region_label` (exists) — single source of the
  reach/lift/transport label; recorders call it, no duplicated phase logic.
- `evals/dataset_support.py` (new, pure numpy/scipy) — decode + KDE.
- `gciql_profile_aggregate._phase_action_scene()` (new) — pure
  pandas/matplotlib consumer.

## Component 1 — Recorder change (persist `t`, `region`)

Additive change in the shared pure seam `gciql_profile.episodes_to_frames`
and the FB analogue in `evals/representation_profile.py`. Both already build
per-episode `cube[T,3]`, `eff[T,3]`, `grip[T]`, `transport_mask`.

- `t` — int step index within episode (`enumerate` over the existing loop).
- `region` — `reach`/`lift`/`transport` from
  `_profile_core.region_label(ep, thr)`, called once per episode, zipped
  per step.

New `value_steps` schema:
`task, episode, t, outcome, region, d, V, transport, cube_x, cube_y`
(`outcome` already persisted; unchanged.)

Thresholds pinned to `Thresholds()` defaults (`eps_reach=0.06`,
`delta_lift=0.03`, `tau_grip=0.5`) so `region`/`transport` stay consistent
with already-published T4 numbers.

**One-time operational cost:** regenerate `value_steps` for both methods —
GCIQL via `scripts/gciql_profile.py` (10 seeds, JAX venv); FB via
`scripts/representation_profile.py` (FB `s*_final` seed dirs, torch host).
`value_landscape`, `coverage`, `phase_funnel` are untouched; existing
`cmp_*` figures and the IQM/curve work are unaffected.

## Component 2 — `evals/dataset_support.py` (new, pure)

`dataset_cube_xy(buffer_dir, ref_cube_xy, n_files=200) -> ndarray[N,2]`
1. Candidate 2-column windows over `physics` (MuJoCo qpos) and
   `observation`, gathered across sampled buffer episodes.
2. **Verification gate:** select the window whose 2D distribution best
   matches `ref_cube_xy` (cube xy already in `value_steps` from rollouts):
   axis-range overlap + 2D-histogram cosine similarity ≥ threshold. If none
   passes, **raise** (never plot a guessed slice).
3. Return decoded `[N,2]`.

`support_kde(xy, grid_x, grid_y) -> ndarray[ny,nx]` — scipy Gaussian KDE on
the V heatmap's world-grid, normalized to [0,1]; 2D-histogram fallback if
KDE covariance is singular.

Decoded xy computed once, cached to
`analysis/dataset_support_cube_xy.npy`, reused across all 10 figures. The
buffer is task-agnostic play data → support is shared across tasks;
per-task differences come from V/arrows/goal.

## Component 3 — `_phase_action_scene()` (renderer)

Inputs: per-seed `value_steps` for both methods (existing `_seed_concat`),
`analysis/scene/{topdown.png,calib.json}` (existing
`scene_calib.world_to_px`), cached dataset-support xy.

For each `task` in task1..5 × `method` in {FB, GCIQL} → one figure,
2 rows {success, transport_fail} × 3 cols {reach, lift, transport}.

Per panel (filter: task, method, outcome, region):
- **Background:** schematic top-down via `world_to_px` (offscreen MuJoCo
  unavailable on host; calibration exact — caption says "schematic").
- **Color:** mean of `Vz` on a 40×40 cube-xy grid (reuse `_bin_vz_grid`).
  `Vz` = V z-scored per method per task (across that task's steps, all
  phases/outcomes). `viridis`, α≈0.55, masked where empty.
- **Support contour:** 2–3 thin grey `contour` lines from the KDE on the
  same grid.
- **Star:** that task's goal cube xy, white star / black edge.
- **Arrows:** 12×12 quiver; per coarse cell, mean per-step displacement
  `cube_xy[t+1] − cube_xy[t]` (rows ordered by `episode`,`t`; same
  task/method/outcome/region). Cells with `< n_min` (default 5) skipped.
  Fixed reference scale + scale-bar annotation.
- **Sparse panels:** `< n_min` steps → background+star only, caption
  "insufficient steps (n=…)". GCIQL `transport_fail` per task is expected
  to be sparse; surfaced, not hidden.

Outputs: `analysis/gciql_profile/aggregate/phase_scene/{task}__{method}.png`
(10 files) + `_index.md`. No automatic copy into `PAPER/` (figures/ is
curated; wiring is a separate explicit step).

Pure seam for testing: `_phase_action_fields(df, grid) ->
(Vz_grid, quiver_uv, counts)` — pure, unit-tested. The matplotlib wrapper
gets only a smoke test (consistent with existing `_value_scene`).

## Testing

- `tests/test_dataset_support.py` — verifier finds a known slice; raises on
  wrong-only fixture; KDE shape/normalization; histogram fallback.
- `tests/test_gciql_profile.py` (extend) — `episodes_to_frames` emits `t`
  (0..T-1 per episode) and `region` ∈ {reach,lift,transport}; existing
  assertions unchanged.
- `tests/test_phase_action_scene.py` — pure `_phase_action_fields`:
  synthetic 2-episode frame → asserted z-scored V bin, known displacement
  arrow, per-cell counts; empty frame returns masked grid without raising.
- Full suite stays green (currently 66 + new).

## Verification before "done"

Re-profile → aggregate → full suite; confirm all 10 figures exist and are
non-empty; spot-check one panel's arrows/Vz against raw `value_steps`
(same raw-sanity discipline as the IQM table check).

## Paper wiring (separate explicit step)

After figures are generated and eyeballed, copy a chosen subset into
`PAPER/fb-cube-failure/figures/` and extend the existing
`\subsection{Characterizing FB's Losing Strategy}`. 10 figures is heavy for
the paper; recommend embedding only the most diagnostic subset (e.g., the
FB vs GCIQL transport-phase pair for 1–2 representative tasks), full set
kept in `analysis/`. Exact selection decided at wiring time.

## Out of scope (YAGNI)

k-NN / Q re-evaluation; fresh sim rollouts for arrows; distance-threshold
phase taxonomy; pooled / aggregate-across-task variants; real MuJoCo
background render.
