# Design: Extend the diagnostic profiling pipeline to pixel observations

Date: 2026-05-20
Status: approved (brainstorming)
Branch: `feat/cube-analysis-pipeline`

## Goal

Extend the "rest of" the cube-failure diagnostic pipeline — the deep probes
beyond learning curves (phase/outcome funnel, dataset-support coverage,
value-gradient, backward-map resolution) — to run on **FB pixel** checkpoints,
and **fully generalize the GCIQL profiler** so GCIQL-pixel checkpoints slot in
when they exist.

This round is **pipeline + smoke test only**: make the per-checkpoint profilers,
the coverage probe, the GCIQL profiler, and the aggregates obs_type-aware; add a
pixel runner script and a fast smoke test verified on a real pixel checkpoint.
The full 5-seed run, the diagnostic figures, and the paper section are deferred
to a later round (the user runs the heavy job, then we wire figures + paper).

## Context

### What already works for pixels (verified by code reading)

- `evals/analysis.py:build_env_and_agent(cfg)` builds the agent from `cfg`, so a
  pixel config builds the FB agent **with the DrQ encoder**.
- `agents/fb/model.py:backward_map/forward_map` encode raw obs internally
  (`self._backward_map(self._bw_encoder(self._normalize(obs)))`), so the FB
  probes that call `model.forward_map(obs,z,a)` / `model.backward_map(obs)` work
  on raw pixel obs.
- `data/ogbench.py` (pixel path) stores `next.physics` alongside pixel obs, so
  the analysis variable (cube xyz = `physics[:,14:17]`) is available even when
  the policy input is pixels.
- Phase signals come from privileged env state, independent of obs_type:
  `evals/phase_probe.py` reads `info["privileged/block_0_pos"]` (cube),
  `info["proprio/effector_pos"]` (eff), `info["proprio/gripper_opening"]` (grip).
- The aggregate scripts (`representation_profile_aggregate.py`,
  `gciql_profile_aggregate.py`) are parquet-driven (glob `s*_final/<name>.parquet`)
  and obs_type-agnostic.

### The one probe that does NOT transfer

`evals/_profile_core.py:probe_coverage` computes nearest-neighbor distance
(`m_dist`) in **raw observation space** (`ep["obs"]` normalized against the
buffer's `next_obs`). For pixels that is image space — ill-posed and
high-dimensional. Decision: compute pixel coverage in **physics space**,
specifically **cube-xyz** (the only buffer physics index reliably identified;
matches the existing cube-coverage construction in
`gciql_profile_aggregate.py`, which reads `physics[:,14:16]`).

## Chosen approach (Approach A)

Generalize `probe_coverage` with a `feature` seam and derive the feature from
`cfg.obs_type` inside the profilers. No new CLI flag. State path unchanged.
Rejected: explicit `--coverage-space` flag everywhere (Approach B, extra surface,
YAGNI) and separate pixel-specific profiler scripts (Approach C, ~1000 lines of
duplication).

### Components

#### 1. `evals/_profile_core.py:probe_coverage` (modify, core)

New signature:
```python
def probe_coverage(episodes, ref_feat, thr, seed=0, feature="obs"):
```
- `feature="obs"` (default): unchanged — per-episode `ep["obs"]` normalized
  against `ref_feat` (the buffer `next_obs`), NN distance. State path is
  byte-for-byte equivalent to today.
- `feature="cube"`: per-episode `ep["cube"][:, :3]` normalized against
  `ref_feat` (the buffer `physics[:,14:17]`), NN distance.
- The normalize + NN body is shared; only the per-episode feature matrix and the
  reference matrix differ. Pure and unit-testable (no MuJoCo).

#### 2. FB profiler (`scripts/representation_profile.py`, `evals/representation_profile.py`)

- `run_representation_profile(...)` selects coverage feature + ref by obs_type:
  - pixels → `probe_coverage(eps, physics_ref[:,14:17], thr, feature="cube")`,
    where `physics_ref = sample_buffer(...)["physics"]`.
  - state → `probe_coverage(eps, next_obs_ref, thr, feature="obs")` (unchanged).
- Thread `obs_type` (or a derived `coverage_feature`) into
  `run_representation_profile` from `representation_profile.py` (it already has
  `cfg.obs_type`). No other FB changes — value/z-decoding/B-resolution probes
  already encode pixel obs through the model and read `physics[:,14:17]`.

#### 3. GCIQL profiler (`scripts/gciql_profile.py`) — fully generalized now

- Read `obs_type` from `flags.json` (default `state`).
- `_jax_rollout`: feed obs to the jax agent regardless of shape; record physics
  signals (`privileged/block_0_pos`, `proprio/effector_pos`,
  `proprio/gripper_opening`) for analysis, as the FB rollout does.
- `_load_ref_obs`: add a pixel branch returning cube-xyz physics ref
  (`physics[:,14:17]`) instead of stacked state obs.
- Coverage call uses `feature="cube"` when `obs_type == "pixels"`.
- Pure logic (coverage feature selection, ref loading) is unit-tested with
  stubs/synthetic arrays. The end-to-end pixel jax rollout is **structured but
  unverified** until GCIQL pixel checkpoints exist — documented as such.

#### 4. Aggregates (`representation_profile_aggregate.py`, `gciql_profile_aggregate.py`)

- No schema change: coverage still emits an `nn_dist` column (now physics-space
  for pixels), value_steps still carry `cube_x/cube_y` (already physics-derived),
  so the existing tables/figures render unchanged on pixel roots.
- Pixel aggregation is **FB-only** now via
  `representation_profile_aggregate.py --root analysis/representation_profile_pixel`.
- The FB-vs-GCIQL comparison in `gciql_profile_aggregate.py` stays deferred until
  GCIQL pixel exists (reserved-slot pattern, same as the curves work). Verify the
  FB-only path runs without a GCIQL root.

#### 5. Runner (`scripts/run_pixel_profile.sh`, new)

- Loops the 5 FB pixel seeds. For each seed dir under
  `RESULTS/fb-pixel-results/*__s{n}`:
  - `representation_profile.py --config <seed>/.hydra/config.yaml
    --checkpoint <seed>/checkpoints/final.pt
    --out analysis/representation_profile_pixel/s{n}_final --data-path <data>
    --mujoco-gl <gl>`
  - `phase_probe.py` for the same seed (per its CLI) →
    `analysis/phase_probe_pixel/s{n}_final/`.
- Then `representation_profile_aggregate.py --root analysis/representation_profile_pixel`.
- Guarded GCIQL branch: if a `--gciql-root <dir>` of GCIQL-pixel checkpoints is
  passed, run `gciql_profile.py` per seed and the comparison aggregate.
- Params via env/args: `SEEDS`, results root, `DATA_PATH`, `MUJOCO_GL`,
  `N_EPISODES`, `BUFFER_SAMPLE`.

#### 6. Smoke test (`scripts/smoke_pixel_profile.sh`, new) — run this round

- 1 seed (s0), `--n-episodes 2 --buffer-sample 256`, real
  `RESULTS/fb-pixel-results/*__s0/checkpoints/final.pt`.
- Asserts: the five FB parquets are written under
  `analysis/representation_profile_pixel/s0_final/`, `coverage.parquet` has the
  `nn_dist` column and non-empty rows, and the run used cube-space coverage
  (`feature="cube"`).
- Executed locally to verify the obs_type-aware FB profiling works on a real
  pixel checkpoint (MuJoCo + ogbench import OK here; state profiling already ran
  locally).

## Data flow

```
RESULTS/fb-pixel-results/<__s{n}>/{.hydra/config.yaml, checkpoints/final.pt}
  -> run_pixel_profile.sh -> representation_profile.py (+ phase_probe.py)   [obs_type=pixels]
       -> analysis/representation_profile_pixel/s{n}_final/*.parquet        (coverage in cube space)
            -> representation_profile_aggregate.py
                 -> analysis/representation_profile_pixel/aggregate/  (T1/T2/T3 tables + bars)

(later) GCIQL pixel ckpts -> gciql_profile.py -> analysis/gciql_profile_pixel/
            -> gciql_profile_aggregate.py (FB-vs-GCIQL comparison)
```

## Testing

- New `tests/test_profile_coverage_feature.py`: `probe_coverage(feature="cube")`
  on synthetic episodes + cube ref; assert NN distances match a hand-computed
  small case; assert `feature="obs"` path is unchanged.
- New unit coverage of the obs_type→coverage_feature selection (pure function).
- Existing `tests/test_representation_profile.py`, `tests/test_gciql_profile.py`,
  `tests/test_phase_action_scene.py` must stay green (state path untouched).
- Smoke (`smoke_pixel_profile.sh`) executed on the real s0 pixel checkpoint.
- All via `.venv/bin/python` (rliable arch8 shim + torch live there).

## Risks & mitigations

- **GCIQL pixel path unverified**: no GCIQL pixel checkpoints exist; the jax
  pixel rollout cannot be end-to-end tested now. Mitigation: unit-test the pure
  pieces, gate the runner's GCIQL branch behind an explicit `--gciql-root`, and
  document the path as untested-until-data.
- **Cube-only coverage is narrower than the state obs-space coverage**: the
  state coverage used the full obs vector; pixel coverage uses cube xyz only,
  because cube (`physics[:,14:17]`) is the one reliably identified buffer index.
  Documented; richer physics features can be added later behind the same seam.
- **Throughput**: encoding the buffer sample through DrQ for B-resolution can be
  slow on CPU; the smoke uses `--buffer-sample 256`. The full run is the user's
  heavy job.
- **Pixel obs shape at probe time**: the eval/probe env applies frame-stacking
  (merged change), so `ep["obs"]` matches the model's expected input; the smoke
  validates this on a real checkpoint.

## Out of scope this round

- The full 5-seed FB pixel profiling run.
- Diagnostic figures and the paper section update (next round, after the heavy
  run produces parquets).
- Recomputing the existing **state** coverage in physics space (kept as-is).

## Commit boundary

Commit only `scripts/`, `evals/`, `tests/`. `analysis/` and `RESULTS/` stay
untracked (see [[project-untracked-artifacts]]). Run everything via
`.venv/bin/python` ([[rliable-arch8-shim]]).
