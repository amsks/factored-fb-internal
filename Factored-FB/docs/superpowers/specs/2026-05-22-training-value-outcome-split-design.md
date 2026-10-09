# Design: Outcome-split value analysis on training states (T1 mirror)

Date: 2026-05-22
Status: approved (brainstorming)
Branch: `feat/cube-analysis-pipeline`

## Goal

Extend the existing training-state value analysis (the committed `training_value`
pipeline, per-phase ρ(V, −d)) to ask the **T1 question on training data**: does
the learned value *discriminate outcomes*? On rollouts (Finding 3 / T1) FB's
value↔(−distance) gradient was **not** higher on `success` than on
`transport_fail` episodes (it was lower). We test whether the same failure to
discriminate appears when we label **training** states by whether the goal is
actually reachable from them.

Scope: **state observations only** (both methods state-based). FB-policy,
FB-data, and GCIQL value, per phase, split by a per-state outcome label.

## What "previously" used (the thing we mirror)

Rollout-side (`evals/representation_profile.py` + `evals/phase_probe.py`):
- `outcome` per **episode** via `classify_phases`: `success` = the env success
  flag OR-accumulated over the rollout (cube within **~0.04 m** of goal at any
  step); else `transport_fail` if the furthest fail phase was transport; else
  `other`.
- Metric (T1): per-episode Spearman ρ(V, −d) over the **transport region**,
  compared `success` vs `transport_fail`.
- A rollout is **200 steps** (`manipspace max_episode_steps=200`).

Training data has no env success flag (reward ≡ 0, discount ≡ 1, no goals — it is
`cube-single-play` play data, 1000 episodes × 1001 steps). So the label is
reconstructed from recorded `physics`.

## Outcome label (the one new definition)

For each sampled training state at episode step `t` and each of the 5 task goals
`g`:

```
success_bound(state, g) = min || cube[t : t+H] - g || < THRESH
```

with `H = 200` (the eval horizon) and `THRESH = 0.04 m` (the eval success
criterion). Else `fail_bound`. This is the faithful translation of the rollout
`success` label — same threshold, same "ever reaches within the horizon"
semantics — applied as a sliding 200-step window because play episodes are
1001 steps (≈5 eval horizons). States in the last <200 steps use their shorter
remaining tail; a state with an empty tail is `fail_bound`.

The label is **per (state, task)** — a state can be success-bound for one goal
and fail-bound for another, matching the per-task structure of the existing
pipeline.

## Context / reuse

- Offline data: `datasets/cube-single-play-v0/buffer/episode_*.npz`
  (`physics` 21-d; cube xyz = `physics[:,14:17]`). Episodes are one continuous
  play trajectory (no within-episode resets).
- The label needs each sampled state's `(episode_id, step_id)` so we can read its
  **own** future tail. The current `_sample_states` concatenates episodes and
  discards that identity → it gets extended to also return `ep_id, step_id`.
- Existing pipeline (committed): `evals/training_value.py` (pure helpers),
  `scripts/training_value_profile.py` (`--method fb` writes `fb_values.parquet`
  + `training_states.npz` + `phase_composition.parquet`; `--method gciql` writes
  `gciql_values.parquet`), `scripts/training_value_aggregate.py` (tables/curves/
  scenes, single-dir and multi-seed `p*/` pooling).
- State checkpoints + seeds: as in the committed multi-seed run (FB
  `RESULTS/Factored-FB-cube-run/*__s{seed}/checkpoints/final.pt`; GCIQL
  `RESULTS/gciql_20260518_201030/.../sd*` step 1000000; 6 seed pairs `p0..p5`).

## Chosen approach

Extend the existing scripts in place — **no new scripts**. Rejected: a separate
outcome-split script (would duplicate the sampler, the two-venv state sharing,
and the multi-seed pooling).

### Components

#### 1. `evals/training_value.py` — new pure helpers (TDD)

- `horizon_reach_label(d_future, thresh, horizon) -> bool` — `True` iff
  `min(d_future[:horizon]) < thresh`; `False` on empty tail.
- `outcome_spearman_table(df, value_col)` — ρ(V, −d) grouped by
  `(region, outcome)`, mean/std over tasks (extends `phase_spearman_table`).
- `value_discrimination(df, value_col)` — per `(region)`: AUC of `value_col`
  separating `success_bound` vs `fail_bound`, and mean ΔV after per-task
  z-scoring. AUC ≈ 0.5 ⇒ no discrimination.

#### 2. `scripts/training_value_profile.py --method fb` (`.venv`)

- `_sample_states` also returns `ep_id, step_id` per sampled state.
- After phase classification, compute the `[n_states, n_tasks]` outcome matrix:
  per task `g`, per-episode cube→`g` distance arrays, then
  `horizon_reach_label` on each sampled state's tail.
- Add an `outcome` column (`"success_bound"`/`"fail_bound"`) to the per-task rows
  of `fb_values.parquet`.
- Store the outcome matrix in `training_states.npz` (key `outcome`,
  shape `[n_states, n_tasks]`, dtype `bool`; `True` = success_bound) alongside
  existing keys. Both passes map `True`→`"success_bound"`, `False`→`"fail_bound"`
  when building parquet rows.

#### 3. `scripts/training_value_profile.py --method gciql` (`.venv-jax-cpu`)

- Read `outcome` from `training_states.npz`; attach `outcome[i, ti-1]` to each
  row of `gciql_values.parquet`. No recompute → labels identical to FB's.

#### 4. `scripts/training_value_aggregate.py` (`.venv`)

Add, for both single-dir and multi-seed pooling:
- `spearman_outcome_table.{md,parquet}` — ρ(V, −d) per phase × outcome × method.
  **Pathology replicates iff ρ is not higher for `success_bound` than
  `fail_bound`.**
- `value_discrimination.{md,parquet}` — AUC + mean ΔV per phase × method.
- `value_vs_dist_by_outcome.png` — value-vs-distance curves faceted by phase,
  two lines (success/fail) per method.
- `outcome_composition.{md,parquet}` — fraction `success_bound` per phase × task
  (sanity: class not degenerate).

### Data flow

```
datasets/cube-single-play-v0
  -> profile --method fb (.venv): + outcome col in fb_values.parquet,
                                  + outcome matrix in training_states.npz
  -> profile --method gciql (.venv-jax-cpu): + outcome col in gciql_values.parquet
  -> aggregate (.venv): + spearman_outcome_table, value_discrimination,
                        value_vs_dist_by_outcome.png, outcome_composition
```

## Testing

Unit (`.venv`, synthetic arrays, no MuJoCo/torch/jax) in
`tests/test_training_value.py`:
- `horizon_reach_label`: reach just inside vs outside `H`; reach but beyond
  `thresh`; truncated tail; empty tail → `False`.
- `value_discrimination`: AUC ≈ 1 on a separable toy, ≈ 0.5 on random.
- `outcome_spearman_table`: monotone toy gives ρ ≈ +1 per (region, outcome).

Smoke: FB then GCIQL on a few hundred states against real state checkpoints;
assert `outcome` column present with both classes, npz has `outcome`; aggregate
emits the new table + one by-outcome curve.

## Re-run

Regenerate FB + GCIQL passes across the 6 seed pairs `p0..p5` (the outcome label
is not recoverable from the existing npz — `ep_id/step_id` were not stored), then
re-aggregate. Same seeds/checkpoints as the committed multi-seed run.

## Risks & mitigations

- **Degenerate class:** play data may rarely bring the cube within 4 cm of a
  specific eval goal in 200 steps → tiny `success_bound` class, noisy ρ/AUC.
  Mitigation: `outcome_composition` reports per-phase success-bound fractions;
  if a phase×task cell is too small, flag (and ρ/AUC are computed with the
  existing nan-safe aggregation).
- **Truncated tails (last <200 steps):** ~20% of states have a shorter tail;
  use whatever future steps exist. Documented, not special-cased beyond empty
  tail → `fail_bound`.
- **Two-venv label consistency:** GCIQL reads the stored matrix (never
  recomputes) → identical labels.

## Out of scope

- Pixel observations (state only).
- The per-phase (non-outcome) table/curves/scenes — already shipped; this only
  adds the outcome split.
- Discounted-hindsight-return / net-progress label variants (considered,
  rejected in favor of the T1-faithful bounded-reach label).

## Commit boundary

Commit `evals/`, `scripts/`, `tests/`, and this spec/plan. `analysis/`,
`RESULTS/`, `datasets/` stay untracked ([[project-untracked-artifacts]]). FB +
aggregate + tests under `.venv`, GCIQL under `.venv-jax-cpu` ([[project-venvs]]).
