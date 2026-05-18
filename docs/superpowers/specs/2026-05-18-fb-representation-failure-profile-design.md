# FB Representation-Failure Profile — Design Spec

**Date:** 2026-05-18
**Status:** Approved (design); pending implementation plan
**Builds on:** phase-probe
(`docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md`,
`evals/phase_probe.py`) and the checkpoint pipeline
(`evals/analysis.py`).

## 1. Question

The phase-probe established (7 seeds, n=10): FB **reaches** the cube
~100% and **grasps+lifts** it ~92%, but ~44% of baseline episodes fail
at **transport/placement** (cube ends ~0.085 m from a 0.04 m goal).
S2 (pre-grasped) does not beat S0. This profile asks **why**, in terms
of the forward-backward representation: *what is the representation not
capturing that prevents reliable placement?*

Four FB-specific probes, aggregated over the 7 complete seeds
(s3–s8, s10), synthesised into one data-filled narrative.

## 2. Shared Machinery

- Load agent+checkpoint via `evals.analysis.load_cfg`/
  `build_env_and_agent`/`load_checkpoint` (device cpu, `--data-path`
  override, `--mujoco-gl glfw`; mujoco stays lazy via the
  `phase_probe._ensure_mujoco` pattern).
- Task z and relabel statistics from
  `evals.ogbench.OGBenchEvaluator._infer_z(task)` — it already returns
  `relabel_reward#nonzero`, `#zero`, `#num_samples`, `#mean`.
- `Q_z(s,a) = (F · z).sum(-1)` where
  `F = model._forward_map(model._left_encoder(s), z, a)`
  (shape `[num_parallel, batch, z_dim]`); diagnostics use the
  **ensemble mean** over the F heads (the representation's central
  estimate — not the pessimistic min the actor uses; recorded in
  `metrics.json`).
- `V(s) = Q_z(s, π(s))` with `π` = the flow-BC actor (`agent.act`).
- Cube xyz = `physics[..., 14:17]`, goal =
  `env.unwrapped.cur_task_info['goal_xyzs'][tb]`,
  `tb = getattr(env.unwrapped, '_target_block', 0)`,
  `d = ‖cube − goal‖`. Per-task reward via
  `envs.ogbench.get_relabel_fn(domain, task)`.
- S0 rollouts reused from `evals.phase_probe.rollout_with_phase_signals`
  (extended to also record per-step `obs` and `action`), each episode
  labelled via `evals.phase_probe.classify_phases` →
  outcome ∈ {`success`, `transport_fail`, `other`}. The driving
  contrast is **success vs transport_fail** on the S0 scenario.

## 3. The Four Probes

### 3.1 Value landscape
Per step of S0 rollouts record `d_t`, `Q_t = Q_z(s_t, a_t)`,
`V_t = Q_z(s_t, π(s_t))`, and the phase from `classify_phases`
signals. Aggregate `V` binned by `d`, split by outcome.

- **Headline metric:** slope of `V` w.r.t. `−d` (OLS) within the
  *transport region* (steps after the cube is first secured), computed
  per episode then averaged, for `success` vs `transport_fail`, plus the
  success−fail gap.
- **Saturation:** mean `V` at the secure-moment vs at episode end.

A flat fail-slope (≈0) with a positive success-slope ⇒ the
representation provides no value gradient guiding placement.

### 3.2 z-decoding + relabel sparsity
- **Sparsity:** per task, `relabel_pos_frac =
  relabel_reward#nonzero / relabel_reward#num_samples` from
  `_infer_z` — how few goal-satisfying states the play data contains.
- **z-decoding:** over an offline subsample of `next_obs` (size
  `--buffer-sample`), score `B(next_obs) · z`; take top-`--topk`.
  Report `topk_mean_d` and
  `topk_pct_at_goal = mean(d_topk < 0.04)`.

If top-`B·z` states are far from goal (lifted, not placed), z does not
encode the placement target.

### 3.3 B-space placement resolution
On the same offline subsample compute `B(next_obs)` and true `d`.

- **Ridge probe** `B → d` with held-out split: report **R²**.
- **Separability:** linear-SVM (or logistic) accuracy distinguishing
  *placed* (`d < 0.04`) from *near-miss* (`0.04 ≤ d < 0.12`).

Low R²/accuracy ⇒ B cannot represent the fine placement distinction,
so the successor measure `F·B` cannot either.

### 3.4 Off-support / coverage
Reference = the offline `obs` subsample, standardised (per-dim
mean/std from the subsample). Per S0 rollout step, NN (L2) distance from
`s_t` to the nearest reference obs (batched cdist on the subsample).
Aggregate mean NN-dist by phase (reach/lift/transport) × outcome.

Tests whether transport_fail episodes leave training-data support
during the transport phase (FB extrapolation breakdown).

## 4. Per-Checkpoint Profiler

`scripts/representation_profile.py` — one checkpoint (caller loops
seeds; the aggregate step runs all 7).

**CLI:** `--config` (req), `--checkpoint` (req), `--out` (req),
`--device` (cpu), `--tasks` (default all 5), `--n-episodes` (10),
`--seed`, `--mujoco-gl`, `--data-path`, `--buffer-sample` (20000),
`--topk` (256). Heavy imports inside `main()` after `parse_args`
(fast `--help`); the build-env from `build_env_and_agent` is closed
immediately (unused; renderer/physics resource hygiene).

**Flow per task:** `_infer_z` → z + relabel_metrics; draw offline
subsample (`next_obs`, `physics`, relabel reward); profiling S0
rollouts (obs+action+signals) classified by outcome; compute the four
probes.

**Outputs → `analysis/representation_profile/<name>/`:**
- `value_landscape.parquet` — per step: task, episode, outcome, d, Q,
  V, phase
- `z_decoding.parquet` — per task: relabel_pos_frac, topk_mean_d,
  topk_pct_at_goal
- `b_resolution.parquet` — per task: r2, placed_vs_nearmiss_acc
- `coverage.parquet` — per step: task, outcome, phase, nn_dist
- `metrics.json` — config, checkpoint, buffer_sample, topk, z norm,
  ensemble-aggregation note (provenance)

All buffer-side probes operate on the configurable subsample (default
20k) to stay tractable at ~1M-transition buffers × 7 seeds.

## 5. Cross-Seed Aggregate + Story

`scripts/representation_profile_aggregate.py` reads every
`<root>/s*_final/` probe parquet, tags `seed`, combines the 7 seeds.

**Tables (parquet + markdown), mean±std across seeds:**
- **T1 value gradient** — per task: transport-region slope(`V` vs `−d`)
  for success vs transport_fail, and the gap.
- **T2 z/data sparsity** — per task: relabel_pos_frac, topk_mean_d,
  topk_pct_at_goal.
- **T3 B resolution** — per task: ridge R², placed-vs-near-miss acc.
- **T4 coverage** — mean NN-dist by phase × outcome.

**Plots:**
- `value_vs_dist.png` — `V` binned by `d`, success vs transport_fail,
  mean±std band over seeds (primary figure).
- `sparsity_bars.png` — relabel_pos_frac & topk_pct_at_goal per task.
- `b_resolution_bars.png` — R² & separability per seed.
- `coverage_curves.png` — NN-dist vs phase progress, success vs fail.

**`story.md`** — a templated synthesis whose every quantitative claim is
filled from the aggregated tables; the causal phrasing is fixed text,
the evidence is data-substituted (no hard-coded conclusions). It ties
sparsity (T2) → weak/placement-blind z and B (T2/T3) → absent value
gradient in transport (T1) → off-support drift at failure (T4).

## 6. Architecture & Files (additive only)

```
evals/representation_profile.py              # probe + helper functions
scripts/representation_profile.py            # per-checkpoint CLI
scripts/representation_profile_aggregate.py  # 7-seed tables/plots/story
tests/test_representation_profile.py
```
Reused unmodified: `evals.analysis`, `evals.phase_probe`
(rollout+classify+`_ensure_mujoco`), `OGBenchEvaluator._infer_z`,
`envs.ogbench.get_relabel_fn`/`create_ogbench_env`, `data.ogbench`.
No training/eval code modified.

`evals/representation_profile.py` public surface:
`q_values(model, obs, action, z)`, `v_values(model, obs, z, agent)`,
`rollout_for_profile(env, agent, z, n_episodes, thr)`,
`probe_value_landscape(episodes)`, `probe_z_decoding(model, next_obs,
physics, reward, goal, z, topk)`, `probe_b_resolution(B, d)`,
`probe_coverage(episodes, ref_obs)`,
`run_representation_profile(...)`.

## 7. Error Handling

- Non-cube/manip env → clear error via
  `evals.phase_probe.ensure_manip_env`.
- Zero relabel positives → `topk_pct_at_goal = 0`,
  `relabel_pos_frac = 0` with a flagged warning (a finding, not a
  crash).
- Degenerate ridge (constant `d`, or <2 classes for separability) →
  metric = `nan` with a warning.
- `--buffer-sample` capped at the buffer size; rollouts/subsamples
  seeded from `cfg.seed`/`--seed` for reproducibility.
- Per-task `try/finally env.close()`; a per-task failure warns and
  continues; remaining outputs still written.
- Aggregate tolerates a missing seed or probe file: it is skipped and
  the gap is noted in `story.md`.
- Device/render: `--device`, `torch.load(map_location=)`,
  `--mujoco-gl` set before env import; mujoco import stays lazy.

## 8. Testing (TDD, no MuJoCo)

Pure logic unit-tested with stubs/synthetic arrays:
- `q_values`/`v_values` — ensemble-mean shape and value correctness on a
  stub model exposing `_left_encoder`/`_forward_map`.
- `probe_value_landscape` — monotone `V(−d)` ⇒ positive slope; flat ⇒
  ≈0; success/fail split correct; transport-region masking correct.
- `probe_z_decoding` — constructed B,z with known goal states ⇒ exact
  top-k %-at-goal and relabel fraction.
- `probe_b_resolution` — B linear in d ⇒ R²≈1; random ⇒ R²≈0;
  separability accuracy on separable vs overlapping sets.
- `probe_coverage` — synthetic obs ⇒ correct NN-dist by phase/outcome.
- aggregate — synthetic per-seed parquets ⇒ correct mean±std tables;
  `story.md` contains the filled numbers and required section headers.

Real MuJoCo run is the documented acceptance step (runs on this Mac as
`phase_probe` did): per-seed profile for s3–s8,s10 then aggregate.
`scripts/representation_profile.py` is thin glue; its logic lives in the
unit-tested `evals/representation_profile.py`.

## 9. Non-Goals / YAGNI

- No retraining, fine-tuning, or representation modification — analysis
  only.
- No new rollout machinery — extend `phase_probe`'s rollout to also
  record obs/action; do not fork it.
- No non-cube domains; single-cube only.
- No per-seed narrative — one aggregate `story.md` (7-seed mean±std).
- No causal-claim hard-coding — `story.md` states observed numbers and
  the implied reading only.
- No modification of existing pipeline/training/eval files.

## 10. Build Order

1. `evals/representation_profile.py`: `q_values`/`v_values` (+ tests)
2. `rollout_for_profile` (extends phase rollout with obs/action) (+ tests)
3. `probe_value_landscape` (+ tests)
4. `probe_z_decoding` + `probe_b_resolution` (+ tests)
5. `probe_coverage` + `run_representation_profile` (+ tests)
6. `scripts/representation_profile.py` CLI (+ `--help`)
7. `scripts/representation_profile_aggregate.py` (tables/plots/story) (+ tests)
8. Documented acceptance: per-seed s3–s8,s10 then aggregate
