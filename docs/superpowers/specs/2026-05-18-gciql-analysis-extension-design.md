# Spec: Extend the analysis pipeline to GCIQL + FB-vs-GCIQL paper

**Date:** 2026-05-18
**Status:** approved (design), pending implementation plan

## Problem

`main` (merged into `feat/cube-analysis-pipeline`) adds a GCIQL baseline: a
verbatim vendored OGBench (JAX/flax) trained on `cube-single-play-v0`. Run
data for 10 seeds lives at
`RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0NN_*/`
(`params_<step>.pkl`, `flags.json`, `train.csv`, `eval.csv`).

The existing analysis pipeline profiles the **FB** agent only
(`evals/representation_profile.py`, `evals/phase_probe.py`,
`scripts/representation_profile*.py`). We must extend it to GCIQL and produce
an FB-vs-GCIQL comparison in the paper draft.

GCIQL is goal-conditioned IQL: `GCValue` V(s,g), `GCDiscreteCritic`/critic
Q(s,g,a), `GCActor` π(a|s,g). It has **no `z`, no forward/backward maps**, so
the FB-specific probes T2 (z-decoding/relabel sparsity) and T3 (B-resolution)
have no principled analog and are out of scope.

## Goals

- Profile GCIQL on the **transferable** probes: T1 value landscape
  (Spearman ρ between V and −d over the transport region), T4 off-support
  coverage, and the reach→grasp→lift→transport phase funnel.
- Reuse byte-identical T1/T4/coverage/phase math between FB and GCIQL (no
  metric drift — the comparison must be fair).
- Aggregate 10 GCIQL seeds and produce an FB-vs-GCIQL comparison.
- Fold the comparison into the paper, **preserving the user's
  hand-restructured layout** by porting that layout into the generator.
- Respect the hard isolation rule: the JAX/GCIQL track never shares a
  process or venv with the PyTorch FB track.

## Non-goals

- T2/T3 analogs for GCIQL (no principled mapping; YAGNI).
- Retraining or modifying GCIQL / vendored OGBench / FB training code.
- Changing the FB metrics or the existing FB per-seed outputs' schema.

## Verified integration facts

- Checkpoint: `save_agent` →
  `pickle({'agent': flax.serialization.to_state_dict(agent)})` at
  `params_<epoch>.pkl`. Restore: build skeleton via
  `GCIQLAgent.create(seed, ex_obs, ex_goals, ex_actions, config)` then
  `flax.serialization.from_state_dict(agent, load_dict['agent'])`
  (`third_party/ogbench/impls/utils/flax_utils.py:162-198`).
- Policy: `agent.sample_actions(observations=obs, goals=goal,
  temperature=0)` (deterministic eval).
- Value: `agent.network.select('value')(obs, goal)` → V(s,g).
- Goal: OGBench eval does `env.reset(options=dict(task_id=task_id,
  render_goal=...))` then `goal = info['goal']`; cube has 5 tasks
  (`task_id` 1–5), matching `eval.csv` columns
  `evaluation/task{1..5}_*_success`.
- `evals/phase_probe.py` is torch-free at import (torch imported lazily
  inside `rollout_with_phase_signals`): `Thresholds`, `classify_phases`,
  `step_signals` are importable in a torch-free venv.
- `evals/representation_profile.py` imports `torch` at module top, so its
  pure helpers (`_spearman`, `transport_mask`, `_region`,
  `probe_coverage`) are NOT importable torch-free → must be relocated.
- `scripts/representation_profile_aggregate.py` verdict helpers
  (`_isnan`, `_verdict_t1.._t4`, `_synthesis`, threshold constants) are
  already torch-free.

## Design

### Component 1 — `evals/_profile_core.py` (new, torch-free)

Relocate (cut, not copy) into this module:
- From `representation_profile.py`: `_spearman`, `transport_mask`,
  `_region`, `probe_coverage`.
- From `representation_profile_aggregate.py`: threshold constants
  (`T1_RHO_MIN`, `T1_RHO_GAP`, `T2_*`, `T3_*`, `T4_OFFSUPPORT_RATIO`),
  `_isnan`, `_verdict_t1`, `_verdict_t2`, `_verdict_t3`, `_verdict_t4`,
  `_synthesis`.

`representation_profile.py` and `representation_profile_aggregate.py`
re-import these names from `_profile_core` (e.g.
`from evals._profile_core import _spearman, transport_mask, _region,
probe_coverage`). Pure relocation: **zero behavior change**, existing 49
tests must stay green unchanged. `_profile_core` imports only
`numpy`/`pandas` (pandas lazily inside functions as today).

### Component 2 — `scripts/gciql_profile.py` (new; runs in the JAX venv)

A self-contained per-checkpoint profiler. It DOES import jax/flax/vendored
OGBench and therefore only runs inside the isolated JAX venv (it is the
inference analog of `run_gciql.py`'s isolation; the FB pipeline never
imports it).

CLI:
```
python scripts/gciql_profile.py \
  --run-dir RESULTS/gciql_.../factored-fb-gciql/sd001_... \
  --step 1000000 \
  --out analysis/gciql_profile/s1_final \
  --tasks 1,2,3,4,5 --n-episodes 10 --scenario S0 \
  --dataset-path datasets/cube-single-play-v0
```

Behavior:
1. Read `<run-dir>/flags.json` → `env_name`, agent config, obs settings,
   training seed.
2. For each `task_id`: build the vendored OGBench env
   (`third_party/ogbench/impls` on `sys.path`; use OGBench's own env
   factory as `impls/main.py` does), reset with
   `options=dict(task_id=task_id)`, `goal=info['goal']`.
3. Build `GCIQLAgent.create(...)` skeleton from config + example
   obs/goal/action drawn from the env, `restore_agent(<run-dir>,
   <step>)`.
4. Roll out the deterministic policy for `--n-episodes` per task
   (scenario S0 = vanilla reset; S1/S2 deferred — S0 only for v1).
   Per step record: `step_signals(info)` (reused from
   `evals.phase_probe`), `V = network.select('value')(obs, goal)`,
   `success`, and cube→goal distance `d`.
5. Per episode: `classify_phases` → outcome
   (`success`/`transport_fail`/`other`) and furthest phase;
   `rho_V_negd = _profile_core._spearman(V[mask], -d[mask])` with the
   same guard (≥5 transport steps, non-degenerate d & V);
   coverage rows via `_profile_core.probe_coverage` against the offline
   cube dataset reference (the same `datasets/cube-single-play-v0`
   buffer the FB pipeline used, loaded as raw `next/observation`).
6. Write, schema-aligned with the FB per-seed outputs:
   - `value_landscape.parquet`: `task, episode, outcome, rho_V_negd,
     V_at_secure, V_at_end, n_transport_steps`
   - `coverage.parquet`: `task, outcome, region, nn_dist`
   - `phase_funnel.parquet`: `task, episode, furthest_phase, outcome`
   - `metrics.json`: run-dir, step, tasks, n_episodes, env_name,
     `value_fn = "GCValue V(s,g)"`.

Pure, JAX-free seams (unit-testable here): `flags.json` parsing,
episode-record → (rho/coverage/outcome/funnel) assembly. The jax
load/rollout path is exercised by an integration smoke (run where the
JAX venv exists).

### Component 3 — `scripts/gciql_profile_aggregate.py` (new) + comparison

1. Aggregate `analysis/gciql_profile/s*_final/` (10 seeds) →
   `aggregate/`: `T1_value_gradient.parquet`,
   `T4_coverage.parquet`, `funnel.parquet`, plots
   (`value_rho_bars.png`, `coverage_curves.png`, `funnel.png`), and a
   `story.md` with per-probe verdicts (`_profile_core._verdict_t1/_t4`,
   `_synthesis`) — same format as the FB Readout, restricted to
   T1/T4/funnel.
2. FB-vs-GCIQL comparison in `aggregate/comparison/`: read the existing
   FB aggregate (`analysis/representation_profile/aggregate/`) and emit:
   - `fb_vs_gciql.parquet`: rows = method × {ρ_success, ρ_fail, T1
     verdict, coverage transport nn_dist (fail/success), transport-fail
     funnel rate}.
   - `cmp_value_rho.png` (grouped bars: FB vs GCIQL, ρ by outcome),
     `cmp_funnel.png`, `cmp_coverage.png`.
   - `comparison.md`: a short data-driven synthesis contrasting FB and
     GCIQL on the three shared axes.
   Missing FB aggregate → clear FileNotFoundError naming the producing
   command.

### Component 4 — paper generator

`evals/paper_fragment.py` `build_fragment` is re-templated to **reproduce
the user's current hand-restructured `fb_cube_failure.tex` layout
verbatim**:
- `\section{Where and Why FB Fails on Cube Manipulation}` with
  subsections, in order: `Experiment Design \& Phase Probing`,
  `Methodology: Representation Probes`, `Results \& Empirical Synthesis`,
  `Discussion \& Core Narrative`.
- Self-contained document (full preamble, inline booktabs tables,
  unicode→TeX), data-driven from the aggregates (continuation of
  `2e31611`).

Extended with GCIQL:
- Under `Results \& Empirical Synthesis`, a new
  `\subsection{FB vs GCIQL: Shared-Axis Comparison}` containing
  `cmp_value_rho.png`, `cmp_funnel.png`, `cmp_coverage.png` and an
  inline `fb_vs_gciql` booktabs table.
- In `Discussion \& Core Narrative`, an appended sentence built from
  `comparison.md`'s synthesis contrasting FB and GCIQL.
- Generator now takes the GCIQL comparison dir; if absent it emits the
  FB-only document unchanged (graceful, so FB-only regeneration still
  works).

`scripts/build_paper_fragment.py` gains `--gciql-root`
(default `analysis/gciql_profile/aggregate/comparison`).

### Testing

- `_profile_core`: existing FB tests must pass unchanged (proves the
  relocation is behavior-preserving). Add one direct import test.
- `gciql_profile.py`: TDD pure seams with JAX-free fakes — `flags.json`
  parsing; an episode-records list → expected `value_landscape`/
  `coverage`/`funnel` frames (stub V and signals). No jax import in
  tests.
- `gciql_profile_aggregate.py`: TDD with synthetic per-seed parquets
  (mirrors the existing aggregate test), incl. the comparison given a
  synthetic FB aggregate.
- `paper_fragment.py`: extend `tests/test_paper_fragment.py` to assert
  the user's subsection titles are present in order, the document is
  self-contained, tables inline, and (when a GCIQL comparison dir is
  supplied) the comparison subsection + figures appear; FB-only path
  still works.
- Full suite green.

### Execution / feasibility

- Buildable and fully unit-verifiable here (no jax needed for the pure
  seams, aggregate, generator).
- GCIQL inference needs jax[cpu]+flax+vendored ogbench. Primary path:
  a local CPU venv (plain `uv venv .venv-jax-local`, no `/dev/shm`;
  `requirements-jax.txt`). Attempt locally; if darwin JAX/MuJoCo install
  fails, the user runs `scripts/gciql_profile.py` (10 seeds) + the
  aggregate on the Linux box where GCIQL runs live. The aggregate,
  comparison, and paper generation run anywhere.

## Risks / mitigations

- **Vendored-OGBench cube `info` keys** may differ from the FB-side
  wrapper that `step_signals` expects (`proprio/effector_pos`,
  `proprio/gripper_opening`, `privileged/block_0_pos`, `success`). The
  smoke step prints the env's `info` keys; if they differ, add a small
  key-mapping shim in `gciql_profile.py` (only place that touches the
  vendored env).
- **`GCIQLAgent.create` config reconstruction** from `flags.json`: the
  exact `config`/example-shapes call is verified to exist but its
  precise arguments are confirmed during the smoke; the pure tests don't
  depend on it.
- **jax[cpu] on macOS**: unknown until attempted; non-blocking — the
  Linux fallback path keeps the deliverable usable.
- **`datasets/cube-single-play-v0`** must be the same underlying OGBench
  data for a fair T4 comparison; it is (FB used the OGBench download).

## Acceptance

- `pytest tests/ -q` green (FB suite unchanged + new tests).
- `scripts/gciql_profile.py --help` parses; pure seams unit-tested.
- With GCIQL per-seed parquets present (run here if jax[cpu] installs,
  else by the user on Linux): aggregate + comparison produce
  `analysis/gciql_profile/aggregate/{...}` and
  `.../comparison/{fb_vs_gciql.parquet, cmp_*.png, comparison.md}`.
- `scripts/build_paper_fragment.py` regenerates
  `PAPER/fb-cube-failure/fb_cube_failure.tex` reproducing the user's
  restructured layout and adding the FB-vs-GCIQL subsection — verified
  by the structure assertions in the tests.
