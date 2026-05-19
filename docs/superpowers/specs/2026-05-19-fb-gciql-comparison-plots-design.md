# Spec: FB-vs-GCIQL comparison plots in the paper

**Date:** 2026-05-19
**Status:** approved (design), pending implementation plan

## Problem

The FB-vs-GCIQL comparison currently ships a single plot
(`cmp_value_rho.png`, mean-only grouped bars of Spearman ρ by outcome).
The headline scientific contrast — FB's flat value landscape vs GCIQL's
goal-tracking one — has no visual in the paper, and the GCIQL profiler
does not persist the per-step `(d, V)` data needed to draw it.

## Goals

- Add four FB-vs-GCIQL comparison figures to the paper, computed from
  the real per-seed data:
  1. V binned by cube→goal distance, FB vs GCIQL × success/transport-fail.
  2. Per-episode Spearman ρ distribution (box/violin), FB vs GCIQL × outcome.
  3. Outcome-distribution funnel (success / transport_fail / other),
     FB vs GCIQL.
  4. Off-support coverage nn-distance by region, FB vs GCIQL.
- Persist GCIQL per-step `(d, V)` so plot 1 is possible.
- Keep FB and GCIQL on byte-identical metrics; figures auto-flow into the
  user's hand-restructured paper layout via the generator.

## Non-goals

- Re-running the FB sweep (its per-seed parquets already exist, 7 seeds).
- Finer GCIQL-only phase granularity (reach/grasp) in the *comparison*
  funnel — FB's representation pipeline only persists the
  success/transport_fail/other `outcome`, so the shared comparison uses
  that for an apples-to-apples bar chart. (GCIQL's own
  `phase_funnel.parquet` keeps its finer detail for the GCIQL-only story.)
- New metrics or thresholds.

## Verified facts

- FB per-seed parquets exist:
  `analysis/representation_profile/s*_final/{value_landscape,value_steps,
  coverage}.parquet` (7 seeds). `value_steps` schema:
  `task, episode, outcome, d, V, transport`. `value_landscape` has
  per-episode `rho_V_negd, outcome, task`.
- GCIQL per-seed parquets exist (10 seeds):
  `analysis/gciql_profile/s*_final/{value_landscape,coverage,
  phase_funnel}.parquet` — **no** `value_steps`.
- `scripts/gciql_profile.py` `_jax_rollout` already records per-step
  `D` and `VV`; the episode dict carries `d`, `V`. `episodes_to_frames`
  returns `(value_landscape, coverage, funnel)` today.
- `scripts/gciql_profile_aggregate.py` `_comparison(...)` currently
  writes `fb_vs_gciql.parquet`, `cmp_value_rho.png`, `comparison.md`,
  reading only the FB *aggregate* `T1_value_gradient.parquet`.
- `evals/paper_fragment.py` `build_fragment(..., gciql_cmp=...)` embeds
  exactly one figure (`cmp_value_rho.png`) + the parquet table +
  `comparison.md` rendered prose, inside
  `\subsection{FB vs GCIQL: Shared-Axis Comparison}`.
- GCIQL full sweep ≈ 5.6 min (10 seeds × 5 tasks × 10 eps), CPU JAX
  venv `.venv-jax-cpu` (already installed locally).

## Design

### Component A — GCIQL per-step output

`scripts/gciql_profile.py`:
- `episodes_to_frames(...)` returns a **4-tuple**
  `(value_landscape, coverage, funnel, value_steps)`. `value_steps` is a
  long-form DataFrame, one row per step, schema **exactly** matching FB:
  `task, episode, outcome, d, V, transport` (`transport` = the per-step
  `transport_mask` bool; `outcome` = the episode outcome).
- `main()` writes `value_steps.parquet` alongside the existing three.
- All callers/tests updated to the 4-tuple.
- Re-run the 10-seed GCIQL sweep to materialize `value_steps.parquet`.

### Component B — comparison aggregate (4 plots)

`scripts/gciql_profile_aggregate.py`:
- `aggregate(...)` / `_comparison(...)` gain a `fb_seed_root` (default
  `analysis/representation_profile`) and use `gciql` per-seed root
  (`root`) so both methods' per-seed `value_steps`, `value_landscape`,
  `coverage` are loadable. The existing FB-aggregate `T1` read stays for
  the `fb_vs_gciql.parquet` summary row.
- New helper `_load_seed(root, name)`: concat `s*_final/<name>.parquet`
  (reuse the existing `_load` glob logic).
- Emit into `.../comparison/`:
  1. `cmp_value_vs_dist.png` — for each (method ∈ {FB,GCIQL}, outcome ∈
     {success, transport_fail}) bin `d` into 9 quantile-edged bins over
     the pooled d, plot mean `V` per bin (4 curves; skip a curve whose
     subset is empty). Built from FB+GCIQL `value_steps`.
  2. `cmp_rho_box.png` — boxplot of per-episode `rho_V_negd` for the four
     groups FB-success, FB-transport_fail, GCIQL-success,
     GCIQL-transport_fail (drop NaN). From FB+GCIQL `value_landscape`.
  3. `cmp_funnel.png` — grouped bar of normalized `outcome` fraction
     (success/transport_fail/other) FB vs GCIQL. From FB+GCIQL
     `value_landscape`.
  4. `cmp_coverage.png` — grouped bar of mean `nn_dist` by `region`,
     FB vs GCIQL. From FB+GCIQL `coverage`.
- Keep `cmp_value_rho.png`, `fb_vs_gciql.parquet`, `comparison.md`.
- A missing FB per-seed root (no `value_steps`) → skip only the plots
  that need it, with a printed warning; never crash the GCIQL aggregate.

### Component C — paper generator

`evals/paper_fragment.py` `build_fragment`:
- Replace the single `cmp_value_rho.png` include with a sorted glob over
  `cmp_*.png` in `gciql_cmp`; copy each into `figures/` and emit one
  `\begin{figure}` per file, caption = filename stem with underscores →
  spaces, title-cased (via the existing `_fig`/`_tex` helpers).
- Table (`fb_vs_gciql.parquet`) and `comparison.md` prose unchanged.
- FB-only path (`gciql_cmp=None`) unchanged.
- `scripts/build_paper_fragment.py` unchanged (already passes
  `--gciql-root`).

### Testing (TDD)

- `tests/test_gciql_profile.py`:
  - `episodes_to_frames` now returns 4 frames; update existing tests and
    add an assertion that `value_steps` has schema
    `{task, episode, outcome, d, V, transport}` and one row per step.
  - Aggregate/comparison test: build synthetic FB per-seed and GCIQL
    per-seed dirs (value_landscape, value_steps, coverage) and assert
    `comparison/{cmp_value_vs_dist,cmp_rho_box,cmp_funnel,cmp_coverage}.png`
    + `fb_vs_gciql.parquet` + `comparison.md` all exist.
- `tests/test_paper_fragment.py`: with a `gciql_cmp` dir containing
  several `cmp_*.png`, assert each filename appears in an
  `\includegraphics` in the generated `.tex` (not just one).
- Full suite green.

### Execution

1. Implement A/B/C with TDD; full `pytest tests/ -q` green.
2. Re-run GCIQL sweep (`.venv-jax-cpu`, ~6 min) to write
   `value_steps.parquet` for 10 seeds.
3. `scripts/gciql_profile_aggregate.py` → 4 `cmp_*.png` + combined frame.
4. `scripts/build_paper_fragment.py --paper-subdir fb-cube-failure` →
   paper regenerated with the four comparison figures in the user's
   layout.
5. Report the figures + the data-driven `comparison.md` synthesis.

## Risks / mitigations

- **GCIQL `value_steps` size**: 10 seeds × 5 tasks × 10 eps × ≤1000
  steps ≈ ≤0.5 M rows — trivial parquet. Bin in-memory; fine.
- **Empty `transport_fail` subset for a curve** (possible per method):
  skip that curve rather than error.
- **FB/GCIQL obs dimensionality differs** only matters for coverage,
  which is already each-method-relative (standardised to its own ref);
  bars compare nn_dist magnitudes, which is the intended read.
- **Re-run drift**: GCIQL inference is deterministic given the fixed
  PRNG seed and `temperature=0`, so re-running only adds the new
  `value_steps` output; per-episode numbers reproduce.

## Acceptance

- `pytest tests/ -q` green (existing 57 + updated/added tests).
- After the GCIQL re-run + aggregate:
  `analysis/gciql_profile/aggregate/comparison/` contains
  `cmp_value_vs_dist.png`, `cmp_rho_box.png`, `cmp_funnel.png`,
  `cmp_coverage.png`, `cmp_value_rho.png`, `fb_vs_gciql.parquet`,
  `comparison.md`.
- `PAPER/fb-cube-failure/fb_cube_failure.tex` reproduces the user's
  4-subsection layout and embeds all five `cmp_*.png` under
  "FB vs GCIQL: Shared-Axis Comparison".
