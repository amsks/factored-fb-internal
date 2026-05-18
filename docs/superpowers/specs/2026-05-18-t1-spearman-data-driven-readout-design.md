# Spec: T1 Spearman metric + data-driven Readout

**Date:** 2026-05-18
**Status:** approved (design), pending implementation plan

## Problem

The 7-seed acceptance run of the FB representation-failure profile exposed two
defects in the analysis layer (not the rollout/probe infrastructure, which
works):

1. **T1 value-gradient metric is uninterpretable.** `probe_value_landscape`
   reports a per-episode OLS slope of `V` on `−d` over the transport region.
   `V` (the ensemble FB Q-value) is unnormalised and ~O(2000) with a range of
   −112 → 4450; per-episode slopes come out O(100–1000) with a cross-seed
   std 5–20× the mean (e.g. task3 transport_fail `1154 ± 2278`). The metric
   neither has meaningful units nor separates `success` from `transport_fail`
   (often the reverse of the expected ordering).

2. **`story.md` "Readout" is a hardcoded template.** It is copied verbatim
   from the original plan and asserts T1/T4 conclusions as fact. The 7-seed
   data contradicts those claims (T1 inconclusive; T4 nn_dist ~2.2 across all
   regions/outcomes, i.e. failures are *not* more off-support). Presented
   as-is it is misleading.

## Goals

- Replace T1 with a scale-free, outlier-robust monotonicity metric.
- Make the Readout generated from the actual aggregate numbers, with an
  explicit per-probe verdict and a synthesis that only claims what the
  verdicts collectively support.
- No new third-party dependency.
- Keep the per-seed CLI / orchestrator interfaces stable apart from the one
  renamed column.

## Non-goals

- Re-deriving T2/T3/T4 metrics (they produce stable, interpretable numbers).
- Changing rollout, phase-probe, or checkpoint-loading code.
- Tuning the FB model or investigating *why* V is O(2000) (separate question).

## Design

### 1. T1 metric — `probe_value_landscape` (`evals/representation_profile.py`)

- Replace the per-episode column `slope_V_negd` with **`rho_V_negd`** =
  Spearman rank correlation between `V` and `−d`, computed over the episode's
  transport-region steps (`transport_mask`).
- Spearman implemented dependency-free: average ranks via
  `pandas.Series.rank()` (pandas is already a dependency), then Pearson
  correlation of the ranks (`numpy`). No scipy.
- Degeneracy guard — `rho_V_negd = nan` unless **all** hold:
  - `n_transport ≥ 5`
  - `ptp(d[transport]) > 1e-9`
  - `ptp(V[transport]) > 1e-9`

  Rationale: a failed transport with a flat V (zero variance) must yield
  `nan`, not a spurious `0`, so it is excluded from aggregation rather than
  diluting the mean.
- Unchanged outputs: `V_at_secure`, `V_at_end`, `n_transport_steps`,
  `episode`, `outcome`, `task`; the long-form per-step `(task, episode,
  outcome, d, V, transport)` frame is unchanged.

### 2. Aggregate — `scripts/representation_profile_aggregate.py`

- Rename `slope_V_negd` → `rho_V_negd` in the T1 grouped table and the
  success / transport-fail series.
- Replace the hardcoded Readout block with a generated **per-probe verdict +
  synthesis**. Verdict thresholds are module-level named constants, and the
  Readout text states explicitly that they are heuristic.

  | Probe | SUPPORTS (failure hypothesis) | Other verdicts |
  |---|---|---|
  | **T1** | `rho_success ≥ 0.15` and `rho_success − rho_fail ≥ 0.10` | `rho_fail − rho_success ≥ 0.10` → `CONTRADICTS`; else `WEAK` |
  | **T2** | `relabel_pos_frac < 0.05` and `topk_pct_at_goal < 0.5` | `WEAK` |
  | **T3** | `r2 < 0.5` and `placed_vs_near_acc < 0.75` | `r2 ≥ 0.7` and `acc ≥ 0.9` → `RESOLVES` (contradicts failure); else `WEAK` |
  | **T4** | `nn_dist(transport_fail) ≥ 1.25 × nn_dist(success)` (transport region) | `NEUTRAL` |

  - Constants: `T1_RHO_MIN=0.15`, `T1_RHO_GAP=0.10`,
    `T2_SPARSE_FRAC=0.05`, `T2_TOPK_AT_GOAL=0.5`,
    `T3_R2_FAIL=0.5`, `T3_ACC_FAIL=0.75`, `T3_R2_RESOLVE=0.7`,
    `T3_ACC_RESOLVE=0.9`, `T4_OFFSUPPORT_RATIO=1.25`.
  - nan-safety: if a probe's inputs are entirely `nan`/missing, its verdict is
    `INSUFFICIENT DATA` and it is excluded from the synthesis sentence.
  - Readout format (one line per probe, then a synthesis line):
    ```
    ## Readout
    _Heuristic verdicts (thresholds in representation_profile_aggregate.py):_
    - T1 value gradient: rho_success=<m>, rho_fail=<m> -> <VERDICT> (<reason>)
    - T2 sparsity: relabel_pos_frac=<m>, topk_pct_at_goal=<m> -> <VERDICT> (<reason>)
    - T3 B-resolution: r2=<m>, placed_vs_near_acc=<m> -> <VERDICT> (<reason>)
    - T4 coverage: fail=<m>, success=<m> (transport) -> <VERDICT> (<reason>)
    Synthesis: <names probes that SUPPORT vs not; no claim beyond verdicts>.
    ```
  - The existing data sections above the Readout (Sparsity/B/Value/Coverage
    mean±std) stay; only "Value gradient (T1)" relabels slope→rho.

### 3. Tests (TDD)

- `tests/test_representation_profile.py`:
  - Rewrite `test_value_landscape_slope_success_vs_fail` →
    `test_value_landscape_rho_success_vs_fail`: a strictly-monotone-V success
    episode → `rho_V_negd ≈ 1`; a constant-V transport_fail episode →
    `rho_V_negd` is `nan`. (The `_fake_ep` helper builds ≥5 transport steps so
    the guard passes for the success case.)
  - Add `test_value_landscape_rho_guard_short_episode`: <5 transport steps →
    `nan`.
  - Update the aggregate-test fixtures (`slope_V_negd` → `rho_V_negd`) and add
    assertions that `story.md` contains the per-probe verdict tokens and the
    computed numbers for a crafted input where the verdicts are known.
- Full suite (`pytest tests/ -q`) must stay green.

### 4. Re-run

Because `value_landscape.parquet`'s schema changes (`slope_V_negd` →
`rho_V_negd`), the per-seed sweep must be re-run for s3–s8, s10 (≈ same
runtime as the prior run), followed by the aggregate and paper-fragment
rebuild. The other four parquet schemas are unchanged.

## Risks / trade-offs

- Spearman ignores effect size (only monotonic ordering). Acceptable: the
  question T1 answers is qualitative ("does value rise as the cube nears the
  goal"), and the unnormalised slope's effect size was meaningless anyway.
- Heuristic thresholds are judgement calls; mitigated by making them named
  constants and labelling the Readout as heuristic.
- Re-running the sweep costs the same wall-clock as before; unavoidable given
  the schema change.

## Acceptance

- `pytest tests/ -q` green.
- Re-run per-seed sweep + aggregate; `story.md` Readout shows per-probe
  verdicts computed from the data (no hardcoded prose), and the T1 section
  reports `rho_V_negd` in [−1, 1] (or nan), not O(100+) slopes.
