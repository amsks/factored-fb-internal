# Findings — privileged-z probe on coverage-balanced FB (2026-05-25)

Probe: `scripts/privileged_z_probe.py` (run in a `feat/cube-analysis-pipeline` worktree
against the `/dev/shm` checkpoints of the running coverage-PRIORITIZED sweep:
α=0.5, ortho_coef=1000, weight_diag=on, weight_z=off). Task 1, transport-phase states,
near-goal subset = distance ≤ 40th percentile. Metric: Spearman(V, −cube→goal distance).

It asks: replace the actual goal vector `z=B(g)` (reward inference) with the BEST free
`z*` (optimized against the frozen forward map F). `z*` ≫ `z=B(g)` ⇒ B's goal inference
is the bottleneck; `z* ≈ z=B(g)` ⇒ the forward path F can't resolve the terminal at all.

## What we saw

| near-goal Spearman(V, −d) | reweighted α=0.5 | baseline (spec, ortho=1000) |
|---|---|---|
| `z=B(g)` — actual goal vector | **+0.056 ± 0.060** (10 seeds @600k; range −0.04…+0.17; 5/10 > baseline) | 0.038 (chance) |
| `z*` — best-free-z (forward-path ceiling) | **+0.277 ± 0.034** (10/10 > baseline; range 0.22…0.32) | 0.201 |

Single-seed trend was noisy / non-monotonic (seed 0: `z=B(g)` 0.110→0.071, `z*` 0.373→0.294
from 200k→400k), which is why we aggregated over 10 seeds at 600k.

## Interpretation

- **`z*` (the achievable ceiling) is reliably lifted**: 0.28 vs 0.20, all 10 seeds above
  baseline. The forward path F, retrained under ρ′, exposes more recoverable terminal geometry.
- **`z=B(g)` (the vector the value actually uses) did NOT meaningfully improve**: 0.056 ± 0.060
  straddles chance (only 5/10 above the 0.038 baseline). The earlier seed-0 0.11 was the high
  tail of the noise.
- The gap `z=B(g)` (0.056) ≪ `z*` (0.277) is wide and **unclosed** → B's goal inference is
  still the bottleneck, and this (prioritized) reweight did not fix it.
- Even the lifted `z*` ≈ 0.28 is far below F's featurizer geometry (ridge `left_enc→d` R² ≈ 0.53
  in the spec) → the bilinear `V = Fᵀz` is **also** lossy. Two ceilings remain.

## Implication

The value uses `z=B(g)`, not `z*`. Since `z=B(g)` near-goal is ≈ chance/baseline, this predicts
**no terminal-success gain over baseline** for the prioritized variant — consistent with eval
success sitting ~0.14 (baseline range) and the "learning curve looks like normal FB" observation.

Per the spec decision tree this is closer to the **kill** outcome for coverage-balancing as a
success fix: B(g) isn't responding and the bilinear is lossy, so the binding constraint looks like
B's goal-encoding capacity and/or the forward path (rank/architecture), not the ρ measure alone.

## Caveats
- 600k, not converged (1M); trend flat-to-down, not encouraging.
- The matched ortho=1.0 "normal FB" probe errored (NA) — comparison is to the spec's reported
  ortho=1000 baseline (the correct control for this experiment anyway).
- Task 1 only; α=0.5 only; weight_diag=on / weight_z=off (the prioritized variant).
- Probe methodology: `_spearman` patched pandas-free (numpy average-rank) and the factorization
  ridge reference made optional, to run under the torch `.venv` without pandas.

## What this motivates (ties to the code review)
1. The **weight_diag ablation** (pure ρ′ vs prioritized) — the current run is prioritized; pure ρ′
   (diagonal unweighted) is the clean reference-measure test.
2. **prioritized + weighted-z**: give F and the actor rare-goal `z=B(g)` exposure — tests whether
   `z=B(g)` fails to sharpen because F/π never train on rare-goal z's. (Prioritized for the sweep.)
3. If neither moves `z=B(g)` and success, pivot to the rank/architecture lever (z_dim/L_dim, aux
   geometry head on B/F), since z* ≪ 0.53 implicates the bilinear too.
