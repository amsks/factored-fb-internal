---
name: spec10-l1-calibration-fails-20260728
description: "CORRECTED 2026-07-28 night: the L1 65-215x miscalibration was a HARNESS BUG (no env.reset between MC rollouts); negativity (~45% M<0) survives and is real; fixed harness re-runs = L1fix_*.json"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a9f88cb-6be5-4d02-a66c-f8dcdd47169f
  modified: 2026-07-28T21:18:47.801Z
---

**⚠ SUPERSEDED IN PART by the 2026-07-28 five-agent audit — see
[[phase2-audit-20260728]] and PHASE2-LEDGER.md §9.**

## What was WRONG (do not cite)

The headline "set-mass off 65–215×, Spearman +0.09…+0.35, gate fails at
6,500–21,500%" was **~90% a harness bug**: `torchfb_full_measure.py` never
called `env.reset()` between MC rollouts, so the gymnasium TimeLimit counter
carried over and **31 of 32 starts truncated after ONE step** — the "MC ground
truth" was a single-step occupancy. Verified in `L1_p2_s4{2,3}.json`: start #0
mc_sum≈162–182 (real), every other start mc_max = exactly 1.000. On the one
clean start, model/MC ≈ **2–4×**, consistent with the model-side total mass
(190 vs ~87 measurable in H=200). The "self-check exonerates the harness" logic
was flawed: it was model-side only and in hindsight *condemned* the per-box
numbers (implied MC total ~1–3 ≪ 86.6, impossible — the cube is always
somewhere).

**Fix applied 2026-07-28:** `env.reset()` before every MC rollout (comment at
the site cites the audit). The four `L1_nn*.json` (written ~21:10 by the tfbnn
auto-run) used the buggy harness — **void, do not read**. Fixed re-runs on all
6 phase-2 ckpts (control + nn arms) → `results/spec10_torchfb/L1fix_*.json`,
tmux `l1fix`.

## What SURVIVES (MC-free, model-side)

- **Negativity is real**: frac(M̃<0) ≈ 30–46% on the λ=0 phase-2 object; a
  measure is non-negative by definition. The `nonneg_coef` hinge penalty kills
  it (λ=1 → ~3%, λ=10 → ~0.6%) while preserving success (0.90–0.96).
- **N̂ head**: corr 0.994 vs exact kernel counts, median rel-err 23.7%, +29%
  bias.
- Task success scores the actor, not the measure (E5's qualitative point
  stands, though its 0.295 argmax number is a uniform-random-action proxy).

**How to apply:** teleport-style MC evals MUST reset the env first (TimeLimit
counter carries over silently — exit is `truncated=True` after 1 step, easy to
miss). Before trusting any calibration verdict, sanity-check the implied MC
total mass against Σγ^k. The real calibration state of the phase-2 object is
whatever `L1fix_*.json` says — likely ~2–4× off, not 100×.
