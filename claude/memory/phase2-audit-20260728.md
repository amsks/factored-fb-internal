---
name: phase2-audit-20260728
description: "2026-07-28 five-agent audit + corrected L1: the phase-2 object IS ~set-level calibrated (head-ratio 1.1-1.2x, ~1.0 after tail); the 100x verdict was a no-env.reset harness bug; nonneg penalty breaks calibration; several ledger claims corrected"
metadata: 
  node_type: memory
  type: project
  originSessionId: f9914d57-bb6c-4942-9c0c-35975b9301d4
  modified: 2026-07-28T23:44:59.223Z
---

Five parallel auditors verified every phase-2 experiment against configs, run
dirs, and raw logs; findings + corrections written to **PHASE2-LEDGER.md §9**
(the authoritative record). Companion: [[spec10-l1-calibration-fails-20260728]].

## The headline reversal (corrected L1, `results/spec10_torchfb/L1fix_*.json`)

λ=0 phase-2 + trained N̂ head: set-mass ratio **1.10–1.23×** at hw 0.8/1.6
(≈1.0× after the γ^H=0.134 tail correction), Spearman **+0.72–0.79**. The
spec-10 15% gate is ~met at those scales. **"Not a measure" is retracted at set
level.** Remaining defects: 45% pointwise negativity (sign-cancels inside
boxes), small-box (hw0.4) 1.2–1.9×, ordering precision.

**`nonneg_coef` is NOT the fix**: it kills negativity and keeps success
(0.90–0.96) but **inflates set-mass 2.5–3.6×** monotonically in λ — the hinge
pushes negatives up with nothing conserving total mass. Sign and scale trade
off; do not adopt as-is.

## Audit corrections that change claims (full detail: ledger §9)

- "First-passage required" was NEVER TESTED — the CellW0 "occupancy" control
  actually had `measure_first_passage: true` + 5 other knobs flipped. Keep only
  "ortho λ∈{1→eff.~100×} is toxic here".
- 3-arm deconfound: "privileged ball makes FB worse" BROKEN (anchor was
  ortho_coef=1000 vs control 1.0; p≈0.12). Keep: control ≪ 0.93 ⇒ the
  {bounded, no-ortho, scale_norm} bundle explains the jump (unsplit).
- Canonical phase-2 row was 0.940 (2 seeds), not 0.955 — third seed was
  cross-attributed from a phase-1 re-eval.
- azB "adopted" unestablished (no 500k run exists; opposite seed trends; part
  of margin = eval-time actor_z_greedy_frac knob).
- E2 "actor drift" and E8 α-reversal: underpowered (tie on matched seed / 1
  seed) — relabel OPEN.
- SMB B3 slope column: category error for fp/zx (first-passage vs dwell), and
  the surrogate omits 1/N̂ so the calibrated slope target is ~73, not 1 —
  visz's 25–28 is near-correct, STRENGTHENING "visz is the calibrated object".
- E5's 0.295 argmax uses 16 uniform-random actions (chance 1/16) — qualitative
  refutation stands, the number is proxy-bound.
- Wing-B harness is sound (no reset bug); multi-env table + freeze/warm-start
  verified clean.

## The trap that caused all this

**Teleport-style MC evals must `env.reset()` before every rollout** — the
gymnasium TimeLimit counter carries over silently and every rollout after the
first truncates at step 1. Detection heuristic that would have caught it: the
implied MC total mass (~1–3) was impossible vs Σγ^k≈87 (the cube is always
somewhere). Fixed in `scripts/rescuing_fb/torchfb_full_measure.py` with an
audit comment; void artifacts: `L1_p2_*.json`, `L1_nn*.json` (superseded by
`L1fix_*`).

## L2 zero-shot test RUN (spec 12, results `L2_{A,B}_p2_s4{2,3}.json`)

48 never-trained rewards, naive reward_inference, λ=0 ckpts, eval-only.
**Zero-shot transfer WORKS for concentrated novel rewards**: control gap_frac
ball0.4 0.87/0.74, ball0.8 0.70/0.61, **twozone 0.84/0.89 (composition
works)**; degrades with bandwidth: smooth0.4 ~0.5, smooth0.8 0.23/0.08,
ball1.6 ≈ floor (no steering). Prediction Spearman 0.62–0.90 (gate met for
balls ≤0.8); **scale off by exactly N̄≈0.014–0.03 = the missing 1/N̂ density
division, quantitatively confirmed**. Ledger §9.10 has the full table.

**L2w (1/N̂-weighted inference) RUN — fork resolved:** prediction scale fixed
(0.79–2.2, was 0.014–0.03) but control regressed where it worked (twozone
0.85→0.46, ball0.8 0.65→0.48) and broad rewards still ≈floor. ⇒ **naive z for
acting, weighted for prediction; the remaining defect is the ACTOR's
z-manifold** (trained z law = 50% gaussian + 50% B(single goal)). Next is a
training-side cell: extend `sample_mixed_z` with synthetic-reward inference
z's. Ledger §9.10 has both tables. Also still open: mass-conserving nonneg,
small-box calibration, L7/L8 ordering fixes, phase-2 cost off cube-single.
Quote phase-2 calibration ONLY from L1fix numbers.
