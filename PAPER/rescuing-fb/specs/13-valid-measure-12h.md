# Spec 13 — Why the successor measure still isn't "fully working", and the 12-hour path

2026-07-29 ~23:00 UTC. Companion to PHASE2-LEDGER §9.15 (pre-registration lives
there; this file holds the reasoning).

## 1. Scoreboard: what is already solved

Judged against "a trainable successor measure that works on manipulation", the
record as of tonight:

| requirement | status | evidence |
|---|---|---|
| trains stably (no ortho, no divergence) | ✅ | bounded residual; ms_abs_mean → 0.19 vs control → ~1000 (§9.3) |
| controls (task success) | ✅ 0.95 | p2+rzK50 0.948/0.950 vs stock FB 0.42 (§9.11b G1) |
| set-level calibration | ✅ ~1.1–1.2× | L1fix head ratios, Spearman 0.72–0.79 (§9.9) |
| dwell semantics real (measure ≠ reaching value) | ✅ 1.33× | §9.13 absolute-return table |
| zero-shot transfer, concentrated rewards | ✅ 0.8–0.95 gap_frac | §9.10/9.11b |
| composition | ✅ 1.35 (beats single-zone oracle) | §9.11b twozone |
| broad-bandwidth rewards | ◐ ball1.6 fixed (0.07→0.5), smooth0.8 0.33 vs 0.4 bar | §9.11b |
| **pointwise validity (M ≥ 0)** | ❌ 50% negative points, **24–29% of mass** | §9.15 D1 |
| small-box calibration (hw 0.4) | ❌ 1.2–1.9× | §9.9 |
| phase 2 off cube-single | ❌ costs −0.14…−0.19 success | §2 |
| unsupervised region discovery | ❌ region is hand-specified (effector result softens: 0.823) | §9.14 |

The method already *is* a working successor measure at the set level. What it is
NOT is a valid measure pointwise — and the small-box miscalibration and ordering
imprecision are plausibly the same defect seen at different scales.

## 2. The debate: why is pointwise validity still failing?

**Position A ("it's a bug — keep penalizing").** The hinge killed negativity
(45%→0.5%) while keeping success. Just tune it. — *Rebuttal:* §9.9 showed the
hinge inflates set mass 2.5–3.6× monotonically in λ. Sign and scale trade off
because nothing conserves total mass; the hinge shovels negative mass up without
anywhere for the excess to go. This is a missing-constraint problem, not a
missing-coefficient problem.

**Position B ("it's cosmetic — negatives live where M≈0").** If negative entries
were tiny fluctuations around zero in unvisited regions, set-level calibration
would already close the case. — *Refuted tonight by measurement:* D1 shows
negative entries carry 24–29% of absolute mass at ~36% of positive magnitude,
and negativity is *highest in the top density quartile* (0.55–0.59). These are
not fringe fluctuations; they are structural oscillations in the well-fit region.

**Position C ("it's a theorem — signed low-rank must ring").** `M(s,a,·)` lives
in span{B₁..B₅₀} with B frozen. The training target is a *boxcar* kernel
(0/1 ball, ε=0.4) over a 3-D workspace holding far more than 50 distinguishable
balls. The L2-optimal rank-50 fit of a discontinuous indicator oscillates
(Gibbs); it rings hardest exactly where the fit works hardest — high-density
regions — which is precisely D1's quartile gradient. Corollary: {rank 50,
boxcar kernel, plain L2 regression, M ≥ 0} is over-constrained. Pick three.

Position C is the one the evidence supports, and it directly generates the fix
menu:

1. **Relax "plain L2"**: keep the hinge but give it the missing anchor — a
   total-mass conservation penalty `((1−γ)·E_j[M_ij/N̂_j] − 1)² → 0`. The
   identity `E_{g~ρ}[M(s,g)/ρ(B_ε(g))] = 1/(1−γ)` holds exactly for the
   visitation fixed point (the cube is always somewhere). Sign hinge + mass
   anchor = project onto {nonneg, right total mass} while Bellman allocates
   within it. → **arm mcA**.
2. **Relax "boxcar"**: smooth the kernel. A quartic/biweight `(1−(d/h)²)²` with
   h=0.6 has comparable mass to the 0.4 ball but is C¹ — far less
   high-frequency content, far less ringing at fixed rank. Semantics stay
   "soft region dwell mass"; N̂ (column mean) and every readout generalize
   unchanged. → **arm mcC** (alone), **arm mcB** (with 1).
3. **Relax "rank 50"**: d=128 (GCIQL bilinear-d128 already beats d50 on
   success). Requires phase-1 retrain — ~7h chain; does not fit alongside the
   above tonight on the GPU budget we're allowed. Queued as follow-up.
4. **Relax "signed"** (the real algorithmic-insight candidate): make
   non-negativity structural — softplus the outputs of F *and* B so
   `M = F·B ≥ 0` by construction (non-negative factorization). Preserves the
   FB API: `Q_r = ⟨F, z_r⟩` stays exact for r ≥ 0 since z_r = Σ rⱼBⱼ ≥ 0.
   Cost: non-negative rank ≫ rank for sharp kernels, so it likely *needs*
   option 2 and/or 3 to have capacity; fresh training dynamics; warm-start
   mismatch. High risk for a 12h window → this is the pre-registered NULL
   escape hatch, not tonight's arm.

## 3. Do we need another algorithmic insight?

**Mostly no — one small one, and it is now identified.** Everything on the
"solved" scoreboard came from insights already in hand (bounded residual ×
region structure interaction §9.14; z-manifold coverage §9.11b; density
bookkeeping 1/N̂ §9.10). The last defect is not mysterious: it is the standard
price of fitting a sharp indicator with a signed truncated basis. The "insight"
is simply that **validity must be made structural (mass anchor + kernel
smoothness matched to rank), not penalized after the fact**. If tonight's arms
fail their gates, THEN a genuinely new parameterization (nonneg factorization,
option 4) is required — that outcome is pre-registered as the NULL.

Out of scope tonight (honestly parked, not forgotten): smooth0.8 (dose-response
says push reward_z_ratio / bandwidth next), phase-2 cost off cube-single
(suspect: same z-manifold mechanism — rzK50 should be re-run on cube-double; the
reward-z law improving cube-single success 0.887→0.950 is suggestive), p1-at-1M
iso-budget control (§9.13), unsupervised region discovery (the ICLR paper's
next-paper question).

## 4. The 12-hour schedule

- 23:00–00:00 — D1 diagnostic (DONE, §9.15), pre-registration (DONE), knobs
  `mass_coef` + `hit_kernel` TDD'd default-off byte-identical.
- 00:00–00:30 — smokes: defaults reproduce control loss; each knob moves the
  loss (the `shared`-whitelist lesson: prove the knob reaches the agent).
- 00:30–03:30 — 6 jobs (3 arms × 2 seeds), GPUs 0–4,7, tmux `mcons`; phase-2
  only (phase 1 reused), 500k steps ≈ 2.5h.
- 03:30–05:30 — chained per job: L1fix → L2-B → L2-A → D1 neg-mass.
- 05:30–07:00 — verdict vs gates → ledger §9.15b. Buffer ~4h for one repair
  cycle (e.g. mass_coef rescale) if a guard fails.

Gates G1–G4 + NULL: see ledger §9.15. An arm passing all four IS the
deliverable: a trainable, controlling, calibrated, non-negative,
zero-shot-transferring successor measure on cube manipulation.
