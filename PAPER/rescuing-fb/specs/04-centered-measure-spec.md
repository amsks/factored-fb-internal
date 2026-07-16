# T10 — Centered (advantage) successor measure: the measure-native rescue

*2026-07-16, pre-registered. Motivation: every failed measure cell trains raw M, whose loss is
dominated by the shared occupancy baseline M̄ — the action-contingent component ΔM (what control
needs; atlas findings.tex §"control needs ΔM") is ~0.2% of Q-variance and gets ~no gradient.
Centering gives ΔM its own loss and scale. This is atlas program "idea #2" made concrete.
If T10 works, the paper claim upgrades from "we replaced the measure loss" to "we fixed it."*

## 1. Objects

On the jax GCIQL-flow substrate (cube-single state, d=50, all instruments exist):

- φ(s,a), ψ(g) as in bilinear; behavior anchor μ = the frozen goal-agnostic BC flow (exists in
  every flow cell).
- **Centered features**: Δφ(s,a) = φ(s,a) − b(s), where b(s) is trained by regression to
  sg[mean_{ã∼μ} φ(s,ã)] over K=8 flow samples per state (fallback if too slow: K=1 dataset
  action, i.e. the dueling b — but flag it, dataset-action centering is biased toward the
  behavior mode). b-loss scale-normed by sg[mean(φ²)] (the dueling v1 explosion lesson).
- **Advantage measure**: ΔM(s,a,g) = Δφ(s,a)·ψ(g)/√d.
- **Baseline**: U(s,g), two variants below.

## 2. The two arms

### T10-a — grounded baseline + centered-measure advantage (the "coexistence" arm)

- U(s,g): plain scalar **IQL expectile TD** (the proven grounding), MLP or bilinear-V head.
- ΔM: **centered measure TD** — the usual B×B measure residual computed on ΔM instead of M
  (off-diag squared term + diag attraction term, both on diff = ΔM − γ·sg[ΔM_target(s′,a′,·)],
  a′ ∼ one-shot actor), with its OWN scale-norm (sg mean(ΔM²)) and coefficient λ_ΔM ∈ {0.3, 1}.
- Actor Q-term consumes **Q = U + ΔM·z-read** — and CRITICALLY the actor's q-normalizer is
  sg|Q_full|, never sg|ΔM| (see §4 risk 1).
- Read: does a *measure-trained* advantage coexist with / substitute for the scalar-TD
  advantage? Success ≥0.85. Also run the rerank audit on the ΔM ordering.

### T10-b — measure-only, centered + split bootstrap (the pure measure-native arm)

No scalar TD anywhere:
- U leg = M̄ trained by **behavior-anchored measure TD**: bootstrap with a′ = the DATASET next
  action (SARSA) — the fixed-policy measure operator, which IS a γ-contraction (this is the
  one-step-FB insight applied to the baseline leg).
- ΔM leg = centered measure TD with a′ ∼ π (the improvement lives only in the advantage).
- Actor consumes U + ΔM as above.
- Read: best existing measure-only is 0.088. **≥0.5 = a real measure-native rescue** (headline);
  0.2–0.5 = partial, interesting; <0.2 = arm dead.

### Controls / anchors (no new runs needed)
plain bilinear IQL 0.904 · measure-only 0.076–0.088 · dueling-scalar-TD (Arch 11, other track —
floored at v2 100k; see §4) · E2 λ≤1 raw-measure aux 0.86–0.92.

## 3. Pre-registered claim boundary (joint with T8)

- T10-b ≥0.5 → "the measure objective fails on manipulation *because it is uncentered and
  policy-bootstrapped everywhere*; centering + split bootstrap fixes it." Strongest paper.
- T10-a ≥0.85 but T10-b fails → "a centered measure can *teach the advantage* but cannot ground
  the baseline; scalar grounding remains necessary" — RG-FB framing stands, with a
  measure-taught advantage as an optional component.
- Both fail AND T8 (faithful one-step FB) fails → license to print "no known form of measure-TD
  grounding supports cube manipulation" and RG-FB is the way.

## 4. Pre-registered risks (why naive centering is dangerous)

1. **The dueling warning (Arch 11 floored).** The scalar-TD analog of this arch
   (Q = U_MLP + A_bil, actor consuming A_bil only) floored at 0.05 even after the b-loss
   rescale. Suspected cause (handoff note): the actor's q-normalizer on |A| ≈ 0 (E_μ[A]≈0 by
   construction) → huge effective actor LR. T10 therefore (i) normalizes by sg|Q_full|,
   (ii) actor consumes U+ΔM not ΔM-only, (iii) watches the 50k eval — if floored like dueling,
   diagnose the actor loss scale FIRST before touching the measure terms. If the other track's
   dueling_v2 result is in by implementation time, read it first.
2. **Unsupported counterfactuals don't appear by centering.** The temporal-misalignment doc
   (grasp-gate horizon vs γ-mass; 17 drops/1M) caps what ANY ρ-supervised measure can know.
   Honest prior: T10-b plateaus in the 0.3–0.5 band rather than reaching 0.85.
3. **Scale coupling**: ΔM's natural scale is ~30× smaller than M̄'s; every term gets its own
   scale-norm (per-matrix, as in measure_scale_norm), and q_mean/value bands are watched
   (q_mean must stay ≤0 and bounded; the ψ(s,g) divergence taught us these can run away).
4. b(s) fidelity: report ‖E_μφ − b‖/‖φ‖ so a lazy b doesn't silently un-center ΔM.

## 5. Implementation notes

- New knobs in gciql_flow.py (default-off, assert-latticed): `centered_measure ∈ {off, aux
  (T10-a), only (T10-b)}`, `center_b_samples=8`, `centered_measure_coef`, split-bootstrap flag
  for the U leg. Reuses: measure_loss machinery (gf:695-845), dueling's b infrastructure
  (gf:119-172, scale-normed b-loss), the flow μ sampler (_flow_rollout).
- OWNERSHIP: gciql_flow.py is held by the RG-FB v0 agent — T10 implementation starts when it
  completes (its actor_vf_cond change is orthogonal; rebase trivially).
- 3 seeds × {T10-a λ=1, T10-b} first (6 runs, 500k, eval@50k), λ=0.3 dose only if λ=1 is
  interesting. GPUs: whatever is free of {0,2,3,4,7} at dispatch. tmux `t10`. Group names
  `GCIQL-State-Flow-CenteredMeasure-{Aux,Only}-d50`.
- After 500k: standard C3 eval + rerank audit on ΔM (is the measure-taught ordering good?) +
  the T2 DR3/co-adaptation stats on the resulting φ.
