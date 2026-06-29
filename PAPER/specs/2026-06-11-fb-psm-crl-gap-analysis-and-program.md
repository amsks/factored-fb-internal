# Why successor measures fail at manipulation — formal gap analysis & research program

**Date:** 2026-06-11 · **Status:** working synthesis (living doc)
**Inputs:** all probes in `analysis/` (phase funnel, factorization, counterfactual value,
aliasing, task-distribution, z-decode), the IWR paper
(`~/git/baselines/Factored_URL_in_Manipulation`), loss-magnitude audits from training
logs, GCIQL-Flow actor ablation results.

---

## 0. New evidence pulled today

- **GCIQL-Flow (IQL critic + FQL flow actor, 10×1M state): IQM 72.3** — *better* than
  GCIQL-TD3 (68.3), and far above FB-FlowBC (50.0), PSM-o1000 (36.7), RLDP (22.7).
  ⇒ **The flow/BC actor stack is not the bottleneck.** With a good critic the same
  actor reaches 72%. The whole FB-family deficit lives in the value path
  (representation × readout × inference).
- **Loss decomposition at convergence** (from train logs):
  - PSM-o1000: total −63,999 ≈ 1000·orth(−63.96) + measure(−38) → ortho term **~1700×**
    the measure terms.
  - FB plain (ortho1000): total −24,348 ≈ ortho(−25,024) + measure(+676) → **~37×**.
  - And turning ortho UP **helps** (FB cube winner ortho1000; PSM 36.7 vs 30.7).
  ⇒ The best configs largely *suppress* the method's own representation-shaping
  objective in favor of generic whitening. The FB/PSM measure loss contributes
  *negatively* on cube. (Caveat: loss scale ≠ gradient scale — gradient audit pending.)

## 1. The three objectives, formally, and where they differ

All three estimate functionals of the discounted successor measure
M^π(s,a,·) = Σ_t γ^t P(s_{t+1+}=· | s,a,π).

| | object represented | parameterization | policy family | task inference |
|---|---|---|---|---|
| **CRL** | **log** dM^π/dρ̄ (NCE-optimal critic) | bilinear φ(s,a)ᵀψ(g) | single π (goal-cond.) | none (g is the task) |
| **FB** | dM^{π_z}/dρ (the **density itself**) | bilinear F(s,a,z)ᵀB(g), E_ρ[BBᵀ]=I | joint family {π_z} | z_r = E_ρ[r·B] (linear) |
| **PSM** | M projected on learned basis φ | ψ(s,a,z)ᵀz, ψ = SF of φ (TD, φ detached) | family {π_z} | reward regression on φ |

Three structural differences that matter for manipulation:

**(a) Density vs log-density.** FB/PSM represent the *density* linearly; CRL the
*log*. In near-deterministic, mode-switched manipulation the successor density has
enormous dynamic range concentrated on thin reachable manifolds; a bounded bilinear
form must spend rank on dynamic range, while a log-form compresses it. (Eysenbach
2024's linear-Gaussian story = the regime where both are easy; IWR shows manipulation
leaves that regime.)

**(b) What the L2(ρ) objective optimizes.** The FB measure loss is an L2(ρ)-weighted
regression of the resolvent (I−γP)^{-1}; its optimum at rank d is the top-d singular
space of M under ρ — i.e. capacity is allocated by **temporal persistence under the
behavior process**, weighted by ρ-mass. In play data, arm/cube *position* is the slow,
persistent variable; the interaction variable ω (free/held/slipping) toggles quickly
and occupies little ρ-mass. Spectral truncation therefore buries ω first. This is the
formal mechanism behind our measurements: FB's B has lift-R²=0.02 while a
whitening-dominated φ (PSM ortho1000, persistence-agnostic) preserves lift (0.81).
**The method's own objective deletes the control-critical variable; generic whitening
keeps it.** This *explains the ortho1000 paradox.*

**(c) The bilinear/low-rank readout.** Both FB and CRL share Q (or energy) =
inner product of two embeddings. IWR's switched-affine analysis says the future
representation is mode-conditioned (ψ_{t+1} = A_ω ψ_t + ω b_t); a single bilinear form
smooths across the mode boundary. Our counterfactual probe quantifies it: bilinear
values rank in-hand grasps at chance (PSM 0.49, RLDP 0.52, FB ~0.6) while an MLP
critic on the same data reads 0.89.

## 2. Where exactly FB/PSM fail — four candidate bottlenecks, with verdicts

- **B1 Representation (FB-specific): CONFIRMED.** The measure loss orders features by
  persistence → ω invisible in B (z-decode). Killing the measure loss with ortho1000
  *improves* success. PSM's φ escapes this partially (different basis objective +
  whitening) — its φ encodes everything (lift 0.81, contact 0.96, hand-dist 0.76).
  → The 2×2 factorial now running (FB-z128, PSM-z50, both ortho1000) settles
  algorithm-vs-capacity for this.
- **B2 Readout: CONFIRMED, the binding constraint for PSM.** Basis contains the
  signal (R²(B→d)=0.90, separability 0.97); bilinear value reads none of it
  (cf-AUC 0.49, aliasing 0.61, unchanged across ortho1→1000 despite +10pp success).
- **B3 Task inference: CONFIRMED secondary.** z_r=E_ρ[rB] is the L2(ρ)-projection of
  r on span(B); it is the *optimal* z only if m=FᵀB exactly. Under truncation error the
  optimal z solves a weighted least-squares problem instead. Evidence: privileged-z
  probe (best free z* ranks 0.28 near-goal vs z=B(g) at 0.04 ≈ chance); the
  coverage-balanced ρ′ reweight raised the F-side ceiling but z=B(g) stayed at chance
  (the pre-registered kill).
- **B4 Actor: RULED OUT.** GCIQL-Flow ≥ GCIQL-TD3 with the same flow/BC stack.
  Note the BC prior *does* carry the manipulation competence (grasping) for everyone —
  the value only steers — which is why chance-level values still yield 25–50%: failure
  concentrates where steering conflicts with holding (the funnel's maintain/transport
  collapse, value blind to grasp retention).

## 3. The deep direction: renewal decomposition of M across the interaction boundary

The unifying object. For mode-switched dynamics with interaction set I (contact
manifold) and first hitting time τ from (s,a), the strong Markov property gives
**exactly** (no Gaussian idealization):

  M^π(s,a,·) = M_pre^π(s,a,·) + E[ γ^τ 1(τ<∞) · M^π(s_τ, ·) ]

where M_pre is the successor measure of the process **killed at I**, and the second
term factors **through the interaction manifold**: (first-passage kernel onto I) ∘
(successor measure from I). Within each mode the kernels are smooth and genuinely
low-rank; the *product through a low-dimensional boundary* is what a flat bilinear
FᵀB cannot express at low rank — rank-d L2(ρ) truncation smooths exactly the boundary,
producing the aliasing we measure. IWR's switched-affine model is the local
linearization of this; their resampling fix concentrates data at the boundary, our
proposal makes the *architecture* respect it.

**Provable claim to write (Proposition 1):** for a two-mode switched system with
cross-mode coupling only through I, a mode-factored representation
m = Σ_ω 1[ω] F_ωᵀB_ω + (bridge term h(s,a)ᵀ C(g)) achieves ε error with rank
d₁+d₂+r_I (r_I = rank of the boundary coupling, small if I is low-dim), while a flat
bilinear form needs rank scaling with the product structure / suffers Ω(coupling mass)
error at the same rank. Sketch via SVD of the cross-mode block of M.

### Solution program (ranked by depth × feasibility)

1. **Bridge-FB / mode-factored successor measure** (the deep one). Gate ω̂(s,a)
   (learned from contact features, or IWR-style distance heuristic to start);
   per-mode (F_ω, B_ω) + explicit first-passage bridge factor. Derivation +
   prototype. Zero-shot inference decomposes per mode: z_ω = E[r B_ω].
2. **Error-aware task inference** (quick mathematical win, no retraining).
   Lemma: under model error, optimal z_r = argmin_z E_ν[(F(s,a,z)ᵀz − Q̂_r)²]
   (GLS with the F-metric), reducing to E_ρ[rB] iff m=FᵀB exactly and ν=ρ.
   Implement on existing checkpoints (relabel buffer gives Q̂ samples via
   on-policy rollouts or k-step returns); measure success delta.
3. **Matched-readout control** (decisive for B2, paper already calls for it):
   frozen PSM-o1000 basis + small MLP critic g(ψ,z) trained on relabeled data;
   report cf-AUC + success. If MLP-on-frozen-basis ≈ GCIQL, B2 is proven causal.
4. **Log-linear FB** (reconciles FB↔CRL): m = exp(FᵀB); derive the inference rule
   (variational, no longer closed-form). More speculative; do after 1–3.

### Experiment queue (cheap → heavy)
- [x] GCIQL-Flow comparison (B4 ruled out) — done 2026-06-11.
- [x] Loss-magnitude decomposition (ortho dominance) — done.
- [x] **Gradient-scale audit — done 2026-06-11** (`scripts/grad_scale_audit.py`,
      `analysis/grad_audit/grad_scale_audit.json`; mean over 8 real batches, per ckpt,
      grads on B/phi params; g_total = g_measure + C·g_ortho by linearity).
      Result REFINES the "switched off" story:
      - **fb_plain (C=1000): ratio C||g_o||/||g_m|| ≈ 1.2–1.9 across ALL of training**
        — FB's measure term is NOT suppressed; gradient-wise it is comparable to the
        whitening. Yet B still ends lift-blind (R²=0.02) ⇒ the lift-blindness is the
        measure objective's *active preference* (persistence ordering), not an artifact
        of suppression. Strengthens B1.
      - **psm_o1000: ratio 26 → 4.5** (declining) — whitening dominates EARLY and sets
        the basis geometry; the measure term never overcomes it ⇒ interaction info
        preserved (lift 0.81). **psm_tuned (C=1): ratio 0.01–0.02** — measure fully in
        control ⇒ basis collapses (PR 68/128, lift 0.65) and success is worse (30.7 vs
        36.7). Gradient-level *causal* account of the ortho1000 paradox for PSM.
      - cos(g_m, g_o) ≈ 0–0.2 everywhere: the two terms shape nearly orthogonal
        subspaces (not direct opposition — the measure term repurposes whatever
        capacity whitening leaves unconstrained).
- [ ] Q-steering audit at eval: rollouts with the Q-guidance ablated from the flow
      actor vs full — how much the value steers, per method.
- [x] **(2) error-aware z-inference ladder — done 2026-06-11** (`scripts/z_inference_ladder.py`,
      `analysis/z_inference/`; 3 methods × 10 seeds × 5 tasks × 20 eps, paired vs L0).
      **VERDICT: inference is NOT a unilaterally recoverable lever.**
      - L1 (GLS): ≈0 at ortho1000 (B already white: fb +2.7 n.s., psm −0.3); at ortho1
        (psm_tuned) it is **catastrophic** (−22.0*): the GLS z is mathematically the
        better reward projection but lands outside the actor's z-training distribution
        (cos_to_L0 0.44) and the policy collapses.
      - L2/L2b (Bellman-consistent z): −13 to −34 everywhere; success is ~monotone in
        cos_to_L0 across all methods/rules.
      - **Mechanism:** at eval, z is a *policy-family index*; the actor's z→behavior map
        was shaped by the training-z distribution (B(g)-cloud + sphere), with weak
        Q-coupling (BC-dominated actor). Improving the z for the VALUE moves it off the
        index manifold the POLICY understands. Together with the privileged-z result
        (better-ranking z* exist but z=B(g) ≈ chance), this shows the value's task
        embedding and the policy's task index have **decoupled semantics** — a training-
        level defect no inference rule can fix. ⇒ raises the priority of training-level
        fixes (Bridge-FB; goal-distribution interventions) over eval-time patches.
- [x] **(3) MLP matched-readout control — done 2026-06-11** (`scripts/mlp_readout_control.py`,
      `analysis/readout_control/`; MLP Q(norm_obs,a,z) + IQL-expectile TD, frozen
      PSM-o1000 basis, FB-style goals: uniform batch-perm + 0.2 diag; 3 seeds × 150k;
      scored on the stored counterfactual grasps).
      **VERDICT — REFRAMES B2: the readout FORM is not the binding constraint.**
      Final cf-AUC 0.477 / 0.494 / 0.524 ≈ chance, statistically identical to the
      bilinear on the same grasps (0.478/0.527/0.423). Training curves flat-to-declining
      (0.53→0.48), so not undertrained. An MLP trained with **FB's goal distribution**
      converges to the same controllability-blindness as the bilinear. GCIQL's 0.89 uses
      the same MLP form but **trajectory-future (HER-style) goals** (p_trajgoal=0.625).
      ⇒ The blindness is carried by the **training goal distribution / TD signal**
      (uniform-buffer goals make post-acquisition controllability irrelevant to the
      regression target), not by the inner-product form per se. The readout-form 2×2 is
      now: bilinear+uniform 0.49 / MLP+uniform 0.50 / MLP+traj 0.89 (GCIQL) /
      **bilinear+traj = MISSING CELL** (≈ retraining FB-family with trajectory goals —
      converges with the IWR/zHER direction in the IWR-FB spec).
- [ ] 2×2 factorial readout (running): FB-z128 / PSM-z50 @ ortho1000 → algorithm vs
      capacity for B1.
- [ ] (1) Bridge-FB prototype.
