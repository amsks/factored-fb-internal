---
name: audit-sf-measure-20260706
description: "READ BEFORE citing SF/measure results (2026-07-06 deep audit, code+wandb): SF-feature-TD collapse claim RETRACTED (semantic confound — no reward in loop, linear proxy r=<s,g> has wrong argmax; sf_loss converged fine); IQL+measure interpolation 18/18 CRASHED (scale blowup Q 20→512, grads 12k) = empty cell needing scale control; B1 fb_forward_z-under-IQL ALREADY RAN = 0.848/0.836/0.848 (untried-doc stale); exp4 actor_z_infer null is an ARTIFACT (z ≈ 99% noise: 10%-random mask @ psi at B=1024)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2991a478-54cc-4bf7-945b-90376b93d2c7
---

**2026-07-06 deep-dive audit** of the SF-feature-TD + measure-graft experiments (code `tools/wandb_mode_shim/gciql_flow.py` + full wandb pull). Follows [[handoff-sf-ddpgbc-20260706]] and corrects it.

## Corrections to prior conclusions
1. **SF-feature-TD "collapse in the SF critic/representation" — RETRACTED.** Rewards never
   enter the SF pipeline; readout ⟨φ,g⟩ (identity) / frozen-kernel is a linear proxy whose
   argmax ≠ goal (config comment even said "confounded — identity b makes reward linear g.s"
   but conclusions dropped it). Wandb signature confirms: sf_loss converged (~5), φ/Q scales
   healthy (~29/~24) in agnostic cells → well-fit critic for the WRONG reward, not a collapse.
   Ref (raymondchua): DMC locomotion rewards ARE ~linear in state (premise valid there);
   their working sf_simple fits task weights by reward regression (line 322). The
   flow-vs-DDPGBC SF control says nothing about actors (both on a wrong-reward critic).
   Honest reading: goal indicators aren't linearly expressible in identity/frozen bases →
   strengthens the readout-side thesis (FB's learned-B+ortho is the device meant to fix this).
   Secondary bugs: goal-dependent masks in the agnostic-φ target (datasets.py:242);
   min(q1,q2) over independent SF heads (ref uses min-L2-norm target, known gap #2).
2. **IQL+λ·measure interpolation: 18/18 runs CRASHED** (mc0.3/1.0/10, both-legs + SA-legs).
   SA-legs die ≤15k; both-legs mc1.0/10 die ~305–340k with Q inflating 20→512, grad_norm 12.4k
   (anchor: |Q|~22, grad ~41). The unbounded ⟨F,z⟩ inflation (torch ISFB finding) is on the JAX
   side too, uncontrolled: MeasureOnly/C4/FBz/D1 all run at actor q_mean 2.5k–4.8k, grads
   12–24k. → interpolation is an EMPTY cell; rerun needs scale control (coefs {.01,.03,.1},
   measure loss normalized by sg|M|, or stop-grad/residual-head variants).
3. **B1-FBzIQL (fb_forward_z under IQL) ALREADY RAN: 0.848/0.836/0.848 @500k** — the
   untried-design-space.md listing it as untried is STALE. The z-indexed IQL value family
   (USFA-IQL core cell) works at ~85 (like two-tower 86 / B4 87).
4. **Exp4 actor_z_infer ≈ 0 is an ARTIFACT, not a negative**: `_infer_actor_z` z = (10%-random
   mask + eye) @ ψ → at B=1024 ~102 random ψ's vs 1 true → z ≈ 99% noise; models a dense
   random reward, not the goal indicator. Proper inference-consistent actor conditioning
   (weight ψ_j by relabeled r(s_j,g_i), matching G4 eval z=E[rψ]) is UNTESTED and is the
   direct fix candidate for the G4 93→7 zero-shot gap.

## Trustworthy after audit
IQL anchors (90), B-cells (B4 87, two-tower 86, B1 85), ortho1000-kills-IQL (7.2). Measure
band 9.9→25.5→16.5 = real as-implemented, with caveats: uncontrolled scale regime; z never
sampled as random directions (only ψ(goal-mix)); no termination masking in measure TD (FB-
faithful but ≠ IQL's absorbing goal). Mechanistic datum from the pull: measure off-diag TD
residual 6347 (z-blind φ) vs 266 (fb_forward_z) — density unfittable without task index, yet
fitting it well still only yields 25% success (supports supervise-what-you-consume /
projection-mismatch mechanism).

## Session theory context (brainstorm preceding audit)
Q1 mechanism candidates: (1) supervise-what-you-consume (vector/density supervision, scalar
consumption); (2) ρ-weighted L2 vs task-weighted error (arm-dominated occupancy on cube;
coincide on antmaze — explains domain flip); (3) B's triple duty + ortho-isometry vs value
objective tension; underneath: Q* is piecewise-max in reward, not linear → linear readout and
max operator fundamentally trade off. Q2: winning form φ(s,a)·ψ(g) is BVN's *baseline* (BVN
claims it's too weak — contradicted on cube); bilinear GCIQL is already an SF trained by
scalar TD; G4 z=ψ(g) 93 proves factorization functional. Q3 ranked: A1 inference-consistent
actor + A2 regression z=Cov(ψ)⁻¹E[rψ] (post-hoc whitening, not ortho loss) on the 90 critic;
B1/USFA-IQL (core cell now known = 85); quarantined measure head (stop-grad); scale-controlled
interpolation.

[[handoff-sf-ddpgbc-20260706]] [[handoff-sf-featuretd-20260704]] [[sota-bilinear-paper]]
[[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]]
