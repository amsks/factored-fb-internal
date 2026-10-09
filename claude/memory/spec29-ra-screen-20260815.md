---
name: spec29-ra-screen-20260815
description: "2026-08-15 spec29 R-A G29-2 SCREEN: the dual-corrected index WORKS. R-A/kernel cube-single 500k s0 = .812 (mean>=350k .743) vs BC floor .111, prior z arms .096-.108, g-anchor .93 — the FIRST working z-conditioned Density-FB, ~7.5x prior z and ~87% of the g anchor. R-A/latent = .368, also far off the floor. ⚠ 1 SEED each, eval noise real (400k dip to .564), and G29-3 reward families NOT RUN so the general-interface claim is NOT yet licensed"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-15T23:37:29.476Z
---

Measure track. Spec `PAPER/Multistep-FB/specs/29-ra-dual-index.md`. Closes the
repair opened by [[spec28-p0-z-canonicalization-20260814]].

**★ G29-2 SCREEN (cube-single, 500k, seed 0, config-verified):**

| arm | @500k | mean(≥350k) | max | min(last5) |
|---|---|---|---|---|
| **R-A/kernel** (`encode=dual`, `actor_reward=kernel`) | **.812** | **.743** | .836 | .564 |
| R-A/latent (`encode=dual`, `actor_reward=latent`) | .368 | .291 | .368 | .208 |

Anchors: BC floor **.111**, g-indexed **.93**, prior z arms **.096-.108** (cs) and
**.017** (cd). **R-A/kernel is the first z-conditioned Density-FB that works** —
~7.5x the prior z arms, ~87% of the g anchor at 500k (80% on the averaged
number). R-A/latent at .368 is also far off the floor, so even the closed-form
`B̃ᵀẑ` readout partially works now where it read .108 before.

Config (both): `measure_density_z_encode=dual`, `enc_tau=5e-4`,
`ridge_lambda_mult=.01`, `z_rand_frac=.5` unchanged. Single-variable delta from
spec-28 cell (ii) except `enc_tau>0`, which `dual` refuses to run without.

**Why it worked — the chain, each link measured:**
1. Σ_B is rank-1 dominated (eff_rank 1.0002/50) ⇒ primal `E[rB]` ≈ B's dominant
   direction, near-ORTHOGONAL to the dual `(Σ+λI)⁻¹m` (cos .010-.025).
2. So `z(g)=√d·norm(B̄(g))` collapsed to a single ray (pairwise cos .998-1.000,
   PR2 1.0) and the actor was exactly goal-invariant (sens .000 vs floor .19).
3. The ridge SOLVE de-collapses it (G29-0: PR2 1.0 → 9-38; the primal-moment
   control stays at PR2 1.0, so it is the solve, not reward-weighting).
4. Under training the dual index STAYS de-collapsed: `z_cos_spread` .128→.149
   (kernel), vs .113 fully-spread / .326 collapsed references. Plateaued, did NOT
   re-collapse (rose to .157 at 185k then settled .148-.154).
5. G29-1 @100k: actor goal-sensitivity **.132** (63% of its .210 floor) vs
   **.000** for every prior z arm — and the pathology INVERTED (sphere-z′ .065 <
   goal-z′ .132; previously sphere moved actors 1.9-2.0 and goals 0.000).
   ρ(q_g,q_z) **.929** vs −.026 on the g anchor ⇒ P0b's prediction realized in a
   trained model. Latent arm did NOT lift at 100k (.006), and that dissociation
   predicted the final .812 vs .368 split ⇒ **the actor-reward choice (actual
   reward samples vs `B̃ᵀẑ`) is the differentiator**, which is what R-A predicted.

**⚠ DO NOT OVERCLAIM — three limits:**
- **1 SEED each.** Spec 29 promotes discriminating cells to cube-double + s1.
- **Eval noise is real**: kernel dipped to .564 at 400k then read .812 at 500k.
  Quote .812 (500k, project convention) AND .743 (mean ≥350k); a single-eval
  reading at exactly 500k could be a favourable draw.
- **G29-3 NOT RUN.** The zero-shot reward-family eval through the dual interface
  is the binding general-reward-interface claim. Goal reaching alone does not
  license that sentence in the paper.
- This is a **NEW ARM**: archived z24/z25 numbers are not comparable to it in one
  table without saying so.

Related: [[spec28-p0-z-canonicalization-20260814]], [[spec27-q2-z-readout-20260814]],
[[spec25-z-coupling-20260813]], [[spec24-z-index-20260813]].
