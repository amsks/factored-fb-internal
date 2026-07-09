# mdose campaign results (2026-07-08): the why-answer for Bellman-on-representation

**Question:** why does adding a Bellman/TD constraint on the representation itself (FB
successor measure on the critic's φ/ψ) wreck bilinear GCIQL (0.90 under scalar IQL TD)?

**Campaign:** 21 runs @500k (7 cells × 3 seeds), tmux `mdose`, launched 2026-07-08.
Commits: `9d87bc2` (ψ probe), `f10356e` (measure+ddpgbc fix, configs), `5e8c912` (z-mix knob).
Probe data: `analysis/e2_psi_probe/all18.json` (E2 cells), `analysis/e2_psi_probe/mdose15.json`.
Guardrail: critic q_mean ∈ [−100,10] full-history; λ=10-sgphi 1/3 and λ=30-sgphi 3/3 violated
(flagged below); all other cells clean.

## The dose curve (cube, critic legs; success = eval overall_success mean/3 seeds @500k;
## cos/AUC = e2_measure_psi_probe on 50k relabeled dataset states, 5-task mean)

| cell | success | cos(z₀,ψg) | AUC exact-z | AUC moment-z |
|---|---|---|---|---|
| base bilinear-d50 | 0.90 | 0.883 | 0.912 | 0.955 |
| sgphi λ=1 (E2) | 0.84 | 0.815 | 0.810 | 0.852 |
| sgphi λ=3 | **0.15** | 0.920* | **0.727** | 0.746 |
| sgphi λ=10 (1 q-viol) | 0.02 | 0.786 | 0.646 | 0.636 |
| sgphi λ=30 (3 q-viol, uninterpretable) | 0.00 | 0.510 | 0.536 | 0.522 |
| **shared λ=10** | **0.92** | 0.931 | 0.962 | 0.960 |
| shared λ≤1 (E2) | 0.85–0.93 | ≈base | ≈base | ≈base |

*cos rising at λ=3 while AUC collapses = ψ collapsing toward a dominant subspace (all
ψ-derived vectors correlate); rank-AUC of the readout, not cos, is the load-bearing metric.

Value-leg (control) trace: healthy at λ=3 (aucE 0.952) while the critic is already broken
(0.727) → the failure originates in the CRITIC's ψ, then propagates to the value net via the
Q-target chain by λ=10 (value aucE 0.696) and kills actor extraction.

## Domain flip (occupancy mechanism)

| cell | success |
|---|---|
| cube measure-only (flow) | 0.099 |
| cube measure-only best graft (C1 fbz) | 0.255 |
| **antmaze measure-only (ddpgbc)** | **0.79** (0.78/0.80; s0 pending) |
| antmaze IQL-bilinear (ddpgbc anchor) | 0.837 |

The identical objective family reaches ~95% of the scalar-TD anchor (0.79/0.837) on antmaze. The
Bellman-on-representation constraint is NOT intrinsically broken — it fails specifically
where the data occupancy is arm-dominated and value structure is interaction-conditioned
(cube). Direct support for the ρ-weighted-L2 mechanism and the project's main thesis.

## Fidelity rescue refuted (C5)

| cell | success | reads |
|---|---|---|
| C5a = C4 + measure_z_random_frac 0.5 | 0.069 | z-coverage alone: no rescue (≤ plain measure) |
| C5b = C5a + actor_cond=z (D1 knob) | 0.165 | = D1's 0.165 exactly; zmix adds nothing |

FB's mixed z-sampling (the last audited graft-fidelity gap) does NOT close measure-graft →
torch-FB-48. Residual candidates: two-tower F, actor trained across mixed z, torch HPs —
but the domain flip makes the implementation residual secondary to the story.
(Termination masking, the other audit caveat, was traced and dropped: tM is stop-gradded ⇒
absorbing-goal masking is gradient-inert on the diagonal.)

## The answer (three parts)

1. **Not intrinsic:** the measure objective supports goal-reaching (antmaze 0.79). Not scale
   (E2 guardrail), not structure (B1/two-tower/B4 = 85–87 on the 90-critic).
2. **On cube it is a fixed-point geometry conflict on ψ:** the measure's preferred backward-map
   geometry degrades the scalar-TD value ranking dose-responsively (AUC 0.91→0.81→0.73→0.65),
   with the success cliff at λ∈(1,3]; the damage starts in the critic's ψ (the consumed leg)
   and propagates value-net → actor. Mechanism consistent with ρ-weighted density fitting on
   arm-dominated occupancy vs the goal-task-weighted structure the readout needs.
3. **Scalar TD masks the conflict via φ co-adaptation with ≥10× margin:** shared legs at λ=10
   hold full success (0.92) and base geometry (AUC 0.962) where the quarantined dose is fatal.
   This is why the E2 interpolation looked "inert," and why measure-as-aux neither helps nor
   hurts: it is out-competed, not compatible.

Corollary (E1a + probe): the moment-z out-ranks the exact goal embedding on the scalar-TD
critic (AUC 0.955 vs 0.912) — the bilinear factorization is zero-shot-capable as-is; nothing
about FB's inference API requires the measure objective on this domain.
