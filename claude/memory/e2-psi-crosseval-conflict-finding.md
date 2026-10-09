---
name: e2-psi-crosseval-conflict-finding
description: "E2 prize cross-eval DONE (2026-07-07): measure objective is NOT inert — stopgrad-phi cells show ACTIVE psi-geometry degradation (cos 0.893→0.815, aucE 0.921→0.810 at λ=1) while shared-legs cells are unchanged → scalar IQL TD MASKS a real objective conflict by letting phi compensate; measure never HELPS psi in any cell"
metadata: 
  node_type: memory
  type: project
  originSessionId: 942da339-fbce-4c5a-84ad-c7933b2fc811
---

**E2 ψ cross-eval (option (a), value-level) — DONE 2026-07-07.** Completes
[[e2-measure-mix-campaign]] (its last open item). Script
`scripts/e2_measure_psi_probe.py` (committed); results `analysis/e2_psi_probe/all18.json`;
raw log `/dev/shm/e2probe.log`.

## Protocol
18 ckpts @500k (3× plain Bilinear-d50 base + 5 MeasureMix cells ×3 seeds), 5 cube tasks,
50k relabeled DATASET states (E1a-faithful, no on-policy caveat). Per ckpt/task/leg:
dataset-moment `z_v0=Σ r·ψ(s)`, `cos(z_v0, ψ(g))`, rank-AUC of bilinear readout under
exact ψ(g) and z_v0. **CRITIC legs = the treated leg** (E2 ran `measure_legs='critic'` —
gotcha: `infer_ablation_gciql.py` as-is probes the VALUE net = wrong leg); value legs = control.

## Result (cell means, critic legs: cos_v0 / auc_exact / auc_v0)
| cell | cos | aucE | auc0 |
|---|---|---|---|
| base d50 | .893 | .921 | .955 |
| l001 shared | .888 | .930 | .960 |
| l01 shared | .885 | .914 | .948 |
| l1 shared | .896 | .917 | .966 |
| l01 **sgphi** | **.835** | .926 | .964 |
| l1 **sgphi** | **.815** | **.810** | **.852** |

Value-leg control: ALL cells ≈ base (cos .898–.903, aucE .951–.958) — value net untouched, probe clean.

## Reading — flips "coexists inertly" into "conflict masked by scalar TD"
1. **Shared-legs (φ free): geometry unchanged** at every λ ≤ 1 → the scalar IQL gradient
   dominates/absorbs the measure gradient; success flat 0.85–0.93 for the same reason.
2. **stopgrad-φ (measure isolated to ψ): active degradation, dose-responsive** — cos drops
   .893→.835 (λ=.1) →.815 (λ=1), and at λ=1 the readout itself breaks (aucE .810, auc0 .852);
   all 6 sgphi seeds below the base seed range (.883–.900). The measure's preferred ψ geometry
   CONFLICTS with the scalar-TD value geometry; you only see it when φ can't co-adapt.
   (Matches: l1-sgphi was also the lowest success cell, 0.80/0.86/0.87.)
3. **Measure never helps**: no cell improves cos or AUC over base → no z-inference-geometry
   benefit from the Bellman-on-representation constraint at any dose. Combined with
   measure-only 9.9–25.5: alone it builds a worse-for-readout geometry; grafted it is
   suppressed (shared) or corrosive (isolated).
4. Side datum: auc_v0 > auc_exact everywhere (base .955 vs .921) — dataset-moment z is a
   better state-ranker than the exact goal embedding (consistent with E1a 0.955).

## Why-question status (bridge thesis)
The Bellman constraint on the representation "messes up" bilinear GCIQL because its
fixed-point geometry for ψ is a DIFFERENT geometry than the scalar-TD one (mechanism:
supervise-what-you-consume / linear-density vs max-Bellman tension), not because of scale
(E2 killed that) or structure (B1/B4 85–87). Next discriminators: λ>1 sgphi dose-response
(find the cliff), antmaze measure-only (occupancy hypothesis), fidelity knobs
(z-sampling mix + termination masking) to close measure-graft 25.5 → torch-FB 48.

[[e2-measure-mix-campaign]] [[audit-sf-measure-20260706]] [[zeroshot-gciql-e1a-works]]
