---
name: spec10-full-measure-brainstorm-20260726
description: 2026-07-26 spec10 written — plan to promote phase-2 ball object to a full successor measure; keystone = M̃/N̂ density ratio + z-search inference; E3 ridge NEGATIVE
metadata: 
  node_type: memory
  type: project
  originSessionId: 90c8189d-90e2-4163-a00e-c2b245ce160f
  modified: 2026-07-26T07:20:47.130Z
---

Bridge track, 2026-07-26. Wrote `PAPER/rescuing-fb/specs/10-full-successor-measure-brainstorm.md`
(uncommitted): the plan for closing the 7-gap list from ball-visitation → full successor measure.

Load-bearing content:
- **Keystone**: learned object = ε-mollified measure M̃ = (K_ε*M)(g); its TD fixed point does NOT
  inherit ρ(g) (that defect is inference-side only), and "double counting" = kernel smoothing.
  Density = ratio of two bounded regressions: m̂ = M̃/N̂ with N̂(g)=ρ(B_ε(g)) fit by plain MC
  regression on the hit-matrix column mean (no TD; can be post-hoc on frozen ψ). Same-kernel
  smoothing/boundary effects cancel in the ratio. Corrected moment readout = reweight by 1/N̂.
- **Route change**: replace open-loop moment/ridge z-inference with measure-side integration +
  z-search/continuous GPI: maximize Q̂_r(s,a,z)=Σ_j r_j/N̂(g_j)·F(s,a,z)ᵀB(g_j). One route for all
  reward families.
- **E3 landed, ridge NEGATIVE**: s1 dense moment 0.128 → ridge ≈0.46 (partial only), and ridge
  HURTS s0 (0.896 → 0.47–0.65). Anisotropy diagnosis at best partial ⇒ [[spec08-launched-20260724]]
  E3 read is "misaim elsewhere"; supports z-search route.
- **Code fact**: ZX actor greedy term is the diagonal readout F(s,a,ψ(g))ᵀB(g) ONLY — π_z is
  untrained at random/mixture z (gciql_flow.py actor_loss ~L959). Top suspect for dense split and
  direct-z anomalies. Fixes: FB-style Fᵀz greedy or mixture-matched greedy (recycle _sample_tasks
  arms), one per cell.
- **E5 landed**: strict action-independence of R REFUTED (argmax agreement 0.295/0.299, ICC
  0.24/0.29) → phase-3 interaction-conditioned renewal head R(s_T,g) proposed.
- Practical: quantile-calibrated ε (target positives/row) = fix for E7/E8 conjunction sparsity.
- Ladder L1–L10 in the spec; L1–L4 are eval-only on existing ckpts (zx, visz, FP-NoOrtho).
