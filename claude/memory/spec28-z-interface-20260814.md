---
name: spec28-z-interface-20260814
description: "2026-08-14 spec 28 WRITTEN from audited external feedback — moment-vs-coefficient / primal-vs-dual gauge mechanism explains spec27's reward→index-at-chance; program = P0 eval-only probes → 2×2 conditioning×readout → repair fork R-A (ridge-dual index + external reward samples) / R-B (reward-embedding actor)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-14T19:36:39.916Z
---

Spec: `PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md`. Not yet implemented.

**Audit verdict on the external feedback (fork-verified against code):** core claim CORRECT and NEW — softmax kills the linear identity `Q=FᵀE[rB]`, so `z_r=E[rB]` is no longer a reward→index inverse; CE pins only `FᵀB` so B's geometry is gauge-arbitrary; `z(g)=√d·norm(B̄(g))` is a PRIMAL vector while the readout `B̃ᵀz/√d` needs the DUAL coefficient `(Σ_B+λI)⁻¹m`. This EXPLAINS [[spec27-q2-z-readout-20260814]]'s assignment-at-chance. Feedback's §7/§1.2 restate our own DENSITY-FB-EXPLAINED §6.5 / spec26 findings.

**Audit corrections:** g-path "exact kernel" is `_delta_phi` — query-set softmax-NORMALIZED (new cells must match); C2 already computes the ridge object but only as a training loss (solo flat .108/ztv .009), and the P0 probe showed B IS ridge-decodable (R² .68–.70 vs .466 RBF) ⇒ geometry exists, interface never uses it; ‖z‖=√d everywhere ⇒ ridge index is readout-only on existing ckpts (conditioning on it = OOD). z25 stage-1 FINAL: C1 .096/.054 (tilt self-deleted), C2 .108/.009, C3 .108/.602, full .108/.588 — success never off BC floor .111.

**Program:** P0 eval-only on existing ckpts (z24 ~/exp_spec24, z25 ~/exp_spec25_stage1, g-anchor d18_cs_dens_s0): P0a action-ranking Spearman_a(q_g,q_z)+spans+Δ_actor+actor z-sensitivity (ztv never measured the ACTOR); P0b point/moment/ridge reward reconstruction (gate G28-1); P0c e2b rerun with ridge index (coordinate-percentile metric, NOT success — dead-actor trap ×3, bit-exact control gate mandatory). → P1 2×2 (cells ii z+kernel, iii g+Bᵀz; ~30–50 lines each; random-z rows mask q via existing `_density_z_masked` pattern; skip if P0a says q_g span≈0 under z). → P2 fork: R-A dual-corrected index ẑ=√d·norm((Σ+λI)⁻¹m̂_r) + actor integrates ACTUAL reward samples (general interface preserved — binding user req); R-B FRE-style reward-embedding actor with behaviorally-grounded task prior.

**Why:** the .736→.017 collapse changed conditioning AND readout at once; never attributable to z-conditioning until cell ii runs.
**How to apply:** implement P0 first (hours, no training); NO ortho loss (convicted toxin [[firstpassage-measure-works-20260722]]), no tilt sweeps, never quote policy-success from dead-actor pipelines.
