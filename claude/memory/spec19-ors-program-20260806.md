---
name: spec19-ors-program-20260806
description: "2026-08-06 spec19/19a/19b/20 WRITTEN+COMMITTED (a8f1df8) — ORS program (occupancy reward shaping from the behavior-mode density measure) + dwell-balanced FB, with priority order P0–P4; user will hand to an implementing agent"
metadata: 
  node_type: memory
  type: project
  originSessionId: 73a4df12-2b06-49d2-98a3-c282720c698b
  modified: 2026-08-06T21:48:46.897Z
---

**Specs 19/19a/19b/20 (commit a8f1df8, PAPER/Multistep-FB/specs/) — user-approved program, NOT implemented; user plans to hand implementation to another agent.** ORS (ICLR 2026, arXiv 2604.20627 — verified real: W₂ shaping from learned occupancy, optimal-policy-preserving, OGBench cd 45/triple 37/scene 80 vs GCIQL 40/7/51) sourced from [[spec18-density-measure-20260806]]'s density head, NOT classical FᵀB (torch-FB deconfound: linear diag diverges on wide-support rewards; −d²(x,g) is maximally wide).

**Architecture:** 19a = `measure_density_behavior` knob (F input [s,a] no goal, SARSA bootstrap with `batch['next_actions']` — needs a small default-off GCDataset extension at third_party/ogbench/impls/utils/datasets.py; actor becomes pure BC flow, actor_q_center refused as inert). 19b = transcription duty FIRST (paper's reward composition/normalization conventions, PSM precedent), then `ors_reward_ckpt` extractor knobs on gciql_flow (frozen ors_f/ors_b modules via freeze_critic_ckpt-style machinery, pool 4096 static nonpytree, r_W replaces batch['rewards'] in value/critic losses, 2-head convention = MEAN of softmaxes not min), plus `ors_calibration_probe.py` — HARD GATE Spearman ≥.65 vs dataset empirical discounted future distance BEFORE any extractor launch. 20 = dwell detector (`make_dwell_weights.py`, global-stationary + post-motion rule protects pre-grasp and cube1-rest-while-cube2-moves rows; α=L^−p, p=0.5, report gate) + torch FB `reweight_mode="dwell"` ROW weights (existing framework weights COLUMNS — new orthogonal path; ortho term stays unweighted).

**Priority order (the user asked for this explicitly):** P0 = G4 Spearman (density + part K256, STILL unrun) + BC-only control + d17/d18 CSV refresh — gates all interpretation. P1 = 19a runs (cs/cd/scene seed 0) → 19b probe gate → extractor wave (cs tie-gate ±.05 vs density; cd >.549 = headline; scene vs hindsight .927 = honest bar). P2 = dwell weights + report (parallel-safe with P1). P3 = dwell-FB wave (pre-registered: null strengthens spec18 estimator attribution). P4 deferred/gated: classical-FB ORS comparison, cube-triple, dwell-density.

**Why:** dense grounding at every (s,a,g) is the constructive answer to §9.16 (hindsight-only, 0.046% conjunctive hits); g-conditioned extraction respects the stack-divergence z-conditioning verdict; dwell stays IN the shaping occupancy (occupancy=arrival×dwell — dwell is signal for ORS) and is only reweighted in the P3 representation arm.

**How to apply:** hand 19a → 19b → 20 to implementing agents in P0→P3 order; each contract is self-contained TDD with lattices and numpy-reference tests. If continuing this track, read spec 19 §3 pre-registered interpretations before quoting any result.
