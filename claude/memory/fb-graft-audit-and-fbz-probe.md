---
name: fb-graft-audit-and-fbz-probe
description: Audit of the FB-measure graft on bilinear GCIQL (root cause = forward map not z-conditioned) + the fb_forward_z fix and running 3-seed probe
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c50f5c8-10e2-42be-b25e-9a708249ec30
---

2026-06-30. Audited the FB successor-measure graft on bilinear GCIQL (`tools/wandb_mode_shim/gciql_flow.py::measure_loss`) after `GCIQL-Bil-MeasureOnly-O1000` stalled at ~8% (seeds 0.072/0.056/0.136). The new `measure_ortho_coef` term (FB orthonormality on psi) is mathematically faithful to FB's `ortho_loss` (`agents/fb/agent.py:711-720`) — NO bug — but it patches the wrong leg.

**Root cause (structural, not a bug):** FB is `M(s,a,z)=F(s,a,**z**)·B(s')` — a z-INDEXED family. GCIQL's `GCBilinearValue` critic is `phi(s,a)·psi(g)` with NO task/z on the forward leg, so the grafted measure is the measure of a single goal-randomized policy (bootstrap `a_next~actor(nobs,random vgoal)` is incoherent across columns). Secondary: `B:=psi` is never hard-normalized — FB pins ‖B‖=‖z‖=√d via a `Norm` layer + uses ortho_coef=**1.0** (the "FB uses 1000" comment is wrong; 1000 only in one coverage-balanced spec). The graft has neither Norm nor the right coef.

**Fix implemented (TDD, committed? NO — UNCOMMITTED):** new `config.fb_forward_z` (default False) + `GCFBValue(GCBilinearValue)` in gciql_flow.py: forward map `phi([s,a,z])` with `z=stop_grad(mean_heads(psi(z_goals)))`, so `M[i,j]=F(s_i,a_i,z_i)·B(s'_j)` is a genuine z-family. Wired into create() (asserts measure_only), measure_loss critic-leg (z_goals=value_goals), and actor Q-readout (z_goals=goals). Tests: `tests/test_gciql_fb_forward_z.py` (2 new, green) + existing gciql_flow/zcond suites green.

**Probe DONE (2026-06-30):** 3 seeds finished @500k, run_group `GCIQL-Bil-FBz-O1000` (wandb amsks). Result `evaluation/overall_success` = **0.255** (0.208/0.256/0.300) vs **0.088** z-blind baseline → **~2.9x**. Confirms the forward-map z-conditioning is the dominant structural gap. Config = exact 8% baseline + `fb_forward_z=True` (single-variable A/B): cube-single-play, bilinear d50, measure_only, measure_coef=1, measure_legs=critic, measure_ortho_coef=1000, bc_coeff=3. NOTE: both FB and GCIQL here use the FLOW actor — see [[bridge-comparison-both-flow-actors]].

**COMMITTED 2026-06-30:** commit `2812892` on `gciql-fb-bridge` (code + `tests/test_gciql_fb_forward_z.py`, 2 tests green). NOT YET PUSHED — origin push blocked on auth (no gh/ssh/token/credential-helper on box; VSCode cred socket expired). origin still at `3833331`; 2 commits ahead unpushed (2812892 fbz + e389c03 hiql-flow). **OPEN:** push when auth available; fold 0.255 into PAPER/; optional secondary fix (hard √d Norm on psi + ortho_coef=1.0). Relates to [[zg-identity-fb-experiment]] (learned-B is FB's bottleneck), [[psm-measure-growth-diagnosis]], [[gciql-flow-agent]].
