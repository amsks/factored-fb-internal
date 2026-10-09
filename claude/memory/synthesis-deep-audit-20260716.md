---
name: synthesis-deep-audit-20260716
description: "2026-07-16 deep audit + synthesis: unified bilinear mechanism (advantage pooling + DR3 stabilizer + extraction match), RG-FB proposal (FB API + scalar grounding, pieces all validated), quick-test queue T1-T9. Full doc: PAPER/synthesis-deep-audit-20260716.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6efa5bd0-7172-48a3-b6c9-f40de80eff30
---

Full synthesis written to **`PAPER/synthesis-deep-audit-20260716.md`** (audit verdicts, mechanism,
9 ranked quick tests, RG-FB spec, cross-OGBench plan, lit index). Produced from code audits of
`gciql_flow.py` + `agents/fb/` and two literature sweeps.

Key claims (see doc for evidence):
- **Mechanism**: bilinear win = cross-goal **advantage pooling** (action signal <1% of Q variance;
  rank-d ψ(g) forces one shared φ basis to accumulate it; MLP fits baseline and its action-ordering
  is rank-collapse noise) + **DR3** explains the ψ(s,g) divergence (shared-s co-adaptation channel;
  ψ(g) closes it) + extraction match (flow=selection reads ordering; DDPG reads the off-manifold
  quasimetric residual — fatal only on narrow action manifolds like cube play data → explains the
  antmaze DDPG+BC flip).
- **Code smells that matter**: BC flow vf is GOAL-AGNOSTIC (gciql_flow.py:1512,632) — prime suspect
  for antmaze flow collapse (both critics ~0.24-0.33); torch FB uses mean-not-min targets (VC-FB's
  diagnosed optimism failure); repo's `onestep` FB zeroed z so Zheng-style one-step FB is UNTESTED.
- **RG-FB proposal**: keep FB API, ground with hindsight scalar expectile TD. All pieces validated:
  bilinear-d50 0.904 + G4 z-cond actor 0.921 + E1a moment inference 0.955 (used the G4 agent,
  verified in scripts/eval_zeroshot_inference.py) + E2 low-λ measure aux free. New work = one
  end-to-end agent + generality probe + cross-domain.
- **Quick tests**: T1 goal-smoothed MLP rerank (pooling test), T2 DR3 statistic on exp5 ckpts,
  T3 off-peak probe on antmaze/scene, T4 goal-conditioned vf on antmaze (highest-leverage training
  run), T5 param controls, T6 zero-shot generality, T7 FB min/conservatism, T8 faithful one-step FB,
  T9 true-advantage rank.
- Key external refs: Zheng/Jayanth/Eysenbach 2602.11399 (FB non-contraction, one-step FB, d≥|S×A|);
  BVN 2204.13695; DR3 2112.04716; VC-FB 2309.15178; BREEZE 2510.15382; ZOL 2602.01962.

Related: [[handoff-bilinear-mechanism-20260716]] [[handoff-oodgrid-generalization-20260713]]
[[zeroshot-gciql-e1a-works]] [[bridge-comparison-both-flow-actors]]
