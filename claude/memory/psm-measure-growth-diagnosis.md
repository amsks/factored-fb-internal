---
name: psm-measure-growth-diagnosis
description: "PSM \"exploding\" q/offdiag is benign unbounded successor-measure growth, not the cause of poor success"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54b70bad-1cae-4220-b346-499c5304e5f4
---

PSM-FlowBC cube-single state campaign (group `psm_flowbc_state`, 10 seeds, 1M) showed q→2200-3500 and sf/psm offdiag→~2000, which looked like "exploding metrics." Diagnosis (2026-06-03, wandb + added norm instrumentation, commit bd90983):

- All seeds **finite** (no NaN). It's slow unbounded growth, not numerical blow-up.
- `phi_norm` and `z_norm` are pinned at √z_dim=11.31 (normalized) — NOT the source.
- `sf_psi_norm` grows unbounded (9.9→353+ by 18k), inflating `sf_M_diag` (3.7→1763+) and `q` (−5→711+). `sf_targetM_diag` tracks `sf_M_diag` every step → self-reinforcing TD bootstrap.
- **Mechanism:** the diagonal measure-loss term `sf_diag = -diagonal(diff).mean()` is LINEAR (rewards a larger self-measure without bound); the only quadratic anchor is on the OFF-diagonal. So the successor measure ψ has no direct magnitude regularizer. Structural to the FB/PSM objective AS TRANSCRIBED from reference (FB shows it too: q~1200) — not a transcription bug.
- **Key:** `corr(q_max, success_final)=+0.36` — the growth does NOT cause the policy failure; if anything mildly helps. The REAL problem is separate: success peaks ~0.34 @600k then DECAYS to ~0.18 @1M (all seeds).

**Why:** "metrics exploding" and "poor/decaying performance" are two different phenomena here; taming the measure growth (sf_psi normalization / weight decay / diagonal anchor / grad clip / lower discount) is a stability/cosmetic fix, NOT expected to fix the success ceiling or late decay.

**How to apply:** Diagnostic norm logging now lives in agents/psm/agent.py + flow_bc/agent.py (phi/psi/M/targetM/z norms; behavior-neutral). When debugging PSM perf, pursue the decay thread (actor over-optimizing an inflating Q, pessimism, BC coeff) separately from the measure-scale thread. Relates to [[crl-actor-loss-critic-gradient]], [[tdmpc2-cube-failure-diagnosis]].
