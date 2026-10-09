---
name: lrcheck-monitoring-protocol
description: How the user wants the PSM FB-matched lr_phi re-check monitored — always compare NEW vs OLD at the same step for both LRs
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 519ba2a4-1b0e-432b-a53e-bb70d5862f27
---

When monitoring the PSM FB-matched `lr_phi` re-check (tmux `sweep_psm_lrcheck`), the user
wants EVERY check to compare the NEW FB-matched runs against the OLD-HP runs **at the same
step**, for **both** `lr_phi=1e-4` AND `1e-5`, to see how the new basis progresses relative
to the old. One command does it:

`.venv/bin/python scripts/lrcheck_progress.py` (defaults: NEW = newest `logs/grid_*fbmatch_lrcheck*`,
OLD = `logs/grid_20260608_080705`). It prints, per lr_phi, NEW vs OLD vs Δ for eval_success
(latest common eval) + critic metrics (q, sf_M_diag, sf_offdiag, orth_loss_offdiag) at the
NEW runs' current step.

**Why:** the FB-matching (`z_dim 128→50`, `discount 0.98→0.99`, `target_tau 0.01→0.005`,
see [[psm-rldp-sweep-campaign]]) moved the HP basis, so the OLD sweep's `lr_phi=1e-5` pick is
stale; the user wants to watch the new trajectory side-by-side with the old one it's replacing.

**How to apply:** lead each monitoring report with that NEW-vs-OLD-at-same-step table for both
LRs. Eval_success is the headline (meaningful only after the first real eval ~50–100k); the
training-metric deltas track expected HP signatures (disc↑→q↑, z_dim↓→sf_M_diag↓).
