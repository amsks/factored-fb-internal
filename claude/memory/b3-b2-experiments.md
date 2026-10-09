---
name: b3-b2-experiments
description: "Error-aware z-inference ladder (B3) + MLP matched-readout control (B2) — design, where they run, early signals"
metadata: 
  node_type: memory
  type: project
  originSessionId: b798f8b4-b3ec-4145-b887-98d31d022aab
---

Part of the gap-analysis program `PAPER/specs/2026-06-11-fb-psm-crl-gap-analysis-and-program.md` ([[psm-readout-ablation]], [[psm-rldp-probe-suite]]). Both launched 2026-06-11, CPU-only on 1M checkpoints (`step_1000000.pt` explicitly — NB `final.pt` is now the 1.5M extension!).

**B3 — z-inference ladder** (tmux `zladder`, `scripts/z_inference_ladder.py`, pure rules in `evals/z_inference.py`, tests `tests/test_z_inference.py` 5-green): rules L0=E[rB] (current), L1=GLS (E[BBᵀ])⁻¹E[rB], L2=Bellman-consistent z (Adam on the single z, sphere-projected, semi-gradient actions), L2b=L2 with reward-balanced residual (plain L2 drifts to a "null task" — 98% zero rewards dominate). 3 methods (fb_plain, psm_o1000, psm_tuned=ortho1) × 10 seeds × 5 tasks × 4 rules × 20 episodes. Out: `analysis/z_inference/<method>/sN.json`. Logs /tmp/zladder_logs/. Watcher bvqj37qtm.
Predictions: L1>L0 only at ortho1 (non-orthonormal B); L2b>L0 ⇒ inference is a recoverable gap.

**B2 — MLP matched-readout control** (tmux `readout_control`, `scripts/mlp_readout_control.py`): frozen PSM-o1000 basis/normalizer/actor; train MLP Q(norm_obs(s),a,z) + V (IQL expectile 0.9) with z=project(φ(g)) for goals sampled FB-style (batch-perm uniform + 0.2 diag g=s'); reward = cube-dist<0.04, absorb at goal. Eval = cf-AUC on the STORED counterfactual grasps (`analysis/value/repsep/cf_records_psm_psm_o1000_ms*.parquet` — has obs+success labels per grasp, NO rollouts needed; bilinear baseline on same grasps ~0.478). Seeds 0-2 × 150k steps. Out: `analysis/readout_control/sN.json`. Watcher bzkaqtfhz.
**Early signal (3k-step smoke): MLP already 0.574 vs bilinear 0.478** on identical grasps; tasks 4/5 at 0.80/0.68. If it converges ≥0.8 ⇒ bilinear readout form is causally B2.

**VERDICTS (2026-06-11/12):** B3 ladder — inference NOT recoverable (L1≈0 at o1000, catastrophic at o1; L2/L2b −13..−34 everywhere; success monotone in cos-to-L0 ⇒ value/policy z-semantics DECOUPLED). B2 readout — MLP on frozen basis + FB-style uniform goals converges to CHANCE (0.48-0.52 ≈ bilinear) ⇒ **readout FORM not the constraint; the GOAL DISTRIBUTION carries the controllability signal** (GCIQL = MLP + 62.5% trajectory-future goals → 0.89). Gradient audit: FB measure term ACTIVE (ratio~1.4, not suppressed) yet B lift-blind ⇒ persistence-ordering is the objective's active preference; PSM o1000 ratio 26→4.5 (whitening locks basis early), o1 ratio 0.01 (measure in control → worse basis) = causal account of ortho1000 paradox.

**Trajectory-goal builds (2026-06-11 night):** `buffers/transition.py` episode_last_index/sample_future_idx + `DictBuffer.sample(future_gamma=)` → aligned `batch["future"]`; `evals/traj_goals.py` re-exports; MLP+traj cell via `mlp_readout_control.py --goal-mode traj` (p_traj=0.625, RUNNING 3 seeds CPU, out `analysis/readout_control/sN_traj.json`); **FB-zHER** `train_goal_mode=traj` (z=B(own discounted future), aligned/no-perm; agent+train.py+fb.yaml, default perm=byte-identical). 23 tests green (`test_traj_goals.py`, `test_iwr_weights.py`, `test_fb_reweight.py`). IWR-FB runs (fb_state_iwr, fb_state_iwr_wz; 3 seeds @500k, tmux `fb_iwr`, GPUs 4-7) + zHER queued behind psm_z50 factorial (tmux `fb_zher_queue`, group `fb_state_zher`). ortho100 still PAUSED.
