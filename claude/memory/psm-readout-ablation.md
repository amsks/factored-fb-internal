---
name: psm-readout-ablation
description: PSM-vs-FB gap ablation launched 2026-06-09 (pessimism=0 + ortho high); why fb_state_plain is the right baseline
metadata: 
  node_type: memory
  type: project
  originSessionId: 73bd5d8a-de5e-4c2d-a10c-a539bd53878c
---

Follow-on to [[psm-rldp-sweep-campaign]]. Tuned PSM plateaus ~0.30 success vs FB ~0.52 on cube-single. Investigating the *remaining* PSM-vs-FB differences.

**Key non-obvious facts (verified from wandb configs 2026-06-09):**
- `fb_state_plain` (0.52) and `fb_state_L256` are actually `agent=fb_flowbc` (noise_conditioned_actor, bc_coeff=3), NOT plain TD3 FB. So the actor stack is ALREADY matched to `psm_state_tuned` (psm_flowbc). The whole gap is in the representation/critic + readout, not the policy head.
- The `cube-single-ortho*` wandb groups are also fb_flowbc (FB cube sweep); fb_flowbc winner = `ortho_coef=1000, lr_b=1e-4`.
- psm.yaml defaults == the tuned config (z_dim=128, discount=0.98, lr_phi=1e-5, ortho_coef=1, target_tau=0.01, actor_pessimism_penalty=0.5). "Our config" = psm.yaml defaults.

**Reframing:** task-distribution probe says PSM φ resolves the interaction var BETTER than FB B (AUC 0.94 vs 0.75) — yet PSM's policy succeeds LESS. So the gap is in turning a good basis into a good policy (readout/Q/actor), not basis quality. Prime suspects: (1) actor_pessimism_penalty=0.5 in PSM vs 0 in FB (with num_parallel=2 the unc term is just |Q1-Q2|, a noisy persistent conservative bias); (2) dual-ψ indirection — actor's Q reads sf_psi·z regressed onto DETACHED φ, so the actor gradient never reshapes φ; (3) FB ortho on B at coef 1000 vs PSM ortho on φ at coef 1 (PSM sweep only tried 0.1–10).

**Launched 2026-06-09 19:05 (tmux `psm_pess0`, `psm_orthohi`), 500k × 3 seeds, via `scripts/run_grid_sweep.sh` (added `GPU_IDS` env knob to target idle GPUs):**
- `psm_state_pess0`: psm_flowbc, actor_pessimism_penalty=0, else defaults. GPUs 2,3,4 (1 seed/GPU = fastest).
- `psm_state_orthohi`: psm_flowbc, ortho_coef ∈ {100,1000}, else defaults. GPUs 5,6,7 (2/GPU).
- Tuned 1M runs (8 seeds) still on GPUs 0,1, untouched.

Selection metric mirrors the sweep: mean eval/success over seeds at ~500k. First eval at 50k. Win = beats tuned 0.30 → promote to 1M×10.

**Result of the 3-seed/500k orthohi sweep (read 2026-06-10):** `ortho_coef=1000` is the best & most stable PSM config seen — final≈peak mean 0.34 (no late decay), beating `ortho_coef=1` (final 0.27) and original tuned ~0.30. `ortho_coef=100` peaks highest (mean 0.41) but DECAYS to 0.19 by 500k. So ortho1000 dominates at 3 seeds → promoted to 1M×10.

**1M×10 ortho1000 DIAGNOSTIC DONE 2026-06-11** (probe suite via `scripts/run_psm_orthohi1000_probe_suite.sh`, outputs `analysis/profiles/psm_o1000_phase/`, `analysis/factorization/psm_o1000/`, `analysis/value/aliasing_psm_o1000/`, cf `cf_value_psm_psm_o1000_ms*`): **confirms readout-not-basis.** ortho1000 phase success 38.4% (vs ortho1 28.6%, +10pp) BUT counterfactual AUC 0.488±0.025 (≈chance, vs ortho1 0.516 — no better) and aliasing 0.61 (identical to ortho1); factorization still B2, R²(B→d)=0.90. Success gain is decoupled from readout: ortho1000 grasps LESS (secured 79.8 vs 87.9) but converts secured→success better (48% vs 33%) = better transport EXECUTION, not a better value ranker. So basis tuning buys success without fixing the bilinear readout's interaction-blindness — same lesson as the killed coverage-balanced reweight ([[fb-coverage-balanced]]).

**1M×10 promotion LAUNCHED 2026-06-10 16:29 (tmux `psm_orthohi_1M`):** both `ortho_coef∈{100,1000}`, lr_phi=1e-5, 10 seeds → 1M, same wandb group `psm_state_orthohi__ortho_coef{100,1000}__lr_phi1e-5` (entity amsks). Two `run_grid_sweep.sh` schedulers (disjoint GPUs, ~650MB/job): **resume** s0–s2 from existing `step_400000.pt` (400k→1M, re-does 100k — `final.pt` is unusable for resume since `_CKPT_RE` only matches `step_<N>.pt`, so resume_from=final.pt yields start_step=0) on GPUs 1,3; **fresh** s3–s9 (0→1M) on GPUs 5,6,7. Logs `logs/grid_20260610_162911/`. Next: summarize at 1M — does ortho1000 close the gap to FB 0.52?
