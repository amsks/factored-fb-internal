---
name: partb-campaign-handoff-20260906
description: "2026-09-06 Part B (td_jepa reward-inference route) campaign PREPARED: launcher scripts/launch_partb_tier1.sh (9 runs: density_fb_z cs/cd x3, fb_tdj cd x3, 1M, relabel eval), slurm array scripts/partb_tier1.slurm, HF consolidation scripts/hf_upload_checkpoints.py -> amsks/factored-fb-checkpoints (private, 120 runs, 11.3 GB, upload started), handoff docs/specs/2026-09-06-partb-reward-inference-campaign.md. Commit ef35c94 on density-fb (68 files incl. other track's relabel WIP, tests 20/20) — PUSH FAILED: no git credentials on box (no gh, no ssh key, VS Code cred socket dead); user must push."
metadata:
  type: project
---

Two eval routes, never mixed in one column: Route A goal-conditioned (OGBench standard,
final-step success) and Route B td_jepa reward inference (10k relabelled samples, shift 1.0,
any-step, 10 eps/task). See [[hilp-tdjc-cube-not-inert-20260905]] for why Route B matters.

Tier 1 = 9 runs (~10 GPU-h each). Tier 1b = eval-only `scripts/relabel_rescore.py` on 500k
cube ckpts (density_fb_z, rldp, psm, fb_ref) — a first pass was launched on GPU 0 in tmux
`fable:rescore` writing analysis/relabel_rescore_cs500k.jsonl; verify it completed. Tier 2 =
scene (needs a scene relabel reward first; current one is cube-only). Tier 3 = RLDP/PSM at 1M
(24 runs, optional).

Gotchas: cube_double needs `measure_hit_spec=19:22:0.4,28:31:0.4` override for density_fb_z;
`density_fb_z` eval.csv is the OWN-POOL reward-sample route, not goal-conditioned; expect a
possible drop vs .82 from pool-composition sensitivity (report, don't tune). td_jepa's published
cube numbers not on this box — copy from their paper; they are best-of-2x2 over 10 seeds.

**2026-09-06 (late) ABLATION MATRIX prepared** (`docs/specs/2026-09-06-goal-vs-z-ablation-matrix.md`,
`scripts/launch_ablation_matrix.sh`, `scripts/ablation_matrix.slurm`): HILP/RLDP x {vanilla,density}
x {z-cond, goal-cond} x {obj, state} + Density-FB 2x2, both cube envs, 1M, 3 seeds = 114 pending
(tier1 Density-FB 24, tier2 retrofits 45, tier3 vanilla 45). NEW code: `goal_kernel` ported into
hilp_density.py (whitened goal index + kernel reward), configs hilp_density_gk(_oc),
rldp_density_gk_oc, hilp_gc(_oc) (hilp_mix_ratio 1.0), rldp_gc(_oc) (train_goal_ratio 1.0),
density_fb_z_state; aliases in impls/agents; tests/impls/test_goal_conditioned_variants.py (12 green);
real-data train smokes OK. Actor-drift audit: HILP/RLDP vanilla actors use Q=ψᵀz / Fᵀz (no reward
basis → not affected); density retrofits use a frozen φ/B pool (anchored by Hilbert/SP+ortho
losses; constant shifts cancel in the softmax) → not the Density-FB bug. Commits UNPUSHED (no creds).
