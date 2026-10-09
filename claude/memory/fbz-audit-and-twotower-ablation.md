---
name: fbz-audit-and-twotower-ablation
description: Systematic audit of GCIQL-Bil-FBz vs real FB (remaining differences) + the two-tower forward-map ablation (impl + launched)
metadata: 
  node_type: memory
  type: project
  originSessionId: dc3d4721-b93f-4cb9-946c-0c8e8c0ef658
---

2026-06-30. Audited `GCIQL-Bil-FBz-O1000` (the fb_forward_z measure graft, @500k=25.5) against reference FB (`agents/fb/`, `fb_state_plain` ~48) to enumerate remaining differences. See also [[bridge-comparison-both-flow-actors]], [[fb-graft-audit-and-fbz-probe]].

**Audit verdict — what's matched (controls OK):** discount 0.99, target tau 0.005, latent/z_dim 50, num_parallel F-heads 2, flow actor, ortho 1000, pure-measure, EMA targets. **Two user re-checks RESOLVED:** (1) z-distribution — GCIQL value_goals = mix [cur .2/traj .5/random .3] (`gciql.py:296`), so the graft DOES train F on a traj/cur/random spread of goal-z; only sphere-uniform random-z + √d projection missing (folds into #3). NOT a standalone issue. (9) eval protocol — `evals/ogbench.py:175` infers FB z by relabeling vs the SAME task goal (z=Σr·B(s')≈B(goal)); both fed same goals, differ only in actor conditioning (=#2). RESOLVED.

**Remaining REAL differences (ranked):** #2 actor/critic conditioning mismatch — FBz uses `actor_cond='goal'` (actor on raw goal) while critic F uses z=ψ(g); FB conditions both on z. Caveat: current `actor_cond='z'` sources z from the VALUE net (`_value_psi`), untrained under measure_only — fixing #2 needs the actor to source z from the CRITIC ψ. #3 no norm_z/project_z (z,B norms float). #4 forward-map arch (flat concat MLP([s,a,z]) vs FB two-tower MLP[s,a]⊗MLP[s,z]). #5 graft applies 1/√d scaling FB doesn't. #6 ψ ensembled (2 heads) vs single B (leave). #7 B MLP differs (leave). #8 single joint Adam@3e-4 vs separate@1e-4 (leave).

**#4 DONE (TDD green, UNCOMMITTED):** new `GCTwoTowerCritic` + `_TwoTowerPhi` + config `fwd_two_tower` in `tools/wandb_mode_shim/gciql_flow.py`. Two-tower F=trunk(MLP[s,a], MLP[s, sg(ψ(g))]) on the IQL bilinear critic (z=ψ(g) computed internally → drop-in, NO measure loss). Tests `tests/test_gciql_two_tower_critic.py` (2, green) + 15 gciql_flow/zcond regression green. Config `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_twotower.yaml`. **LAUNCHED** 3 seeds @500k: tmux `twotower` (GPUs 0/1/2), run_group `GCIQL-State-Flow-Bilinear-d50-TwoTower`, logs `/dev/shm/twotower_logs/s{0,1,2}.log`, wandb amsks. Compare vs d50 baseline @500k = **90.4** (architecture-only: does FB's two-tower cost the IQL bilinear?). If neutral/+, port to FB.

**ORTHO-ONLY cell DONE+LAUNCHED (TDD green, UNCOMMITTED):** user asked if we'd tried "bilinear GCIQL + orthonormality only, nothing else from FB" — NO, ortho was locked inside measure_loss. Added `ortho_loss()` + config `iql_ortho_coef` (applies FB ||E[ψψᵀ]-I|| on the IQL critic's ψ(value_goals), NO measure bootstrap). Tests `tests/test_gciql_iql_ortho_only.py` (2, green). Config `..._d50_ortho1000.yaml`, coef=1000 (FB cube value). LAUNCHED 3 seeds @500k: tmux `ortho` GPUs 3/4/5, run_group `GCIQL-State-Flow-Bilinear-d50-Ortho1000`, logs `/dev/shm/ortho_logs/`. vs d50 90.4. If coef=1000 tanks on scale, ladder {1,100}.

**PENDING (user directed, not yet done):** #2 (actor_cond='z' sourced from CRITIC ψ for measure_only/fb_forward_z), #3 (optional z/B sphere-norm knob), #5 (knob to drop the measure's 1/√d scaling). Launch path: `.venv/bin/python run_gciql.py --config-name=<cfg> seed=$s ... wandb_mode=online` (torch venv has hydra; spawns jax child). [[always-run-in-tmux]] [[jax-gpu-packing-preallocate]] [[wandb-entity-amsks]]
