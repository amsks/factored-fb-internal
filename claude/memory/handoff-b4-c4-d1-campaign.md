---
name: handoff-b4-c4-d1-campaign
description: "FB-structure endpoint campaign (B4/C4/D1) + two new gciql_flow knobs (project_z, actor_z_from_critic); RUNNING + deferred ablation watcher"
metadata: 
  node_type: memory
  type: project
  originSessionId: 93f5c9a5-a5c6-42e4-bb06-d18aac9bae68
---

2026-07-01. From the untried-design-space catalog (`PAPER/fb-gciql-bridge/untried-design-space.md`),
launched the two "payoff" endpoints + D1 in parallel; the leave-one-out ablations are scheduled
behind a watcher. See [[fb-graft-audit-and-fbz-probe]], [[fbz-audit-and-twotower-ablation]],
[[fb-state-run-name-map]] (bridge run→success map).

**Two NEW gciql_flow.py knobs (TDD green, UNCOMMITTED):**
- `project_z` (default False) — hard-projects z and B=psi onto the sqrt(d) sphere (FB's
  ||B||=||z||=sqrt(d) Norm). Module field on GCFBValue + GCTwoTowerCritic; helper `_sphere(x,d)`;
  wired in create() only for those two critics. Tests `tests/test_gciql_project_z.py` (2).
- `actor_z_from_critic` (default False) — sources the flow actor's z from the CRITIC psi
  (=B, measure-trained) not the value net (audit #2 / D1); needs actor_cond='z'. New `_critic_psi`
  + `_actor_z` dispatcher; wired in actor_loss, sample_actions, AND measure_loss's a_next bootstrap
  (that bootstrap passed raw vgoals — broke under actor_cond='z'; now maps via _actor_z). Tests
  `tests/test_gciql_actor_z_from_critic.py` (3). **Gotcha fixed:** `_critic_psi` must key ex_a off
  `goals.shape[:-1]` (eval passes a single UNBATCHED goal) and head-mean psi over axis 0 always.

**Code-grounded corrections to the catalog's B/C decomposition:** (1) `fwd_two_tower` and
`fb_forward_z` are MUTUALLY EXCLUSIVE (two implementations of forward-map z-conditioning), not
stackable — so B is built on two-tower, C on flat fbz. (2) C1 (the 25.5 FBz run) ALREADY had
measure_ortho_coef=1000, so C's ortho rung is a CHANGE 1000→1.0, not an add.

**LAUNCHED 2026-07-01 — 3 run-sets, 9 seeds @500k, tmux `bridgeends`, logs /dev/shm/bridge_ends_logs/:**
- **B4** `GCIQL-State-Flow-Bilinear-d50-B4-FBstruct` = IQL bilinear d50 + fwd_two_tower + project_z
  + iql_ortho_coef=1.0. Q: does ALL FB structure survive on the 90 base? (two-tower alone=86.4.)
- **C4** `-C4-FBrebuild` = measure_only + fb_forward_z + project_z + measure_ortho_coef=1.0. Q:
  rebuild FB from the measure side → reach plain-FB 48? (C1 fbz@ortho1000 = 25.5.)
- **D1** `-D1-ActorZCritic` = FBz-O1000 cell + actor_cond=z + actor_z_from_critic. Single-variable
  vs the 25.5 FBz baseline (only the actor's z-source changes; eval also routes via _actor_z=critic psi).
Configs `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_{B4,C4,D1}.yaml`. All 3 smoke-passed
end-to-end (1200 steps + eval, exit 0). Collect: B4 vs 90.4; C4 vs 48; D1 vs 25.5.
B4 is ~4x slower/step (two-tower+project_z+ortho) → B4 seeds are the long pole.

**GPU packing GOTCHA (2026-07-01):** launched 9 seeds packing 2 onto GPU0 (preallocate=false) —
the two co-initializing jax procs on GPU0 DEADLOCKED at step 0 (0% util) while the 7 one-per-GPU
runs climbed fine. Fix: killed both, relaunched B4_s0 alone on GPU0, deferred D1_s2 via
`/dev/shm/d1s2_watch.sh` (tmux `bridgeends:d1s2wait` — waits for <8 cube procs, then launches D1
seed=2 on the idlest GPU). LESSON: never co-init 2 jax procs on one GPU; stagger or one-per-GPU
(refines [[jax-gpu-packing-preallocate]], which enables packing but not concurrent init). Net:
8 endpoint seeds one-per-GPU + D1_s2 deferred; all 9 will complete.

**DEFERRED ablations (leave-one-out, config-only, 4 sets/12 seeds):** tmux `bridgewatch` runs
`/dev/shm/bridge_ablation_watch.sh` — waits until no `main.py --env_name=cube-single` procs remain,
then launches into tmux `bridgeabl` (logs /dev/shm/bridge_ablation_logs/): `B_ttproj` (two-tower+proj,
no ortho), `B_ttortho` (two-tower+ortho1, no proj), `C_proj` (fbz+proj+ortho1000), `C_ortho1`
(fbz+ortho1, no proj). Brackets each endpoint by dropping one knob.

**NOT auto-scheduled (needs code):** "two-tower under the measure loss" (faithful FB forward for C) —
GCTwoTowerCritic isn't wired into measure_loss; that's the one remaining rung requiring new code.

**OPEN:** collect @500k results (wandb amsks/factored-fb, key `evaluation/overall_success`); commit
the 2 knobs + 7 configs + 2 test files (currently UNCOMMITTED on `gciql-fb-bridge`); fold results into
the ladder tex + catalog. [[always-run-in-tmux]] [[jax-gpu-packing-preallocate]] [[wandb-entity-amsks]]
