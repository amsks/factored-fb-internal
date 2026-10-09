---
name: handoff-identity-psi-20260704
description: "READ FIRST (2026-07-04) — ISFB torch campaign FINISHED = NULL (coord 0.33/awr 0.27/hard 0.08, all < plain FB 48). Next experiment = successor-FEATURE vector TD on φ with identity base b (φ-only); a first attempt mis-built it as the successor-MEASURE loss (identity_psi) and was REVERTED. new PAPER/PROJECT-MAP.md consolidates everything"
metadata:
  node_type: memory
  type: project
  originSessionId: 2ad16711-4f75-4873-a0f0-27a559213354
---

**READ FIRST. 2026-07-04.** Supersedes [[handoff-torch-isfb-20260702]] for the live thread.

## 0. CORRECTION — the SF experiment is FEATURE TD, not MEASURE (do not repeat my error)
User's plan (2026-07-03 transcript, session c1a38cc3): *"train the phi embedding with TD
constraint on the representation (SF style), keep psi as identity, not train the goal embedding."*
= train **only φ(s,a)** with a **successor-FEATURE vector TD**:
`φ(s,a) ← b(s) + γ·φ_target(s',a')`, `L_SF = mean_i ‖φ(s_i,a_i) − sg[b(s_i)+γ m_i φ_target(s'_i,a'_i)]‖²`,
readout `Q=⟨φ(s,a),g⟩`, base `b(s)=s` (identity), d=obs dim 28. **Per-sample vector regression.**
This is NOT the successor-MEASURE loss (`measure_loss`, the B×B `M[i,j]=⟨φ_i,ψ_j⟩` density matrix,
diag+offdiag contrastive TD). I first mis-built it AS the measure loss under a knob `identity_psi`
(GCIdentityPsiValue), TDD-green + launched, then **REVERTED ALL OF IT**. **BUILT CORRECTLY + RUNNING
(2026-07-04):** knob **`sf_feature_td`** in `gciql_flow.py` — critic module `GCIdentityBValue`
(φ MLP→d, base leg `_psi(g)=g` identity, param-free) + method **`sf_loss`** (the per-sample vector
TD above; b(s)=obs current-state, a'~flow-actor(s',g), target_critic, mean L2) + `total_loss` branch
(SF-only, drops IQL) + create() asserts (bilinear, value_latent_dim==goal_dim=28, forces
actor_cond='goal') + `sf_coef=1.0`. Tests `tests/test_gciql_sf_feature_td.py` (4 green incl. the
**γ=0 proof `L_SF==mean((φ−obs)²)`** = per-sample regression, not a measure matrix); no sibling
regressions; real-env smoke exit 0. **LAUNCHED:** tmux `sffeaturetd`, script
`/dev/shm/sf_feature_td_launch.sh`, logs `/dev/shm/sf_feature_td_logs/`, 3 seeds @500k GPUs 0–2,
group **`GCIQL-State-Flow-SF-FeatureTD-IdentityB`**, config
`configs/gciql/cube_single_state_gciql_flow_sf_feature_td.yaml`. Identity b only (frozen-random
base deferred). Anchors: FB `zg_iql` 0.44 / `zg_her` 0.19 (confounded — identity b ⇒ linear reward
g·s, q blew up ~1460). Ref repo: raymondchua/simple_successor_features ("Learning SF the Simple
Way", Chua NeurIPS24). Locked choice: b(s)=current state s_i (standard SF Bellman); next-state is the alt.

**2 RUNS IN PARALLEL (user, 2026-07-04):** T1 identity (b(s)=s, d=28, tmux `sffeaturetd`, group
`GCIQL-State-Flow-SF-FeatureTD-IdentityB`, GPUs 0-2) + T2 FROZEN random base (b(s)=ψ(s), ψ=frozen
random MLP via `GCFrozenBValue`, `sf_base='frozen'`, d=**50** to match FB/bilinear-GCIQL rank, tmux
`sffrozen50`, group `GCIQL-State-Flow-SF-FeatureTD-Frozen50`, GPUs 3-5, config `..._frozen50.yaml`).
Frozen = the SF anti-collapse fix (learnable ψ under SF-loss-only → φ=ψ=0 collapse; not run unless we
want to *show* the collapse = optional T3). `sf_base` knob branches sf_loss's base b and the critic
class (GCIdentityBValue vs GCFrozenBValue); frozen skips the d==goal_dim assert. Tests now 6 green
(added ψ-frozen-after-training + γ=0 frozen `L_SF==mean((φ−ψ(s))²)`). Both smokes exit 0. Collect
both @500k vs zg_iql 0.44 / zg_her 0.19.

**COMMITTED 2026-07-04 as `663a3f5`** on gciql-fb-bridge (NOT pushed — box auth blocked). This
commit also checkpointed the prior uncommitted GCIQL-track bridge work sharing gciql_flow.py
(untried B1/C2/exp4, isfb-jax, project_z, iql-ortho, two-tower) + their configs/tests, so the tree
is self-consistent. PAPER/ (incl. PROJECT-MAP.md) intentionally left untracked. Torch ISFB
(agents/fb/isfb/) was already committed earlier (b58e317). Only PAPER artifacts remain uncommitted.

## 1. ISFB torch campaign FINISHED — result = NULL
`isfbtorch2` (group `ISFB-Torch-Exp1-fixed`, 9 runs @500k cube) completed, GPUs idle.
`success/overall` @500k means: **coord 0.33** (s0/1/2=.30/.34/.36), **awr 0.27** (.26/.20/.34),
**hard 0.08** (.10/.06/.08, collapsed orth~1000). ALL below plain torch FB (~48) and the
reward-TD ceiling (~0.65). → the in-sample measure-improvement operator does NOT rescue FB on
the faithful substrate; **the null holds**. Not yet written into isfb.tex. Caveat: unbounded
⟨F,z⟩/M-scale inflation (ortho coef 1 may be too weak vs stock-FB 1000); hard variant collapsed.

## 2. NEW consolidated map — `PAPER/PROJECT-MAP.md`
Single living orientation page (no such doc existed before; user asked to "refresh the working
project map"). Consolidates the two-layer finding, the bridge grid, the ISFB null, the current
identity_psi run, and the remaining queue (D1/B1→B4/C2→C4/E1-E2). Update it when a campaign turns.

## 3. [REVERTED — see §0] the mis-built `identity_psi` (successor-MEASURE, WRONG objective)
Freeze the GCIQL bilinear goal leg to **identity** (ψ(g)=g, z=g, param-free) while keeping the
**vector SF-style measure TD on φ**, on the jax bilinear GCIQL-flow substrate. Tests the
co-adaptation hypothesis: learned-ψ measure collapses (MeasureOnly 9.9, C4 16.5) maybe because
φ,ψ co-adapt; freezing ψ removes that DoF. Spec `docs/superpowers/specs/2026-07-03-identity-psi-sf-measure-td-design.md`.
- **Code (UNCOMMITTED, stacked on the uncommitted untried B1/C2/exp4 gciql_flow.py changes):**
  `GCIdentityPsiValue(GCBilinearValue)` in `tools/wandb_mode_shim/gciql_flow.py` — φ=MLP output
  dim=latent_dim, `_psi(g)=g` (identity, no goal_encoder), ψ broadcast to φ's ensemble shape
  `[2,B,d]`; `fb_forward_z` attr concatenates z=sg(g) into φ. create() routing: `identity_psi`
  takes precedence over fb_forward_z/two_tower; asserts value_arch=bilinear, **value_latent_dim==
  goal_dim** (cube state obs_dim=**28**, act_dim=5), measure_ortho_coef==0; forces actor_cond='goal'.
  get_config default `identity_psi=False`.
- **Tests (5 green, `tests/test_gciql_identity_psi.py`, jax venv):** ψ=identity(g); φ dim==obs dim
  & Q finite; no ψ params; measure trains finite (PRI); identity_psi+fb_forward_z trains finite (FBZ).
  No regression in test_gciql_untried_bridge/two_tower. Real-env smoke (30 steps + eval) exit 0.
- **RUNNING:** tmux `identitypsi`, script `/dev/shm/identitypsi_launch.sh`, logs
  `/dev/shm/identitypsi_logs/`. 6 runs @500k = {PRI task-agnostic φ, FBZ z-cond φ(s,a,z=g)}×seed{0,1,2},
  GPUs 0–5, wandb amsks groups **`GCIQL-State-Flow-IdentityPsi-Measure`** (PRI) /
  **`...-Measure-FBz`** (FBZ). Configs `configs/gciql/cube_single_state_gciql_flow_identity_psi_measure{,_fbz}.yaml`.

## 4. NEXT
1. Collect identity_psi @500k: `evaluation/overall_success` vs FB `zg_her` 0.19 / MeasureOnly 9.9
   / C4 16.5. Diagnostic (more important than raw success): did freezing ψ stop the q≈1460 blowup
   (check q/M scale) and move φ off the 9–25 band? Expected regime is linear-reward (accepted).
2. Write the ISFB null into isfb.tex; fold identity_psi result into PROJECT-MAP + the bridge note.
3. Then remaining queue (D1, B1→B4, C2→C4). Commit the stacked gciql_flow.py + isfb code when ready
   (push blocked on box auth per prior handoffs).
[[handoff-torch-isfb-20260702]] [[handoff-isfb-and-untried-20260701b]] [[jax-venv-location]]
[[jax-gpu-packing-preallocate]] [[always-run-in-tmux]] [[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]] [[commit-no-claude-coauthor]]
