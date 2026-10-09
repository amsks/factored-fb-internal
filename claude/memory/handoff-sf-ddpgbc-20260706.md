---
name: handoff-sf-ddpgbc-20260706
description: "READ FIRST (2026-07-06) — SF-feature-TD re-run with the STOCK DDPGBC actor (no-flow-actor control): does swapping the flow actor for the gaussian DDPG+BC actor lift SF off the ~0.09-0.16 collapse? 4 cells {agnostic,cond}x{identity d28,frozen d50} x3 seeds @500k RUNNING (tmux sfddpgbc). Also: the original 8-cell FLOW-actor campaign is now COMPLETE (all @500k)."
metadata:
  node_type: memory
  type: project
  originSessionId: 27023f28-08b1-455e-b5c4-83a722ff4aba
---

**READ FIRST. 2026-07-06.** Follows [[handoff-sf-featuretd-20260704]] (the flow-actor SF campaign).

## Flow-actor 8-cell campaign is COMPLETE (all @500k, 3 seeds, wandb amsks/factored-fb)
Final means: IdentityB 0.097 / Frozen50 0.092 / IdentityCond 0.120 / **FrozenCond 0.157 (best)**;
all 4 ORTHO1000 cells ≈ 0 (ortho reg on φ collapses the critic; one FrozenCond-Ortho1k seed failed@405k).
Headline holds: NO SF-representation-TD cell escapes ~0.09–0.16 toward `zg_iql` 0.44. Ortho knob is dead.

## NEW campaign: SF with the STOCK DDPGBC actor (no-flow-actor control) — DONE @500k
**RESULT (12/12 finished, 3 seeds): IdentityB 0.027 / Frozen50 0.013 / IdentityCond 0.071 /
FrozenCond 0.127.** vs FLOW-actor (0.097/0.092/0.120/0.157): DDPGBC is ≤ flow in EVERY cell.
**CLEAN NEGATIVE: the flow actor was NOT the bottleneck** — swapping actors doesn't rescue SF, if
anything it's worse; all 4 stay ≪ zg_iql 0.44. Collapse lives in the SF CRITIC/representation, not the
actor. Same sub-trend as flow: task-cond helps (FrozenCond best), base (identity vs frozen) doesn't.
tmux `sfddpgbc` exited (all runs done).

Q was: was the FLOW actor the bottleneck behind the SF collapse? (strong project prior: flow actor is
repeatedly the killer — antmaze, decouple-critic). Swap `actor_kind: ddpgbc` (stock GCIQL gaussian
DDPG+BC, alpha=0.3), SF-feature-TD critic UNCHANGED. Clean 2×2 {agnostic,cond}×{identity d28,frozen d50}.
- Configs: `configs/gciql/cube_single_state_gciql_flow_sf_feature_td{,_frozen50,_identity_cond,_frozen_cond}_ddpgbc.yaml`.
  Groups `GCIQL-State-Flow-SF-FeatureTD-{IdentityB,Frozen50,IdentityCond,FrozenCond}-DDPGBC`. 3 seeds @500k.
- DROPPED the 4 ortho cells (user scope decision): ortho hits the CRITIC not the actor → ddpgbc can't rescue.
- Launch script `/dev/shm/sf_ddpgbc_launch.sh`, logs `/dev/shm/sf_ddpgbc_logs/`. 8 immediate (seeds 0,1 of
  all 4 cells) + 4 seed-2 on a SINGLE free-GPU watcher. NOTE: watcher placed id_s2 on gpu6 during the boot
  transient (co-init race) → gpu6 doubles 2 small state runs (~2.75GB, harmless). 3 seed-2 runs wait for
  the first wave to hit 500k.

## Code change (UNCOMMITTED, on gciql-fb-bridge) — TDD, 11/11 green
`tools/wandb_mode_shim/gciql_flow.py`: the ddpgbc path assumed the flow actor and crashed on the
`fb_forward_z` (task-conditioned) cells. Fixed in TWO places:
1. `actor_loss`: when `actor_kind=='ddpgbc'` AND `fb_forward_z`, run a local DDPG+BC loss that passes the
   z=goal leg to the critic `critic(obs,goals,q_actions,goals)` (stock delegates with only 3 args → crash).
2. `sf_loss`: next action `a'~π(s',g)` — ddpgbc uses `actor(nobs,vgoals).mode()`, not the noise-flow call.
New test `test_sf_ddpgbc_actor_with_conditioned_critic_trains_finite` in `tests/test_gciql_sf_feature_td.py`.

## NEXT
1. Collect 4 DDPGBC cells @500k (`evaluation/overall_success`, mean/3 seeds). If they ALSO sit ~0.09–0.16
   → the collapse is the SF *critic/representation*, not the actor (strengthens thesis). If any jumps
   toward 0.44 → the flow actor WAS masking a usable SF critic (big result — revisit all flow-actor SF).
2. Commit the whole SF increment (task-cond + ortho from 663a3f5 era + this ddpgbc fix). Push blocked on box auth.
3. Fold into PROJECT-MAP §5 / isfb.tex.

Collect: `.venv/bin/python -c "import wandb;api=wandb.Api();[print(g,[round(r.summary.get('evaluation/overall_success',-1),3) for r in api.runs('amsks/factored-fb',{'group':g})]) for g in ['GCIQL-State-Flow-SF-FeatureTD-'+x+'-DDPGBC' for x in ['IdentityB','Frozen50','IdentityCond','FrozenCond']]]"`

[[handoff-sf-featuretd-20260704]] [[jax-venv-location]] [[jax-gpu-packing-preallocate]] [[always-run-in-tmux]] [[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]] [[commit-no-claude-coauthor]]
