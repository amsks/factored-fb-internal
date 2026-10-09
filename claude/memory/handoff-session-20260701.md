---
name: handoff-session-20260701
description: "READ FIRST — 2026-07-01 session handover: 3 chained campaigns running/queued, uncommitted code, and the Q-max-flow HIQL agent to build next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 93f5c9a5-a5c6-42e4-bb06-d18aac9bae68
---

**READ FIRST. 2026-07-01 session end.** Three campaigns are chained on the GPU box; a batch of
code is TDD-green but UNCOMMITTED; and one new agent is spec'd to build next. Companions:
[[handoff-b4-c4-d1-campaign]], [[hiql-subgoalsteps-bug-and-comparison]], [[fb-graft-audit-and-fbz-probe]].

## 1. Live GPU pipeline (auto-chains; check `tmux ls`)
1. **tmux `bridgeends`** — 8 endpoint seeds one-per-GPU: B4/C4/D1 ×3 (minus D1_s2). At handover:
   B4 ~90–130k (SLOW, long pole), C4/D1 ~350–365k /500k. `bridgeends:d1s2wait` launches D1_s2 on the
   first freed GPU (deferred to dodge co-init deadlock — see [[handoff-b4-c4-d1-campaign]]).
2. **tmux `bridgewatch`** → launches 4 leave-one-out ablations (`B_ttproj,B_ttortho,C_proj,C_ortho1`,
   12 seeds) into tmux `bridgeabl` once endpoints drain.
3. **tmux `hiqlcmpwatch`** → launches the 4 HIQL comparison cfgs (12 seeds, correct HPs) into tmux
   `hiqlcmp` once the ablations drain (gate = ablation "launched" marker + zero `main.py --env_name=`).
Collect (wandb amsks/factored-fb, key `evaluation/overall_success`): B4 vs 90.4 · C4 vs 48 · D1 vs
25.5 · ablations bracket each endpoint · HIQL cube-flow MLP-vs-bilinear@sg10 + antmaze MLP-vs-bilinear.

## 2. UNCOMMITTED (branch `gciql-fb-bridge`) — commit next session
- `tools/wandb_mode_shim/gciql_flow.py` (M): 2 new knobs `project_z` (sqrt(d) sphere-Norm on z,B)
  + `actor_z_from_critic` (D1) + measure_loss actor-bootstrap fix + `_critic_psi` unbatched fix.
- Tests (all green): `tests/test_gciql_{project_z,actor_z_from_critic,two_tower_critic,iql_ortho_only}.py`.
- 11 configs `configs/gciql/*.yaml` (B4/C4/D1, twotower/ortho1000, 4 ablations, 2 cube-hiql, 2 antmaze-hiql).
- `PAPER/fb-gciql-bridge/{fb-gciql-bridge.tex (M: 2 ladder rows + bilinear-V fix),
  untried-design-space.md, morph-table-500k.md}` — PAPER is gitignored (don't commit).
- Also still-unpushed prior commits 2812892 (fbz) + e389c03 (hiql-flow); origin at 3833331 (push blocked on box auth).

## 3. NEXT TASK — build the Q-max-flow HIQL agent ("hierarchical GCIQL low level")
The audit ([[hiql-subgoalsteps-bug-and-comparison]]) showed HIQL's flow port is GCIVL-style
AWR-weighted flow-BC (reweights BC, bounded by data), NOT GCIQL's Q-max flow (FQL, extracts an
improved policy). The winning GCIQL ingredient was never applied to HIQL. Build it TDD-first:

**New agent `tools/wandb_mode_shim/hiql_qmax_flow.py`** (subclass HIQLAgent):
- KEEP unchanged: HIQL IVL value V(s,g) + `goal_rep` subgoal rep (for high-level adv + subgoal space);
  the HIGH actor (AWR or flow subgoal proposer — do NOT Q-max it, it outputs reps); value_arch=**mlp**
  (audit: bilinear hurts the hierarchical rep).
- ADD a LOW-level GCIQL: critic `Q_low(s,a,w)` + value `V_low(s,w)` where `w = goal_rep([s; g_low])`
  (g_low = HIQL's low_actor_goal, a near-future state). Train by IQL expectile on the low-level
  goal-reaching reward toward g_low (gc_negative: 0 if reached else -1). This is literally GCIQL with
  goal := the subgoal rep. Reuse gciql_flow's value_loss/critic_loss.
- LOW actor = FQL flow: `-Q_low.mean()/|Q_low| + bc_coeff*distill` (reuse gciql_flow.actor_loss),
  conditioned on (s, w). Replaces HIQL's AWR-weighted-flow low actor.
- sample_actions: high proposes subgoal rep (as hiql_flow) → low FQL flow-actor produces the action.
- **THE RISK to nail first (TDD):** the low-level reward/termination definition — reaching g_low
  within subgoal_steps — must match how HGCDataset builds `low_actor_goals`. Get this one test right
  before anything else.
- TDD tests: low critic/value train finite; low FQL actor trains finite (Q rises, not critic-inflating —
  see [[crl-actor-loss-critic-gradient]]); full update finite; sample_actions shapes (batched + single).
- Then config (cube flow, sg=10; optionally antmaze sg=25), smoke, slot into a watcher after `hiqlcmp`.
Caveat: HIQL's ceiling on cube is bounded by the high-level's subgoal quality; Q-max lifts the LOW
level (subgoal reaching = what GCIQL does at 90), so expect improvement but maybe not to 90.

## 4. Watch-outs
- NEVER co-init 2 jax procs on one GPU (deadlocks at step 0); one-per-GPU or the deferred/throttled
  watcher pattern. [[jax-gpu-packing-preallocate]] [[always-run-in-tmux]] [[wandb-entity-amsks]]
- No LaTeX toolchain on the box — rebuild the bridge PDF off-box.
