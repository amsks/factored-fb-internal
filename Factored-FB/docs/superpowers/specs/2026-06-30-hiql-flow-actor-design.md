# HIQL FlowBC actor — design

**Date:** 2026-06-30
**Status:** approved (design), pre-implementation

## Goal

Add a flow-matching BC actor to HIQL, behind a knob that selects which of HIQL's
two actors becomes flow, so we can run the 3 actor ablations (`low`, `high`,
`both`) and compare against stock HIQL (bilinear-d50 ≈ 0.159 @500k cube-single)
and the `gcivl_flow` results. This is the HIQL analog of `gciql_flow` /
`gcivl_flow` (action policy → AWR/Q-guided flow actor), extended to HIQL's
hierarchy.

## Background (HIQL structure, vendored `agents/hiql.py`)

HIQL has two AWR actors over an IVL expectile value + a state-dependent,
length-normalized subgoal representation `goal_rep([s; g])` (dim `rep_dim`):

- **low_actor** `GCActor → action_dim`, conditioned on `obs` + the subgoal rep
  `goal_reps = goal_rep([obs; low_actor_goals])` (passed with `goal_encoded=True`;
  stop-grad unless `low_actor_rep_grad`). AWR weight
  `exp_a = min(exp(low_alpha·adv), 100)`, `adv = V(s',low_g) − V(s,low_g)`.
- **high_actor** `GCActor → rep_dim` (proposes a subgoal rep), conditioned on
  `obs` + raw goal. Target `goal_rep([obs; high_actor_targets])`. AWR weight
  `exp_a = min(exp(high_alpha·adv), 100)`, `adv = V(high_target,high_g) − V(s,high_g)`.
- **eval** (`sample_actions`): high_actor proposes a rep → length-normalize
  (×√rep_dim) → low_actor produces the action conditioned on that rep.

`gcivl_flow.py` is the template: AWR-weighted flow-matching BC with a
goal-conditioned `GCVectorField` velocity field (goal concatenated into the vf's
obs input), `weight = min(exp(alpha·adv),100)`, per-sample-weighted flow-matching
loss, Euler-rollout sampling.

## Design

New agent `tools/wandb_mode_shim/hiql_flow.py` → `HIQLFlowAgent(HIQLAgent)`,
registered `agent_name='hiql_flow'` (via the shim's `sitecustomize`, like the
other `*_flow` agents). Never edits the vendored OGBench tree.

**Config knob:** `flow_levels ∈ {'low','high','both'}`, default `'low'`. Selects
which actor(s) are flow; the other stays stock Gaussian-AWR (inherited from
HIQLAgent). Inherits `value_arch ∈ {mlp,bilinear}` unchanged, so every ablation
can run on either value head.

**Low-level flow actor** (when `flow_levels ∈ {low,both}`): replace `low_actor`
with `low_actor_vf` = `GCVectorField(action_dim)`. Loss = AWR-weighted
flow-matching BC, reusing stock HIQL's `adv`/`exp_a` exactly:
`goal_reps = goal_rep([obs; low_actor_goals])` (stop-grad unless
`low_actor_rep_grad`); flow target `vel = action − x₀`; velocity net conditioned
on `concat(obs, goal_reps)`; loss `mean(exp_a · ‖pred − vel‖²)`. Sample: Euler
rollout → clip[-1,1].

**High-level flow actor** (when `flow_levels ∈ {high,both}`): replace `high_actor`
with `high_actor_vf` = `GCVectorField(rep_dim)`. Loss = AWR-weighted flow-matching
BC toward `target = goal_rep([obs; high_actor_targets])` (`stop_gradient` on the
rep target, matching stock high_actor which does not push gradients into goal_rep
via the target); velocity net conditioned on `concat(obs, goals)`; weight
`exp_a = min(exp(high_alpha·adv),100)`. Sample: Euler rollout over `rep_dim` →
length-normalize (×√rep_dim).

**total_loss:** `value_loss + low_term + high_term`, where each `*_term` is the
flow loss or the inherited stock actor loss per `flow_levels`. value / goal_rep /
target_value losses inherited from HIQLAgent unchanged (including the target_value
soft update in `update`).

**sample_actions:** high stage → subgoal rep (flow rollout + length-normalize, or
stock high_actor sample + normalize); low stage → action (flow rollout, or stock
low_actor sample), conditioned on that rep. Handle batched + unbatched (state) and
the encoder/event-rank logic as in the other `*_flow` agents.

**create():** rebuild HIQL's network dict (goal_rep, value, target_value) and add
`low_actor_vf` and/or `high_actor_vf` (and keep `low_actor`/`high_actor` only when
that level is stock). Example init args: low vf over `concat(obs, ex_rep)` ×
action; high vf over `concat(obs, goal)` × `rep_dim`.

## Testing (TDD, jax-subprocess harness like `test_gciql_flow_injection`)

1. `low_actor_vf` output depends on the conditioning subgoal rep (changing
   `goal_reps` changes the predicted velocity).
2. `high_actor_vf` produces a `rep_dim`-vector; after sample-time normalization
   its norm is √rep_dim.
3. Agent builds + trains a few steps with finite losses for **each** of
   `flow_levels ∈ {low, high, both}` (state, mlp; one bilinear smoke).
4. `sample_actions` returns correct action shape, clipped, for batched and
   unbatched observations.
5. `hiql_flow` registered in `agents.agents`.

## Ablations to run after merge

`flow_levels = low / high / both` @500k × 3 seeds, cube-single state, on the value
head(s) of interest (bilinear-d50 first; mlp optional). Compare to stock HIQL
(0.159 bilinear) and gcivl_flow.

## Non-goals

- No pixel/DrQ path in this pass (state only; encoder path written but untested).
- No change to HIQL's value, goal_rep, or subgoal sampling logic.
- No new value architecture (reuses inherited mlp|bilinear).
