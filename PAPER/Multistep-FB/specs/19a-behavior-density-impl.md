# Spec 19a — `measure_density_behavior` implementation contract

Audience: the implementing agent. Self-contained; science context in
`specs/19-ors-program.md`. Parent knob: `measure_density` (spec 18,
IMPLEMENTED — read its shipped code in `tools/wandb_mode_shim/gciql_flow.py`
before starting; this contract is a small delta on it). TDD; default-off
must be BIT-identical.

## 0. What is being built

A BEHAVIOR-POLICY variant of the spec-18 continuous density head: the
occupancy model ORS needs is d^{π_D}(·|s,a) — the dataset behavior
policy's discounted future object-configuration distribution. Two changes
relative to `measure_density`:

1. **No goal conditioning.** F tower input is `[s, a]` (the behavior
   policy ignores goals; conditioning on g would only add estimation
   burden). The density loss does not consume `value_goals`.
2. **SARSA bootstrap.** The bootstrap action is the DATASET's next action
   `a′_D = batch['next_actions']`, not a sample from the learned actor.
   (The learned flow actor is goal-conditioned BC — bootstrapping with it
   would estimate a different policy's occupancy.)

Everything else is spec-18 verbatim: queries = the batch's own next-state
object coords, `k = _delta_phi(nobs, queries)`, CE to
`sg[(1−γ)k + γw̄⁺]`, 2-headed F / single B, target towers, τ, ε-scaled
B input. The actor in this mode is PURE BC flow (its q-term is disabled —
there is no goal-conditioned measure to read); the run's product is the
frozen (F_D, B_D) towers, consumed by spec 19b. The actor checkpoint is a
byproduct and is not evaluated.

## 1. GCDataset: `next_actions` (in `third_party/ogbench/impls/utils/datasets.py`)

Add a config flag (mirror how existing optional keys are gated; default
OFF must leave every produced batch dict byte-identical):

- When on, `batch['next_actions']` = the action logged at the
  `next_observations` index, using EXACTLY the same trajectory-boundary
  index arithmetic the class already uses to build `next_observations`
  (read that code first; do not invent new indexing).
- Where that index has no valid same-trajectory action (final state of a
  trajectory), substitute the CURRENT action `batch['actions']` for that
  row and count it: expose the substitution fraction so the agent can log
  it (info key `density_sarsa_subst`; on OGBench play this should be
  ≈ 1/traj_len — assert < 5% in a test on the real cube-single npz if
  cheap, else on a synthetic layout that mirrors ogbench's).
- The flag is plumbed from the agent config in `create()`-adjacent
  wiring the same way existing dataset-facing knobs are (grep how
  `gc_negative`/frame-stack knobs travel; follow that path).

## 2. Changes in `tools/wandb_mode_shim/gciql_flow.py`

### 2.1 `get_config()` (next to the spec-18 knobs)

```python
config.measure_density_behavior = False  # spec 19a: behavior-policy variant of
  # measure_density -- F tower input [s,a] (no goal), bootstrap action = the dataset's
  # next action (SARSA). Product = frozen (F_D, B_D) for the spec-19b ORS reward.
  # Requires measure_density=True; actor q-term disabled (pure BC flow).
  # False = off, bit-identical.
```

One knob. `measure_density_dim/_temp/_hard` are shared with spec 18.

### 2.2 `density_loss` delta

When `measure_density_behavior`:
- F/target-F input: `jnp.concatenate([obs, acts], -1)` /
  `jnp.concatenate([nobs, a_next], -1)` with
  `a_next = batch['next_actions']` (no stop_gradient needed — it is data;
  keep the rng path untouched so the off-branch stays bit-identical).
- Everything else (queries, k, w̄⁺ composition, CE, info keys) unchanged.
  Add info key `density_sarsa_subst` (from the dataset flag's counter or
  recompute the substitution mask if plumbing it is cheaper).

### 2.3 `actor_loss`

When `measure_density_behavior`: SKIP the spec-18 density q-term branch
entirely — the actor loss reduces to the pure BC-flow terms (verify what
the flow actor loss looks like with the q-term contribution removed;
there may already be a q-weight/actor-loss structure that makes this a
clean branch — mirror whatever the cleanest existing off-switch is; if
none exists, branch around the q-term with the config flag).
`actor_q_center` becomes irrelevant in this mode — see lattice.

### 2.4 `create()` / lattice

- Registration: same modules as spec 18 but with the F example input
  `[ex_observations, ex_actions]` (no goals) when behavior mode is on.
- Lattice (extend the spec-18 block):

```python
if config.get('measure_density_behavior', False):
    assert config.get('measure_density', False), "behavior mode is a variant of measure_density"
    # goal-free F: the spec-18 goal-conditioned asserts that reference the actor's
    # q-term/actor_q_center are replaced in this mode:
    assert config.get('actor_kind', 'flow') == 'flow'
    assert not config.get('actor_q_center', False), \
        "behavior mode has no actor q-term; actor_q_center would be an inert knob (17b lesson: refuse)"
```

  All other spec-18 lattice refusals (mutual exclusion with 17a/17b/fp,
  state-obs-only, no ortho, no warm start, ...) apply unchanged.
- The dataset `next_actions` flag must be forced on by this knob's config
  path (and off otherwise) — a behavior run that silently bootstrapped
  with garbage would be a fake result.

### 2.5 `update` override

Unchanged from spec 18 (the density target-update branches already fire
on `measure_density`).

## 3. Tests — `tests/test_gciql_measure_density_behavior.py`

Same harness as the spec-18 suite. Base config = spec-18 base +
`measure_density_behavior=True`.

1. `test_default_off_bit_identical` — behavior flag off (but
   measure_density on): loss bit-identical to the shipped spec-18 arm on
   a fixed batch (proves the branch is airtight); both off: identical to
   no-knob.
2. `test_f_input_width` — F tower's first dense layer expects
   obs_dim + action_dim (no goal width) in behavior mode.
3. `test_sarsa_bootstrap_matches_numpy` — numpy reference of the CE with
   the bootstrap features computed at `[nobs, batch['next_actions']]`;
   assert the learned-actor path is NOT called (e.g. patch/monitor the
   actor module or compare against a batch where actor params are
   randomized — bootstrap must be invariant to actor params).
4. `test_actor_is_pure_bc` — `jax.grad(actor_loss)`: zero over
   density_f/density_b/target leaves AND the loss value equals the BC
   flow terms alone (compute the q-term-free reference).
5. `test_next_actions_boundary` — synthetic dataset with ogbench-style
   trajectory layout: interior rows get actions[idx+1]; final-state rows
   get the current action; substitution fraction correct.
6. `test_lattice` — refuses: behavior without measure_density; with
   measure_delta / measure_delta_partition / measure_first_passage; with
   warm_start_ckpt. Accepts base.
7. `test_end_to_end_updates` — 30 updates: finite, simplex, no NaN.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density_behavior.py -x -q`,
then the spec-18 suite (shared code touched) — all green.

## 4. Configs (write after green; launch per spec 19 §1 P1.1)

`configs/gciql/{cube_single,cube_double,scene}_state_measure_density_behavior_d50.yaml`
— copy `cube_single_state_measure_density_d50.yaml` scaffolding; set
`measure_density_behavior: true`; hit_specs: cs `"19:22:0.4"`, cd/scene =
the multi-block specs the d17 delta configs use (read them). run_groups
`GCIQL-{CubeSingle,CubeDouble,Scene}-Flow-Measure-DensityBehavior-d50`.
Seed 0 first (spec 19 §1). Note cd/scene are FIRST training runs for the
density family on multi-object — also record the standing cd/sc eval
cadence (eval_interval 10000).

## 5. Out of scope

- The r_W reward and extractor (spec 19b).
- Any dwell weighting (spec 20).
- Goal-conditioned SARSA (meaningless), z-anything, pixel path.
