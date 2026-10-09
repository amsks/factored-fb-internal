# Spec 19a — `measure_density_behavior` implementation report

Status: GREEN. Branch `gciql-fb-bridge`, tree left dirty (no commits per hard rules).

## Files touched

- `tools/wandb_mode_shim/gciql_flow.py`
- `third_party/ogbench/impls/utils/datasets.py`
- `tests/test_gciql_measure_density_behavior.py` (new)
- `configs/gciql/cube_single_state_measure_density_behavior_d50.yaml` (new)
- `configs/gciql/cube_double_state_measure_density_behavior_d50.yaml` (new)
- `configs/gciql/scene_state_measure_density_behavior_d50.yaml` (new)

## Touch points (file:line)

**Dataset (`third_party/ogbench/impls/utils/datasets.py`)**
- `GCDataset.sample`, L225–242: gated on `self.config.get('next_actions', False)`.
  Uses the SAME trajectory-boundary arithmetic `sample_goals`/`measure_nstep` already
  use (`self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]`) — not the
  naive `min(idx+1, size-1)` the pre-existing `OGBENCH_NEXT_ACTIONS=1` sitecustomize
  shim (`tools/wandb_mode_shim/sitecustomize.py:161-179`, untouched) uses, which is
  only boundary-safe for compact/`valids`-gated datasets. Cube/scene datasets are
  REGULAR (explicit `next_observations`, `ogbench/utils.py` `compact_dataset=False`
  default), where a bare `idx+1` at a trajectory's last row crosses into the next
  trajectory's first action. Final-state rows substitute `batch['actions']`;
  `batch['next_actions_subst']` (float32 0/1 per row) exposes the substitution mask.

**Agent (`tools/wandb_mode_shim/gciql_flow.py`)**
- `get_config()` L3729 `measure_density_behavior=False`, L3734 `next_actions=False`.
- `density_loss` L1922 `behavior = self.config.get(...)`; L1924-1930 goal-free F
  input (`[obs,acts]` vs `[obs,acts,g]`); L1936-1946 SARSA bootstrap
  (`a_plus = batch['next_actions']`, target F called on `[nobs, a_plus]`, no actor
  call, no rng consumption) vs the untouched actor-draw branch; L1987-1989
  `info['density_sarsa_subst'] = jnp.mean(batch['next_actions_subst'])` (a plain
  mean of the row indicator — matches spec's "fraction" statistic exactly, no
  mean/median mixup).
- `actor_loss` L1051-1073: `measure_density`'s q-term branch now checks
  `measure_density_behavior` first and sets `q1 = q2 = None` (density towers never
  called) instead of computing the goal-kernel readout. L1097-1104 and L1113-1127:
  `q1 is None` / `q is None` guards make `actor_loss` fall through to
  `bc_loss` (+`bc_flow_loss`) alone, skipping the q/q_scale normalization entirely;
  `q_mean`/`q_abs_mean` report `0.0` in this mode.
- `create()` registration L3433-3438: `ex_f` example input drops the goal leg
  (`[ex_observations, ex_actions]`) when behavior mode is on, so `density_f`'s first
  Dense layer is actually sized `obs_dim+action_dim`, not just called that way.
- Lattice L3014-3039 (spec-18 block): L3026-3027 the `actor_q_center` assert is now
  `if not measure_density_behavior: assert actor_q_center`. L3050-3071 (new,
  extending the spec-18 block): standalone `if measure_density_behavior:` block —
  requires `measure_density=True`, `actor_kind='flow'`, refuses
  `actor_q_center=True` (17b fake-NULL lesson), and forces the dataset flag on.

**Forcing mechanism (the load-bearing part).** `create()` originally did
`config = dict(config)` (L2536, now L2545) — a COPY — before any lattice mutation,
so the pre-existing `config['actor_cond']='goal'`-style patterns elsewhere in the
file only ever mutated a local copy, never anything visible outside `create()`. That
is fine for those (agent-internal) knobs but would have SILENTLY FAILED to reach
`GCDataset`: in the real entrypoint (`third_party/ogbench/impls/main.py:57-84`),
`train_dataset = dataset_class(Dataset.create(**train_dataset), config)` runs
*before* `agent_class.create(...)`, and `GCDataset.config` is a live reference to
the **same** `FLAGS.agent` ConfigDict object — every `.sample()` call reads
`self.config.get('next_actions', False)` fresh, so whatever that object holds at
call time (after `create()` returns) is what the dataset sees. I added
`orig_config = config` at L2543 (before the copy) to keep a handle on the caller's
original object, and the lattice block (L3070-3071) now writes
`config['next_actions'] = True` (local copy, so `agent.config`/every assert below
also sees it) **and** `orig_config['next_actions'] = True` (the caller's object —
this is what actually reaches `GCDataset`). Verified directly: `test_lattice` in
the new suite asserts `cfg['next_actions'] is True` on the SAME dict object passed
into `GCIQLFlowAgent.create(...)` after the call returns — this would have failed
under the pre-fix `config = dict(config)`-only version (confirmed: it did, on the
first test run, before this fix was added).

## Verification output

**1. New suite**
```
.venv/bin/python -m pytest tests/test_gciql_measure_density_behavior.py -q
======================== 7 passed in ~79s ========================
```
All 7 spec §3 tests green: default-off bit-identical (vs `git show HEAD`, both the
behavior-off/density-on arm AND the fully-off arm), F input width, SARSA bootstrap
numpy match + actor-param invariance, actor-is-pure-BC (grad zero over density/
critic leaves, loss == bc_flow_loss + bc_coeff*distill_error exactly), GCDataset
next_actions boundary arithmetic + substitution fraction, lattice refusals +
`next_actions` forcing, 30-step end-to-end.

**2. Shared-code regression (spec 18/17a/17b)**
```
.venv/bin/python -m pytest tests/test_gciql_measure_density.py tests/test_gciql_measure_delta.py tests/test_gciql_measure_delta_partition.py -q
=================== 1 failed, 30 passed in 335.05s ===================
FAILED tests/test_gciql_measure_density.py::test_default_off_bit_identical
```
This failure is PRE-EXISTING and unrelated to spec 19a: that test diffs the
current module against `git show HEAD:tools/wandb_mode_shim/gciql_flow.py` under
the hardcoded assumption `"measure_density" not in baseline_src` (i.e. that HEAD
predates spec 18). HEAD already contains spec 18 (`measure_density`, 24
occurrences, confirmed via `git show HEAD:... | grep -c measure_density` before any
edit in this session) — the test's own baseline target went stale when spec 18
landed on `main`/this branch's HEAD, independent of anything touched here. My
edits never modify `tests/test_gciql_measure_density.py`.

**3. GCDataset-touching suites (next_actions off-path byte-identical)**
```
.venv/bin/python -m pytest tests/test_gciql_measure_nstep.py tests/test_gciql_simplesf.py tests/test_goal_mix.py tests/test_gciql_sf_feature_td.py tests/test_ogbench_drq_injection.py -q
======================== 36 passed in 346.45s ========================
```

**4. Config construction (agent creation, no lattice trip)**
```
cube_single_state_measure_density_behavior_d50.yaml -> OK, next_actions forced: True
cube_double_state_measure_density_behavior_d50.yaml -> OK, next_actions forced: True
scene_state_measure_density_behavior_d50.yaml       -> OK, next_actions forced: True
```
Hit-specs confirmed verbatim: cs `19:22:0.4`; cd `19:22:0.4,28:31:0.4`; scene
`19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6`. `eval_interval`: cs 50000,
cd/scene 10000. `run_group`: `GCIQL-{CubeSingle,CubeDouble,Scene}-Flow-Measure-
DensityBehavior-d50`. None of the three configs sets `actor_q_center` (the lattice
would refuse it).

## Deviations from the literal spec text

- Spec's lattice pseudocode (§2.4) shows the behavior asserts as one flat block;
  I split the `actor_q_center` check across two sites (the pre-existing spec-18
  block, now conditional, plus the new standalone block) because the spec-18 block
  UNCONDITIONALLY asserted `actor_q_center=True`, which would otherwise always fire
  first and contradict the new block on any behavior-mode config. Net effect
  matches the spec's intent exactly (behavior mode refuses `actor_q_center=True`).
- Added the `orig_config`/dual-write forcing mechanism, not specified in the doc at
  that level of detail — required because `create()`'s pre-existing `config =
  dict(config)` copy would otherwise make the "force `next_actions` on" bullet a
  no-op outside `create()`'s own return value.
- `density_sarsa_subst` is only added to `density_loss`'s info dict when behavior
  mode is on (absent, not zero, when off) — keeps the off-path info dict untouched
  for the bit-identical gate.

## Concerns

- The one pre-existing failing test (`test_gciql_measure_density.py::
  test_default_off_bit_identical`) should be re-pointed to a later baseline commit
  or otherwise fixed by whoever owns spec 18's test file — flagging, not fixing,
  since it's out of this task's scope and touching a committed test file wasn't
  requested.
- Training runs not launched (per hard rules); configs are seed 0 only, as spec 19
  §1 P1.1 specifies (3 seeds on cs gated on the P1.3 tie check, not part of this task).
