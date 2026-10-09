# Task 17b-agent report — `measure_delta_partition` agent-side implementation

Scope: §2 of `PAPER/Multistep-FB/specs/17b-delta-partition-impl.md`, entirely inside
`tools/wandb_mode_shim/gciql_flow.py`, class `GCIQLFlowAgent`. Tests were pre-written
(test-first) in `tests/test_gciql_measure_delta_partition.py` by the parallel
"17b-tests" wave (`.superpowers/sdd/spec17-delta/task-17b-tests-report.md`); this
report covers only the implementation that turns them green. 17a (`measure_delta`)
was NOT touched except reading its existing mutual-exclusion assert. No commits, no
`git add`/`stash`/`checkout`; tree left dirty per instructions.

## Touch points (file:line, current state of `tools/wandb_mode_shim/gciql_flow.py`)

- **Imports** (:11-19): added `from typing import Any` and `import numpy as np`
  (neither was previously imported at module scope; both required by the spec's
  literal code — `Any` for the dataclass field annotation, `np.load`/`np.asarray`
  for the prototype file).
- **§2.1 `get_config()` — 3 knobs** (~:3411-3419, right after the 17a `measure_delta`
  knob block and before `measure_expectile`): `measure_delta_partition = False`,
  `measure_delta_protos = ''`, `measure_delta_coef = 1.0`, docstrings copied
  verbatim from spec §2.1. (Not required by any test — the harness builds cfg dicts
  directly — but required by the spec for real launches via YAML overrides, since
  ml_collections ConfigDict rejects unknown-key overrides.)
- **§2.2 dataclass field** (~:744-750): `GCIQLFlowAgent` had ZERO extra dataclass
  fields before this change (pure `class GCIQLFlowAgent(GCIQLAgent): """docstring"""`
  then methods) — added `delta_protos: Any = None` as the first statement in the
  class body.
- **§2.3 `_delta_phi` staticmethod** (~:1276-1296): inserted directly after
  `_hit_matrix` (unchanged), before `_measure_terms`. Implementation matches the
  spec exactly: per-block `/eps` division on both `obs_rows` and `protos` (offset by
  cumulative block width), squared-distance logits, `jax.nn.softmax(-1)`.
- **§2.4 `delta_partition_loss`** (~:1759-1793): inserted directly after
  `measure_loss`'s `return loss, info`, before `centered_measure_loss`. Matches the
  spec's code verbatim: online `p` via `delta_head(..., params=grad_params)`;
  bootstrap `noise = jax.random.normal(rng, acts.shape)` drawn straight off the
  passed-in `rng` (**no split** — required bit-exactly by test 3); `a_plus` via the
  flow `actor` module, stop-gradded; `p_bar` via `target_delta_head` (its own live
  params, no override); `phi_n = self._delta_phi(nobs, ...)`; `p_tgt = sg[(1-γ)φ(x')+
  γ p̄]`; `ce = -mean(sum(p_tgt·log(p+1e-8)))`; `loss = measure_delta_coef * ce`. Info
  keys: `delta_ce`, `delta_p_entropy`, `delta_tgt_entropy`, `delta_p_max`,
  `delta_cell_change` (all 5, as the spec's §2.4 info-key list requires).
- **§2.5 `total_loss` wiring** (~:2168-2176): inside the pre-existing
  `if self.config.get('measure_only', False):` branch, added
  `if self.config.get('measure_delta_partition', False): m_loss, m_info =
  self.delta_partition_loss(...) else: m_loss, m_info = self.measure_loss(...)`
  before the `for k, v in m_info.items(): info[f'measure/{k}'] = v` loop — so 17b's
  info keys are prefixed `measure/delta_ce` etc., exactly mirroring the stock
  `measure/measure_loss` prefixing pattern the tests substring-match against. No
  other line in `total_loss` touched; the `rng, m_rng = jax.random.split(rng)` line
  and everything after the branch (task_prior_critic_mix, return) is unchanged.
- **§2.6 `actor_loss` q-term** (~:1024-1034 area, inside the flow-actor path): added
  a new `elif self.config.get('measure_delta_partition', False):` branch **between**
  the pre-existing `elif z_infer is not None:` branch and the final
  `else: zarg = ...` branch (rather than an `if` physically before the final
  `q1, q2 = self.network.select('critic')(...)` line — functionally identical
  "runs before the stock branch," see Deviations below). Computes
  `z_g = sg[_delta_phi(goals, self.delta_protos, blocks)]`,
  `logits_a = delta_head([obs, actor_actions, goals])` (**no `params=grad_params`**
  — frozen, confirmed by test 6), `q1, q2 = (softmax(logits_a,-1) * z_g[None]).sum(-1)`.
  Does not touch the `rng, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)`
  line or `actor_actions` sampling above it — both are computed identically
  regardless of the knob, satisfying risk #1 (rng-split sequence unchanged).
- **§2.7 `create()`**:
  - **Assert-lattice** (~:2745-2779): new `if config.get('measure_delta_partition',
    False):` block immediately after the existing 17a mutual-exclusion assert (17a's
    own block, at ~:2741-2742, was read but not edited). Every check from the spec's
    §2.7 code is present, but I **split every combined `assert X and Y`
    into separate named asserts** (see Deviations) so each refusal has its own
    message. Full mapping verified below.
  - **Module registration** (~:3105-3125): inserted between the `nhat_head` block
    and `if config.get('actor_kind', 'flow') == 'ddpgbc':`. Loads
    `delta_protos_np = np.load(config['measure_delta_protos'])`, asserts
    `ndim==2 and shape[1]==d_c and shape[0]>=2` (own message), builds
    `dh_def = ensemblize(MLP, 2)((*value_hidden_dims, K), activate_final=False,
    layer_norm=...)`, registers `network_info['delta_head']` and
    `network_info['target_delta_head']` (via `copy.deepcopy(dh_def)`) with example
    input `concat([ex_observations, ex_actions, ex_goals], -1)`.
    `delta_protos_np = None` is initialized unconditionally above the `if`, so the
    variable is always defined for the later `return`.
  - **Target-init mirror** (~:3299-3303): right after the pre-existing
    `params['modules_target_critic'] = params['modules_critic']`, added
    `if config.get('measure_delta_partition', False): params['modules_target_delta_head']
    = params['modules_delta_head']`.
  - **Return** (~:3304-3307): `delta_protos = jnp.asarray(delta_protos_np,
    jnp.float32) if delta_protos_np is not None else None`; `return cls(rng,
    network=network, config=..., delta_protos=delta_protos)`.
- **§2.8 `update` override** (~:2266-2284, inserted between `total_loss` and
  `sample_actions`): `@jax.jit def update(self, batch):` — identical body to the
  inherited `GCIQLAgent.update` (`agents/gciql.py:157`) plus
  `if self.config.get('measure_delta_partition', False): self.target_update(
  new_network, 'delta_head')` after the stock `self.target_update(new_network,
  'critic')` call. Confirmed `third_party/ogbench/impls/main.py` calls
  `agent.update(batch)` (no other entry point) per the task brief; did not re-verify
  independently since the task brief already confirmed it.

## Assert-lattice mapping (test 8, hand-verified via direct subprocess calls, not inferred)

| `test_assert_lattice` case | Assertion message actually raised |
|---|---|
| `measure_only=False` | `measure_delta_partition (17b) replaces the measure loss: set measure_only=True` |
| `measure_delta=True` | `measure_delta rides the hit-region path: set measure_first_passage=True` **(fired by 17a's own lattice, not 17b's)** — see below |
| `measure_first_passage=True` | `measure_delta_partition does not use the region-matrix path: set measure_first_passage=False` |
| `measure_delta_protos=''` | `measure_delta_partition needs measure_delta_protos (.npy path)` |
| wrong-width protos `[8,5]` | `measure_delta_protos must be [K>=2, 3], got (8, 5)` |
| `actor_q_center=False` | `measure_delta_partition requires actor_q_center=True (p.z_g carries the action-independent level phi(x0).z_g)` |
| `warm_start_ckpt='/nonexistent/...'` | `measure_delta_partition is incompatible with warm_start_ckpt` |
| `measure_ortho_coef=1.0` | `no ortho anywhere near measure_delta_partition (fp-noortho lesson)` |
| base config | no assertion (accepted) |

**Flagged risk #2 resolved**: 7 of 8 refusals fire the intended 17b-specific named
assertion. The `measure_delta=True` case is the one exception: `base_cfg()` sets
`measure_first_passage=False`, and the pre-existing 17a block (`if
config.get('measure_delta', False): assert config.get('measure_first_passage',
False), "measure_delta rides the hit-region path..."`) runs textually *before* my
17b block and raises first on that unrelated condition — my own 17b mutual-exclusion
assert (`assert not config.get('measure_delta', False), "measure_delta (17a) and
measure_delta_partition (17b) are mutually exclusive"`) is reachable and correct
(verified independently: setting `measure_delta=True, measure_first_passage=True,
measure_fp_absorbing=False` alongside `measure_delta_partition=True` does hit my
assert's exact message) but is dead code on the specific input `test_assert_lattice`
exercises. This is inherent to the two lattices both existing in the same `create()`
and is not a bug — test 8 only asserts "some AssertionError fires," which holds.

## Deviations from the spec's literal code (and why)

1. **Assert-lattice: split combined asserts.** Spec §2.7's code block has three
   compound asserts (`actor_kind=='flow' and actor_cond=='goal'`; `nstep==0 and not
   eps_multiscale`; `centered_measure=='off' and task_prior=='dirac' and
   task_prior_actor=='dirac'`). I split each into separate `assert` statements with
   individual messages. Same set of refused configs, same behavior — just clearer
   `AssertionError` text per failure mode, consistent with the task brief's ask to
   verify "each refusal fires *its own* named assertion."
2. **`actor_loss` q-term: `elif` instead of a bare `if` before the final line.** The
   spec's snippet reads "add a branch BEFORE it" without full surrounding context;
   I implemented it as a new `elif` clause slotted between the existing
   `elif z_infer is not None:` and the final `else:` of the same if/elif/elif chain
   (rather than nesting a stand-alone `if measure_delta_partition: ... else: <original
   body>` inside the final `else:`). Net effect is identical — when the knob is on,
   `q1, q2` are set by the new branch and the stock `else:` body (zarg/greedy_pick/
   critic call) never executes — but it doesn't require duplicating or re-indenting
   the existing `else:` block's `zarg`/`greedy_pick` code, so the diff against the
   pre-existing branch is smaller. Since the lattice already forces `value_arch !=
   'dueling'` (measure_only excludes dueling elsewhere) and `actor_cond=='goal'`
   (so `z_infer is None`), the new `elif` is reached whenever
   `measure_delta_partition=True`, matching the spec's intent.
3. **Local variable names `blocks17b`, `_blocks17b_create`, `_K17b`, `_d_c17b`.**
   Not in the spec; used to avoid shadowing the `blocks` name already bound
   elsewhere in `actor_loss`'s scope, and to signal these are 17b-owned temporaries
   inside `create()`'s otherwise unrelated body.

## Test output

### 1. `tests/test_gciql_measure_delta_partition.py` — 10/10 green

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py -q
tests/test_gciql_measure_delta_partition.py ..........                   [100%]
======================== 10 passed in 101.42s (0:01:41) ========================
```
(Re-run after the `get_config()` knob addition — unchanged, still 10/10 in 101.21s.)

### 2. `tests/test_gciql_measure_delta.py` (17a) — 8/8 green, unaffected

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta.py -q
tests/test_gciql_measure_delta.py ........                               [100%]
========================= 8 passed in 78.38s (0:01:18) =========================
```

### 3. Regression neighborhood — 26/26 green

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_nstep.py tests/test_gciql_measure_hit_spec.py \
    tests/test_gciql_measure_axes.py tests/test_gciql_measure_scale.py tests/test_gciql_measure_zmix.py -q
tests/test_gciql_measure_nstep.py ......                                 [ 23%]
tests/test_gciql_measure_hit_spec.py ......                              [ 46%]
tests/test_gciql_measure_axes.py ......                                  [ 69%]
tests/test_gciql_measure_scale.py .....                                  [ 88%]
tests/test_gciql_measure_zmix.py ...                                     [100%]
======================== 26 passed in 282.25s (0:04:42) ========================
```

## Grad hygiene (crl f7b623d lesson) — confirmed by test 6, both directions

- `actor_loss`'s new branch calls `delta_head` **without** `params=grad_params` →
  `jax.grad(actor_loss)` grad-norm over `modules_delta_head` +
  `modules_target_delta_head` + `modules_critic` < 1e-9.
- `delta_partition_loss`'s bootstrap action is `stop_gradient(actor(...))` →
  `jax.grad(delta_partition_loss)` grad-norm over `modules_actor` < 1e-9; over
  `modules_delta_head` > 1e-8 (correctly trained); over `modules_target_delta_head`
  == 0 (no gradient path — it's soft-updated outside the loss, per §2.8).

## Not touched (per scope)

`scripts/rescuing_fb/make_delta_prototypes.py`, `configs/gciql/
cube_single_state_measure_delta_partition_d50.yaml`, `tests/
test_gciql_measure_delta_partition.py` (test file itself), and the 17a
(`measure_delta`) code path — read-only. No training launched, no real prototypes
generated.

---

## Round 1 fixes (independent Opus review, 2 launch blockers + 1 medium + 1 low)

### CRITICAL 1 — partition arm trained nothing (measure_coef double-gate)

**Root cause**: `total_loss`'s `measure_only` branch computed
`loss = self.config.get('measure_coef', 1.0) * m_loss + actor_loss` unconditionally.
`measure_coef` defaults to **0.0** in `get_config()` (:3391, pre-existing, untouched),
and spec 17b §6's shipped launch config deliberately drops it (since the 17b path
doesn't need it — `delta_partition_loss` already applies `measure_delta_coef`
internally). Result: `delta_partition_loss`'s CE was multiplied by 0.0 a second time,
giving `delta_head` exactly zero gradient. `total_loss` still returned a finite loss
and a plausible `delta_ce`/entropy info dict, reading as the spec's pre-registered
scientific NULL rather than a training bug.

**Fix** (`tools/wandb_mode_shim/gciql_flow.py`, `total_loss`, ~:2183-2202): split the
`measure_only` branch's final loss assembly —
```python
if self.config.get('measure_delta_partition', False):
    loss = m_loss + actor_loss              # delta_partition_loss already applied measure_delta_coef
else:
    loss = self.config.get('measure_coef', 1.0) * m_loss + actor_loss
```
The non-partition (`measure_loss`) path is byte-identical to before.

**Regression tests added** (`tests/test_gciql_measure_delta_partition.py`):
- `test_shipped_config_trains_delta_head` (new, #11): builds a config from the
  shipped knob set (`measure_coef` absent → defaults to 0.0, matching
  `cube_single_state_measure_delta_partition_d50.yaml`), asserts
  `jax.grad(total_loss)` has nonzero norm over `modules_delta_head`, and that 30
  `update()` steps measurably move `delta_head` params.
- `test_target_head_init_and_soft_update` (#4) — **rewritten**, see HIGH finding below.

**HIGH finding (pre-existing test bug, exposed by the CRITICAL 1 fix)**: the
original test 4 computed `expected = tau*dh1 + (1-tau)*tdh0` using `dh1`, the
**post**-update online params. But `target_update` (base class, `agents/gciql.py:147`,
unmodified) reads `self.network.params[...]` — the **pre**-update online params —
when computing the soft update, not the just-produced `new_network`'s params. Before
the CRITICAL 1 fix, `dh1 == dh0` exactly (zero gradient), so the two formulas
coincided and the test passed trivially regardless of whether the `update()`
override existed. After the CRITICAL 1 fix, `dh1 != dh0`, exposing a genuine ~1.5e-6
discrepancy between the test's (wrong) formula and the code's actual behavior — the
test started failing for the WRONG reason (a bug in the test, not in the code).
Verified by direct experiment (`jax.grad`-based probe): the exact formula the code
computes is `tau*dh0 + (1-tau)*tdh0` (diff 0.0 to full float32 precision), vs the
test's `tau*dh1 + (1-tau)*tdh0` (diff 1.5e-6). On a single update step this formula
is inherently degenerate (target==online at init, so `target_update`'s own formula
reduces to "no change" whether or not it's called) — no single-step check can be
override-sensitive. **Rewrote test 4 to run two `update()` steps**: the SECOND
target's expected value is computed from the online/target params as they stood
**going into** the second call (i.e. `dh1`/`tdh1`, captured after step 1), matching
`target_update`'s actual semantics exactly (atol=1e-7, effectively 0). Also added an
explicit "delta_head actually moved after step 1" assertion so the test's own
gradient-flow premise is checked, not assumed.

**Verified by hand** (both before and after the test rewrite): temporarily deleted
the `update()` override's `if self.config.get('measure_delta_partition', False):
self.target_update(new_network, 'delta_head')` line (replaced with a commented-out
placeholder, `__pycache__` cleared to force recompilation), re-ran test 4 — **fails**
(`AssertionError: 1.4999896e-06` under the corrected two-step formula, meaning the
target genuinely stayed frozen relative to the moved online params). Restored the
override line — test 4 **passes** again. Confirms the test is now genuinely
sensitive to the override, both before my rewrite (with the wrong formula it failed
for the wrong numerical reason) and after (it fails/passes for the RIGHT reason).

### CRITICAL 2 — `delta_protos` broke restore of every pre-17b checkpoint

**Root cause**: `delta_protos` was declared `Any = None`, a default **pytree**
(dynamic) field. `restore_agent` (`third_party/ogbench/impls/utils/flax_utils.py:198`)
calls `flax.serialization.from_state_dict(agent, load_dict['agent'])` on the WHOLE
agent object, not just `agent.network`. flax's generated `from_state_dict` for a
`flax.struct` dataclass iterates only over the pytree-node fields and raises
`ValueError: Missing field <name>` for any that's absent from the state dict. Every
pre-17b checkpoint's `load_dict['agent']` dict has exactly `{'rng', 'network'}` (it
was pickled before `delta_protos` existed on the class at all), so restore would
raise `Missing field delta_protos` for ANY agent object built with the post-17b code
— regardless of whether `measure_delta_partition` is on or off in that run's config.

**Fix**: made `delta_protos` a **static/nonpytree field**
(`tools/wandb_mode_shim/gciql_flow.py` :746-763):
```python
delta_protos: Any = nonpytree_field(default=None)
```
(imported `nonpytree_field` from `utils.flax_utils`, the same helper `config` already
uses.) Verified against flax's own source
(`flax/struct.py`, `dataclass()`): `to_state_dict`/`from_state_dict` for a
`flax.struct` dataclass are generated to walk **only** `data_fields` (the
`pytree_node=True` fields, i.e. `meta_fields`/static fields are entirely excluded).
So a nonpytree `delta_protos` is (a) never emitted by `to_state_dict` — old
checkpoints written by the fixed code would ALSO never mention it — and (b) never
required or touched by `from_state_dict` — restoring leaves the field at whatever
value the constructing `create()` call gave it, exactly the spec's original ("old
checkpoints keep loading") intent, now actually true.

Static fields must be hashable for `jax.jit`'s cache key (a raw numpy/jnp array is
not, and its `==` is elementwise/ambiguous), so the stored value is now a **hashable
nested tuple of Python floats** (`[K, d_c]` row-major), built in `create()`:
```python
delta_protos = (tuple(tuple(float(v) for v in row) for row in delta_protos_np)
                if delta_protos_np is not None else None)
```
`_delta_phi` (`tools/wandb_mode_shim/gciql_flow.py` :1317-1338) now does
`protos = jnp.asarray(protos, jnp.float32)` as its first line, so every call site
(including test call sites that pass a raw array directly) works unchanged; inside
`jax.jit`, the nested tuple becomes a folded constant baked into the trace, and the
per-call Python-level hash of a ~256×3 nested tuple is negligible (reviewer's own
characterization, confirmed by the test suite's wall-clock times showing no
regression).

**Regression test added**: `test_delta_protos_not_serialized_and_old_checkpoint_restores`
(new, #12, `tests/test_gciql_measure_delta_partition.py`) — proves (a)
`flax.serialization.to_state_dict` of both an OFF-knob and an ON-knob agent never
emits a `delta_protos` key and only ever has `{'rng','network'}`; (b)
`flax.serialization.from_state_dict` against a state dict containing exactly
`{'rng','network'}` (the literal shape of a pre-17b checkpoint's agent dict, restored
into an agent whose config/architecture matches — exactly what `restore_agent` does
in practice) succeeds without raising, and the restored agent's `delta_protos` is
left untouched (still whatever the constructing agent had — a static field is not an
argument `from_state_dict` can override).

**Hand-verification against the real anchor checkpoint** (as instructed), run in the
jax venv (`/dev/shm/.venv-jax/bin/python`), config reconstructed from the
checkpoint's own `flags.json`, `obs_dim`/`action_dim` (37/5) confirmed via
`ogbench.make_env_and_datasets('cube-double-play-v0')`:
```
checkpoint: /home/adityamohan/exp_spec08/factored-fb/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0/params_500000.pkl

agent created OK, delta_protos = None
checkpoint agent dict keys: ['rng', 'network']
RESTORE SUCCEEDED. delta_protos on restored agent: None
step: 500001
```
This is the exact pkl the reviewer reproduced the failure against; with the fix it
restores cleanly and reports the correct trained step count (500001).

### MEDIUM 3 — gate G2 instrument (`measure_delta_msq`)

Spec 17 §6 gate G2 ("no zero-flux collapse: msq bounded away from 0") had no
readout — `_measure_terms` computes `msq = mean(M**2)` internally for
`measure_scale_norm` but discards it, and the scale-normed loss terms are
insensitive to `msq` by construction (they divide it out). Added
`info['measure_delta_msq'] = jnp.mean(M ** 2)` **on the 17a `measure_delta` path
only** (`tools/wandb_mode_shim/gciql_flow.py`, `measure_loss`, inside the existing
`if self.config.get('measure_delta', False):` block, gated identically to
`measure_delta_event_rate`/`measure_delta_hit_level` — same `if delta_event_rate is
not None:` guard at the info-assembly site, so the off path and the plain
first-passage path emit no new key). `tests/test_gciql_measure_delta.py::test_info_keys`
extended to assert the key is present-and-finite-and-nonnegative when
`measure_delta=True`, and absent both when `measure_delta=False` and on the plain
(absorbing) first-passage path — all three sub-cases pass.

### LOW 4 — stale docstring

`tests/test_gciql_measure_delta_partition.py`'s module docstring (previously
claiming tests 1-8/10 are expected-red pre-implementation) rewritten to reflect the
current state: all 12 tests (10 original + 2 round-1 regression tests) are green,
including the round-1 fixes.

## Verification (final state, after all round-1 fixes)

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py tests/test_gciql_measure_delta.py -q
tests/test_gciql_measure_delta_partition.py ............                 [ 60%]
tests/test_gciql_measure_delta.py ........                               [100%]
======================== 20 passed in 203.94s (0:03:23) ========================

$ .venv/bin/python -m pytest tests/test_gciql_measure_nstep.py tests/test_gciql_measure_hit_spec.py \
    tests/test_gciql_measure_axes.py tests/test_gciql_measure_scale.py tests/test_gciql_measure_zmix.py -q
tests/test_gciql_measure_nstep.py ......                                 [ 23%]
tests/test_gciql_measure_hit_spec.py ......                              [ 46%]
tests/test_gciql_measure_axes.py ......                                  [ 69%]
tests/test_gciql_measure_scale.py .....                                  [ 88%]
tests/test_gciql_measure_zmix.py ...                                     [100%]
======================== 26 passed in 280.54s (0:04:40) ========================
```

## Not touched (per coordinator's ruling)

- The unreachable 17b mutual-exclusion assert in the `measure_delta=True` test-8 case
  (17a's own lattice refuses that combination first) — left as-is, dead code only.
- `measure_scale_norm` / learning rate — the delta-vs-ballvisit scale difference is a
  science property of the recipe, not touched.

---

## Round 2 fixes (scoped re-review — all round-1 fixes verdicted ADDRESSED; 3 LOW cleanup items, code already committed at 79a9c6b and the wave is live)

### Item 1 — stale docstring (LOW)

`tests/test_gciql_measure_delta_partition.py`'s module docstring said "All 10 tests"
(there are 12 after round 1's additions) and pointed CRITICAL 2's regression test at
a nonexistent `tests/test_gciql_checkpoint_restore.py`. Corrected: "All 12 tests",
and named the actual regression tests in THIS file —
`test_shipped_config_trains_delta_head` (#11, CRITICAL 1) and
`test_delta_protos_not_serialized_and_old_checkpoint_restores` (#12, CRITICAL 2).

### Item 2 — tautological restore assertion (LOW, but "the exact regression that
nearly burned the wave")

The round-1 version of test 12 asserted `restored.delta_protos ==
agent_off.delta_protos`, comparing `None == None` for an OFF-knob agent — true
regardless of whether restore preserves static fields correctly. Rewrote the test to
be a real, discriminating check:
- Two ON-knob agents built from **different** prototype files (same K/d_c, so the
  network's `delta_head`/`target_delta_head` module shapes match — only the
  prototype **values** differ): `agent_on` (seed 0) and `agent_on2` (seed 42).
- `agent_on2` is restored from a `{'rng','network'}` dict sourced from `agent_on`'s
  own `to_state_dict` (a dict that, like a real checkpoint, never mentions
  `delta_protos`).
- Asserts (a) the restore succeeds; (b) `restored2.delta_protos ==
  agent_on2.delta_protos` — the agent's **own, full, unclobbered** prototype tuple,
  not `None`; (c) `restored2.delta_protos != agent_on.delta_protos` — proving it
  wasn't silently overwritten by the source agent's (different) prototypes either;
  (d) `_delta_phi` on the restored prototypes reproduces `agent_on2`'s pre-restore
  output bit-for-bit (`np.array_equal`).
- Per the review's explicit instruction, also documents the genuinely-impossible
  case rather than hiding it: restoring an ON-knob agent from a dict whose
  **network** was built entirely without `delta_head`/`target_delta_head` modules
  (the literal shape of a real pre-17b checkpoint, sourced from the OFF-knob
  agent's `to_state_dict`) is asserted to **raise**, and the raised message is
  checked to name the missing `delta_head`/`modules_delta_head` module (from
  `flax.serialization`'s plain-dict restore path, `_restore_dict`) and to **not**
  mention `delta_protos` — proving the failure is a structural network-architecture
  mismatch (expected: that checkpoint never trained a `delta_head`), not a
  delta_protos regression. This is exactly what `restore_agent` does in practice
  (the checkpoint's config always matches its own architecture), so it is not a
  real-world blocker — just documented rather than left unstated.

### Item 3 — `measure_coef` silently inert on the 17b path (LOW)

Added an assert-lattice line in `create()`
(`tools/wandb_mode_shim/gciql_flow.py`, inside the `if
config.get('measure_delta_partition', False):` block, next to the `measure_ortho_coef`
check):
```python
assert config.get('measure_coef', 0.0) <= 0.0, \
    "measure_delta_partition ignores measure_coef (... use measure_delta_coef instead ...)"
```
Refuses `measure_coef > 0` (whether the caller relied on the ambient `.get`
fallback of 1.0 or an explicit positive value) with a message naming
`measure_delta_coef` as the correct knob, so a later dose sweep aimed at the wrong
knob fails loudly at `create()` instead of silently producing identical runs with
misleading recorded provenance. `config.get('measure_coef', 0.0) <= 0.0` accepts
both an absent key (the shipped config's own knob set) and an explicit `0.0`.
Extended `test_assert_lattice` with `refuses(measure_coef=1.0)`,
`refuses(measure_coef=0.1)`, and confirmed `not refuses(measure_coef=0.0)` (the
shipped config's own knob set must not trip it).

**Confirmed against the actual shipped config** (constructed the real agent from
`configs/gciql/cube_single_state_measure_delta_partition_d50.yaml`'s
`agent_overrides`, real cube-single env dims `obs=28, act=5`, the real generated
prototypes at `results/delta_protos/cube_single_K256_s0.npy`):
```
measure_coef in cfg: 0.0
real obs/act dims: 28 5
AGENT CONSTRUCTED OK. delta_protos shape (K): 256 d_c: 3
```

## Verification (final state, after round-2 fixes)

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py tests/test_gciql_measure_delta.py -q
tests/test_gciql_measure_delta_partition.py ............                 [ 60%]
tests/test_gciql_measure_delta.py ........                               [100%]
======================== 20 passed in 205.37s (0:03:25) ========================
```

No changes to `tools/wandb_mode_shim/gciql_flow.py` beyond the single new
`measure_coef` assert; no changes to `configs/gciql/
cube_single_state_measure_delta_partition_d50.yaml`, `scripts/rescuing_fb/
make_delta_prototypes.py`, or 17a. Nothing committed/staged (rules unchanged); the
code the wave is running on (commit 79a9c6b) is untouched by these cleanup items —
they are test-file and one new defensive assert only.
