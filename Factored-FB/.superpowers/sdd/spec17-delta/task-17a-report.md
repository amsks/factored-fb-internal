# Task 17a report — `measure_delta` implementation

Spec: `PAPER/Multistep-FB/specs/17a-delta-measure-impl.md`. Scope: 3 touch points in
`tools/wandb_mode_shim/gciql_flow.py`, `tests/test_gciql_measure_delta.py`, 5 config
yamls. Nothing outside this scope was touched. No commits/staging/git-history changes
made (per hard rules); working tree left dirty as instructed.

## Files changed / created

- `tools/wandb_mode_shim/gciql_flow.py` (edited, 3 touch points)
- `tests/test_gciql_measure_delta.py` (created, 8 tests)
- `configs/gciql/cube_double_state_measure_delta_d50.yaml` (created)
- `configs/gciql/cube_double_state_measure_delta_hs_d50.yaml` (created)
- `configs/gciql/scene_state_measure_delta_d50.yaml` (created)
- `configs/gciql/scene_state_measure_delta_hs_d50.yaml` (created)
- `configs/gciql/cube_single_state_measure_delta_d50.yaml` (created)

## 1. Code changes (`tools/wandb_mode_shim/gciql_flow.py`)

### 1.1 `get_config()` — new knob

`config.measure_delta = False` inserted immediately after `config.measure_nstep_lambda`
(now at gciql_flow.py:3198-3203), verbatim comment from spec 17a §1.1.

### 1.2 `measure_loss` — fp region branch

- Sentinel `delta_event_rate = delta_hit_level = None` added next to the existing
  `nstep_k_mean = None` line (gciql_flow.py:1400, right after the pre-existing line
  ~1399).
- The delta block inserted immediately after the non-multiscale
  `hit = self._hit_matrix(nobs, col_states, blocks, M.dtype)` line, i.e. inside the
  `else:` of `if ms_spec: ... else: ...` (gciql_flow.py:1525-1534). Transcribed
  verbatim from spec: `hit_n = hit; hit_c = self._hit_matrix(obs, col_states, blocks,
  M.dtype); hit = hit_n - hit_c; delta_event_rate = jnp.mean(jnp.abs(hit));
  delta_hit_level = jnp.mean(hit_n)`.
- Info-append block added at the bottom of `measure_loss`, mirroring
  `measure_nstep_k_mean` (gciql_flow.py:1713-1715): `if delta_event_rate is not None:
  info['measure_delta_event_rate'] = delta_event_rate; info['measure_delta_hit_level']
  = delta_hit_level`.
- No changes to the `_measure_terms(...)` call site, `_measure_terms` itself, or the
  multiscale/expectile/nhat_head branches, exactly as the spec specifies (they compose
  or refuse via the lattice, not via new branch logic).

### 1.3 `create()` — assert-lattice

New `if config.get('measure_delta', False): ...` block inserted right after the
spec-16 `measure_nstep` lattice's closing line (`assert config['encoder'] is None, ...
window gather)`) and before the RG-FB `task_prior` lattice comment
(gciql_flow.py:2631-2655). Transcribed verbatim from spec 17a §1.3, including the
`measure_delta_partition` mutual-exclusion guard (via `config.get(...)`, so 17a merges
independently of whether 17b's knob exists yet).

## 2. Tests (`tests/test_gciql_measure_delta.py`)

8 tests (7 from spec §3 + 1 extra static check for the "no `jnp.clip`" boundedness
claim, split out of `test_bounded_target` into its own
`test_bounded_target_no_clip_in_source` so the numeric check and the source-text check
have independent failure modes):

1. `test_default_off_bit_identical` — param tree + `measure_loss` bit-identical vs a
   config predating the knob, for tpd batch AND hindsight.
2. `test_delta_target_matches_numpy_reference` — constructed batch realizing all 4
   transition types; monkeypatches `_measure_terms` to capture the exact `(M, tM, hit)`
   fed to it by `measure_loss` (avoids hand-replicating the flow actor's stochastic
   bootstrap — M/tM are provably unchanged by `measure_delta`, only `hit` moves), then
   recomputes the loss with a from-scratch numpy formula and checks atol 1e-5, for both
   tpd values. Also carries the spec §4 grad-hygiene assertion:
   `jax.grad(actor_loss)` w.r.t. `params["modules_critic"]` is exactly zero on a delta
   config.
3. `test_hit_c_reads_observations` — pure-exit row (obs inside, nobs outside) must
   yield -1 and `measure_delta_event_rate > 0`; a batch dwelling far outside every
   column ball yields `measure_delta_event_rate == 0`.
4. `test_identity_tabular` — pure numpy (no jax subprocess), 3-state deterministic
   chain, `J = (1-gamma)*M - h(x0)` to 1e-6 for gamma in {0.9, 0.99} via power
   iteration of both Bellman equations.
5. `test_bounded_target` (+ `test_bounded_target_no_clip_in_source`) — captured hit
   values are a subset of {-1, 0, 1}; the delta code block (sliced from source between
   the `measure_delta` and `measure_nhat_head` markers) contains no `clip`.
6. `test_info_keys` — the two info keys present iff `measure_delta=True`; absent on
   both the delta-off ball-visitation path and the plain absorbing first-passage path.
7. `test_assert_lattice` — all 7 refusals + `measure_delta_partition` refusal, plus 3
   accepts (base, hindsight, stopgrad_psi, warm-start-shaped empty string).

**Verification that the tests are load-bearing, not vacuous**: I temporarily sabotaged
gciql_flow.py:1532 (`hit = hit_n - hit_c` → `hit = hit_n + hit_c`) and reran the suite;
3 of 8 tests failed as expected (`test_delta_target_matches_numpy_reference`,
`test_hit_c_reads_observations`, `test_bounded_target`), then reverted the sabotage and
confirmed all 8 pass again. Transcript below is the clean (post-revert) run.

## 3. Configs (`configs/gciql/*.yaml`)

Each is a byte-equal copy of its named base (spec 17a §5 table) except the deltas the
spec names. All 5 were sanity-checked by merging `get_config()` + `agent_overrides` and
calling `GCIQLFlowAgent.create(...)` directly (mirrors what `train.py` does at launch)
— all 5 compose without assertion errors:

- `cube_double_state_measure_delta_d50.yaml` — base
  `cube_double_state_measure_ballvisit_d50.yaml` + `measure_delta: true`,
  `measure_fp_absorbing: false` (already true in base, kept explicit),
  `run_group: GCIQL-CubeDouble-Flow-Measure-Delta-d50`, `eval_interval: 10000`.
- `cube_double_state_measure_delta_hs_d50.yaml` — base
  `cube_double_state_measure_ballvisit_hs_d50.yaml` + same deltas, keeps
  `measure_testpoint_dist: hindsight`,
  `run_group: GCIQL-CubeDouble-Flow-Measure-Delta-Hindsight-d50`.
- `scene_state_measure_delta_d50.yaml` — base `scene_state_measure_ballvisit_d50.yaml`
  + analogous deltas, hit_spec unchanged (5-way scene region),
  `run_group: GCIQL-Scene-Flow-Measure-Delta-d50`.
- `scene_state_measure_delta_hs_d50.yaml` — base
  `scene_state_measure_ballvisit_hs_d50.yaml` + analogous deltas,
  `run_group: GCIQL-Scene-Flow-Measure-Delta-Hindsight-d50`.
- `cube_single_state_measure_delta_d50.yaml` — base
  `cube_single_state_measure_ballvisit_d50.yaml` + `measure_delta: true`,
  `measure_stopgrad_psi: false` (**overridden from the base's `true`** — the ⚠ warning
  in spec §5 is real: the base is the phase-2 FrozenPsi recipe and this arm trains
  from scratch with no warm start), `eval_interval` left at 50000 (unchanged — this is
  not a phase-2 warm-start arm, no 0→10k shock window to instrument),
  `run_group: GCIQL-CubeSingle-Flow-Measure-Delta-d50` (new naming convention —
  matches the `GCIQL-CubeSingle-...` prefix already used by the concurrent 17b config
  `cube_single_state_measure_delta_partition_d50.yaml`, rather than the legacy
  `GCIQL-State-...` prefix on the older cube-single ballvisit config).

No `run_group` string was given verbatim by the spec for the cs config; I constructed
one following the repo's "group names lie — never reuse" discipline and the naming
convention already established by the sibling 17b config in the same directory.

## 4. Verification output

### `tests/test_gciql_measure_delta.py`
```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/adityamohan/git/Factored-FB
configfile: pyproject.toml
plugins: hydra-core-1.3.2, anyio-4.14.2
collected 8 items

tests/test_gciql_measure_delta.py ........                               [100%]

========================= 8 passed in 78.81s (0:01:18) =========================
```

### Regression neighborhood
```
$ .venv/bin/python -m pytest tests/test_gciql_measure_nstep.py tests/test_gciql_measure_hit_spec.py tests/test_gciql_measure_axes.py tests/test_gciql_measure_scale.py tests/test_gciql_measure_zmix.py -q
============================= test session starts ==============================
collected 26 items

tests/test_gciql_measure_nstep.py ......                                 [ 23%]
tests/test_gciql_measure_hit_spec.py ......                              [ 46%]
tests/test_gciql_measure_axes.py ......                                  [ 69%]
tests/test_gciql_measure_scale.py .....                                  [ 88%]
tests/test_gciql_measure_zmix.py ...                                     [100%]

======================== 26 passed in 280.17s (0:04:40) ========================
```
All green pre- and post-change; the diff could not plausibly have caused any of these
(the only touch point in a shared code path is the sentinel-init line and the
info-append line, both no-ops when `measure_delta` is absent/False, and both are
outside any of these suites' exercised branches except through the unconditional
`measure_loss` call, which the bit-identical tests in those suites already re-verify).

### Config sanity (not a pytest suite; ad hoc script mirroring train.py's merge)
```
OK   configs/gciql/cube_double_state_measure_delta_d50.yaml  run_group=GCIQL-CubeDouble-Flow-Measure-Delta-d50
OK   configs/gciql/cube_double_state_measure_delta_hs_d50.yaml  run_group=GCIQL-CubeDouble-Flow-Measure-Delta-Hindsight-d50
OK   configs/gciql/scene_state_measure_delta_d50.yaml  run_group=GCIQL-Scene-Flow-Measure-Delta-d50
OK   configs/gciql/scene_state_measure_delta_hs_d50.yaml  run_group=GCIQL-Scene-Flow-Measure-Delta-Hindsight-d50
OK   configs/gciql/cube_single_state_measure_delta_d50.yaml  run_group=GCIQL-CubeSingle-Flow-Measure-Delta-d50
```

## 5. Deviations from the spec, and why

1. **Split test 5 into two functions.** Spec §3 test 5 (`test_bounded_target`) asks for
   both a numeric bound check and a "no `jnp.clip` in the path" source check. I kept
   the numeric part in `test_bounded_target` and moved the source-grep into
   `test_bounded_target_no_clip_in_source` (outside the jax subprocess — it's a plain
   file read). This gives 8 test functions instead of the spec's 7-named list, but
   covers every assertion the spec lists; nothing was weakened or dropped.
2. **Test 2's off-by-one geometry.** For `tpd='batch'`, `col_states = next_observations`
   itself, so the diagonal of the hit matrix (`hit_n[i,i]`) is always 1 by construction
   (self-distance 0) — there is no "outside→outside" or "exit" transition available on
   the diagonal. I therefore planted the four transition types as OFF-diagonal entries
   against a fixed goal column (row 0 doubling as both the column's own test point and
   a trivial dwell case), and verified them via `hit_ref[i, 0]` for i=1..4. This matches
   the spec's intent ("realizes all four transition types against the batch's own nobs
   columns") but the specific row/column indices were my choice since the spec doesn't
   pin them.
3. **`capture_terms` monkeypatch instead of hand-replicating the network forward pass.**
   The spec says to "rebuild M and tM exactly as `measure_loss_of` does (call the critic
   directly for phi/psi)". I instead monkeypatched the static method `_measure_terms`
   to capture its actual call arguments from inside a real `measure_loss` invocation.
   This is provably equivalent (M/tM/gamma/scale_norm are computed identically whether
   `measure_delta` is on or off — the delta code only rewrites `hit` after M/tM already
   exist) and avoids a second, error-prone hand-copy of the flow actor's stochastic
   bootstrap sampling (which would need to reproduce the exact rng-consumption order to
   be bit-exact). I verified this reasoning is sound by reading the touch-point code
   path in full (gciql_flow.py:1412-1655) before choosing this approach.
4. **cube-single `run_group` string** — not given verbatim by the spec; constructed as
   described in section 3 above.

Nothing in the spec's 7 named tests or the §4 grad-hygiene assertion was weakened,
skipped, or altered in meaning.

## 6. What I could not independently verify

- I did not launch any training (out of scope, per instructions) and no checkpoints
  exist yet, so I could not do an end-to-end `update()` smoke test against a *warm-started*
  checkpoint for the cd/sc configs (only construction-time composition + measure_loss
  correctness were verified, which is what the spec asks for at this stage).
- I did not verify `warm_start_ckpt` splicing itself (spec explicitly puts the
  `--delta` L1fix/L2 harness conversion and any launching out of scope for 17a).
