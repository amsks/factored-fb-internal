# Spec 18 task-agent report

Implemented §4 (agent code), §5 (tests), and the two configs of §7 for
`measure_density` (continuous normalized successor density).
Spec: `PAPER/Multistep-FB/specs/18-density-measure.md`.

## Files changed

- `/home/adityamohan/git/Factored-FB/tools/wandb_mode_shim/gciql_flow.py`
- `/home/adityamohan/git/Factored-FB/tests/test_gciql_measure_density.py` (new)
- `/home/adityamohan/git/Factored-FB/configs/gciql/cube_single_state_measure_density_d50.yaml` (new)
- `/home/adityamohan/git/Factored-FB/configs/gciql/cube_single_state_measure_density_hard_d50.yaml` (new)

## Touch points in `gciql_flow.py`

| Spec section | What | Location (post-edit) |
|---|---|---|
| §4.1 get_config | 4 knobs `measure_density`, `measure_density_dim`, `measure_density_temp`, `measure_density_hard` | inserted after `config.measure_delta_coef = 1.0` (~line 3616-3626) |
| §4.2 query helper | `_density_queries(nobs, blocks)` staticmethod | inserted after `_delta_phi` return (~line 1342-1350) |
| §4.2 (implicit) | `_raw_block_cols(obs_rows, blocks)` staticmethod — the RAW twin used as `_delta_phi`'s `protos` arg, so the kernel path never double-scales | inserted immediately after `_density_queries` (~line 1352-1359) |
| §4.3 loss | `density_loss(self, batch, grad_params, rng)` | inserted after `delta_partition_loss`, before `centered_measure_loss` (~line 1858-1920) |
| §4.4 total_loss wiring | `measure_only` branch: `elif measure_density: density_loss` instead of `measure_loss`; loss = `m_loss + actor_loss` (NOT gated by `measure_coef`), combined with 17b's existing ungated branch via an `or` | ~line 2245-2263 |
| §4.5 actor q-term | `elif measure_density:` branch before the stock `fb_forward_z`/else fallback, both towers called at frozen params (no `params=grad_params`) | ~line 1054-1069 |
| §4.6 registration | `density_f`/`density_b`/`target_density_f`/`target_density_b` modules, registered only when `measure_density=True` | in `create()`, right after the 17b `delta_head` registration block (~line 3290-3307) |
| §4.6 target init mirror | `params['modules_target_density_f'] = params['modules_density_f']`; same for `_b` | right after the 17b `target_delta_head` mirror (~line 3482-3484) |
| §4.6 assert-lattice | mutual-exclusion asserts added to BOTH 17a's and 17b's existing blocks (`assert not config.get('measure_density', False)`), plus the full `measure_density` lattice block (transcribed verbatim from spec §4.6) | 17a mutex ~line 2899; 17b mutex ~line 2942; density block ~line 2945-2966 |
| §4.7 update override | extended the existing 17b `update()` override with `if measure_density: target_update('density_f'); target_update('density_b')` | ~line 2402-2421 |

## Lattice mapping (measure_density block, as shipped)

```
assert measure_only
assert not measure_delta and not measure_delta_partition
assert not measure_first_passage
assert actor_kind=='flow' and actor_cond=='goal'
assert actor_q_center
assert encoder is None
assert measure_density_temp > 0.0
assert measure_density_dim >= 2
assert measure_ortho_coef == 0.0
assert measure_coef <= 0.0                      # 17b lesson, transcribed
for bad in (fb_forward_z, sf_feature_td, measure_expectile, measure_nhat_head,
            actor_z_infer, fwd_two_tower, isfb, freeze_critic_ckpt, warm_start_ckpt):
    assert not bad
assert measure_nstep == 0 and not measure_eps_multiscale
assert centered_measure=='off' and task_prior=='dirac' and task_prior_actor=='dirac'
assert not measure_delta_protos            # density has no prototype file
```

Plus, symmetrically, `measure_delta` (17a) and `measure_delta_partition` (17b) now
each assert `not measure_density` so all three knobs are pairwise mutually exclusive.

## Deviations from the spec's literal code (and why)

1. Added `_raw_block_cols` as a small named staticmethod instead of inlining
   `<concatenated raw block columns of nobs>` three times (density_loss,
   actor_loss q-term, create()'s registration comment references it implicitly
   via `_density_queries` for `ex_b`). Purely a de-duplication; behavior is
   exactly the spec's "plain block slice (no scaling)".
2. `density_loss` additionally computes `w = jax.nn.softmax(logits, -1)` (not in
   the spec's code block, which only builds `logp`) purely for the info-key
   diagnostics (`density_ess`, `density_w_entropy`). This does not touch the CE
   gradient path — `ce` is built from `logp` exactly as specified.
3. `total_loss`'s ungate condition was written as
   `measure_delta_partition or measure_density` (single shared branch) rather
   than a third parallel `if`/`elif`, to avoid duplicating the CRITICAL-1
   comment block; behavior is identical to a dedicated branch since the two
   knobs are mutually exclusive by the lattice.
4. No new pytree/data field was added anywhere (spec §4.1's "Nothing static
   beyond config" honored) — verified structurally: `density_f`/`density_b`
   and their targets are ordinary registered network modules, not agent
   dataclass fields.

## Test results

```
$ JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gciql_measure_density.py -x -q
10 passed in 118.18s

$ JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gciql_measure_delta.py tests/test_gciql_measure_delta_partition.py -q
20 passed in 203.69s

$ JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gciql_measure_nstep.py tests/test_gciql_measure_hit_spec.py tests/test_gciql_measure_axes.py tests/test_gciql_measure_scale.py tests/test_gciql_measure_zmix.py -q
26 passed in 279.95s
```

Total: 56/56 green, no regressions in the 17a/17b siblings or the measure-axis
neighborhood.

## Config construction check (item 4)

Built agents from both shipped configs (with `encoder=None`/`frame_stack=None`
CLI-time overrides applied, mirroring how `train.py` builds the config) under
`/dev/shm/.venv-jax`, CPU-only:

```
configs/gciql/cube_single_state_measure_density_d50.yaml       OK  run_group=GCIQL-CubeSingle-Flow-Measure-Density-d50
  registered: modules_density_b, modules_density_f, modules_target_density_b, modules_target_density_f
configs/gciql/cube_single_state_measure_density_hard_d50.yaml  OK  run_group=GCIQL-CubeSingle-Flow-Measure-DensityHard-d50
  registered: modules_density_b, modules_density_f, modules_target_density_b, modules_target_density_f
```

Neither config trips the lattice. Both drop `measure_delta_protos` (density has
no prototype file — the lattice refuses it) and drop `measure_coef` (left at its
0.0 default — the 17b lesson, so nothing silently zeroes the CE).

## Concerns / open items (not this agent's scope)

- §6 (`scripts/rescuing_fb/eval_density_zeroshot.py`) and §7's wave launch are
  explicitly out of scope and untouched.
- The two configs use `value_arch: bilinear`, `value_latent_dim: 50` as dead
  weight (17b precedent — the lattice doesn't require this, but keeps the
  critic/value modules' shapes consistent with the sibling cells for any
  shared tooling that inspects them).
- `measure_density_hard` config additionally sets `measure_density_hard: true`;
  everything else matches the PRIMARY config's knob set exactly, satisfying
  the "run-group names lie, audit by config" standing rule.

---

## Round-2 review addendum (opus review + coordinator follow-up)

### 1. BLOCKER — G2 gate instrument (fixed)

`density_loss` (`tools/wandb_mode_shim/gciql_flow.py:1918-1938`) now computes
`ess_rows = 1/sum(w**2, axis=-1)` (shape `[2,B]`) once and logs four stats:
`density_ess` (mean, kept for continuity), `density_ess_median` (the actual
G2 gate statistic per spec §7), and `density_ess_h0`/`density_ess_h1`
(per-head means) so a one-head collapse is visible even though double_q's
min at actor/eval means the sick head is what the policy/eval actually feel.
Covered by `test_ce_matches_numpy_reference` (numpy reference for all four
stats against `density_loss`'s own info dict, atol 1e-5, plus a
mean-vs-median-must-differ guard against a copy-paste that ships `jnp.mean`
under the median key) and `test_end_to_end_updates` (all four finite and in
`[1,M]` for 30 update steps).

### 2. MEDIUM — lattice gap (fixed)

Added two refusals to the `measure_density` lattice
(`gciql_flow.py:2988-2995`): `value_arch != 'dueling'` and
`not task_prior_critic_mix`. Note for the record: `value_arch='dueling'` is
ALSO already refused transitively by the pre-existing generic dueling lattice
block (`gciql_flow.py:2766-2788`, `assert not config.get('measure_only',
False)`, and `measure_density` requires `measure_only=True`) — so this
specific composition was not actually reachable before the fix either. The
explicit assert is added anyway for a clear spec-18-specific error message
and as defense-in-depth against a future refactor of that generic block.
`task_prior_critic_mix` is a live gap closed here: `total_loss`'s
`measure_only` branch runs it unconditionally on top of whichever measure
loss (`gciql_flow.py:~2330`), and does not require `task_prior_actor='mix'`
to be reachable in a way `measure_density`'s own lattice would otherwise
catch. Both covered by `test_assert_lattice`.

### 3. LOW — three tests strengthened

- **Target/actor grad tautology.** Confirmed the reviewer's finding
  mathematically: `TrainState.__call__`'s own contract
  (`third_party/ogbench/impls/utils/flax_utils.py:90-97`, "the default
  behavior is to stop the gradients... you need to explicitly provide the
  parameters to flow the gradients") means `density_loss`'s
  `target_density_f`/`target_density_b` calls (no `params=grad_params`)
  already sever those params from `grad_params` **structurally**, before
  `stop_gradient(t)` ever runs. `t` has zero live dependency on `grad_params`
  regardless of that wrapper, so no test that calls the real `density_loss`
  can be sensitive to deleting it — verified empirically below. Added
  `test_target_stopgrad_is_load_bearing`: a self-contained reconstruction of
  `t` that DELIBERATELY calls the target towers WITH `params=grad_params`
  (the live path the wrapper is defense-in-depth against), and shows grad
  w.r.t. target params is exactly 0 with the wrapper and genuinely nonzero
  without it — a path that IS sensitive to the invariant, even though the
  production call site currently is not (this is a repo-wide convention, not
  a spec-18-specific gap — 17b's `delta_partition_loss` has the identical
  structural property around `p_tgt`).
- **`test_hard_immediate_ablation` tail.** Replaced the `eye(B)==eye(B).T`
  tautology with a hard-vs-soft cross-check at matched params/batch/rng:
  asserts `density_k_self` is 1.0 under hard, well under 0.5 under soft (on a
  batch with the hit-spec block tightened so the soft kernel actually spreads
  mass instead of degenerating to near-one-hot — the default N(0,1) spread
  gave k_self≈0.90, too weak a contrast), and CE/`k` matrices differ.
- **`test_default_off_bit_identical`.** Now loads `git show
  HEAD:tools/wandb_mode_shim/gciql_flow.py` (verified to contain zero
  `measure_density` occurrences — this repo's actual pre-spec-18 baseline,
  since nothing has been committed) as a separate module and diffs the
  CURRENT knob-off agent's param tree and `total_loss` against THAT one, in
  addition to the original same-code comparison (kept as a cheap redundant
  signal, explicitly labeled non-load-bearing in the docstring).

### 4. LOW — stale K-sweep provenance (fixed)

Corrected `--k 256` → `--k {64,512,1024}` and the header's `K=256` → the
file's actual K in all three
`cube_single_state_measure_delta_partition_K{64,512,1024}_d50.yaml` files
(comment-only; `agent_overrides`/`run_group` were already correct).

### Verification

```
$ JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gciql_measure_density.py -q
11 passed in 138.71s   (10 spec tests + test_target_stopgrad_is_load_bearing)

$ JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gciql_measure_delta.py tests/test_gciql_measure_delta_partition.py -q
20 passed in 203.95s
```

**Item 3's requested stop_gradient experiment — result and an honest caveat.**
Ran it literally: temporarily deleted `jax.lax.stop_gradient(...)` around `t`
in the real `density_loss` (production code), ran the full suite, restored
it, ran again.

```
baseline (sg present):        11 passed in 138.16s
sg deleted (production line): 11 passed in 138.71s   <- suite STAYS GREEN
sg restored:                  11 passed in 138.71s
```

The suite does **not** flip red — and per the structural argument above, no
test that calls the real `density_loss` method can, because `t` never
touches `grad_params` regardless of that specific wrapper (the severing
happens earlier, via the omitted `params=` kwarg on the target-tower calls).
This matches the reviewer's original finding and is not a residual gap: it
is a correct, if redundant, defensive line, exactly mirroring 17b's
`p_tgt = jax.lax.stop_gradient(...)`. What IS now verified, and would catch
a real regression (e.g. a future refactor that starts passing
`params=grad_params` into a target-tower call), is
`test_target_stopgrad_is_load_bearing`'s reconstruction, which flips red
without the wrapper and green with it (shown in its own definition/assert).
Flagging this explicitly rather than reporting a "test now fails" result
that would not be true of the actual production line.

### Files touched (this addendum)

- `tools/wandb_mode_shim/gciql_flow.py` — G2 instruments (§1), lattice gap (§2)
- `tests/test_gciql_measure_density.py` — new test + two strengthened tests (§3)
- `configs/gciql/cube_single_state_measure_delta_partition_K{64,512,1024}_d50.yaml` — comment fix (§4)
