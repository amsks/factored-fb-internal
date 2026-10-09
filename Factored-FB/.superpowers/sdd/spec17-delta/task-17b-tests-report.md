# Task 17b-tests report — `measure_delta_partition` test-first contract

Spec: `PAPER/Multistep-FB/specs/17b-delta-partition-impl.md` (self-contained; science
context `PAPER/Multistep-FB/specs/17-delta-measure.md` §1 variant B). Scope: the
STANDALONE half only — the prototype script, the test file (written test-first
against the not-yet-implemented agent-side contract), and the launch config. Did
**not** touch `tools/wandb_mode_shim/gciql_flow.py` (owned by the parallel
"17b-agent" wave). No commits/staging/git-history changes; tree left dirty per
instructions.

## Files created

- `scripts/rescuing_fb/make_delta_prototypes.py` (new)
- `tests/test_gciql_measure_delta_partition.py` (new, 10 tests)
- `configs/gciql/cube_single_state_measure_delta_partition_d50.yaml` (new)

## 1. `scripts/rescuing_fb/make_delta_prototypes.py` (spec §1.1)

CLI: `--dataset <npz> --hit_spec '<start:stop:eps,...>' --k 256 --seed 0 --out <npy>`.

- Reimplements the `_parse_hit_spec` grammar locally rather than importing it from
  `tools/wandb_mode_shim/gciql_flow.py`: that module transitively imports jax/flax
  (only present in `/dev/shm/.venv-jax`), while this script must run in the plain
  repo `.venv` at launch time before training starts. The grammar is copied verbatim
  (comma-separated `start:stop:eps`, same asserts) and the docstring says so, per the
  spec's explicit fallback instruction.
- `extract_block_columns`: concatenates the hit-spec block columns of `observations`
  in block order → `[N, d_c]`.
- `draw_prototypes`: K **distinct** row indices via `np.random.default_rng(seed).choice(n,
  size=k, replace=False)` — coverage over dataset-visited configurations, not
  clustering. Every returned proto row is literally one of the dataset's own rows.
- `sanity_stats`: K, d_c, per-dim min/max, mean nearest-neighbor distance between
  prototypes (pairwise L2, diagonal excluded via `np.fill_diagonal(..., inf)`).
- Writes float32 `[K, d_c]` via `np.save`; creates parent dirs. Prototypes stored in
  RAW obs units (no ε scaling — that happens at use, inside `_delta_phi`).

## 2. `tests/test_gciql_measure_delta_partition.py` (spec §5, all 10 tests)

Same harness as `tests/test_gciql_measure_nstep.py` (`tests.jax_subprocess.run_jax`,
a `COMMON` f-string preamble, `OBS_D=40`, action dim 5, `d=8`, `B=16`). Base config:
`measure_only=True, measure_first_passage=False, actor_q_center=True,
measure_delta_partition=True, measure_delta_protos=<tmp .npy written inside the
subprocess body>, measure_hit_spec='19:22:0.4', K=8 toy protos` — matches spec §5
preamble exactly. `_parse_hit_spec` is imported directly from `gciql_flow` (already
exists, untouched).

Tests (names verbatim from spec §5):

1. `test_default_off_bit_identical` — knob off (unset vs. explicit `False`): identical
   param-tree leaf count/values, no `modules_delta_head`/`modules_target_delta_head`
   keys, `total_loss` bit-identical.
2. `test_phi_simplex_and_scaling` — `_delta_phi` rows sum to 1, ≥0; argmax = nearest
   prototype on 5 constructed on-prototype points + 1 off-manifold point; **and**, for
   a 2-block spec `'19:22:0.4,28:31:0.8'`, compares `_delta_phi`'s output against a
   from-scratch numpy reference (`phi_np`, divides each block by its own ε before
   squaring) on points offset by exactly `eps_1=0.4` in block 1 vs. `eps_2=0.8` in
   block 2 — an unscaled reference would disagree, so this catches a missing/incorrect
   per-block ε division.
3. `test_ce_matches_numpy_reference` — rebuilds `p`, `p̄`, `φ(x')` via **direct module
   calls** at the agent's own params (bootstrap noise drawn from the *same* rng key
   passed to `delta_partition_loss`, since the spec's code draws
   `noise = jax.random.normal(rng, acts.shape)` directly off the passed-in key with no
   split — reproducible bit-exactly), rebuilds `p_tgt`/CE in numpy/jax, compares to
   `delta_partition_loss`'s returned loss at atol 1e-5. Also asserts `p_tgt` rows sum
   to 1 (pins the (1−γ)/γ convex-combination identity), and — as an explicit
   regression guard for the NEXT-state counting convention (spec §3) — asserts that
   substituting `φ(x)` (current state) for `φ(x')` does **not** reproduce the same
   loss.
4. `test_target_head_init_and_soft_update` — at create, `target_delta_head` params ==
   `delta_head` params elementwise; after one `update()`, new target ==
   `tau*online + (1-tau)*old_target` elementwise (the repo's `target_update` formula,
   `agents/gciql.py:147`). Catches both a missing `update()` override (frozen target
   forever) and a missing init mirror.
5. `test_measure_only_wiring` — `total_loss` info contains a key containing
   `'delta_ce'` (checked by substring since the wiring pattern in this codebase
   prefixes with `'measure/'`, e.g. `info[f'measure/{k}']`, exactly mirroring how
   `measure_loss`'s own `m_info` is wired at `total_loss:2089-2090`) and **no** key
   containing `'measure_loss'`; no `value/`- or `critic/`-prefixed keys (measure_only
   semantics: IQL value/critic losses stay dropped).
6. `test_actor_grad_hygiene` — `jax.grad(actor_loss)`: grad-norm over
   `modules_delta_head` + `modules_target_delta_head` + `modules_critic` < 1e-9; over
   `modules_actor` > 1e-8. `jax.grad(delta_partition_loss)`: grad over `modules_actor`
   < 1e-9 (bootstrap stop-gradded); over `modules_delta_head` > 1e-8; over
   `modules_target_delta_head` < 1e-9 (no gradient path into the target — it is
   soft-updated outside the loss).
7. `test_actor_q_reads_goal_cell` — goals constructed sitting exactly on prototype
   `j*=2`: `z_g=φ(goal)` peaked at `j*`. Reconstructs the actor's own sampled
   `actor_actions` deterministically (`rng, x0_rng, t_rng, n_rng, z_rng =
   jax.random.split(rng, 5); noise = jax.random.normal(n_rng, ...)` — this is
   *pre-existing, unmodified* code in `actor_loss` per spec §2.6, which only inserts a
   branch **before** the final `q1, q2 = ...` line), calls `delta_head` directly with
   those exact actions at frozen params, and checks the hand-computed
   `mean(min(q1,q2))` matches `actor_loss`'s own `info['q_mean']` at atol 1e-4.
8. `test_assert_lattice` — refusals: `measure_only=False`; `measure_delta=True` (17a);
   `measure_first_passage=True`; `measure_delta_protos=''`; a wrong-width protos file
   (`[8,5]` against the base's `d_c=3`); `actor_q_center=False`; a `warm_start_ckpt`
   set; `measure_ortho_coef=1.0`. Accepts the base config (`not refuses()`).
9. `test_prototype_script` — subprocess-runs `make_delta_prototypes.py` (via
   `sys.executable`, i.e. the plain repo `.venv` — **no jax subprocess**, since the
   script itself must not need jax) on a synthetic 200-row npz: output shape
   `[16, 3]`, `float32`, every prototype row found among the dataset's own block rows.
10. `test_end_to_end_updates` — 30 `update()` steps on synthetic data: every step's
    info dict must contain a `'delta_ce'`-substring key with a finite value (and any
    `delta_p_entropy`/`delta_tgt_entropy` keys present must be finite too); no NaN in
    any param leaf after 30 steps; `p` (softmax of the trained `delta_head` on a fresh
    batch) stays on the simplex.

### Conventions pinned (spec §3)

- Simplex (Σ=1, ≥0): tests 2, 3, 10.
- NEXT-state counting convention (`p_tgt` built from `φ(x')`, not `φ(x)`): test 3's
  explicit off-by-one regression assertion.
- Per-block ε scaling: test 2's 2-block numpy-reference comparison.
- Anti-scope (spec §4 — no ortho/mass-anchor/scale_norm/hindsight/warm-start in this
  path): test 8 refuses `measure_ortho_coef>0` and `warm_start_ckpt`; no test asserts
  for any such machinery being present.

### A bug I found and fixed in my own harness (not a gciql_flow issue)

`COMMON` is an f-string (needs `{OBS_D}`/`{D}`/`{K}` substituted), so its own literal
braces are escaped `{{...}}`. Individual test **bodies** passed to `_run(...)` are
plain (non-f) strings — I initially copy-pasted the `{{...}}` escaping into them too,
which Python parses as a *set containing a dict comprehension* (`{ {...} }`), raising
`TypeError: unhashable type: 'dict'`. Fixed by de-escaping to single braces in the 6
body strings that needed a dict comprehension (`tests/test_gciql_measure_delta_partition.py`
lines that build `{k: jnp.asarray(v) for k, v in ...}`); the two occurrences that
legitimately live inside `COMMON` were left double-braced.

## 3. `configs/gciql/cube_single_state_measure_delta_partition_d50.yaml` (spec §6)

Copied the `cube_single_state_measure_ballvisit_d50.yaml` scaffolding (env, steps
500000, log/eval/save intervals, storage, eval_episodes 50). `agent_overrides` set to
exactly the spec's list: `bc_coeff: 3.0`, `value_arch: bilinear`,
`value_latent_dim: 50`, `measure_only: true`, `measure_delta_partition: true`,
`measure_hit_spec: "19:22:0.4"`, `actor_q_center: true`,
`measure_delta_protos: /home/adityamohan/git/Factored-FB/results/delta_protos/cube_single_K256_s0.npy`.
Deliberately **dropped** `measure_coef`, `measure_legs`, `measure_scale_norm`,
`measure_first_passage`, `measure_fp_absorbing`, `measure_stopgrad_psi`,
`measure_ortho_coef` (none touch the 17b path; the lattice refuses `fp` and the rest
would just record misleading provenance). `run_group:
GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50`. No `warm_start_ckpt` (lattice
refuses it — from-scratch cell). Not launched.

## 4. Verification

### 4.1 `-k prototype_script` (must be green — tests only this repo's own script)

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py -q -k prototype_script
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/adityamohan/git/Factored-FB
configfile: pyproject.toml
collected 10 items / 9 deselected / 1 selected

tests/test_gciql_measure_delta_partition.py .                            [100%]

======================= 1 passed, 9 deselected in 0.26s ========================
```

### 4.2 Full suite

```
$ .venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py -q
tests/test_gciql_measure_delta_partition.py .FFFFFFF.F                   [100%]
...
FAILED tests/test_gciql_measure_delta_partition.py::test_phi_simplex_and_scaling
FAILED tests/test_gciql_measure_delta_partition.py::test_ce_matches_numpy_reference
FAILED tests/test_gciql_measure_delta_partition.py::test_target_head_init_and_soft_update
FAILED tests/test_gciql_measure_delta_partition.py::test_measure_only_wiring
FAILED tests/test_gciql_measure_delta_partition.py::test_actor_grad_hygiene
FAILED tests/test_gciql_measure_delta_partition.py::test_actor_q_reads_goal_cell
FAILED tests/test_gciql_measure_delta_partition.py::test_assert_lattice
FAILED tests/test_gciql_measure_delta_partition.py::test_end_to_end_updates
==================== 8 failed, 2 passed in 83.34s (0:01:23) ====================
```

**Green (2/10):** `test_default_off_bit_identical`, `test_prototype_script`. Test 1 is
correctly green even pre-implementation: the "off" path reads no new config keys, so
it is a no-op by construction today, and the implementer must keep it that way (this
is a genuine regression contract, not a false pass).

**Red (8/10), with confirmed root cause per test (from the actual subprocess
stderr, not guessed):**

| test | exception | missing symbol |
|---|---|---|
| `test_phi_simplex_and_scaling` | `AttributeError: type object 'GCIQLFlowAgent' has no attribute '_delta_phi'` | `_delta_phi` staticmethod (§2.3) |
| `test_ce_matches_numpy_reference` | `AttributeError: 'GCIQLFlowAgent' object has no attribute 'delta_partition_loss'` | `delta_partition_loss` method (§2.4) |
| `test_target_head_init_and_soft_update` | `KeyError: 'modules_delta_head'` | module registration in `create()` (§2.7) |
| `test_measure_only_wiring` | `AssertionError` — info keys are `['actor/...', 'measure/measure_diag', 'measure/measure_loss', ...]`, no `delta_ce` | `total_loss` wiring (§2.5) |
| `test_actor_grad_hygiene` | `KeyError: 'modules_delta_head'` | module registration (§2.7) |
| `test_actor_q_reads_goal_cell` | `AttributeError: 'GCIQLFlowAgent' object has no attribute 'delta_protos'` | `delta_protos` dataclass field (§2.2) |
| `test_assert_lattice` | plain `AssertionError` (the `refuses(...)` checks return `False` — no assertions exist yet to raise) | assert-lattice block (§2.7) |
| `test_end_to_end_updates` | `AssertionError: (0, [...no delta_ce key...])` | `total_loss` wiring (§2.5) |

Every failure traces to a symbol/wiring point the spec assigns to the 17b-agent wave
in `gciql_flow.py`; none is a bug in the test itself. (Confirmed by directly reading
each subprocess's `stderr`, not inferred.)

### 4.3 Manual script exercise (synthetic npz, scratchpad only)

```
$ .venv/bin/python - <<'EOF'
import numpy as np
rng = np.random.default_rng(0)
obs = rng.normal(size=(200, 40)).astype(np.float32)
np.savez(".../synthetic_ds.npz", observations=obs)
EOF
$ .venv/bin/python scripts/rescuing_fb/make_delta_prototypes.py \
    --dataset .../synthetic_ds.npz --hit_spec '19:22:0.4' --k 16 --seed 0 \
    --out .../protos_manual.npy

[make_delta_prototypes] hit_spec='19:22:0.4' blocks=((19, 22, 0.4),)
[make_delta_prototypes] dataset=.../synthetic_ds.npz N=200 K=16 d_c=3 seed=0
[make_delta_prototypes] per-dim min: [-1.0713 -2.1866 -1.343 ]
[make_delta_prototypes] per-dim max: [1.8632 1.9539 2.0067]
[make_delta_prototypes] mean nearest-neighbor distance: 1.037742
[make_delta_prototypes] wrote .../protos_manual.npy shape=(16, 3) dtype=float32
```

No real dataset was used anywhere; both the pytest harness and this manual exercise
use synthetic Gaussian data written to `/tmp/.../scratchpad`.

## CONTRACT FOR THE IMPLEMENTER

Everything below is what the tests in `tests/test_gciql_measure_delta_partition.py`
require to exist in `tools/wandb_mode_shim/gciql_flow.py`, `class GCIQLFlowAgent`.
Spec section numbers refer to `17b-delta-partition-impl.md`.

**Config knobs (`get_config()`, §2.1):**
- `config.measure_delta_partition = False` (bool)
- `config.measure_delta_protos = ''` (str, `.npy` path)
- `config.measure_delta_coef = 1.0` (float)

**Dataclass field (§2.2):** `GCIQLFlowAgent.delta_protos: Any = None` — `[K, d_c]`
jnp float32 array when the knob is on, else `None`.

**Staticmethod (§2.3):**
```python
@staticmethod
def _delta_phi(obs_rows, protos, blocks):
```
`obs_rows: [B, obs_dim]`, `protos: [K, d_c]` (raw obs units, columns = the
`_parse_hit_spec`-ordered block concatenation), `blocks` = `_parse_hit_spec(...)`
output. Returns `[B, K]`, softmax over K of
`-0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2`. **Must** divide each block by
its own ε before squaring (test 2 catches an unscaled version). Rows sum to 1, all
entries ≥ 0 (test 2, 3, 10).

**Method (§2.4):**
```python
def delta_partition_loss(self, batch, grad_params, rng):
```
Reads `batch['observations'/'next_observations'/'actions'/'value_goals']`. Computes
online `p = softmax(delta_head([obs,acts,g], params=grad_params))`; bootstrap
`noise = jax.random.normal(rng, acts.shape)` (no split — the rng passed in is used
directly, test 3 relies on this exact behavior to reproduce it bit-exactly);
`a_plus = sg[actor(nobs, g, noise)]`; target `p_bar = softmax(target_delta_head([nobs,
a_plus, g]))` (no `params=` override — reads the network's own live params);
`phi_n = _delta_phi(nobs, self.delta_protos, blocks)` where
`blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))`;
`p_tgt = sg[(1-gamma)*phi_n[None] + gamma*p_bar]`; `ce = -mean(sum(p_tgt *
log(p+1e-8), -1))`; returns `(measure_delta_coef * ce, info)`. `info` must contain
(exact key names, tests use substring match so exact prefixing elsewhere is fine):
`delta_ce`, `delta_p_entropy`, `delta_tgt_entropy`, `delta_p_max`, `delta_cell_change`.

**`total_loss` wiring (§2.5):** inside the existing `if
self.config.get('measure_only', False):` branch, when `measure_delta_partition` is
on, call `delta_partition_loss` **instead of** `measure_loss`, prefixed exactly like
the stock `m_info` loop (`info[f'measure/{k}'] = v`) — test 5/10 check for a
`'delta_ce'`-substring key and the *absence* of any `'measure_loss'`-substring key,
plus no `value/`/`critic/`-prefixed keys.

**`actor_loss` q-term (§2.6):** in the flow-actor path's final `else:` branch (the one
computing `q1, q2 = self.network.select('critic')(obs, goals, actor_actions, *zarg)`),
add a branch **before** it when `measure_delta_partition`:
`z_g = sg[_delta_phi(goals, self.delta_protos, blocks)]`;
`logits_a = delta_head([obs, actor_actions, goals])` — **no `params=grad_params`**
(frozen; test 6 checks the gradient into `modules_delta_head` from `actor_loss` is
~0); `q1, q2 = (softmax(logits_a,-1) * z_g[None]).sum(-1)`. `goals = batch['actor_goals']`
(already bound in scope). Test 7 reconstructs `actor_actions` deterministically from
the pre-existing rng-split sequence (`rng, x0_rng, t_rng, n_rng, z_rng =
jax.random.split(rng, 5); noise = normal(n_rng, ...)`) — do not change that sequence's
position relative to this new branch.

**`create()` (§2.7):**
- Module registration (next to `center_b`/`nhat_head` blocks, so default tree stays
  byte-identical): `network_info['delta_head']` and `network_info['target_delta_head']`
  = `ensemblize(MLP, 2)((*value_hidden_dims, K), activate_final=False, layer_norm=...)`
  (and a `copy.deepcopy` for the target), example input
  `jnp.concatenate([ex_observations, ex_actions, ex_goals], -1)`. `K =
  protos_np.shape[0]`, loaded from `config['measure_delta_protos']` via `np.load`.
- Assert: `protos_np.ndim==2 and protos_np.shape[1]==d_c and protos_np.shape[0]>=2`
  where `d_c = sum(b-a for a,b,e in blocks)` (test 8: wrong-width protos file refused).
- After the existing `params['modules_target_critic'] = params['modules_critic']`
  line: mirror it with `params['modules_target_delta_head'] =
  params['modules_delta_head']` when the knob is on (test 4's init-mirror check).
- Return `cls(..., delta_protos=jnp.asarray(protos_np, jnp.float32))` (else `None`).
- Assert-lattice (one block, mirrors the other knobs' style; test 8 exercises every
  line): requires `measure_only=True`; refuses `measure_delta=True` (17a);
  refuses `measure_first_passage=True`; requires `actor_kind=='flow'` and
  `actor_cond=='goal'`; requires `actor_q_center=True`; requires `config['encoder'] is
  None`; requires `measure_delta_protos` truthy; requires `measure_ortho_coef==0.0`;
  refuses `fb_forward_z`/`sf_feature_td`/`measure_expectile`/`measure_nhat_head`/
  `actor_z_infer`/`fwd_two_tower`/`isfb`/`freeze_critic_ckpt`/`warm_start_ckpt`;
  requires `measure_nstep==0` and no `measure_eps_multiscale`; requires
  `centered_measure=='off'`, `task_prior=='dirac'`, `task_prior_actor=='dirac'`.

**`update()` override (§2.8):** `GCIQLFlowAgent` currently inherits `update` from
`GCIQLAgent` (soft-updates only `'critic'`). Add an override identical to the base
plus: `if self.config.get('measure_delta_partition', False):
self.target_update(new_network, 'delta_head')`. Test 4 fails specifically if this is
missing (target stays frozen at init values forever) even if everything else is
correct.

## STATUS / concerns

- STATUS: all three deliverables complete and verified. 2/10 tests green as expected
  (`test_default_off_bit_identical`, `test_prototype_script`); 8/10 red, every
  red confirmed (via actual subprocess stderr, not inference) to fail on a missing
  17b symbol, not a test bug.
- Concern: test 7's exact-match check (`atol=1e-4` on `q_mean`) depends on the
  implementer not moving the pre-existing `rng, x0_rng, t_rng, n_rng, z_rng =
  jax.random.split(rng, 5)` line relative to where `actor_actions` gets sampled; spec
  §2.6 only asks for a branch inserted at the *q-term* call site, so this should hold,
  but flag it if the implementer's diff touches earlier lines of `actor_loss`.
  Similarly for test 3's rng reproduction inside `delta_partition_loss`
  (`noise = jax.random.normal(rng, ...)` with no split).
- Concern: today, test 8's "wrong-width protos" and "measure_only=False" refusals pass
  vacuously for the wrong reason on some sub-checks (no assert-lattice exists yet to
  raise) rather than a real assertion; once §2.7 lands, re-run test 8 and confirm each
  `refuses(...)` case now fails on the SPECIFIC assertion message named in the
  contract above, not just "some AssertionError somewhere."
