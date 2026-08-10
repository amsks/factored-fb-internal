# Spec 17a — `measure_delta` implementation contract (variant A)

Audience: the implementing agent. Self-contained; science context in
`specs/17-delta-measure.md`. Follow the repo's TDD discipline: write the
tests first, knob default-off must be BIT-identical.

## 0. What is being built

A new default-off knob `measure_delta` on the jax agent
`tools/wandb_mode_shim/gciql_flow.py` (class `GCIQLFlowAgent`). When on, the
region successor-measure TD target's reward changes from the arrival
indicator to the SIGNED EVENT indicator:

    stock (ball-visitation):  diff = M − hit_n           − γ·tM
    delta:                    diff = M − (hit_n − hit_c) − γ·tM

where `hit_n[i,j] = 1[obj(next_observations_i) ∈ region_j]` (already
computed) and `hit_c[i,j] = 1[obj(observations_i) ∈ region_j]` (NEW — same
`_hit_matrix`, rows = `batch['observations']`). Non-absorbing (cont ≡ 1).
Fixed point: `J = (1−γ)·M_visitation − h(x₀)`, bounded in [−1, 1]; same
argmax over actions as the visitation measure at fixed (s, columns).

Everything else — columns (batch tpd = `psi(next_observations)` or
hindsight tpd = `psi(value_goals)`), bilinear M, bootstrap action, ortho
(off), scale_norm, stopgrads, warm-start splicing — is untouched. No
sampler/GCDataset change (observations are already in the batch). No new
modules (param tree unchanged ⇒ warm_start_ckpt works as-is).

## 1. Exact code changes (one file)

`tools/wandb_mode_shim/gciql_flow.py` — three touch points:

### 1.1 `get_config()`

Add next to `measure_fp_absorbing` / `measure_nstep`:

```python
config.measure_delta = False  # spec 17a: delta (temporally-centered) region measure -- the
  # target reward becomes hit(x')-hit(x) (enter +1 / dwell 0 / exit -1), non-absorbing;
  # fixed point J = (1-gamma)*M_visitation - h(x0), bounded in [-1,1], same action argmax as
  # the visitation measure. Requires measure_first_passage=True (the hit-region path) +
  # measure_fp_absorbing=False; refuses nstep/expectile/multiscale/nhat/task_prior.
  # False = off, bit-identical.
```

### 1.2 `measure_loss` — the fp region branch

Locate the fp block in the critic-legs path (grep `if fp:` inside
`measure_loss`; currently ~line 1495). The single-radius path computes
`hit = self._hit_matrix(nobs, col_states, blocks, M.dtype)`. Immediately
after that line (the NON-multiscale path only — the lattice refuses
multiscale), add:

```python
if self.config.get('measure_delta', False):
    # spec 17a: signed event target hit(x')-hit(x). hit_n stays for the info
    # keys; the residual sees only the temporal difference, so the dwell
    # plateau (inside->inside) contributes zero and the level channel is
    # deleted from the supervision (ledger 9.16 / 9.15c motivation).
    hit_n = hit
    hit_c = self._hit_matrix(obs, col_states, blocks, M.dtype)      # [B,B]
    hit = hit_n - hit_c
    delta_event_rate = jnp.mean(jnp.abs(hit))
    delta_hit_level = jnp.mean(hit_n)
```

Initialize `delta_event_rate = delta_hit_level = None` next to the existing
`nstep_k_mean = None` sentinel at the top of `measure_loss`, and append to
`info` at the bottom exactly like `measure_nstep_k_mean`:

```python
if delta_event_rate is not None:
    info['measure_delta_event_rate'] = delta_event_rate
    info['measure_delta_hit_level'] = delta_hit_level
```

NO change is needed at the `_measure_terms(...)` call site: with the lattice
forcing `measure_fp_absorbing=False`, the existing
`absorbing=self.config.get('measure_fp_absorbing', True)` already routes to
the non-absorbing branch (`cont = 1`, squared diagonal, scale_norm dividing
BOTH terms by `msq`). `_measure_terms` itself is untouched — verify it has
no `hit >= 0` assumption (it doesn't: `diff = M - hit - gamma*cont*tM` is
sign-agnostic; `cont` ignores `hit` when `absorbing=False`).

Interaction to be aware of (and covered by the lattice): the
`measure_nhat_head` block reads `hit.mean(axis=0)` as an arrival-rate
target — meaningless for a signed hit; refused. The `elif tpd ==
'hindsight' and hit is None:` branch is unreachable under fp=True; hindsight
+ delta flows through the ordinary `hit is not None` path with columns =
value_goals, which is intended.

### 1.3 `create()` — assert-lattice

Mirror the spec-16 `measure_nstep` lattice block (grep `measure_nstep` in
`create`; currently ~line 2447). Add a `measure_delta` block:

```python
if config.get('measure_delta', False):
    assert config.get('measure_first_passage', False), \
        "measure_delta rides the hit-region path: set measure_first_passage=True"
    assert not config.get('measure_fp_absorbing', True), \
        "measure_delta is non-absorbing by construction: set measure_fp_absorbing=False"
    assert config.get('measure_legs', 'critic') == 'critic', \
        "measure_delta lives on the critic-legs measure path"
    assert config.get('value_arch', 'mlp') == 'bilinear', \
        "measure_delta requires value_arch='bilinear'"
    assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
        "measure_delta needs an active measure loss"
    assert int(config.get('measure_nstep', 0) or 0) == 0, \
        "measure_delta x measure_nstep composition is refused in round 1 (spec 17 §7)"
    for bad in ('measure_expectile', 'measure_nhat_head', 'fb_forward_z'):
        assert not config.get(bad, False), f"measure_delta is incompatible with {bad}"
    assert not config.get('measure_eps_multiscale', ''), \
        "measure_delta x multiscale is untested; refused in round 1"
    assert config.get('task_prior', 'dirac') == 'dirac' \
        and config.get('task_prior_actor', 'dirac') == 'dirac', \
        "measure_delta refuses the task_prior arms (same discipline as the other measure knobs)"
    assert not config.get('measure_delta_partition', False), \
        "measure_delta and measure_delta_partition are mutually exclusive (17a vs 17b)"
```

(`measure_delta_partition` is 17b's knob; guard with `config.get(...)` so
17a merges independently. `measure_testpoint_dist` batch AND hindsight are
both ALLOWED — hindsight is the standing recipe.)

## 2. Semantics the tests must pin (conventions)

- Counting convention: the measure counts arrival at s' (`h(x_{t+1})`,
  t ≥ 0). J's Bellman: `J(s,a) = E[(h(x₁)−h(x₀)) + γ·J(s',a⁺)]` with
  `x₀ = obj(observations)`, `x₁ = obj(next_observations)`. So the code's
  `hit_c` MUST come from `observations` and `hit_n` from
  `next_observations` — an off-by-one here silently changes the object.
- Transition table (per column j): outside→outside 0 · outside→inside +1 ·
  inside→inside 0 · inside→outside −1.
- Identity: for the exact fixed points on any MDP,
  `J = (1−γ)·M_visitation − h(x₀)` under the SAME γ-power convention as the
  code (M_visitation = the `measure_fp_absorbing=False` fixed point).

## 3. Tests — `tests/test_gciql_measure_delta.py`

Copy the structure of `tests/test_gciql_measure_nstep.py` verbatim
(`tests.jax_subprocess.run_jax`, `COMMON` preamble with `base_cfg`/
`make_batch`/`agent_for`/`measure_loss_of`; OBS_D=40, cube-single block
19:22:0.4, d=8, B=16). `base_cfg` for delta:
`measure_first_passage=True, measure_fp_absorbing=False, measure_delta=True`
on top of the measure-only/bilinear/no-ortho/scale_norm base. Tests:

1. `test_default_off_bit_identical` — `measure_delta=False` config: param
   tree identical to a no-knob config AND `measure_loss` bit-identical on
   the same batch (both tpd values).
2. `test_delta_target_matches_numpy_reference` — constructed batch that
   realizes all four transition types against the batch's own nobs columns
   (place cube xyz obs[19:22] / nobs[19:22] inside/outside the 0.4-ball of
   selected columns). Compute the expected loss in numpy: rebuild M and tM
   exactly as `measure_loss_of` does (call the critic directly for phi/psi
   — see the nstep reference test for the pattern), then
   `diff = M − (hit_n−hit_c) − γ·tM`, squared diag, offdiag, scale_norm
   `/msq`, mean over the 2 heads, and compare to the returned loss (atol
   1e-5). Run for tpd batch AND hindsight (hindsight: columns/psi legs at
   value_goals — mirror the code's hindsight recompute).
3. `test_hit_c_reads_observations` — regression against the off-by-one:
   batch where obs is INSIDE the column ball and nobs OUTSIDE (pure exit):
   the immediate target must be −1 for that entry, i.e. loss must match the
   reference built with hit_c from observations; also assert
   `info['measure_delta_event_rate'] > 0` there and `== 0` on a batch whose
   obs/nobs both sit far outside every column ball.
4. `test_identity_tabular` — pure numpy (no agent): a 3-state deterministic
   chain with a 1-state region; compute M_visitation and J by power
   iteration of their Bellman equations (the code's convention: reward at
   the NEXT state) and assert `J = (1−γ)M − h(x₀)` to 1e-6, for two γ.
   This pins the eval-side conversion `M̂ = (h + Ĵ)/(1−γ)` used later by
   the L1fix/L2 harnesses.
5. `test_bounded_target` — with `measure_delta`, the immediate target
   entries lie in {−1, 0, +1} (read back `hit` indirectly: build a batch
   with known geometry and assert the loss equals the reference where the
   signed values are exactly ±1/0; no jnp.clip may exist in the path).
6. `test_info_keys` — `measure_delta_event_rate`/`measure_delta_hit_level`
   present iff the knob is on; absent on the stock path (regression: no key
   leakage when off).
7. `test_assert_lattice` — refusals: delta without fp; delta with
   fp_absorbing=True; delta+nstep; delta+expectile; delta+nhat_head;
   delta+eps_multiscale; delta+fb_forward_z. Accepts: delta (base),
   delta+hindsight, delta+measure_stopgrad_psi, delta+warm-start-shaped
   configs (no ckpt needed — just no assert).

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_delta.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself). After
green, ALSO run the neighbors (regression):
`tests/test_gciql_measure_nstep.py tests/test_gciql_measure_hit_spec.py`
plus the other measure suites the spec-16 session ran (grep tests/ for
`measure`), all green.

## 4. Grad-hygiene note (existing behavior, re-assert)

The flow actor's q-term already evaluates the critic WITHOUT
`params=grad_params` (crl-actor-loss-critic-gradient lesson). Delta changes
what the critic's fixed point IS, not how the actor reads it — add one
assertion to test 2's body: `jax.grad(actor_loss)` w.r.t. critic-path
leaves ≈ 0 on a delta config (copy the check from the memory note: sum
grad-norms over leaves whose path contains 'critic').

## 5. Configs (write after tests are green; do NOT launch)

Six yamls in `configs/gciql/`, each a copy of its base with the delta knobs
and a NEW distinct run_group (group names lie — never reuse):

| new config | base | delta knobs on top |
|---|---|---|
| `cube_double_state_measure_delta_d50.yaml` | `cube_double_state_measure_ballvisit_d50.yaml` | `measure_delta: true`, `measure_fp_absorbing: false` (verify base; set explicitly), run_group `GCIQL-CubeDouble-Flow-Measure-Delta-d50`, `eval_interval: 10000` |
| `cube_double_state_measure_delta_hs_d50.yaml` | `cube_double_state_measure_ballvisit_hs_d50.yaml` | same + keeps `measure_testpoint_dist: hindsight`; run_group `...-Delta-Hindsight-d50` |
| `scene_state_measure_delta_d50.yaml` | `scene_state_measure_ballvisit_d50.yaml` | analogous; scene hit_spec stays `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6` |
| `scene_state_measure_delta_hs_d50.yaml` | `scene_state_measure_ballvisit_hs_d50.yaml` | analogous |
| `cube_single_state_measure_delta_d50.yaml` | `cube_single_state_measure_ballvisit_d50.yaml` | from-scratch cs arm; eval_interval stays 50000; **`measure_stopgrad_psi: false`** (see below) |
| (nstep arms reuse spec 16 configs — not 17a's concern) |

⚠ The ballvisit d50 bases are the PHASE-2 FROZEN-PSI recipe
(`measure_stopgrad_psi: true`, run_group `...-FrozenPsi-...`, warm start
supplied by the launcher). That is correct for the warm-started cd/sc delta
arms (psi carried from phase 1). For the FROM-SCRATCH cube-single delta arm
it is silently fatal — stopgrad_psi with no warm start freezes psi at random
init and the cell floors for a fake reason. The cs config MUST set
`measure_stopgrad_psi: false`.

Keep every other knob byte-equal to the base (bc_coeff 3.0, d50,
measure_only, scale_norm true, ortho 0, actor_q_center true as in base).
The wave launcher (spec 17 §6) passes
`+agent_overrides.warm_start_ckpt=...` / `warm_start_epoch=500000` on the
cd/sc arms exactly like `scripts/rescuing_fb/launch_p2fix_arms.sh`.

## 6. Out of scope for 17a

- The L1fix/L2 harness `--delta` conversion flag (`M̂ = (h+Ĵ)/(1−γ)`) —
  separate change, after checkpoints exist.
- Variant B (17b), the prototype script, any new module.
- Launching anything.
