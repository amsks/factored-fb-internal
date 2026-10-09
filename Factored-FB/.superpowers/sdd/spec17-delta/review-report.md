# Spec 17a / 17b — independent correctness review

Reviewer: read-only gate pass before the 6-job wave launch. All 18 delta tests
re-run green here (`tests/test_gciql_measure_delta.py` 8/8,
`tests/test_gciql_measure_delta_partition.py` 10/10, 179 s). Findings below are
things the green suite does **not** catch. Two are launch blockers.

Verification environment: `/dev/shm/.venv-jax/bin/python`, JAX_PLATFORMS=cpu,
PYTHONPATH = shim + ogbench impls. Scratch scripts under the session scratchpad
(`t_coef.py`, `t_train.py`, `t_restore.py`); nothing in the repo was modified.

---

## 1. CRITICAL — the 17b arm's CE loss is multiplied by ZERO; `delta_head` never trains

**Where:** `configs/gciql/cube_single_state_measure_delta_partition_d50.yaml:35-43`
(agent_overrides) × `tools/wandb_mode_shim/gciql_flow.py:2178`
(`loss = self.config.get('measure_coef', 1.0) * m_loss + actor_loss`)
× `tools/wandb_mode_shim/gciql_flow.py:3391` (`config.measure_coef = 0.0`, the
get_config default).

**Spec clause:** spec 17b §6 explicitly instructs "DROP the measure-matrix knobs
the base carries (`measure_coef`, …): none of them touch the 17b path". That
premise is false — `measure_coef` is read on the shared `measure_only` return
path at line 2178, which 17b routes through (§2.5 says to call
`delta_partition_loss` *inside* that branch). The implementer followed the spec
literally.

**Failure scenario (verified, not inferred):** the shipped yaml leaves
`measure_coef` at its get_config default `0.0`. `total_loss` therefore returns
`0.0 * ce + actor_loss`. Reproduced with exactly the yaml's agent_overrides:

```
measure_coef in effect: 0.0
grad norm delta_head:        0.0
grad norm target_delta_head: 0.0
grad norm actor:             5.90
grad norm actor_vf:          1.35
info keys: [... 'measure/delta_ce', 'measure/delta_p_entropy', ...]
```

and over 30 real `update()` steps:

```
measure_coef=default(0.0): max delta_head param change = 0.000e+00   ce=2.3181
measure_coef=1.0:          max delta_head param change = 7.317e-03   ce=2.3263
```

So `d17_cs_part` would burn 500k steps training a **pure BC actor** whose q-term
reads a *randomly initialised, permanently frozen* head. `delta_ce` still logs a
plausible finite number (it is computed, just not weighted), `delta_p_entropy`
sits near `log K`, and the pre-registered read for that signature is "B's
zero-flux analogue" (spec 17 §4) — i.e. the run would be **misinterpreted as a
scientific NULL** rather than recognised as a wiring bug. G3 would fail for the
wrong reason.

**Fix options (pick one, do not do both silently):** add `measure_coef: 1.0` to
the yaml, **or** stop multiplying by `measure_coef` in the 17b sub-branch (17b
already has its own `measure_delta_coef`). Either way add a lattice assert
`measure_coef * measure_delta_coef > 0` so this cannot recur, and fix spec 17b §6.

---

## 2. CRITICAL — the new `delta_protos` dataclass field breaks `restore_agent` for every pre-existing checkpoint

**Where:** `tools/wandb_mode_shim/gciql_flow.py:749-753` (`delta_protos: Any = None`
on `GCIQLFlowAgent`, a **pytree data field**) ×
`third_party/ogbench/impls/utils/flax_utils.py:198`
(`flax.serialization.from_state_dict(agent, load_dict['agent'])`).

**Spec clause:** spec 17b §2.2 asserts "Old checkpoints keep loading: restore only
touches `['agent']['network']`." That claim is wrong — `restore_agent` calls
`from_state_dict` on the **whole agent**, and flax.struct's registered
`from_state_dict` raises on any data field missing from the state dict.

**Failure scenario (verified):** a real anchor checkpoint,
`/home/adityamohan/exp_spec08/factored-fb/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0/params_500000.pkl`,
has `agent` state-dict top keys `['rng', 'network']`. Restoring it into the new
agent class:

```
RESTORE-OLD FAILED: ValueError Missing field delta_protos in state dict
  while restoring an instance of GCIQLFlowAgent, at path .     (flax 0.12.7)
```

This is a hard crash, not a silent wrong result, but it takes out a lot at once:
`third_party/ogbench/impls/main.py:87` (`--restore_path` resume), and every eval
harness that rebuilds+restores a GCIQLFlowAgent — `scripts/rescuing_fb/t3_offpeak_multienv.py:86`,
`t6_gpi_eval.py:108`, `measure_readout_head.py:191`, `eval_rgfb_zeroshot.py:79`,
`t5c_srank.py:73`, `render_policies_jax.py:66`, `eval_zeroshot_rewards.py:581`,
plus `scripts/phase_probe_gciql.py:123`, `scripts/oodgrid_run_jax.py:93`,
`scripts/profiles/gciql_profile.py:221`, `gciql_feature_extract.py:190`,
`scripts/figures/outcome_clips.py:235`, `scripts/gaussian_vs_flow_gciql.py:59`.

Concretely: the moment this lands, **the spec 17 §6 post-wave battery cannot
evaluate any of its own anchors** (§9.16 phase-1 ckpts, the spec-14 L2 harness on
the 500k ckpts, the results-CSV pipeline). `warm_start_ckpt` itself is unaffected
— gciql_flow.py:3260 reads the raw pickle and splices `['agent']['network']['params']`
by hand — so training would start fine and the breakage would surface only at
collection time, after the GPU days are spent.

**Fix options:** keep the protos out of the agent pytree (e.g. carry the path in
`config`, which is a `nonpytree_field`, and load lazily / bind at create into a
closure or a `pytree_node=False` field holding a plain tuple), or make
`restore_agent` tolerant of missing fields. Whichever is chosen, add a regression
test that restores a 2-key `{'rng','network'}` state dict.

---

## 3. HIGH — the 17b test suite never exercises a *trained* head; tests 4 and 10 pass tautologically

**Where:** `tests/test_gciql_measure_delta_partition.py:68-83` (`base_cfg`, which
omits `measure_coef`), used by tests 3–8 and 10.

Because `base_cfg` inherits `measure_coef = 0.0`, every `update()`-based test runs
the exact broken configuration of finding 1 and cannot distinguish it from a
working one:

- `test_target_head_init_and_soft_update` (line 252) — with zero gradient the
  online head never moves (measured: max param change 0.000e+00 over 30 updates),
  so `online == old_target` forever and the asserted formula reduces to
  `old == tau*old + (1-tau)*old`. **This test passes with the entire `update()`
  override at gciql_flow.py:2266-2284 deleted** — precisely the failure (§2.8:
  "target frozen forever") it was written to catch.
- `test_end_to_end_updates` (line 465) — 30 updates that train nothing. It asserts
  finiteness and simplex-ness of a head at its init values; it would pass with
  `delta_partition_loss` returning a constant.

`test_measure_only_wiring` (line 285) is also insensitive: it checks that
`delta_ce` appears in `info`, which it does at coefficient zero.

**Recommendation:** set `measure_coef: 1.0` in `base_cfg`, and add to test 4/10 an
assertion that `modules_delta_head` params actually change (and that
`target_delta_head` moves *toward* but not *onto* the online head, i.e.
`old != new != online`), plus a falling-CE assertion in test 10.

---

## 4. MEDIUM — gate G2's instrument (`msq`) is not logged anywhere

**Where:** `tools/wandb_mode_shim/gciql_flow.py:1368` (`msq` is computed inside
`_measure_terms` and discarded) vs. `measure_loss`'s info dict at line 1755-1766.

**Spec clause:** spec 17 §6 gate **G2** — "no zero-flux collapse (`msq` bounded
away from 0, event-rate-scaled)".

**Failure scenario:** the 17a arms' primary pre-registered failure signature is
`msq → 0`. Nothing in the run logs exposes `msq`; the info keys are
`measure_loss / measure_offdiag / measure_diag / measure_orth` (all already
scale-normed, hence *insensitive* to the very quantity G2 tests) plus the new
`measure_delta_event_rate / measure_delta_hit_level`. `actor/q_abs_mean` is a
partial proxy but is measured at actor goals, not at the measure's columns. G2
would have to be adjudicated post hoc from checkpoints. One extra info key
(`info['measure_msq'] = jnp.mean(M**2)`) in the fp branch closes it.

---

## 5. MEDIUM — `measure_scale_norm` divisor collapse: the delta arms sit at a very different effective gradient scale than the anchors they are compared to

**Where:** `tools/wandb_mode_shim/gciql_flow.py:1366-1372`
(`offdiag /= msq + 1e-6`, `diagt /= msq + 1e-6`) on the `absorbing=False` path
that spec 17a §1.2 deliberately reuses.

**Spec clause:** spec 17 §2 item 13 / §2 item 10 (scale identities pinned up
front). Not a spec violation — the spec knowingly reuses scale_norm — but the
consequence is unpinned by any test or identity check.

**Failure scenario:** the ball-visitation arms these configs are copied from
regress a target whose level is the dwell mass (ledger §9.16: `q_abs` pins to 43),
so `msq ≈ 1.8e3` and `msq >> eps=1e-6`. The delta target is
`J ∈ [−1, 1]` with typical magnitude ~ the event rate, which on the conjunctive
cd/scene regions is small. As `mean(M²)` falls through `1e-6` the normalisation
saturates on `eps` and the term stops being dimensionless: the per-entry gradient
scale changes by many orders of magnitude relative to the anchor arm **at the same
learning rate**, so a divergence or a dead-flat loss on cd/sc is as likely an
optimisation artifact as the pre-registered "zero-flux collapse" read. Combined
with finding 4 (no `msq` logged) this is not diagnosable from the run.

**Recommendation:** log `msq` (finding 4) and pre-register the reading rule; if
`msq` crosses ~1e-5 the G1/G2 reads on that arm are not interpretable as stated.

---

## 6. LOW — unreachable assert (confirmed; the implementer's own report flagged it)

**Where:** `tools/wandb_mode_shim/gciql_flow.py:2770-2771`
(`assert not config.get('measure_delta')` inside the 17b block).

The 17a block at 2741 runs first and its final assert (2761-2762) already refuses
the `measure_delta ∧ measure_delta_partition` combination. Dead code only — the
refusal itself is intact (`test_assert_lattice`'s
`refuses(measure_delta=True)` passes via the 17a message). No action required
beyond knowing the error message a user will see comes from the 17a block.

I found no *missing* refusal: every refusal named in 17a §3 test 7 and 17b §5
test 8 fires, and all lattice blocks (2741, 2767) execute before the module
registration (3131) and the protos `np.load` (3137), so a bad config never gets
as far as touching the filesystem — except the intentional shape check at
3140-3142, which is reachable and tested.

---

## 7. LOW — the CE is weighted by two coefficients

**Where:** `gciql_flow.py:1794` (`measure_delta_coef * ce`) then `2178`
(`measure_coef * m_loss`).

Once finding 1 is fixed by setting `measure_coef: 1.0`, the effective CE weight is
the product of two knobs recorded in different places. Spec 17b §2.1 presents
`measure_delta_coef` as *the* dose knob. Harmless if both are 1.0, but it is a
provenance trap for any later dose sweep.

---

## 8. LOW — stale test-file docstring

**Where:** `tests/test_gciql_measure_delta_partition.py:19-28` still says
"`GCIQLFlowAgent` does not yet implement … tests 1-8 and 10 here are EXPECTED to
fail (red)". They are green. Misleading for the next reader.

---

## What I checked and found CORRECT

- **17a counting convention.** `hit_c` is built from `batch['observations']`
  (gciql_flow.py:1572) and `hit_n` from `batch['next_observations']` (via the
  `hit` computed at 1565 from `nobs`). The off-by-one regression is genuinely
  pinned by `test_hit_c_reads_observations` (it compares the captured `hit`
  matrix elementwise against a reference built from `observations`; swapping the
  source changes the matrix, not just the event rate).
- **17a fixed point.** With the lattice forcing `measure_fp_absorbing=False`,
  `_measure_terms` (1363-1365) yields `cont=1`, `diff = M − (h(x') − h(x)) − γ·tM`
  ⇒ `J = E[Σ γ^t (h(x_{t+1}) − h(x_t))]`. Algebraically
  `Σ γ^t h(x_t) = h(x₀) + γM`, hence `J = (1−γ)M − h(x₀)` exactly as spec 17a §2
  states; `test_identity_tabular` pins it numerically under the same
  reward-at-next-state convention. `_measure_terms` has no sign assumption on
  `hit`, and no `clip` was introduced (statically checked by
  `test_bounded_target_no_clip_in_source`).
- **17a delta block placement.** Inside `if fp:`, after both the multiscale and
  the single-eps `hit` assignments; multiscale is lattice-refused, and the
  `measure_nhat_head` column-mean consumer (1576-1590, which would be meaningless
  on a signed hit) is lattice-refused too. The hindsight path recomputes
  `col_states = value_goals` *before* the delta block, so `hit_c` uses the same
  columns as `hit_n` — correct.
- **17b target is a genuine probability vector.** `p_tgt = (1−γ)φ(x') + γ·p̄`,
  both terms softmax outputs ⇒ nonneg, sums to 1 (asserted in test 3); counting
  is from the NEXT state (`phi_n` from `nobs`, gciql_flow.py:1791), and the
  wrong-side variant is explicitly shown to give a different loss.
- **Grad hygiene, verified independently of test 6.** A `jax.grad` of `total_loss`
  under a delta-partition config returns **exactly 0.0** for both
  `modules_delta_head` and `modules_target_delta_head` while `modules_actor` and
  `modules_actor_vf` are nonzero — i.e. the actor q-term (gciql_flow.py:1036-1038)
  really does read the head at frozen params, and the bootstrap action
  (1787) is `stop_gradient`-ed. For 17a, the actor q-term is the untouched stock
  critic call at 1058 (no `params=grad_params`).
- **Default-off bit-identity.** No new `jax.random.split` or `jax.random.normal`
  on any off path: 17b's `noise` draw (1786) consumes the same `m_rng` that
  `measure_loss` would have, inside a branch that only exists when the knob is on;
  the `update` override (2266-2284) is line-for-line the base plus a
  trace-time-static conditional; module registration (3132-3148) and the
  target-init mirror (3321-3322) are both gated. Info keys are appended only when
  the knob is on (1764-1766). Param-tree and loss bit-identity are covered by both
  suites' `test_default_off_bit_identical`.
- **17a configs.** Diffed against their ballvisit bases: the only differences are
  `measure_delta: true`, the new run_group, `eval_interval 50000→10000` on cd/sc
  (both arms, both envs), and `measure_stopgrad_psi: true→false` on the
  from-scratch cube-single arm — exactly the spec 17a §5 table, including the
  FrozenPsi trap. hit_specs are byte-equal to the bases (cd
  `19:22:0.4,28:31:0.4`, scene the 5-block string). cube-single keeps
  `eval_interval: 50000`. No run_group in `configs/gciql/` is duplicated.
- **`make_delta_prototypes.py`** — grammar matches `_parse_hit_spec`, draws
  distinct rows without replacement, saves float32 `[K, d_c]` in raw obs units,
  prints the coverage stats the spec asks for. `results/delta_protos/` exists and
  is empty (protos are generated at launch, as specified).
- **Anti-scope (17b §4 / 17 §2).** No ortho, mass/nonneg penalty, scale_norm,
  hindsight, warm-start or added pessimism anywhere in the 17b path; the lattice
  refuses ortho and warm start; the only min() is the stock 2-head `double_q`.
- **`_delta_phi` ε scaling** divides each block by its own ε before the squared
  distance, for both rows and prototype columns, using cumulative block offsets —
  correct for scene's heterogeneous tolerances, and pinned by test 2 against an
  independent numpy reference.
- **NaN risk in `log(p + 1e-8)`** — bounded: an underflowed `p_k = 0` gives a
  finite `-1e8·p_tgt` factor multiplied by a softmax Jacobian that is exactly 0
  there, so no inf/NaN; 30 real updates produce finite params.

---

# Re-review (fix round 1)

Independent verification, read-only. Environment: `/dev/shm/.venv-jax/bin/python`,
`JAX_PLATFORMS=cpu`, PYTHONPATH = shim + ogbench impls. Scratch scripts under the
session scratchpad (`v1.py`–`v5.py`, `t4.py`, plus two patched copies of
`gciql_flow.py` used as controls). Nothing in the repo was modified except this
report. `.venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py
tests/test_gciql_measure_delta.py -q` → **20 passed in 204 s**.

## 1. CRITICAL 1 (partition CE weighted by zero) — **ADDRESSED**

**(a) Shipped-config gradient.** Built `cfg` from `get_config()` + the *exact*
`agent_overrides` block of
`configs/gciql/cube_single_state_measure_delta_partition_d50.yaml`
(`measure_coef` absent ⇒ 0.0, `measure_delta_coef` 1.0):

```
measure_coef in effect: 0.0        measure_delta_coef: 1.0
grad |modules_delta_head|        = 3352.21     (was 0.0 pre-fix)
grad |modules_target_delta_head| = 0.0         (correct: soft-updated, not trained)
grad |modules_actor|             = 2372.72
max delta_head param change over 6 real update() steps = 1.81e-3
```

At the real launch scale (K=256 protos from
`results/delta_protos/cube_single_K256_s0.npy`, B=1024) 12 `update()` steps run
clean: `delta_ce` 5.65, `delta_p_entropy` 5.02 < log K = 5.545.

**(b) Non-partition path numerically unchanged.** I built a control copy of the
*current* file with only the round-1 edits reverted (unconditional
`measure_coef * m_loss`, `measure_delta_msq` removed, `delta_protos` back to a
pytree field) and compared `total_loss` bit-for-bit under two configs whose
`agent_overrides` are copied verbatim from
`cube_single_state_measure_ballvisit_d50.yaml` and
`cube_single_state_measure_delta_d50.yaml`:

```
                fixed                                 reverted-control
ballvisit  7.028288841247559 / 0x40e0e7be     7.028288841247559 / 0x40e0e7be
17a_delta  7.029975891113281 / 0x40e0f590     7.029975891113281 / 0x40e0f590
post-update param-sum (f64 hex)  identical in both rows
```

The `else:` arm at gciql_flow.py:2208 is the original line verbatim; the only
17a-visible change is the extra info key. No existing measure cell is rescaled.

## 2. CRITICAL 2 (`delta_protos` broke restore) — **ADDRESSED** (all four properties)

**(a) Real anchor checkpoint.** Config reconstructed from the run's own
`flags.json['agent']`, obs/act dims 37/5:

```
ckpt agent keys: ['network', 'rng']
created, delta_protos = None
RESTORE OK; delta_protos = None ; params changed by restore: True
```
against `/home/adityamohan/exp_spec08/factored-fb/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0/params_500000.pkl`.
The reverted-control copy reproduces the original failure on the same pkl
(`ValueError: Missing field delta_protos ... at path .`), so this is a genuine
before/after, not a passing tautology.

**(b) jit / re-trace.** `jax.log_compiles(True)` around 6 `update()` steps at K=8
and 12 steps at K=256: **1 XLA compilation** in both cases. No unhashable-static
error; the nested tuple's hash/eq are by value, so the cache key is stable across
steps. K=256/B=1024 step time 0.097 s on CPU with no per-step compile.

**(c) `self.replace()` preservation.** `aa.delta_protos == a.delta_protos` after 6
`update()` steps, type `tuple` — `dataclasses.replace` carries the un-named field.

**(d) Bit-identity of the tuple round-trip.** `np.asarray(jnp.asarray(
agent.delta_protos, jnp.float32)).view(np.uint32)` is `array_equal` to the on-disk
`.npy` viewed as uint32 — for both the toy K=8 file and the real
`cube_single_K256_s0.npy` (256×3 float32). float32 → Python float → float32 is
exact; no coordinate shifts.

**New-checkpoint round trip.** A 17b agent trained 3 steps then
`to_state_dict` → fresh `create()` → `from_state_dict` restores with
`modules_delta_head` present and every leaf equal.

## 3. HIGH (degenerate soft-update test) — **ADDRESSED**

The formula the test asserts is the one `target_update`
(`third_party/ogbench/impls/agents/gciql.py:149-153`) actually computes: both
operands are read from `self.network.params`, i.e. the **pre-step** online and
target params, so `expected = tau*dh1 + (1-tau)*tdh1` at step 2 is correct (the
old `dh1`-as-post-step version was wrong, as the implementer says).

Sensitivity confirmed empirically, not by reasoning alone. I made a copy of the
current file with only the two override lines
(`if measure_delta_partition: self.target_update(new_network, 'delta_head')`)
replaced by `pass`, and ran the test-4 body against both:

```
with override (repo):  max |target2 - expected| = 7.45e-09   PASS (atol 1e-7)
override deleted:      max |target2 - expected| = 1.55e-06   FAIL
```

The discrimination is structural: after step 1, `tdh1 == dh0 != dh1`, so a frozen
target differs from the formula by `tau*(dh1 - tdh1) != 0`. Margins are ~13× below
and ~15× above the 1e-7 tolerance at `tau=0.005` — real, though not generous.
The added "delta_head actually moved after step 1" assertion pins the test's own
gradient-flow premise, so it can no longer regress into the CRITICAL-1 shape.

## 4. MEDIUM (gate G2 instrument) — **ADDRESSED**

`info['measure_delta_msq'] = delta_msq` (gciql_flow.py:1787) with
`delta_msq = jnp.mean(M ** 2)` (:1595), assigned inside the
`if config.get('measure_delta')` block and guarded at the info site by the shared
`if delta_event_rate is not None:`. Emission verified directly on `total_loss`
info dicts:

```
17a_delta : ... measure/measure_delta_event_rate, measure/measure_delta_hit_level,
                measure/measure_delta_msq, ...
ballvisit : (no measure_delta_* keys at all)
```

The reported quantity is the right one: the `M` at :1595 is the same online
measure matrix later handed to `_measure_terms` (:1721), whose internal
`msq = mean(M**2)` is exactly the divisor G2 is about — the term the scale-normed
losses divide out. Paired with the already-present `measure_delta_event_rate`,
G2's "msq bounded away from 0, event-rate-scaled" is now adjudicable from the run
logs. Adding the key does not perturb the loss (verified bit-identical in §1b).

## 5. New breakage introduced by the fixes

**NEW — MEDIUM — prototypes are not in the checkpoint, and a mismatch is silent.**
`gciql_flow.py:3161-3178` loads the partition from
`config['measure_delta_protos']` (an absolute path into untracked
`results/delta_protos/`), and the CRITICAL-2 fix makes `delta_protos` a
*non-serialized* field. A 17b checkpoint therefore records the head's weights but
**not** the partition they were trained against; every post-wave eval rebuilds it
by re-reading that path. Demonstrated: restoring a 17b state dict into an agent
created with a permuted prototype file raises nothing and silently changes the
actor q-term (`q_mean` 0.00322 → 0.00313 at near-init weights; the divergence is
unbounded once the head is trained). Only the shape `[K>=2, 3]` is checked.
Mitigation is operational, not code: freeze/back up
`results/delta_protos/cube_single_K256_s0.npy` before launch and never regenerate
it in place. Consider recording a hash of the file in the run config.

**NEW — LOW — `tests/test_gciql_measure_delta_partition.py:19-33`**: the rewritten
module docstring says "All 10 tests are green" (there are now 12) and points
CRITICAL 2's regression test at `tests/test_gciql_checkpoint_restore.py`, **which
does not exist** — the test is #12 in this same file. The LOW-4 docstring fix
replaced one inaccuracy with another.

**NEW — LOW — `tests/test_gciql_measure_delta_partition.py:605`**: test 12's third
assertion, `restored.delta_protos == agent_off.delta_protos`, is `None == None` —
tautological. The substantive proof is the `to_state_dict` key check plus the
2-key `from_state_dict` call; an ON-knob agent is never restored from an
old-style dict in the suite (I did that by hand instead). No action required, but
the assertion carries no information.

**CARRIED OVER — LOW — `measure_coef` is now inert on the 17b path** (finding 7 of
the first pass, chosen fix option 2). `gciql_flow.py:2199-2208` ignores
`measure_coef` entirely when `measure_delta_partition` is on, and no assert
refuses a config that sets it. Failure mode is benign (a copied
`measure_coef: 0.5` would be silently ignored rather than zeroing the loss), but a
dose sweep run through the wrong knob would produce identical runs with different
recorded provenance.

Findings 5 (scale_norm divisor collapse) and 6/8 of the first pass were
explicitly ruled out of scope by the coordinator and remain as written; finding 5
is now at least *diagnosable* thanks to `measure_delta_msq`.
