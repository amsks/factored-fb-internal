# Spec 21 — `measure_density_hard_mix`: soft/hard mixture for the density immediate term

Status: WRITTEN 2026-08-11. Spec ONLY — nothing here is implemented.
Parents: spec 18 (`measure_density`, cube-single wave collected 3 seeds) +
its hard-immediate ablation `measure_density_hard` (1 seed collected).
Agent: `tools/wandb_mode_shim/gciql_flow.py` (jax). Sibling contracts:
17a/17b/18 mutual-exclusion lattice unchanged.

TDD; default-off must be BIT-identical.

## 0. One paragraph

Spec 18's CE target is `t = sg[(1−γ)·k + γ·w̄⁺]`. The immediate term `k`
currently has two all-or-nothing forms: the soft ε-kernel
`k = _delta_phi(nobs, u_raw)` (default) and, under
`measure_density_hard=True`, the identity one-hot `k = I` — each row's
immediate target puts all its `(1−γ)` mass on the row's own successor
instead of smearing it over kernel-near queries. On cube-single 500k the
soft arm (3 seeds) gets native goal success .896/.928/.964 but zero-shot
dirac .704/.824/.920, dense .64/.672/.864, region .27/.31/.30 (P=16384,
25 ep). The hard arm (1 seed) gets goal success .876 but zero-shot dirac
.848, dense .896, region .51 — better on EVERY zero-shot family, slightly
worse on native goal success. This spec asks whether a convex mixture
`k := (1−α)·κ_ε + α·one-hot` buys the hard arm's zero-shot readout
without the (possible) goal-success cost, and pre-registers the cheap
seed check that must come first.

## 1. Assessment — is a mixture warranted?

Honest answer: **maybe — not yet.** The hard-vs-soft comparison is
currently confounded by seed count: soft has 3 seeds, hard has 1, and the
soft arm's own spread (dirac .704–.920) is wider than the soft-vs-hard
gap on dirac. Worse, the hard seed's mid-run goal-success evals swung
1.00 → .876, so even the "goal-success cost" that motivates a mixture is
a single noisy point. Two cheap facts must land before any mixture code
is written:

1. Does hard's zero-shot win (dirac ≈ .85, region ≈ .51) survive 2 more
   seeds?
2. Is hard's goal-success deficit vs soft real at 3 vs 3 seeds, or seed
   noise?

Decision table (pre-registered):
- Win survives AND cost is real ⇒ the mixture is the right tool; run the
  α wave (§5, stage 2).
- Win survives AND cost is noise ⇒ no mixture needed; adopt
  `measure_density_hard=True` outright and STOP (the mixture knob stays
  unimplemented).
- Win does not survive seeds ⇒ hard was a 1-seed mirage; STOP, no
  mixture, soft stays the default.

The mixture contract below (§3) is specified now so stage 2 can start the
day stage 1's gate reads out, but implementation is gated on the table.

## 2. Decisions pinned

- **Mixture on the immediate term only.** `k := (1−α)·κ_ε + α·I`. The
  bootstrap term `γ·w̄⁺` and the CE loss are untouched; both k variants
  have rows on the probability simplex, so any convex combination does
  too, and the target `t = sg[(1−γ)·k + γ·w̄⁺]` remains a valid
  probability vector per row — the CE projection stays well-posed with no
  renormalization step.
- **`α=0` is bit-identical to soft.** Not merely equal in expectation —
  the same jaxpr path must produce the same floats (guard the branch at
  trace time on `α > 0`, mirroring how `measure_density_hard` branches).
- **`α=1` equals `measure_density_hard=True` numerically** but the two
  knobs REFUSE to be set together (assert-lattice) — one mechanism, one
  knob; the boolean stays for back-compat with the collected d18 run.
- **No schedule on α.** Fixed scalar for the whole run (ml_collections
  scalar, no lists). Annealing soft→hard is anti-scope (§7).
- **Readout unchanged.** The eval harness
  (`scripts/rescuing_fb/eval_density_zeroshot.py`) and the actor q-term
  (`Σ_j w_j·kernel(goal, u_j)`) keep the SOFT ε-kernel everywhere they
  use a kernel — the mixture changes only what the critic is trained
  toward, never how it is read out. (Train-eval kernel match is already a
  spec-18 invariant; a hard actor/eval kernel would be a point-mass
  indicator and is not meaningful off the query grid.)
- **Same slot, same lattice.** Everything spec 18 §4.6 asserts
  (measure_only, flow actor, actor_q_center, state obs, no ortho, no
  coef, mutual exclusion with 17a/17b) applies verbatim; the new knob
  adds one assert, removes none.

## 3. Implementation contract — `tools/wandb_mode_shim/gciql_flow.py`

### 3.1 `get_config()` (next to `measure_density_hard`)

```python
config.measure_density_hard_mix = 0.0   # spec 21: alpha in [0,1]. Immediate term of the
  # density CE target becomes k = (1-alpha)*eps-kernel + alpha*one-hot(own successor).
  # 0.0 = off, bit-identical to soft; 1.0 = numerically the hard ablation (but the two
  # knobs refuse to coexist). Rows of k stay on the simplex for any alpha, so the CE
  # target sg[(1-gamma)*k + gamma*w_bar+] stays a valid per-row probability vector.
```

### 3.2 `density_loss` — the only functional change

Replace the current two-way branch at the `k` construction (currently
lines ~2036–2039) with:

```python
alpha = float(self.config.get('measure_density_hard_mix', 0.0))
if self.config.get('measure_density_hard', False):
    k = jnp.eye(nobs.shape[0])                                # [B, M], M == B (unchanged path)
elif alpha > 0.0:
    k_soft = self._delta_phi(nobs, u_raw, blocks)             # [B, M]
    k = (1.0 - alpha) * k_soft + alpha * jnp.eye(nobs.shape[0])
else:
    k = self._delta_phi(nobs, u_raw, blocks)                  # [B, M] (unchanged path)
```

Everything downstream (`t`, CE, ESS/entropy info keys) is untouched.
`density_k_self` (E[k_ii]) already exists and directly reads the mixture
strength: it interpolates from the soft self-coverage value toward 1.0
as α → 1. No new info key needed; add none.

### 3.3 Assert-lattice (extend the spec-18 block)

```python
a = float(config.get('measure_density_hard_mix', 0.0))
assert 0.0 <= a <= 1.0
assert not (a > 0.0 and config.get('measure_density_hard', False)), \
    "one mechanism, one knob: use hard_mix OR hard, not both"
assert a == 0.0 or config.get('measure_density', False), \
    "hard_mix is meaningless without measure_density"
```

### 3.4 What does NOT change

No new modules, no target-network changes, no `update` override change,
no GCDataset change, no eval-script change, no actor-loss change. The
diff is one config line, one branch in `density_loss`, three asserts.

## 4. Tests — extend `tests/test_gciql_measure_density.py`

Same harness (`tests.jax_subprocess.run_jax`; OBS_D=40, action dim 5,
d=8, B=16, hit_spec `'19:22:0.4'`). Base config as the spec-18 suite plus
`measure_density_hard_mix` where stated.

1. `test_hard_mix_zero_bit_identical` — `hard_mix=0.0` (explicit) vs
   knob absent: param tree byte-identical, `total_loss` bit-identical on
   a fixed batch (the α=0 trace-time guard, mirroring
   `test_default_off_bit_identical`).
2. `test_hard_mix_matches_numpy_reference` — α=0.3: rebuild
   `k = 0.7·_delta_phi + 0.3·I`, then `t` and the CE in numpy from
   direct module calls with the agent's own params (atol 1e-5).
3. `test_hard_mix_simplex` — for α ∈ {0.25, 0.5, 1.0}: rows of k sum to
   1, rows of t sum to 1, all entries ≥ 0; `density_k_self` increases
   monotonically in α on a fixed batch.
4. `test_hard_mix_one_equals_hard` — α=1.0 CE equals
   `measure_density_hard=True` CE on the same batch/params (atol 1e-6).
5. `test_hard_mix_lattice` — refusals: α=0.5 together with
   `measure_density_hard=True`; α=−0.1; α=1.5; α=0.5 without
   `measure_density`. Accepts α=0.5 on the base config.
6. `test_hard_mix_end_to_end` — α=0.5, 30 `update()` steps: finite
   losses, w on the simplex, no NaN in params.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density.py -x -q`,
then the 17a/17b suites (lattice touched) — all green before any launch.

## 5. Pre-registered wave (cube-single, 500k, standing recipe)

All runs: d50 slot settings as the d18 wave (bc_coeff 3.0, eval_interval
50000, wandb entity amsks, tmux, `XLA_PYTHON_CLIENT_PREALLOCATE=false`),
zero-shot readout at P=16384, 25 ep, both dirac/dense/region families.
Distinct run_group per arm; audit by config, not group name.

### Stage 1 — seed the hard arm FIRST (no new code required)

- `d21_cs_hard_s{1,2}` — `measure_density_hard=True`, seeds 1/2
  (seed 0 exists from the d18 wave). Run_group
  `GCIQL-CubeSingle-Flow-Measure-DensityHard-d50` (same as d18 seed 0).

Gate S1 (decides whether stage 2 exists):
- S1a (win survives): 3-seed hard means — zero-shot dirac ≥ .82 (soft's
  3-seed mean) AND region ≥ .40. Fail ⇒ STOP, no mixture, hard was
  noise.
- S1b (cost is real): 3-seed hard goal-success mean < .90 (soft's mean
  is .929). If S1a passes and S1b FAILS (hard also ≥ .90) ⇒ adopt
  hard outright, STOP — no mixture implementation.
- Only S1a ∧ S1b ⇒ implement §3 (TDD, §4) and launch stage 2.

### Stage 2 — mixture (conditional on S1a ∧ S1b)

- `d21_cs_mix25` — α=0.25, seeds 0/1/2. Config
  `configs/gciql/cube_single_state_measure_density_mix25_d50.yaml`,
  run_group `GCIQL-CubeSingle-Flow-Measure-DensityMix25-d50`.
- `d21_cs_mix50` — α=0.5, seeds 0/1/2. Analogous config/group.

Gates (BOTH axes must pass on 3-seed means; passing one is a fail):
- G1 (native axis): goal success ≥ .90.
- G2 (zero-shot axis): dirac ≥ .82 (soft's mean); directional read on
  region (soft ≈ .29, hard ≈ .51) reported but not gated.

Pre-registered interpretations:
- Some α passes G1 ∧ G2 ⇒ the soft kernel's smearing of the `(1−γ)`
  immediate mass degrades point-goal readout, and a partial one-hot
  anchor restores it without breaking the dense supervision that the
  soft kernel provides — adopt the smallest passing α.
- Both α fail G1 (goal success drops with α) ⇒ the hard cost is
  dose-responsive; the two axes trade off through the immediate term —
  report as a RESULT, keep soft default, no further α points.
- Both α fail G2 (zero-shot gain needs α=1) ⇒ the gain is not from the
  one-hot mass per se but from removing the kernel entirely; report,
  do not chase intermediate α.
- Monotone in α on both axes with a crossing ⇒ report the frontier;
  adopting a point is a user decision, not this spec's.

Post-collection: add every run_group to `scripts/results_map.py`, refresh
CSVs (`.venv/bin/python scripts/compile_results_csvs.py --refresh`).

## 6. Anti-scope (do NOT add)

- No cube-double/scene arms until the cube-single gates read out.
- No schedule/annealing on α; no learned α; no per-block α.
- No readout changes: the eval harness, the actor q kernel, and the
  pool machinery stay exactly as spec 18 shipped them.
- No temperature/bandwidth co-sweep with α (one variable per wave).
- No new info keys, penalties, or regularizers riding along.
- No implementation at all unless stage 1's gate S1a ∧ S1b passes —
  the decision table in §1 is binding.

## 7. Open questions carried forward

- If hard wins outright (S1a ∧ ¬S1b): does hard also hold on
  cube-double/scene, where the conjunctive ε-kernel is much sparser?
  Separate wave, separate spec.
- The soft kernel still feeds the actor q-term even when the critic
  trains hard/mixed — whether an α-matched actor readout matters is
  unknown and out of scope here.
- Region-family gap (soft .29 vs hard .51) is the largest effect in the
  table; if the mixture wave runs, region is the most informative
  directional read even though it is ungated.
