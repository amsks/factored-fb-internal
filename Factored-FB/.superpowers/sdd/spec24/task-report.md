# Spec 24 — z-indexed Density-FB — task report

STATUS: DONE. Code + tests written, all suites green, wave config constructs
cleanly, and a real 300-step GPU smoke run through `run_gciql.py` completes
without crashing (round 2, below, after a coordinator-reported launch crash).

## Round 2: real-launch crash + fix
`sample_actions` — the eval-time entrypoint `evaluate()` calls every env
step, including UNBATCHED single-observation calls — had no
`measure_density_z` branch. It fell through to `actor_cond=='goal'`'s plain
path and passed the RAW goal obs into the actor module, which `create()`
correctly registers z-wide (`measure_density_dim`). Fixed by adding an
`elif measure_density_z` branch that maps `goals -> self._density_z_encode(goals)`
(E1's deterministic transform, no rng), with a batch-dim add/strip for the
unbatched case (`_density_z_encode`'s `_density_queries` call slices with an
explicit batch axis, unlike `_actor_z`'s plain Dense calls). No unit test had
exercised `sample_actions` at all (only `actor_loss`/`density_loss` via
`update()`), and the pytest harness's own OBS_D/ACT_D/D (40/5/8) happened to
give g-path width 85 vs z-path width 53 — never coincidentally equal, so a
real regression test would have caught it, it just didn't exist. Added
`test_sample_actions_matches_registered_z_width` with OBS_D2/ACT_D2/DZ2 =
7/3/11 (mutually distinct) that runs `update()` then calls `sample_actions`
in both batched and unbatched shapes — reproduces the exact crash and its
unbatched cousin. Re-ran the full z-suite (9/9) and the sibling suites
(28/28) after the fix — all still green.

### GPU smoke proof (`run_gciql.py`, GPU 6, 300 steps, `train_steps=300 eval_interval=1000000`)
Completed all 300 steps (previously died in <1 min on both seeds). Last 3
`train.csv` rows, all finite:

| step | training/density_ce | training/density_ess_median | training/z_cos_spread |
|---|---|---|---|
| 100 | 4.310378 | 44.924328 | 0.18619317 |
| 200 | 4.394128 | 46.02189 | 0.20361748 |
| 300 | 4.435150 | 46.589287 | 0.21586885 |

(validation/* columns likewise finite.) Output dir left in the scratchpad,
not committed.

## Files changed
- `tools/wandb_mode_shim/gciql_flow.py`: knobs `measure_density_z` /
  `measure_density_z_rand_frac`; helpers `_density_z_encode` (deterministic
  goal→z, also E1's transform), `_density_z` (full mix), `_density_actor_q_z`
  (z-analog of `_density_actor_q`, reward embedding on the TARGET
  `target_density_b`); `density_loss` gets a `z_on` branch (F's `[obs,acts,z]`
  input, bootstrap `pi(s',z)`, `target_density_f`'s `[nobs,a+,z]` input,
  `z_cos_spread` instrument) inserted as a new `elif` before the untouched
  g-indexed `else`; `actor_loss`'s conditioning chain and density q-term
  chain each get one new `elif`; `create()` gets three architecture edits
  (`ex_actor_goals` width, F's `ex_f` width, both gated on
  `measure_density_z`); new lattice block scoping the knob to
  `actor_kind='flow'`, refusing `measure_density_behavior` and an
  out-of-range `z_rand_frac`.
- `tests/test_gciql_measure_density_z.py` (new, 9 tests: the spec's 7 items,
  item 4 split into an actor-q numeric test + a grad-hygiene test, plus the
  round-2 `sample_actions` regression test below).
- `configs/gciql/point_mass_maze_rnd_seed0_measure_density_z_d50.yaml` (new):
  the P0 config verbatim plus `measure_density_z: true`, `run_group`
  changed to `GCIQL-PointMassMaze-RND-Flow-Measure-DensityZ-d50`. `hit_spec`
  reused unmodified.

## Test summary (post-fix)
- `tests/test_gciql_measure_density_z.py -q`: **9/9 passed** (130s).
- `tests/test_gciql_measure_density.py tests/test_gciql_density_actors.py tests/test_gciql_measure_density_behavior.py -q`: **28/28 passed** (343s), unaffected.

## Load-bearing / g-path-unchanged proof
- **z path load-bearing**: `test_grad_hygiene` shows `jax.grad(density_loss)`
  is >1e-8 over `modules_density_f`/`modules_density_b` and `test_end_to_end_updates`
  shows both towers' params actually move after 30 `update()` steps with
  `measure_density_z=True`; `test_actor_q_matches_numpy` confirms the actor's
  `q_mean` is a nonzero function reconstructed independently from
  `target_density_b`/`density_f`/`density_b`. The deliberate-mistake
  sub-test in `test_grad_hygiene` (mirrors the spec-18 suite's
  `test_target_stopgrad_is_load_bearing`) reconstructs z's build with
  `target_density_b` called live (`params=grad_params`) and shows gradient
  into it is nonzero without `stop_gradient(z)`, zero with it shipped —
  proving the wrapper is load-bearing, not decorative.
- **g path unchanged**: `test_default_off_bit_identical` diffs the param
  tree and `total_loss` against the pinned pre-spec-24 SHA
  `602008f1cdaa28cc0b4150e8812ec90e3f6be482` (has `measure_density_behavior`
  but not `measure_density_z`) — exact match, both with the knob explicitly
  off and with no `measure_density` knob at all. The full spec-18/19a/spec-22
  suites (28 tests) stay green untouched.

## Config verification
Ad hoc script (mirroring `run_gciql.py`'s `agent_overrides` merge) loads the
new YAML, merges its overrides onto `get_config()`, and calls
`GCIQLFlowAgent.create()` with point-mass-maze-shaped example tensors
(obs_dim=4, action_dim=2) — constructs cleanly, all four density modules plus
`modules_actor` present, lattice not tripped.

## Concerns
- `measure_density_z` is scoped to `actor_kind='flow'` only (an explicit
  assert added beyond the spec's literal pseudocode) — `ddpgbc`/`awr_flow`
  would need their own z-conditioned architecture and q-branches, out of
  round-1 scope per the spec's own framing (only the flow actor's treatment
  is described in §2).
- Per the spec's anti-scope, no B-conditioning/ortho was added to make the
  sphere-normalized index work — if E1/E2 (evaluation, not implemented here)
  show it needs that, that is the intended finding, not a bug to patch.
- E1/E2/E3 eval scripts and the pre-registered wave launch are explicitly
  out of this task's scope (§3/§4 are evaluation/wave, not agent code); only
  the one P0 config was added per the instructions.
