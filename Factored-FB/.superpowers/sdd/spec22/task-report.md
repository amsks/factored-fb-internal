# Spec 22 — task report

STATUS: DONE. All code/tests written, all suites green, 5 configs construct cleanly.

## Files changed
- `tools/wandb_mode_shim/gciql_flow.py`: `_density_actor_q` helper; flow branch
  refactored to call it; explicit `ddpgbc`/`awr_flow` density branches in
  `actor_loss` (no delegation); lattice relaxed to `actor_kind in ('flow',
  'ddpgbc', 'awr_flow')` with the `actor_q_center` flip for awr; **beyond the
  literal spec text**: fixed `density_loss`'s bootstrap action `a_plus`, which
  hard-coded the flow actor's noise-call signature and would break under
  ddpgbc (GCActor returns a distribution, not an array) and KeyError under
  awr_flow (no one-shot `actor` module registered) — added matching branches
  there (ddpgbc mirrors `measure_loss`'s own pattern; awr_flow Euler-rolls
  `actor_vf`). Flow branch of both `actor_loss` and `density_loss` untouched.
- `tests/test_gciql_density_actors.py` (new, 10 tests).
- 5 new configs in `configs/gciql/`: `antmaze_medium_state_measure_density_d50{,_ddpgbc,_awrflow}.yaml`, `cube_single_state_measure_density_d50_{ddpgbc,awrflow}.yaml`.

## Test summary (all run in foreground, JAX_PLATFORMS=cpu)
- `tests/test_gciql_density_actors.py -v`: **10/10 passed** (125.99s).
- Neighbor suite `test_gciql_measure_density{,_behavior}.py test_gciql_measure_delta{,_partition}.py test_gciql_ors_extractor.py -q`: **45/45 passed** (510.78s).
- 5 configs → `GCIQLFlowAgent.create()`: all 5 construct without tripping the lattice (verified via ad hoc script mirroring `train.py`'s config merge).

## Flow-path bit-identity (§5.1)
Tightened past `np.allclose`: `test_flow_path_bit_identical_after_refactor`
asserts `float(l_base) == float(l1)` **and** the uint32 bit-pattern view of the
loss and of `actor/q_mean`, `actor/bc_flow_loss`, `actor/distill_loss`,
`measure/density_ce`, `measure/density_ess` against the pinned pre-spec-22 SHA
`0332ba0`. Verified independently outside pytest too:
```
loss EXACT equal: True 8.167612075805664 8.167612075805664
loss bit pattern equal: True
actor/q_mean EXACT equal: True 0.0007244835142046213 ...
measure/density_ce EXACT equal: True 5.7385711669921875 ...
param tree EXACT equal (all leaves): True
```
`test_stock_kinds_bit_identical_without_density` applies the same exact/
bit-pattern check to plain ddpgbc/awr_flow (density off) against the same SHA.

## The delegation question — direct evidence
Initial version of `test_ddpgbc_density_never_reads_critic` perturbed
`grad_params['modules_critic']` — **a null test**: neither the shipped density
branch nor the stock delegated `GCIQLAgent.actor_loss` ever reads the critic via
`params=grad_params` (both call `network.select('critic')(...)` with no
`params=`, which always resolves to `self.network.params`). That test would
have passed identically whether or not delegation were active. I caught and
fixed this. The rewritten test:
1. Perturbs the critic **as stored on the agent** (`network.replace(params=...)`,
   +1e3), then recomputes `actor_loss` → loss/`q_mean`/`q_loss` unchanged to the
   **exact float and bit pattern**.
2. Counterfactual: calls the stock `GCIQLAgent.actor_loss(a, ...)` directly on
   both the unperturbed and perturbed agent — asserts these **do** differ
   (proves the perturbation is live and would have shown up under delegation).
3. Asserts the shipped branch's `q_mean` differs from the delegated `q_mean`.
Combined with `test_ddpgbc_density_numpy_reference` (rebuilds q from direct
`density_f`/`density_b` calls, atol 1e-5 match) and `test_ddpgbc_density_grad_hygiene`
(`jax.grad` ≈0 over `modules_critic`/`modules_value`/density towers, >0 over
`modules_actor`), this rules out the 17b fake-NULL species by construction, not
convention. Mirror tests exist for `awr_flow` (numpy reference + grad hygiene,
no delegation path to counterfactual against since awr_flow never delegated).

## Grad hygiene / lattice confirmation
- No `params=grad_params` on any density-tower call in `_density_actor_q` or the
  new branches (grep-verified); `kg` wrapped in `jax.lax.stop_gradient`.
- `actor_q_center=True` required for ddpgbc (refused if False); required False
  for awr_flow (refused if True) — `test_lattice_spec22`.

## Concerns
- The `density_loss` bootstrap-action fix was necessary for ddpgbc/awr_flow to
  even run under `measure_density` but isn't in the spec's literal §4 snippet —
  flagging it explicitly since it's outside the given contract text, though
  required for the wave to be launchable.
