# Spec 19b — ORS extractor implementation report

STATUS: GREEN. Branch `gciql-fb-bridge`, tree dirty, no commits. No training launched.
No 19a checkpoint exists — all runs below use a SYNTHETIC `measure_density_behavior`
checkpoint built in-process; this proves the machinery, not a real 19a checkpoint's r_W.

## Verification (foreground, CPU-only, just re-run)
1. `.venv/bin/python -m pytest tests/test_gciql_ors_extractor.py -q` → **7 passed, 84s**.
2. `.venv/bin/python -m pytest tests/test_gciql_measure_density.py tests/test_gciql_measure_density_behavior.py tests/test_gciql_measure_delta.py tests/test_gciql_measure_delta_partition.py -q` → **38 passed, 401s**, INCLUDING `test_default_off_bit_identical` in both density files (previously red — HEAD had drifted past their `git show HEAD` assumption). The repair skips ONLY if the pinned baseline SHA (`370938f` pre-spec18, `74e861d` pre-spec19a) is unreachable (shallow clone etc.); both were reachable here and the real diff ran and passed — not a skip.
3. `scratchpad/verify_configs.py`, stubbed synthetic ckpt+pool per env (P1.1 hasn't trained) → all 4 new configs construct an agent cleanly, no lattice trip: `cube_single_state_bconly_d50.yaml`, `{cube_single,cube_double,scene}_state_ors_d50.yaml`.

## Three confirmations
- **`ors_reward_scale`**: cube-double **4.0** (=1/0.25, VERIFIED — Table 10's cube-double-play row); scene **1.333333** (=1/0.75, VERIFIED — Table 10's scene-play row); cube-single **4.0** (INFERRED — no Table 10 row exists for cs; borrowed cube-double/-triple's divisor by object-family analogy, flagged in the config header). None uses the paper's released code's single hardcoded `0.75×` multiplier (`0.75*r`, a different arithmetic direction than `r/x` that only numerically coincides with scene's own row).
- **Restore safety**: `ors_pool`/`ors_pool_b_emb`/`ors_tau` are `nonpytree_field()`, same as `delta_protos`. `scratchpad/verify_restore.py` (just ran, PASSED): a pre-ORS agent's `to_state_dict()` top-level keys are `['network', 'rng']` only; saved via the real `save_agent`, restored via the real `restore_agent` onto a current ORS-aware `GCIQLFlowAgent` — param tree byte-identical, `ors_pool`/`ors_pool_b_emb`/`ors_tau` correctly `None`. An ORS-on agent's own state dict also excludes those 3 fields.
- **Reward convention** (`gciql_flow.py:915-916`): `success = batch['masks'] == 0.0; rewards = jnp.where(success, 1.0, r_w)` — success-index rows forced to `1.0`; every other row gets r_W; `batch['rewards']` is never referenced again on that path. `test_reward_substitution` pins this with NaN-poisoned rewards (ORS-on stays finite; an ORS-off control on the same batch goes NaN).

## Concerns
- `ors_reward_ckpt`/`ors_pool_path` in the 3 ORS configs are placeholders (P1.1 hasn't run yet) — filled at launch time, matching `warm_start_ckpt`'s repo-wide convention (never baked into a static config for a not-yet-existing checkpoint).
- Added one lattice refusal beyond the literal spec text: `value_arch='occupancy'` computes its own reward and returns before the ORS substitution is reached — would be silently inert without the refusal (17b fake-NULL hazard).
- Piece 2 (`actor_bc_only`) tested separately: `tests/test_gciql_actor_bc_only.py`, 3 passed.
