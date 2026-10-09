# Spec 17a delta conversion — eval harness (`eval_zeroshot_rewards.py`)

## What was changed

`scripts/rescuing_fb/eval_zeroshot_rewards.py`:

1. New CLI flags `--delta` and `--force-delta-mismatch` (`build_parser`).
2. New pure-numpy/logic functions (placed after `gpi_pick_actions`, in a new
   "spec 17: delta-measure eval-side conversion" section):
   - `hit_indicator(row_obs, col_obs, blocks)` — `[R,C]` region-hit indicator.
     Reuses `GCIQLFlowAgent._hit_matrix` directly (imported lazily from
     `gciql_flow`, read-only — that file was not touched); does not
     reimplement the block/eps geometry.
   - `delta_to_measure(h, J, gamma)` — the identity `Mhat = (h+Jhat)/(1-gamma)`.
     Pure numpy, no jax.
   - `DeltaMismatchError` + `check_delta_consistency(ckpt_is_delta,
     requested_delta, force, run_dir)` — the refusal/warning logic, factored
     out as a standalone pure function so it's unit-testable without a real
     checkpoint.
   - `make_measure_readout(q_at_z_raw, blocks, gamma, delta,
     psi_for_decode, states_for_decode)` — wraps a raw critic readout with the
     conversion. `delta=False` returns `q_at_z_raw` **unchanged** (identity
     passthrough, no wrapping closure at all). `delta=True` decodes `z_r` to
     its nearest dataset-state proxy via the harness's own
     `z_to_goal_proxy_idx` (already used for `actor_cond='goal'` deployment),
     uses that as the region `g` for `h_g(x0)`, and returns
     `(h_g(obs) + Jhat) / (1-gamma)`.
3. `main()` wiring:
   - Right after `cfg = dict(flags["agent"])` (fail fast, before restoring
     anything): `ckpt_is_delta = bool(cfg.get("measure_delta", False))`, then
     `check_delta_consistency(...)` — raises `DeltaMismatchError` on an
     un-forced mismatch, prints `[zsr] FORCE OVERRIDE: ...` on a forced one.
   - Right after `Psi`/`match_data` are built: `gamma =
     float(cfg.get("discount", GAMMA))`; if `--delta`, `delta_blocks =
     _parse_hit_spec(str(cfg.get("measure_hit_spec") or "19:22:0.4"))`
     (checkpoint's own recorded hit spec, never a CLI default); then
     `q_at_z = make_measure_readout(q_at_z, delta_blocks, gamma, args.delta,
     psi_for_decode=Psi, states_for_decode=S)`.
   - Output JSON: `results["meta"]["measure_delta"]` (applied, ckpt flag,
     forced_mismatch, gamma, hit_spec, convention string) is added **only
     when** `args.delta or ckpt_is_delta or args.force_delta_mismatch` — i.e.
     never in the plain off/off case, so the fully-off JSON structure is
     unchanged from before this change.

## 1. Readout sites found (requirement 1)

Grepped the whole file for `q_at_z` / `agent.network.select` / critic reads.
There is exactly **one** raw critic-measure readout site in the harness:
`q_at_z` (`t6_gpi_eval.make_q_at_z`, `Q(s,a,z)=phi(s,a).z/sqrt(d)`), built at
one line and consumed at exactly one call site inside `batched_gpi_rollout`
(the `mixture` family's `--gpi` path). That wrapper is now the sole point the
conversion is applied.

Everything else in the harness (`dirac`/`region`/`dense` success, the
`linear` family, `reachavoid`) reads success from the **environment's own
`info['success']`** after an actor rollout, or from `psi(g)` state
embeddings (`make_psi`) used only for z-inference (`moment_z`/`ridge_z`/
retrieval). Neither needs conversion: psi/columns are explicitly unchanged
by `measure_delta` per spec 17a (only the TD *target* moves), and the
trained actor already argmaxes the reparameterized `J` correctly (spec 17a:
"argmax_a J = argmax_a M" — not a new objective), so rollout-based success
numbers are correct with or without the eval-side conversion.

**Notable finding**: `mixture` (hence `q_at_z`/GPI) is restricted to
cube-single by the harness's own multi-env gate (`families ['mixture'] are
cube-single-only`). So for the pre-registered cd/sc delta arms (spec 17 §6),
no family in this harness ever touches the raw critic value at all — the
conversion's only live surface today is a cube-single `--delta --gpi
--families mixture` run (e.g. the pre-registered `d17_cs_delta` arm).

## 2. Refusal logic (requirement 4)

`check_delta_consistency(ckpt_is_delta, requested_delta, force, run_dir)`:
- `ckpt_is_delta=True, requested_delta=False` → `DeltaMismatchError` unless
  `force=True` (then a `"FORCE OVERRIDE (numbers below are WRONG
  raw-J-as-M): ..."` warning string).
- `requested_delta=True, ckpt_is_delta=False` → `DeltaMismatchError` unless
  `force=True` (then `"FORCE OVERRIDE (numbers below are WRONG
  converted-raw-M): ..."`).
- Matched cases (`True/True`, `False/False`) → `[]`, no exception, `force`
  is a no-op.

Called in `main()` before the checkpoint is restored (fail fast). Verified
live: forcing `--delta` against the non-delta cube-single FP-NoOrtho
checkpoint printed the exact `FORCE OVERRIDE ...` line and proceeded; the
JSON's `meta.measure_delta` block recorded `applied=True,
ckpt_measure_delta=False, forced_mismatch=True`.

## 3. Bit-identity off-path (requirement 6)

Proof is by construction, not just testing: `make_measure_readout(...,
delta=False)` returns the raw `q_at_z_raw` object itself — no wrapping
closure, no cast — so the GPI call site is byte-for-byte the same function
as before this change. `check_delta_consistency(False, False)` returns `[]`
(no side effects). The `meta.measure_delta` JSON key is only added when
delta is "in play" in any way (`args.delta`, `ckpt_is_delta`, or
`--force-delta-mismatch`), so a plain off/off run's meta dict has the exact
same key set as before (verified live: 31 keys, no `measure_delta` key,
against 32 keys with it present on the forced-delta run).

I could not diff a literal "before" vs "after" full JSON byte-for-byte
because the file is untracked in git (no committed baseline to check out)
and, more importantly, **the harness itself is not run-to-run
deterministic even with `--delta` off and a fixed `--seed`** — two identical
off-path invocations against the same checkpoint produced different
`cos_zr_gi`/`choice_frac`/`switch_rate` values (e.g. `cos_zr_gj` 0.518 vs
0.445 across two otherwise-identical runs). This is pre-existing
nondeterminism in the harness (likely env/mujoco reset or jax CPU-BLAS
nondeterminism), unrelated to this change and out of scope to fix — flagging
it here so it isn't later mistaken for a regression from this diff.

## 4. Tests (`tests/test_eval_zeroshot_delta.py`, 10 tests, all green)

```
.venv/bin/python -m pytest tests/test_eval_zeroshot_rewards.py \
  tests/test_eval_zeroshot_delta.py tests/test_gciql_measure_delta.py \
  tests/test_gciql_measure_hit_spec.py -q
=> 45 passed in 129.84s
```
- `test_delta_to_measure_identity_tabular` — mirrors
  `tests/test_gciql_measure_delta.py::test_identity_tabular`'s exact 3-state
  chain / reward-at-next-state convention; checks the harness's own
  `delta_to_measure` recovers `M` from `(h, J)` built by power iteration.
- `test_check_delta_consistency_*` (4 tests) — both refusal directions, the
  force-warning path, matched-case no-op.
- `test_make_measure_readout_off_path_is_bit_identical_passthrough` — asserts
  `readout is stub` (object identity, not just equal output).
- `test_make_measure_readout_delta_true_requires_decode_context` — missing
  psi/states raises `AssertionError`.
- `test_hit_indicator_matches_hit_matrix_bit_exact` (jax subprocess) —
  `hit_indicator` bit-exact vs a direct `GCIQLFlowAgent._hit_matrix` call.
- `test_make_measure_readout_delta_true_applies_conversion` (jax subprocess)
  — a stub `q_at_z_raw` returns known `J`; the wrapped readout decodes `z_r`
  via the real `z_to_goal_proxy_idx`, computes `h` via the real
  `_hit_matrix`, and the result matches `delta_to_measure(hit_indicator(...),
  J, gamma)` exactly, spot-checked against the closed-form numbers.

## 5. What could not be verified end-to-end (no delta checkpoint exists yet)

Ran two live smoke checks (CPU-only, `taskset -c 40-47`,
`JAX_PLATFORMS=cpu`, `/dev/shm/.venv-jax/bin/python`, `--episodes 2
--horizon 25`) against
`exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/
m_firstpassage_noortho_s0` (cube-single, non-delta, `actor_cond=goal`,
`discount=0.99`), `--families mixture --gpi`:
- off-path: ran clean, `meta` has no `measure_delta` key.
- `--delta --force-delta-mismatch`: ran clean, printed the FORCE warning,
  `meta.measure_delta` populated correctly (`gamma=0.99, hit_spec=
  "19:22:0.4"`), `make_measure_readout`'s delta branch executed against the
  real restored agent/`Psi`/`S` without error.

The suggested `exp_spec08/.../BallVisit-Hindsight-d50/p2h_cd_s0` checkpoint
(`actor_cond=goal`, multi-env cube-double) was also restored successfully,
but the harness's own multi-env family gate refuses `mixture` there
(`families ['mixture'] are cube-single-only`) — so it cannot exercise
`q_at_z`/GPI at all, which is consistent with finding 1 above (cd/sc runs
never touch the raw critic readout regardless of delta).

Not verified (impossible without a real delta checkpoint, none exist yet —
first save at 100k, wave at ~30k): that a REAL delta-trained critic's raw
`Jhat` output, run through `make_measure_readout`, actually reproduces the
region's true occupancy `M` (the unit tests substitute a synthetic
`q_at_z_raw` with known `J` values; correctness of the *conversion itself*
against real trained numbers is therefore proven only by the tabular
identity + bit-exact `_hit_matrix` reuse, not by a real-critic sanity check).

## Files changed

- `scripts/rescuing_fb/eval_zeroshot_rewards.py` (edited; `gciql_flow.py`
  not touched, only imported read-only for `GCIQLFlowAgent._hit_matrix` /
  `_parse_hit_spec`)
- `tests/test_eval_zeroshot_delta.py` (new)

## Addendum — exact-weighted h correctness fix (code review round 2)

Code review flagged that the original proxy-decode `h` (nearest dataset
state to `z_r`) was an unnecessary approximation: since `z_r = sum_j w_j
psi(g_j)` and `q_at_z` is linear in `z_r`, the raw critic readout at `z_r`
is automatically `sum_j w_j Jhat(s,a,g_j)` — no approximation needed there —
but the h-term must be the SAME weighted sum `sum_j w_j h_{g_j}(x0)`, not
one proxy state's `h`.

**Fix**: `compute_z_weights(mode, rewards, R_vals, Psi, match_data, k,
ridge_lambda)` reproduces the exact weight vector `w` (full-length, aligned
to the relabel table `S`) each z-inference mode actually used to build the
deployed `z_r`:
- `moment_weights` — `w_j = rewards_j*(R_j or 1)/n`, mirrors `moment_z` +
  renormalize (pinned bit-compatible with `infer_z`/`renewal_head.infer_z`
  per that code's own contract — not independently verifiable without the
  external module, flagged as a trust assumption).
- `retrieval_weights` — `1/len(idx)` on the top-k selected indices, mirrors
  `infer_cell`'s retrieval branch exactly.
- `ridge_weights_exact` — via the push-through identity
  `(A^TA+muI)^-1A^T = A^T(AA^T+muI)^-1`: reduces ridge's `d×d` solve to an
  **`n×n`** solve (`n` = relabel size). Gated by `RIDGE_EXACT_MAX_N=2000`
  (chosen for a sub-few-second CPU solve); above that, falls back to the
  original nearest-state proxy decode, ONLY for ridge. `n=1500` (exact) and
  `n=3000` (fallback) both verified live against the real cube-single
  FP-NoOrtho checkpoint — see logs below.

`make_measure_readout(..., S=, w=, psi_for_decode=)` now takes the
precomputed `w`; `w=None` selects the proxy fallback (ridge-only).
`w`/`h_construction` are computed **per mixture cell** (in `main()`, right
after `infer_cell`), since `r` — hence `w` — differs per cell; the wrapper
itself is still built fresh per cell (cheap: a closure + one `hit_indicator`
matmul per GPI step). Off-path (`--delta` absent) is untouched: `q_at_z_raw`
is used directly, no wrapper ever constructed. Output JSON gained
`meta.measure_delta.h_construction`/`ridge_exact_max_n` (run-level) and
`delta_h_construction` per mixture cell.

**On the "ranking across candidate z's" question**: I built the requested
case (`test_ranking_across_candidate_zs_flips_between_exact_and_proxy`) —
two composite tasks `z_A`, `z_B` (50/50 mixtures) where exact h ranks
`Mhat(A)=0.5/(1-gamma) > Mhat(B)=0`, but the proxy decode, steered onto a
decoy state carrying zero weight in either mixture, ranks `Mhat(B) >
Mhat(A)` — a genuine flip in predicted `Mhat` across two candidate `z`s.
However I also verified, and must report precisely: **within one
`gpi_select` call** (the harness's *actual* GPI action-selection mechanism),
`z_r` is held fixed for the whole comparison and `h_g(x0)` is therefore a
constant added to every candidate — `1/(1-gamma)` positive and `h` additive
both drop out of `argmax`, exactly as I found before this round's fix
(`test_gpi_select_argmax_is_invariant_to_a_per_env_additive_constant`,
pure algebra, holds regardless of exact vs proxy). So: the fix is strictly
required and correct for the `Mhat` **value** (any cross-cell comparison, or
any future direct quoting per spec 17 §6's L1fix/L2 use), but it is
provably inert for today's specific within-GPI-call action argmax — I'm
reporting this precisely rather than eliding it, per your own framing that
"inert" was an acceptable, informative finding if true.

**Tests**: 18 tests in `tests/test_eval_zeroshot_delta.py` (was 10),
covering `moment_weights`/`retrieval_weights`/`ridge_weights_exact`
reconstruction of the deployed `z` exactly, `compute_z_weights` dispatch +
ridge-threshold fallback, the exact-weighted and proxy-fallback readout
paths, the ranking-flip case, and the GPI-invariance companion. Full suite:
`.venv/bin/python -m pytest tests/test_eval_zeroshot_rewards.py
tests/test_eval_zeroshot_delta.py tests/test_gciql_measure_delta.py
tests/test_gciql_measure_hit_spec.py -q` → **53 passed**. Live smoke
(cube-single FP-NoOrtho checkpoint, `--delta --force-delta-mismatch
--families mixture --gpi`): moment mode, ridge at `n=1500`
(`exact_weighted`), and ridge at `n=3000` (`proxy_decode` fallback) all ran
clean end-to-end against the real restored agent/`Psi`/`S`.
