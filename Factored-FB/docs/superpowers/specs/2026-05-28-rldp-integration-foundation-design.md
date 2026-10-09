# Design: RLDP integration — sub-project 1 (Foundation)

**Date:** 2026-05-28
**Scope:** `nn_models.py`, the `DictBuffer` sampling API, and a new `tests/parity/` harness.
**Audience:** anyone implementing the foundation that sub-projects 2 (RLDP + RLDP+FlowBC on state) and 3 (pixel variants) will depend on.

## Context

RLDP (the ICLR-submitted Forward-Backward variant) differs from plain FB in four places: a parallel/horizon buffer, a dynamics-aware predictor head, an extra `_bw_encoder` path, and dramatically lower B-orthonormality regularization. The reference implementation we will port from is **td_jepa's `metamotivo/agents/rldp/`** (`/home/adityamohan/git/baselines/td_jepa/`) — the same upstream codebase our existing FB agent was ported from, so the port recipe is established. The full integration is decomposed into three sequential sub-projects:

1. **Foundation (this spec).** Infrastructure only — no RLDP agent yet.
2. **RLDP + RLDP+FlowBC on state.** New agent code, configs, launchers; depends on (1).
3. **Pixel variants of both.** Visual encoder integration; depends on (2).

This spec covers only (1). It must not add any RLDP agent code, configs, or launchers.

## Problem

Three pieces of infrastructure block sub-projects 2 and 3:

- `VForwardArchiConfig` / `VForwardMap` (RLDP's predictor head module) were **explicitly removed** from our `nn_models.py` when we ported FB ("Removed VForwardMap, VectorField (not used in FB baseline)" — line 6 of `nn_models.py`). They need to come back.
- Our `DictBuffer.sample()` returns single transitions. RLDP needs **horizon-h windows** (td_jepa launches RLDP with `"buffer_type": "parallel"`, `"horizon": 5`).
- The port rigor we agreed on (literal port + parity gate tests) needs a **reusable harness** so sub-projects 2 and 3 don't each re-invent the weight-transplant + numerical-match scaffolding.

## Design

### Components

#### 1. `nn_models.py` — restore `VForwardMap`

- Re-add `VForwardArchiConfig` (dataclass) and `VForwardMap` (`nn.Module`), ported verbatim from `/home/adityamohan/git/baselines/td_jepa/metamotivo/nn_models.py` lines ~280–310. Match the existing td_jepa builder pattern (`cfg.build(...)` returns the module).
- Update the "Removed VForwardMap, VectorField" comment at the top of `nn_models.py` (around line 6) to reflect that `VForwardMap` is back (and `VectorField` remains removed if RLDP doesn't need it — confirm by grep).
- **No** changes to any other class in `nn_models.py`. FB's existing modules stay byte-identical.
- Expected size: ~30 lines added.

#### 2. `buffers/transition.py` (where `DictBuffer` lives) — add `horizon` kwarg to `sample()`

- Add `horizon: int = 1` kwarg to `DictBuffer.sample()`.
- **`horizon == 1`**: behavior **exactly unchanged** — same code path as today. Sub-project 1 must not perturb any FB behavior.
- **`horizon > 1`**: returns a dict where each value has shape `[batch, horizon, ...]` instead of `[batch, ...]`. Windows are **contiguous within a single episode** — sampling must not cross episode boundaries.
- Fast-fail with a clear error if any episode is shorter than the requested horizon (or skip-and-resample those episodes — see Open Questions).
- Expected size: ~40 lines added.

#### 3. `tests/parity/` — reusable weight-transplant + numerical-match harness

New package with three modules:

- `tests/parity/__init__.py` — empty package marker.
- `tests/parity/harness.py` — the helpers:
  - `load_jax_checkpoint(path: str) -> dict[str, np.ndarray]` — loads a jax/flax param tree from disk and returns it flattened to a leaf-name → numpy-array dict.
  - `transplant(src_params: dict, torch_module: nn.Module, mapping: dict[str, str]) -> None` — copies values from `src_params` into `torch_module.state_dict()` using a caller-supplied `mapping` from torch name → src name. Handles shape transposition for jax `(in, out)` vs torch `(out, in)` linear layers. Asserts every torch param was assigned.
  - `assert_forward_match(ref_fn, ours_fn, inputs, atol: float, rtol: float, label: str = "") -> None` — calls both functions on `inputs`, asserts `torch.allclose` within tolerances; on failure, prints the max abs diff + position.
  - `assert_grad_match(ref_module, ours_module, loss_fn, inputs, atol, rtol) -> None` — runs one optimizer step on each and compares post-step weights.

#### Tests shipped in this sub-project

- `tests/test_nn_models_vforward.py` — instantiate `VForwardMap`, check output shape, run forward + backward on a small random batch.
- `tests/test_dict_buffer_horizon.py`:
  - `test_horizon_1_unchanged` — same batch as the existing test path, exact bitwise equality of returned dict (regression gate).
  - `test_horizon_5_shape` — returns `[B, 5, ...]` for each key.
  - `test_horizon_windows_contiguous` — windows sampled from a synthetic 2-episode buffer never cross the episode boundary.
  - `test_horizon_too_long_episode_handled` — appropriate behavior on short episodes (per Open Questions resolution).
- `tests/parity/test_harness_helpers.py` — unit-tests for `transplant` and `assert_*_match` on a synthetic toy `nn.Linear` pair, including the transposition handling.

### Data flow

The `horizon` kwarg lives in the buffer; nothing else changes. Existing call sites (`agent.update(...)` for FB) pass no `horizon` arg, hit the default `horizon=1`, and behave exactly as before. The parity harness is a passive library — only test code consumes it.

### Conventions

- Per `memory/factored-fb-test-env.md`: tests run under `.venv/bin/python -m pytest` (uv broken).
- Per `memory/commit-no-claude-coauthor.md`: commits omit the Claude co-author trailer.
- Per the launcher conventions established in the prior PR: any new file follows the existing layout (no new top-level directories beyond `tests/parity/`).

## Acceptance criteria

1. All four new tests pass: `.venv/bin/python -m pytest tests/test_nn_models_vforward.py tests/test_dict_buffer_horizon.py tests/parity/ -v`.
2. The full existing test suite still passes — no regression. `.venv/bin/python -m pytest tests/ -v` clean (modulo the known pandas-missing skips per `memory/factored-fb-test-env.md`).
3. **FB regression smoke:** `DRY_RUN=1 bash scripts/agents/fb/state.sh` still works (asserts via existing `tests/scripts/test_agent_launchers.py`). One short FB training smoke (5000 steps) shows training loss curves indistinguishable from a pre-change baseline (within RNG variance).
4. `git diff main` touches only: `nn_models.py`, the buffer file, `tests/test_nn_models_vforward.py`, `tests/test_dict_buffer_horizon.py`, `tests/parity/`. No changes to `agents/`, `configs/`, `scripts/`, `train.py`.

## Out of scope

- **Any** RLDP agent / model / config / launcher (sub-projects 2 and 3).
- Pixel-specific encoder work (sub-project 3).
- Changes to `agents/fb/*` — FB stays bit-identical, asserted by acceptance criterion 3.
- Reimplementing `VectorField` (kept removed unless a downstream sub-project demands it).
- Modifying `train.py` or any launcher — the buffer change is purely additive.

## Open risks / questions

- **Short episodes vs horizon=5**: if the OGBench cube buffer contains episodes shorter than 5 steps (unlikely but possible), do we (a) raise a clear error and let the user fix the dataset, or (b) silently re-sample those episodes until we get full-horizon windows? Decision deferred to implementation — defaulting to (a) "raise" is safer; (b) can be added if it turns out to be needed in practice.
- **JAX checkpoint format**: td_jepa saves with `safetensors`. The parity harness assumes safetensors as the canonical jax-side format. If we later need to load native `flax.serialization.to_bytes` checkpoints we extend `load_jax_checkpoint` then; sub-project 1 only needs the safetensors path because that's what td_jepa emits.
- **`assert_grad_match` precision**: training-dynamics parity gates that we ran for FB used `<1e-6` in float64. The harness should support specifying dtype + precision per call so sub-projects 2/3 can use the same gates we did for FB.

## References

- td_jepa source: `/home/adityamohan/git/baselines/td_jepa/metamotivo/`
  - `nn_models.py` lines ~280–310 (`VForwardArchiConfig`, `VForwardMap`)
  - `agents/rldp/{agent.py, model.py}` (the target of sub-project 2)
  - `agents/rldp/flow_bc/` (the target of sub-project 2 — RLDP+FlowBC variant)
  - `scripts/train/proprio/launch_rldp_ogbench.py` (sweep recipe sub-project 2 will mirror)
- Our existing FB port (the recipe to follow):
  - `agents/fb/{agent.py, model.py}` (573 + 168 lines)
  - `agents/fb/flow_bc/{agent.py, model.py}`
- Our FB parity gate precedent: commits `0943a10` (Gate 1 forward parity) and `9b43d2c` (Gate 2 training-dynamics parity).
