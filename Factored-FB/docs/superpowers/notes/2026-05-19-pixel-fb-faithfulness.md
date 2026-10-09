# Pixel-FB Faithfulness Notes (2026-05-19)

## Byte-identical ports (from td_jepa/metamotivo/nn_models.py:329-409)
- `DrQEncoderArchiConfig`, `DrQEncoder`
- `AugmentatorArchiConfig`, `Augmentator`
Verified by an AST per-class source-segment byte comparison
(`ast.get_source_segment`) against td_jepa `metamotivo/nn_models.py`: all four
classes byte-identical, exact byte counts match (246/1547/191/1153 B). Note: a
naive `inspect.getsource`-concatenation check yields a false positive because it
drops the PEP 8 blank lines *between* top-level classes — those blank lines are
the only difference and carry no code; the per-class AST compare is the
authoritative check.

## Intentional, documented deviations
- **Buffer:** repo `DictBuffer` != td_jepa `ParallelBuffer`. We port only the
  frame-stack offset/clamp math (`np.maximum(-i, -timesteps)` ->
  `np.minimum(k, timestep)`); not the parallel prefetch machinery.
  Behaviorally equivalent for offline data: the stacked observation for
  transition t is the deterministic sequence [t-(fs-1) .. t] clamped at the
  episode start.
- **`next.observation`:** reconstructed as "drop oldest of s, append stored
  next frame", matching td_jepa's `obs[1:1+frame_stack]` slice semantics
  without storing a redundant timestep array. Equivalence rests on the data
  loader invariant `next/observation[i] == observation[i+1]` within an episode
  (observation = obs_full[:-1], next/observation = obs_full[1:]).
- **Pixel orientation (CHW vs HWC):** td_jepa's `extract_all.py` stores pixels
  **CHW** (`np.moveaxis(observations, -1, 1)`) and its loader does NOT
  re-transpose. This repo instead stores pixels **HWC** at extraction
  (`scripts/extract_ogbench.py` visual branch, no moveaxis) and converts
  HWC->CHW inside `data/ogbench.py::load_transitions`
  (`np.moveaxis(ep["pixels"], -1, 1)`). The orientation reaching `DrQEncoder`
  (`(N, C, H, W)` after frame-stack -> 9 channels) is therefore identical to
  td_jepa; only the on-disk layout and the location of the single transpose
  differ. This is an intentional, documented deviation chosen so the committed
  loader/buffer/config/env wiring is left untouched.
- **State episodes:** td_jepa also writes an all-zero dummy `pixels` key for
  state envs; this repo does NOT (state loader path never reads `pixels`),
  keeping state episodes compact and byte-compatible with prior runs.
- **`terminated`:** kept all-False per the existing repo convention (offline GC
  data treated as non-terminating), unchanged from the state path.
- **Agent-layer cfg threading (Task 4 correction):** `rgb_encoder_cfg` /
  `augmentator_cfg` are threaded through `FBAgent.__init__` -> `_make_model`
  (mirroring `obs_normalizer_cfg`); `FBFlowBCAgent` forwards via `**fb_kwargs`.
  The original plan's "no agent logic change" assumption was wrong (the FB
  *model* accepted the cfgs but the *agent* layer did not); see the plan's
  "Plan correction (Task 4)" note.

## Verification status (code + tests only; no training/smoke runs this effort)
- Pixel-FB unit/integration tests (Tasks 1-6): all green.
- Pre-existing repo test failures in unrelated analysis/plotting modules
  (missing `pandas` etc.) are out of scope and not regressions.

## Out of scope (recommended follow-up)
- Smoke run (~50k steps) FB-pixel on visual-cube-single-play-v0.
- GCIQL-pixel validation run via scripts/run_gciql_visual.sh.
- FB-pixel vs GCIQL-pixel comparison sweep.
