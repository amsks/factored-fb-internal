# Pixel-FB + GCIQL-pixel Integration — Design

- **Date:** 2026-05-19
- **Status:** Approved design (pre-implementation)
- **Scope decision:** FB-pixel port is the focus; GCIQL-pixel gets a runner script only.
- **Done bar:** Code + unit tests only. No training/smoke runs this effort.
- **Target env:** `visual-cube-single-play-v0` (mirrors the existing GCIQL-pixel config for direct FB-vs-GCIQL comparison).
- **Approach:** A — Faithful vertical port from td_jepa `metamotivo`.

## Background

`nn_models.py` in this repo is a refactor of `td_jepa/metamotivo/nn_models.py`. The pixel
modules (`DrQEncoder`, `Augmentator`) were explicitly removed ("add back when needed").
The FB model/agent already build and call encoder/augmentator slots
(`agents/fb/model.py` builds `_fw_encoder`/`_bw_encoder`/`_augmentator` from
`rgb_encoder_cfg`/`augmentator_cfg`/`obs_normalizer_cfg`; `agents/fb/agent.py:172-182`
already calls them in `aug()`/`enc()`), defaulting to `Identity*`. `RGBNormalizerConfig`
and `PixelWrapper` are already present. So the wiring exists; what is missing is the two
nn modules, the pixel data/frame-stack path, and a pixel domain config.

Source of truth for the faithful port:
`/home/adityamohan/git/baselines/td_jepa/metamotivo/` and
`/home/adityamohan/git/baselines/td_jepa/scripts/train/pixel/launch_fb_ogbench.py`.

## Gap analysis

| Piece | Repo status | td_jepa source |
|---|---|---|
| `DrQEncoder` + `DrQEncoderArchiConfig` | missing | `metamotivo/nn_models.py:329-376` |
| `Augmentator` + `AugmentatorArchiConfig` | missing | `metamotivo/nn_models.py:379-409` |
| `RGBNormalizerConfig` | present (`normalizers.py`) | — |
| `PixelWrapper` (CHW, frame-stack) | present (`envs/wrappers.py`) | — |
| FB encoder/augmentator slots | wired (`agents/fb/model.py`, `agent.py:172-182`) | — |
| Pixel offline data loading | missing (`data/ogbench.py` state-only) | `metamotivo/data_loading/base.py:28-79` |
| Frame-stacking in buffer | missing (`buffers/transition.py` `DictBuffer`) | `metamotivo/buffers/parallel.py:73-89` |
| `configs/domain/visual_cube_single.yaml` | missing | `scripts/train/pixel/launch_fb_ogbench.py` |

## Components

1. **`nn_models.py`** — port verbatim from `td_jepa/metamotivo/nn_models.py:329-409`:
   `DrQEncoder` + `DrQEncoderArchiConfig` (`name="drq"`, `feature_dim`), `Augmentator` +
   `AugmentatorArchiConfig` (`name="random_shifts"`, `pad`). Register both in the
   selectable archi config unions so configs can pick them by discriminated `name`.
   No external deps (pure PyTorch).

2. **`train.py` `make_agent`** — map the Hydra `rgb_encoder` / `augmentator` /
   `obs_normalizer` config blocks into the existing
   `rgb_encoder_cfg` / `augmentator_cfg` / `obs_normalizer_cfg` kwargs of the FB model.
   No `agents/fb/model.py` logic change (slots already build/call them).

3. **`data/ogbench.py`** — add an `obs_type`-gated pixel branch to
   `load_transitions` / `load_ogbench_dataset`: read the `pixels` key, keep `uint8`,
   preserve HWC, no float cast. State path unchanged.

4. **`buffers/transition.py`** — add frame-stacking on sample, faithful to
   `metamotivo/buffers/parallel.py:73-89` (`frame_stack=3`, concat oldest→newest on
   channel axis, episode-boundary clamping via the same offset math). Active only when
   `obs_type=="pixels"`; `frame_stack=1`/state path unchanged.

5. **`configs/domain/visual_cube_single.yaml`** — mirror
   `launch_fb_ogbench.py`: `obs_type: pixels`, `frame_stack: 3`, `fb_flowbc` agent,
   `RGBNormalizerConfig`, `rgb_encoder: drq` `feature_dim=256`,
   `augmentator: random_shifts` `pad=2`, `L_dim=256`, `z_dim=50`, `batch_size=256`,
   `render_height=render_width=64`.

6. **GCIQL-pixel runner** — `scripts/run_gciql_visual*.sh` reusing the existing
   `configs/gciql/cube_single_visual.yaml` (analogous to the state sweep script).
   Delivered + smoke-parse tested only; no training run executed.

## Data flow (faithful to td_jepa)

```
OGBench visual env            → obs (H,W,C) uint8 [0,255]
  PixelWrapper                → (C·frame_stack, H, W) uint8, CHW   [already present]
data/ogbench.py (pixel branch)→ load "pixels" key, uint8, HWC, no float cast
DictBuffer.sample (FS branch) → gather t-2,t-1,t, concat channel → (B, C·3, H, W) uint8
agents/fb/agent.py (unchanged):
  _obs_normalizer = RGBNorm   → (x/255)-0.5, float
  aug()  = _augmentator       → random_shifts, pad=2
  enc()  = _fw/_bw_encoder    → DrQEncoder → (B, 256)
  _left_encoder → FB forward/backward maps (unchanged)
```

## Config plumbing

- `obs_type` (`state`→`pixels`) and new `frame_stack` flow to: env creation
  (`OGBenchEnvConfig` already applies `PixelWrapper` for `pixels`), the data loader,
  buffer sampling, and `make_agent` module selection.
- Selection is by discriminated `name` (`"drq"`, `"random_shifts"`,
  `"RGBNormalizerConfig"`) — the same Hydra/pydantic pattern already used for
  forward/backward/actor archis. No new selection machinery.
- State path unaffected: defaults stay `Identity*`; pixel modules activate only when
  the domain config names them.

## Faithfulness invariants

- Ported module bodies byte-identical to td_jepa where ported (manual diff during
  implementation, documented).
- CHW channel-first; `uint8` retained until `RGBNorm`.
- `frame_stack=3` concat order = oldest→newest on the channel axis.
- Encoder construction order preserved: `_bw_encoder`, `_augmentator`, `_fw_encoder`.

## Testing (new, under `tests/`, CPU, fast, no training)

1. `test_drq_encoder.py` — fake `Box((9,64,64))`: forward `(B,256)` with
   `feature_dim=256`; `output_space.shape==(256,)`; gradient flows; non-3D obs_space
   assertion fires.
2. `test_augmentator.py` — `pad=2`: output shape == input `(B,9,64,64)`;
   dtype/device preserved; non-square assertion fires; output differs from input but
   finite.
3. `test_pixel_data_loading.py` — synthetic npz with `pixels` key: loader returns
   uint8 HWC, correct N; state path still float32 (regression guard).
4. `test_frame_stack_buffer.py` — `frame_stack=3`: sampled obs has `C·3` channels,
   oldest→newest order, episode-boundary clamping matches td_jepa offsets;
   `frame_stack=1`/state path unchanged.
5. `test_visual_cube_config.py` — Hydra-compose `domain=visual_cube_single`, build FB
   model: `_fw_encoder` is `DrQEncoder`, `_augmentator` is `Augmentator`,
   `_obs_normalizer` is `RGBNorm`; state config still builds `Identity*` (regression).

Plus a documented manual faithfulness diff of the ported `DrQEncoder`/`Augmentator`
bodies vs td_jepa source.

## Error handling

- Clear assertion messages: non-3D obs into DrQ/Augmentator; non-square images;
  `obs_type=pixels` but dataset lacks `pixels` key; `frame_stack` unset for a pixel
  domain.
- State path provably untouched — every pixel branch gated on `obs_type` / config
  `name`; enforced by regression assertions in tests 3–5.

## Risks & mitigations

- **Image size:** td_jepa OGBench pixel uses 64×64; config asserts
  `render_height == render_width == 64`.
- **Buffer divergence:** repo `DictBuffer` ≠ td_jepa `ParallelBuffer`; we port only the
  frame-stack sampling math, not the parallel-fetch machinery — intentional deviation,
  behaviorally equivalent for offline data, documented.
- **No training validation this effort:** correctness rests on unit tests + the
  faithfulness diff; a smoke/validation run on `visual-cube-single-play-v0` is explicit
  out-of-scope and the recommended immediate follow-up.

## Non-goals

- No training or smoke runs (deferred follow-up).
- No DMC pixel support; OGBench `visual-cube-single-play-v0` only.
- No `visual-cube-double` or other pixel domains this effort.
- No changes to GCIQL beyond a runner script (its visual config already works).
- No port of td_jepa's `ParallelBuffer` fetch machinery.

## Recommended follow-up (out of scope here)

1. Smoke run (~50k steps) FB-pixel on `visual-cube-single-play-v0`.
2. GCIQL-pixel validation run via the new runner.
3. Full FB-pixel vs GCIQL-pixel comparison sweep.
