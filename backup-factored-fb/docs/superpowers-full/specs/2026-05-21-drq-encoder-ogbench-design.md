# DrQ encoder (JAX) for OGBench GCIQL/GCIVL — design

**Date:** 2026-05-21
**Status:** approved (pending written-spec review)

## Motivation

The pixel baselines (GCIQL, GCIVL, run via vendored OGBench in JAX) use the
`impala_small` encoder, while our FB runs use a DrQ-v2 encoder. The visual
front-end therefore differs between FB and the baselines along **four** axes,
confounding any FB-vs-baseline comparison:

| aspect | FB (`configs/domain/visual_cube_single.yaml`) | OGBench GCIQL/GCIVL (impala) |
|--------|-----------------------------------------------|------------------------------|
| encoder | DrQ-v2 (4× conv-32, k3, s2/1/1/1) + Linear→LN→Tanh→256 | `impala_small` (ResNet stacks) |
| augmentation | random_shifts, pad=2 (bilinear) | random_crop, pad=3, p_aug=0.5 (integer) |
| frame_stack | 3 | 1 (none) |
| normalization | `RGBNorm`: `x/255 − 0.5` | `x/255` (inside impala) |

**Goal:** give GCIQL and GCIVL a DrQ-based visual front-end that matches FB's
entire pixel pipeline (encoder + augmentation + frame stacking + normalization),
so the only remaining difference vs FB is the RL algorithm. The existing impala
runs are **kept** as the OGBench-default baseline; the DrQ-front-end runs are
added alongside.

## Hard constraint

Vendored OGBench (`third_party/ogbench`, pinned SHA
`1d4140997f60c52c6fb0702ec100dc988b18c548`) must **never be edited**. All
behavior changes are injected at OGBench-subprocess startup via the existing
`sitecustomize` shim that `run_gciql.py` already places on the child PYTHONPATH
(`tools/wandb_mode_shim/`).

## Architecture

### Component 1 — JAX DrQ encoder (`tools/wandb_mode_shim/ogbench_drq.py`)

A Flax `nn.Module` mirroring FB's `DrQEncoder` (`nn_models.py:480`) exactly:

- **Input:** raw image, NHWC, uint8 range. Normalize **inside** the module:
  `x = x.astype(float32) / 255.0 − 0.5` (matches FB's `RGBNorm`, *not* impala's
  `/255`).
- **Conv trunk:** `Conv(features=32, kernel=(3,3), strides=2, padding='VALID')
  → relu → [Conv(32, (3,3), strides=1, padding='VALID') → relu] × 3 →
  reshape (..., -1)`. `padding='VALID'` is required (PyTorch DrQ uses padding=0;
  Flax defaults to 'SAME').
- **Projection:** `Dense(256) → LayerNorm(epsilon=1e-5) → tanh`. `feature_dim=256`
  matches FB; `epsilon=1e-5` matches PyTorch's `nn.LayerNorm` default (needed for
  the parity test below).
- Input channel count is inferred by Flax, so the module works whether it sees
  9 channels (obs alone, frame_stack=3 RGB) or 18 (concat obs+goal — see caveat).
- Weight init: `xavier_uniform` (OGBench convention). Architectural — not
  numerically — equal to FB at init.

### Component 2 — random_shifts augmentation (JAX, in the shim)

JAX reimplementation of FB's `Augmentator` (`nn_models.py:530`): replicate-pad by
`pad=2`, then bilinear grid-sample with a per-sample random integer shift in
`[0, 2·pad]`, `align_corners=False`. Applied to every batch (**p_aug=1.0**,
matching DrQ-v2 / FB which always augment). Operates on NHWC 4D image tensors;
non-image keys pass through unchanged.

> Implementation note: confirm FB applies augmentation on every step (DrQ-v2
> convention) during implementation; if FB uses a probability, mirror it. Default
> assumption is always-on (p_aug=1.0).

### Component 3 — injection (sitecustomize shim)

Extend `tools/wandb_mode_shim/sitecustomize.py` (single sitecustomize; Python
imports only the first on sys.path, so all injection lives here):

- **Always**: register `utils.encoders.encoder_modules['drq'] = DrQEncoder`.
  Harmless unless a config selects `encoder: drq`.
- **When `OGBENCH_DRQ_FRONTEND=1`**: monkeypatch `utils.datasets.GCDataset.augment`
  to the random_shifts implementation. Gating ensures impala runs keep OGBench's
  native random_crop aug untouched.

`run_gciql.py` (our code, editable) sets `OGBENCH_DRQ_FRONTEND=1` in the child env
when `agent_overrides.encoder == 'drq'`. The shim dir is already on PYTHONPATH.

### Component 4 — Hydra configs

Cloned from the impala configs, under `configs/gciql/`:

- `cube_single_visual_drq.yaml` (GCIQL): `agent_file: agents/gciql.py`,
  `encoder: drq`, `frame_stack: 3` in `agent_overrides`, `alpha: 1.0`,
  `run_group: factored-fb-gciql-pixel-drq`.
- `cube_single_visual_gcivl_drq.yaml` (GCIVL): `agent_file: agents/gcivl.py`,
  `encoder: drq`, `frame_stack: 3`, `alpha: 10.0`,
  `run_group: factored-fb-gcivl-pixel-drq`.
- Both keep `train_steps: 500000`, `batch_size: 256`, `save_interval: 50000`,
  `eval_interval: 100000`, `eval_episodes: 50`, `eval_on_cpu: 0`.
- Both set `p_aug: 1.0` (always augment). OGBench's p_aug *gate* is in
  vendored `GCDataset.sample()` (`if rand() < p_aug: self.augment(...)`); the
  shim only replaces `augment` (random_shifts), so always-on augmentation
  requires `p_aug: 1.0` in the config, not just the monkeypatch.

The launcher `scripts/run_gciql_visual.sh` already takes `CONFIG=...`, so no
launcher change is needed beyond passing the new config names.

## GCEncoder caveat (documented, not a bug)

In GCIQL/GCIVL the encoder is wrapped in `GCEncoder(concat_encoder=...)`, which
encodes `concat([obs, goal])` on the channel axis (≈18 channels with
frame_stack=3 RGB). FB encodes obs alone (9 channels). The **conv architecture**
is identical across methods; the input channel count differs because the
*algorithms* differ (goal-conditioned concat vs FB's forward-backward). This is
inherent and is not an encoder-level confound.

## Testing / parity

1. **Encoder unit test:** instantiate the JAX DrQ encoder on a dummy
   `(B, 64, 64, 9)` batch → assert output `(B, 256)`; check trunk spatial dims
   (64→31→29→27→25; flattened 25·25·32 = 20000 → Dense 256) match FB's `repr_dim`
   for the same input shape.
2. **Augmentation unit test:** random_shift output shape == input shape; shift
   range within `[0, 2·pad]`; deterministic under a fixed PRNG key; non-image
   keys untouched.
3. **Weight-transplant numerical parity test (hard gate, requested):** verify the
   JAX DrQ encoder and FB's PyTorch `DrQEncoder` produce the **same output for the
   same input under the same parameterization**:
   - Build both encoders for matching input shape.
   - Transplant FB's PyTorch weights into the Flax module:
     conv kernels `[out, in, kh, kw]` (torch) → `[kh, kw, in, out]` (flax);
     `Dense.kernel` = torch `Linear.weight.T`; copy biases; copy LayerNorm
     `scale`/`bias`.
   - Feed identical input, accounting for layout (torch NCHW vs jax NHWC) and the
     normalization boundary (FB applies `RGBNorm` outside the encoder, so apply
     `RGBNorm` to the torch input; the JAX encoder normalizes internally — feed it
     the raw equivalent).
   - Assert `max|out_jax − out_torch| < 1e-5` (float32; conv accumulation order
     differs slightly across frameworks).
   - A pass proves architectural equivalence; a fail localizes the discrepancy
     (padding, layout, eps, projection).
4. **Behavioral smoke run:** ~2k-step single-seed run with the DrQ front-end →
   loss decreases, eval executes, GPU util ~25 it/s with no 20s/it pathology,
   checkpoints written.

## Experiment plan

DrQ-front-end **GCIQL** and **GCIVL** on `visual-cube-single-play-v0`, seeds 0–4,
500k steps, on the healthy GPUs (3,5,6,7, plus GPUs freed once the impala GCIVL
runs finish; the degraded GPUs 0,1,2,4 remain avoided until a reset/reboot).
Impala runs stay as the OGBench-default baseline. New wandb groups
(`factored-fb-{gciql,gcivl}-pixel-drq`) keep DrQ and impala runs separate.

## Out of scope

- No edits to vendored OGBench.
- No changes to FB or the existing impala configs/runs.
- Other environments beyond `visual-cube-single-play-v0` (can extend later).
