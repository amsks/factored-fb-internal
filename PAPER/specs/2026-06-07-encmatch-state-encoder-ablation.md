# Param-matched state encoder ablation (`cube_single_encmatch`)

**Date:** 2026-06-07 · **Branch:** `feat/psm-integration` (or a fresh ablation branch)

## Motivation

The pixel DrQ FB run carries ~15.5M trainable online params; the `fb_state_L256`
ablation carries only ~5.6M. The two DrQ conv encoders (`_fw_encoder` +
`_bw_encoder`, ~5.15M each = ~10.3M) are uncompensated extra capacity, learned
end-to-end from the FB objective (forward_optimizer owns `_fw_encoder`,
backward_optimizer owns `_bw_encoder`; actor grads are stopped via `obs.detach()`
+ `no_grad` so the encoder is critic-trained only). `L_dim=256` only widened the
latent — it did not match that capacity.

This ablation gives the **state** FB+FlowBC two jointly-trained MLP encoders
structurally parallel to DrQ's two conv encoders, parameter-matched to the pixel
run. It isolates the parameter-count / representation-capacity confound: does
matched capacity on the state representation path change the result, independent
of pixels?

## Key identity

If the MLP encoders output `feature_dim=256` (DrQ's feature dim), every
downstream net has identical input dims to the pixel run, hence identical param
counts:

| net | input | params (= pixel) |
|---|---|---|
| `_forward_map` | left(256)+z(50)+act(5) | 2,213,988 |
| `_backward_map` | bw_enc(256) | 946,226 |
| `_actor` | fw_enc(256) | 1,086,469 |
| `_actor_vf` | fw_enc(256) | 925,189 |
| `_left_encoder` | 256→256, 0-layer | 65,792 |
| **downstream subtotal** | | **5,237,664** |

So matching total online (~15.54M) ⟺ the two MLP encoders summing to ~10.30M
(~5.15M each). hidden_dim≈1100, hidden_layers=5 lands within ~0.2%.

MLP param count: `(nl-1)·W² + (nl+284)·W + 768` for input 28 → width W, nl
trunk layers, proj W→256 + LayerNorm. nl=5, W=1100 ⇒ ~5.159M each.

## Components

1. **`nn_models.MLPEncoder` + `MLPEncoderArchiConfig`** — mirrors `DrQEncoder`'s
   interface: `build(obs_space)` → module with `.output_space = Box(feature_dim)`.
   Trunk: `hidden_layers` × (Linear(width) + ReLU); proj: `Linear(width,
   feature_dim) → LayerNorm → Tanh` (same output-norm as DrQ). Flattens obs.
2. **`train.build_pixel_cfgs`** — add `rgb_encoder.name == "mlp"` branch →
   `MLPEncoderArchiConfig(feature_dim, hidden_dim, hidden_layers)`. Default-off:
   no block / `name: drq` unchanged (byte-identical regression).
3. **`configs/domain/cube_single_encmatch.yaml`** — `cube-single-play-v0`,
   `obs_type: state`, `agent: fb_flowbc`, `L_dim: 256`,
   `rgb_encoder: {name: mlp, feature_dim: 256, hidden_dim: 1100, hidden_layers: 5}`,
   `left_encoder: {hidden_dim: 512, hidden_layers: 0, norm: true}`,
   state FB knobs (`bc_coeff: 3.0`, `ortho_coef`, `lr_b`). No augmentator /
   normalizer blocks (stay Identity — no aug on state).
4. **`scripts/agents/fb/state_encmatch.sh`** — mirrors `state.sh`;
   `RUN_GROUP=fb_state_encmatch`, domain `cube_single_encmatch`, 1M steps,
   SEEDS/GPUS knobs, wandb entity `amsks`.
5. **`scripts/count_fb_params.py`** — add the encmatch config; assert total
   online within ~1% of pixel.

## Tests (TDD)

- `MLPEncoder`: output shape `(B, feature_dim)`, `.output_space.shape == (feature_dim,)`,
  param count near target for given (hidden_dim, hidden_layers).
- `build_pixel_cfgs`: `name: mlp` → `MLPEncoderArchiConfig`; no block → `IdentityNNConfig` (regression).
- Agent builds under encmatch + one finite `update()` (smoke); downstream-net
  param breakdown matches pixel.

## Scope / caveat

Matches **parameter count + end-to-end-trained-encoder structure**, NOT
augmentation (DrQ random-shift has no state analogue). This answers "encoder
param matched," not "pixels vs state."
