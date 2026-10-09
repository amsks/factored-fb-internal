# PSM & RLDP pixel (DrQ) + FlowBC — validate and wire

**Date:** 2026-06-03
**Branch:** feat/psm-integration
**Status:** Approved design

## Goal

1. **Validate** the existing PSM+FlowBC+DrQ pixel path with a short *real* run —
   confirm learning behavior, not merely finite losses.
2. **Wire** RLDP+FlowBC+DrQ up to the same point PSM already reached (domain
   config + launcher + smoke test), reusing the FB encoder plumbing RLDP already
   inherits.
3. Leave both ready for the user to launch the full multi-seed campaign.

## Context / current state

- **PSM pixel is complete and smoke-tested.** `configs/domain/visual_cube_single_psm.yaml`
  (DrQ `feature_dim=256`, random-shift `pad=2`, RGB normalizer, `z_dim=50`,
  `batch_size=256`), launcher `scripts/agents/psm/pixel.sh`, and pixel smoke tests
  (`tests/test_psm_pixel_smoke.py`, `tests/test_psm_e2e.py`). Smoke = finite losses
  only; no real-run learning evidence yet.
- **RLDP is state-only.** `RLDPAgent` / `RLDPFlowBCAgent` exist and are registered
  in `train.py` (`rldp` at :249, `rldp_flowbc` at :269). RLDP inherits FB's encoder
  machinery: `RLDPAgent.update()` (agents/rldp/agent.py:78-96) already runs
  `_obs_normalizer` -> `_augmentator` -> `_fw_encoder`/`_bw_encoder` on raw obs,
  same as the FB pixel path. `build_pixel_cfgs` (train.py:70) is agent-agnostic.
  Gap: no pixel domain config, no pixel launcher, no pixel smoke test for RLDP.
- **RLDP consumes windowed `[B, horizon, ...]` batches** (agent.py:63-89). The
  `future_obs` slice (`[:, 1:]`) is reshaped and pushed through normalize -> augment
  -> encode — a per-step image-tensor cost the state path never incurred.

## Approach

Mirror the established "sibling domain per agent" pattern PSM already used
(Approach A). Minimal change surface; expected zero RLDP agent-logic changes.

## Components

### 1. RLDP pixel domain config — `configs/domain/visual_cube_single_rldp.yaml`
Mirror `visual_cube_single_psm.yaml`:
- `defaults: [/agent: rldp_flowbc, _self_]`, `agent: rldp_flowbc`
- `domain: cube-single-play-v0`, `obs_type: pixels`, `frame_stack: 3`
- `obs_normalizer.name: RGBNormalizerConfig`
- `rgb_encoder: {name: drq, feature_dim: 256}`
- `augmentator: {name: random_shifts, pad: 2}`
- Pixel arch deltas: `z_dim: 50`, `batch_size: 256`, `discount: 0.99`,
  `bc_coeff: 3.0`, `actor.hidden_dim: 512`, `actor_vf.hidden_dim: 512`
- **RLDP-specific (must carry, unlike PSM):** `horizon`, the `predictor` block,
  and `ortho_coef` — whatever the state RLDP config sets, preserved here.
- **Pixel-tuned horizon:** start at `horizon: 3` (state default is 5) to bound the
  windowed `future_obs` image-encode memory. Confirm/tune during validation.

### 2. RLDP pixel launcher — `scripts/agents/rldp/pixel.sh`
Mirror `scripts/agents/psm/pixel.sh`: same env-var knobs (`SEEDS`, `GPUS`,
`TRAIN_STEPS`, `RUN_GROUP`, `DATA_PATH`, `DRY_RUN`, etc.), `TRAIN_STEPS=500000`
default, `RUN_GROUP=rldp_pixel`, and
`_run_fb_sweep visual_cube_single_rldp "$RUN_GROUP" "${extra[@]}"`.

### 3. RLDP pixel smoke test — `tests/test_rldp_pixel_smoke.py`
Mirror `tests/test_psm_pixel_smoke.py`, adapted for RLDP:
- Build `RLDPFlowBCAgent` with `obs_space=Box(0,255,(9,32,32))`, DrQ encoder,
  random-shift aug, small `feature_dim`, small `horizon`.
- `test_pixel_front_end_built_from_encoder_output_dim` — `obs_dim` derived from
  encoder `output_space` (not raw `9*32*32`).
- `test_pixel_update_finite` — one **windowed** update yields finite
  `fb_loss`, `sp_loss`, `orth_loss`, `actor_loss`, `bc_flow_loss`.
- **New helper required:** a windowed pixel batch builder producing `[B, horizon, ...]`
  per key (`observation`, `action`, `next.observation`, `next.terminated`). This is
  the one piece not copyable verbatim from the PSM smoke test (PSM is not windowed).

### 4. Verification gate — make_agent forwarding
Confirm `make_agent` passes `rgb_encoder_cfg` / `augmentator_cfg` /
`obs_normalizer_cfg` into `rldp_flowbc` via the `shared` dict (expected: yes,
since it is the same `shared` FB/PSM use). If not, add the forwarding — the only
anticipated code change.

### 5. Short validation runs (user launches full afterward)
- PSM pixel: `scripts/agents/psm/pixel.sh` with `TRAIN_STEPS~=3000`, 1 seed.
- RLDP pixel: same via the new `scripts/agents/rldp/pixel.sh`.
- Success criteria (sanity, NOT convergence):
  - losses finite and not diverging across the short run;
  - encoder params receive non-zero gradients (the DrQ front-end is actually training);
  - an early eval produces *some* reward/success signal.
- **wandb:** short sanity runs default to **wandb disabled** to avoid polluting the
  project with throwaway runs. If any run does log, it MUST use entity `amsks`
  (never the team default) — verify before launching.

## Out of scope
- Other benchmarks / multi-task pixel configs (Approach B).
- The full multi-seed campaign (user launches).
- Any RLDP agent-logic change beyond the possible `make_agent` forwarding fix.

## Risks
- **Windowed pixel memory/throughput.** The `horizon` dimension multiplies
  image-encode cost; may force smaller `horizon`/`batch`. The pixel data-pipeline
  numpy<->jax/host round-trip GPU-starvation gotcha applies to this path too —
  watch GPU utilization during the validation run.
- **make_agent not forwarding pixel cfgs to RLDP** — low risk; small fix if so.

## Verification
- `.venv/bin/python -m pytest tests/test_rldp_pixel_smoke.py` green (env note:
  `uv run` is broken in this repo; use `.venv/bin/python -m pytest`).
- Existing PSM pixel smoke tests stay green.
- `DRY_RUN=1` on both pixel launchers prints the expected hydra command.
- Short validation runs meet the sanity criteria above.
