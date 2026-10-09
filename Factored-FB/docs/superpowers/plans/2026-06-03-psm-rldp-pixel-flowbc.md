# PSM & RLDP pixel (DrQ) + FlowBC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the existing PSM+FlowBC+DrQ pixel path with a short real run, and wire RLDP+FlowBC+DrQ to the same point (domain config + launcher + smoke test).

**Architecture:** RLDP inherits FB's pixel-capable encoder plumbing (`_obs_normalizer` -> `_augmentator` -> `_fw`/`_bw` encoders already run in `RLDPAgent.update()`), and `build_pixel_cfgs` is agent-agnostic. So pixels for RLDP are config + test, mirroring the "sibling domain per agent" pattern PSM used (`visual_cube_single_psm.yaml`). Expected: zero RLDP agent-logic changes.

**Tech Stack:** PyTorch, Hydra configs, OGBench visual envs, DrQ CNN encoder, pytest. Test env note: `uv run` is broken in this repo — use `.venv/bin/python -m pytest`.

---

## File Structure

- Create: `tests/test_rldp_pixel_smoke.py` — RLDP pixel front-end smoke (encoder wiring + finite windowed update).
- Create: `configs/domain/visual_cube_single_rldp.yaml` — sibling pixel domain selecting `rldp_flowbc` + DrQ.
- Create: `scripts/agents/rldp/pixel.sh` — RLDP pixel launcher mirroring `scripts/agents/psm/pixel.sh`.
- Read-only reference: `agents/rldp/agent.py`, `agents/rldp/flow_bc/agent.py`, `train.py:70-94` (`build_pixel_cfgs`), `train.py:269-294` (`rldp_flowbc` registration), `configs/domain/visual_cube_single_psm.yaml`, `tests/test_rldp_flowbc_agent_smoke.py`, `tests/test_psm_pixel_smoke.py`.

Note: `docs/superpowers/**` is gitignored in this repo — plan/spec files live on disk and are not committed. Commit only code/config/test/script files.

---

### Task 1: RLDP pixel front-end smoke test

Proves the inherited FB encoder plumbing works for RLDP on pixels through a windowed update. If this fails because pixel cfgs don't reach the agent, Task 1b fixes `make_agent`; otherwise Task 1b is a no-op verification.

**Files:**
- Test: `tests/test_rldp_pixel_smoke.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""RLDP pixel (DrQ) front-end smoke.

Builds RLDPFlowBCAgent with the DrQ encoder + random-shift augmentator + RGB
normalizer (frame_stack=3 => 9 channels), feeds a *windowed* pixel batch via
DictBuffer.sample(..., horizon=h), and asserts one update step yields finite
losses. Verifies:
  - the model derives obs_dim from the encoder output_space (not raw obs shape),
  - update() applies normalize + aug + enc so the CNN front-end runs on pixels,
  - the windowed SP-loss path (future_obs slice) runs through the encoder.
"""
import math

import gymnasium as gym
import numpy as np
import torch

from agents.rldp.flow_bc.agent import RLDPFlowBCAgent
from buffers.transition import DictBuffer
from nn_models import (
    AugmentatorArchiConfig,
    DrQEncoderArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import RGBNormalizerConfig

C, H, W = 9, 32, 32   # frame_stack=3 RGB, square image
FEATURE_DIM = 32
HORIZON = 3           # pixel-tuned horizon (state default is 5)


def _agent(**kw):
    obs_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
    return RLDPFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=32, horizon=HORIZON,
        ortho_coef=0.01, flow_steps=5, bc_coeff=0.3, lr_actor_vf=3e-4,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        obs_normalizer_cfg=RGBNormalizerConfig(),
        rgb_encoder_cfg=DrQEncoderArchiConfig(feature_dim=FEATURE_DIM),
        augmentator_cfg=AugmentatorArchiConfig(pad=2),
        device="cpu", **kw)


def _pixel_buffer(n_eps=5, ep_len=40):
    total = n_eps * ep_len
    img = lambda: np.random.randint(0, 256, (total, C, H, W), dtype=np.uint8)
    action = np.tanh(np.random.randn(total, 5)).astype(np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=bool)
    for i in range(n_eps):
        terminated[(i + 1) * ep_len - 1] = True
    timestep = np.concatenate([np.arange(ep_len) for _ in range(n_eps)]).astype(np.int64)
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": img(),
        "action": action,
        "reward": reward,
        "next": {"observation": img(), "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_pixel_front_end_built_from_encoder_output_dim():
    a = _agent()
    assert type(a.model._fw_encoder).__name__ == "DrQEncoder"
    assert type(a.model._bw_encoder).__name__ == "DrQEncoder"
    assert type(a.model._augmentator).__name__ == "Augmentator"
    # obs_dim must come from the encoder output_space, not the raw (9,32,32) obs space
    assert a.model._fw_encoder.output_space.shape[0] == FEATURE_DIM


def test_pixel_update_finite():
    a = _agent()
    buf = _pixel_buffer()
    torch.manual_seed(0)
    m = a.update(buf.sample(8, horizon=HORIZON), step=0)
    for k in ["fb_loss", "sp_loss", "orth_loss", "actor_loss", "bc_flow_loss"]:
        assert math.isfinite(float(m[k])), k
```

- [ ] **Step 2: Run test to verify it runs (and see whether plumbing already works)**

Run: `.venv/bin/python -m pytest tests/test_rldp_pixel_smoke.py -v`
Expected: Either PASS (inherited plumbing already forwards pixel cfgs — most likely) or FAIL with an error showing the encoder wasn't built / obs_dim used the raw `9*32*32` shape. If it PASSES, Task 1b is a verification no-op. If it FAILS on cfg forwarding, do Task 1b then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rldp_pixel_smoke.py
git commit -m "test(rldp): pixel DrQ front-end smoke (encoder wiring + windowed finite update)"
```

---

### Task 1b: Verify/fix make_agent forwards pixel cfgs to rldp_flowbc

Only acts if Task 1 failed on cfg forwarding. `build_pixel_cfgs` (train.py:70-94) returns `(obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg)`; confirm these land in the `shared` dict passed to `RLDPFlowBCAgent` at train.py:286-294.

**Files:**
- Read: `train.py` around the `shared = {...}` construction (before line 245) and the `rldp_flowbc` branch (`train.py:269-294`).
- Modify (only if missing): `train.py` — add the three pixel cfgs to `shared` or to the `rldp_flowbc` call.

- [ ] **Step 1: Inspect the shared-dict construction**

Run: `grep -n "shared\b\|rgb_encoder_cfg\|augmentator_cfg\|obs_normalizer_cfg\|build_pixel_cfgs" train.py`
Expected: `shared` already includes `rgb_encoder_cfg`, `augmentator_cfg`, `obs_normalizer_cfg` (the same dict FB/PSM use). If present, no change — this task is a no-op; skip to Step 3.

- [ ] **Step 2: If absent, forward the cfgs**

Add the missing keys to the `shared` dict so the `**shared` expansion in the `rldp_flowbc` branch carries them, matching how `fb_flowbc`/`psm_flowbc` receive them. (Show the exact 3 lines in context when editing; do not introduce a new pattern — copy the FB branch's keys verbatim.)

- [ ] **Step 3: Re-run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_rldp_pixel_smoke.py -v`
Expected: PASS (both tests).

- [ ] **Step 4: Commit (only if a code change was made)**

```bash
git add train.py
git commit -m "fix(rldp): forward pixel cfgs (drq/rgb-norm/augment) into rldp_flowbc"
```

---

### Task 2: RLDP pixel domain config

Sibling domain mirroring `visual_cube_single_psm.yaml` but selecting `rldp_flowbc` and carrying RLDP's `horizon`/`predictor`/`ortho_coef`. Pixel-tuned `horizon: 3`.

**Files:**
- Create: `configs/domain/visual_cube_single_rldp.yaml`

- [ ] **Step 1: Write the config**

```yaml
# @package _global_
# Pixel cube-single RLDP+FlowBC. Mirrors visual_cube_single_psm.yaml but selects
# the rldp_flowbc agent (the FB visual domain hard-codes agent: fb_flowbc, and
# `agent` is a config group so it cannot be force-added on the CLI — hence a
# sibling domain). Carries RLDP-specific horizon/predictor/ortho_coef.
defaults:
  - /agent: rldp_flowbc
  - _self_

domain: cube-single-play-v0      # create_ogbench_env prepends "visual-" for pixels
agent: rldp_flowbc
obs_type: pixels
frame_stack: 3

# Pixel module selection (consumed by train.build_pixel_cfgs)
obs_normalizer:
  name: RGBNormalizerConfig
rgb_encoder:
  name: drq
  feature_dim: 256
augmentator:
  name: random_shifts
  pad: 2

# Architecture deltas vs state (match the FB/PSM visual domain shared pixel knobs)
z_dim: 50
batch_size: 256
discount: 0.99
bc_coeff: 3.0
actor:
  name: noise_conditioned_actor
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
actor_vf:
  hidden_dim: 512
  hidden_layers: 4

# RLDP-specific (pixel-tuned horizon to bound windowed future_obs encode memory)
horizon: 3
ortho_coef: 0.01
predictor:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
```

- [ ] **Step 2: Verify the config composes via Hydra**

Run:
```bash
.venv/bin/python -c "
import hydra
from hydra import compose, initialize
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['domain=visual_cube_single_rldp'])
    print('agent:', cfg.agent)
    print('obs_type:', cfg.obs_type)
    print('rgb_encoder:', cfg.rgb_encoder.name, cfg.rgb_encoder.feature_dim)
    print('horizon:', cfg.horizon)
    assert cfg.agent == 'rldp_flowbc'
    assert cfg.rgb_encoder.name == 'drq'
    assert cfg.horizon == 3
print('OK')
"
```
Expected: prints `agent: rldp_flowbc`, `rgb_encoder: drq 256`, `horizon: 3`, then `OK`. If the top-level config name is not `config`, discover it with `ls configs/*.yaml` and substitute. If `initialize` config_path differs, match the path `train.py` uses (`grep -n "config_path\|config_name" train.py`).

- [ ] **Step 3: Commit**

```bash
git add configs/domain/visual_cube_single_rldp.yaml
git commit -m "feat(rldp): pixel domain config (visual_cube_single_rldp, drq + horizon 3)"
```

---

### Task 3: RLDP pixel launcher

Mirror `scripts/agents/psm/pixel.sh` exactly, swapping the domain to `visual_cube_single_rldp` and the run group to `rldp_pixel`. Drop the PSM-only proto knobs (`MAX_LOG_SEED`, proto `NUM_PARALLEL`); keep `Z_DIM`.

**Files:**
- Create: `scripts/agents/rldp/pixel.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# scripts/agents/rldp/pixel.sh — RLDP+FlowBC on visual_cube_single (pixel obs, frame-stack 3).
# Selects the visual domain + DrQ encoder (via the domain config) and the rldp_flowbc
# agent config group. OGBench visual default is 500k steps.
USAGE="usage: bash scripts/agents/rldp/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=rldp_pixel
    DATA_PATH=/dev/shm/factored-fb/datasets
    Z_DIM=             continuous z dim (default from agent config)
    HORIZON=           SP-loss window length (default 3 from domain config)
    SAVE_EVAL_VIDEOS=false       forward to train.py as save_eval_videos=<val>
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-rldp_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"

# visual_cube_single_rldp selects domain=visual + drq encoder + agent=rldp_flowbc.
extra=()
[[ -n "${Z_DIM:-}" ]]   && extra+=("z_dim=$Z_DIM")
[[ -n "${HORIZON:-}" ]] && extra+=("horizon=$HORIZON")

export TRAIN_STEPS
_run_fb_sweep visual_cube_single_rldp "$RUN_GROUP" "${extra[@]}"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/agents/rldp/pixel.sh`

- [ ] **Step 3: Dry-run to verify the emitted command**

Run: `DRY_RUN=1 SEEDS='0' GPUS='0' bash scripts/agents/rldp/pixel.sh`
Expected: prints a `train.py` invocation containing `domain=visual_cube_single_rldp` and the configured train steps, without launching training. (If `_run_fb_sweep` uses a different DRY_RUN sentinel, check `scripts/agents/_common.sh` and match it; PSM's pixel.sh advertises `DRY_RUN=1`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/agents/rldp/pixel.sh
git commit -m "feat(rldp): pixel launcher (visual_cube_single_rldp, 500k default)"
```

---

### Task 4: PSM pixel short validation run

Confirm the *existing* PSM pixel path actually learns (not just finite losses). Sanity, not convergence.

**Files:** none (run + observe).

- [ ] **Step 1: Launch a short PSM pixel run, wandb disabled**

Run (1 seed, ~3000 steps, no wandb to avoid throwaway runs polluting the project):
```bash
DRY_RUN=1 SEEDS='0' GPUS='0' TRAIN_STEPS=3000 bash scripts/agents/psm/pixel.sh
```
First confirm the dry-run command looks right, then re-run without `DRY_RUN=1` to actually launch. If the launcher/`_common.sh` doesn't expose a wandb-disable knob, add `wandb.mode=disabled` (or the repo's equivalent) to the hydra extra args — discover the exact key with `grep -rn "wandb" train.py configs/ | head`. If a run must log, it MUST use entity `amsks` (never the team default).

- [ ] **Step 2: Observe the sanity criteria**

During/after the run, confirm:
- losses (`psm_loss`, `sf_loss`, `actor_loss`, `bc_flow_loss`) stay finite and do not diverge across the 3000 steps;
- the DrQ encoder is actually training — its params receive non-zero gradients (spot-check via a brief logged grad-norm, or confirm encoder weights changed between step 0 and step 3000);
- an early eval produces *some* reward/success signal (any nonzero is a pass at this scale).
- Watch GPU utilization — per the known pixel data-pipeline gotcha, host<->device round-trips can starve the GPU; note if util is pathologically low.

- [ ] **Step 3: Record the outcome**

Note pass/fail of each criterion in the run notes (no commit). If it fails to learn, STOP and surface findings before the campaign — this is the validation gate the user asked for.

---

### Task 5: RLDP pixel short validation run

Same sanity check for the newly-wired RLDP pixel path.

**Files:** none (run + observe).

- [ ] **Step 1: Launch a short RLDP pixel run, wandb disabled**

Run:
```bash
DRY_RUN=1 SEEDS='0' GPUS='0' TRAIN_STEPS=3000 bash scripts/agents/rldp/pixel.sh
```
Confirm the dry-run command, then re-run without `DRY_RUN=1`. Same wandb policy as Task 4 (disabled for sanity; `amsks` if logging).

- [ ] **Step 2: Observe the sanity criteria**

Confirm:
- losses (`fb_loss`, `sp_loss`, `orth_loss`, `actor_loss`, `bc_flow_loss`) stay finite and do not diverge;
- the DrQ encoder params receive non-zero gradients;
- an early eval produces *some* reward/success signal;
- **windowed-pixel memory/throughput:** confirm the `horizon: 3` windowed batch fits in GPU memory at `batch_size=256` and GPU util is reasonable. If OOM or severe slowdown, reduce `batch_size` (e.g. 128) or `horizon` (e.g. 2) in `configs/domain/visual_cube_single_rldp.yaml`, commit the tweak, and re-run.

- [ ] **Step 3: Record the outcome**

Note pass/fail of each criterion. If memory/horizon was tuned, the committed config now reflects the validated settings. Report both validation outcomes to the user; they launch the full multi-seed campaign.

---

## Notes for the implementer
- Test env: `uv run` is broken here — always use `.venv/bin/python -m pytest`.
- `docs/superpowers/**` is gitignored — never `git add` plan/spec files (they'll be rejected; that's expected).
- Do not modify RLDP agent logic unless Task 1b proves a forwarding gap — the design assumes inherited FB plumbing is sufficient.
- wandb entity is `amsks` for any real logging; sanity runs default to disabled.
