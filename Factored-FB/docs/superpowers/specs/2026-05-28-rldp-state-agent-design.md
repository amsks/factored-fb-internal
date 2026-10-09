# Design: RLDP integration — sub-project 2 (RLDP + RLDP+FlowBC on state)

**Date:** 2026-05-28
**Scope:** `agents/rldp/`, `configs/agent/rldp*.yaml`, `scripts/agents/rldp{,_flowbc}/state.sh`, and one conditional in `train.py`.
**Audience:** anyone landing plain RLDP and RLDP+FlowBC as state-only agents on cube-single. Pixel variants are sub-project 3.

## Context

Sub-project 2 follows sub-project 1 (Foundation), which already restored `VForwardArchiConfig`/`VForwardMap`, added `horizon` kwarg to `DictBuffer.sample()`, and shipped `tests/parity/harness.py`. The foundation pieces are unconditionally available on `rldp-foundation`; sub-project 2 assumes they are merged.

This sub-project takes td_jepa's reference RLDP implementation (`/home/adityamohan/git/baselines/td_jepa/metamotivo/agents/rldp/`, ~470 LOC across 6 files) and adopts it **as our implementation** — not as a vendored fixture, not behind a parity gate. The user's explicit call: "td_jepa's RLDP files BECOME `agents/rldp/`". The only changes during the copy are (a) import path rewrites and (b) the buffer-interface adapter that lets the agent consume our `DictBuffer.sample(horizon=h)` output.

Pixel-specific code paths in the td_jepa RLDP files (DrQ encoder hooks, RGB augmentation, pixel obs spaces) are kept in the copy but **not exercised** in this sub-project — sub-project 3 wires them up.

## Problem

Three concrete pieces are missing for a state-only RLDP run on cube-single:

- **Agent implementation**: no `agents/rldp/` directory exists. The closest precedent is `agents/fb/` (which we ported earlier from td_jepa). The same recipe applies, but the source files differ (RLDP adds `_predictor`, SP loss, and a different optimizer wiring).
- **Configs**: no `configs/agent/rldp.yaml`. The closest precedent is `configs/agent/fb.yaml` + `fb_flowbc.yaml`.
- **Launchers**: no `scripts/agents/rldp/` directory. Mirrors `scripts/agents/fb/`.

Plus one mechanical change: `train.py:278` currently calls `buffer.sample(cfg.batch_size)`. RLDP needs `horizon=cfg.horizon`. One conditional handles both code paths.

## Design

### Files created (six agent files, two configs, two launchers, four test modules)

```
agents/rldp/
├── __init__.py                      # empty
├── agent.py                         # ~234 LOC; copied from td_jepa rldp/agent.py
├── model.py                         # ~42 LOC; copied from td_jepa rldp/model.py
└── flow_bc/
    ├── __init__.py                  # empty
    ├── agent.py                     # ~116 LOC; copied from td_jepa rldp/flow_bc/agent.py
    └── model.py                     # ~71 LOC; copied from td_jepa rldp/flow_bc/model.py

configs/agent/
├── rldp.yaml                        # inherits fb.yaml; adds horizon, predictor, lowered ortho_coef
└── rldp_flowbc.yaml                 # inherits fb_flowbc.yaml + same additions

scripts/agents/
├── rldp/state.sh                    # mirrors fb/state.sh; sets agent=rldp
└── rldp_flowbc/state.sh             # mirrors fb/state.sh; sets agent=rldp_flowbc

tests/
├── test_rldp_model.py               # build, forward, param count
├── test_rldp_agent_smoke.py         # 10-step update on synthetic data
├── test_rldp_flowbc_model.py        # build, forward, param count
└── test_rldp_flowbc_agent_smoke.py  # 10-step update on synthetic data
```

Plus two appended test cases in `tests/scripts/test_agent_launchers.py` and one conditional edit in `train.py`.

### Copy-and-adapt rules

For each of the six agent files copied from td_jepa:

1. **Replace imports.** td_jepa uses package-relative paths inside `metamotivo/`. Rewrite to our flat layout:
   - `from ...nn_models import X` → `from nn_models import X`
   - `from ...base_model import load_model` → match our equivalent (we may not need `load_model` — keep the import only if used)
   - `from ..fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig` → `from agents.fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig`
   - `from ..fb.model import FBModel, FBModelArchiConfig, FBModelConfig` → `from agents.fb.model import FBModel, FBModelArchiConfig, FBModelConfig`
   - `from ...td3.nn_models import SimpleVectorFieldArchiConfig` → check if our `nn_models.py` re-exports this; otherwise vendor in the same way.
2. **Preserve the Meta CC BY-NC 4.0 copyright header.**
3. **Do not refactor the loss code.** Any logic edit beyond what's necessary to compile is out of scope — preserves the "copy" character of the port.

### Buffer integration — the one design decision specific to our codebase

td_jepa's `RLDPAgent.update()` calls `replay_buffer["train"].sample(cfg.train.batch_size)` and expects a mixed-granularity batch (single-transition `observation`, `action`, `next/observation`, `next/terminated`; windowed `next/traj_observation`, `next/traj_action`). Our `DictBuffer.sample(batch_size, horizon=h)` returns everything as `[B, h, ...]`.

**Adaptation lives inline in `agents/rldp/agent.py::RLDPAgent.update()`.** The first ~10 lines reshape the windowed batch:

```python
# Inline adapter: windowed batch -> td_jepa-shaped intermediates.
obs        = batch["observation"][:, 0]                  # [B, obs_dim]
action     = batch["action"][:, 0]                       # [B, action_dim]
next_obs   = batch["next"]["observation"][:, 0]          # [B, obs_dim]
terminated = batch["next"]["terminated"][:, 0]           # [B]
future_obs = batch["observation"][:, 1:]                 # [B, h-1, obs_dim]
future_act = batch["action"][:, 1:]                      # [B, h-1, action_dim]
# Then proceed exactly as td_jepa's update_fb expects: pass future_obs.reshape(...) and future_act.
```

This keeps `DictBuffer` generic (no RLDP-specific code in the buffer) and matches the spirit of td_jepa's own `update()` which also does inline reshapes.

### `train.py` — single conditional change

Around line 278:
```python
batch = buffer.sample(cfg.batch_size)
```
becomes:
```python
horizon = getattr(cfg, "horizon", 1)
batch = buffer.sample(cfg.batch_size, horizon=horizon)
```

`horizon` is `1` for all existing agents (FB, FB+FlowBC) so behavior is byte-identical. RLDP configs set `horizon: 5`.

### Configs

`configs/agent/rldp.yaml`:
```yaml
# @package _global_
defaults:
  - fb
  - _self_

agent: rldp
horizon: 5                   # RLDP samples horizon-5 windows for SP loss
ortho_coef: 0.01             # much lower than FB's 1000 — encoder absorbs B-regularization
predictor:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
```

`configs/agent/rldp_flowbc.yaml`:
```yaml
# @package _global_
defaults:
  - fb_flowbc
  - _self_

agent: rldp_flowbc
horizon: 5
ortho_coef: 0.01
predictor:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
```

Both inherit everything else (z_dim, lr_*, network architecture for forward/backward/actor) from their parent FB config.

### Launchers

`scripts/agents/rldp/state.sh` and `scripts/agents/rldp_flowbc/state.sh` mirror `scripts/agents/fb/state.sh`. Defaults: `SEEDS="0 1 2 3 4"`, `GPUS="0 1"`, `TRAIN_STEPS=1000000`, `RUN_GROUP=rldp_state` / `rldp_flowbc_state`. The leaf calls `_run_fb_sweep cube_single "$RUN_GROUP" agent=rldp ...` (passing `agent=rldp` as an extra Hydra arg). MODE/REWEIGHT knobs from `fb/state.sh` are not relevant to RLDP — left out of the new launchers.

### Tests

- `tests/test_rldp_model.py`: build `RLDPModel(obs_space=Box(20), action_dim=4)`, run `model._forward_map(obs, z, action)` and `model._predictor(B(obs), action)` on small batches, assert output shapes and that `_predictor` is registered as a torch submodule.
- `tests/test_rldp_agent_smoke.py`: build a synthetic 200-transition `DictBuffer` across 5 episodes of length 40, build `RLDPAgent`, run `agent.update(batch)` 10 times with `horizon=5`, assert no NaN in returned metrics dict and that `len(metrics) > 0` per step.
- Same shape for `_flowbc` variants.
- Append to `tests/scripts/test_agent_launchers.py`:
  ```python
  def test_rldp_state_dry_run():
      r = _dry_run("rldp/state.sh")
      assert r.returncode == 0
      assert "agent=rldp" in r.stdout
      assert "domain=cube_single" in r.stdout

  def test_rldp_flowbc_state_dry_run():
      r = _dry_run("rldp_flowbc/state.sh")
      assert r.returncode == 0
      assert "agent=rldp_flowbc" in r.stdout
  ```

### Conventions

- Per `memory/factored-fb-test-env.md`: tests run via `.venv/bin/python -m pytest` (uv broken).
- Per `memory/commit-no-claude-coauthor.md`: commits omit the Claude co-author trailer.
- Existing pandas-missing test failures in the suite remain skipped as known-pre-existing.

## Acceptance criteria

1. All new tests pass: `.venv/bin/python -m pytest tests/test_rldp_model.py tests/test_rldp_agent_smoke.py tests/test_rldp_flowbc_model.py tests/test_rldp_flowbc_agent_smoke.py tests/scripts/test_agent_launchers.py -v`.
2. Sub-project 1 tests still pass: `.venv/bin/python -m pytest tests/test_nn_models_vforward.py tests/test_dict_buffer_horizon.py tests/parity/ -v` clean.
3. FB launcher tests still pass: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k fb` clean (no regressions on FB paths).
4. Both new launchers dry-run cleanly:
   - `DRY_RUN=1 bash scripts/agents/rldp/state.sh 2>&1 | tail -3` prints a `train.py agent=rldp horizon=5 ...` command.
   - Same for `rldp_flowbc/state.sh`.
5. **End-to-end smoke**: one short real training run (5000 steps, single seed) on `/dev/shm/factored-fb/datasets/cube-single-play-v0`:
   ```
   TRAIN_STEPS=5000 SEEDS=0 GPUS=0 bash scripts/agents/rldp/state.sh
   ```
   completes without crashing. Logs show loss values that are finite (no NaN). This is the only acceptance gate that touches real GPUs.
6. Diff bounded to: `agents/rldp/`, `configs/agent/rldp.yaml`, `configs/agent/rldp_flowbc.yaml`, `scripts/agents/rldp/state.sh`, `scripts/agents/rldp_flowbc/state.sh`, `tests/test_rldp_*`, `tests/scripts/test_agent_launchers.py`, and a single conditional in `train.py`.

## Out of scope

- **Pixel variants** of RLDP / RLDP+FlowBC (sub-project 3).
- Hyperparameter sweeps. Single default config; sweep configs come later if needed.
- Multi-stage campaign launchers (like `run_crl_flowbc_campaign.sh`). Standard `SEEDS=...` pattern of `state.sh` is enough.
- Any modification to `agents/fb/`. FB stays byte-identical.
- Parity gates against an external td_jepa reference. The in-tree code is the implementation.
- Running td_jepa's RLDP for comparison. Out of scope for the port itself; later experimentation may launch td_jepa side-by-side, but that's user-initiated.

## Open risks / questions

- **`SimpleVectorFieldArchiConfig` location**: td_jepa imports it from `metamotivo/agents/td3/nn_models.py`. Our `nn_models.py` exports `SimpleVectorFieldArchiConfig` at the top level (used by FB+FlowBC). Verify during implementation that the import rewrite resolves correctly; if not, copy in whichever class is missing.
- **`base_model.load_model`**: td_jepa's `RLDPModel.load(...)` calls `load_model` from `base_model.py`. We may or may not have an equivalent. If not, drop the `load` classmethod from the copy (we don't currently load RLDP from disk during training — only at eval, which is a separate code path that can be added when needed).
- **`compile`/`cudagraphs` config fields**: td_jepa's `setup_training` references `self.cfg.cudagraphs` and `self.cfg.compile`. Confirm these exist on our FBAgentConfig; if not, default-add them to `agents/fb/agent.py::FBAgentConfig` (small additive change, won't affect FB behavior).
- **Loss values on 5000-step smoke**: RLDP losses (FB + SP + ortho + Q) should be non-NaN but won't necessarily look meaningful at 5000 steps. The smoke gate is "doesn't crash and produces finite numbers", not "converges to a useful value". Real performance evaluation lives in sub-project 4 (full sweeps).

## References

- td_jepa source: `/home/adityamohan/git/baselines/td_jepa/metamotivo/agents/rldp/{agent.py, model.py, flow_bc/}`
- td_jepa launcher we're loosely mirroring: `scripts/train/proprio/launch_rldp_ogbench.py`
- Our FB precedent:
  - `agents/fb/{agent.py, model.py}` (the port template)
  - `configs/agent/fb.yaml` + `fb_flowbc.yaml`
  - `scripts/agents/fb/state.sh`
- Sub-project 1 foundation (must be merged before this work starts):
  - `nn_models.py` — `VForwardArchiConfig` + `VForwardMap`
  - `buffers/transition.py` — `DictBuffer.sample(horizon=h)`
  - `tests/parity/harness.py` — unused here, available for sub-project 3
