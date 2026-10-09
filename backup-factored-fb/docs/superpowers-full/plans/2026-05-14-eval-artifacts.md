# Per-run Eval Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land checkpoints + eval-episode GIFs in per-run Hydra output directories (currently overwritten under `$REPO/checkpoints/`), gated behind an opt-in `save_eval_videos` flag, and re-run `cube_single ortho=1000 lr_b=1e-4` to capture them.

**Architecture:** Repoint `cfg.save_dir` to `${hydra:runtime.output_dir}/checkpoints` so artifacts colocate with the existing Hydra per-run dir. Add a `record=False` flag to `envs/rollout.py:rollout()` that collects `env.render()` frames per episode. `OGBenchEvaluator.run(step)` writes the first episode of each task as `<eval_videos_dir>/step_<N>/<task>.gif` via `imageio.mimsave` and logs each to `wandb.Video`. Plumb `EXTRA_ARGS` through `scripts/run_sweep.sh` so the targeted re-run uses the existing sweep machinery.

**Tech Stack:** PyTorch, Hydra 1.3, OGBench (mujoco), wandb, imageio.

**Spec:** `docs/superpowers/specs/2026-05-14-eval-artifacts-design.md`

---

## File Structure

| File | Role |
|---|---|
| `configs/train.yaml` | Add `eval_videos_dir`, `save_eval_videos`; repoint `save_dir` to Hydra runtime output dir |
| `envs/rollout.py` | Add `record` param + frame collection; return 3-tuple `(stats, infos, frames)` |
| `evals/ogbench.py` | Constructor gains `save_videos`, `video_dir`, `use_wandb`; `run()` gains `step` arg, writes GIFs + `wandb.Video` |
| `train.py` | Mkdir video dir; pass new kwargs to evaluator; call `evaluator.run(step=step)` |
| `scripts/run_sweep.sh` | Append `$EXTRA_ARGS` to `python train.py` invocation |
| `requirements.txt` | Pin `imageio>=2.31` explicitly |
| `scripts/smoke_test_artifacts.sh` | New: short end-to-end smoke run that prints expected artifact tree |

---

## Task 1: Add imageio dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Check whether imageio is already importable**

Run: `python -c "import imageio; print(imageio.__version__)"`

If it prints a version ≥ 2.31, the dependency is already present transitively — still pin it explicitly. If it errors, install will happen at Step 3.

- [ ] **Step 2: Add the pin to `requirements.txt`**

Replace the entire file with:

```
# Core
torch>=2.3
numpy>=1.26

# Environments & datasets
ogbench>=1.1.2
gymnasium>=0.29

# Config
hydra-core>=1.3
omegaconf>=2.3
pydantic>=2.0            # required by base_config.py / BaseConfig

# Checkpointing
safetensors>=0.4.4

# Data
h5py>=3.11.0

# Logging
wandb>=0.22.3
tqdm

# Video / GIF writing
imageio>=2.31
```

- [ ] **Step 3: Install (idempotent if already present)**

Run: `pip install -r requirements.txt`
Expected: `Requirement already satisfied: imageio>=2.31` (or fresh install if missing).

- [ ] **Step 4: Verify import + GIF writer entrypoint**

Run: `python -c "import imageio; import numpy as np; imageio.mimsave('/tmp/_t.gif', np.zeros((3,4,4,3),dtype=np.uint8), format='GIF', fps=30, loop=0); print('ok')"`
Expected: `ok` printed and `/tmp/_t.gif` is a valid GIF (`file /tmp/_t.gif` → `GIF image data`).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "deps: pin imageio>=2.31 for eval GIF writing"
```

---

## Task 2: Add `record` parameter to rollout

**Files:**
- Modify: `envs/rollout.py`

- [ ] **Step 1: Write a minimal smoke script for the new behaviour**

Create `scripts/_smoke_rollout.py` (temporary — deleted in Step 6):

```python
"""Smoke test for envs.rollout with record=True. Builds a real cube env."""
import numpy as np
import torch

from envs.ogbench import create_ogbench_env
from envs.rollout import rollout


class DummyAgent:
    device = "cpu"
    def act(self, obs, **_):
        action_dim = 5  # cube action_dim; doesn't matter, env clips
        return torch.zeros(1, action_dim)


env, _ = create_ogbench_env("cube-single-singletask-task1-v0", seed=0, obs_type="state")
try:
    stats, infos, frames = rollout(env, DummyAgent(), num_episodes=1, record=True)
finally:
    env.close()

assert frames is not None, "frames should not be None when record=True"
assert len(frames) == 1, f"expected 1 episode of frames, got {len(frames)}"
assert len(frames[0]) >= 2, f"expected at least reset+1step frames, got {len(frames[0])}"
arr = np.stack(frames[0])
assert arr.dtype == np.uint8, f"expected uint8, got {arr.dtype}"
assert arr.ndim == 4 and arr.shape[-1] == 3, f"expected [T,H,W,3], got {arr.shape}"
print(f"OK frames shape={arr.shape}")

# Backward-compat: record=False returns 3-tuple with frames=None
env2, _ = create_ogbench_env("cube-single-singletask-task1-v0", seed=0, obs_type="state")
try:
    stats, infos, frames = rollout(env2, DummyAgent(), num_episodes=1, record=False)
finally:
    env2.close()
assert frames is None, f"frames must be None when record=False, got {type(frames)}"
print("OK record=False returns frames=None")
```

- [ ] **Step 2: Run the smoke script to confirm it FAILS**

Run: `cd /home/adityamohan/git/Factored-FB && python scripts/_smoke_rollout.py`
Expected: `TypeError: rollout() got an unexpected keyword argument 'record'` (or similar — current signature has no `record` param).

- [ ] **Step 3: Modify `envs/rollout.py`**

Replace the entire file with:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def rollout(
    env: Any,
    agent: Any,
    num_episodes: int,
    ctx: torch.Tensor | None = None,
    record: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[List[List[np.ndarray]]]]:
    observation, info = env.reset()
    returns, lengths, infos = [0.0], [0], [[info]]
    ctx = {} if ctx is None else {"z": ctx}
    frames: Optional[List[List[np.ndarray]]] = [[env.render()]] if record else None
    while True:
        input_dict = {"obs": torch.tensor(observation, device=agent.device, dtype=torch.float32)[None], **ctx}
        action = agent.act(**input_dict).cpu().numpy()[0]
        observation, reward, terminated, truncated, info = env.step(action)
        if record:
            frames[-1].append(env.render())
        done = terminated or truncated
        returns[-1] += reward
        lengths[-1] += 1
        infos[-1] += [info]
        if done:
            if len(returns) >= num_episodes:
                break
            observation, info = env.reset()
            returns.append(0.0)
            lengths.append(0)
            infos.append([info])
            if record:
                frames.append([env.render()])
    return {"reward": returns, "length": lengths}, infos, frames
```

- [ ] **Step 4: Run the smoke script and confirm both assertions PASS**

Run: `cd /home/adityamohan/git/Factored-FB && python scripts/_smoke_rollout.py`
Expected output:
```
OK frames shape=(<T>, 64, 64, 3)
OK record=False returns frames=None
```

- [ ] **Step 5: Delete the smoke script**

Run: `rm /home/adityamohan/git/Factored-FB/scripts/_smoke_rollout.py`

- [ ] **Step 6: Commit**

```bash
git add envs/rollout.py
git commit -m "feat(rollout): add record param to capture env.render() frames"
```

---

## Task 3: Update OGBenchEvaluator to accept video options + step arg

**Files:**
- Modify: `evals/ogbench.py`

- [ ] **Step 1: Replace `evals/ogbench.py` with the updated evaluator**

Replace the entire file with:

```python
"""evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Sample a batch from the offline buffer.
  2. Relabel with `get_relabel_fn(domain, task)(next.physics, action)`.
  3. Shift rewards by `shift_reward` (default 1.0).
  4. Infer z = project(sum r_i * B(next_obs_i)).
  5. Roll out `n_episodes` episodes via envs/rollout.py.
  6. Report mean/std success and reward per task, plus relabel-inference
     debug metrics (matches td_jepa's OGBenchRewardEvaluation.run output).
  7. Aggregate across tasks: eval/reward, eval/reward#std, eval/success.
  8. (Optional) Save first episode per task as <video_dir>/step_<N>/<task>.gif
     and upload to wandb as wandb.Video when save_videos=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
from envs.rollout import rollout


class OGBenchEvaluator:
    def __init__(
        self,
        domain: str,
        agent,
        offline_buffer,
        relabel_size: int = 10_000,
        n_episodes: int = 10,
        shift_reward: float = 1.0,
        obs_type: str = "state",
        seed: int = 0,
        device: str = "cpu",
        save_videos: bool = False,
        video_dir: Optional[Path | str] = None,
        use_wandb: bool = False,
    ):
        # Match td_jepa's OGBenchRewardEvaluation: do NOT build envs upfront.
        # Building all 5 eval envs + 5 relabel-fn envs at startup would add
        # ~50s of mujoco init even on runs that never call run(). Build them
        # lazily inside run() / _infer_z() and tear them down after each task.
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.obs_type = obs_type
        self.seed = seed
        self.device = device
        self.tasks = ALL_TASKS.get(domain, [])
        self.save_videos = save_videos
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.use_wandb = use_wandb
        if self.save_videos and self.video_dir is None:
            raise ValueError("save_videos=True requires video_dir to be set")

    def run(self, step: int = 0) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        all_rewards: list[np.ndarray] = []
        all_successes: list[np.ndarray] = []
        for task in self.tasks:
            env, _ = create_ogbench_env(task, seed=self.seed, obs_type=self.obs_type)
            try:
                z, relabel_metrics = self._infer_z(task)
                stats, infos, frames = rollout(
                    env, self.agent, self.n_episodes, ctx=z, record=self.save_videos
                )
            finally:
                env.close()

            per_ep_success = np.array(
                [any(step_info.get("success", False) for step_info in ep_info) for ep_info in infos],
                dtype=np.float32,
            )
            per_ep_reward = np.array(stats["reward"], dtype=np.float32)

            metrics[f"{task}/success"] = float(np.mean(per_ep_success))
            metrics[f"{task}/reward"] = float(np.mean(per_ep_reward))
            metrics[f"{task}/reward#std"] = float(np.std(per_ep_reward))
            for k, v in relabel_metrics.items():
                metrics[f"{task}/{k}"] = float(v)

            all_rewards.append(per_ep_reward)
            all_successes.append(per_ep_success)

            if self.save_videos and frames is not None and len(frames) > 0:
                self._save_gif(task=task, step=step, episode_frames=frames[0])

        if all_rewards:
            rewards_all = np.concatenate(all_rewards)
            successes_all = np.concatenate(all_successes)
            metrics["eval/reward"] = float(np.mean(rewards_all))
            metrics["eval/reward#std"] = float(np.std(rewards_all))
            metrics["eval/success"] = float(np.mean(successes_all))

        return metrics

    def _save_gif(self, *, task: str, step: int, episode_frames: list) -> None:
        import imageio

        ep_arr = np.stack(episode_frames)  # [T, H, W, 3] uint8
        step_dir = self.video_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        gif_path = step_dir / f"{task}.gif"
        imageio.mimsave(gif_path, ep_arr, format="GIF", fps=30, loop=0)
        print(f"[eval] wrote {gif_path}")

        if self.use_wandb:
            import wandb
            wandb.log(
                {f"eval_video/{task}": wandb.Video(str(gif_path), fps=30, format="gif")},
                step=step,
            )

    def _infer_z(self, task: str) -> Tuple[Tensor, Dict[str, float]]:
        relabel_fn = get_relabel_fn(self.domain, task)  # builds + closes a temp env internally
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return self.agent.model.sample_z(1).squeeze(0), {}

        batch = self.offline_buffer.sample(n)

        if "physics" not in batch["next"]:
            raise KeyError(
                "Buffer batch is missing batch['next']['physics']. "
                "Re-load the dataset with data.ogbench.load_ogbench_dataset "
                "(updated to place physics under next/)."
            )

        phys_np = batch["next"]["physics"].cpu().numpy()
        act_np = batch["action"].cpu().numpy()

        rewards_np = relabel_fn(phys_np, act_np)               # [N, 1]
        rewards_np = rewards_np + self.shift_reward
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)

        non_zero = int(np.count_nonzero(rewards_np.ravel()))
        relabel_metrics = {
            "relabel_reward#mean": float(np.mean(rewards_np)),
            "relabel_reward#nonzero": float(non_zero),
            "relabel_reward#zero": float(rewards_np.size - non_zero),
            "relabel_reward#num_samples": float(rewards_np.size),
        }

        z = self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device),
            reward=rewards,
        )
        return z, relabel_metrics
```

Note the rename `step` → `step_info` in the per-step success comprehension so it doesn't shadow the new `step: int` argument on `run()`.

- [ ] **Step 2: Static sanity check — confirm the module still imports**

Run: `cd /home/adityamohan/git/Factored-FB && python -c "from evals.ogbench import OGBenchEvaluator; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Check the validation guard fires**

Run: `cd /home/adityamohan/git/Factored-FB && python -c "
from evals.ogbench import OGBenchEvaluator
try:
    OGBenchEvaluator(domain='cube-single-play-v0', agent=None, offline_buffer=None, save_videos=True)
    print('FAIL: should have raised')
except ValueError as e:
    print(f'OK: {e}')
"`
Expected: `OK: save_videos=True requires video_dir to be set`.

- [ ] **Step 4: Commit**

```bash
git add evals/ogbench.py
git commit -m "feat(eval): write per-task eval GIFs + log wandb.Video"
```

---

## Task 4: Update `configs/train.yaml`

**Files:**
- Modify: `configs/train.yaml`

- [ ] **Step 1: Replace `configs/train.yaml`**

Replace the entire file with:

```yaml
defaults:
  - _self_
  - domain: antmaze_medium      # domain config selects its own agent via its defaults list
  - override hydra/launcher: basic

# Domain (overridden by domain/*.yaml)
domain: antmaze-medium-navigate-v0
obs_type: state
data_path: datasets
n_transitions: null         # null = load full dataset
load_n_episodes: 1000

# Training
num_train_steps: 1_000_000
batch_size: 256
eval_every: 50_000
log_every: 1_000
save_every: 200_000
seed: 42
device: cuda

# Eval
eval_n_episodes: 10
eval_relabel_size: 10_000
eval_shift_reward: 1.0
save_eval_videos: false     # opt-in: record first episode per task as GIF

# Logging
use_wandb: false
wandb_project: factored-fb
wandb_run_name: null            # default: ${domain}__${agent}__s${seed}
wandb_entity: null              # team / username; null -> default workspace
wandb_group: null               # useful for --multirun (e.g. group=fb_antmaze)
wandb_tags: []                  # list of strings, e.g. [baseline, antmaze]
wandb_save_code: true           # snapshot train.py and configs/ into the run

# Artifact paths (per-run, resolved against Hydra's runtime output dir)
save_dir: ${hydra:runtime.output_dir}/checkpoints
eval_videos_dir: ${hydra:runtime.output_dir}/eval_videos

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}__${domain}__${agent}__s${seed}
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}__s${seed}
```

- [ ] **Step 2: Verify Hydra resolves the new interpolations**

Run: `cd /home/adityamohan/git/Factored-FB && python -c "
import hydra
from omegaconf import OmegaConf
with hydra.initialize(config_path='configs', version_base='1.3'):
    cfg = hydra.compose(config_name='train', return_hydra_config=True)
    print('save_dir:', cfg.save_dir)
    print('eval_videos_dir:', cfg.eval_videos_dir)
    print('save_eval_videos:', cfg.save_eval_videos)
"`
Expected output (paths are absolute and end with `checkpoints` / `eval_videos`):
```
save_dir: /.../outputs/<timestamp>__.../checkpoints
eval_videos_dir: /.../outputs/<timestamp>__.../eval_videos
save_eval_videos: False
```

- [ ] **Step 3: Commit**

```bash
git add configs/train.yaml
git commit -m "config: per-run save_dir + eval_videos_dir under Hydra output_dir"
```

---

## Task 5: Wire video config into `train.py`

**Files:**
- Modify: `train.py:181-216`

- [ ] **Step 1: Replace the evaluator construction + save-dir block**

In `train.py`, find this block (currently around lines 181-194):

```python
    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        shift_reward=cfg.eval_shift_reward,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        device=cfg.device,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
```

Replace it with:

```python
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(cfg.eval_videos_dir)
    if cfg.save_eval_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        shift_reward=cfg.eval_shift_reward,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        device=cfg.device,
        save_videos=cfg.save_eval_videos,
        video_dir=video_dir,
        use_wandb=cfg.use_wandb,
    )
```

- [ ] **Step 2: Update the `evaluator.run()` call site**

In the same file, find (currently around line 216):

```python
        if cfg.eval_every > 0 and (step == 0 or step % cfg.eval_every == 0):
            print(f"[eval] step={step}")
            log_metrics(evaluator.run(), step, use_wandb=cfg.use_wandb, prefix="eval/reward/")
```

Replace with:

```python
        if cfg.eval_every > 0 and (step == 0 or step % cfg.eval_every == 0):
            print(f"[eval] step={step}")
            log_metrics(evaluator.run(step=step), step, use_wandb=cfg.use_wandb, prefix="eval/reward/")
```

- [ ] **Step 3: Static sanity check**

Run: `cd /home/adityamohan/git/Factored-FB && python -c "import train; print('ok')"`
Expected: `ok` (no syntax / import errors).

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "train: pass save_videos/video_dir to evaluator; run(step=step)"
```

---

## Task 6: Plumb `EXTRA_ARGS` through `scripts/run_sweep.sh`

**Files:**
- Modify: `scripts/run_sweep.sh`

- [ ] **Step 1: Add the env var default at the top**

Find the block of env-var defaults near the top of the script:

```bash
DATA_PATH="${DATA_PATH:-/dev/shm/datasets}"
```

Add immediately after it:

```bash
EXTRA_ARGS="${EXTRA_ARGS:-}"
```

- [ ] **Step 2: Append `$EXTRA_ARGS` to the `python train.py` invocation**

Find the `python train.py` call inside the launch loop:

```bash
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            domain="$domain" \
            ortho_coef="$ortho" \
            lr_b="$lr_b" \
            seed="$seed" \
            data_path="$DATA_PATH" \
            use_wandb=true \
            wandb_entity=amsks \
            wandb_group="$group" \
            "wandb_tags=[sweep-${TIMESTAMP}]" \
            wandb_run_name="$run_name" \
        > "$log" 2>&1
```

Replace with (note unquoted `$EXTRA_ARGS` so multiple tokens split into separate args):

```bash
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            domain="$domain" \
            ortho_coef="$ortho" \
            lr_b="$lr_b" \
            seed="$seed" \
            data_path="$DATA_PATH" \
            use_wandb=true \
            wandb_entity=amsks \
            wandb_group="$group" \
            "wandb_tags=[sweep-${TIMESTAMP}]" \
            wandb_run_name="$run_name" \
            $EXTRA_ARGS \
        > "$log" 2>&1
```

- [ ] **Step 3: Echo `EXTRA_ARGS` in the startup banner**

Find the banner block:

```bash
echo "[sweep] data_path:  $DATA_PATH"
echo ""
```

Insert before the blank line:

```bash
echo "[sweep] extra_args: ${EXTRA_ARGS:-(none)}"
```

So the result is:

```bash
echo "[sweep] data_path:  $DATA_PATH"
echo "[sweep] extra_args: ${EXTRA_ARGS:-(none)}"
echo ""
```

- [ ] **Step 4: Lint with `bash -n`**

Run: `bash -n /home/adityamohan/git/Factored-FB/scripts/run_sweep.sh`
Expected: no output (script parses).

- [ ] **Step 5: Dry-run with a no-op `python` to confirm the args propagate**

Run:
```bash
cd /home/adityamohan/git/Factored-FB && \
PATH=/tmp:$PATH cat > /tmp/python <<'EOF'
#!/usr/bin/env bash
echo "ARGS=$*" >&2
EOF
chmod +x /tmp/python
SEEDS="1" DOMAINS="cube_single" ORTHOS="1000" LR_BS="1e-4" \
  N_GPUS=1 JOBS_PER_GPU=1 EXTRA_ARGS="save_eval_videos=true foo=bar" \
  bash scripts/run_sweep.sh 2>&1 | grep -E "ARGS=|extra_args" | head -5
rm /tmp/python
```
Expected: a line `ARGS=domain=cube_single ortho_coef=1000 ... save_eval_videos=true foo=bar` and a line `[sweep] extra_args: save_eval_videos=true foo=bar`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_sweep.sh
git commit -m "sweep: plumb EXTRA_ARGS through python train.py invocation"
```

---

## Task 7: End-to-end smoke test on cube_single

**Files:**
- Create: `scripts/smoke_test_artifacts.sh`

- [ ] **Step 1: Create the smoke test script**

Write `scripts/smoke_test_artifacts.sh`:

```bash
#!/usr/bin/env bash
# Smoke test: tiny cube_single run with save_eval_videos=true.
# Verifies per-run artifact layout (checkpoints + eval_videos) end-to-end.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_PATH="${DATA_PATH:-/dev/shm/datasets}"

echo "[smoke] launching short cube_single run..."
python train.py \
    domain=cube_single \
    seed=0 \
    data_path="$DATA_PATH" \
    num_train_steps=100 \
    eval_every=50 \
    save_every=100 \
    log_every=100 \
    eval_n_episodes=2 \
    eval_relabel_size=64 \
    load_n_episodes=4 \
    save_eval_videos=true \
    use_wandb=false

# Find the most-recent outputs/ dir from this run.
LATEST_DIR=$(ls -1dt outputs/*cube-single*s0* 2>/dev/null | head -1)
echo "[smoke] inspecting $LATEST_DIR"

echo ""
echo "[smoke] checkpoints:"
ls -la "$LATEST_DIR/checkpoints/" || { echo "FAIL: no checkpoints/"; exit 1; }

echo ""
echo "[smoke] eval_videos:"
find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | sort

EXPECTED_GIFS=15   # 3 eval rounds (step 0, 50, 100) * 5 tasks
ACTUAL_GIFS=$(find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | wc -l)
if [ "$ACTUAL_GIFS" -ne "$EXPECTED_GIFS" ]; then
    echo "FAIL: expected $EXPECTED_GIFS GIFs, got $ACTUAL_GIFS"
    exit 1
fi

# Spot-check one GIF is a valid image.
SAMPLE_GIF=$(find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | head -1)
python -c "
import imageio.v3 as iio
arr = iio.imread('$SAMPLE_GIF')
assert arr.ndim == 4 and arr.shape[-1] == 3, f'bad shape: {arr.shape}'
assert arr.dtype.name == 'uint8', f'bad dtype: {arr.dtype}'
print(f'OK $SAMPLE_GIF shape={arr.shape}')
"

echo ""
echo "[smoke] PASS — $LATEST_DIR contains checkpoints/ and $ACTUAL_GIFS GIFs"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /home/adityamohan/git/Factored-FB/scripts/smoke_test_artifacts.sh`

- [ ] **Step 3: Run the smoke test**

Run: `bash /home/adityamohan/git/Factored-FB/scripts/smoke_test_artifacts.sh`

Expected: ends with `[smoke] PASS — outputs/...<timestamp>__cube-single-play-v0__fb_flowbc__s0/ contains checkpoints/ and 15 GIFs`. Also expect 1 checkpoint (`step_100.pt`) plus `final.pt` listed.

If it fails with "no checkpoints/" then `cfg.save_dir` resolution is broken — re-check Task 4 Step 2 output. If it fails with the GIF count, list the actual tree and diagnose.

- [ ] **Step 4: Verify a regression baseline (save_eval_videos=false)**

Run:
```bash
cd /home/adityamohan/git/Factored-FB && \
python train.py \
    domain=cube_single \
    seed=0 \
    data_path="${DATA_PATH:-/dev/shm/datasets}" \
    num_train_steps=100 \
    eval_every=50 \
    save_every=100 \
    log_every=100 \
    eval_n_episodes=2 \
    eval_relabel_size=64 \
    load_n_episodes=4 \
    use_wandb=false
```

Then check the latest output dir:

```bash
LATEST=$(ls -1dt outputs/*cube-single*s0* | head -1)
echo "checkpoints: $(ls $LATEST/checkpoints/ 2>/dev/null)"
echo "videos: $(ls $LATEST/eval_videos/ 2>/dev/null || echo '(absent, as expected)')"
```

Expected: `checkpoints: final.pt step_100.pt` and `videos: (absent, as expected)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_test_artifacts.sh
git commit -m "scripts: end-to-end smoke test for per-run artifacts"
```

---

## Task 8: Launch the targeted re-run

**Files:** (no edits — operational task)

- [ ] **Step 1: Decide seed count**

Default `SEEDS="1 2 3"` matches the original sweep's reduced-seed runs. Increase to `"1 2 3 4 5"` if you want more variance; decrease to `"1"` for a single sanity run before committing GPU-hours.

- [ ] **Step 2: Launch inside tmux**

Run:
```bash
tmux new -s video_replay -d "cd /home/adityamohan/git/Factored-FB && \
  DOMAINS=cube_single ORTHOS=1000 LR_BS=1e-4 SEEDS='1 2 3' \
  N_GPUS=1 JOBS_PER_GPU=3 \
  EXTRA_ARGS='save_eval_videos=true' \
  bash scripts/run_sweep.sh 2>&1 | tee logs/video_replay.log"
```

Expected immediately: tmux session `video_replay` exists (`tmux ls` shows it). Attach with `tmux a -t video_replay`.

- [ ] **Step 3: Watch the first eval round complete**

Inside tmux (or `tail -f logs/video_replay.log`), wait for the first `[eval] step=0` lines. ~1-2 min into each job, you should see `[eval] wrote outputs/.../eval_videos/step_0/cube-single-singletask-task1-v0.gif` etc.

Expected: 5 GIF write lines per job at step 0. If wandb is happy, the run page also shows `eval_video/<task>` panels.

- [ ] **Step 4: Spot-check disk layout for one in-flight run**

Run:
```bash
LATEST=$(ls -1dt outputs/*cube-single*s1*ortho1000* | head -1)
echo "=== $LATEST ==="
ls "$LATEST/checkpoints/" 2>/dev/null || echo "(no checkpoints yet)"
find "$LATEST/eval_videos/" -name '*.gif' | sort | head -10
```

Expected: `eval_videos/step_0/` populated with 5 GIFs; `checkpoints/` empty until `step % save_every == 0` (200k).

- [ ] **Step 5: Let it run to completion**

Total runtime is ~24-30 GPU-hours per seed at 1M steps. With 3 seeds on 1 GPU at JOBS_PER_GPU=3, expect ~30 hours wall-clock. Detach tmux (`Ctrl-b d`) and check back periodically.

- [ ] **Step 6 (post-run): Verify final artifact tree**

After the sweep completes, run:
```bash
for d in outputs/*cube-single*ortho1000*lrb1e-4*; do
    echo "=== $d ==="
    echo "  checkpoints: $(ls $d/checkpoints/ | wc -l) files"
    echo "  eval_videos: $(find $d/eval_videos/ -name '*.gif' | wc -l) gifs"
done
```

Expected per run: 6 checkpoints (step_200000, step_400000, step_600000, step_800000, step_1000000, final) and 100 GIFs (5 tasks × 21 eval rounds: step_0, step_50000, …, step_1000000).

- [ ] **Step 7: (Optional) commit the launch as a documented invocation**

If you want the exact command preserved in the repo, append it to `scripts/run_sweep.sh`'s top-of-file usage comments (don't add a new script — keep that single entry point). Otherwise skip this step.

---

## Self-Review

**Spec coverage:**
- Repoint `save_dir` → Task 4 ✓
- New `eval_videos_dir`, `save_eval_videos` → Task 4 ✓
- `record` in rollout → Task 2 ✓
- Evaluator GIF write + wandb.Video → Task 3 ✓
- `train.py` wiring → Task 5 ✓
- `EXTRA_ARGS` in sweep → Task 6 ✓
- `imageio` dep → Task 1 ✓
- Smoke test + regression check → Task 7 ✓
- Targeted re-run command → Task 8 ✓

**Placeholder scan:** No `TBD`, `TODO`, or "implement later" markers. All steps contain executable code or commands.

**Type/name consistency:**
- `record: bool = False` consistent across Task 2 (rollout) and Task 3 (evaluator passes `record=self.save_videos`).
- `save_videos` / `video_dir` / `use_wandb` consistent across Task 3 (init), Task 5 (train.py passes them).
- `run(step: int = 0)` matches Task 5 call site `evaluator.run(step=step)`.
- Variable rename `step → step_info` in the success comprehension noted in Task 3 to avoid shadowing the new `step` param.
- `step_{step}.pt` vs `eval_videos/step_{step}/<task>.gif` — checkpoint and video subdir naming both use `step_<N>` without zero-padding, consistent with current `train.py:209`.
