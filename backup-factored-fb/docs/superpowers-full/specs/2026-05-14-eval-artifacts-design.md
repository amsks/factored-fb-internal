# Per-run eval artifacts: checkpoints + GIF videos

**Date:** 2026-05-14
**Status:** Approved (awaiting user spec review)

## Motivation

We want to investigate what FB-FlowBC actually learns on `cube-single-play-v0`
when trained at `ortho_coef=1000`, `lr_b=1e-4` — a specific cell of
`scripts/run_sweep.sh`'s grid. To do that we need (a) the model checkpoints
across training so we can probe intermediate behaviour, and (b) recordings of
eval episodes so we can watch the policy.

Today neither is possible:

1. **Checkpoints overwrite across runs.** `configs/train.yaml:36` pins
   `save_dir: checkpoints/`, and Hydra 1.3 defaults to `hydra.job.chdir=False`.
   Every run resolves that path against `$REPO/checkpoints/`. The Hydra
   per-run directories in `outputs/<timestamp>__<domain>__<agent>__s<seed>/`
   exist but contain only `.hydra/` config snapshots + the log — no model
   weights, no artifacts.
2. **No video capture.** `envs/rollout.py` never calls `env.render()`, and
   `OGBenchEvaluator.run()` (`evals/ogbench.py:39-77`) returns scalar metrics
   only.

## Goals

- Each training run writes its checkpoints into a directory unique to that
  run (no cross-run clobbering).
- For a targeted re-run of `cube_single, ortho=1000, lr_b=1e-4`, capture eval
  episodes as `.gif` files and upload them to Weights & Biases as
  `wandb.Video` so they're browsable inline in the run UI.
- The targeted re-run uses the existing `scripts/run_sweep.sh` machinery
  (parameterised by env vars) — no parallel script to maintain.
- Existing sweeps (`bash scripts/run_sweep.sh` with defaults) must remain
  bit-for-bit identical in their training loop. Video capture is opt-in.

## Non-goals

- Changing eval frequency or eval episode count defaults.
- Changing the sweep grid (`ortho × lr_b × seed × domain`).
- Restructuring the evaluator beyond what's needed for frame capture.
- Render-resolution flags (we keep the existing 64×64 default).
- Saving every eval episode as a GIF — only the first per task (see below).
- Mp4 / other video formats. GIF only.

## Design overview

```
outputs/<timestamp>__cube-single-play-v0__fb_flowbc__s1__ortho1000__lrb1e-4/
├── .hydra/                          # unchanged — Hydra config snapshot
├── train.log                        # unchanged
├── checkpoints/                     # NEW location (was: $REPO/checkpoints/)
│   ├── step_200000.pt
│   ├── step_400000.pt
│   ├── step_600000.pt
│   ├── step_800000.pt
│   ├── step_1000000.pt              # save_every=200k triggers at final step too
│   └── final.pt                     # separate post-loop save
└── eval_videos/                     # NEW
    ├── step_0/
    │   ├── cube-single-singletask-task1-v0.gif
    │   ├── cube-single-singletask-task2-v0.gif
    │   ├── cube-single-singletask-task3-v0.gif
    │   ├── cube-single-singletask-task4-v0.gif
    │   └── cube-single-singletask-task5-v0.gif
    ├── step_50000/
    │   └── ...
    ├── ...
    └── step_1000000/                # final eval — sorts with others numerically
        └── ...
```

Five GIFs per eval round (one per task — the first of `n_episodes=10`
episodes), 20 eval rounds over a 1M-step run → 100 GIFs per run at ~50 KB
each = ~5 MB on disk. Same payload uploaded to wandb under
`eval_video/<task>` keyed by step.

## Components

### 1. Config (`configs/train.yaml`)

Replace `save_dir: checkpoints/` with absolute interpolation against Hydra's
runtime output dir, and add two new keys:

```yaml
save_dir: ${hydra:runtime.output_dir}/checkpoints
eval_videos_dir: ${hydra:runtime.output_dir}/eval_videos
save_eval_videos: false
```

We use `${hydra:runtime.output_dir}` rather than setting
`hydra.job.chdir=true` so that any cwd-relative file IO elsewhere stays
unaffected. `${hydra:runtime.output_dir}` resolves correctly for both
single-run mode (`hydra.run.dir`) and multirun mode
(`hydra.sweep.dir/hydra.sweep.subdir`).

### 2. Frame capture (`envs/rollout.py`)

Add an optional `record=False` parameter. When true, call `env.render()`
after each `env.reset()` and `env.step()` and accumulate frames per episode.

```python
def rollout(env, agent, num_episodes, ctx=None, record=False):
    observation, info = env.reset()
    returns, lengths, infos = [0.0], [0], [[info]]
    ctx = {} if ctx is None else {"z": ctx}
    frames = [[env.render()]] if record else None
    while True:
        input_dict = {"obs": torch.tensor(observation, ...)[None], **ctx}
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
            returns.append(0.0); lengths.append(0); infos.append([info])
            if record:
                frames.append([env.render()])
    return {"reward": returns, "length": lengths}, infos, frames
```

`env.render()` returns `np.ndarray[H, W, 3] uint8` at the 64×64 default set
by `create_ogbench_env(render_height=64, render_width=64)`
(`envs/ogbench.py:55-67`).

**Backward compatibility:** the function returns a 3-tuple instead of a
2-tuple. The only caller is `OGBenchEvaluator`, which we update in lockstep.
When `record=False` the third element is `None` and the cost of capture is
zero — branch never executes.

### 3. Evaluator (`evals/ogbench.py`)

`OGBenchEvaluator.__init__` gains three params:

```python
def __init__(self, ..., save_videos=False, video_dir=None, use_wandb=False):
    ...
    self.save_videos = save_videos
    self.video_dir = Path(video_dir) if video_dir is not None else None
    self.use_wandb = use_wandb
```

`run()` gains a `step: int` argument so it knows which subdir to write to.
Inside the per-task loop:

```python
stats, infos, frames = rollout(env, self.agent, self.n_episodes, ctx=z,
                                record=self.save_videos)
...
if self.save_videos and frames:
    import imageio
    ep_frames = np.stack(frames[0])    # [T, H, W, 3] uint8
    step_dir = self.video_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    gif_path = step_dir / f"{task}.gif"
    imageio.mimsave(gif_path, ep_frames, format="GIF", fps=30, loop=0)
    if self.use_wandb:
        import wandb
        wandb.log(
            {f"eval_video/{task}": wandb.Video(str(gif_path), fps=30, format="gif")},
            step=step,
        )
```

Only the first episode (`frames[0]`) is saved — matches the "1 GIF per task
per eval" cost target. Eval reset uses the same seed, so frame 0 is
deterministic across eval rounds.

### 4. Train wiring (`train.py`)

Three diffs:

1. After `save_dir = Path(cfg.save_dir); save_dir.mkdir(...)`, compute and
   mkdir `video_dir = Path(cfg.eval_videos_dir)` (only when
   `cfg.save_eval_videos`).
2. Pass `save_videos=cfg.save_eval_videos`, `video_dir=video_dir`,
   `use_wandb=cfg.use_wandb` into `OGBenchEvaluator(...)`.
3. Change both `evaluator.run()` call sites (the t=0 / `eval_every` one
   inside the main loop) to `evaluator.run(step=step)`.

The final `torch.save(..., "final.pt")` is unchanged. With the defaults
`num_train_steps=1_000_000`, `eval_every=50_000`, the loop already triggers
an eval at `step == 1_000_000`, so the GIFs at `eval_videos/step_1000000/`
correspond to the same weights as `checkpoints/final.pt`.

### 5. Sweep script (`scripts/run_sweep.sh`)

Add support for an optional `EXTRA_ARGS` env var that gets appended verbatim
to the `python train.py ...` invocation. Default empty.

```bash
EXTRA_ARGS="${EXTRA_ARGS:-}"
...
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

(Unquoted `$EXTRA_ARGS` so multiple tokens split into separate args.)

### 6. Dependency (`requirements.txt`)

Add `imageio>=2.31` explicitly. Likely already pulled in transitively
through other deps, but pinning it makes intent clear.

## Targeted re-run command

```bash
DOMAINS=cube_single \
ORTHOS=1000 \
LR_BS=1e-4 \
SEEDS="1 2 3" \
N_GPUS=1 \
JOBS_PER_GPU=3 \
EXTRA_ARGS="save_eval_videos=true" \
bash scripts/run_sweep.sh
```

3 seeds × 1 domain × 1 ortho × 1 lr_b = 3 jobs. On a single GPU, all three
can run in parallel (`JOBS_PER_GPU=3`). Each lands in its own Hydra
`outputs/...` directory with full `checkpoints/` + `eval_videos/`.

## Testing strategy

**Smoke test (fast, no wandb):**

```bash
python train.py \
  domain=cube_single \
  num_train_steps=100 \
  eval_every=50 \
  save_every=100 \
  save_eval_videos=true \
  use_wandb=false
```

Should produce:
- `outputs/.../checkpoints/step_100.pt` + `final.pt`
- `outputs/.../eval_videos/step_0/*.gif` (5 files)
- `outputs/.../eval_videos/step_50/*.gif` (5 files)
- `outputs/.../eval_videos/step_100/*.gif` (5 files)

**Regression check (videos disabled):**

```bash
python train.py \
  domain=cube_single \
  num_train_steps=100 \
  eval_every=50 \
  save_every=100 \
  use_wandb=false
```

Should produce the same `.pt` files as above, no `eval_videos/` directory at
all, and identical scalar eval metrics to a pre-change run (we'll diff
`train/` and `eval/` keys against a baseline run on `main`).

**GIF integrity:**

```bash
python -c "
import imageio.v3 as iio
arr = iio.imread('outputs/.../eval_videos/step_50/cube-single-singletask-task1-v0.gif')
print(arr.shape, arr.dtype)  # expect (T, 64, 64, 3) uint8
"
```

## Risks + open questions

- **Render speed:** `env.render()` on cube envs typically takes 1-3 ms per
  call. With ~200 steps × 5 tasks × `n_episodes=10` = 10k steps per eval, an
  unguarded record would add ~30s per eval. We only render 1 episode per
  task = 1k frames per eval ≈ 3s — acceptable.
- **`wandb.Video` quirks:** giving it a file path bypasses the
  numpy-array codepath that re-encodes to mp4 internally. Direct GIF upload
  is supported.
- **Memory:** at 64×64×3 uint8 × 1k frames per eval, peak frame memory is
  ~12 MB — negligible.

## Files touched

| File | Change |
|---|---|
| `configs/train.yaml` | Repoint `save_dir`; add `eval_videos_dir`, `save_eval_videos` |
| `envs/rollout.py` | Add `record=False`; return `(stats, infos, frames)` |
| `evals/ogbench.py` | New init kwargs; `run()` accepts `step`; write GIF + wandb.Video |
| `train.py` | Mkdir video dir; pass kwargs to evaluator; `run(step=step)` |
| `scripts/run_sweep.sh` | Append `$EXTRA_ARGS` to the train invocation |
| `requirements.txt` | Add `imageio>=2.31` |

Files **not** touched: `agents/`, `data/`, `buffers/`, `normalizers.py`,
`nn_models.py`, `base_*.py`, `configs/agent/*`, `configs/domain/*`.
