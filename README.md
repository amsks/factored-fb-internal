# Factored-FB

A clean, standalone implementation of **Forward-Backward (FB) representations** for offline goal-conditioned reinforcement learning on [OGBench](https://github.com/seohongpark/ogbench).

This repo is a faithful port of the FB / FlowBC agents from [facebookresearch/td_jepa](https://github.com/facebookresearch/td_jepa/tree/main/metamotivo) to state-based OGBench. DMC support, the `exca` launcher, and all non-FB agents are dropped.

---

## Method

FB representations decompose the Q-function as:

```
Q(s, a, z) = F(s, z, a) · B(s')
```

**F** (forward map) predicts successor features; **B** (backward map) maps observations to a latent goal space **z**. At test time a task is specified by inferring **z** as `z = project(Σ r_i · B(next_obs_i))` on relabelled offline transitions — no fine-tuning required.

- `fb_flowbc`: a noise-conditioned actor distilled from a flow-matching velocity field
  `v_θ(obs, x_t, t)`. The velocity field is trained on dataset actions; the actor is
  trained with `Q = F·z` plus a BC term that distills the ODE rollout of `v_θ`.
  Used for cube / scene / puzzle.

Two agent variants:

| Agent | Actor | Use case |
|---|---|---|
| `fb` | TruncatedNormal (DDPG-style mean) | Antmaze / locomotion |
| `fb_flowbc` | Flow-matching velocity field + distilled noise actor | Cube / puzzle / scene manipulation |

---

## Installation

**Python ≥ 3.11** required. The repo is set up for [`uv`](https://github.com/astral-sh/uv) (a `.venv/` is committed via `pyvenv.cfg`), but plain `pip` works too.

```bash
git clone <repo-url> Factored-FB
cd Factored-FB
uv venv --python 3.11             # creates .venv/
source .venv/bin/activate
make install                      # uv pip install -r requirements.txt
```

CUDA is the default device; CPU users should pass `device=cpu` on the CLI.

### Verify the install (no data needed)

```bash
make smoke
```

Runs `scripts/smoke.py` — builds both `FBAgent` and `FBFlowBCAgent` from every domain config and runs five update steps on synthetic Box observations. ~20 seconds, no OGBench dataset required.

---

## Getting the OGBench data

The training loop expects per-episode `.npz` files at `<data_path>/<domain>/buffer/*.npz` (default `data_path: datasets`). One command pulls down the official OGBench datasets and splits them into the expected layout:

```bash
make data                                             # all state-based domains
# or a single env:
python scripts/extract_ogbench.py --env antmaze-medium-navigate-v0 --output_folder datasets
```

The script (`scripts/extract_ogbench.py`) calls `ogbench.utils.make_env_and_datasets` to download into OGBench's default cache, then writes `datasets/<env>/buffer/episode_XXXXXX_<length>.npz` with the keys `observation`, `action`, `physics`, `reward`, `discount`.

If you already have the OGBench `.npz` files cached somewhere else:

```bash
python scripts/extract_ogbench.py --dataset_path /path/to/cache --output_folder datasets
```

---

## Quick start

```bash
# Antmaze medium (standard FB)
make train-ant-medium                                 # or: python train.py domain=antmaze_medium

# Cube manipulation (FlowBC)
make train-cube                                       # or: python train.py domain=cube_single

# Seeded sweep
make sweep-ant-medium                                 # or: python train.py --multirun seed=1,2,3 domain=antmaze_medium

# CPU run (no CUDA)
python train.py domain=antmaze_medium device=cpu

# Override anything on the CLI
python train.py domain=antmaze_medium ortho_coef=100 lr_b=1e-5 seed=7

# With wandb logging (entity/group help when running sweeps under a shared account)
python train.py domain=antmaze_medium use_wandb=true \
    wandb_entity=my-team wandb_group=fb-antmaze wandb_tags='[baseline]'
```

**Domain configs** live in `configs/domain/*.yaml` and each one selects the right agent via its own `defaults:` list — `domain=antmaze_*` pulls in `fb.yaml`; `domain=cube_*`, `scene`, `puzzle_3x3` pull in `fb_flowbc.yaml`. You normally do not need to pass `agent=` explicitly.

Available domain config names: `antmaze_medium`, `antmaze_large`, `antmaze_giant`, `cube_single`, `cube_double`, `scene`, `puzzle_3x3`.

---

## Configuration

All hyperparameters live in `configs/`:

```
configs/
├── train.yaml                   ← top-level (training loop, eval, logging)
├── agent/
│   ├── fb.yaml                  ← standard FB defaults (td_jepa OGBench launch)
│   └── fb_flowbc.yaml           ← inherits fb.yaml; noise actor + flow-matching VF
└── domain/
    ├── antmaze_medium.yaml      ← pulls in /agent: fb
    ├── antmaze_large.yaml       ← pulls in /agent: fb
    ├── antmaze_giant.yaml       ← pulls in /agent: fb
    ├── cube_single.yaml         ← pulls in /agent: fb_flowbc; bc_coeff=3.0
    ├── cube_double.yaml         ← pulls in /agent: fb_flowbc; bc_coeff=3.0
    ├── scene.yaml               ← pulls in /agent: fb_flowbc; bc_coeff=3.0
    └── puzzle_3x3.yaml          ← pulls in /agent: fb_flowbc; bc_coeff=3.0
```

Override anything on the CLI:

```bash
python train.py domain=antmaze_medium ortho_coef=100 lr_b=1e-5 seed=7
```

### Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `z_dim` | 50 | Latent goal dimension |
| `L_dim` | 50 | Left-encoder output dimension |
| `hidden_dim` (F, B, left_encoder, actor) | 512 | Width of MLP blocks |
| `forward.hidden_layers` | 2 | Forward map depth |
| `backward.hidden_layers` | 4 | Backward map depth |
| `left_encoder.hidden_layers` | 4 | Left encoder depth |
| `forward.num_parallel` | 2 | F ensemble heads |
| `ortho_coef` | 1.0 | B orthonormality weight (sweep 100/1000 for manipulation) |
| `actor_std` | 0.2 | TruncatedNormal std (FB only; ignored by FlowBC) |
| `stddev_clip` | 0.3 | Action sample clip |
| `train_goal_ratio` | 0.5 | Fraction of z drawn from B(next_obs[perm]) |
| `f_target_tau`, `b_target_tau` | 0.005 | Polyak soft-update rates |
| `bc_coeff` (FlowBC) | 0.3 antmaze / 3.0 manip | Distillation BC weight |
| `flow_steps` (FlowBC) | 10 | Euler steps for the action-flow ODE |
| `actor_encode_obs` | false | If true, actor sees the left-encoded representation |

---

## Directory structure

```
Factored-FB/
├── agents/
│   ├── base.py                 # BaseAgent ABC, soft_update / hard_update
│   └── fb/
│       ├── model.py            # FBModel: networks + encoders + targets + z utilities
│       ├── agent.py            # FBAgent: M-matrix TD, contrastive ortho, F·z actor
│       └── flow_bc/
│           ├── model.py        # FBFlowBCModel: adds _actor_vf velocity field
│           └── agent.py        # FBFlowBCAgent: FM loss + RL + distillation BC
├── buffers/
│   └── transition.py           # DictBuffer — nested-dict circular buffer
├── configs/                    # Hydra YAML configs (see Configuration)
├── data/
│   └── ogbench.py              # load_ogbench_dataset → DictBuffer
├── envs/
│   ├── ogbench.py              # create_ogbench_env, get_relabel_fn, ALL_TASKS
│   ├── rollout.py              # rollout(env, agent, n_episodes, ctx)
│   ├── wrappers.py             # PixelWrapper (unused — kept for future pixel work)
│   └── gym_spaces.py           # Space JSON serialisation (for checkpoint metadata)
├── evals/
│   └── ogbench.py              # OGBenchEvaluator: relabel → infer z → rollout
├── scripts/
│   ├── extract_ogbench.py      # Download + split OGBench .npz → per-episode files
│   └── smoke.py                # Synthetic pipeline smoke (no data needed)
├── base_config.py              # Pydantic BaseConfig
├── base_model.py               # BaseModel, save_model, load_model
├── nn_models.py                # All network building blocks
├── normalizers.py              # BatchNorm / Identity / RGB normalizers
├── utils.py                    # set_seed_everywhere, EveryNStepsChecker, …
├── train.py                    # Hydra entry point
├── Makefile
└── requirements.txt
```

---

## Data layout

After `make data`, episode files live at `<data_path>/<domain>/buffer/episode_XXXXXX_<length>.npz` (default `data_path` = `datasets/`). Each `.npz` contains arrays of shape `[T, ·]` over one episode:

| Key | Shape | Description |
|---|---|---|
| `observation` | [T, obs\_dim] | Proprioceptive state |
| `action` | [T, act\_dim] | |
| `physics` | [T, phys\_dim] | Full physics state; used for next-state reward relabelling |
| `discount` | [T] | 0 at terminal steps |

`load_ogbench_dataset` converts T-step episodes into T−1 transition tuples and
stores them in a `DictBuffer` under the layout:

```
batch["observation"]            [B, obs_dim]
batch["action"]                 [B, act_dim]
batch["next"]["observation"]    [B, obs_dim]
batch["next"]["physics"]        [B, phys_dim]
batch["next"]["terminated"]     [B, 1]
```

---

## Evaluation

`OGBenchEvaluator` iterates all 5 task variants per domain. For each:

1. Sample `eval_relabel_size` transitions from the offline buffer.
2. Relabel rewards with `get_relabel_fn(domain, task)(batch["next"]["physics"], action)`.
3. Add `eval_shift_reward` (default 1.0) — shifts sparse rewards from `{−k, …, 0}` to `{1−k, …, 1}`.
4. Compute `z = project(Σ r_i · B(next_obs_i))` via `agent.model.reward_inference`.
5. Roll out `eval_n_episodes` episodes via `envs/rollout.py`.
6. Log `eval/{task}/success` and `eval/{task}/reward`.

### Per-run artifacts (checkpoints + eval videos)

Every run writes its artifacts into its own Hydra output directory
(`outputs/<timestamp>__<domain>__<agent>__s<seed>/`), so parallel sweep runs
no longer clobber a shared `checkpoints/`:

```
outputs/<run>/
├── checkpoints/        # step_<N>.pt at every save_every + final.pt
└── eval_videos/        # only when save_eval_videos=true
    └── step_<N>/<task>.gif
```

Video capture is **opt-in and off by default** (`save_eval_videos: false`).
When enabled, the first eval episode per task is recorded as a GIF and, with
`use_wandb=true`, uploaded as a `wandb.Video` under `eval_video/<task>`. Paths
are configurable via `save_dir` / `eval_videos_dir` (both default to the run's
Hydra output dir). Headless rendering works out of the box (`MUJOCO_GL=egl`).

```bash
# Capture eval GIFs for a run
python train.py domain=cube_single save_eval_videos=true

# Same, via the sweep script (EXTRA_ARGS is appended to every job)
EXTRA_ARGS="save_eval_videos=true" bash scripts/run_sweep.sh

# End-to-end check of the artifact layout
bash scripts/smoke_test_artifacts.sh
```

---

## GCIQL baseline (OGBench, JAX)

For comparing FB against OGBench's GCIQL on cube manipulation. GCIQL runs as a
**verbatim vendored copy of OGBench** (`third_party/ogbench/`, JAX) in a
**separate environment** — it never shares a process or venv with the PyTorch
FB track. A thin Hydra launcher (`run_gciql.py`) translates config into
OGBench's exact CLI flags and subprocesses OGBench's own `impls/main.py`.

Default storage is tmpfs (`/dev/shm`, RAM-backed, no sudo, **ephemeral on
reboot**). For runs that must survive a reboot, use the NVMe override.

```bash
# tmpfs (default, no sudo):
make install-jax-shm                   # isolated .venv-jax in /dev/shm
make smoke-gciql                       # end-to-end check (state + visual)
VARIANT=state  SEEDS="0 1 2" bash scripts/run_gciql.sh
VARIANT=visual SEEDS="0 1 2" bash scripts/run_gciql.sh

# durable NVMe (one-time, real SSH terminal — needs sudo):
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/scratch1 && sudo mount /dev/nvme1n1 /mnt/scratch1
sudo chown "$USER":"$USER" /mnt/scratch1
make install-jax
python run_gciql.py --config-name cube_single_state storage=nvme
```

Datasets/checkpoints live under the chosen storage root; OGBench's
`~/.ogbench` cache is symlinked there. FB vs GCIQL is compared via each
method's own success metric (their evaluators differ by design); align runs in
wandb with a shared `run_group`.

---

## Citation

```bibtex
@article{bagatella2025td,
  title={TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning},
  author={Bagatella, Marco and Pirotta, Matteo and Touati, Ahmed and Lazaric, Alessandro and Tirinzoni, Andrea},
  journal={arXiv preprint arXiv:2510.00739},
  year={2025}
}
```

---

## License

Code adapted from [facebookresearch/td\_jepa](https://github.com/facebookresearch/td_jepa) is subject to the **CC BY-NC 4.0** license.