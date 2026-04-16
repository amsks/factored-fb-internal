# FB Baseline on OGBench — Design Spec

**Date:** 2026-04-16  
**Status:** Approved  
**Goal:** Port ForwardBackward (FB) representations from `td_jepa/metamotivo` into a clean, standalone `Factored-FB` repo that can run FB baselines on OGBench tasks. This provides the baseline on top of which the "Factored" FB variant will be built.

---

## Source

All core logic is extracted from `/Users/adityamohan/git/Austin/td_jepa/metamotivo/`. DMC support, the `exca` cluster launcher, `tyro` CLI, and all non-FB agents (ICVF, TD-JEPA, Laplacian, BYOL) are dropped.

---

## Directory Structure

```
Factored-FB/
├── agents/
│   ├── fb/
│   │   ├── agent.py          # FBAgent (antmaze + state-based tasks)
│   │   ├── model.py          # FBModel: ForwardMap, BackwardMap, LeftEncoder, Actor
│   │   └── flow_bc/
│   │       └── agent.py      # FBFlowBCAgent (cube/puzzle/scene manipulation)
│   └── base.py               # Shared base classes / soft-update utilities
├── envs/
│   └── ogbench.py            # OGBench env factory, PixelWrapper, reward relabelers
├── buffers/
│   └── transition.py         # DictBuffer (state) + ParallelBuffer (pixels)
├── data/
│   └── ogbench.py            # Offline .npz loader → buffer
├── evals/
│   └── ogbench.py            # Reward inference + rollout evaluation
├── nn_models.py              # ForwardMap, BackwardMap, LeftEncoder, Actor nets
├── train.py                  # Main entry point (argparse)
├── configs/                  # Stub for future Hydra YAML configs
│   └── .gitkeep
└── requirements.txt          # Dependencies for uv pip
```

---

## Components

### `nn_models.py`
Neural network building blocks extracted from `td_jepa/metamotivo/nn_models.py`:
- `BackwardMap`: obs → z (LayerNorm → Tanh → MLP → normalize to sqrt(z_dim))
- `ForwardMap`: (L(obs), z, action) → z prediction; supports parallel ensemble heads
- `LeftEncoder`: obs → intermediate representation (shallow MLP)
- `SimpleActor`: (L(obs), z) → Gaussian action distribution
- `NoiseConditionedActor`: (obs, z, noise, t) → velocity field (for flow matching)

### `agents/fb/model.py`
`FBModel`: wraps all networks + target network copies. Handles:
- z normalization (`project_z`)
- Mixed z sampling (50% from `B(next_obs)`, 50% random Gaussian)
- Soft target updates
- Reward inference (solves `min ||B·w - r||^2` → returns z)

### `agents/fb/agent.py`
`FBAgent`: training logic for antmaze / state-obs tasks.
- FB loss: diagonal + off-diagonal terms on `F·B^T` matrix
- Orthonormality regularizer on B: `ortho_coef * (||Cov_offdiag||^2 + ||1 - Cov_diag||)`
- Pessimism via ensemble disagreement: `Q_target = Q_mean - pessimism * Q_std`
- Three optimizers: `opt_fb` (F+B+L), `opt_actor`

### `agents/fb/flow_bc/agent.py`
`FBFlowBCAgent`: extends `FBAgent` for manipulation tasks.
- Replaces Gaussian actor with flow-matching (`NoiseConditionedActor`)
- Adds behavior cloning loss (`bc_coeff` weighted)
- Used for cube, puzzle, scene domains where action distributions are multi-modal

### `envs/ogbench.py`
- `create_ogbench_env(domain, obs_type, ...)`: wraps `ogbench.make_env_and_datasets`
- `PixelWrapper`: frame stacking along channel dim
- Task-specific reward functions: `cube_reward_fn`, `puzzle_reward_fn`, `scene_reward_fn`, `ant_reward_fn`
- Domain lists: `ANT_DOMAINS`, `CUBE_DOMAINS`, `PUZZLE_DOMAINS`, `SCENE_DOMAINS`

### `data/ogbench.py`
- Loads offline `.npz` episodes from `data_path/domain/buffer/*.npz`
- Returns `DictBuffer` (state) or `ParallelBuffer` (pixels)
- Default: 1000 episodes

### `buffers/transition.py`
- `DictBuffer`: in-memory tensor storage, O(1) uniform sampling
- `ParallelBuffer`: on-demand episode loading, supports frame stacking, caps GPU memory

### `evals/ogbench.py`
- `OGBenchEvaluator`: for each eval task, runs reward inference from a batch of relabelled transitions, then rolls out the policy for N episodes
- Logs: `{domain}/success`, `{domain}/reward`

---

## Data Flow

```
.npz files
    └── data/ogbench.py → DictBuffer/ParallelBuffer
                               │
                         train.py loop
                               │
                    ┌──────────┴──────────┐
                    │                     │
              sample batch           eval every N steps
                    │                     │
              FBAgent.update()    OGBenchEvaluator.run()
              ├─ encode obs              │
              ├─ sample mixed z     reward_inference(B)
              ├─ FB loss                 │
              ├─ ortho loss          rollout(env, agent)
              └─ actor loss              │
                                    log success/reward
```

---

## Entry Point (`train.py`)

```bash
python train.py \
  --domain antmaze-medium-navigate-v0 \
  --data_path datasets/ \
  --agent fb \
  --num_train_steps 1000000 \
  --seed 42

python train.py \
  --domain cube-single-play-v0 \
  --data_path datasets/ \
  --agent fb_flowbc \
  --bc_coeff 0.3 \
  --ortho_coef 100
```

All hyperparameters are `argparse` arguments with defaults matching td_jepa's OGBench configs. `--use_wandb` flag gates W&B logging. The argparse structure is kept flat and explicit so migration to `hydra.main` is a straightforward swap.

### Key argparse defaults (OGBench):

| Arg | Default | Notes |
|-----|---------|-------|
| `z_dim` | 50 | Latent goal space dim |
| `L_dim` | 50 | Left encoder output dim |
| `batch_size` | 256 | |
| `discount` | 0.99 | |
| `ortho_coef` | 1.0 | 100–1000 for manipulation |
| `f_target_tau` | 0.005 | |
| `b_target_tau` | 0.005 | |
| `bc_coeff` | 0.0 | >0 for FlowBC only |
| `train_goal_ratio` | 0.5 | Fraction of z from B(next_obs) |
| `num_train_steps` | 1_000_000 | |
| `eval_every` | 50_000 | |
| `lr_f` | 1e-4 | |
| `lr_b` | 1e-4 | |
| `lr_actor` | 1e-4 | |

---

## Dependencies (`requirements.txt`)

```
torch>=2.3
ogbench>=1.1.2
wandb>=0.22.3
h5py>=3.11.0
safetensors>=0.4.4
tqdm
numpy
```

Python >= 3.11. Install with:
```bash
uv pip install -r requirements.txt
```

No `tyro`, no `exca`, no `pandas`.

---

## What Is NOT Included

- DMC environments or data loaders
- Pixel/visual OGBench support (can be added later from td_jepa)
- Cluster/SLURM launcher
- Non-FB agents (ICVF, TD-JEPA, Laplacian, BYOL)
- Hyperparameter sweep infrastructure (deferred to Hydra)

---

## Future Hydra Migration Path

When ready:
1. Replace `argparse` block in `train.py` with `@hydra.main(config_path="configs", config_name="train")`
2. Move defaults from argparse into `configs/train.yaml`
3. Add override configs per domain in `configs/domain/`

No changes needed to agent, env, buffer, or eval code.
