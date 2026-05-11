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

**Python ≥ 3.11** required.

```bash
git clone https://github.com/your-org/Factored-FB
cd Factored-FB
make install          # uv pip install -r requirements.txt
make install-dev      # also installs ruff + pytest
```

Or with plain pip:

```bash
pip install -r requirements.txt
```

---

## Quick start

```bash
# Verify everything is wired (~30 s, no full dataset needed)
make smoke

# Full antmaze medium run
make train-ant-medium

# Cube manipulation with FlowBC actor
make train-cube

# Seeded sweep
make sweep-ant-medium
```

Or directly with Hydra:

```bash
python train.py domain=antmaze-medium-navigate-v0
python train.py domain=antmaze-medium-navigate-v0 agent=fb_flowbc
python train.py domain=cube_single                         # domain preset
python train.py --multirun seed=1,2,3 domain=antmaze-medium-navigate-v0
```

---

## Configuration

All hyperparameters live in `configs/`:

```
configs/
├── train.yaml              ← base config (all defaults)
├── agent/
│   ├── fb.yaml
│   └── fb_flowbc.yaml      ← bc_coeff=0.3, ortho_coef=100
├── domain/
│   ├── antmaze_medium.yaml
│   ├── antmaze_large.yaml
│   └── cube_single.yaml    ← sets agent=fb_flowbc automatically
└── experiment/
    └── smoke_test.yaml     ← 500-step fast pipeline check
```

Override anything on the command line:

```bash
python train.py domain=antmaze-medium-navigate-v0 ortho_coef=10 lr_f=3e-4 seed=7
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
│   ├── base.py               # BaseAgent ABC, soft_update / hard_update
│   └── fb/
│       ├── model.py          # FBModel: networks + targets + z utilities
│       ├── agent.py          # FBAgent: FB/ortho/actor losses
│       └── flow_bc/
│           └── agent.py      # FBFlowBCAgent: NoiseConditionedActor + BC
├── buffers/
│   └── transition.py         # DictBuffer — generic nested-dict circular buffer
├── configs/                  # Hydra YAML configs
├── data/
│   └── ogbench.py            # load_transitions, load_ogbench_data → DictBuffer
├── envs/
│   ├── ogbench.py            # create_ogbench_env, get_relabel_fn, ALL_TASKS
│   ├── rollout.py            # rollout(env, agent, n_episodes, ctx)
│   ├── wrappers.py           # PixelWrapper (frame stacking)
│   └── gym_spaces.py         # Space JSON serialisation
├── evals/
│   └── ogbench.py            # OGBenchEvaluator: relabel → infer z → rollout
├── tests/                    # pytest suite (see Testing section above)
├── base_config.py            # Pydantic BaseConfig (from metamotivo)
├── base_model.py             # BaseModel, save_model, load_model
├── nn_models.py              # All network building blocks (from metamotivo)
├── normalizers.py            # BatchNorm / Identity / RGB normalizers
├── utils.py                  # set_seed_everywhere, EveryNStepsChecker, …
├── train.py                  # Main entry point (Hydra)
├── Makefile
└── requirements.txt
```

---

## Data layout

OGBench episode files are expected at:

```
<data_path>/<domain>/buffer/ep0.npz
<data_path>/<domain>/buffer/ep1.npz
...
```

Each `.npz` contains arrays of shape `[T, ·]` over one episode:

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