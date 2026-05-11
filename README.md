# Factored-FB

A clean, standalone implementation of **Forward-Backward (FB) representations** for offline goal-conditioned reinforcement learning on [OGBench](https://github.com/seohongpark/ogbench).

This repo is a faithful port of the FB / FlowBC agents from [facebookresearch/td_jepa](https://github.com/facebookresearch/td_jepa/tree/main/metamotivo) to state-based OGBench. DMC support, the `exca` launcher, and all non-FB agents are dropped.

---

## Method

FB representations decompose the Q-function as:

```
Q(s, a, z) = F(s, z, a) · B(s')
```

**F** (forward map) predicts successor features; **B** (backward map) maps observations to a latent goal space **z**. At test time a task is specified by inferring **z** via a closed-form least-squares solve on relabelled offline transitions — no fine-tuning required.

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
| `hidden_dim` | 1024 | Width of F and actor MLPs |
| `n_ensemble` | 2 | ForwardMap parallel heads |
| `ortho_coef` | 1.0 | B orthonormality weight; 100+ for manipulation |
| `pessimism` | 1.0 | Ensemble std coefficient in Q target |
| `train_goal_ratio` | 0.5 | Fraction of z from B(next\_obs) vs random |
| `actor_std` | 0.1 | TruncatedNormal std during training |
| `bc_coeff` | 0.0 | BC loss weight (FlowBC only) |

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
| `observation` | [T, obs\_dim] | Proprioceptive state (= qpos for ant tasks) |
| `action` | [T, act\_dim] | |
| `physics` | [T, phys\_dim] | Full physics state; used for reward relabelling |
| `discount` | [T] | 0 at terminal steps |

`load_transitions` converts T-step episodes into T-1 transition tuples and stores them in a `DictBuffer` under the layout:

```
batch["observation"]          [B, obs_dim]
batch["action"]               [B, act_dim]
batch["next"]["observation"]  [B, obs_dim]
batch["next"]["terminated"]   [B]
```

---

## Evaluation

`OGBenchEvaluator` iterates all 5 task variants per domain. For each:

1. Sample a batch from the offline buffer and relabel with `get_relabel_fn(domain, task)`
2. Solve `min_w ||B(obs)·w − r||²` (closed form) for **z**
3. Roll out `eval_n_episodes` episodes via `envs/rollout.py`
4. Log `eval/{task}/success` and `eval/{task}/reward`

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