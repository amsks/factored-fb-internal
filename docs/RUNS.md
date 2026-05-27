# RUNS — how to launch every agent × task (cube-single)

Single source for the run commands. Two stacks:

| Stack | Agents | Entry point | Configs |
|---|---|---|---|
| **JAX / OGBench** (vendored) | GCIQL, GCIVL, CRL, **CRL+FlowBC** | `run_gciql.py` → `third_party/ogbench/impls/main.py` | `configs/gciql/*.yaml` |
| **PyTorch FB** (td_jepa port) | **FB** (= FB+FlowBC on cube) | `train.py` | `configs/domain/*.yaml` + `configs/agent/*.yaml` |

## Conventions (read once)
- **Interpreters:** JAX stack runs `<repo>/.venv/bin/python run_gciql.py`, which spawns `/dev/shm/.venv-jax/bin/python main.py`. FB stack runs `<repo>/.venv/bin/python train.py` (PyTorch).
- **Storage is volatile.** Durable `/` is full, so everything writes to **`/dev/shm`** (RAM) — JAX via `configs/gciql/storage/shm.yaml` (`output_root=/dev/shm/gciql_outputs`); FB via the launcher pinning `hydra.run.dir`/`TMPDIR`/`WANDB_DIR` to `/dev/shm`. **Copy results out before any reboot.**
- **wandb → `amsks` / project `factored-fb`** for BOTH stacks (JAX via the shim's `GCIQL_WANDB_ENTITY=amsks`; FB via `configs/train.yaml wandb_entity: amsks`). Never `al-laq-fb`.
- **GPUs:** JAX preallocates a GPU-memory fraction → run JAX jobs on GPUs free of other JAX jobs. PyTorch FB uses little VRAM but contends on compute. Launchers map `seed i → GPU i % nGPU`. Run everything **in tmux**.

## Prerequisites / data
```bash
# JAX/OGBench data (native .npz, already present): ~/.ogbench/data/<env>.npz  (-> /dev/shm/ogbench_data)
# FB data (per-episode + physics) — extract once from the OGBench .npz:
/dev/shm/.venv-jax/bin/python scripts/extract_ogbench.py \
    --env cube-single-play-v0 --dataset_path ~/.ogbench/data \
    --output_folder /dev/shm/factored-fb/datasets
# Coverage-balanced density (only needed for FB reweight_alpha>0):
.venv/bin/python -m scripts.cube_density \
    --data /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer --out analysis/cube_density.npz
```

---

## Matrix (cube-single) — config / how to run

| Agent | State | Pixel (impala) | Pixel (DrQ) |
|---|---|---|---|
| **GCIQL** | `cube_single_state` | `cube_single_visual` | `cube_single_visual_drq` |
| **GCIVL** | — *(no config)* | `cube_single_visual_gcivl` | `cube_single_visual_gcivl_drq` |
| **CRL** (plain) | `cube_single_state_crl` | — *(no config)* | — *(no config)* |
| **CRL+FlowBC** | `cube_single_state_crl_flowbc` | — | `cube_single_visual_crl_flowbc_drq` |
| **FB** (= fb_flowbc) | `train.py domain=cube_single` | — | `train.py domain=visual_cube_single` |

Gaps (no config yet): GCIVL-state, plain-CRL-pixel, plain-FB on cube (`agent=fb` does **not** override the domain's `fb_flowbc`). OGBench cube recipes (alpha) baked into the configs: GCIQL=1.0, GCIVL=10.0, CRL=3.0.

---

## JAX / OGBench runs (GCIQL · GCIVL · CRL · CRL+FlowBC)

**Single seed (direct):**
```bash
.venv/bin/python run_gciql.py --config-name <CONFIG> seed=0
# e.g. GCIQL state:        --config-name cube_single_state
#      GCIQL pixel (DrQ):  --config-name cube_single_visual_drq
#      GCIVL pixel:        --config-name cube_single_visual_gcivl
#      CRL state:          --config-name cube_single_state_crl
#      CRL+FlowBC state:   --config-name cube_single_state_crl_flowbc
#      CRL+FlowBC pixel:   --config-name cube_single_visual_crl_flowbc_drq
# Overridable on the CLI: seed=, train_steps=, save_interval=, eval_interval=, run_group=, wandb_mode=
```

**Multi-seed in tmux (state, any JAX config via CONFIG=):**
```bash
tmux new -s crl_plain
CONFIG=cube_single_state_crl GPUS="0 1" SEEDS="0 1 2 3 4" bash scripts/run_state_crl.sh
#   scripts/run_state_crl.sh        — plain CRL (CONFIG-parameterized; default cube_single_state_crl)
#   scripts/run_state_crl_flowbc.sh — CRL+FlowBC state (10 seeds, seed i->GPU i%nGPU)
# Knobs: SEEDS, GPUS, TRAIN_STEPS, SAVE_INTERVAL, EVAL_INTERVAL, RUN_GROUP, DRY_RUN=1
```
(For GCIQL/GCIVL state/pixel multi-seed, run `run_state_crl.sh` with `CONFIG=<that config>` and an appropriate `RUN_GROUP`, or loop `run_gciql.py` over seeds.)

---

## PyTorch FB runs (FB / FB+FlowBC)

**Coverage-balanced reweight sweep (the current experiment), in tmux:**
```bash
tmux new -s fb_rw
# defaults to PRIORITIZED + weighted-z, alpha=0.5, ortho=1000, 10 seeds:
GPUS="2 3 4 5 6 7" bash scripts/run_fb_reweight.sh
# knobs: ALPHA, WEIGHT_DIAG(true=prioritized/false=pure rho'), WEIGHT_Z, CLIP, ORTHO_COEF, SEEDS, GPUS, DRY_RUN=1
```

**Plain FB baseline (single seed), state — outputs pinned to /dev/shm:**
```bash
RUN=/dev/shm/factored-fb/runs/$(date +%H%M%S)__fb__s0
CUDA_VISIBLE_DEVICES=2 TMPDIR=/dev/shm/tmp_fb WANDB_DIR=/dev/shm/wandb_fb \
.venv/bin/python train.py domain=cube_single seed=0 device=cuda \
    ortho_coef=1000 num_train_steps=1000000 \
    data_path=/dev/shm/factored-fb/datasets use_wandb=true \
    hydra.job.chdir=true "hydra.run.dir=$RUN"
# reweighted variant: add reweight_alpha=0.5 reweight_density_path=$PWD/analysis/cube_density.npz weight_diag=true weight_z=true
```
**Pixel FB:** `domain=visual_cube_single` (obs_type=pixels, frame_stack=3; OGBench visual default is `num_train_steps=500000`).

**One-step FB (+ FlowBC actor)** — `onestep=true` makes F z-independent (behavior successor features) and bootstraps with the dataset's next action (SARSA); the FlowBC actor does one improvement step. Default-off (byte-identical FB). Works with the existing extracted dataset (`next.action` is derived at load time):
```bash
# state, one-step FB + FlowBC:
.venv/bin/python train.py domain=cube_single onestep=true seed=0 device=cuda \
    data_path=/dev/shm/factored-fb/datasets use_wandb=true \
    hydra.job.chdir=true "hydra.run.dir=/dev/shm/factored-fb/runs/$(date +%H%M%S)__onestepfb__s0"
# pixel: domain=visual_cube_single onestep=true
```

---

## Reference: launchers in `scripts/`
| script | what |
|---|---|
| `run_state_crl.sh` | plain CRL (or any JAX state config via `CONFIG=`), multi-seed, tmux |
| `run_state_crl_flowbc.sh` | CRL+FlowBC state, 10 seeds, tmux (post critic-shield fix) |
| `run_fb_reweight.sh` | coverage-balanced FB sweep (`WEIGHT_DIAG`/`WEIGHT_Z`/`ALPHA`), tmux |
| `run_crl_flowbc_campaign.sh` | original 2-track (state + pixel) CRL+FlowBC campaign launcher |
| `fb_after_crl.sh` | watcher: auto-launch the FB sweep once GPUs free |
| `extract_ogbench.py` | build FB per-episode dataset from the OGBench `.npz` |
| `cube_density.py` | B0 goal-agnostic cube-position density for the FB reweight |

## Monitoring
```bash
tmux ls                                   # sessions: crl_plain / crl_state / fb_rw / ...
tail -f /dev/shm/<log_dir>/*.log          # per-seed logs (each launcher prints its log dir)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# wandb: https://wandb.ai/amsks/factored-fb  (filter by run_group)
```
