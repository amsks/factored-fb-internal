# GCIQL via vendored OGBench (JAX) alongside PyTorch FB

**Date:** 2026-05-18
**Status:** Approved (awaiting user spec review)
**Branch:** `gciql-pixel-integration`

## Motivation

We want to compare our PyTorch Forward-Backward (FB / FB-FlowBC) implementation
on cube manipulation against a strong OGBench baseline, **GCIQL**
(goal-conditioned IQL), on `cube-single-play-v0` in **both state-based and
pixel-based** variants.

OGBench's GCIQL is implemented in **JAX/Flax**, and OGBench's training pipeline
is structurally different from Factored-FB's FB pipeline:

| | Factored-FB (FB) | OGBench (GCIQL) |
|---|---|---|
| Data | flat transitions (`DictBuffer`) | trajectory-aware `GCDataset`, future-goal + geometric goal sampling |
| Conditioning | inferred `z` from reward relabel | explicit goal `g` (state/image) |
| Eval | relabel → `infer_z` → rollout(z) | env-provided `info['goal']`, success from `info['success']` |
| Pixel encoder | DrQ (td_jepa) | IMPALA (`impala_small`) |
| Framework | PyTorch + Hydra | JAX/Flax + ml_collections/absl |

A PyTorch reimplementation of GCIQL would risk subtle divergence from OGBench's
reported numbers and invalidate the comparison. The decision is therefore to
**run GCIQL as verbatim OGBench (JAX)**, vendored into this repo, driven by a
**thin Hydra launcher**, with full dependency and process isolation from the
PyTorch FB track.

## Goals

- Vendor the entire OGBench repository into Factored-FB, byte-for-byte
  upstream, never edited.
- Run GCIQL on `cube-single-play-v0` (state) and `visual-cube-single-play-v0`
  (pixel) using **OGBench's exact hyperparameters**.
- Drive runs from a Hydra launcher consistent with the rest of the repo, while
  the JAX training pipeline executes exactly as upstream OGBench (config via
  OGBench's own CLI flags; pipeline untouched).
- Full isolation: a separate JAX virtualenv, separate output tree, and a
  `subprocess` boundary so the JAX and PyTorch tracks never share a process,
  imports, or config system.
- The existing PyTorch FB track (`train.py`, Hydra, `.venv`) is unchanged.

## Non-goals

- Porting GCIQL (or any OGBench agent) to PyTorch.
- The pixel-FB port from td_jepa — deferred to a separate spec.
- A shared/unified evaluation harness across FB and GCIQL (their eval protocols
  differ; see "Comparison" below).
- Modifying OGBench's training loop, agent code, `ml_collections` configs, or
  absl flags.
- Changing the FB track in any way.

## Architecture: two non-interacting tracks

```
Factored-FB/
├── train.py                  # PyTorch FB track — UNCHANGED, uses .venv
├── run_gciql.py              # NEW: thin Hydra launcher — imports NO jax
├── configs/
│   ├── train.yaml            # unchanged
│   └── gciql/                # NEW: Hydra configs encoding OGBench's exact HPs
│       ├── cube_single_state.yaml
│       └── cube_single_visual.yaml
├── third_party/
│   └── ogbench/              # NEW: full verbatim copy of OGBench (no .git)
│       ├── impls/            #   training pipeline (main.py, agents/gciql.py …)
│       ├── ogbench/          #   env + dataset library source
│       ├── assets/           #   (~4 MB) committed
│       └── PROVENANCE.md     #   upstream URL + commit + copy date
├── requirements-jax.txt      # NEW: isolated JAX deps (from ogbench pyproject)
├── Makefile                  # NEW target: install-jax
└── scripts/
    ├── run_gciql.sh          # NEW: env-var parameterised launcher
    └── smoke_test_gciql.sh   # NEW: end-to-end smoke (state + visual)

/mnt/scratch1/   (spare 875 GB NVMe — mounted by the user via real SSH; see Prereqs)
├── .venv-jax/                # isolated JAX+CUDA env (never mixes with .venv)
├── ogbench_data/             # OGBENCH_DATASET_DIR (auto-downloaded datasets)
└── gciql_outputs/<run>/      # checkpoints + eval CSV + logs
```

**Shared surface:** the git repo, the NVMe disk, and (optionally) a wandb
project for human-side comparison. **Not shared:** Python process, imports,
virtualenv, config system. The track boundary is a single `subprocess.run`.

## Components

### 1. Vendored OGBench (`third_party/ogbench/`)

- Source: `/home/adityamohan/git/baselines/ogbench`
  (`https://github.com/seohongpark/ogbench.git`).
- Upstream commit: `1d4140997f60c52c6fb0702ec100dc988b18c548` (2026-01-14).
- Copied **excluding `.git/`** (~41 MB working tree; trivially fits the full
  root disk, which has ~4.2 GB free).
- **Never edited.** Any behavioural change must be expressed as an OGBench CLI
  flag from the launcher, never as a source edit.
- `third_party/ogbench/PROVENANCE.md` records the upstream remote, the exact
  commit SHA, the copy date, and the rule "do not modify; upgrade by
  re-vendoring at a new pinned commit."

### 2. Isolated JAX environment (`requirements-jax.txt`, `.venv-jax`)

- `requirements-jax.txt` derived from `third_party/ogbench/pyproject.toml`
  training extras: `jax[cuda12]` (>=0.4.26), `flax` (>=0.8.4), `distrax`
  (>=0.1.5), `ml_collections`, `optax`, `wandb`, plus `tqdm`, `matplotlib`,
  `moviepy` as required by `impls/main.py`.
- The `ogbench` env/dataset library is installed **from the vendored source**
  (`pip install -e third_party/ogbench`), not from PyPI — keeping the whole
  thing self-contained and pinned to the vendored commit (no external version
  drift). This is the consequence of vendoring the full repo (Approach B).
- Installed into `/mnt/scratch1/.venv-jax` (on the NVMe, not the full root
  disk) via a `make install-jax` target (uv-based, mirroring `make install`).
- The PyTorch `.venv` is never touched; the two environments never share an
  interpreter.

### 3. Thin Hydra launcher (`run_gciql.py`)

- `@hydra.main(config_path="configs", config_name=...)` over the
  `configs/gciql/` group.
- Responsibilities, in order:
  1. Resolve the Hydra cfg into OGBench's **exact** CLI flag list
     (`--env_name=…`, `--agent=agents/gciql.py`, `--agent.alpha=…`, etc.).
  2. Create the per-run output dir under
     `/mnt/scratch1/gciql_outputs/<timestamp>__<env>__gciql__s<seed>/`
     (mirrors the existing per-run artifact convention).
  3. Set environment for the child: `OGBENCH_DATASET_DIR=/mnt/scratch1/ogbench_data`,
     `CUDA_VISIBLE_DEVICES`, `MUJOCO_GL=egl`, `WANDB_*` if enabled.
  4. `subprocess.run(["/mnt/scratch1/.venv-jax/bin/python", "main.py", *flags],
     cwd="third_party/ogbench/impls", env=…)`, streaming child stdout/stderr to
     the run dir's log.
  5. Propagate the child exit code; record run metadata (resolved flags,
     git SHA of Factored-FB, vendored OGBench SHA) as a JSON sidecar.
- **`run_gciql.py` never imports `jax`/`flax`.** It is pure orchestration so
  that Hydra/PyTorch deps and JAX deps never coexist in one interpreter.

### 4. Configs — OGBench's EXACT cube hyperparameters

Sourced from `third_party/ogbench/impls/hyperparameters.sh` and
`impls/agents/gciql.py:get_config()`.

`configs/gciql/cube_single_state.yaml`:

| Flag | Value | Origin |
|---|---|---|
| `env_name` | `cube-single-play-v0` | hyperparameters.sh:603 |
| `agent` | `agents/gciql.py` | hyperparameters.sh:603 |
| `agent.alpha` | `1.0` | hyperparameters.sh:603 (cube override) |
| `eval_episodes` | `50` | hyperparameters.sh:603 |
| `train_steps` | `1_000_000` | main.py default |
| `agent.expectile` | `0.9` | gciql.py default |
| `agent.discount` | `0.99` | gciql.py default |
| `agent.tau` | `0.005` | gciql.py default |
| `agent.lr` | `3e-4` | gciql.py default |
| `agent.batch_size` | `1024` | gciql.py default |
| `agent.actor_hidden_dims` | `(512,512,512)` | gciql.py default |
| `agent.value_hidden_dims` | `(512,512,512)` | gciql.py default |
| `agent.layer_norm` | `true` | gciql.py default |
| `agent.const_std` | `true` | gciql.py default |
| `agent.actor_loss` | `ddpgbc` | gciql.py default |
| `agent.encoder` | `None` | gciql.py default (state) |

`configs/gciql/cube_single_visual.yaml` (deltas vs state, from
hyperparameters.sh:837–838):

| Flag | Value |
|---|---|
| `env_name` | `visual-cube-single-play-v0` |
| `agent.alpha` | `1.0` |
| `agent.batch_size` | `256` |
| `agent.encoder` | `impala_small` |
| `agent.p_aug` | `0.5` |
| `train_steps` | `500_000` |
| `eval_on_cpu` | `0` |
| `eval_episodes` | `50` |

Goal-sampling and reward defaults (unchanged from `gciql.py`): value goals
`(p_curgoal, p_trajgoal, p_randomgoal) = (0.2, 0.5, 0.3)` with
`value_geom_sample=true`; actor goals `(0.0, 1.0, 0.0)` with
`actor_geom_sample=false`; `gc_negative=true`. These are **not** re-specified
in our YAML (we do not override them) — documented here only for provenance.

`cube_double_*` variants may be added later by mirroring
hyperparameters.sh:616–617 / 850–851; not required for the initial comparison.

### 5. Launch + smoke scripts

- `scripts/run_gciql.sh`: env-var parameterised (`SEEDS`, `VARIANT=state|visual`,
  `WANDB_*`, …), mirrors `scripts/run_sweep.sh` ergonomics, invokes
  `run_gciql.py`.
- `scripts/smoke_test_gciql.sh`: mirrors `scripts/smoke_test_artifacts.sh`.
  Runs state and visual with tiny `train_steps` and small `eval_interval`.

## Data flow

```
Hydra cfg (configs/gciql/*.yaml)
  └─ run_gciql.py resolves → OGBench CLI flag list
       └─ subprocess: .venv-jax/bin/python third_party/ogbench/impls/main.py …
            ├─ ogbench.make_env_and_datasets() downloads/loads cube dataset
            │     → /mnt/scratch1/ogbench_data
            ├─ GCIQLAgent trains (JAX, exactly upstream)
            └─ checkpoints + eval CSV (+ optional wandb)
                  → /mnt/scratch1/gciql_outputs/<run>/
  └─ launcher captures exit code + log path + metadata sidecar
```

Nothing crosses back into the PyTorch process.

## Comparison (human-side, not code-coupled)

Pass a shared `--wandb` project/group flag so GCIQL runs sit next to FB runs in
wandb for side-by-side viewing. **Known semantic gap (documented, not hidden):**
FB's evaluator (relabel → `infer_z` → rollout) and OGBench's evaluator
(env-provided goals, `info['success']`) are different protocols. The legitimate
comparison is *each method's own success metric on the same cube task*
(FB `eval/success` vs OGBench `evaluation/overall_success`), not a shared eval
harness. We will state this caveat wherever results are reported.

## Prerequisites (hard gate before `make install-jax`)

The instance root disk is 100% full (~4.2 GB free). The JAX env, datasets, and
outputs must live on a spare instance-store NVMe. `sudo` is not available
non-interactively from the agent harness, so the user runs this once in a
**real interactive SSH session**:

```bash
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/scratch1
sudo mount /dev/nvme1n1 /mnt/scratch1
sudo chown "$USER":"$USER" /mnt/scratch1
df -hT /mnt/scratch1
```

Caveat: instance-store is **wiped on instance stop/terminate** (survives
reboot). Checkpoints worth keeping must be pushed off-box (rsync/S3).

## Risks & open questions

- **JAX CUDA match:** the `jax[cuda12]` wheel must match the instance driver.
  The smoke test asserts `jax.devices()` reports the GPU; if not, pin a
  compatible jaxlib build.
- **Network egress:** `make_env_and_datasets` downloads datasets on first use;
  the instance must allow outbound HTTPS. Smoke test surfaces this early.
- **Visual dataset size:** multi-GB; NVMe (875 GB) is sized fine; root disk
  untouched.
- **Headless render:** OGBench renders video episodes during eval;
  `MUJOCO_GL=egl` is set for the child (matches FB track's existing default).
- **NVMe ephemerality:** documented; not solved here (out of scope).

## Testing strategy

**Smoke (fast):** `scripts/smoke_test_gciql.sh` runs both variants with
`train_steps≈2000`, small `eval_interval`, and asserts:
1. `jax.devices()` sees the GPU (printed by a one-line preflight in the script).
2. Dataset loads from `/mnt/scratch1/ogbench_data` (no write to `/`).
3. A checkpoint file appears under the run dir.
4. An `evaluation/overall_success` scalar is logged to the run CSV.

**Validation (slow):** one real `cube-single-play-v0` state run; sanity-check
final `overall_success` against OGBench's reported GCIQL cube-single number
(order-of-magnitude agreement, not exact, due to seed/hardware).

**Isolation check:** confirm `run_gciql.py` does not import jax (static check),
and that the PyTorch `.venv` is unmodified after `make install-jax`.

## Files touched

| File | Change |
|---|---|
| `third_party/ogbench/**` | NEW — verbatim vendored copy (no `.git`) |
| `third_party/ogbench/PROVENANCE.md` | NEW — upstream URL + commit + rule |
| `run_gciql.py` | NEW — thin Hydra launcher (no jax import) |
| `configs/gciql/cube_single_state.yaml` | NEW |
| `configs/gciql/cube_single_visual.yaml` | NEW |
| `requirements-jax.txt` | NEW — isolated JAX deps |
| `Makefile` | NEW target `install-jax` (+ `run-gciql`, `smoke-gciql`) |
| `scripts/run_gciql.sh` | NEW |
| `scripts/smoke_test_gciql.sh` | NEW |
| `README.md` | Add a short "GCIQL (OGBench, JAX) baseline" section |
| `.gitignore` | Ensure `/mnt/scratch1` artifacts / local outputs ignored |

Files **not** touched: `train.py`, `agents/`, `configs/train.yaml`,
`configs/agent/*`, `configs/domain/*`, `evals/`, `envs/`, `buffers/`,
`data/`, `nn_models.py`, `.venv` — the entire PyTorch FB track.
