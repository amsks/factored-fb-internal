# CRL on cube-single (state + pixel/DrQ) — Design

**Status:** approved (pending user review)
**Date:** 2026-05-22
**Author:** brainstormed with Claude

## Goal

Add OGBench's **CRL** (Contrastive RL) goal-conditioned baseline to the cube-single
experiment matrix, on two observation modalities:

1. **State** — `cube-single-play-v0`, MLP encoder (no visual front-end).
2. **Pixel + DrQ** — `visual-cube-single-play-v0` with FB's DrQ visual front-end
   (encoder + random_shifts augmentation + frame_stack=3 + RGBNorm), identical to
   the GCIQL/GCIVL DrQ configs, so the only baseline-vs-FB difference is the RL
   algorithm.

This mirrors the existing GCIQL/GCIVL work. **CRL needs no agent/Python code
changes** — it already exposes `encoder`/`frame_stack`/`p_aug` placeholders and
reads the global `encoder_modules`, so the DrQ injection (`sitecustomize.py`) and
`run_gciql.py`'s `OGBENCH_DRQ_FRONTEND` gating apply unchanged.

## Critical context (verified against the codebase)

- **CRL agent** (`third_party/ogbench/impls/agents/crl.py`): `get_config()` declares
  `encoder` (placeholder str), `frame_stack` (placeholder int), `p_aug` (0.0),
  `dataset_class='GCDataset'`, `actor_loss='ddpgbc'` (default), `alpha=0.1` (default).
  It instantiates `encoder_modules[config['encoder']]` **five** times:
  `critic_state`, `critic_goal`, `value_state`, `value_goal`, and a
  `GCEncoder(concat_encoder=...)` for the actor. The actor's GCEncoder encodes
  `concat([obs, goal])` → ~18 channels at frame_stack=3 (vs 9 for obs alone); Flax
  infers input channels, so DrQEncoder handles it. This is more encoder copies than
  GCIQL/GCIVL but the same registration path — the smoke run validates it.
- **OGBench's recommended CRL recipe for cube-single** (`impls/hyperparameters.sh`):
  - state `cube-single-play-v0`: `--agent.alpha=3.0`
  - pixel `visual-cube-single-play-v0`: `--agent.alpha=3.0 --agent.batch_size=256
    --agent.encoder=impala_small --agent.p_aug=0.5`
  - For the **DrQ** variant we keep `alpha=3.0`, `batch_size=256` and swap
    `encoder=drq, frame_stack=3, p_aug=1.0` — exactly the deltas the GCIQL/GCIVL
    DrQ configs applied over their `impala_small` recipes.
- **`run_gciql.py` is agent-agnostic.** `build_ogbench_argv` emits `--agent=<agent_file>`
  and `--agent.<k>=<v>` for every entry in `agent_overrides`; `build_child_env` sets
  `OGBENCH_DRQ_FRONTEND=1` whenever `agent_overrides.encoder == 'drq'`. No code change.
- **Hard constraint:** never edit `third_party/ogbench` (pinned). Unchanged here.
- **Existing config conventions** (the templates to mirror):
  - `configs/gciql/cube_single_state.yaml` — state: 1M steps, `eval_on_cpu=1`,
    `save_interval=1000000`, `eval_interval=100000`, `eval_episodes=50`,
    `log_interval=5000`.
  - `configs/gciql/cube_single_visual_drq.yaml` — pixel/DrQ: 500k steps,
    `eval_on_cpu=0`, `save_interval=50000`, encoder/frame_stack/p_aug as above.
- **Launcher** (`scripts/run_drq_pixel_sweep.sh` + `scripts/_drq_worker.sh`): fixed
  GPU set, **static** round-robin of jobs across GPUs, one detached worker per GPU
  running its slice **sequentially**. No dynamic GPU pickup. Two limitations for
  this work: (a) it builds jobs as CONFIGS×SEEDS with a single SEEDS list — it can't
  express 10 state seeds + 5 pixel seeds in one call; (b) the worker hardcodes log
  names as `{gciql,gcivl}_pixel_drq_s<seed>_<ts>.log` — wrong label for `crl` and
  for `state`.
- **Storage:** all outputs land on `/dev/shm` (tmpfs, **reboot-wiped**), same as the
  DrQ runs. Run groups: `factored-fb-crl` (state), `factored-fb-crl-pixel-drq` (pixel).

## Scope (decided)

- **State:** 10 seeds (0–9). **Pixel/DrQ:** 5 seeds (0–4). 15 runs total.
- **Smoke-test first:** 2k-step `wandb=disabled` run of each config before the sweep.
- **GPUs:** start on the free, healthy GPUs **6 and 7** now; launch a **second batch
  on 3 and 5** for the remaining not-yet-started seeds once the in-flight GCIVL DrQ
  runs free them (~4–5h). Avoid degraded 0/1/2/4.

## Components

### 1. Two Hydra configs (`configs/gciql/`)

**`cube_single_state_crl.yaml`** — clone of `cube_single_state.yaml` with only:
`agent_file: agents/crl.py`, `agent_overrides: {alpha: 3.0}`, `run_group:
factored-fb-crl`. Everything else (env, 1M steps, eval_on_cpu=1, intervals)
identical, so the only difference vs the GCIQL state baseline is the algorithm.

**`cube_single_visual_crl_drq.yaml`** — clone of `cube_single_visual_drq.yaml` with
only: `agent_file: agents/crl.py`, `run_group: factored-fb-crl-pixel-drq`, and
`agent_overrides: {alpha: 3.0, batch_size: 256, encoder: drq, frame_stack: 3,
p_aug: 1.0}` (alpha 1.0→3.0 is the CRL recipe; the DrQ deltas are unchanged).

### 2. Launcher generalization (`scripts/`)

Generalize the existing scripts (keep one launcher, don't fork) so they handle
heterogeneous configs and correct log names:

- **`_drq_worker.sh`:** name logs from the **config name** —
  `logs/<cfg>_s<seed>_<ts>.log` (e.g. `cube_single_state_crl_s0_<ts>.log`,
  `cube_single_visual_crl_drq_s3_<ts>.log`). Removes the wrong `pixel_drq` label;
  unambiguous across algo/variant. (Affects only future runs; completed DrQ runs
  keep their existing log files.)
- **`run_drq_pixel_sweep.sh`:** accept an explicit **`JOBS`** override (space-separated
  `config:seed`) that bypasses the CONFIGS×SEEDS construction. This expresses the
  heterogeneous 10+5 matrix in one call and lets the second batch (on 3,5) take just
  the remaining jobs. When `JOBS` is unset, behavior is unchanged (back-compat).

Each GPU's worker runs one job at a time, so pixel runs (~17 GB) never share a GPU —
no OOM risk from concurrency. **Job ordering:** front-load the 5 pixel jobs ahead of
the state jobs in the `JOBS` list so the headline DrQ comparison finishes first.

### 3. Config argv tests (`tests/test_drq_configs.py`)

Add two test functions mirroring the existing DrQ config tests: compose each new
config and assert the resulting `build_ogbench_argv` contains the expected
`--agent=agents/crl.py`, `--agent.alpha=3.0`, run_group, env_name, and (pixel only)
`--agent.encoder=drq --agent.frame_stack=3 --agent.p_aug=1.0`. TDD: write first,
watch them fail (MissingConfigException), then add the configs.

## Data flow

`run_gciql.py --config-name <cfg>` → Hydra resolves config → `build_ogbench_argv`
emits absl flags → `build_child_env` sets `OGBENCH_DRQ_FRONTEND=1` for the pixel/DrQ
config (encoder=drq) and leaves it unset for state → subprocess runs
`.venv-jax/bin/python main.py` with the shim on PYTHONPATH → `sitecustomize.py`
registers `encoder_modules['drq']` and (pixel only) swaps `GCDataset.augment` →
checkpoints/CSVs written to `/dev/shm/gciql_outputs/factored-fb/<run_group>/sd<seed>_<ts>/`.

## Testing & verification

1. **Unit:** `tests/test_drq_configs.py` (the two new argv tests) pass under
   `.venv/bin/python -m pytest`.
2. **Smoke (manual gate, GPU + .venv-jax):** 2k-step `wandb=disabled` run of each
   config; confirm CRL builds (state) and `encoder=drq` resolves for CRL's 5-encoder
   + 18-ch actor concat (pixel) with no shape error, loss decreasing, a checkpoint
   written, healthy throughput (not the ~20 s/it degraded-GPU pathology).
3. **Sweep launch:** batch 1 on GPUs 6,7 (all 15 jobs round-robin, pixel first);
   batch 2 on 3,5 for remaining seeds once GCIVL frees them.
4. **Regression:** existing `tests/test_drq_configs.py` cases and a representative DrQ
   smoke still behave (launcher back-compat when `JOBS` unset).

## Deviations / risks

- **CRL instantiates 5 encoders** vs GCIQL/GCIVL's fewer — higher GPU memory for the
  pixel runs. The smoke run verifies it fits on one 23 GB GPU before the sweep.
- **`/dev/shm` is tmpfs** — a reboot wipes checkpoints/CSVs. Out of scope here
  (same as DrQ), but worth copying completed runs to durable storage afterward.
- **State at 1M steps × 10 seeds** is the heaviest piece; sequential per-GPU
  execution is acceptable (user confirmed). State runs are MLP-only and cheap per
  step relative to pixel.

## Out of scope

- Persisting `/dev/shm` outputs to durable storage (separate decision).
- CRL on any env other than cube-single; other DrQ variants; analysis/plots.
- Editing vendored OGBench.
