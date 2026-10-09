# Design: per-agent run entry points for cube-single

**Date:** 2026-05-28
**Scope:** `scripts/` and `README.md`
**Audience:** anyone (user, collaborator, paper reader) who wants to launch an agent×observation combination on the cube-single env.

## Problem

`scripts/` currently has 19 shell scripts (~1,300 lines), with substantial duplication: 4 GCIQL scripts, 3 CRL variants, separate scripts for one-step FB and reweighted FB, etc. The current README's Quick Start only mentions `make train-cube` and never names GCIQL/GCIVL/CRL/CRL+FlowBC. `docs/RUNS.md` documents recipes but is itself a manual rather than an entry point. A new arrival has no obvious starting command for "run agent X on cube state/pixel."

## Design

### Directory layout

```
scripts/
├── agents/
│   ├── _common.sh                # shared boilerplate
│   ├── fb/
│   │   ├── state.sh              # FB+FlowBC on cube_single
│   │   └── pixel.sh              # FB+FlowBC on visual_cube_single
│   ├── gciql/
│   │   ├── state.sh
│   │   └── pixel.sh
│   ├── gcivl/
│   │   └── pixel.sh              # no state config — file omitted
│   ├── crl/
│   │   └── state.sh              # no pixel config — file omitted
│   └── crl_flowbc/
│       ├── state.sh
│       └── pixel.sh              # DrQ-only
├── _drq_worker.sh                # internal helper, unchanged
├── resource_guard.sh             # unchanged
├── fb_after_crl.sh               # campaign orchestrator, unchanged
├── run_crl_flowbc_campaign.sh    # unchanged
├── resume_crl_flowbc_campaign.sh # unchanged
├── monitor_crl_campaign.sh       # unchanged
├── run_sweep.sh                  # KEPT (antmaze FB launcher)
├── run_sweep_slurm.sh            # KEPT (antmaze FB SLURM)
└── smoke_test_*.sh               # unchanged
```

8 leaf scripts under `agents/` (fb×2 + gciql×2 + gcivl×1 + crl×1 + crl_flowbc×2). Matrix gaps (GCIVL-state, plain-CRL-pixel) get no file rather than a stub.

### Per-leaf contract

Every `scripts/agents/<agent>/<obs>.sh`:

- Sources `scripts/agents/_common.sh` for boilerplate (tmux launch, GPU pinning, `/dev/shm` setup, wandb defaults, log dir, dry-run preview).
- **Default behavior**: multi-seed sweep in a detached tmux session, prints attach command.
- **Universal env knobs**: `SEEDS` (default `"0 1 2 3 4"`), `GPUS` (default `"0 1"`), `TRAIN_STEPS`, `RUN_GROUP`, `DATA_PATH`, `DRY_RUN=1`.
- **Per-agent knobs**:
  - `fb/`: `MODE` ∈ {`vanilla`, `onestep`} (default `vanilla`); `REWEIGHT_ALPHA` (default `0` = disabled).
  - `gciql/pixel.sh`, `gcivl/pixel.sh`: `ENCODER` ∈ {`impala`, `drq`} (default `impala`).
- **Help**: `--help` prints the script's contract (defaults, knobs, examples).
- **Single seed** = `SEEDS=0 GPUS=0 bash scripts/agents/<agent>/<obs>.sh`.
- **GPU mapping**: seed *i* → GPU index `i % nGPU` from the `GPUS` list (existing convention).

### Shared helper (`scripts/agents/_common.sh`)

Sourced by every leaf. Exposes:

| Function | Purpose |
|---|---|
| `_setup_dev_shm` | Create `/dev/shm/{factored-fb/runs,tmp_fb,wandb_fb,gciql_outputs}` if missing. |
| `_assert_data <path>` | Fail fast with a clear message if the extracted dataset is missing. |
| `_gpu_for_seed <seed> <gpus_str>` | Round-robin seed→GPU map. |
| `_log_dir <run_group> <seed>` | Canonical per-seed log dir under `/dev/shm/<run_group>/`. |
| `_tmux_kick <session> <cmd>` | Launch detached; print `tmux attach -t <session>` to stdout. |
| `_dry_run_preview` | If `DRY_RUN=1`, echo the full command set instead of executing. |
| `_print_usage` | Helper consumed by each leaf's `--help`. |

No leaf script duplicates GPU/tmux/dir logic.

### README rewrite

`docs/RUNS.md` is folded into README and deleted. README gains:

- A **"Run any agent on cube-single"** section near the top with a 2-column matrix table:
  - Rows: FB, FB (one-step), FB (reweight), GCIQL, GCIVL, CRL, CRL+FlowBC.
  - Columns: state, pixel.
  - Each cell: the exact one-line invocation (e.g., `bash scripts/agents/fb/state.sh`), or `—` for matrix gaps.
- A **knobs reference** subsection listing the universal env vars + per-agent knobs.
- A **prerequisites** subsection (kept from RUNS.md): data extraction, density file for reweight, JAX venv setup.
- A **monitoring** subsection (kept from RUNS.md): tmux ls, log paths, nvidia-smi, wandb URL.
- A **two-stack context** sidebar (kept from RUNS.md): why some agents run under `.venv` vs the JAX subprocess.

Existing README sections (Method, Installation, Configuration, Directory structure, Data layout, Evaluation, Citation, License) stay; only Quick Start is replaced and RUNS.md content is inlined.

### Deletions (9 scripts, ~520 lines)

- `run_gciql.sh`, `run_gciql_state.sh`, `run_gciql_state_sweep.sh`, `run_gciql_visual.sh`
- `run_state_crl.sh`, `run_state_crl_flowbc.sh`
- `run_fb_reweight.sh`, `run_onestep_fb.sh`
- `run_drq_pixel_sweep.sh`
- `docs/RUNS.md`

### Explicitly NOT touched

- `run_sweep.sh`, `run_sweep_slurm.sh` — kept for antmaze FB runs. README notes that antmaze still uses `run_sweep.sh`.
- Campaign orchestrators (`run_crl_flowbc_campaign.sh`, `resume_*`, `monitor_*`, `fb_after_crl.sh`) — different purpose (coordinated multi-agent experiments), not single-agent entry points.
- `_drq_worker.sh`, `resource_guard.sh` — internal helpers.
- `smoke_test_*.sh` — install verification, not run launchers.
- Configs (`configs/gciql/*.yaml`, `configs/domain/*.yaml`, etc.) — unchanged; the new leaves just reference them.

## Acceptance criteria

1. A new user can run `bash scripts/agents/fb/state.sh --help` and see exactly what knobs exist.
2. The 5 representative cube launches all succeed with their default flags (smoke-only — no need to train to completion):
   - `bash scripts/agents/fb/state.sh DRY_RUN=1`
   - `bash scripts/agents/gciql/pixel.sh ENCODER=drq DRY_RUN=1`
   - `bash scripts/agents/crl/state.sh DRY_RUN=1`
   - `bash scripts/agents/crl_flowbc/pixel.sh DRY_RUN=1`
   - `bash scripts/agents/fb/state.sh MODE=onestep REWEIGHT_ALPHA=0.5 DRY_RUN=1`
3. `README.md` has a matrix table at the top of "Quick Start" with one line per (agent, obs) cell.
4. `docs/RUNS.md` no longer exists; nothing in the repo references it.
5. The 9 deleted scripts are absent and not referenced by any kept script.
6. `scripts/run_sweep.sh` still works for antmaze (untouched).

## Out of scope

- Antmaze launchers (kept as-is).
- Refactoring `train.py`, `run_gciql.py`, configs, or any Python code.
- New agent variants or hyperparameter additions.
- SLURM / cloud launchers for the new tree (env-var contract is portable; SLURM wrappers can be added later if needed).

## Open risks

- **wandb groups**: existing scripts each set their own `RUN_GROUP`. The new leaves use the convention `<agent>_<obs>` as default; user can override. Risk: cross-script comparisons in wandb that grouped on old labels will need filter updates.
- **Campaign scripts reference deleted scripts?** Need to verify in implementation; if so, redirect inside the campaign script to the new leaf path.
