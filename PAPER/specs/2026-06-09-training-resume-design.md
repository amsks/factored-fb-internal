# Design: Crash-Recovery Resume for Training Runs

**Date:** 2026-06-09
**Status:** design — awaiting review
**Scope:** PyTorch `train.py` (FB / PSM / RLDP, state + pixel) + verify/document JAX
`run_gciql.py` resume + `scripts/run_grid_sweep.sh` launcher flag.

## Problem

Long runs (1M × 10 seeds for `psm_state_tuned` / `rldp_state_tuned`) can die
mid-flight (preemption, OOM, host reboot). Today a dead PyTorch run must restart
from step 0 — wasted GPU-hours. We want to relaunch and continue from the last
checkpoint.

Two concrete gaps surfaced while planning the 1M launch:

1. **No resume path in `train.py`.** The loop only ever *saves*
   (`train.py:408-410`, `:440`); there is no load/continue path. (The agent has
   `.load()` at `agents/base.py:62` but the trainer never calls it.)
2. **Output dirs are timestamp-based** (`configs/train.yaml:43`,
   `hydra.run.dir = outputs/${now}__${domain}__${agent}__s${seed}`), so a
   relaunch writes to a *new* dir and could never find the prior checkpoint. This
   same non-determinism caused concurrent same-second sweep runs to collide into
   one dir and overwrite each other's checkpoints (observed: 17 dirs for 18 runs;
   a dir whose `.hydra/overrides.yaml` did not match its `final.pt`).

What is **already** in place (do not rebuild):

- PyTorch checkpoints already include **optimizer state** —
  `agents/fb/agent.py:561-573` `state_dict()` saves `forward/backward/actor_optimizer`
  (this is why `.pt` files are ~174 MB). Target nets are `nn.Module` submodules,
  also captured.
- The replay buffer is the **static offline dataset** reloaded from `data_path`
  (`train.py:353`); there is no online buffer to persist.
- **JAX/GCIQL resume already works**: `run_gciql.py:48-52` passes
  `restore_path`/`restore_epoch` to the vendored OGBench `main.py:86-87`
  (`restore_agent` restores params + opt state from `params_<epoch>.pkl`), and the
  wandb shim (`tools/wandb_mode_shim/sitecustomize.py`) handles `GCIQL_STEP_OFFSET`
  (shift saved epochs + wandb/CSV steps), `GCIQL_CSV_APPEND`, and `GCIQL_EXP_NAME`.

## Faithfulness bar (decided)

**Crash-recovery, not bit-exact.** Restore weights + optimizer + step; re-seed RNG
fresh. The continued run is statistically equivalent, not bit-identical. Rationale:
CUDA + TF32 (`train.py:31`) already make mid-stream bit-exactness impossible, and
the project's existing GCIQL resume re-keys PRNG rather than restoring it — we match
that convention. Bit-exact (persisting torch/cuda/numpy/python RNG + buffer sampling
order) is explicitly **out of scope**.

## Design

### 1. Checkpoint sidecar (PyTorch)

- `step_N.pt` stays **exactly** as today: raw `agent.state_dict()`. Preserves the
  existing contract for every analysis/eval script that does `agent.load(path)`.
- At each checkpoint, additionally write one `train_state.json` in `save_dir`
  (overwritten each time), holding:
  ```json
  {"step": N, "wandb_run_id": "<id|null>", "latest": "step_N.pt"}
  ```
  A single overwritten file always points at the newest checkpoint — no per-step
  proliferation, trivial to read on resume.

### 2. Deterministic save_dir (decided: include)

- Change `configs/train.yaml` `hydra.run.dir` to run-name-based:
  `outputs/${run_dir_name}`, where `run_dir_name` defaults to `${wandb_run_name}`
  and falls back to `${domain}__${agent}__s${seed}` when `wandb_run_name` is unset.
- Stable across relaunches (enables resume) **and** collision-free per run
  (fixes the overwrite/provenance bug).
- **Clobber guard:** if `resume` is false and `save_dir` already contains
  `step_*.pt` or `final.pt`, abort with a clear error (`pass force=true to
  overwrite`). Prevents silently destroying a prior run's checkpoints.

### 3. Resume logic (`train.py`)

New Hydra config keys (in `configs/train.yaml`):
- `resume: false` — bool.
- `resume_from: null` — explicit checkpoint path.
- `force: false` — allow overwriting a non-empty save_dir when not resuming.

Resolution order at startup:
1. `resume_from` set → load that `.pt`; read its sibling `train_state.json` for the
   step + wandb id (fall back to parsing `step_<N>.pt` if no sidecar).
2. else `resume=true` → find latest `step_*.pt` in `save_dir` + read
   `train_state.json`. If none found, error (resume requested but nothing to
   resume).
3. else → fresh run (apply clobber guard).

On resume: `agent.load_state_dict(torch.load(ckpt))`, `start_step = step`,
`set_seed(cfg.seed + start_step)` (avoid replaying the same minibatch sequence),
and run `for step in range(start_step, cfg.num_train_steps + 1)`.

Loop-range semantics: `step_N.pt` is saved at the top of iteration `step==N`,
i.e. after `N` completed updates (steps `0..N-1`). Resuming at `range(N, …)`
re-checkpoints at `N` (idempotent), evals at `N`, then does update `N` — correct
continuation with no skipped or double-counted updates.

### 4. wandb continuation

- Capture `wandb.run.id` right after `wandb.init` and write it into every
  `train_state.json`.
- On resume, `wandb.init(id=saved_id, resume="allow", …)` → one continuous run /
  curve. If the saved id is null (run had `use_wandb=false`, or died before first
  checkpoint), start fresh.

### 5. Launcher (`scripts/run_grid_sweep.sh`)

- Add optional `RESUME=1` env → append `resume=true` to every generated command.
- Because save_dir is now deterministic per run-name, re-running the same launcher
  invocation resumes jobs that have a checkpoint and starts the rest fresh — a
  one-command crash recovery for a whole grid.

### 6. JAX / GCIQL

- Core resume already functional — no code change required.
- Add a resume **smoke test** (tiny run → restore → assert continuation) and a
  short README/CLAUDE note documenting `restore_path`/`restore_epoch`/offset usage.
- *Optional ergonomic parity:* a `resume=true` convenience in `run_gciql` that
  auto-derives `restore_path`/`restore_epoch` from the latest `params_*.pkl` in the
  run's output dir, mirroring the PyTorch `resume=true` UX. Mark optional; drop if
  it complicates the plan.

## Testing

- **Unit:** sidecar write/read round-trip; latest-`step_*.pt` discovery (incl.
  empty dir, multiple checkpoints); loop-range math (resume at N runs N..max);
  clobber-guard trigger.
- **Integration smoke (PyTorch):** tiny run (e.g. `num_train_steps≈60`,
  `save_every=20`) for **both state and pixel** (pixel via the synthetic
  frame-stacked smoke buffer pattern used in existing pixel tests); kill/stop
  after the first checkpoint, resume, assert: it finishes, steps are monotonic and
  reach the target, wandb id reused (when wandb on), no clobber error.
- **Integration smoke (GCIQL):** tiny run → restore_epoch → assert it continues
  and step offset applies.
- Bit-exact equivalence is **not** asserted (crash-recovery bar).

## Out of scope

- Bit-exact / RNG-faithful resume.
- Persisting/checkpointing an online replay buffer (training is offline).
- Auto-detect/auto-resume on launch without an explicit flag.
- Changing checkpoint cadence or retention policy (`save_every` unchanged).

## Affected files

- `train.py` — resume resolution, sidecar write, loop range, wandb resume, clobber guard.
- `configs/train.yaml` — `resume`, `resume_from`, `force`, run-name-based `hydra.run.dir`.
- `scripts/run_grid_sweep.sh` — `RESUME=1` passthrough.
- `run_gciql.py` (+ shim) — optional `resume=true` convenience; docs.
- `tests/` — new unit + smoke tests per above.
