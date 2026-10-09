# FB-Pixel Relaunch on `amsks` Workspace with Eval GIFs — Design Spec

**Date:** 2026-05-19
**Status:** Approved (pending written-spec review)
**Branch:** `gciql-pixel-integration`

## Goal

Move the FB-pixel results into the user's wandb workspace (`amsks/factored-fb`)
and produce per-task eval GIFs during training, matching the state-FB pattern.
**No source changes** — both knobs are already Hydra-overridable.

## Context

- Two existing runs sit on the `al-laq-fb` team workspace:
  - `vxo86krs` — 50k smoke (finished, no GIFs).
  - `c6i3in90` — 1M run, ~148k/1M, still running, no GIFs.
- `train.py` already accepts `wandb_entity` (`configs/train.yaml:32`, default `null`) and `save_eval_videos` (`configs/train.yaml:26`, default `false`). When `save_eval_videos=true`, `OGBenchEvaluator(save_videos=True, video_dir=…)` writes per-task GIFs under `${hydra:runtime.output_dir}/eval_videos` and uploads them via `wandb.Video`. Lines `train.py:229-245`.
- `train.py` has **no resume-from-checkpoint hook**; first checkpoint isn't written until `save_every=200_000`, so the ~148k of progress is unrecoverable. User accepted the sunk cost.

## Decisions (locked with user)

1. **Stop and restart from scratch** under `wandb_entity=amsks` with `save_eval_videos=true`. Loses ~90 min; gains a clean run in the user's workspace with in-training GIFs at every eval, matching state-FB exactly.
2. **Mirror the two existing al-laq-fb runs** to `amsks/factored-fb` via `wandb sync -e amsks -p factored-fb <local-dir>` for historical visibility. The smoke (`vxo86krs`) is finished and syncs cleanly; the partial full (`c6i3in90`) is synced *after* SIGTERM lets wandb finalize its local dir.
3. **Graceful stop**, not SIGKILL: `kill -TERM` and wait for exit, so wandb closes the run, finalizes the dir, and flushes any pending uploads.
4. **GPU 1**, same dataset at `/dev/shm/factored-fb/datasets`. Smoke run remains untouched (already finished).
5. **Run name** for the new run: `visual-cube-single__fb_flowbc__s0__full1M_amsks` to disambiguate from the al-laq-fb run.

## Components

### Component 1 — Graceful stop of the current full run
- Identify pid: `pgrep -fa "train.py domain=visual_cube_single .* full1M$"`.
- Send `SIGTERM` to that pid; wait until the process exits (poll `pgrep` until empty). Hard cap the wait at ~120 s; if it does not exit, escalate to `SIGINT` then `SIGKILL` and log that wandb dir may be partial.
- Confirm the local dir `wandb/run-20260519_201504-c6i3in90` contains a finalized run (presence of `wandb-events.jsonl` / `wandb-summary.json` is acceptable proxy; do not block if missing — just note it).

### Component 2 — Mirror existing runs into `amsks/factored-fb`
- For each of the two local dirs, run:
  ```
  .venv/bin/python -m wandb sync -e amsks -p factored-fb <wandb/run-...>
  ```
- Expected: each command prints a new run URL under `wandb.ai/amsks/factored-fb/...`. Captured to a log file for the report. Failures are non-fatal (don't block the relaunch).

### Component 3 — Relaunch the full 1M run under `amsks` with GIFs
- Command (background, GPU 1, fresh log file):
  ```
  CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
      domain=visual_cube_single \
      data_path=/dev/shm/factored-fb/datasets \
      seed=0 use_wandb=true \
      wandb_entity=amsks \
      save_eval_videos=true \
      wandb_run_name=visual-cube-single__fb_flowbc__s0__full1M_amsks \
      > logs/fb_pixel_full_s0_amsks.log 2>&1 &
  ```
- Pre-flight: a Hydra-compose sanity check (no training) asserting
  `cfg.wandb_entity == "amsks"`, `cfg.save_eval_videos is True`, paths intact.
- Healthy-startup watch: wait until the log shows either a `[step ]` training
  line, an `eval_videos/` GIF written (`*.gif` under the new run's
  `eval_videos/`), or a failure signature.
- Confirm the printed wandb URL points at `wandb.ai/amsks/factored-fb/...`.

## Data Flow

```
Stop full run (SIGTERM pid 3481130)  ──┐
                                       ├─→ wandb sync -e amsks -p factored-fb (mirror)
Smoke local dir vxo86krs ──────────────┘
                                              ↓
                                       wandb.ai/amsks/factored-fb/<new-ids>
                                              ↑
                                              │
Relaunch train.py wandb_entity=amsks save_eval_videos=true
   ├─ writes ckpts under outputs/<ts>__cube-single-play-v0__fb_flowbc__s0/checkpoints
   ├─ writes per-task GIFs under outputs/<ts>.../eval_videos/step_<N>/<task>.gif
   └─ logs metrics + wandb.Video to wandb.ai/amsks/factored-fb/<new-run-id>
```

## Testing

- **Pre-flight test** (cheap, before launch): Hydra `compose("train", overrides=[...])` and assert `wandb_entity`, `save_eval_videos`, `data_path`, `frame_stack` are set as expected.
- **Healthy-startup verification** (operational, post-launch): block on the log until first `[step ]` line OR first `eval_videos/...gif` appears OR a fatal traceback / OOM / process death. Report wandb URL, ensure entity is `amsks`.
- **GIF presence verification**: after the first eval interval (step 0), confirm at least one `<output_dir>/eval_videos/step_0/<task>.gif` exists. If absent at step 0, that is a regression in the existing committed GIF code (out of scope — would surface as a separate bug).
- No new pytest needed (no source change; existing pixel-FB suite already covers the wiring).

## Out of Scope

- Adding a resume-from-checkpoint code path to `train.py` (user declined Q1's "add resume hook" option).
- Deleting the original `al-laq-fb` runs (user deletes from the wandb UI if/when desired).
- Re-running the 50k smoke (user declined Q2's "also rerun smoke" option).
- Any change to `OGBenchEvaluator` GIF code or `wandb.Video` upload — relying on the already-committed state-FB pattern.

## Risks

- **wandb sync of the partial `c6i3in90` dir** may yield a slightly-incomplete-looking history (no final summary line, last-step truncation). Acceptable; the new authoritative run will be the relaunch.
- **Graceful-stop timeout**: if the running process ignores SIGTERM, escalation to SIGKILL is allowed; wandb local dir may then be partial — sync may emit warnings, not errors. Non-blocking.
- **Disk / tmpfs**: dataset already on `/dev/shm` (273 GB free, RAM ample); no change.
- **GPU 1** is the one currently in use by the to-be-killed run; after stop it frees and the relaunch reuses it.
