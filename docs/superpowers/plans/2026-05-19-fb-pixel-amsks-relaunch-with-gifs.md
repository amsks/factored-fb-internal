# FB-Pixel Relaunch on `amsks` with Eval GIFs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the in-flight FB-pixel full run on `al-laq-fb`, mirror the two existing al-laq-fb runs (smoke + partial full) into the `amsks/factored-fb` workspace, and relaunch the full 1M run under `amsks` with eval GIFs enabled, matching the state-FB pattern.

**Architecture:** Pure config + ops — no source edits. Both knobs (`wandb_entity`, `save_eval_videos`) are already Hydra-overridable in `train.py`; eval GIFs and `wandb.Video` upload are already implemented in `OGBenchEvaluator`. Three sequential ops: graceful stop → wandb sync mirroring → relaunch + healthy-startup verification.

**Tech Stack:** Python, Hydra, wandb (`wandb sync -e -p`), PyTorch, bash.

**Spec:** `docs/superpowers/specs/2026-05-19-fb-pixel-amsks-relaunch-with-gifs-design.md`

**Conventions (locked):**
- Commit messages: **do not** add a Claude `Co-Authored-By` trailer (repo convention).
- Tests run with `.venv/bin/python -m pytest` (`uv run` is broken here).
- No source changes. Plan tasks are operational + a single Hydra-compose sanity check; no new pytest is added because no production code changes.
- Existing `al-laq-fb` runs are left alone (user deletes from wandb UI later if desired). The 50k smoke is NOT re-run (spec out-of-scope #3).

**Critical pid:** the current al-laq-fb full run pid was **3481130** at design time; **re-discover at execution time** (`pgrep -fa "train.py domain=visual_cube_single .* full1M$"`) — do not hard-code.

---

## File Structure

- **No source files modified.** Operational only.
- **Logs produced (new):**
  - `logs/wandb_sync_amsks.log` — wandb sync output for both mirror commands.
  - `logs/fb_pixel_full_s0_amsks.log` — relaunched 1M run's stdout/stderr.
- **Wandb local dirs touched (read-only sync):**
  - `wandb/run-20260519_195608-vxo86krs` (smoke, finished).
  - `wandb/run-20260519_201504-c6i3in90` (partial full, will be finalized by Task 1).
- **Output dirs created by the new run:** `outputs/<ts>__cube-single-play-v0__fb_flowbc__s0/` with `checkpoints/` and `eval_videos/step_<N>/<task>.gif`.

---

### Task 1: Graceful stop of the running full run

**Files:** none modified.

- [ ] **Step 1: Re-discover the live pid (do NOT trust the design-time pid)**

Run:
```bash
PID=$(pgrep -f 'train.py domain=visual_cube_single .* full1M$' | head -1)
echo "pid=$PID"
```
Expected: a numeric pid (e.g., `pid=3481130`). If empty, the run has already exited — skip Task 1 Steps 2–3 and proceed to Task 2.

- [ ] **Step 2: Send SIGTERM (graceful) and wait for exit, up to ~120 s**

Run (in repo root):
```bash
PID=$(pgrep -f 'train.py domain=visual_cube_single .* full1M$' | head -1)
[ -z "$PID" ] && { echo "(already gone)"; exit 0; }
kill -TERM "$PID"
for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || { echo "exited cleanly"; exit 0; }
    sleep 2
done
echo "still alive after 120s; escalating to SIGINT"
kill -INT "$PID" 2>/dev/null || true
for _ in $(seq 1 15); do
    kill -0 "$PID" 2>/dev/null || { echo "exited after SIGINT"; exit 0; }
    sleep 2
done
echo "still alive after SIGINT; escalating to SIGKILL (wandb dir may be partial)"
kill -KILL "$PID" 2>/dev/null || true
```
Expected: `exited cleanly` (normal path). Note in the report if SIGINT/SIGKILL escalation occurred — partial wandb sync output may follow in Task 2.

- [ ] **Step 3: Confirm the wandb local dir is closed**

Run:
```bash
ls -la wandb/run-20260519_201504-c6i3in90 | head
ls wandb/run-20260519_201504-c6i3in90/files/ 2>/dev/null | head
```
Expected: dir exists and contains the run's files. The presence of `wandb-metadata.json` and a `logs/` subdir is a good proxy for "the run was tracked". `wandb-summary.json` *may* be missing if SIGKILL was needed; that is acceptable — `wandb sync` still uploads the captured history.

---

### Task 2: Mirror existing `al-laq-fb` runs into `amsks/factored-fb`

**Files:** none modified. Writes `logs/wandb_sync_amsks.log`.

- [ ] **Step 1: Sync the finished smoke run (`vxo86krs`)**

Run (repo root):
```bash
.venv/bin/python -m wandb sync -e amsks -p factored-fb \
    wandb/run-20260519_195608-vxo86krs \
    2>&1 | tee -a logs/wandb_sync_amsks.log
```
Expected: output ends with a `View run` line containing a new URL of the form `https://wandb.ai/amsks/factored-fb/runs/<new-id>`. Capture this URL for the report. Non-fatal warnings (e.g., "already synced") are acceptable.

- [ ] **Step 2: Sync the stopped partial full run (`c6i3in90`)**

Run (repo root):
```bash
.venv/bin/python -m wandb sync -e amsks -p factored-fb \
    wandb/run-20260519_201504-c6i3in90 \
    2>&1 | tee -a logs/wandb_sync_amsks.log
```
Expected: same as Step 1 — a new amsks URL printed. If SIGKILL was used in Task 1, expect a warning about a missing final summary; the captured history (up to the kill point) still uploads. Capture the URL.

- [ ] **Step 3: Verify the two new amsks URLs are reachable**

Run:
```bash
.venv/bin/python - <<'EOF'
import re, sys
log = open("logs/wandb_sync_amsks.log").read()
urls = re.findall(r"https://wandb\.ai/amsks/factored-fb/runs/[A-Za-z0-9]+", log)
urls = list(dict.fromkeys(urls))  # de-dup, preserve order
assert len(urls) >= 2, f"expected 2 mirror URLs, found {urls}"
print("mirrored runs:")
for u in urls[:2]:
    print(" ", u)
import wandb
api = wandb.Api()
for u in urls[:2]:
    rid = u.rsplit("/", 1)[-1]
    r = api.run(f"amsks/factored-fb/{rid}")
    print(f"  {rid}: state={r.state}  history_rows={r.lastHistoryStep}")
EOF
```
Expected: both URLs printed, each resolving via the wandb API with `state` and a `history_rows` count > 0. If `lastHistoryStep` is `None` for the partial run, that's tolerable — Task 2 is best-effort historical mirroring, not a blocker for Task 3.

---

### Task 3: Pre-launch Hydra compose sanity for the amsks relaunch

**Files:** none modified.

- [ ] **Step 1: Hydra-compose the exact override set we'll launch with**

Run (repo root):
```bash
.venv/bin/python - <<'EOF'
from hydra import compose, initialize
with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(
        config_name="train",
        overrides=[
            "domain=visual_cube_single",
            "data_path=/dev/shm/factored-fb/datasets",
            "seed=0",
            "use_wandb=true",
            "wandb_entity=amsks",
            "save_eval_videos=true",
            "wandb_run_name=visual-cube-single__fb_flowbc__s0__full1M_amsks",
        ],
    )
assert cfg.wandb_entity == "amsks",            cfg.wandb_entity
assert cfg.save_eval_videos is True,           cfg.save_eval_videos
assert cfg.use_wandb is True,                  cfg.use_wandb
assert cfg.obs_type == "pixels",               cfg.obs_type
assert cfg.frame_stack == 3,                   cfg.frame_stack
assert cfg.domain == "cube-single-play-v0",    cfg.domain
assert str(cfg.data_path) == "/dev/shm/factored-fb/datasets", cfg.data_path
assert int(cfg.num_train_steps) == 1_000_000,  cfg.num_train_steps
print("OK: compose; loader will read",
      f"{cfg.data_path}/{cfg.domain}/buffer")
print("OK: wandb -> amsks/factored-fb (entity=amsks)")
print("OK: save_eval_videos=True; GIFs will be written under "
      "${hydra:runtime.output_dir}/eval_videos")
EOF
```
Expected: three `OK:` lines, no assertion failure. If anything fails, STOP and reconcile — do not launch.

- [ ] **Step 2: Confirm scratch + GPU are healthy**

Run:
```bash
df -h /dev/shm | tail -1
ls /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/ | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
```
Expected: tmpfs has ≥ ~30 GB free, the buffer dir has 1000 files, GPU 1 is now idle (≤ ~200 MiB used after Task 1 freed it). If GPU 1 still shows the old run's footprint (~6 GB), Task 1 did not fully release the GPU — wait ~10 s and re-check before launching.

---

### Task 4: Launch the amsks relaunch run

**Files:** none modified. Writes `logs/fb_pixel_full_s0_amsks.log`.

- [ ] **Step 1: Background-launch the run on GPU 1**

Run (repo root):
```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
    domain=visual_cube_single \
    data_path=/dev/shm/factored-fb/datasets \
    seed=0 use_wandb=true \
    wandb_entity=amsks \
    save_eval_videos=true \
    wandb_run_name=visual-cube-single__fb_flowbc__s0__full1M_amsks \
    > logs/fb_pixel_full_s0_amsks.log 2>&1 &
echo "launched pid=$! -> logs/fb_pixel_full_s0_amsks.log"
```
Expected: prints a pid. The process runs detached.

- [ ] **Step 2 (executor note):** This is a long-running training job (~11 h at 25 FPS). Do NOT busy-wait. Use a single `run_in_background` Bash `until` loop polling the log for the healthy-startup signal or a failure — exactly the pattern used for the prior smoke/full launches in this branch.

- [ ] **Step 3: Watch the log for healthy startup (first train step OR first GIF OR failure)**

Run (repo root, in background):
```bash
until grep -qE '^\[step +[0-9]' logs/fb_pixel_full_s0_amsks.log \
   || ls outputs/2026-05-19_*__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/*.gif 2>/dev/null | grep -q . \
   || grep -qiE 'Error executing job|RuntimeError|AssertionError|FileNotFoundError|out of memory|CUDA error|Killed' logs/fb_pixel_full_s0_amsks.log \
   || ! pgrep -f 'train.py domain=visual_cube_single .* full1M_amsks$' >/dev/null; do
    sleep 20
done
echo "=== WATCH EXITED ==="
echo "--- proc ---"
pgrep -fa 'train.py domain=visual_cube_single .* full1M_amsks$' || echo "(gone)"
echo "--- wandb url ---"
grep -aoE 'https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+' logs/fb_pixel_full_s0_amsks.log | head -1
echo "--- milestones ---"
grep -aE '^\[train\]|^\[eval|^\[step|Error executing job' logs/fb_pixel_full_s0_amsks.log | tail -8
echo "--- gifs so far ---"
ls outputs/2026-05-19_*__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/*.gif 2>/dev/null | head
```
Expected (success): process ALIVE, a wandb URL whose path starts with `wandb.ai/amsks/factored-fb/runs/`, at least one `[train] buffer size: 1,000,000` milestone, and either a `[step ...]` line OR at least one GIF under `eval_videos/step_0/`. If the watch exits because of a failure signature or process death, STOP and capture/report — do not relaunch blindly.

- [ ] **Step 4: Verify the run landed in `amsks/factored-fb` via API**

Run (repo root):
```bash
.venv/bin/python - <<'EOF'
import re, wandb
url = re.search(r"https://wandb\.ai/(\S+)/runs/([A-Za-z0-9]+)",
                open("logs/fb_pixel_full_s0_amsks.log").read())
assert url, "no wandb URL in log yet"
entity_project, rid = url.group(1), url.group(2)
assert entity_project == "amsks/factored-fb", entity_project
api = wandb.Api()
r = api.run(f"{entity_project}/{rid}")
print(f"OK: amsks/factored-fb/{rid}  state={r.state}  history_rows={r.lastHistoryStep}")
print(f"  config keys present: wandb_entity={r.config.get('wandb_entity')}  "
      f"save_eval_videos={r.config.get('save_eval_videos')}")
EOF
```
Expected: prints `OK: amsks/factored-fb/<rid>  state=running  history_rows=<N>` (or `finished` if it somehow finished — unlikely for a 1M run within the watch window), with `wandb_entity=amsks` and `save_eval_videos=True` confirmed from the wandb-stored config.

---

### Task 5: GIF presence verification (after first eval interval)

**Files:** none modified.

- [ ] **Step 1: Wait for the first eval's GIFs to land, then assert presence**

Train.py runs an eval at `step=0` (`[eval] step=0`) before the first training step; if `save_eval_videos=true`, `OGBenchEvaluator.run()` writes per-task GIFs under
`outputs/<ts>__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/<task>.gif`
and uploads them as `wandb.Video`. Run (repo root, in background):

```bash
until ls outputs/2026-05-19_*__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/*.gif 2>/dev/null | grep -q . \
   || grep -qiE 'Error executing job|RuntimeError|AssertionError|FileNotFoundError|out of memory|CUDA error|Killed' logs/fb_pixel_full_s0_amsks.log \
   || ! pgrep -f 'train.py domain=visual_cube_single .* full1M_amsks$' >/dev/null; do
    sleep 30
done
echo "=== GIF WATCH EXITED ==="
GIF_GLOB="outputs/2026-05-19_*__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/*.gif"
GIFS=$(ls $GIF_GLOB 2>/dev/null | wc -l)
echo "step_0 GIFs found: $GIFS"
ls -lh $GIF_GLOB 2>/dev/null | head -5
```
Expected: `step_0 GIFs found: 5` (five eval tasks in `ALL_TASKS["cube-single-play-v0"]`), each a small GIF file (sub-MB to a few MB). If `0`, that is a regression in the committed GIF code — capture the log, STOP, and report (out of scope to fix here per spec; would surface as a separate debugging task).

- [ ] **Step 2: Confirm the wandb run has `eval_video/...` keys**

Run (repo root):
```bash
.venv/bin/python - <<'EOF'
import re, wandb
url = re.search(r"https://wandb\.ai/(\S+)/runs/([A-Za-z0-9]+)",
                open("logs/fb_pixel_full_s0_amsks.log").read())
ep, rid = url.group(1), url.group(2)
r = wandb.Api().run(f"{ep}/{rid}")
keys = [k for k in r.summary.keys() if "eval_video" in k or "Video" in k or "video" in k.lower()]
print("video-ish summary keys:", keys[:10])
# Also scan the first few history rows for video media-type keys
h = list(r.scan_history(min_step=0, max_step=1))
vid_keys = sorted({k for row in h for k in row if "eval_video" in k.lower() or "video" in k.lower()})
print("video-ish history keys (step<=1):", vid_keys[:10])
EOF
```
Expected: at least one video-ish key surfaced (either in summary or in early history). The exact namespace depends on the committed shim (`feat(gciql): log eval video under FB eval_video/ namespace`, commit `ff6e05b`). If the run hasn't streamed videos yet (slow upload), `vid_keys` may be empty for a few minutes; re-run this step after another ~2 min. If still empty after ~10 min, STOP and report (potential silent video-upload issue — investigate separately).

---

### Task 6: Final report + handoff

**Files:** none modified.

- [ ] **Step 1: Compose the run-status summary**

Run (repo root):
```bash
.venv/bin/python - <<'EOF'
import re, wandb
log = open("logs/fb_pixel_full_s0_amsks.log").read()
url = re.search(r"https://wandb\.ai/(\S+)/runs/([A-Za-z0-9]+)", log)
ep, rid = url.group(1), url.group(2)
r = wandb.Api().run(f"{ep}/{rid}")
print(f"new amsks run: https://wandb.ai/{ep}/runs/{rid}")
print(f"  state={r.state}  history_rows={r.lastHistoryStep}")
for k in ("train/fb_loss","train/actor_loss","train/bc_flow_loss",
          "eval/reward/eval/reward","eval/reward/eval/success"):
    if k in r.summary: print(f"  {k}={r.summary[k]}")

# Also report the mirrored runs
sync_log = open("logs/wandb_sync_amsks.log").read()
mirrors = list(dict.fromkeys(
    re.findall(r"https://wandb\.ai/amsks/factored-fb/runs/[A-Za-z0-9]+", sync_log)
))[:2]
print("mirrored runs:")
for u in mirrors: print(" ", u)
EOF
```
Expected: a single status block with the new amsks run URL, latest metrics, and the two mirrored run URLs from Task 2. Report this verbatim to the user, plus note the dataset is on tmpfs (`/dev/shm/factored-fb/datasets`) and will not survive a reboot.

- [ ] **Step 2: Note the operational state for the user**

Report:
- New amsks run URL + state (running) + latest metrics block from Step 1.
- Two mirrored URLs (smoke + partial-full).
- Reminder: long-running (~11 h at ~25 FPS), GIFs at every eval interval, checkpoints every 200k steps under `outputs/<ts>.../checkpoints`.
- Reminder: `/dev/shm/factored-fb/datasets` is RAM-backed; do not reboot mid-run, and the dataset can be regenerated with the Task 3 command from the previous extraction plan.

---

## Self-Review

**Spec coverage:**
- Decision 1 (stop & restart from scratch under amsks with GIFs) → Tasks 1, 4 ✓
- Decision 2 (mirror existing al-laq-fb runs to amsks via `wandb sync -e -p`) → Task 2 ✓
- Decision 3 (graceful SIGTERM with SIGINT/SIGKILL escalation) → Task 1 Step 2 ✓
- Decision 4 (GPU 1, dataset on `/dev/shm`, smoke untouched) → Task 3 Step 2 + Task 4 Step 1 ✓
- Decision 5 (run name `visual-cube-single__fb_flowbc__s0__full1M_amsks`) → Task 3 Step 1 + Task 4 Step 1 ✓
- Component 1 (graceful stop) → Task 1 ✓
- Component 2 (wandb sync mirroring) → Task 2 ✓
- Component 3 (relaunch + healthy-startup watch + entity verification + GIF presence) → Tasks 3, 4, 5 ✓
- Testing: pre-flight Hydra-compose sanity → Task 3 Step 1; healthy-startup watch → Task 4 Step 3; GIF presence verification → Task 5 ✓
- Out-of-scope respected: no resume hook added, no source changes, no smoke rerun, no al-laq-fb deletion ✓
- Risks acknowledged: SIGKILL → partial sync (Task 1 Step 2 + Task 2 Step 2 warning), GPU release wait (Task 3 Step 2), tmpfs reminder (Task 6 Step 2) ✓

**Placeholder scan:** no TBD/TODO/"similar to Task N"; every command shows full text + expected output + explicit STOP condition on failure.

**Type / interface consistency:**
- Hydra override `wandb_entity=amsks` appears identically in Task 3 Step 1 (sanity-check) and Task 4 Step 1 (launch).
- `save_eval_videos=true` appears identically in both places.
- `wandb_run_name=visual-cube-single__fb_flowbc__s0__full1M_amsks` is consistent across launch (Task 4) and the watch pattern `train.py domain=visual_cube_single .* full1M_amsks$` (Task 4 Step 3, Task 5 Step 1).
- Output dir glob `outputs/2026-05-19_*__cube-single-play-v0__fb_flowbc__s0/eval_videos/step_0/*.gif` is identical across Tasks 4 Step 3 and 5 Step 1.
- Log paths `logs/wandb_sync_amsks.log` (Task 2) and `logs/fb_pixel_full_s0_amsks.log` (Task 4) are referenced consistently by their downstream tasks (Tasks 4 Step 4, 5 Step 2, 6 Step 1).
