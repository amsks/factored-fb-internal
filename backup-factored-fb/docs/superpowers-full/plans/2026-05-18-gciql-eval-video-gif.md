# GCIQL eval-video GIF + FB key scheme — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GCIQL's wandb eval video log as a GIF (fps=30, loop=0, FB's exact `imageio` params) and additionally appear under FB's `eval_video/` key namespace, without editing vendored OGBench.

**Architecture:** Extend the auto-imported shim `tools/wandb_mode_shim/sitecustomize.py`. Inside its existing `GCIQL_FB_ENV` branch, (a) wrap `wandb.Video` so an OGBench-style `(t,c,H,W)` uint8 ndarray passed with `format='mp4'` is re-encoded to a GIF via `imageio.mimsave`, and (b) extend the existing `wandb.log` re-key wrapper so a `video` entry is also logged under `eval_video/all_tasks` (original `video` kept). Any failure falls back to the original mp4 path; inert unless `GCIQL_FB_ENV` is set.

**Tech Stack:** Python, wandb 0.27, imageio 2.37 (JAX venv `/dev/shm/.venv-jax`), numpy. Tests are plain `assert` scripts (repo convention, see `scripts/test_run_gciql.py`) run with the JAX venv interpreter.

---

## File Structure

- **Modify:** `tools/wandb_mode_shim/sitecustomize.py` — add the `wandb.Video` GIF wrapper and the `video → eval_video/all_tasks` re-key. Single responsibility unchanged (it is the no-edit-vendored wandb adapter for the GCIQL child).
- **Create:** `scripts/test_wandb_shim_video.py` — standalone, no GPU/JAX/network. Loads the shim fresh per case via `importlib`, asserts behavior on the real `wandb.Video` and a stubbed `wandb.log`.
- **Modify:** `Makefile` — add a `test-wandb-shim` target (parallels existing `test-gciql`).

The currently-running sweep is unaffected (sitecustomize is read at child-process start); the new format applies on the next relaunch.

---

## Task 1: `wandb.Video` → GIF wrapper

**Files:**
- Create: `scripts/test_wandb_shim_video.py`
- Modify: `tools/wandb_mode_shim/sitecustomize.py` (inside the `if _fb_env:` block)

- [ ] **Step 1: Write the failing test**

Create `scripts/test_wandb_shim_video.py`:

```python
"""Verifies tools/wandb_mode_shim video GIF re-encoding + FB re-key.

Run with the JAX venv (has wandb/imageio/numpy):
    /dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import imageio
import numpy as np
import wandb

SHIM = (
    Path(__file__).resolve().parent.parent
    / "tools" / "wandb_mode_shim" / "sitecustomize.py"
)

_ORIG_VIDEO = wandb.Video
_ORIG_LOG = wandb.log
_ORIG_INIT = wandb.init
_GATE_VARS = (
    "GCIQL_WANDB_FORCE_MODE",
    "GCIQL_FB_ENV",
    "GCIQL_WANDB_PROJECT",
    "GCIQL_WANDB_ENTITY",
)
_load_count = 0


def load_shim(env):
    """Restore real wandb attrs, set env, exec the shim fresh."""
    global _load_count
    wandb.Video = _ORIG_VIDEO
    wandb.log = _ORIG_LOG
    wandb.init = _ORIG_INIT
    for k in _GATE_VARS:
        os.environ.pop(k, None)
    os.environ.update(env)
    _load_count += 1
    spec = importlib.util.spec_from_file_location(
        f"_shim_under_test_{_load_count}", SHIM
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# Case 1: OGBench-style ndarray+mp4 -> GIF-backed wandb.Video
load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"})
arr = np.zeros((4, 3, 8, 8), dtype=np.uint8)  # (t, c, H, W) like OGBench
v = wandb.Video(arr, fps=15, format="mp4")
assert isinstance(v, _ORIG_VIDEO), type(v)
assert getattr(v, "_format", None) == "gif", getattr(v, "_format", None)
assert str(getattr(v, "_path", "")).endswith(".gif"), getattr(v, "_path", None)
print("CASE 1 ok: ndarray+mp4 -> gif")

# Case 2: passthrough for a non-mp4 path arg (no re-encode, identity of path)
gp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
gp.close()
imageio.mimsave(gp.name, np.zeros((2, 8, 8, 3), np.uint8), format="GIF",
                fps=30, loop=0)
load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"})
v2 = wandb.Video(gp.name, fps=5, format="gif")
assert str(getattr(v2, "_path", "")) == gp.name, getattr(v2, "_path", None)
os.unlink(gp.name)
print("CASE 2 ok: non-mp4 path passthrough")

# Case 3: inert when no gate var set (wandb.Video unchanged identity)
load_shim({})
assert wandb.Video is _ORIG_VIDEO, "Video should be untouched when inert"
print("CASE 3 ok: inert without GCIQL_FB_ENV")

print("ALL VIDEO-WRAPPER CASES PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py`
Expected: FAIL at CASE 1 — `AssertionError` on `_format` (currently the shim does not wrap `wandb.Video`, so `wandb.Video(arr, format="mp4")` either errors invoking moviepy or returns a non-gif Video). Any non-pass output counts as the expected failing state.

- [ ] **Step 3: Add the `wandb.Video` wrapper to the shim**

In `tools/wandb_mode_shim/sitecustomize.py`, inside the `if _fb_env:` block, immediately after the line:

```python
            _env_base = _fb_env[:-3] if _fb_env.endswith("-v0") else _fb_env
```

insert:

```python

            # OGBench logs its tiled eval clip as
            # wandb.Video(ndarray, fps=15, format='mp4') where the array is
            # (t, c, H, W) uint8 (get_wandb_video -> reshape_video in
            # third_party/ogbench/impls/utils/log_utils.py). Re-encode it as a
            # GIF with FB's exact imageio params so GCIQL eval media matches
            # FB's recording (evals/ogbench.py).
            import numpy as _np
            import tempfile as _tempfile
            import imageio as _imageio

            _orig_video = _wandb.Video

            def _video(data_or_path, caption=None, fps=None, format=None):
                try:
                    if isinstance(data_or_path, _np.ndarray) and format == "mp4":
                        _frames = _np.transpose(data_or_path, (0, 2, 3, 1))
                        _tmp = _tempfile.NamedTemporaryFile(
                            suffix=".gif", delete=False
                        )
                        _tmp.close()
                        _imageio.mimsave(
                            _tmp.name, _frames, format="GIF", fps=30, loop=0
                        )
                        return _orig_video(
                            _tmp.name, caption=caption, fps=30, format="gif"
                        )
                except Exception:
                    pass
                return _orig_video(
                    data_or_path, caption=caption, fps=fps, format=format
                )

            _wandb.Video = _video
```

- [ ] **Step 4: Run test to verify CASES 1–3 pass**

Run: `/dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py`
Expected: prints `CASE 1 ok`, `CASE 2 ok`, `CASE 3 ok`, `ALL VIDEO-WRAPPER CASES PASS`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/sitecustomize.py scripts/test_wandb_shim_video.py
git commit -m "feat(gciql): re-encode OGBench eval video as GIF in wandb shim"
```

---

## Task 2: Re-key `video` → `eval_video/all_tasks`

**Files:**
- Modify: `tools/wandb_mode_shim/sitecustomize.py` (the `_fb_keys` function)
- Modify: `scripts/test_wandb_shim_video.py` (append a log re-key case)
- Modify: `Makefile` (add `test-wandb-shim` target)

- [ ] **Step 1: Add the failing log re-key test**

In `scripts/test_wandb_shim_video.py`, replace the final line
`print("ALL VIDEO-WRAPPER CASES PASS")` with:

```python
print("ALL VIDEO-WRAPPER CASES PASS")

# Case 4: wandb.log({'video': ...}) also emits eval_video/all_tasks,
# keeps the original 'video', and still mirrors scalar FB keys.
_captured = {}


def _recorder(data=None, *args, **kwargs):
    _captured.clear()
    if isinstance(data, dict):
        _captured.update(data)
    return None


wandb.log = _recorder  # shim will wrap THIS as its original
load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"})
_SENTINEL = object()
wandb.log(
    {
        "video": _SENTINEL,
        "evaluation/overall_success": 1.0,
        "evaluation/task1_pick_success": 0.5,
    },
    step=7,
)
assert _captured.get("eval_video/all_tasks") is _SENTINEL, _captured
assert _captured.get("video") is _SENTINEL, "original video key must remain"
assert _captured.get("eval/reward/eval/success") == 1.0, _captured
assert (
    _captured.get("eval/reward/cube-single-play-singletask-task1-v0/success")
    == 0.5
), _captured
print("CASE 4 ok: video re-keyed to eval_video/all_tasks (+ scalars)")

print("ALL CASES PASS")
```

- [ ] **Step 2: Run test to verify CASE 4 fails**

Run: `/dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py`
Expected: CASES 1–3 pass, then FAIL at CASE 4 — `AssertionError` because `_fb_keys` does not yet emit `eval_video/all_tasks`.

- [ ] **Step 3: Add the video re-key in `_fb_keys`**

In `tools/wandb_mode_shim/sitecustomize.py`, in the `_fb_keys` function, the loop currently reads:

```python
                for k, v in list(data.items()):
                    if k == "evaluation/overall_success":
                        extra["eval/reward/eval/success"] = v
                        continue
                    m = _task_re.match(k)
```

Insert a `video` branch so it becomes:

```python
                for k, v in list(data.items()):
                    if k == "evaluation/overall_success":
                        extra["eval/reward/eval/success"] = v
                        continue
                    if k == "video":
                        extra["eval_video/all_tasks"] = v
                        continue
                    m = _task_re.match(k)
```

- [ ] **Step 4: Run test to verify all cases pass**

Run: `/dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py`
Expected: `CASE 1 ok` … `CASE 4 ok`, final line `ALL CASES PASS`, exit 0.

- [ ] **Step 5: Add Makefile target**

In `Makefile`, after the `test-gciql:` target block:

```makefile
test-gciql:
	python scripts/test_run_gciql.py
```

add:

```makefile

## Unit test for the wandb shim video GIF + re-key behavior
test-wandb-shim:
	/dev/shm/.venv-jax/bin/python scripts/test_wandb_shim_video.py
```

and append `test-wandb-shim` to the `.PHONY:` line at the top of the file
(the line beginning `.PHONY: install ...`).

- [ ] **Step 6: Run the Makefile target to confirm it wires up**

Run: `make test-wandb-shim`
Expected: same output as Step 4, ending `ALL CASES PASS`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add tools/wandb_mode_shim/sitecustomize.py scripts/test_wandb_shim_video.py Makefile
git commit -m "feat(gciql): log eval video under FB eval_video/ namespace"
```

---

## Task 3: Offline end-to-end verification (no relaunch of the live sweep)

**Files:** none (verification only)

- [ ] **Step 1: Tiny offline GCIQL run exercising the eval video path**

Run:

```bash
cd /home/adityamohan/git/Factored-FB
rm -rf /dev/shm/vidcheck_hydra /dev/shm/gciql_outputs/OGBench/_vidcheck
CUDA_VISIBLE_DEVICES=0 timeout 600 python run_gciql.py \
  --config-name cube_single_state storage=shm seed=888 \
  train_steps=2 log_interval=1 eval_interval=1 save_interval=1000 \
  eval_episodes=1 wandb_mode=offline run_group=_vidcheck \
  hydra.run.dir=/dev/shm/vidcheck_hydra 2>&1 | tail -5
```

Expected: completes, prints `Encoding video...` and the `[run_gciql] ...`
launch line, no traceback.

- [ ] **Step 2: Confirm the offline run recorded a GIF (not mp4) for the eval video**

Run:

```bash
D=$(ls -1dt /tmp/tmp*/wandb/offline-run-* 2>/dev/null | head -1)
echo "run dir: $D"
find "$D" -name '*.gif' | head
find "$D" -name '*.mp4' | head
ls "$D"/files/media/videos 2>/dev/null || true
```

Expected: at least one `.gif` under the offline run's media; no eval `.mp4`
video. (Presence of `.gif` confirms the wrapper fired end-to-end through real
OGBench.)

- [ ] **Step 3: Clean verification artifacts**

Run:

```bash
rm -rf /dev/shm/vidcheck_hydra /dev/shm/gciql_outputs/OGBench/_vidcheck "$D"
```

Expected: no output, exit 0. (Does not touch the live sweep `201030`.)

- [ ] **Step 4: Final commit if anything was left staged**

```bash
git status --short
```

Expected: clean except untracked `checkpoints/` and `logs/` (pre-existing,
intentionally untracked).

---

## Self-Review

**Spec coverage:**
- GIF format / fps=30 / loop=0 via FB's `imageio.mimsave` → Task 1, Step 3.
- Single tiled layout kept (no de-tiling) → wrapper only re-encodes, never reshapes → Task 1.
- Additive `eval_video/all_tasks`, original `video` kept → Task 2, Step 3 + CASE 4 assertions.
- No vendored edits; only `sitecustomize.py`, test, Makefile → File Structure.
- Gated on `GCIQL_FB_ENV`, inert otherwise → Task 1 CASE 3.
- Error handling falls back to original → wrapper `try/except` (Task 1 Step 3); covered implicitly by passthrough CASE 2.
- Lazy `numpy/tempfile/imageio` imports inside the active branch → Task 1 Step 3.
- Rollout: live sweep untouched, verified offline → Task 3.

**Placeholder scan:** none — all steps contain full code/commands and expected output.

**Type/name consistency:** `_orig_video`/`_video`/`_wandb.Video`, `_fb_keys` key `eval_video/all_tasks`, env `GCIQL_FB_ENV`, test helper `load_shim`, attrs `_format`/`_path` (verified present in wandb 0.27) — consistent across Tasks 1–2 and the spec.
