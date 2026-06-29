# Training Resume (Crash-Recovery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a dead PyTorch training run (FB/PSM/RLDP, state or pixel) be relaunched and continue from its last checkpoint, with one continuous wandb curve; verify+document the already-working JAX/GCIQL resume and add a launcher convenience.

**Architecture:** Extract pure, torch-free resume helpers into `resume.py` (sidecar I/O, latest-checkpoint discovery, resume-plan resolution, clobber guard) and unit-test them in isolation. Wire them into `train.py` around the existing checkpoint/loop code. Make per-run output dirs deterministic (run-name-based) so a relaunch finds the prior checkpoint. JAX already supports resume via `restore_path`/`restore_epoch`; add a smoke test + docs.

**Tech Stack:** Python 3.11, PyTorch, Hydra/OmegaConf, pytest, bash. JAX/Flax for the GCIQL path (vendored OGBench).

**Spec:** `PAPER/specs/2026-06-09-training-resume-design.md`

**Test runner (this repo):** `uv run` is broken — always use `.venv/bin/python -m pytest`. Do NOT switch pytest to importlib mode (it breaks ~9 modules' collection; those pre-existing collection errors, mostly missing pandas, are not regressions). Commits in this repo OMIT the `Co-Authored-By: Claude` trailer.

---

## File Structure

- **Create `resume.py`** (repo root, next to `train.py`) — pure helpers: `write_train_state`, `read_train_state`, `find_latest_checkpoint`, `resolve_resume` (+ `ResumePlan`), `assert_no_clobber`. No torch import → fast to test.
- **Create `tests/test_resume.py`** — unit tests for all of the above.
- **Modify `train.py`** — import helpers; reorder so `save_dir`/resume-plan resolve before `wandb.init`; load checkpoint; re-seed; loop from `start_step`; write sidecar at each checkpoint; resume wandb run.
- **Modify `configs/train.yaml`** — add `resume`, `resume_from`, `force`; change `hydra.run.dir` to run-name-based.
- **Modify `scripts/run_grid_sweep.sh`** — `RESUME=1` appends `resume=true` to each job.
- **Create `scripts/smoke_resume.sh`** — tiny real end-to-end PyTorch resume smoke (state + pixel).
- **Create `scripts/smoke_resume_gciql.sh`** — GCIQL restore smoke.

---

## Task 1: Sidecar read/write helpers

**Files:**
- Create: `resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resume.py
import json
from pathlib import Path

import pytest

from resume import write_train_state, read_train_state


def test_write_then_read_roundtrip(tmp_path):
    write_train_state(tmp_path, step=400000, wandb_run_id="abc123", latest="step_400000.pt")
    got = read_train_state(tmp_path)
    assert got == {"step": 400000, "wandb_run_id": "abc123", "latest": "step_400000.pt"}


def test_write_creates_dir_and_overwrites(tmp_path):
    sub = tmp_path / "checkpoints"
    write_train_state(sub, step=10, wandb_run_id=None, latest="step_10.pt")
    write_train_state(sub, step=20, wandb_run_id=None, latest="step_20.pt")
    got = read_train_state(sub)
    assert got["step"] == 20 and got["latest"] == "step_20.pt"
    # exactly one sidecar file, valid JSON
    assert json.loads((sub / "train_state.json").read_text())["step"] == 20


def test_read_missing_returns_none(tmp_path):
    assert read_train_state(tmp_path) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resume'`.

- [ ] **Step 3: Write minimal implementation**

```python
# resume.py
"""Crash-recovery resume helpers for train.py.

Pure and torch-free: sidecar I/O, latest-checkpoint discovery, resume-plan
resolution, and the clobber guard. Kept out of train.py so they unit-test
without importing torch/hydra/agents. See
PAPER/specs/2026-06-09-training-resume-design.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple, Optional, Union

TRAIN_STATE_FILE = "train_state.json"
_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")

PathLike = Union[str, Path]


def write_train_state(save_dir: PathLike, step: int, wandb_run_id: Optional[str],
                      latest: str) -> None:
    """Write/overwrite the single sidecar that points at the newest checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(step), "wandb_run_id": wandb_run_id, "latest": latest}
    (save_dir / TRAIN_STATE_FILE).write_text(json.dumps(payload, indent=2))


def read_train_state(save_dir: PathLike) -> Optional[dict]:
    """Return the sidecar dict, or None if it does not exist."""
    p = Path(save_dir) / TRAIN_STATE_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add resume.py tests/test_resume.py
git commit -m "feat(resume): train_state sidecar read/write helpers"
```

---

## Task 2: Latest-checkpoint discovery

**Files:**
- Modify: `resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_resume.py
from resume import find_latest_checkpoint


def _touch(p: Path):
    p.write_bytes(b"x")


def test_find_latest_picks_highest_step(tmp_path):
    for n in (200000, 400000, 600000):
        _touch(tmp_path / f"step_{n}.pt")
    _touch(tmp_path / "final.pt")          # must be ignored
    _touch(tmp_path / "notes.txt")         # must be ignored
    path, step = find_latest_checkpoint(tmp_path)
    assert step == 600000
    assert path == tmp_path / "step_600000.pt"


def test_find_latest_empty_or_missing(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_resume.py::test_find_latest_picks_highest_step -q`
Expected: FAIL — `ImportError: cannot import name 'find_latest_checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to resume.py
def find_latest_checkpoint(save_dir: PathLike):
    """Return (Path, step) of the highest-numbered step_<N>.pt, or None.

    Ignores final.pt and any non step_<N>.pt file.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return None
    best = None
    for f in save_dir.glob("step_*.pt"):
        m = _CKPT_RE.match(f.name)
        if m is None:
            continue
        n = int(m.group(1))
        if best is None or n > best[1]:
            best = (f, n)
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add resume.py tests/test_resume.py
git commit -m "feat(resume): latest-checkpoint discovery"
```

---

## Task 3: Resume-plan resolution

**Files:**
- Modify: `resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_resume.py
from resume import resolve_resume, ResumePlan


def test_resolve_fresh_when_not_resuming(tmp_path):
    plan = resolve_resume(tmp_path, resume=False, resume_from=None)
    assert plan == ResumePlan(ckpt_path=None, start_step=0, wandb_run_id=None)


def test_resolve_auto_uses_latest_and_sidecar(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    (tmp_path / "step_400000.pt").write_bytes(b"x")
    write_train_state(tmp_path, step=400000, wandb_run_id="run42", latest="step_400000.pt")
    plan = resolve_resume(tmp_path, resume=True, resume_from=None)
    assert plan.ckpt_path == tmp_path / "step_400000.pt"
    assert plan.start_step == 400000
    assert plan.wandb_run_id == "run42"


def test_resolve_auto_errors_when_nothing_to_resume(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path, resume=True, resume_from=None)


def test_resolve_explicit_path(tmp_path):
    ckpt = tmp_path / "step_600000.pt"
    ckpt.write_bytes(b"x")
    write_train_state(tmp_path, step=600000, wandb_run_id="rid", latest="step_600000.pt")
    plan = resolve_resume(tmp_path, resume=False, resume_from=str(ckpt))
    assert plan.ckpt_path == ckpt
    assert plan.start_step == 600000
    assert plan.wandb_run_id == "rid"


def test_resolve_explicit_path_missing_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path, resume=False, resume_from=str(tmp_path / "nope.pt"))


def test_resolve_explicit_path_no_sidecar_parses_step(tmp_path):
    ckpt = tmp_path / "step_300000.pt"
    ckpt.write_bytes(b"x")
    plan = resolve_resume(tmp_path, resume=False, resume_from=str(ckpt))
    assert plan.start_step == 300000
    assert plan.wandb_run_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_resume.py -k resolve -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_resume'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to resume.py
class ResumePlan(NamedTuple):
    ckpt_path: Optional[Path]   # None => fresh run
    start_step: int             # 0 for a fresh run
    wandb_run_id: Optional[str]


def _step_from_ckpt_name(path: PathLike) -> int:
    m = _CKPT_RE.match(Path(path).name)
    return int(m.group(1)) if m else 0


def resolve_resume(save_dir: PathLike, resume: bool,
                   resume_from: Optional[str]) -> ResumePlan:
    """Decide how to start training.

    Priority: explicit resume_from > resume=true (auto-find latest) > fresh.
    Raises FileNotFoundError if a resume is requested but nothing is found.
    """
    if resume_from:
        ckpt = Path(resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt}")
        ts = read_train_state(ckpt.parent) or {}
        step = int(ts["step"]) if ts.get("latest") == ckpt.name \
            else _step_from_ckpt_name(ckpt)
        return ResumePlan(ckpt, step, ts.get("wandb_run_id"))

    if resume:
        latest = find_latest_checkpoint(save_dir)
        if latest is None:
            raise FileNotFoundError(
                f"resume=true but no step_*.pt found in {save_dir}")
        ckpt, step = latest
        ts = read_train_state(save_dir) or {}
        if ts.get("latest") == ckpt.name:
            step = int(ts["step"])
        return ResumePlan(ckpt, step, ts.get("wandb_run_id"))

    return ResumePlan(None, 0, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add resume.py tests/test_resume.py
git commit -m "feat(resume): resume-plan resolution (explicit/auto/fresh)"
```

---

## Task 4: Clobber guard

**Files:**
- Modify: `resume.py`
- Test: `tests/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_resume.py
from resume import assert_no_clobber


def test_clobber_raises_on_existing_checkpoints(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        assert_no_clobber(tmp_path, resume=False, force=False)


def test_clobber_raises_on_existing_final(tmp_path):
    (tmp_path / "final.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        assert_no_clobber(tmp_path, resume=False, force=False)


def test_clobber_allows_when_empty(tmp_path):
    assert_no_clobber(tmp_path, resume=False, force=False)  # no raise


def test_clobber_skipped_when_resume_or_force(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    assert_no_clobber(tmp_path, resume=True, force=False)   # no raise
    assert_no_clobber(tmp_path, resume=False, force=True)   # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_resume.py -k clobber -q`
Expected: FAIL — `ImportError: cannot import name 'assert_no_clobber'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to resume.py
def assert_no_clobber(save_dir: PathLike, resume: bool, force: bool) -> None:
    """Refuse to start a fresh run on top of an existing run's checkpoints."""
    if resume or force:
        return
    save_dir = Path(save_dir)
    existing = list(save_dir.glob("step_*.pt")) + list(save_dir.glob("final.pt"))
    if existing:
        raise FileExistsError(
            f"{save_dir} already holds {len(existing)} checkpoint file(s). "
            f"Pass resume=true to continue, or force=true to overwrite.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: PASS (15 passed).

- [ ] **Step 5: Commit**

```bash
git add resume.py tests/test_resume.py
git commit -m "feat(resume): clobber guard for fresh runs"
```

---

## Task 5: Config keys + deterministic output dir

**Files:**
- Modify: `configs/train.yaml`

- [ ] **Step 1: Add resume config keys**

In `configs/train.yaml`, after the `# Training` block (after line `seed: 42`, before `device: cuda`), add:

```yaml
# Resume (crash-recovery; see PAPER/specs/2026-06-09-training-resume-design.md)
resume: false               # true => auto-find latest step_*.pt in save_dir and continue
resume_from: null           # explicit checkpoint path; takes priority over resume
force: false                # allow a fresh run to overwrite a non-empty save_dir
```

- [ ] **Step 2: Make the per-run output dir deterministic**

In `configs/train.yaml`, change the `hydra.run.dir` line FROM:

```yaml
    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}__${domain}__${agent}__s${seed}
```

TO:

```yaml
    dir: outputs/${oc.select:wandb_run_name,${domain}__${agent}__s${seed}}
```

This makes `save_dir` (`${hydra:runtime.output_dir}/checkpoints`) stable across relaunches — when the launcher passes `wandb_run_name=...`, the dir is that name; otherwise it falls back to the old `${domain}__${agent}__s${seed}` form. Leave the `hydra.sweep` block unchanged.

- [ ] **Step 3: Verify the interpolation resolves**

Run:
```bash
.venv/bin/python -c "
from hydra import compose, initialize
with initialize(version_base='1.3', config_path='configs'):
    cfg = compose(config_name='train', overrides=['wandb_run_name=psm_state_tuned__s0'], return_hydra_config=True)
    print('run.dir =', cfg.hydra.run.dir)
"
```
Expected: prints `run.dir = outputs/psm_state_tuned__s0`. Then run with no override:
```bash
.venv/bin/python -c "
from hydra import compose, initialize
with initialize(version_base='1.3', config_path='configs'):
    cfg = compose(config_name='train', return_hydra_config=True)
    print('run.dir =', cfg.hydra.run.dir)
"
```
Expected: prints a dir of the form `outputs/<domain>__<agent>__s42` (no timestamp). If `oc.select` errors on your OmegaConf version, fall back to adding a top-level `run_dir_name: ${oc.select:wandb_run_name,${domain}__${agent}__s${seed}}` key and using `dir: outputs/${run_dir_name}`.

- [ ] **Step 4: Commit**

```bash
git add configs/train.yaml
git commit -m "feat(resume): resume config keys + deterministic run-name output dir"
```

---

## Task 6: Wire resume into train.py

**Files:**
- Modify: `train.py` (imports near line 33-53; `main` body lines 322-445)

- [ ] **Step 1: Import the helpers**

Add to the import block in `train.py` (e.g. after the `from normalizers import ...` line, near line 53):

```python
from resume import assert_no_clobber, resolve_resume, write_train_state
```

- [ ] **Step 2: Resolve save_dir + resume plan BEFORE wandb.init**

In `main`, the current order is: `set_seed` → `wandb.init` (327-339) → env/buffer/agent → `save_dir = Path(cfg.save_dir)` (373). Move `save_dir` resolution and add the plan resolution to just AFTER `print(OmegaConf.to_yaml(cfg))` (line 325) and BEFORE the `if cfg.use_wandb:` block. Insert:

```python
    save_dir = Path(cfg.save_dir)
    plan = resolve_resume(save_dir, cfg.get("resume", False), cfg.get("resume_from"))
    if plan.ckpt_path is None:
        assert_no_clobber(save_dir, cfg.get("resume", False), cfg.get("force", False))
    save_dir.mkdir(parents=True, exist_ok=True)
    if plan.ckpt_path is not None:
        print(f"[resume] from {plan.ckpt_path} at step {plan.start_step}")
```

Then DELETE the now-duplicate `save_dir = Path(cfg.save_dir)` and its `save_dir.mkdir(...)` at the old location (lines 373-374).

- [ ] **Step 3: Resume the wandb run instead of starting a new one**

Replace the `wandb.init(...)` call (lines 330-339) with a version that reuses the saved id when resuming:

```python
        init_kwargs = dict(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.getcwd(),
            save_code=cfg.wandb_save_code,
        )
        if plan.wandb_run_id:
            init_kwargs["id"] = plan.wandb_run_id
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = None
```

(Note: add the `else: wandb_run_id = None` so the variable always exists. Keep the `run_name = ...` line above unchanged.)

- [ ] **Step 4: Load the checkpoint and re-seed after the agent is built**

Immediately after `agent = make_agent(...)` and its print (line 370-371), add:

```python
    if plan.ckpt_path is not None:
        agent.load_state_dict(torch.load(plan.ckpt_path, map_location=cfg.device))
        set_seed(cfg.seed + plan.start_step)   # avoid replaying identical minibatches
        print(f"[resume] agent state loaded; continuing from step {plan.start_step}")
```

- [ ] **Step 5: Start the loop at start_step and write the sidecar on checkpoint**

Change the training loop header (line 406) FROM `for step in range(0, cfg.num_train_steps + 1):` TO:

```python
    for step in range(plan.start_step, cfg.num_train_steps + 1):
```

And change the checkpoint block (lines 408-411) FROM:

```python
        if step > 0 and step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}.pt"
            torch.save(agent.state_dict(), ckpt)
            print(f"[ckpt] {ckpt}")
```

TO:

```python
        if step > 0 and step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}.pt"
            torch.save(agent.state_dict(), ckpt)
            write_train_state(save_dir, step, wandb_run_id, ckpt.name)
            print(f"[ckpt] {ckpt}")
```

(`wandb_run_id` is in scope from Step 3.)

- [ ] **Step 6: Smoke-import to catch wiring errors**

Run: `.venv/bin/python -c "import train; print('import ok')"`
Expected: `import ok` (no NameError/SyntaxError). This does not run training.

- [ ] **Step 7: Run the full unit suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_resume.py tests/test_build_pixel_cfgs_mlp.py -q`
Expected: PASS (all resume tests + pixel-cfg tests).

- [ ] **Step 8: Commit**

```bash
git add train.py
git commit -m "feat(resume): wire crash-recovery resume into train.py"
```

---

## Task 7: Launcher RESUME passthrough

**Files:**
- Modify: `scripts/run_grid_sweep.sh`

- [ ] **Step 1: Add the RESUME env knob**

In `scripts/run_grid_sweep.sh`, in the env-defaults block (near line 38, beside `DRY_RUN="${DRY_RUN:-0}"`), add:

```bash
RESUME="${RESUME:-0}"            # 1 => append resume=true to every job (relaunch-continue)
```

- [ ] **Step 2: Append resume=true to each command when RESUME=1**

In the command-building loop, change the `cmd="$PY train.py ..."` assignment (lines 50-54) so the final line includes a resume suffix. Replace line 54
`wandb_tags=[grid-$TS] save_eval_videos=false"`
with:

```bash
wandb_tags=[grid-$TS] save_eval_videos=false${RESUME:+$( [[ "$RESUME" == "1" ]] && echo ' resume=true' )}"
```

(If that inline form is awkward, instead add after the `cmd=...` heredoc: `[[ "$RESUME" == "1" ]] && cmd="$cmd resume=true"`.)

- [ ] **Step 3: Verify with DRY_RUN**

Run:
```bash
AGENT=psm_flowbc KNOB_A_NAME=ortho_coef KNOB_A_VALS=1 KNOB_B_NAME=lr_phi KNOB_B_VALS=1e-5 \
SEEDS=0 RESUME=1 DRY_RUN=1 bash scripts/run_grid_sweep.sh | grep -c "resume=true"
```
Expected: `1` (the single job's command contains `resume=true`). Re-run with `RESUME=0` and expect `0`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_grid_sweep.sh
git commit -m "feat(resume): RESUME=1 launcher flag to relaunch-continue a grid"
```

---

## Task 8: PyTorch resume integration smoke (state + pixel)

**Files:**
- Create: `scripts/smoke_resume.sh`

This is a real end-to-end smoke (needs the offline dataset at `DATA_PATH` and a GPU). It is a host smoke script, not a pytest (mirrors existing `scripts/smoke_*.sh`).

- [ ] **Step 1: Write the smoke script**

```bash
#!/usr/bin/env bash
# scripts/smoke_resume.sh — end-to-end crash-recovery resume smoke for train.py.
# Runs a tiny job to its first checkpoint, then resumes and asserts it continues
# to completion at the right step. Covers state by default; pass OBS=pixel for the
# pixel path. Needs the offline dataset at DATA_PATH and a GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
DOMAIN="${DOMAIN:-cube_single}"
AGENT="${AGENT:-fb_flowbc}"
NAME="smoke_resume_$$"
SAVE_DIR="$REPO/outputs/$NAME/checkpoints"

common=( "$PY" train.py agent="$AGENT" domain="$DOMAIN" data_path="$DATA_PATH"
         use_wandb=false eval_every=0 log_every=20 save_every=40
         load_n_episodes=20 wandb_run_name="$NAME" seed=0 )

echo "[smoke] phase 1: fresh run to 60 steps (checkpoints at 40)"
"${common[@]}" num_train_steps=60
test -f "$SAVE_DIR/step_40.pt" || { echo "FAIL: no step_40.pt"; exit 1; }
test -f "$SAVE_DIR/train_state.json" || { echo "FAIL: no sidecar"; exit 1; }
"$PY" - "$SAVE_DIR" <<'PY'
import sys; from resume import read_train_state
ts = read_train_state(sys.argv[1]); assert ts and ts["step"] == 40, ts
print("[smoke] sidecar OK:", ts)
PY

echo "[smoke] phase 2: resume to 120 steps"
"${common[@]}" num_train_steps=120 resume=true
test -f "$SAVE_DIR/step_120.pt" || { echo "FAIL: resume did not reach step_120.pt"; exit 1; }
test -f "$SAVE_DIR/final.pt" || { echo "FAIL: no final.pt after resume"; exit 1; }
echo "[smoke] PASS ($AGENT / $DOMAIN)"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/smoke_resume.sh`

- [ ] **Step 3: Run the state smoke**

Run: `bash scripts/smoke_resume.sh`
Expected: ends with `[smoke] PASS (fb_flowbc / cube_single)`. The phase-2 log should show training starting from `[resume] from .../step_40.pt at step 40` and the loop continuing (not restarting at 0).

- [ ] **Step 4: Run the pixel smoke**

Run: `DOMAIN=visual_cube_single AGENT=fb_flowbc bash scripts/smoke_resume.sh`
(Use whatever pixel domain config exists for FB; if the FB pixel domain has a different name, substitute it. The point is to exercise the pixel obs path through the same resume code.)
Expected: ends with `[smoke] PASS`. If the pixel dataset is not staged, document that and run only when it is.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_resume.sh
git commit -m "test(resume): end-to-end state+pixel resume smoke script"
```

---

## Task 9: GCIQL resume smoke + docs (optional convenience)

**Files:**
- Create: `scripts/smoke_resume_gciql.sh`
- Modify: `run_gciql.py` (optional convenience only)

JAX/GCIQL resume already works (`run_gciql.py:48-52` → vendored `third_party/ogbench/impls/main.py:86-87` `restore_agent`; wandb shim handles `GCIQL_STEP_OFFSET`/`GCIQL_CSV_APPEND`). This task only verifies + documents it.

- [ ] **Step 1: Write the GCIQL resume smoke script**

```bash
#!/usr/bin/env bash
# scripts/smoke_resume_gciql.sh — verify GCIQL restore continues a run.
# Runs a tiny GCIQL job that saves a checkpoint, then restarts with
# restore_path/restore_epoch and asserts it continues past the restore epoch.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAX_PY="${JAX_PY:-/dev/shm/.venv-jax/bin/python}"   # see memory: jax venv is volatile /dev/shm
echo "[smoke] GCIQL resume — see run_gciql.py restore_path/restore_epoch passthrough"
echo "[smoke] phase 1: short run that writes params_<epoch>.pkl"
# Fill in the minimal run_gciql invocation used by scripts/smoke_test_gciql.sh,
# with a small train_steps and a save_interval that produces >=1 checkpoint.
# Then phase 2: re-run with restore_path=<dir> restore_epoch=<epoch> and assert
# the new run's first saved epoch > restore_epoch (GCIQL_STEP_OFFSET applied).
echo "[smoke] (adapt from scripts/smoke_test_gciql.sh; assert epoch offset)"
```

Before finalizing, open `scripts/smoke_test_gciql.sh` and copy its concrete `run_gciql` invocation into phases 1 and 2 (set a small `train_steps`/`save_interval`; phase 2 adds `restore_path=<output_dir> restore_epoch=<N>`). Assert phase 2 produces a checkpoint at epoch `> N`.

- [ ] **Step 2: Run it**

Run: `bash scripts/smoke_resume_gciql.sh`
Expected: phase 2 continues past the restore epoch (a saved epoch greater than the restore epoch appears), confirming params+opt restore and step offset.

- [ ] **Step 3: Document resume in the repo**

Add a short "Resuming runs" note (PyTorch: `resume=true` / `resume_from=<ckpt>`, deterministic run-name dirs, `RESUME=1` launcher flag; JAX: `restore_path`/`restore_epoch`) to the repo's docs location (e.g. a `docs/` note or the relevant README). Keep it to ~15 lines.

- [ ] **Step 4 (OPTIONAL): `resume=true` convenience for run_gciql**

Only if cheap: in `run_gciql.py`, when `cfg.get("resume")` is true and no `restore_path` is set, auto-derive `restore_path` = the run's output dir and `restore_epoch` = the highest `params_*.pkl` epoch found there, then fall through to the existing passthrough (lines 48-52). Add a unit test in `scripts/test_run_gciql.py` mirroring its existing arg-building tests. If this complicates the passthrough, SKIP it — explicit `restore_path`/`restore_epoch` already works.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_resume_gciql.sh run_gciql.py 2>/dev/null || git add scripts/smoke_resume_gciql.sh
git commit -m "test(resume): GCIQL resume smoke + docs"
```

---

## Self-Review

**Spec coverage:**
- Sidecar (design §1) → Task 1. ✓
- Deterministic save_dir + clobber guard (§2) → Tasks 4, 5. ✓
- Resume logic / config keys / loop range / re-seed (§3) → Tasks 3, 5, 6. ✓
- wandb same-run continuation (§4) → Task 6 Step 3. ✓
- Launcher RESUME (§5) → Task 7. ✓
- JAX verify + smoke + docs + optional convenience (§6) → Task 9. ✓
- Testing: unit (Tasks 1-4), integration smoke state+pixel (Task 8), GCIQL smoke (Task 9). ✓
- Out-of-scope items (bit-exact, online buffer, auto-resume) — correctly absent.

**Placeholder scan:** No "TBD/TODO" in code steps. Task 9 Step 1 intentionally references copying the concrete invocation from `scripts/smoke_test_gciql.sh` (the canonical GCIQL command lives there; do not duplicate-guess it) and Step 4 is explicitly optional — both are instructions with a named source, not blind placeholders.

**Type consistency:** `write_train_state(save_dir, step, wandb_run_id, latest)`, `read_train_state(save_dir)→dict|None`, `find_latest_checkpoint(save_dir)→(Path,int)|None`, `resolve_resume(save_dir, resume, resume_from)→ResumePlan(ckpt_path, start_step, wandb_run_id)`, `assert_no_clobber(save_dir, resume, force)` — names/signatures match between definitions and all call sites (train.py Steps 2-5).

**Pre-flight assumption to confirm during execution:** all four agents (FB/PSM/RLDP flowbc) implement `state_dict()`/`load_state_dict()` round-tripping optimizers like `agents/fb/agent.py:561-573`. FB and PSM confirmed (174 MB checkpoints include optimizer moments). Task 8's smoke exercises the agent under `AGENT=...`; run it once per agent family (fb_flowbc, psm_flowbc, rldp_flowbc) to confirm load round-trips before the real 1M launch.
