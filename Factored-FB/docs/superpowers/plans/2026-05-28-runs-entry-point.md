# Per-agent run entry points implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 9 redundant per-agent shell launchers with a clear 8-leaf `scripts/agents/<agent>/<obs>.sh` tree backed by a shared `_common.sh`, and fold `docs/RUNS.md` into the README as the single source of truth.

**Architecture:** A small `_common.sh` exposes two high-level entry points (`_run_fb_sweep`, `_run_jax_sweep`) plus utility helpers (GPU mapping, log dirs, DRY_RUN preview). Each leaf script is ~15-30 lines: sources `_common.sh`, sets per-agent defaults, builds variant flags, and calls the right entry point. Tests are pytest modules that invoke each leaf with `DRY_RUN=1` and assert the printed command tokens.

**Tech Stack:** bash, pytest (existing `.venv` venv), no new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-28-runs-entry-point-design.md`

**Conventions:**
- All commits omit the Co-Authored-By Claude trailer (per repo convention — see `memory/commit-no-claude-coauthor.md`).
- Tests run via `.venv/bin/python -m pytest` (per `memory/factored-fb-test-env.md` — `uv run` is broken).
- New scripts must work without sourcing the JAX venv (the `_common.sh` only orchestrates; the JAX venv is invoked through `run_gciql.py`).

---

## File structure

**New files:**
- `scripts/agents/_common.sh` — shared helpers + two sweep entry points
- `scripts/agents/fb/state.sh`
- `scripts/agents/fb/pixel.sh`
- `scripts/agents/gciql/state.sh`
- `scripts/agents/gciql/pixel.sh`
- `scripts/agents/gcivl/pixel.sh`
- `scripts/agents/crl/state.sh`
- `scripts/agents/crl_flowbc/state.sh`
- `scripts/agents/crl_flowbc/pixel.sh`
- `tests/scripts/__init__.py` (empty, marks the dir as a pytest package)
- `tests/scripts/test_common_sh.py`
- `tests/scripts/test_agent_launchers.py`

**Modified:**
- `README.md` — replace `## Quick start` (lines 77-104) with a matrix-driven section; fold in the entirety of `docs/RUNS.md` (data prerequisites, knobs, monitoring, two-stack context); note that antmaze uses `scripts/run_sweep.sh`.

**Deleted:**
- `scripts/run_gciql.sh`
- `scripts/run_gciql_state.sh`
- `scripts/run_gciql_state_sweep.sh`
- `scripts/run_gciql_visual.sh`
- `scripts/run_state_crl.sh`
- `scripts/run_state_crl_flowbc.sh`
- `scripts/run_fb_reweight.sh`
- `scripts/run_onestep_fb.sh`
- `scripts/run_drq_pixel_sweep.sh`
- `docs/RUNS.md`

---

## Task 1: Shared helper `scripts/agents/_common.sh` + unit tests

**Files:**
- Create: `scripts/agents/_common.sh`
- Create: `tests/scripts/__init__.py` (empty)
- Create: `tests/scripts/test_common_sh.py`

- [ ] **Step 1: Create the empty pytest package marker**

```bash
mkdir -p scripts/agents/fb scripts/agents/gciql scripts/agents/gcivl scripts/agents/crl scripts/agents/crl_flowbc tests/scripts
: > tests/scripts/__init__.py
```

- [ ] **Step 2: Write the failing test for `_common.sh` helpers**

Create `tests/scripts/test_common_sh.py`:

```python
"""Unit tests for scripts/agents/_common.sh helper functions.
Each test sources the script under bash and asserts on the function's stdout/exit code.
"""
import subprocess
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "_common.sh"


def _run(snippet: str, **env_extra) -> subprocess.CompletedProcess:
    """Source _common.sh in a fresh bash, run the snippet, return CompletedProcess."""
    env = {"PATH": "/usr/bin:/bin"}
    env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", f"set -u; source {COMMON}; {snippet}"],
        capture_output=True, text=True, env=env,
    )


def test_common_sh_exists():
    assert COMMON.is_file(), f"missing {COMMON}"


def test_gpu_for_seed_round_robin():
    # 4 seeds (0..3) across 3 GPUs ("0 1 2") -> 0,1,2,0
    out = _run('for s in 0 1 2 3; do _gpu_for_seed "$s" "0 1 2"; done').stdout.split()
    assert out == ["0", "1", "2", "0"], out


def test_gpu_for_seed_single_gpu():
    out = _run('_gpu_for_seed 5 "7"').stdout.strip()
    assert out == "7", out


def test_log_dir_format():
    # _log_dir <run_group> <seed> -> /dev/shm/<run_group>_<TS>/seed_<seed>.log
    out = _run('_log_dir test_group 3').stdout.strip()
    assert out.startswith("/dev/shm/test_group_"), out
    assert out.endswith("/seed_3.log"), out


def test_dry_run_preview_short_circuits():
    # _dry_run_preview echoes "[DRY] <args>" when DRY_RUN=1 and returns 0
    r = _run('_dry_run_preview foo bar baz', DRY_RUN="1")
    assert r.returncode == 0
    assert "[DRY] foo bar baz" in r.stdout


def test_dry_run_preview_no_op_when_unset():
    r = _run('_dry_run_preview foo bar baz')
    assert r.returncode == 0
    assert "[DRY]" not in r.stdout


def test_assert_data_passes_when_dir_exists(tmp_path):
    d = tmp_path / "buf"; d.mkdir()
    r = _run(f'_assert_data {d}')
    assert r.returncode == 0


def test_assert_data_fails_when_missing(tmp_path):
    r = _run(f'_assert_data {tmp_path / "nope"}')
    assert r.returncode != 0
    assert "MISSING dataset" in r.stderr
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_common_sh.py -v
```

Expected: ALL FAIL (no `_common.sh` yet, so `test_common_sh_exists` fails first; the others fail because `source` errors).

- [ ] **Step 4: Implement `_common.sh`**

Create `scripts/agents/_common.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/_common.sh — shared helpers + sweep entry points for scripts/agents/*.
# Sourced (never executed) by each leaf. Provides:
#   _gpu_for_seed <seed> <gpus_str>          -> echoes GPUS[seed % nGPU]
#   _log_dir <run_group> <seed>              -> echoes /dev/shm/<run_group>_<TS>/seed_<seed>.log
#   _dry_run_preview <cmd...>                -> if DRY_RUN=1, echoes "[DRY] <cmd...>" and returns 0
#   _assert_data <path>                      -> exits 1 with a clear message if path missing
#   _setup_dev_shm                            -> mkdir -p /dev/shm/{tmp_fb,wandb_fb,tmp_crl,wandb_crl,factored-fb/runs,gciql_outputs}
#   _print_usage                              -> reads $USAGE and prints to stderr
#   _run_fb_sweep <domain> <run_group> [extra hydra args...]
#                                            -> SEEDS/GPUS/TRAIN_STEPS-driven PyTorch FB sweep (train.py)
#   _run_jax_sweep <config-name> <run_group> [extra args...]
#                                            -> SEEDS/GPUS/TRAIN_STEPS-driven JAX sweep (run_gciql.py)
# Per-leaf env knobs (read by _run_*_sweep):
#   SEEDS (default "0 1 2 3 4"), GPUS (default "0 1"), TRAIN_STEPS, RUN_GROUP, DATA_PATH,
#   STAGGER (default 3 for fb / 2 for jax), DRY_RUN (default 0).
set -uo pipefail
trap '' HUP

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

_gpu_for_seed() {
    # $1=seed, $2=space-separated GPU list
    local seed="$1"
    local -a gpus
    read -ra gpus <<< "$2"
    local n=${#gpus[@]}
    echo "${gpus[$(( seed % n ))]}"
}

_TS_SHARED="${_TS_SHARED:-$(date +%Y%m%d_%H%M%S)}"
_log_dir() {
    # $1=run_group, $2=seed -> /dev/shm/<run_group>_<TS>/seed_<seed>.log
    local group="$1" seed="$2"
    local dir="/dev/shm/${group}_${_TS_SHARED}"
    mkdir -p "$dir"
    echo "${dir}/seed_${seed}.log"
}

_dry_run_preview() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "[DRY] $*"
        return 0
    fi
    return 1   # signals: not a dry run, caller should proceed
}

_assert_data() {
    [[ -d "$1" ]] || { echo "MISSING dataset $1 — run scripts/extract_ogbench.py" >&2; exit 1; }
}

_setup_dev_shm() {
    mkdir -p /dev/shm/tmp_fb /dev/shm/wandb_fb \
             /dev/shm/tmp_crl /dev/shm/wandb_crl \
             /dev/shm/factored-fb/runs /dev/shm/gciql_outputs
}

_print_usage() { printf '%s\n' "${USAGE:-(no usage defined)}" >&2; }

# Parse --help once, before any sweep runs.
_maybe_help() {
    for a in "$@"; do
        [[ "$a" == "--help" || "$a" == "-h" ]] && { _print_usage; exit 0; }
    done
}

# ── PyTorch FB sweep (train.py) ────────────────────────────────────────────
# Usage: _run_fb_sweep <domain> <run_group> [hydra args...]
_run_fb_sweep() {
    local domain="$1" run_group="$2"; shift 2
    local -a extra_args=("$@")

    local seeds="${SEEDS:-0 1 2 3 4}"
    local gpus="${GPUS:-0 1}"
    local steps="${TRAIN_STEPS:-1000000}"
    local data="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
    local stagger="${STAGGER:-3}"
    local wandb_entity="${WANDB_ENTITY:-amsks}"

    _setup_dev_shm
    export TMPDIR="${TMPDIR:-/dev/shm/tmp_fb}"
    export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_fb}"
    _assert_data "$data/cube-single-play-v0/buffer"

    echo "[fb-sweep] domain=$domain group=$run_group steps=$steps seeds=[$seeds] gpus=[$gpus]"
    echo "[fb-sweep] data=$data  extra_args=(${extra_args[*]:-})"

    local -a seed_arr; read -ra seed_arr <<< "$seeds"
    local s gpu log run_dir
    for s in "${seed_arr[@]}"; do
        gpu="$(_gpu_for_seed "$s" "$gpus")"
        log="$(_log_dir "$run_group" "$s")"
        run_dir="/dev/shm/factored-fb/runs/${_TS_SHARED}__${run_group}__s${s}"
        local cmd=("$PYTHON" train.py
            "domain=$domain"
            "num_train_steps=$steps"
            "data_path=$data"
            "seed=$s" device=cuda
            use_wandb=true "wandb_entity=$wandb_entity" "wandb_group=$run_group"
            "wandb_run_name=${run_group}_s${s}"
            hydra.job.chdir=true "hydra.run.dir=$run_dir"
            "${extra_args[@]}")
        echo "[fb-sweep] seed=$s -> GPU $gpu  log=$log"
        if _dry_run_preview "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
        sleep "$stagger"
    done
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        wait
        echo "[fb-sweep] ALL DONE"
    fi
}

# ── JAX OGBench sweep (run_gciql.py) ───────────────────────────────────────
# Usage: _run_jax_sweep <config-name> <run_group> [extra key=val args...]
_run_jax_sweep() {
    local config="$1" run_group="$2"; shift 2
    local -a extra_args=("$@")

    local seeds="${SEEDS:-0 1 2 3 4}"
    local gpus="${GPUS:-0 1}"
    local steps="${TRAIN_STEPS:-1000000}"
    local stagger="${STAGGER:-2}"
    local wandb_mode="${WANDB_MODE:-online}"
    local storage="${STORAGE:-shm}"

    _setup_dev_shm
    export TMPDIR="${TMPDIR:-/dev/shm/tmp_crl}"
    export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_crl}"

    echo "[jax-sweep] config=$config group=$run_group steps=$steps seeds=[$seeds] gpus=[$gpus]"
    echo "[jax-sweep] wandb_mode=$wandb_mode storage=$storage  extra_args=(${extra_args[*]:-})"

    local -a seed_arr; read -ra seed_arr <<< "$seeds"
    local s gpu log
    for s in "${seed_arr[@]}"; do
        gpu="$(_gpu_for_seed "$s" "$gpus")"
        log="$(_log_dir "$run_group" "$s")"
        local cmd=("$PYTHON" run_gciql.py
            --config-name "$config"
            "seed=$s" "train_steps=$steps"
            "storage=$storage" "wandb_mode=$wandb_mode" "run_group=$run_group"
            "hydra.run.dir=/dev/shm/${run_group}_${_TS_SHARED}/hydra_s${s}"
            "${extra_args[@]}")
        echo "[jax-sweep] seed=$s -> GPU $gpu  log=$log"
        if _dry_run_preview "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
        sleep "$stagger"
    done
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        wait
        echo "[jax-sweep] ALL DONE"
    fi
}
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_common_sh.py -v
```

Expected: ALL PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/agents/_common.sh tests/scripts/__init__.py tests/scripts/test_common_sh.py
git commit -m "feat(scripts): shared _common.sh for agent launchers + unit tests"
```

---

## Task 2: `scripts/agents/fb/state.sh`

**Files:**
- Create: `scripts/agents/fb/state.sh`
- Create: `tests/scripts/test_agent_launchers.py` (start of file)

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_agent_launchers.py`:

```python
"""Integration tests: each scripts/agents/<agent>/<obs>.sh leaf with DRY_RUN=1
must exit 0 and print the expected command tokens. No actual jobs are launched."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _dry_run(leaf: str, **env_extra) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "SEEDS": "0", "GPUS": "0"}
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(REPO / "scripts" / "agents" / leaf)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


# ── fb/state.sh ────────────────────────────────────────────────────────────
def test_fb_state_vanilla_dry_run():
    r = _dry_run("fb/state.sh")
    assert r.returncode == 0, r.stderr
    assert "[DRY]" in r.stdout
    assert "train.py" in r.stdout
    assert "domain=cube_single" in r.stdout
    assert "onestep=true" not in r.stdout
    assert "reweight_alpha" not in r.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in r.stdout


def test_fb_state_onestep_mode():
    r = _dry_run("fb/state.sh", MODE="onestep")
    assert r.returncode == 0, r.stderr
    assert "onestep=true" in r.stdout


def test_fb_state_reweight_alpha():
    r = _dry_run("fb/state.sh", REWEIGHT_ALPHA="0.5")
    assert r.returncode == 0, r.stderr
    assert "reweight_alpha=0.5" in r.stdout
    assert "weight_diag=true" in r.stdout
    assert "weight_z=true" in r.stdout
    assert "reweight_density_path=" in r.stdout


def test_fb_state_multi_seed():
    # 3 seeds round-robin across 2 GPUs -> seed 0->GPU 0, seed 1->GPU 1, seed 2->GPU 0
    r = _dry_run("fb/state.sh", SEEDS="0 1 2", GPUS="0 1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("[DRY] CUDA_VISIBLE_DEVICES=") == 3
    assert "seed=0" in r.stdout and "seed=1" in r.stdout and "seed=2" in r.stdout


def test_fb_state_help_exits_zero():
    r = subprocess.run(
        ["bash", str(REPO / "scripts/agents/fb/state.sh"), "--help"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0
    assert "fb/state.sh" in r.stderr or "fb/state.sh" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v
```

Expected: ALL FAIL (no `fb/state.sh` yet).

- [ ] **Step 3: Implement `scripts/agents/fb/state.sh`**

Create `scripts/agents/fb/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/fb/state.sh — FB+FlowBC on cube_single (state).
# Variants via env vars:
#   MODE=vanilla|onestep             (default vanilla)
#   REWEIGHT_ALPHA=<float>           (default unset/0; >0 enables coverage-balanced reweight)
# Universal knobs: SEEDS, GPUS, TRAIN_STEPS, RUN_GROUP, DATA_PATH, DRY_RUN.
USAGE="usage: bash scripts/agents/fb/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'   space-separated seeds
    GPUS='0 1'          space-separated GPU ids (seed i -> GPUS[i % nGPU])
    TRAIN_STEPS=1000000
    RUN_GROUP=fb_state  wandb group + log dir name
    DATA_PATH=/dev/shm/factored-fb/datasets
    MODE=vanilla        vanilla | onestep
    REWEIGHT_ALPHA=     >0 enables coverage-balanced reweight (needs analysis/cube_density.npz)
    DRY_RUN=1           print commands instead of launching"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-fb_state}"
MODE="${MODE:-vanilla}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0}"
ORTHO_COEF="${ORTHO_COEF:-1000}"
LR_B="${LR_B:-1e-4}"

extra=("ortho_coef=$ORTHO_COEF" "lr_b=$LR_B")
[[ "$MODE" == "onestep" ]] && extra+=("onestep=true")
if [[ "$REWEIGHT_ALPHA" != "0" ]]; then
    DENSITY="${DENSITY:-$REPO/analysis/cube_density.npz}"
    extra+=("reweight_alpha=$REWEIGHT_ALPHA"
            "reweight_clip=${REWEIGHT_CLIP:-10.0}"
            "reweight_density_path=$DENSITY"
            "weight_diag=true" "weight_z=true")
fi

_run_fb_sweep cube_single "$RUN_GROUP" "${extra[@]}"
```

- [ ] **Step 4: Make the script executable**

Run: `chmod +x scripts/agents/fb/state.sh`

- [ ] **Step 5: Run tests to confirm they pass**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k fb_state
```

Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/agents/fb/state.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/fb/state.sh launcher + tests (FB+FlowBC, MODE/REWEIGHT_ALPHA knobs)"
```

---

## Task 3: `scripts/agents/fb/pixel.sh`

**Files:**
- Create: `scripts/agents/fb/pixel.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append new tests)

- [ ] **Step 1: Append tests to `tests/scripts/test_agent_launchers.py`**

Append:

```python
# ── fb/pixel.sh ────────────────────────────────────────────────────────────
def test_fb_pixel_vanilla_dry_run():
    r = _dry_run("fb/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "train.py" in r.stdout
    assert "domain=visual_cube_single" in r.stdout
    assert "[DRY]" in r.stdout


def test_fb_pixel_onestep():
    r = _dry_run("fb/pixel.sh", MODE="onestep")
    assert r.returncode == 0
    assert "onestep=true" in r.stdout
    assert "domain=visual_cube_single" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k fb_pixel
```

Expected: 2 FAIL.

- [ ] **Step 3: Implement `scripts/agents/fb/pixel.sh`**

Create `scripts/agents/fb/pixel.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/fb/pixel.sh — FB+FlowBC on visual_cube_single (pixel obs, frame-stack 3).
# OGBench visual default is 500k steps. Variants: MODE=vanilla|onestep.
USAGE="usage: bash scripts/agents/fb/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=fb_pixel
    DATA_PATH=/dev/shm/factored-fb/datasets
    MODE=vanilla                 vanilla | onestep
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-fb_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
MODE="${MODE:-vanilla}"

extra=()
[[ "$MODE" == "onestep" ]] && extra+=("onestep=true")

export TRAIN_STEPS
_run_fb_sweep visual_cube_single "$RUN_GROUP" "${extra[@]}"
```

- [ ] **Step 4: Make executable**

Run: `chmod +x scripts/agents/fb/pixel.sh`

- [ ] **Step 5: Run tests to confirm pass**

Run:
```bash
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k fb_pixel
```

Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/agents/fb/pixel.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/fb/pixel.sh launcher + tests (visual_cube_single, MODE knob)"
```

---

## Task 4: `scripts/agents/gciql/state.sh`

**Files:**
- Create: `scripts/agents/gciql/state.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

Append to `tests/scripts/test_agent_launchers.py`:

```python
# ── gciql/state.sh ─────────────────────────────────────────────────────────
def test_gciql_state_dry_run():
    r = _dry_run("gciql/state.sh")
    assert r.returncode == 0, r.stderr
    assert "run_gciql.py" in r.stdout
    assert "--config-name cube_single_state" in r.stdout
    assert "[DRY]" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gciql_state`
Expected: 1 FAIL.

- [ ] **Step 3: Implement `scripts/agents/gciql/state.sh`**

Create `scripts/agents/gciql/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/gciql/state.sh — GCIQL on cube_single (state).
# OGBench's exact recipe: 1M steps, alpha=1.0 (config defaults).
USAGE="usage: bash scripts/agents/gciql/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=gciql_state
    WANDB_MODE=online            online | offline | disabled
    STORAGE=shm                  shm | nvme
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-gciql_state}"
_run_jax_sweep cube_single_state "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + run tests**

```bash
chmod +x scripts/agents/gciql/state.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gciql_state
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agents/gciql/state.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/gciql/state.sh launcher + test"
```

---

## Task 5: `scripts/agents/gciql/pixel.sh`

**Files:**
- Create: `scripts/agents/gciql/pixel.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

```python
# ── gciql/pixel.sh ─────────────────────────────────────────────────────────
def test_gciql_pixel_default_impala():
    r = _dry_run("gciql/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual" in r.stdout
    assert "cube_single_visual_drq" not in r.stdout


def test_gciql_pixel_drq_encoder():
    r = _dry_run("gciql/pixel.sh", ENCODER="drq")
    assert r.returncode == 0
    assert "--config-name cube_single_visual_drq" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gciql_pixel`
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Create `scripts/agents/gciql/pixel.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/gciql/pixel.sh — GCIQL on visual_cube_single (pixel).
# ENCODER=impala (default OGBench impala_small) | drq (DrQ front-end).
USAGE="usage: bash scripts/agents/gciql/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=gciql_pixel
    ENCODER=impala               impala | drq
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

ENCODER="${ENCODER:-impala}"
case "$ENCODER" in
    impala) CONFIG=cube_single_visual ;;
    drq)    CONFIG=cube_single_visual_drq ;;
    *) echo "ENCODER must be impala|drq (got: $ENCODER)" >&2; exit 1 ;;
esac

RUN_GROUP="${RUN_GROUP:-gciql_pixel_${ENCODER}}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep "$CONFIG" "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + test**

```bash
chmod +x scripts/agents/gciql/pixel.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gciql_pixel
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agents/gciql/pixel.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/gciql/pixel.sh launcher + tests (ENCODER=impala|drq)"
```

---

## Task 6: `scripts/agents/gcivl/pixel.sh`

**Files:**
- Create: `scripts/agents/gcivl/pixel.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

```python
# ── gcivl/pixel.sh ─────────────────────────────────────────────────────────
def test_gcivl_pixel_default_impala():
    r = _dry_run("gcivl/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual_gcivl" in r.stdout
    assert "cube_single_visual_gcivl_drq" not in r.stdout


def test_gcivl_pixel_drq_encoder():
    r = _dry_run("gcivl/pixel.sh", ENCODER="drq")
    assert r.returncode == 0
    assert "--config-name cube_single_visual_gcivl_drq" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gcivl_pixel`
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Create `scripts/agents/gcivl/pixel.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/gcivl/pixel.sh — GCIVL on visual_cube_single (pixel only — no state config).
# ENCODER=impala (default) | drq.
USAGE="usage: bash scripts/agents/gcivl/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000
    RUN_GROUP=gcivl_pixel
    ENCODER=impala               impala | drq
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

ENCODER="${ENCODER:-impala}"
case "$ENCODER" in
    impala) CONFIG=cube_single_visual_gcivl ;;
    drq)    CONFIG=cube_single_visual_gcivl_drq ;;
    *) echo "ENCODER must be impala|drq (got: $ENCODER)" >&2; exit 1 ;;
esac

RUN_GROUP="${RUN_GROUP:-gcivl_pixel_${ENCODER}}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep "$CONFIG" "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + test**

```bash
chmod +x scripts/agents/gcivl/pixel.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k gcivl_pixel
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agents/gcivl/pixel.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/gcivl/pixel.sh launcher + tests (ENCODER=impala|drq)"
```

---

## Task 7: `scripts/agents/crl/state.sh`

**Files:**
- Create: `scripts/agents/crl/state.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

```python
# ── crl/state.sh ───────────────────────────────────────────────────────────
def test_crl_state_dry_run():
    r = _dry_run("crl/state.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_state_crl" in r.stdout
    assert "cube_single_state_crl_flowbc" not in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k "crl_state and not flowbc"`
Expected: 1 FAIL.

- [ ] **Step 3: Implement**

Create `scripts/agents/crl/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/crl/state.sh — plain CRL (contrastive RL, DDPG+BC actor) on cube_single (state).
USAGE="usage: bash scripts/agents/crl/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=crl_state
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-crl_state}"
_run_jax_sweep cube_single_state_crl "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + test**

```bash
chmod +x scripts/agents/crl/state.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k "crl_state and not flowbc"
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agents/crl/state.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/crl/state.sh launcher + test (plain CRL state)"
```

---

## Task 8: `scripts/agents/crl_flowbc/state.sh`

**Files:**
- Create: `scripts/agents/crl_flowbc/state.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

```python
# ── crl_flowbc/state.sh ────────────────────────────────────────────────────
def test_crl_flowbc_state_dry_run():
    r = _dry_run("crl_flowbc/state.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_state_crl_flowbc" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k crl_flowbc_state`
Expected: 1 FAIL.

- [ ] **Step 3: Implement**

Create `scripts/agents/crl_flowbc/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/crl_flowbc/state.sh — CRL + FlowBC actor on cube_single (state).
USAGE="usage: bash scripts/agents/crl_flowbc/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=crl_flowbc_state
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-crl_flowbc_state}"
_run_jax_sweep cube_single_state_crl_flowbc "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + test**

```bash
chmod +x scripts/agents/crl_flowbc/state.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k crl_flowbc_state
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agents/crl_flowbc/state.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/crl_flowbc/state.sh launcher + test"
```

---

## Task 9: `scripts/agents/crl_flowbc/pixel.sh`

**Files:**
- Create: `scripts/agents/crl_flowbc/pixel.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append)

- [ ] **Step 1: Append tests**

```python
# ── crl_flowbc/pixel.sh ────────────────────────────────────────────────────
def test_crl_flowbc_pixel_dry_run():
    r = _dry_run("crl_flowbc/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual_crl_flowbc_drq" in r.stdout
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k crl_flowbc_pixel`
Expected: 1 FAIL.

- [ ] **Step 3: Implement**

Create `scripts/agents/crl_flowbc/pixel.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/crl_flowbc/pixel.sh — CRL + FlowBC actor on visual_cube_single (DrQ pixel front-end).
USAGE="usage: bash scripts/agents/crl_flowbc/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000
    RUN_GROUP=crl_flowbc_pixel
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-crl_flowbc_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep cube_single_visual_crl_flowbc_drq "$RUN_GROUP"
```

- [ ] **Step 4: Make executable + test**

```bash
chmod +x scripts/agents/crl_flowbc/pixel.sh
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k crl_flowbc_pixel
```

Expected: 1 PASS.

- [ ] **Step 5: Run the full launcher test module to confirm nothing broke**

Run: `.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v`
Expected: 14 PASS (all leaves).

- [ ] **Step 6: Commit**

```bash
git add scripts/agents/crl_flowbc/pixel.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/crl_flowbc/pixel.sh launcher + test"
```

---

## Task 10: Rewrite README; delete `docs/RUNS.md`

**Files:**
- Modify: `README.md` (replace `## Quick start` section, lines 77-104 + insert new sections after)
- Delete: `docs/RUNS.md`

- [ ] **Step 1: Read current README so the edit is surgical**

Run: `sed -n '77,104p' README.md`

(Confirms the Quick start block is exactly the 28-line section to replace.)

- [ ] **Step 2: Replace the Quick start section with the new run-matrix section + folded RUNS.md content**

Use `Edit` to replace the entire `## Quick start` block (from `## Quick start` through the line before `## Configuration`) with the block below. The replacement folds in the entire `docs/RUNS.md` content (the file is being deleted).

```markdown
## Running an agent

All cube-single agents live under `scripts/agents/<agent>/<obs>.sh`. Each leaf is a multi-seed sweep with sensible defaults; single-seed runs are `SEEDS=0 GPUS=0 bash ...`.

### Matrix (cube-single)

| Agent | State | Pixel |
|---|---|---|
| **FB+FlowBC** | `bash scripts/agents/fb/state.sh` | `bash scripts/agents/fb/pixel.sh` |
| **GCIQL** | `bash scripts/agents/gciql/state.sh` | `bash scripts/agents/gciql/pixel.sh` |
| **GCIVL** | — *(no config)* | `bash scripts/agents/gcivl/pixel.sh` |
| **CRL** (plain) | `bash scripts/agents/crl/state.sh` | — *(no config)* |
| **CRL+FlowBC** | `bash scripts/agents/crl_flowbc/state.sh` | `bash scripts/agents/crl_flowbc/pixel.sh` |

`--help` on any leaf prints its knobs. Example smoke runs:
```bash
bash scripts/agents/fb/state.sh --help
SEEDS=0 GPUS=0 DRY_RUN=1 bash scripts/agents/gciql/pixel.sh ENCODER=drq
```

### Universal env knobs

| Knob | Default | What |
|---|---|---|
| `SEEDS` | `"0 1 2 3 4"` | Space-separated seeds. Single seed: `SEEDS=0`. |
| `GPUS` | `"0 1"` | Space-separated GPU ids. Seed *i* → `GPUS[i % nGPU]`. |
| `TRAIN_STEPS` | `1000000` state / `500000` pixel | Per-seed training steps. |
| `RUN_GROUP` | `<agent>_<obs>` | wandb group and log dir name. |
| `DATA_PATH` | `/dev/shm/factored-fb/datasets` | FB stack only. JAX stack uses `~/.ogbench/data`. |
| `WANDB_MODE` | `online` (JAX) | `online`, `offline`, `disabled`. JAX stack only. |
| `STORAGE` | `shm` (JAX) | `shm`, `nvme`. JAX stack only. |
| `DRY_RUN` | `0` | `1` prints commands without launching. |

### Per-agent knobs

| Leaf | Knob | Values |
|---|---|---|
| `fb/{state,pixel}.sh` | `MODE` | `vanilla` (default), `onestep` |
| `fb/state.sh` | `REWEIGHT_ALPHA` | `0` (default, off), `>0` enables coverage-balanced reweight (needs `analysis/cube_density.npz`) |
| `gciql/pixel.sh`, `gcivl/pixel.sh` | `ENCODER` | `impala` (default), `drq` |

### Prerequisites

```bash
# JAX/OGBench data (native .npz, present after install): ~/.ogbench/data/<env>.npz
# FB data (per-episode + physics) — extract once from the OGBench .npz:
/dev/shm/.venv-jax/bin/python scripts/extract_ogbench.py \
    --env cube-single-play-v0 --dataset_path ~/.ogbench/data \
    --output_folder /dev/shm/factored-fb/datasets
# Coverage-balanced density (only needed for FB REWEIGHT_ALPHA>0):
.venv/bin/python -m scripts.cube_density \
    --data /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer \
    --out analysis/cube_density.npz
```

### Monitoring

```bash
tmux ls                                       # tmux sessions
tail -f /dev/shm/<run_group>_*/seed_*.log     # per-seed logs (each launch prints the dir)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# wandb: https://wandb.ai/amsks/factored-fb  (filter by run_group)
```

### Two-stack context (why some agents use `.venv` directly and others spawn a JAX subprocess)

- **PyTorch FB** (FB, FB+FlowBC) runs natively under `.venv/bin/python train.py …`.
- **JAX/OGBench** (GCIQL, GCIVL, CRL, CRL+FlowBC) runs via `.venv/bin/python run_gciql.py …`, which Hydra-launches the vendored OGBench `third_party/ogbench/impls/main.py` under `/dev/shm/.venv-jax/bin/python`. Configs live in `configs/gciql/*.yaml`.
- Storage is volatile: durable `/` is full, so everything writes to `/dev/shm`. Copy results out before any reboot.
- wandb defaults to entity `amsks`, project `factored-fb` for both stacks.

### Antmaze

The new `scripts/agents/` tree is cube-only. For antmaze FB sweeps, use the older launcher:
```bash
bash scripts/run_sweep.sh           # local
bash scripts/run_sweep_slurm.sh     # SLURM
```

---
```

(Note: the trailing `---` is the section separator before `## Configuration`.)

- [ ] **Step 3: Delete `docs/RUNS.md`**

Run: `git rm docs/RUNS.md`

- [ ] **Step 4: Verify nothing in the repo references `docs/RUNS.md`**

Run:
```bash
grep -rn "RUNS\.md\|docs/RUNS" . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=wandb --exclude-dir=logs --exclude-dir=checkpoints || echo "no references"
```

Expected: `no references` (or only matches inside the spec/plan docs in `docs/superpowers/`, which are local-only and may keep historical refs).

If a real reference is found (e.g., in another script's comment), update it to point at the README section instead.

- [ ] **Step 5: Render-check the README locally**

Open `README.md` in your editor or run:
```bash
head -130 README.md
```

Confirm the matrix table sits cleanly between `## Installation` (or wherever Quick start used to be) and `## Configuration`.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/RUNS.md
git commit -m "docs(readme): per-agent run matrix; fold RUNS.md content into README"
```

---

## Task 11: Delete the 9 superseded launchers

**Files:**
- Delete: `scripts/run_gciql.sh`, `scripts/run_gciql_state.sh`, `scripts/run_gciql_state_sweep.sh`, `scripts/run_gciql_visual.sh`, `scripts/run_state_crl.sh`, `scripts/run_state_crl_flowbc.sh`, `scripts/run_fb_reweight.sh`, `scripts/run_onestep_fb.sh`, `scripts/run_drq_pixel_sweep.sh`

- [ ] **Step 1: Verify no kept script references the doomed ones**

Run:
```bash
for f in run_gciql.sh run_gciql_state.sh run_gciql_state_sweep.sh run_gciql_visual.sh \
         run_state_crl.sh run_state_crl_flowbc.sh run_fb_reweight.sh \
         run_onestep_fb.sh run_drq_pixel_sweep.sh; do
    hits=$(grep -rln "$f" scripts/ 2>/dev/null | grep -v "^scripts/$f$" || true)
    [[ -n "$hits" ]] && echo "REFERENCED: $f -> $hits"
done
echo "done"
```

Expected: only `done` printed (no `REFERENCED:` lines). If any references appear, update them to point at the equivalent `scripts/agents/<agent>/<obs>.sh` leaf before deleting.

- [ ] **Step 2: Verify campaign scripts still work after refactor**

The campaign orchestrators (`run_crl_flowbc_campaign.sh`, `resume_crl_flowbc_campaign.sh`, `fb_after_crl.sh`) may shell out to old launchers. Check:
```bash
grep -En "run_(gciql|state_crl|fb_reweight|onestep_fb|drq_pixel)" \
    scripts/run_crl_flowbc_campaign.sh scripts/resume_crl_flowbc_campaign.sh scripts/fb_after_crl.sh
```

If any matches: replace each reference with the new leaf path (`scripts/agents/<agent>/<obs>.sh`) using `Edit`, then commit the campaign script update **before** deleting the old launcher.

- [ ] **Step 3: Delete the 9 scripts**

Run:
```bash
git rm scripts/run_gciql.sh scripts/run_gciql_state.sh scripts/run_gciql_state_sweep.sh \
       scripts/run_gciql_visual.sh scripts/run_state_crl.sh scripts/run_state_crl_flowbc.sh \
       scripts/run_fb_reweight.sh scripts/run_onestep_fb.sh scripts/run_drq_pixel_sweep.sh
```

- [ ] **Step 4: Re-run the full test suite to confirm nothing depended on the old scripts**

Run: `.venv/bin/python -m pytest tests/scripts/ -v`
Expected: 14 PASS (8 from `_common.sh` + 8+ from launcher tests; exact count = 8 helper + 14 launcher = 22).

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(scripts): delete 9 superseded per-agent launchers (replaced by scripts/agents/)"
```

---

## Task 12: Acceptance smoke — run all 5 spec dry-runs end-to-end

**Files:** none (verification only).

- [ ] **Step 1: Run the 5 acceptance commands from the spec**

```bash
echo "=== 1. fb/state vanilla ===" && bash scripts/agents/fb/state.sh DRY_RUN=1 2>&1 | tail -5
echo && echo "=== 2. gciql/pixel drq ===" && DRY_RUN=1 ENCODER=drq bash scripts/agents/gciql/pixel.sh 2>&1 | tail -5
echo && echo "=== 3. crl/state ===" && DRY_RUN=1 bash scripts/agents/crl/state.sh 2>&1 | tail -5
echo && echo "=== 4. crl_flowbc/pixel ===" && DRY_RUN=1 bash scripts/agents/crl_flowbc/pixel.sh 2>&1 | tail -5
echo && echo "=== 5. fb/state onestep + reweight ===" && DRY_RUN=1 MODE=onestep REWEIGHT_ALPHA=0.5 bash scripts/agents/fb/state.sh 2>&1 | tail -8
```

Expected: each command exits 0, every block contains `[DRY] CUDA_VISIBLE_DEVICES=` lines.

Wait — `DRY_RUN=1` as a positional arg in command 1 won't work with bash; the script reads env vars, not CLI args. Update step 1 above to use proper env-var prefixing:

```bash
echo "=== 1. fb/state vanilla ===" && DRY_RUN=1 bash scripts/agents/fb/state.sh 2>&1 | tail -5
echo && echo "=== 2. gciql/pixel drq ===" && DRY_RUN=1 ENCODER=drq bash scripts/agents/gciql/pixel.sh 2>&1 | tail -5
echo && echo "=== 3. crl/state ===" && DRY_RUN=1 bash scripts/agents/crl/state.sh 2>&1 | tail -5
echo && echo "=== 4. crl_flowbc/pixel ===" && DRY_RUN=1 bash scripts/agents/crl_flowbc/pixel.sh 2>&1 | tail -5
echo && echo "=== 5. fb/state onestep + reweight ===" && DRY_RUN=1 MODE=onestep REWEIGHT_ALPHA=0.5 bash scripts/agents/fb/state.sh 2>&1 | tail -8
```

- [ ] **Step 2: Verify `--help` on each leaf**

```bash
for f in scripts/agents/fb/state.sh scripts/agents/fb/pixel.sh \
         scripts/agents/gciql/state.sh scripts/agents/gciql/pixel.sh \
         scripts/agents/gcivl/pixel.sh \
         scripts/agents/crl/state.sh \
         scripts/agents/crl_flowbc/state.sh scripts/agents/crl_flowbc/pixel.sh; do
    echo "=== $f --help ==="
    bash "$f" --help 2>&1 | head -3
done
```

Expected: each prints a usage block starting with `usage: bash scripts/agents/...`.

- [ ] **Step 3: Final sanity scan — repo has no broken references**

```bash
grep -rn "run_gciql\.sh\|run_state_crl\|run_fb_reweight\|run_onestep_fb\|run_drq_pixel\|RUNS\.md" \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=docs/superpowers \
    --exclude-dir=wandb --exclude-dir=logs --exclude-dir=checkpoints --exclude-dir=__pycache__ \
    . || echo "no stale references"
```

Expected: `no stale references` (or only matches in the spec/plan files under `docs/superpowers/`, which are local-only).

- [ ] **Step 4: If everything is green, no commit needed (verification only)**

If a real issue surfaces (e.g., a missed reference), fix it with `Edit`, re-run step 3, and commit:
```bash
git add <fixed file>
git commit -m "chore: fix stale launcher reference after agent-tree refactor"
```

---

## Self-Review

**1. Spec coverage:** Every spec item is covered:
- 8 leaves under `scripts/agents/` → Tasks 2-9 (one each).
- `_common.sh` shared helpers → Task 1.
- Per-leaf contract (defaults, env knobs, `--help`, single-seed via `SEEDS=0 GPUS=0`) → tests in Tasks 2-9 enforce all of these.
- README rewrite + RUNS.md fold + deletion → Task 10.
- 9 deletions → Task 11.
- Acceptance criteria (5 dry-runs, `--help`, no stale refs) → Task 12.
- Antmaze kept-and-noted → Task 10 README section.
- Campaign script reference check → Task 11 step 2.

**2. Placeholder scan:** No "TBD", "TODO", "similar to". Every code block is the actual content.

**3. Type consistency:** Helper function names used identically across tasks (`_run_fb_sweep`, `_run_jax_sweep`, `_gpu_for_seed`, `_log_dir`, `_dry_run_preview`, `_assert_data`, `_setup_dev_shm`, `_maybe_help`). Env knob names consistent across leaves (`SEEDS`, `GPUS`, `MODE`, `REWEIGHT_ALPHA`, `ENCODER`, `RUN_GROUP`, `TRAIN_STEPS`, `DRY_RUN`).

No issues found in self-review.
