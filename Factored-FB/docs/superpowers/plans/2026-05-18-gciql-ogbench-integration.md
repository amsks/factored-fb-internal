# GCIQL via vendored OGBench (JAX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run OGBench's GCIQL (JAX) on cube manipulation (state + visual) from inside Factored-FB via a thin Hydra launcher, with the OGBench pipeline executing byte-for-byte upstream and fully isolated from the PyTorch FB track.

**Architecture:** The whole OGBench repo is vendored verbatim under `third_party/ogbench/`. A pure-Python Hydra launcher `run_gciql.py` (no `jax` import) translates Hydra configs into OGBench's exact absl CLI flags and `subprocess`-launches `third_party/ogbench/impls/main.py` inside an isolated `.venv-jax` on a spare NVMe. The PyTorch `.venv` / FB track is untouched.

**Tech Stack:** Python, Hydra/OmegaConf (launcher, in `.venv`), vendored OGBench JAX/Flax (in `.venv-jax`), uv for env management, bash scripts for orchestration.

**Spec:** `docs/superpowers/specs/2026-05-18-gciql-ogbench-integration-design.md`

**Key facts established from the OGBench source (so the launcher is exact):**
- `impls/main.py` absl flags: `--run_group` (str), `--seed` (int), `--env_name` (str), `--save_dir` (str), `--restore_path`, `--restore_epoch`, `--train_steps`, `--log_interval`, `--eval_interval`, `--save_interval`, `--eval_tasks`, `--eval_episodes`, `--eval_temperature`, `--eval_gaussian`, `--video_episodes`, `--video_frame_skip`, `--eval_on_cpu`, and `config_flags --agent` (default `agents/gciql.py`) with `--agent.<key>=<value>` overrides.
- `main.py:50` rewrites `save_dir` to `os.path.join(save_dir, 'OGBench', run_group, exp_name)`.
- `setup_wandb` always calls `wandb.init(mode='online')`; to run without network use env `WANDB_DISABLED=true` (hard off-switch, honored regardless of the explicit mode kwarg). No vendored-code edit.
- Datasets load via `ogbench.make_env_and_datasets(env_name, ...)` with NO dataset-dir flag; default dir is `~/.ogbench/data` (`ogbench/utils.py:10`). Redirect to NVMe by symlinking `~/.ogbench -> /mnt/scratch1/ogbench_data` (no code edit).
- GCIQL `get_config()` defaults (do NOT re-specify; only override deviations): `lr=3e-4`, `batch_size=1024`, `actor_hidden_dims=(512,512,512)`, `value_hidden_dims=(512,512,512)`, `layer_norm=True`, `discount=0.99`, `tau=0.005`, `expectile=0.9`, `actor_loss='ddpgbc'`, `alpha=0.3`, `const_std=True`, `encoder=None`, goal-sampling `(0.2,0.5,0.3)/(0.0,1.0,0.0)`, `gc_negative=True`, `p_aug=0.0`.
- Cube overrides (from `impls/hyperparameters.sh`): state `--agent.alpha=1.0 --eval_episodes=50`; visual `--env_name=visual-cube-single-play-v0 --agent.alpha=1.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5 --train_steps=500000 --eval_on_cpu=0 --eval_episodes=50`.
- OGBench `pyproject.toml` core deps: `mujoco>=3.1.6`, `dm_control>=1.0.20`, `gymnasium[mujoco]`; `[train]` extra: `jax[cuda12]>=0.4.26`, `flax>=0.8.4`, `distrax>=0.1.5`, `ml_collections`, `matplotlib`, `moviepy`, `wandb`. `main.py` also imports `tqdm`.

**Agent-runnable now:** Tasks 1–6 (vendoring, manifests, configs, launcher + test, scripts, Makefile/README/gitignore). **User-gated (needs mounted NVMe + GPU + network):** Task 7 (mount, install, smoke validation) — exact commands + expected output provided; do not mark complete until the user reports results.

---

### Task 1: Vendor the OGBench repository verbatim

**Files:**
- Create: `third_party/ogbench/**` (copy of `/home/adityamohan/git/baselines/ogbench` minus `.git/`)
- Create: `third_party/ogbench/PROVENANCE.md`

- [ ] **Step 1: Copy the repo excluding `.git`**

Run:
```bash
mkdir -p third_party
rsync -a --exclude='.git' /home/adityamohan/git/baselines/ogbench/ third_party/ogbench/
```

- [ ] **Step 2: Verify the copy is complete and `.git` is absent**

Run:
```bash
test ! -e third_party/ogbench/.git && \
test -f third_party/ogbench/impls/main.py && \
test -f third_party/ogbench/impls/agents/gciql.py && \
test -f third_party/ogbench/pyproject.toml && \
test -d third_party/ogbench/ogbench && echo "VENDOR OK"
```
Expected: `VENDOR OK`

- [ ] **Step 3: Write `third_party/ogbench/PROVENANCE.md`**

```markdown
# OGBench — Vendored Copy (do not modify)

- **Upstream:** https://github.com/seohongpark/ogbench.git
- **Pinned commit:** 1d4140997f60c52c6fb0702ec100dc988b18c548 (2026-01-14)
- **Vendored on:** 2026-05-18
- **Source clone:** /home/adityamohan/git/baselines/ogbench

## Rules

This directory is a **byte-for-byte copy** of upstream OGBench. It MUST NOT be
edited. GCIQL must run exactly as upstream so results stay comparable to
OGBench's published numbers.

- All behavioural changes are expressed as OGBench CLI flags by the launcher
  `run_gciql.py` — never as edits here.
- To upgrade: re-vendor at a new pinned commit and update this file.
```

- [ ] **Step 4: Verify provenance commit matches the source clone**

Run:
```bash
git -C /home/adityamohan/git/baselines/ogbench rev-parse HEAD
grep -q 1d4140997f60c52c6fb0702ec100dc988b18c548 third_party/ogbench/PROVENANCE.md && echo "PROVENANCE OK"
```
Expected: the printed SHA equals `1d4140997f60c52c6fb0702ec100dc988b18c548` and `PROVENANCE OK`. (If the SHA differs, update PROVENANCE.md to the printed value before continuing.)

- [ ] **Step 5: Commit**

```bash
git add third_party/ogbench
git commit -m "vendor: OGBench @1d41409 verbatim (no .git) for JAX GCIQL track

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Isolated JAX dependency manifest

**Files:**
- Create: `requirements-jax.txt`

- [ ] **Step 1: Write `requirements-jax.txt`**

```text
# Isolated environment for the vendored OGBench (JAX) GCIQL track.
# Installed into /mnt/scratch1/.venv-jax — NEVER mixed with the PyTorch .venv.
# The `ogbench` env/dataset library is installed from the VENDORED source
# (pinned to the vendored commit), pulling its [train] extra (jax/flax/etc.).
-e ./third_party/ogbench[train]
tqdm
```

- [ ] **Step 2: Verify the editable target resolves**

Run:
```bash
test -f third_party/ogbench/pyproject.toml && \
grep -q 'jax\[cuda12\]' third_party/ogbench/pyproject.toml && echo "JAX REQS OK"
```
Expected: `JAX REQS OK`

- [ ] **Step 3: Commit**

```bash
git add requirements-jax.txt
git commit -m "deps: requirements-jax.txt (vendored ogbench[train], isolated env)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Hydra configs encoding OGBench's exact cube hyperparameters

**Files:**
- Create: `configs/gciql/cube_single_state.yaml`
- Create: `configs/gciql/cube_single_visual.yaml`

- [ ] **Step 1: Write `configs/gciql/cube_single_state.yaml`**

```yaml
# Thin-launcher config. Values map 1:1 to OGBench CLI flags; the OGBench
# pipeline runs verbatim. Only deviations from gciql.py get_config() defaults
# go under agent_overrides (here: alpha=1.0 per impls/hyperparameters.sh).
variant: state
env_name: cube-single-play-v0
agent_file: agents/gciql.py
seed: 0
train_steps: 1000000
log_interval: 5000
eval_interval: 100000
save_interval: 1000000
eval_episodes: 50
eval_on_cpu: 1
run_group: factored-fb-gciql
agent_overrides:
  alpha: 1.0
wandb_mode: online            # online | offline | disabled
jax_python: /mnt/scratch1/.venv-jax/bin/python
output_root: /mnt/scratch1/gciql_outputs
ogbench_home_symlink: ~/.ogbench
```

- [ ] **Step 2: Write `configs/gciql/cube_single_visual.yaml`**

```yaml
# Visual cube: deltas vs state per impls/hyperparameters.sh:837-838.
variant: visual
env_name: visual-cube-single-play-v0
agent_file: agents/gciql.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 0
run_group: factored-fb-gciql
agent_overrides:
  alpha: 1.0
  batch_size: 256
  encoder: impala_small
  p_aug: 0.5
wandb_mode: online
jax_python: /mnt/scratch1/.venv-jax/bin/python
output_root: /mnt/scratch1/gciql_outputs
ogbench_home_symlink: ~/.ogbench
```

- [ ] **Step 3: Verify YAML parses**

Run:
```bash
python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ['configs/gciql/cube_single_state.yaml','configs/gciql/cube_single_visual.yaml']]; print('YAML OK')"
```
Expected: `YAML OK`

- [ ] **Step 4: Commit**

```bash
git add configs/gciql
git commit -m "config: GCIQL Hydra configs with OGBench exact cube HPs (state+visual)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Thin Hydra launcher `run_gciql.py` (TDD the flag builder)

The only logic with correctness risk is translating config → OGBench argv. We
test it first with a plain dict (no Hydra, no jax), matching this repo's
script-based verification convention (cf. `scripts/smoke.py`).

**Files:**
- Create: `scripts/test_run_gciql.py` (assertion script — the failing test)
- Create: `run_gciql.py`

- [ ] **Step 1: Write the failing test `scripts/test_run_gciql.py`**

```python
"""Verification for run_gciql.build_ogbench_argv (no Hydra/jax needed)."""
import sys

from run_gciql import build_ogbench_argv

STATE = dict(
    variant="state", env_name="cube-single-play-v0", agent_file="agents/gciql.py",
    seed=0, train_steps=1000000, log_interval=5000, eval_interval=100000,
    save_interval=1000000, eval_episodes=50, eval_on_cpu=1,
    run_group="factored-fb-gciql", agent_overrides={"alpha": 1.0},
    output_root="/mnt/scratch1/gciql_outputs",
)
argv = build_ogbench_argv(STATE)
assert argv == [
    "--env_name=cube-single-play-v0",
    "--agent=agents/gciql.py",
    "--seed=0",
    "--train_steps=1000000",
    "--log_interval=5000",
    "--eval_interval=100000",
    "--save_interval=1000000",
    "--eval_episodes=50",
    "--eval_on_cpu=1",
    "--run_group=factored-fb-gciql",
    "--save_dir=/mnt/scratch1/gciql_outputs",
    "--agent.alpha=1.0",
], argv

VISUAL = dict(
    variant="visual", env_name="visual-cube-single-play-v0",
    agent_file="agents/gciql.py", seed=3, train_steps=500000, log_interval=5000,
    eval_interval=100000, save_interval=500000, eval_episodes=50, eval_on_cpu=0,
    run_group="factored-fb-gciql",
    agent_overrides={"alpha": 1.0, "batch_size": 256,
                     "encoder": "impala_small", "p_aug": 0.5},
    output_root="/mnt/scratch1/gciql_outputs",
)
v = build_ogbench_argv(VISUAL)
assert "--env_name=visual-cube-single-play-v0" in v
assert "--seed=3" in v
assert "--agent.batch_size=256" in v
assert "--agent.encoder=impala_small" in v
assert "--agent.p_aug=0.5" in v
assert "--agent.alpha=1.0" in v
# agent overrides come last, sorted by key for determinism
assert v.index("--agent.alpha=1.0") < v.index("--agent.batch_size=256")

# Isolation: the launcher module must NOT import jax/flax.
assert "jax" not in sys.modules and "flax" not in sys.modules, sorted(sys.modules)

print("RUN_GCIQL ARGV OK")
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `python scripts/test_run_gciql.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_gciql'`

- [ ] **Step 3: Implement `run_gciql.py`**

```python
"""Thin Hydra launcher for the vendored OGBench (JAX) GCIQL track.

This module is PURE ORCHESTRATION and MUST NOT import jax/flax/the OGBench
training code. It translates a Hydra config into OGBench's exact absl CLI
flags and runs `third_party/ogbench/impls/main.py` in the isolated .venv-jax
via subprocess. The two tracks never share a process or environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
OGBENCH_IMPLS = REPO_ROOT / "third_party" / "ogbench" / "impls"


def build_ogbench_argv(cfg: Mapping[str, Any]) -> list[str]:
    """Translate launcher config -> OGBench main.py absl argv (deterministic)."""
    argv = [
        f"--env_name={cfg['env_name']}",
        f"--agent={cfg['agent_file']}",
        f"--seed={cfg['seed']}",
        f"--train_steps={cfg['train_steps']}",
        f"--log_interval={cfg['log_interval']}",
        f"--eval_interval={cfg['eval_interval']}",
        f"--save_interval={cfg['save_interval']}",
        f"--eval_episodes={cfg['eval_episodes']}",
        f"--eval_on_cpu={cfg['eval_on_cpu']}",
        f"--run_group={cfg['run_group']}",
        f"--save_dir={cfg['output_root']}",
    ]
    overrides = cfg.get("agent_overrides") or {}
    for key in sorted(overrides):
        argv.append(f"--agent.{key}={overrides[key]}")
    return argv


def _preflight(cfg: Mapping[str, Any]) -> Path:
    """Fail fast with actionable messages if the JAX track isn't provisioned."""
    jax_python = Path(str(cfg["jax_python"])).expanduser()
    if not jax_python.exists():
        sys.exit(
            f"[run_gciql] JAX interpreter not found: {jax_python}\n"
            "Mount the NVMe and run `make install-jax` first "
            "(see the spec's Prerequisites section)."
        )
    symlink = Path(str(cfg["ogbench_home_symlink"])).expanduser()
    if not symlink.exists():
        sys.exit(
            f"[run_gciql] {symlink} missing. Create it so OGBench writes "
            "datasets to the NVMe:\n"
            f"  ln -s /mnt/scratch1/ogbench_data {symlink}"
        )
    out_root = Path(str(cfg["output_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    return jax_python


@hydra.main(version_base="1.3", config_path="configs/gciql",
            config_name="cube_single_state")
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    argv = build_ogbench_argv(plain)
    jax_python = _preflight(plain)

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    mode = str(plain.get("wandb_mode", "online"))
    if mode == "disabled":
        env["WANDB_DISABLED"] = "true"
    elif mode == "offline":
        env["WANDB_MODE"] = "offline"

    run_meta = {
        "argv": argv,
        "factored_fb_sha": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "ogbench_pinned": "1d4140997f60c52c6fb0702ec100dc988b18c548",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(str(plain["output_root"])) / "last_run_gciql.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"[run_gciql] {jax_python} main.py {' '.join(argv)}")

    proc = subprocess.run(
        [str(jax_python), "main.py", *argv],
        cwd=str(OGBENCH_IMPLS), env=env,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python scripts/test_run_gciql.py`
Expected: `RUN_GCIQL ARGV OK`

- [ ] **Step 5: Static isolation check (no jax/flax import anywhere in the launcher)**

Run: `! grep -nE '^\s*(import|from)\s+(jax|flax)' run_gciql.py && echo "ISOLATION OK"`
Expected: `ISOLATION OK`

- [ ] **Step 6: Commit**

```bash
git add run_gciql.py scripts/test_run_gciql.py
git commit -m "feat(gciql): thin Hydra launcher subprocessing vendored OGBench (no jax import)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Launch + smoke shell scripts

**Files:**
- Create: `scripts/run_gciql.sh`
- Create: `scripts/smoke_test_gciql.sh`

- [ ] **Step 1: Write `scripts/run_gciql.sh`**

```bash
#!/usr/bin/env bash
# Launch GCIQL (vendored OGBench, JAX) via the thin Hydra launcher.
# Env-var parameterised, mirrors scripts/run_sweep.sh ergonomics.
#   VARIANT=state|visual  SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VARIANT="${VARIANT:-state}"
SEEDS="${SEEDS:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG="cube_single_${VARIANT}"

echo "[gciql] variant=$VARIANT seeds=$SEEDS wandb=$WANDB_MODE config=$CONFIG"
for seed in $SEEDS; do
    echo "[gciql] $(date +%T)  launching seed=$seed"
    python run_gciql.py --config-name "$CONFIG" \
        seed="$seed" wandb_mode="$WANDB_MODE"
    echo "[gciql] $(date +%T)  done seed=$seed"
done
```

- [ ] **Step 2: Write `scripts/smoke_test_gciql.sh`**

```bash
#!/usr/bin/env bash
# Smoke: tiny GCIQL run for state + visual. Verifies the vendored OGBench +
# isolated JAX env + NVMe data path work end-to-end. Requires mounted NVMe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

JAX_PY="${JAX_PY:-/mnt/scratch1/.venv-jax/bin/python}"
echo "[smoke-gciql] JAX device check:"
"$JAX_PY" -c "import jax; print('jax devices:', jax.devices())"

for VARIANT in state visual; do
    echo "[smoke-gciql] === $VARIANT ==="
    python run_gciql.py --config-name "cube_single_${VARIANT}" \
        train_steps=2000 log_interval=1000 eval_interval=1000 \
        save_interval=2000 eval_episodes=2 wandb_mode=disabled

    RUN_DIR=$(ls -1dt /mnt/scratch1/gciql_outputs/OGBench/*/* 2>/dev/null | head -1)
    echo "[smoke-gciql] inspecting $RUN_DIR"
    test -f "$RUN_DIR/eval.csv" || { echo "FAIL: no eval.csv"; exit 1; }
    grep -q "overall_success\|success" "$RUN_DIR/eval.csv" \
        || { echo "FAIL: no success metric logged"; exit 1; }
    ls "$RUN_DIR"/params_*.pkl >/dev/null 2>&1 \
        || { echo "FAIL: no checkpoint written"; exit 1; }
    echo "[smoke-gciql] $VARIANT PASS"
done
echo "[smoke-gciql] ALL PASS"
```

(Note: OGBench `save_agent` writes `params_<epoch>.pkl` under the run dir; the
final eval at `i==1` and at `eval_interval` writes `eval.csv` rows with the
`evaluation/overall_success` column.)

- [ ] **Step 3: Make executable and verify bash syntax**

Run:
```bash
chmod +x scripts/run_gciql.sh scripts/smoke_test_gciql.sh
bash -n scripts/run_gciql.sh && bash -n scripts/smoke_test_gciql.sh && echo "BASH SYNTAX OK"
```
Expected: `BASH SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_gciql.sh scripts/smoke_test_gciql.sh
git commit -m "scripts: GCIQL launch + end-to-end smoke (state+visual)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Makefile targets, README section, .gitignore

**Files:**
- Modify: `Makefile` (add targets)
- Modify: `README.md` (add a section)
- Modify: `.gitignore` (ignore local run metadata)

- [ ] **Step 1: Append targets to `Makefile`**

Add (after the existing `smoke:` target block):
```makefile
install-jax:
	@test -d /mnt/scratch1 || { echo "Mount the NVMe at /mnt/scratch1 first (see spec Prerequisites)"; exit 1; }
	uv venv --python 3.11 /mnt/scratch1/.venv-jax
	uv pip install --python /mnt/scratch1/.venv-jax/bin/python -r requirements-jax.txt
	@test -e ~/.ogbench || ln -s /mnt/scratch1/ogbench_data ~/.ogbench
	@mkdir -p /mnt/scratch1/ogbench_data /mnt/scratch1/gciql_outputs
	@echo "[install-jax] done. Verify: /mnt/scratch1/.venv-jax/bin/python -c 'import jax; print(jax.devices())'"

run-gciql:
	python run_gciql.py --config-name cube_single_state

smoke-gciql:
	bash scripts/smoke_test_gciql.sh
```

- [ ] **Step 2: Verify the Makefile parses and shows the new targets**

Run: `make -n install-jax >/dev/null 2>&1; make help 2>/dev/null | grep -q . ; grep -qE '^(install-jax|run-gciql|smoke-gciql):' Makefile && echo "MAKEFILE OK"`
Expected: `MAKEFILE OK`

- [ ] **Step 3: Add a README section**

Insert before the `## Citation` section in `README.md`:
```markdown
## GCIQL baseline (OGBench, JAX)

For comparing FB against OGBench's GCIQL on cube manipulation. GCIQL runs as a
**verbatim vendored copy of OGBench** (`third_party/ogbench/`, JAX) in a
**separate environment** — it never shares a process or venv with the PyTorch
FB track. A thin Hydra launcher (`run_gciql.py`) translates config into
OGBench's exact CLI flags and subprocesses OGBench's own `impls/main.py`.

Prerequisite (one-time, run in a real SSH terminal — needs sudo):

```bash
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/scratch1 && sudo mount /dev/nvme1n1 /mnt/scratch1
sudo chown "$USER":"$USER" /mnt/scratch1
```

Then:

```bash
make install-jax                       # isolated .venv-jax on the NVMe
make smoke-gciql                       # end-to-end check (state + visual)

# Full runs (OGBench's exact cube hyperparameters):
VARIANT=state  SEEDS="0 1 2" bash scripts/run_gciql.sh
VARIANT=visual SEEDS="0 1 2" bash scripts/run_gciql.sh
```

Outputs/datasets/checkpoints live under `/mnt/scratch1` (instance-store —
wiped on instance stop; rsync/S3 anything worth keeping). FB vs GCIQL is
compared via each method's own success metric (their evaluators differ by
design); align runs in wandb with a shared `run_group`.
```

- [ ] **Step 4: Add `.gitignore` entries**

Append to `.gitignore`:
```
# GCIQL/JAX track local artifacts (live on the NVMe, never committed)
last_run_gciql.json
/mnt/
```

- [ ] **Step 5: Verify README + gitignore edits applied**

Run: `grep -q "GCIQL baseline (OGBench, JAX)" README.md && grep -q "last_run_gciql.json" .gitignore && echo "DOCS OK"`
Expected: `DOCS OK`

- [ ] **Step 6: Commit**

```bash
git add Makefile README.md .gitignore
git commit -m "build/docs: install-jax + run/smoke targets, README GCIQL section

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: NVMe provisioning, env install, smoke validation (USER-GATED)

This task requires a mounted spare NVMe, a GPU, and network egress. The agent
cannot run it (sudo needs a real TTY; root disk is full; JAX needs the GPU).
Provide these to the user and **do not mark complete until the user reports the
expected output**.

- [ ] **Step 1: User mounts the NVMe (real interactive SSH session)**

```bash
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/scratch1
sudo mount /dev/nvme1n1 /mnt/scratch1
sudo chown "$USER":"$USER" /mnt/scratch1
df -hT /mnt/scratch1
```
Expected: `df` shows `/mnt/scratch1` mounted with ~875 GB free.
(Caveat: instance-store — wiped on instance stop/terminate.)

- [ ] **Step 2: Install the isolated JAX env**

Run: `make install-jax`
Expected: `.venv-jax` created on the NVMe; `~/.ogbench` symlink created;
`/mnt/scratch1/{ogbench_data,gciql_outputs}` exist; no errors.

- [ ] **Step 3: Verify JAX sees the GPU**

Run: `/mnt/scratch1/.venv-jax/bin/python -c "import jax; print(jax.devices())"`
Expected: a non-empty device list containing a `cuda`/`gpu` device (not only
`CpuDevice`). If CPU-only, pin a jaxlib build matching the instance CUDA
driver in `requirements-jax.txt` and re-run `make install-jax`.

- [ ] **Step 4: Run the end-to-end smoke**

Run: `make smoke-gciql`
Expected: `[smoke-gciql] ALL PASS` (state + visual each produce `eval.csv`
with a success column and a `params_*.pkl` checkpoint under
`/mnt/scratch1/gciql_outputs/OGBench/factored-fb-gciql/<exp>/`).

- [ ] **Step 5: Validation run (sanity vs OGBench's reported number)**

Run: `VARIANT=state SEEDS="0" WANDB_MODE=offline bash scripts/run_gciql.sh`
Expected: completes 1M steps; final `evaluation/overall_success` in `eval.csv`
is in the same ballpark as OGBench's published GCIQL `cube-single-play-v0`
result (order-of-magnitude agreement; exact match not expected across
seed/hardware). Record the number for the FB-vs-GCIQL comparison.

- [ ] **Step 6: Mark plan complete**

Once the user confirms Steps 1–5 outputs, the integration is validated. Push
any checkpoints worth keeping off the ephemeral NVMe (rsync/S3).

---

## Self-Review

**Spec coverage:**
- Vendor whole OGBench verbatim + PROVENANCE → Task 1 ✓
- Isolated JAX env (vendored `[train]`) → Task 2, Task 6 (`install-jax`), Task 7 ✓
- Thin Hydra launcher, no jax import, subprocess boundary → Task 4 (incl. isolation checks) ✓
- Exact cube hyperparameters (state + visual) → Task 3 (+ argv test in Task 4) ✓
- NVMe prerequisite + ephemerality caveat → Task 6 README, Task 7 ✓
- Launch + smoke scripts → Task 5 ✓
- Comparison caveat documented → Task 6 README ✓
- Datasets to NVMe without code edit (`~/.ogbench` symlink) → Task 4 preflight, Task 6 `install-jax` ✓
- wandb offline/disabled without code edit (`WANDB_DISABLED`/`WANDB_MODE`) → Task 4 ✓
- FB/PyTorch track untouched → no task modifies it; verified by file list ✓

**Placeholder scan:** No TBD/TODO; all code, configs, and commands are concrete.

**Type/name consistency:** `build_ogbench_argv(cfg)` signature and the exact
argv list are identical between Task 4's test (Step 1) and implementation
(Step 3); config keys (`env_name`, `agent_file`, `agent_overrides`,
`output_root`, `jax_python`, `ogbench_home_symlink`, `wandb_mode`) match across
Tasks 3, 4, 5, 6. Output path `/mnt/scratch1/gciql_outputs/OGBench/<run_group>/<exp>`
consistent between the launcher (`--save_dir`), smoke script glob, and Task 7.

No gaps found.
