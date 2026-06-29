# PSM & RLDP hyperparameter sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give PSM and RLDP the same state-modality hyperparameter tuning FB got, by making PSM's `ortho_coef` a live knob and adding a generic grid-sweep launcher + selection summarizer.

**Architecture:** One-line guarded change wires PSM's existing `ortho_coef` config into the live φ-orthonormality term (default 1.0 ⇒ bit-exact to reference). A method-parameterized bash launcher (`run_grid_sweep.sh`) round-robins a 2-knob grid across GPUs. A log-reading summarizer aggregates `eval/success` per grid point to pick the winner per method.

**Tech Stack:** PyTorch (agents), Hydra (`train.py` overrides), bash (launcher), Python+pytest (summarizer + tests). Repo root `/home/adityamohan/git/Factored-FB`; venv `.venv` (use `.venv/bin/python`, NOT `uv run`).

---

### Task 1: Make PSM `ortho_coef` a live knob

**Files:**
- Modify: `agents/psm/agent.py` (the `_update_psm` ortho combine, currently `psm_loss = psm_loss + 1 * orth_loss`)
- Modify: `agents/psm/agent.py` (header docstring — note the intentional deviation)
- Test: `tests/test_psm_ortho_coef_live.py` (new; does NOT require the baselines reference, so it never skips)

- [ ] **Step 1: Write the failing test**

Create `tests/test_psm_ortho_coef_live.py`. It reuses the float64 agent harness from the equivalence test:

```python
"""tests/test_psm_ortho_coef_live.py — proves PSM's ortho_coef actually scales the
phi orthonormality term in _update_psm (was hardcoded `1 * orth_loss`; made
configurable for the ablation sweep). Self-contained: no baselines reference needed.
"""
import torch

from tests.test_psm_reference_equivalence import _build_agent, _fixed_inputs, B, LR, BETAS, EPS, WD


def _run_update(coef):
    torch.manual_seed(0)              # identical weight init across calls
    agent = _build_agent()
    agent.ortho_coef = coef
    agent.optim_phi = torch.optim.Adam(agent.model.phi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_psm_psi = torch.optim.Adam(agent.model.psm_psi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    obs, action, next_obs, goal, next_action, z_psm, z_sf, discount = _fixed_inputs()
    return agent._update_psm(obs, action, discount, next_obs,
                             torch.arange(B), goal, z_psm, next_action=next_action)


def test_ortho_coef_scales_psm_loss():
    m1 = _run_update(1.0)
    m10 = _run_update(10.0)
    # same weights+inputs => the orthonormality term itself is identical
    assert torch.equal(m1["orth_loss"], m10["orth_loss"])
    # total psm_loss must reflect the coefficient on the ortho term
    assert torch.allclose(m1["psm_loss"],  m1["psm_diag"]  + m1["psm_offdiag"]  + 1.0  * m1["orth_loss"])
    assert torch.allclose(m10["psm_loss"], m10["psm_diag"] + m10["psm_offdiag"] + 10.0 * m10["orth_loss"])
    # and the knob actually changes the loss (fails while hardcoded to 1)
    assert not torch.equal(m1["psm_loss"], m10["psm_loss"])
```

If `BETAS`/`EPS` are not module-level constants in `tests/test_psm_reference_equivalence.py`, check that file's top and import the actual names it defines (it constructs Adam with `betas=BETAS, eps=EPS` in `test_psm_update_bit_exact`, so they exist). If they are inline literals there, replace the imports with `BETAS=(0.9, 0.999); EPS=1e-8` in this test.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_psm_ortho_coef_live.py -q`
Expected: FAIL — `test_ortho_coef_scales_psm_loss` fails the `allclose(... + 10.0 * orth_loss)` and the `not torch.equal` asserts, because the loss currently hardcodes `1 * orth_loss`.

- [ ] **Step 3: Make `ortho_coef` live**

In `agents/psm/agent.py`, inside `_update_psm`, change:

```python
            psm_loss = psm_loss + 1 * orth_loss
```
to:
```python
            psm_loss = psm_loss + self.ortho_coef * orth_loss
```

Leave the SF-branch line `sf_loss = sf_loss + 0 * self.ortho_coef * orth_loss` UNCHANGED (faithful no-op; φ is detached there).

- [ ] **Step 4: Update the header docstring**

In `agents/psm/agent.py`, the module docstring currently says the `0*ortho_coef` SF term etc. are "preserved exactly." Append a sentence:

```
NOTE: _update_psm's phi-orthonormality term is `self.ortho_coef * orth_loss`
(default 1.0 == the reference's hardcoded `1 *`, so the bit-exact equivalence test
still passes). ortho_coef is an intentional ablation knob BEYOND the paper, which
fixes the orthonormality weight at 1. See PAPER/specs/2026-06-08-psm-rldp-
hyperparameter-sweep.md.
```

- [ ] **Step 5: Run the new test + the equivalence test (must both pass)**

Run: `.venv/bin/python -m pytest tests/test_psm_ortho_coef_live.py tests/test_psm_reference_equivalence.py -q`
Expected: PASS. The equivalence test still passes because at the default `ortho_coef=1.0`, `1.0 * orth_loss` is bit-identical to `1 * orth_loss`. (If `baselines/PSM` is not importable, the equivalence tests `skip` — that is acceptable; the new test must still PASS.)

- [ ] **Step 6: Commit**

```bash
git add agents/psm/agent.py tests/test_psm_ortho_coef_live.py
git commit -m "feat(psm): make ortho_coef a live knob in _update_psm (default 1.0, bit-exact)"
```

---

### Task 2: Generic grid-sweep launcher

**Files:**
- Create: `scripts/run_grid_sweep.sh` (method-parameterized 2-knob grid; reuses the GPU-slot scheduler pattern from `scripts/run_sweep.sh`)

- [ ] **Step 1: Write the launcher**

Create `scripts/run_grid_sweep.sh`:

```bash
#!/usr/bin/env bash
# scripts/run_grid_sweep.sh — generic 2-knob grid sweep for ANY train.py agent.
# Generalizes run_sweep.sh (which hardcodes FB's ortho_coef/lr_b) to arbitrary
# hydra knob names, so PSM (ortho_coef x lr_phi) and RLDP (ortho_coef x lr_b)
# reuse one scheduler. Round-robins jobs across GPUs; run inside tmux.
#
# Env knobs:
#   AGENT=psm_flowbc | rldp_flowbc | fb_flowbc   (hydra `agent=`)
#   DOMAIN=cube_single
#   KNOB_A_NAME=ortho_coef   KNOB_A_VALS="0.1 1 10"
#   KNOB_B_NAME=lr_phi|lr_b  KNOB_B_VALS="1e-4 1e-5"
#   SEEDS="0 1 2"            TRAIN_STEPS=500000
#   N_GPUS=8  JOBS_PER_GPU=3
#   RUN_GROUP_BASE=psm_state_sweep
#   DATA_PATH=/dev/shm/factored-fb/datasets
#   WANDB_ENTITY=amsks
#   DRY_RUN=1                (print commands, launch nothing)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/grid_$TS"

AGENT="${AGENT:?set AGENT}"
DOMAIN="${DOMAIN:-cube_single}"
KNOB_A_NAME="${KNOB_A_NAME:?set KNOB_A_NAME}"; KNOB_A_VALS="${KNOB_A_VALS:?set KNOB_A_VALS}"
KNOB_B_NAME="${KNOB_B_NAME:?set KNOB_B_NAME}"; KNOB_B_VALS="${KNOB_B_VALS:?set KNOB_B_VALS}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
N_GPUS="${N_GPUS:-8}"; JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
N_PAR=$(( N_GPUS * JOBS_PER_GPU ))
RUN_GROUP_BASE="${RUN_GROUP_BASE:-${AGENT}_${DOMAIN}_sweep}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
WANDB_ENTITY="${WANDB_ENTITY:-amsks}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"
_tag() { echo "${1//./p}"; }   # 1e-4 -> 1e-4 ; 0.1 -> 0p1 (filesystem-safe)

declare -a CMDS LOGS NAMES
for a in $KNOB_A_VALS; do
  for b in $KNOB_B_VALS; do
    for s in $SEEDS; do
      grp="${RUN_GROUP_BASE}__${KNOB_A_NAME}$(_tag "$a")__${KNOB_B_NAME}$(_tag "$b")"
      name="${grp}__s${s}"
      log="$LOG_DIR/${name}.log"
      cmd="python train.py agent=$AGENT domain=$DOMAIN \
$KNOB_A_NAME=$a $KNOB_B_NAME=$b seed=$s \
num_train_steps=$TRAIN_STEPS data_path=$DATA_PATH \
use_wandb=true wandb_entity=$WANDB_ENTITY wandb_group=$grp wandb_run_name=$name \
wandb_tags=[grid-$TS] save_eval_videos=false"
      CMDS+=("$cmd"); LOGS+=("$log"); NAMES+=("$name")
    done
  done
done

N=${#CMDS[@]}
echo "[grid] AGENT=$AGENT  $N jobs  grid ${KNOB_A_NAME}{$KNOB_A_VALS} x ${KNOB_B_NAME}{$KNOB_B_VALS} x seeds{$SEEDS}"
echo "[grid] $N_GPUS GPUs x $JOBS_PER_GPU/GPU = $N_PAR parallel  steps=$TRAIN_STEPS  logs -> $LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  for i in "${!CMDS[@]}"; do
    gpu=$(( i % N_GPUS ))
    echo "GPU$gpu  ${NAMES[$i]}"
    echo "   CUDA_VISIBLE_DEVICES=$gpu ${CMDS[$i]}"
  done
  exit 0
fi

declare -a SLOT_PIDS=(); for ((s=0;s<N_PAR;s++)); do SLOT_PIDS+=(""); done
find_free_slot() {
  while true; do
    for ((s=0;s<N_PAR;s++)); do
      pid="${SLOT_PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then echo "$s"; return; fi
    done
    sleep 3
  done
}
for i in "${!CMDS[@]}"; do
  slot=$(find_free_slot); gpu=$(( slot % N_GPUS ))
  echo "[grid] $(date +%T) launch ${NAMES[$i]} on GPU$gpu (slot $slot)"
  ( cd "$REPO"; CUDA_VISIBLE_DEVICES="$gpu" ${CMDS[$i]} > "${LOGS[$i]}" 2>&1; \
    echo "[grid] $(date +%T) DONE ${NAMES[$i]} rc=$?" ) &
  SLOT_PIDS[$slot]="$!"
done
wait
echo "[grid] all done -> $LOG_DIR"
```

- [ ] **Step 2: Dry-run the PSM grid and verify the command list**

Run:
```bash
DRY_RUN=1 AGENT=psm_flowbc DOMAIN=cube_single \
  KNOB_A_NAME=ortho_coef KNOB_A_VALS="0.1 1 10" \
  KNOB_B_NAME=lr_phi     KNOB_B_VALS="1e-4 1e-5" \
  SEEDS="0 1 2" RUN_GROUP_BASE=psm_state_sweep \
  bash scripts/run_grid_sweep.sh | head -40
```
Expected: header `AGENT=psm_flowbc  18 jobs`, and per-line commands containing `agent=psm_flowbc`, `ortho_coef=<v>`, `lr_phi=<v>`, `seed=<s>`, `num_train_steps=500000`, distinct `wandb_group`/`wandb_run_name` per (ortho,lr) and round-robined `CUDA_VISIBLE_DEVICES`. Confirm there are exactly 18 job lines (3×2×3).

- [ ] **Step 3: Dry-run the RLDP grid**

Run:
```bash
DRY_RUN=1 AGENT=rldp_flowbc DOMAIN=cube_single \
  KNOB_A_NAME=ortho_coef KNOB_A_VALS="0.001 0.01 0.1" \
  KNOB_B_NAME=lr_b       KNOB_B_VALS="1e-4 1e-5" \
  SEEDS="0 1 2" RUN_GROUP_BASE=rldp_state_sweep \
  bash scripts/run_grid_sweep.sh | head -40
```
Expected: `AGENT=rldp_flowbc  18 jobs`; commands contain `agent=rldp_flowbc`, `ortho_coef=<v>`, `lr_b=<v>`. Exactly 18 job lines.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_grid_sweep.sh
git commit -m "feat(sweep): generic 2-knob grid-sweep launcher (method-parameterized)"
```

---

### Task 3: Sweep summarizer + winner selection

**Files:**
- Create: `scripts/sweep_lib.py` (pure helpers: parse final eval/success from a run log; aggregate over seeds; pick winner)
- Create: `scripts/summarize_sweep.py` (driver: scan a `logs/grid_*` dir → table + winner; IO, TDD-exempt)
- Test: `tests/test_sweep_lib.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sweep_lib.py`:

```python
import numpy as np
import pytest

from scripts.sweep_lib import parse_final_success, aggregate_grid, pick_winner


def test_parse_final_success_takes_last(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[step 0] eval/success=0.0000\n"
        "noise line\n"
        "[step 250000] eval/success=0.3000\n"
        "[step 500000] eval/success=0.4200\n"
    )
    assert parse_final_success(str(log)) == pytest.approx(0.42)


def test_parse_final_success_missing_returns_none(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("no eval here\n")
    assert parse_final_success(str(log)) is None


def test_aggregate_grid_means_over_seeds():
    # (ortho, lr) -> list of per-seed final successes
    rows = {("1", "1e-4"): [0.4, 0.6, 0.5], ("10", "1e-4"): [0.0, 0.1, 0.05]}
    agg = aggregate_grid(rows)
    assert agg[("1", "1e-4")]["mean"] == pytest.approx(0.5)
    assert agg[("10", "1e-4")]["mean"] == pytest.approx(0.05)
    assert agg[("1", "1e-4")]["n"] == 3


def test_pick_winner_is_highest_mean():
    agg = {("1", "1e-4"): {"mean": 0.5, "std": 0.08, "n": 3},
           ("10", "1e-4"): {"mean": 0.05, "std": 0.04, "n": 3}}
    assert pick_winner(agg) == ("1", "1e-4")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sweep_lib.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.sweep_lib'`.

- [ ] **Step 3: Implement the pure helpers**

Create `scripts/sweep_lib.py`:

```python
"""scripts/sweep_lib.py — pure helpers for grid-sweep selection (unit-tested)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_SUCC = re.compile(r"eval/success[=:\s]+([0-9.]+)")


def parse_final_success(log_path: str) -> Optional[float]:
    """Last eval/success value in a run log, or None if absent."""
    last = None
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = _SUCC.search(line)
                if m:
                    last = float(m.group(1))
    except FileNotFoundError:
        return None
    return last


def aggregate_grid(rows: Dict[Tuple[str, str], List[float]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """(knob_a, knob_b) -> {mean, std, n} over per-seed final successes."""
    out = {}
    for key, vals in rows.items():
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[key] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    return out


def pick_winner(agg: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[str, str]:
    """Grid point with the highest mean success (NaN means treated as -inf)."""
    def score(kv):
        m = kv[1]["mean"]
        return -np.inf if np.isnan(m) else m
    return max(agg.items(), key=score)[0]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sweep_lib.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Write the driver (IO, TDD-exempt)**

Create `scripts/summarize_sweep.py`:

```python
"""scripts/summarize_sweep.py — aggregate a grid_* log dir into a winner table.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/summarize_sweep.py \
      logs/grid_<TS> --knob-a ortho_coef --knob-b lr_phi --out analysis/sweep/psm_state
Run names are <base>__<knobA><val>__<knobB><val>__s<seed>.log (filesystem tag: '.'->'p').
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.sweep_lib import aggregate_grid, parse_final_success, pick_winner


def _untag(t: str) -> str:
    return t.replace("p", ".") if re.fullmatch(r"\d+p\d+", t) else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--knob-a", required=True)
    ap.add_argument("--knob-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pat = re.compile(rf"{args.knob_a}([0-9pe.-]+)__{args.knob_b}([0-9pe.-]+)__s(\d+)\.log$")
    rows = defaultdict(list)
    for log in sorted(Path(args.log_dir).glob("*.log")):
        m = pat.search(log.name)
        if not m:
            continue
        a, b, _seed = _untag(m.group(1)), _untag(m.group(2)), m.group(3)
        rows[(a, b)].append(parse_final_success(str(log)))

    agg = aggregate_grid(rows)
    winner = pick_winner(agg) if agg else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Sweep: {args.knob_a} x {args.knob_b}  ({args.log_dir})\n",
             f"| {args.knob_a} | {args.knob_b} | mean success | std | n |",
             "|---|---|---|---|---|"]
    for (a, b), s in sorted(agg.items(), key=lambda kv: -(-1 if kv[1]['mean']!=kv[1]['mean'] else kv[1]['mean'])):
        star = " **<-- winner**" if (a, b) == winner else ""
        lines.append(f"| {a} | {b} | {s['mean']:.3f} | {s['std']:.3f} | {s['n']} |{star}")
    lines.append(f"\n**Winner:** {args.knob_a}={winner[0]}, {args.knob_b}={winner[1]}" if winner else "\n(no runs found)")
    (out.with_suffix(".md")).write_text("\n".join(lines))
    (out.with_suffix(".json")).write_text(json.dumps(
        {f"{a}|{b}": s for (a, b), s in agg.items()} | {"winner": winner}, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out}.md / {out}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_lib.py scripts/summarize_sweep.py tests/test_sweep_lib.py
git commit -m "feat(sweep): log-reading grid summarizer + winner selection (tested helpers)"
```

---

### Task 4: Launch the sweeps and select winners (operational)

This task runs the long GPU campaign; it produces no committed code. Do it after Tasks 1–3 are merged and on the box with 8 free GPUs.

- [ ] **Step 1: Verify GPUs are free**

Run: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
Expected: all 8 GPUs ~0% / low memory before launching.

- [ ] **Step 2: Launch the PSM grid in tmux** ([[always-run-in-tmux]])

```bash
tmux new-session -d -s sweep_psm "cd /home/adityamohan/git/Factored-FB && \
AGENT=psm_flowbc DOMAIN=cube_single \
KNOB_A_NAME=ortho_coef KNOB_A_VALS='0.1 1 10' \
KNOB_B_NAME=lr_phi     KNOB_B_VALS='1e-4 1e-5' \
SEEDS='0 1 2' TRAIN_STEPS=500000 N_GPUS=8 JOBS_PER_GPU=3 \
RUN_GROUP_BASE=psm_state_sweep \
bash scripts/run_grid_sweep.sh 2>&1 | tee logs/sweep_psm_master.log"
```

- [ ] **Step 3: Launch the RLDP grid in tmux** (after PSM frees slots, or share GPUs if memory allows — state models are small)

```bash
tmux new-session -d -s sweep_rldp "cd /home/adityamohan/git/Factored-FB && \
AGENT=rldp_flowbc DOMAIN=cube_single \
KNOB_A_NAME=ortho_coef KNOB_A_VALS='0.001 0.01 0.1' \
KNOB_B_NAME=lr_b       KNOB_B_VALS='1e-4 1e-5' \
SEEDS='0 1 2' TRAIN_STEPS=500000 N_GPUS=8 JOBS_PER_GPU=3 \
RUN_GROUP_BASE=rldp_state_sweep \
bash scripts/run_grid_sweep.sh 2>&1 | tee logs/sweep_rldp_master.log"
```

Record both sessions in the [[tmux-sessions-registry]] memory.

- [ ] **Step 4: Summarize + pick winners** (when all `*.log` show `[step 500000]`)

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_sweep.py logs/grid_<PSM_TS> \
  --knob-a ortho_coef --knob-b lr_phi --out analysis/sweep/psm_state
PYTHONPATH=. .venv/bin/python scripts/summarize_sweep.py logs/grid_<RLDP_TS> \
  --knob-a ortho_coef --knob-b lr_b   --out analysis/sweep/rldp_state
```
Expected: a table per method with the winning `(ortho_coef, lr_*)` flagged.

- [ ] **Step 5: Launch winner runs at 1M × 10 seeds**

For each method, re-run the winning combo at full budget (mirrors `fb_state_plain`):
```bash
tmux new-session -d -s psm_tuned "cd /home/adityamohan/git/Factored-FB && \
AGENT=psm_flowbc DOMAIN=cube_single \
KNOB_A_NAME=ortho_coef KNOB_A_VALS='<winner_ortho>' \
KNOB_B_NAME=lr_phi     KNOB_B_VALS='<winner_lr>' \
SEEDS='0 1 2 3 4 5 6 7 8 9' TRAIN_STEPS=1000000 N_GPUS=8 JOBS_PER_GPU=2 \
RUN_GROUP_BASE=psm_state_tuned \
bash scripts/run_grid_sweep.sh 2>&1 | tee logs/psm_tuned_master.log"
```
(analogously `rldp_state_tuned` with its winner). These `*_tuned` 10-seed groups are the numbers reported in `PAPER/`.

---

## Notes

- `PAPER/` (incl. these spec/plan files) is gitignored by repo convention — do NOT commit the plan/spec, only the code (Tasks 1–3).
- wandb entity is `amsks` ([[wandb-entity-amsks]]); never the team default.
- Use `.venv/bin/python -m pytest`, not `uv run` ([[factored-fb-test-env]]).
