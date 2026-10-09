# GCIQL Analysis Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile GCIQL on the transferable probes (T1 value landscape, T4 coverage, phase funnel), aggregate 10 seeds, build an FB-vs-GCIQL comparison, and fold it into the paper preserving the user's hand-restructured layout.

**Architecture:** Extract torch-free shared math into `evals/_profile_core.py` so a JAX-venv GCIQL profiler reuses byte-identical metrics. Add `scripts/gciql_profile.py` (runs in the isolated JAX venv: loads `params_*.pkl`, rolls out the goal-conditioned policy, emits FB-schema-aligned parquets), `scripts/gciql_profile_aggregate.py` (10-seed aggregate + FB-vs-GCIQL comparison), and re-template the paper generator to the user's structure + a comparison subsection.

**Tech Stack:** Python, numpy, pandas (torch-free core); JAX/flax + vendored OGBench (`third_party/ogbench/impls`) for GCIQL inference only; pytest.

Spec: `docs/superpowers/specs/2026-05-18-gciql-analysis-extension-design.md`

---

## Verified codebase facts

- `evals/__init__.py` is empty (torch-free package import).
- `evals/phase_probe.py` torch-free at import; provides `Thresholds`
  (`eps_reach,delta_lift,k_steps,tau_grip`), `classify_phases(signals,
  thr)`, `step_signals(info)` (reads `proprio/effector_pos`,
  `privileged/block_0_pos`, `proprio/gripper_opening`, `success`).
- `evals/representation_profile.py`: `import torch` at line 15;
  `transport_mask` 64-69, `_spearman` 103-115, `_region` 232-239,
  `probe_coverage` 242-272.
- `scripts/representation_profile_aggregate.py`: `import numpy as np`
  L15, `import pandas as pd` L16, `_SEED_RE` L26; threshold consts
  29-37; `_isnan` 40-44; `_verdict_t1` 47-54; `_verdict_t2` 57-62;
  `_verdict_t3` 65-72; `_verdict_t4` 75-80; `_synthesis` 83-95.
- GCIQL checkpoint: `pickle({'agent': flax state_dict})` at
  `params_<epoch>.pkl`; restore via `GCIQLAgent.create(...)` skeleton +
  `flax.serialization.from_state_dict`
  (`third_party/ogbench/impls/utils/flax_utils.py:162-198`,
  `restore_agent(agent, restore_path, restore_epoch)`).
- Policy: `agent.sample_actions(observations, goals, temperature=0)`;
  Value: `agent.network.select('value')(obs, goal)`.
- OGBench eval pattern (`third_party/ogbench/impls/utils/evaluation.py`):
  `env.reset(options=dict(task_id=task_id))`, `goal=info['goal']`.
- GCIQL run dirs:
  `RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0NN_*/`
  with `params_{100000..1000000}.pkl`, `flags.json`, `eval.csv`
  (`evaluation/task{1..5}_*_success`, `evaluation/overall_success`).
- `scripts/build_paper_fragment.py`: args `--phase-root`,`--repr-root`,
  `--out`,`--paper-subdir`; calls `build_fragment(phase_root, repr_root,
  out, paper_subdir)`.
- User's `PAPER/fb-cube-failure/fb_cube_failure.tex` structure:
  `\section{Where and Why FB Fails on Cube Manipulation}` →
  `\subsection{Experiment Design \& Phase Probing}` →
  `\subsection{Methodology: Representation Probes}` →
  `\subsection{Results \& Empirical Synthesis}` →
  `\subsection{Discussion \& Core Narrative}`.

---

## Task 1: `evals/_profile_core.py` — torch-free shared math

**Files:**
- Create: `evals/_profile_core.py`
- Modify: `evals/representation_profile.py`
- Modify: `scripts/representation_profile_aggregate.py`
- Test: `tests/test_profile_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_profile_core.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import _profile_core as pc


def test_profile_core_is_torch_free():
    import importlib
    import evals._profile_core as m
    src = Path(m.__file__).read_text()
    assert "import torch" not in src
    importlib.reload(m)  # imports cleanly with numpy/pandas only


def test_spearman_monotone_and_degenerate():
    assert pc._spearman(np.array([1.0, 2, 3, 4, 5]),
                         np.array([1.0, 2, 3, 4, 5])) > 0.99
    assert np.isnan(pc._spearman(np.ones(5), np.arange(5.0)))


def test_verdict_helpers_relocated():
    assert pc._verdict_t1(0.40, 0.02)[0] == "SUPPORTS"
    assert pc._verdict_t4(3.0, 1.0)[0] == "SUPPORTS"
    syn = pc._synthesis({"T1": ("SUPPORTS", ""), "T4": ("NEUTRAL", "")})
    assert syn.startswith("Synthesis:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_core.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals._profile_core'`.

- [ ] **Step 3: Create `evals/_profile_core.py`**

Create `evals/_profile_core.py` with exactly these relocated definitions
(copied verbatim from their current locations):

```python
"""evals/_profile_core.py — torch-free probe math shared by the FB
(PyTorch) and GCIQL (JAX) analysis tracks. numpy/pandas only; never
import torch/jax here so it loads in either venv."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _spearman(y: np.ndarray, x: np.ndarray) -> float:
    """Spearman rho (Pearson on average ranks) of y vs x; nan if either
    side is constant or fewer than 2 points. Outlier-robust and
    scale-free, unlike an OLS slope on the unnormalised value V."""
    import pandas as pd

    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def transport_mask(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step boolean: cube lifted above table and gripper closed."""
    cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(ep["grip"], dtype=np.float64).reshape(-1)
    lift = cube[:, 2] - float(ep["table_z"])
    return (lift > thr.delta_lift) & (grip > thr.tau_grip)


def _region(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step region label: 'transport' if lifted+gripped, else
    'lift' if gripper closed, else 'reach'."""
    grip = np.asarray(ep["grip"], np.float64).reshape(-1)
    tm = np.asarray(ep["transport_mask"], bool)
    out = np.where(tm, "transport",
                   np.where(grip > thr.tau_grip, "lift", "reach"))
    return out


def probe_coverage(episodes, ref_obs: np.ndarray, thr,
                   max_ref: int = 5000, seed: int = 0) -> "Any":
    """Per-step nearest-neighbour L2 distance to the standardised offline
    reference, tagged by outcome and region."""
    import pandas as pd

    ref = np.asarray(ref_obs, np.float64)
    if len(ref) > max_ref:
        rng = np.random.default_rng(seed)
        ref = ref[rng.choice(len(ref), max_ref, replace=False)]
    mu = ref.mean(0)
    sd = ref.std(0) + 1e-6
    refn = (ref - mu) / sd
    rows = []
    for ep in episodes:
        obs = np.asarray(ep["obs"], np.float64)
        if obs.shape[0] == 0:
            continue
        on = (obs - mu) / sd
        d2 = (on ** 2).sum(1, keepdims=True) + (refn ** 2).sum(1) \
            - 2.0 * on @ refn.T
        nn = np.sqrt(np.maximum(d2.min(1), 0.0))
        reg = _region(ep, thr)
        for r, n in zip(reg, nn):
            rows.append({"outcome": ep["outcome"], "region": str(r),
                         "nn_dist": float(n)})
    return pd.DataFrame(rows)


T1_RHO_MIN = 0.15
T1_RHO_GAP = 0.10
T2_SPARSE_FRAC = 0.05
T2_TOPK_AT_GOAL = 0.5
T3_R2_FAIL = 0.5
T3_ACC_FAIL = 0.75
T3_R2_RESOLVE = 0.7
T3_ACC_RESOLVE = 0.9
T4_OFFSUPPORT_RATIO = 1.25


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except TypeError:
        return True


def _verdict_t1(rho_succ, rho_fail):
    if _isnan(rho_succ) or _isnan(rho_fail):
        return "INSUFFICIENT DATA", "no valid transport-region episodes"
    if rho_succ >= T1_RHO_MIN and rho_succ - rho_fail >= T1_RHO_GAP:
        return "SUPPORTS", "success value rises toward goal; fail flatter"
    if rho_fail - rho_succ >= T1_RHO_GAP:
        return "CONTRADICTS", "fail value rises more than success"
    return "WEAK", "no clear success/fail gradient separation"


def _verdict_t2(frac, topk_at_goal):
    if _isnan(frac) or _isnan(topk_at_goal):
        return "INSUFFICIENT DATA", "no relabel/z-decoding stats"
    if frac < T2_SPARSE_FRAC and topk_at_goal < T2_TOPK_AT_GOAL:
        return "SUPPORTS", "data goal-sparse; z's top states off-goal"
    return "WEAK", "data not strongly goal-sparse or z points at goal"


def _verdict_t3(r2, acc):
    if _isnan(r2) or _isnan(acc):
        return "INSUFFICIENT DATA", "no B-resolution stats"
    if r2 < T3_R2_FAIL and acc < T3_ACC_FAIL:
        return "SUPPORTS", "B barely resolves placement distance"
    if r2 >= T3_R2_RESOLVE and acc >= T3_ACC_RESOLVE:
        return "RESOLVES", "B cleanly resolves placement (contradicts)"
    return "WEAK", "B partially resolves placement"


def _verdict_t4(fail_nn, succ_nn):
    if _isnan(fail_nn) or _isnan(succ_nn):
        return "INSUFFICIENT DATA", "missing transport-region coverage"
    if fail_nn >= T4_OFFSUPPORT_RATIO * succ_nn:
        return "SUPPORTS", "transport-fails sit further off data support"
    return "NEUTRAL", "fails not meaningfully more off-support"


def _synthesis(verdicts) -> str:
    supports = [k for k, (v, _) in verdicts.items() if v == "SUPPORTS"]
    contra = [k for k, (v, _) in verdicts.items()
              if v in ("CONTRADICTS", "RESOLVES")]
    if supports:
        s = (", ".join(supports)
             + " support the representation-failure hypothesis")
    else:
        s = "no probe supports the representation-failure hypothesis"
    if contra:
        s += "; " + ", ".join(contra) + " argue against it"
    return "Synthesis: " + s + "."
```

- [ ] **Step 4: Re-point `representation_profile.py` at the shared module**

In `evals/representation_profile.py`, delete the now-relocated
definitions: `transport_mask` (lines ~64-69), `_spearman` (~103-115),
`_region` (~232-239), `probe_coverage` (~242-272). Add, immediately
after the `import torch` line (line 15):

```python

from evals._profile_core import (  # noqa: E402  (re-exported for callers)
    _region,
    _spearman,
    probe_coverage,
    transport_mask,
)
```

(Internal callers `probe_value_landscape` (`_spearman`),
`rollout_for_profile` (`transport_mask`), and `probe_coverage`/`_region`
keep working via these names; `rp.transport_mask` etc. stay importable.)

- [ ] **Step 5: Re-point `representation_profile_aggregate.py` at the shared module**

In `scripts/representation_profile_aggregate.py`, delete the threshold
constants (lines ~29-37) and `_isnan`,`_verdict_t1`,`_verdict_t2`,
`_verdict_t3`,`_verdict_t4`,`_synthesis` (lines ~40-95). Immediately
after the `_SEED_RE = re.compile(r"s(\d+)_final")` line add:

```python

from evals._profile_core import (  # noqa: E402
    _isnan, _synthesis, _verdict_t1, _verdict_t2, _verdict_t3,
    _verdict_t4, T1_RHO_GAP, T1_RHO_MIN, T2_SPARSE_FRAC,
    T2_TOPK_AT_GOAL, T3_ACC_FAIL, T3_ACC_RESOLVE, T3_R2_FAIL,
    T3_R2_RESOLVE, T4_OFFSUPPORT_RATIO,
)
```

(`REPO_ROOT` is already on `sys.path` in that script before this import;
keep this import after that `sys.path.insert`.)

- [ ] **Step 6: Run the new test + the full suite (behaviour preserved)**

Run: `.venv/bin/python -m pytest tests/test_profile_core.py tests/test_representation_profile.py tests/test_paper_fragment.py tests/test_phase_probe.py -q`
Expected: PASS. Then `.venv/bin/python -m pytest tests/ -q` → all green
(the relocation is behaviour-preserving; the existing FB tests are the
regression guard).

- [ ] **Step 7: Commit**

```bash
git add evals/_profile_core.py evals/representation_profile.py scripts/representation_profile_aggregate.py tests/test_profile_core.py
git commit -m "refactor(profile): torch-free _profile_core shared by FB+GCIQL"
```

---

## Task 2: `scripts/gciql_profile.py` — pure JAX-free seams

**Files:**
- Create: `scripts/gciql_profile.py` (pure helpers only this task)
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gciql_profile.py`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile as gp
from evals.phase_probe import Thresholds


def test_parse_flags_reads_env_and_seed(tmp_path):
    (tmp_path / "flags.json").write_text(json.dumps(
        {"env_name": "cube-single-play-v0", "seed": 10,
         "agent": "agents/gciql.py"}))
    info = gp.parse_flags(tmp_path)
    assert info["env_name"] == "cube-single-play-v0"
    assert info["seed"] == 10


def _ep(outcome, d, v, grip, cube_z, table_z=0.02):
    T = len(d)
    return {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.asarray(d, np.float64),
        "V": np.asarray(v, np.float64),
        "grip": np.asarray(grip, np.float64),
        "cube": np.column_stack([np.zeros(T), np.zeros(T),
                                 np.asarray(cube_z)]),
        "table_z": table_z, "success": outcome == "success",
        "task": "task1", "episode": 0,
    }


def test_episodes_to_frames_t1_t4_funnel():
    # 6 transport steps, V strictly rises as d -> 0  => rho ~ +1
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6                    # lifted (table_z 0.02)
    eps = [_ep("success", d, v, grip, cube_z)]
    vl, cov, fun = gp.episodes_to_frames(eps, Thresholds(),
                                         ref_obs=np.zeros((10, 4)))
    assert set(vl.columns) >= {"task", "episode", "outcome",
                               "rho_V_negd", "n_transport_steps"}
    assert vl.iloc[0]["rho_V_negd"] > 0.99
    assert vl.iloc[0]["outcome"] == "success"
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}
    assert set(fun.columns) >= {"task", "episode", "furthest_phase",
                                "outcome"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.gciql_profile'`.

- [ ] **Step 3: Create `scripts/gciql_profile.py` with the pure seams**

Create `scripts/gciql_profile.py`:

```python
"""scripts/gciql_profile.py — per-checkpoint GCIQL profiler.

Runs ONLY in the isolated JAX venv (it imports jax/flax/vendored
OGBench inside main()/_jax_*). The pure seams below (flags parsing,
episode-records -> frames) are JAX-free and unit-tested.

T1 (Spearman rho(V(s,g), -d)), T4 (off-support coverage) and the phase
funnel are computed with evals._profile_core / evals.phase_probe so the
FB and GCIQL numbers are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import _spearman, probe_coverage, transport_mask
from evals.phase_probe import Thresholds, classify_phases


def parse_flags(run_dir) -> Dict[str, Any]:
    """Read OGBench flags.json from a GCIQL seed dir."""
    data = json.loads((Path(run_dir) / "flags.json").read_text())
    return {"env_name": data.get("env_name", "cube-single-play-v0"),
            "seed": int(data.get("seed", 0)),
            "agent": data.get("agent", "agents/gciql.py")}


def episodes_to_frames(episodes: List[Dict[str, Any]], thr: Thresholds,
                       ref_obs: np.ndarray):
    """Episode records -> (value_landscape, coverage, funnel) frames.

    Each record: obs[T,O], d[T], V[T], grip[T], cube[T,3], table_z,
    success(bool), task(str), episode(int). Pure; no jax.
    """
    vl_rows, fun_rows, cov_eps = [], [], []
    for ep in episodes:
        d = np.asarray(ep["d"], np.float64)
        V = np.asarray(ep["V"], np.float64)
        tm = transport_mask(ep, thr)
        sig = {
            "eff": np.zeros((len(d), 3)),
            "cube": np.asarray(ep["cube"], np.float64).reshape(-1, 3),
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "goal": np.asarray(ep["cube"], np.float64).reshape(-1, 3)[-1],
            "success": bool(ep["success"]),
            "table_z": float(ep["table_z"]),
            "length": len(d),
        }
        cls = classify_phases(sig, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        n_t = int(tm.sum())
        if n_t >= 5 and np.ptp(d[tm]) > 1e-9 and np.ptp(V[tm]) > 1e-9:
            rho = _spearman(V[tm], -d[tm])
            v_secure, v_end = float(V[tm][0]), float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        vl_rows.append({"task": ep["task"], "episode": ep["episode"],
                        "outcome": outcome, "rho_V_negd": rho,
                        "V_at_secure": v_secure, "V_at_end": v_end,
                        "n_transport_steps": n_t})
        fun_rows.append({"task": ep["task"], "episode": ep["episode"],
                         "furthest_phase": cls["fail_phase"]
                         if not cls["success"] else "success",
                         "outcome": outcome})
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
    cov = probe_coverage(cov_eps, ref_obs, thr)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gciql_profile.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): JAX-free flags + episodes->frames seams"
```

---

## Task 3: `scripts/gciql_profile.py` — JAX rollout + CLI

**Files:**
- Modify: `scripts/gciql_profile.py`
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Add a CLI-parse failing test**

Append to `tests/test_gciql_profile.py`:

```python
def test_build_argparser_has_required_flags():
    ap = gp.build_argparser()
    ns = ap.parse_args([
        "--run-dir", "/x/sd001", "--step", "1000000",
        "--out", "/tmp/o", "--tasks", "1,2,3", "--n-episodes", "4",
        "--dataset-path", "datasets/cube-single-play-v0"])
    assert ns.run_dir == "/x/sd001" and ns.step == 1000000
    assert ns.tasks == "1,2,3" and ns.n_episodes == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k build_argparser -q`
Expected: FAIL — `AttributeError: module 'scripts.gciql_profile' has no attribute 'build_argparser'`.

- [ ] **Step 3: Add the argparser + JAX rollout + main**

Append to `scripts/gciql_profile.py`:

```python
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="a GCIQL seed dir (contains flags.json, "
                         "params_<step>.pkl)")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--scenario", default="S0", choices=["S0"])
    ap.add_argument("--dataset-path",
                    default="datasets/cube-single-play-v0")
    ap.add_argument("--max-steps", type=int, default=1000)
    return ap


def _load_ref_obs(dataset_path: str, max_n: int = 5000) -> np.ndarray:
    """Stack next/observation from the OGBench cube npz buffer (same
    data the FB coverage probe used)."""
    buf = Path(dataset_path) / "buffer"
    files = sorted(buf.glob("episode_*.npz"))[:200]
    obs = []
    for f in files:
        with np.load(f) as z:
            key = "observations" if "observations" in z else "obs"
            obs.append(np.asarray(z[key], np.float32))
    arr = np.concatenate(obs, 0) if obs else np.zeros((1, 1), np.float32)
    if len(arr) > max_n:
        rng = np.random.default_rng(0)
        arr = arr[rng.choice(len(arr), max_n, replace=False)]
    return arr


def _jax_rollout(run_dir: str, step: int, task_ids: List[int],
                 n_episodes: int, max_steps: int):
    """Load the GCIQL checkpoint and roll out the goal-conditioned
    policy. Imports jax/flax/vendored OGBench — JAX venv only."""
    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb_impls) not in sys.path:
        sys.path.insert(0, str(ogb_impls))
    import ogbench  # vendored
    from agents.gciql import GCIQLAgent
    from utils.flax_utils import restore_agent
    import jax

    flags = parse_flags(run_dir)
    episodes: List[Dict[str, Any]] = []
    for task_id in task_ids:
        env = ogbench.make_env_and_datasets(
            flags["env_name"], compact_dataset=True, env_only=True) \
            if hasattr(ogbench, "make_env_and_datasets") else \
            ogbench.make_env(flags["env_name"])
        ex_obs, info = env.reset(options=dict(task_id=task_id))
        goal = info["goal"]
        ex_act = env.action_space.sample()
        agent = GCIQLAgent.create(
            flags["seed"],
            np.asarray(ex_obs)[None],
            np.asarray(goal)[None],
            np.asarray(ex_act)[None],
            GCIQLAgent.get_config(),
        )
        agent = restore_agent(agent, str(run_dir), step)
        value_fn = jax.jit(lambda o, g: agent.network.select("value")(
            o[None], g[None])[0])
        for ep_i in range(n_episodes):
            obs, info = env.reset(options=dict(task_id=task_id))
            goal = info["goal"]
            O, A, EFF, CUBE, GRIP, VV, D = [], [], [], [], [], [], []
            succ = False
            for _ in range(max_steps):
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(obs), goals=np.asarray(goal),
                    temperature=0.0))
                v = float(np.asarray(value_fn(
                    np.asarray(obs, np.float32),
                    np.asarray(goal, np.float32))))
                obs, _, term, trunc, info = env.step(a)
                eff = np.asarray(info["proprio/effector_pos"], np.float64)
                cube = np.asarray(info["privileged/block_0_pos"],
                                  np.float64)
                grip = float(np.asarray(
                    info["proprio/gripper_opening"]).reshape(-1)[0])
                gxyz = np.asarray(info.get("privileged/target_block_pos",
                                           cube), np.float64)
                O.append(np.asarray(obs, np.float32)); A.append(a)
                EFF.append(eff); CUBE.append(cube); GRIP.append(grip)
                VV.append(v)
                D.append(float(np.linalg.norm(cube - gxyz)))
                succ = succ or bool(info.get("success", False))
                if term or trunc:
                    break
            episodes.append({
                "obs": np.asarray(O, np.float32),
                "d": np.asarray(D), "V": np.asarray(VV),
                "grip": np.asarray(GRIP),
                "cube": np.asarray(CUBE),
                "table_z": float(CUBE[0][2]) if CUBE else 0.02,
                "success": succ,
                "task": f"task{task_id}", "episode": ep_i})
        env.close()
    return episodes, flags


def main() -> None:
    args = build_argparser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    episodes, flags = _jax_rollout(
        args.run_dir, args.step, task_ids, args.n_episodes,
        args.max_steps)
    ref = _load_ref_obs(args.dataset_path)
    vl, cov, fun = episodes_to_frames(episodes, Thresholds(), ref)
    vl.to_parquet(out / "value_landscape.parquet")
    cov.to_parquet(out / "coverage.parquet")
    fun.to_parquet(out / "phase_funnel.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "step": args.step,
        "tasks": task_ids, "n_episodes": args.n_episodes,
        "env_name": flags["env_name"],
        "value_fn": "GCValue V(s,g)",
    }, indent=2))
    print(f"[gciql_profile] done -> {out}")


if __name__ == "__main__":
    main()
```

> **Smoke note (run where the JAX venv exists; see Task 6):** the exact
> `ogbench` env factory, `GCIQLAgent.get_config`, and the cube `info`
> keys (`privileged/target_block_pos`) are confirmed against the
> vendored copy during the Task 6 smoke; if the env factory name or an
> info key differs, fix it only inside `_jax_rollout` (the single place
> that touches the vendored env) — the pure seams and their tests are
> unaffected.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (3 tests; `build_argparser` parses; no jax imported by
the pure tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gciql_profile.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): JAX checkpoint rollout + CLI"
```

---

## Task 4: `scripts/gciql_profile_aggregate.py` + FB-vs-GCIQL comparison

**Files:**
- Create: `scripts/gciql_profile_aggregate.py`
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_gciql_profile.py`:

```python
def _seed_dir(root, s):
    d = root / f"s{s}_final"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "episode": 0, "outcome": "success",
         "rho_V_negd": 0.6, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": 6},
        {"task": "task1", "episode": 1, "outcome": "transport_fail",
         "rho_V_negd": 0.0, "V_at_secure": 0.5, "V_at_end": 0.5,
         "n_transport_steps": 6},
    ]).to_parquet(d / "value_landscape.parquet")
    pd.DataFrame([
        {"task": "task1", "outcome": "transport_fail",
         "region": "transport", "nn_dist": 3.0},
        {"task": "task1", "outcome": "success",
         "region": "transport", "nn_dist": 1.0}],
    ).to_parquet(d / "coverage.parquet")
    pd.DataFrame([
        {"task": "task1", "episode": 0, "furthest_phase": "success",
         "outcome": "success"},
        {"task": "task1", "episode": 1, "furthest_phase": "transport",
         "outcome": "transport_fail"}],
    ).to_parquet(d / "phase_funnel.parquet")


def test_gciql_aggregate_and_comparison(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    for s in (1, 2):
        _seed_dir(groot, s)
    # synthetic FB aggregate for the comparison
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "outcome": "success", "rho_V_negd": 0.13},
        {"task": "task1", "outcome": "transport_fail",
         "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg)
    assert (out / "story.md").exists()
    assert (out / "T1_value_gradient.parquet").exists()
    cmp = out / "comparison"
    assert (cmp / "fb_vs_gciql.parquet").exists()
    assert (cmp / "comparison.md").exists()
    txt = (cmp / "comparison.md").read_text()
    assert "FB" in txt and "GCIQL" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k gciql_aggregate -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.gciql_profile_aggregate'`.

- [ ] **Step 3: Create `scripts/gciql_profile_aggregate.py`**

Create `scripts/gciql_profile_aggregate.py`:

```python
"""scripts/gciql_profile_aggregate.py — 10-seed GCIQL aggregate
(T1/T4/funnel) + FB-vs-GCIQL comparison. Reuses _profile_core verdicts
so GCIQL and FB are scored identically."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evals._profile_core import (  # noqa: E402
    _synthesis, _verdict_t1, _verdict_t4)

_SEED_RE = re.compile(r"s(\d+)_final")


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(root).glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/gciql_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3g} ± {s.std(ddof=0):.2g}"


def aggregate(root, out, fb_aggregate=None) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vl = _load(root, "value_landscape")
    cv = _load(root, "coverage")
    fun = _load(root, "phase_funnel")

    t1 = (vl.groupby(["task", "outcome"])["rho_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")
    fun.to_parquet(out / "funnel.parquet")

    rho_s = float(vl[vl.outcome == "success"]["rho_V_negd"].mean())
    rho_f = float(vl[vl.outcome == "transport_fail"]["rho_V_negd"].mean())
    cvt = cv[cv.region == "transport"]
    fail_nn = float(cvt[cvt.outcome == "transport_fail"]["nn_dist"].mean())
    succ_nn = float(cvt[cvt.outcome == "success"]["nn_dist"].mean())
    verdicts = {"T1": _verdict_t1(rho_s, rho_f),
                "T4": _verdict_t4(fail_nn, succ_nn)}

    fig, ax = plt.subplots(figsize=(6, 4))
    g = vl.groupby("outcome")["rho_V_negd"].mean()
    ax.bar([str(i) for i in g.index], g.values)
    ax.set_title("GCIQL value gradient")
    ax.set_ylabel("Spearman rho(V, -d)")
    fig.tight_layout(); fig.savefig(out / "value_rho_bars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title("GCIQL coverage"); ax.set_ylabel("nn_dist")
    fig.tight_layout(); fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    fc = fun.groupby("furthest_phase").size()
    ax.bar([str(i) for i in fc.index], fc.values)
    ax.set_title("GCIQL phase funnel")
    fig.tight_layout(); fig.savefig(out / "funnel.png", dpi=120)
    plt.close(fig)

    lines = [
        "# GCIQL cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "", "## Value gradient (T1)",
        f"- Spearman ρ(V vs −d) success: "
        f"{_ms(vl[vl.outcome=='success']['rho_V_negd'])}",
        f"- Spearman ρ(V vs −d) transport-fail: "
        f"{_ms(vl[vl.outcome=='transport_fail']['rho_V_negd'])}",
        "", "## Coverage (T4)",
        f"- transport nn_dist (transport_fail): "
        f"{_ms(cvt[cvt.outcome=='transport_fail']['nn_dist'])}",
        "", "## Readout",
        f"- T1 value gradient: rho_success={rho_s:.3g}, "
        f"rho_fail={rho_f:.3g} -> {verdicts['T1'][0]} "
        f"({verdicts['T1'][1]})",
        f"- T4 coverage: fail={fail_nn:.3g}, success={succ_nn:.3g} "
        f"-> {verdicts['T4'][0]} ({verdicts['T4'][1]})",
        _synthesis(verdicts),
    ]
    (out / "story.md").write_text("\n".join(lines))

    if fb_aggregate is not None:
        _comparison(out / "comparison", vl, cvt, fb_aggregate,
                    rho_s, rho_f, fail_nn, succ_nn, verdicts)
    print(f"[gciql_profile_aggregate] {vl['seed'].nunique()} seeds -> {out}")


def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    fb_t1 = pd.read_parquet(
        Path(fb_aggregate) / "T1_value_gradient.parquet")

    def _fb_rho(oc):
        sub = fb_t1[fb_t1.outcome == oc]
        col = "mean" if "mean" in sub.columns else "rho_V_negd"
        return float(sub[col].mean()) if len(sub) else float("nan")

    fb_rs, fb_rf = _fb_rho("success"), _fb_rho("transport_fail")
    fbv = _verdict_t1(fb_rs, fb_rf)
    rows = [
        {"method": "FB", "rho_success": fb_rs, "rho_fail": fb_rf,
         "T1_verdict": fbv[0]},
        {"method": "GCIQL", "rho_success": g_rs, "rho_fail": g_rf,
         "T1_verdict": g_verd["T1"][0]},
    ]
    cmp = pd.DataFrame(rows)
    cmp.to_parquet(cdir / "fb_vs_gciql.parquet")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [fb_rs, g_rs], 0.4, label="success")
    ax.bar(x + 0.2, [fb_rf, g_rf], 0.4, label="transport_fail")
    ax.set_xticks(x); ax.set_xticklabels(["FB", "GCIQL"])
    ax.set_ylabel("Spearman rho(V, -d)"); ax.legend()
    ax.set_title("FB vs GCIQL: value gradient")
    fig.tight_layout(); fig.savefig(cdir / "cmp_value_rho.png", dpi=120)
    plt.close(fig)

    (cdir / "comparison.md").write_text("\n".join([
        "# FB vs GCIQL — shared-axis comparison", "",
        f"- FB    value gradient: rho_s={fb_rs:.3g}, rho_f={fb_rf:.3g} "
        f"-> {fbv[0]}",
        f"- GCIQL value gradient: rho_s={g_rs:.3g}, rho_f={g_rf:.3g} "
        f"-> {g_verd['T1'][0]}",
        f"- GCIQL coverage transport nn_dist: fail={g_fnn:.3g}, "
        f"success={g_snn:.3g} -> {g_verd['T4'][0]}",
        "",
        "Both methods are compared only on axes that transfer "
        "(value gradient, coverage, phase funnel); FB-specific "
        "z-decoding (T2) and B-resolution (T3) have no GCIQL analog.",
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "gciql_profile"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fb-aggregate",
                    default=str(REPO_ROOT / "analysis"
                               / "representation_profile" / "aggregate"))
    args = ap.parse_args()
    root = Path(args.root)
    fb = Path(args.fb_aggregate)
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gciql_profile_aggregate.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): 10-seed aggregate + FB-vs-GCIQL comparison"
```

---

## Task 5: Paper generator — user's structure + GCIQL comparison

**Files:**
- Modify: `evals/paper_fragment.py`
- Modify: `scripts/build_paper_fragment.py`
- Test: `tests/test_paper_fragment.py`

- [ ] **Step 1: Add failing assertions for the user's structure + GCIQL**

In `tests/test_paper_fragment.py`, inside `test_build_fragment`, after
the existing `assert "\\begin{tabular}" in tex` line add:

```python
    # user's hand-restructured layout, reproduced by the generator
    assert "Where and Why FB Fails on Cube Manipulation" in tex
    for sub in ("Experiment Design", "Methodology: Representation Probes",
                "Results", "Discussion"):
        assert sub in tex
    # GCIQL comparison folded in when a comparison dir is supplied
    gcmp = tmp_path / "gcmp"
    gcmp.mkdir()
    pd.DataFrame([{"method": "FB", "rho_success": 0.13,
                   "rho_fail": 0.23, "T1_verdict": "WEAK"},
                  {"method": "GCIQL", "rho_success": 0.4,
                   "rho_fail": 0.05, "T1_verdict": "SUPPORTS"}]
                 ).to_parquet(gcmp / "fb_vs_gciql.parquet")
    (gcmp / "comparison.md").write_text(
        "# cmp\n- FB ...\n- GCIQL ...")
    (gcmp / "cmp_value_rho.png").write_bytes(b"\x89PNG")
    pf.build_fragment(phase, repr_, tmp_path / "PAPER3", "x",
                      gciql_cmp=gcmp)
    t3 = (tmp_path / "PAPER3" / "x" / "fb_cube_failure.tex").read_text()
    assert "FB vs GCIQL" in t3
    assert "cmp_value_rho.png" in t3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py::test_build_fragment -q`
Expected: FAIL — `build_fragment() got an unexpected keyword argument 'gciql_cmp'` (and structure strings absent).

- [ ] **Step 3: Re-template `build_fragment` to the user's structure + GCIQL**

In `evals/paper_fragment.py`, change the `build_fragment` signature to
add the keyword:

```python
def build_fragment(phase_root, repr_root, out_dir, subdir: str,
                   figures: Sequence[str] = _DEFAULT_FIGS,
                   gciql_cmp=None) -> Path:
```

Replace the document-assembly list (the `doc = [...]` block) so the
section/subsection titles exactly match the user's restructured file and
a GCIQL block is appended into Results when `gciql_cmp` is provided.
Replace from `doc = [` through the `]` that closes it with:

```python
    gcmp_block = ""
    if gciql_cmp is not None:
        gciql_cmp = Path(gciql_cmp)
        cmp_png = gciql_cmp / "cmp_value_rho.png"
        if cmp_png.exists():
            shutil.copyfile(cmp_png, figdir / "cmp_value_rho.png")
        cmp_tbl = ""
        pq = gciql_cmp / "fb_vs_gciql.parquet"
        if pq.exists():
            import pandas as pd
            cmp_tbl = df_to_booktabs(
                pd.read_parquet(pq), caption=_tex("FB vs GCIQL"),
                label="tab:fb-vs-gciql")
        cmp_md = ""
        cmp_md_p = gciql_cmp / "comparison.md"
        if cmp_md_p.exists():
            cmp_md = _story_to_latex(cmp_md_p.read_text())
        gcmp_block = "\n".join([
            r"\subsection{FB vs GCIQL: Shared-Axis Comparison}",
            _fig("figures", "cmp_value_rho.png",
                 (figdir / "cmp_value_rho.png").exists(),
                 _tex("FB vs GCIQL: value gradient")),
            cmp_tbl, cmp_md])

    doc = [
        "% Auto-generated self-contained document; regenerate via "
        "scripts/build_paper_fragment.py",
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\title{Where and Why FB Fails on Cube Manipulation}",
        r"\author{FB representation-failure profile (auto-generated)}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        _abstract_from_story(story_md),
        r"\end{abstract}",
        "",
        r"\section{Where and Why FB Fails on Cube Manipulation}",
        r"\label{sec:fb-cube-failure}",
        "",
        r"\subsection{Experiment Design \& Phase Probing}",
        r"We decompose each cube episode into reach $\to$ grasp/lift "
        r"$\to$ transport, and probe the policy from three start "
        r"states: S0 (baseline), S1 (skip-reach), S2 (pre-grasped, "
        r"physically clamped, no weld). The phase-probe over seeds "
        r"s3--s8, s10 localises where the rollout breaks.",
        figs_phase,
        "",
        r"\subsection{Methodology: Representation Probes}",
        r"Four representation probes on the FB checkpoints: (i) value "
        r"landscape $V(s)=Q_z(s,\pi(s))$, scored as the Spearman "
        r"$\rho$ between $V$ and $-d$ (cube--goal distance) over the "
        r"transport region; (ii) z-decoding and relabel sparsity; "
        r"(iii) ridge $R^2$ of $B(s)\!\to\!d$ and placed-vs-near "
        r"separability; (iv) off-support nearest-neighbour distance. "
        r"Probes are aggregated (mean$\pm$std) over the seven seeds "
        r"and each is assigned a heuristic verdict.",
        "",
        r"\subsection{Results \& Empirical Synthesis}",
        figs_repr,
        tbls,
        _story_to_latex(story_md),
        gcmp_block,
        "",
        r"\subsection{Discussion \& Core Narrative}",
        r"The evidence localises the failure to the \emph{"
        r"representation}, not the value landscape or data coverage. "
        r"The play data almost never shows the cube at the goal and "
        r"the inferred task vector $z$'s highest-scoring states are "
        r"mostly off-goal (T2); the backward map $B$ barely linearly "
        r"resolves placement distance (T3). With a scale-free metric "
        r"the value function provides no gradient that separates "
        r"successes from transport-failures (T1 weak), and failures "
        r"are not measurably further off data support than successes "
        r"(T4 neutral). FB lifts the cube but lacks a learned, "
        r"sharply-encoded target to deliver it.",
        "",
        r"\end{document}",
        "",
    ]
```

- [ ] **Step 4: Add `--gciql-root` to the build CLI**

In `scripts/build_paper_fragment.py`, after the `--paper-subdir`
argument add:

```python
    ap.add_argument("--gciql-root",
                    default=str(REPO_ROOT / "analysis" / "gciql_profile"
                               / "aggregate" / "comparison"))
```

and change the `build_fragment(...)` call to:

```python
    from pathlib import Path as _P
    g = _P(args.gciql_root)
    base = build_fragment(args.phase_root, args.repr_root, args.out,
                          args.paper_subdir,
                          gciql_cmp=g if g.exists() else None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py -q`
Expected: PASS (3 tests; FB-only path and GCIQL-comparison path both
green).

- [ ] **Step 6: Commit**

```bash
git add evals/paper_fragment.py scripts/build_paper_fragment.py tests/test_paper_fragment.py
git commit -m "feat(paper): user-structured template + FB-vs-GCIQL subsection"
```

---

## Task 6: Full verification + GCIQL run + paper regen

**Files:** none (operational).

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, zero failures.

- [ ] **Step 2: Attempt a local CPU JAX venv (feasibility)**

```bash
uv venv --python 3.11 .venv-jax-local
UV_CACHE_DIR=.uv-cache-jax uv pip install --python .venv-jax-local/bin/python -r requirements-jax.txt
.venv-jax-local/bin/python -c "import jax, flax, ogbench; print('jax', jax.__version__)"
```

If this fails on darwin, STOP this step and hand Steps 3–4 to the user
to run on the Linux box where GCIQL runs live (the commands are
identical, substituting the Linux JAX venv for `.venv-jax-local`).

- [ ] **Step 3: Smoke one seed, then run all 10**

Smoke (verifies env factory / `GCIQLAgent.get_config` / cube info keys;
fix only inside `_jax_rollout` if a key/name differs):

```bash
RUN=$(ls -d RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd001_*)
.venv-jax-local/bin/python scripts/gciql_profile.py \
  --run-dir "$RUN" --step 1000000 --out /tmp/gciql_smoke_s1 \
  --tasks 1 --n-episodes 1 --dataset-path datasets/cube-single-play-v0
```
Expected: `/tmp/gciql_smoke_s1/{value_landscape,coverage,phase_funnel}.parquet`
+ `metrics.json`; `value_landscape.parquet` has a `rho_V_negd` column.

Then all 10 seeds → `analysis/gciql_profile/sNN_final/`:

```bash
i=0
for RUN in RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0*_*; do
  i=$((i+1))
  .venv-jax-local/bin/python scripts/gciql_profile.py \
    --run-dir "$RUN" --step 1000000 \
    --out analysis/gciql_profile/s${i}_final \
    --tasks 1,2,3,4,5 --n-episodes 10 \
    --dataset-path datasets/cube-single-play-v0
done
```

- [ ] **Step 4: Aggregate + comparison + paper regen**

```bash
.venv/bin/python scripts/gciql_profile_aggregate.py
.venv/bin/python scripts/build_paper_fragment.py --paper-subdir fb-cube-failure
```
Expected: `analysis/gciql_profile/aggregate/{T1_value_gradient.parquet,
T4_coverage.parquet,funnel.parquet,story.md,*.png}` and
`.../comparison/{fb_vs_gciql.parquet,cmp_value_rho.png,comparison.md}`;
`PAPER/fb-cube-failure/fb_cube_failure.tex` reproduces the user's
4-subsection layout and contains the `FB vs GCIQL` subsection.

- [ ] **Step 5: Report**

Read `analysis/gciql_profile/aggregate/comparison/comparison.md` and
report the FB-vs-GCIQL verdicts to the user. Do not commit
`analysis/`/`PAPER/` artifacts unless the user asks.

---

## Self-review

- **Spec coverage:** Component 1 → Task 1; Component 2 → Tasks 2–3;
  Component 3 → Task 4; Component 4 → Task 5; execution/feasibility →
  Task 6. T1/T4/funnel only (no T2/T3) honoured in Tasks 2–4. Isolation
  honoured: `gciql_profile.py` imports jax only inside `_jax_rollout`;
  pure seams + all other tasks torch/jax-free. Paper preserves the
  user's exact subsection titles (Task 5 Step 3) and is generator-driven
  (reconciliation choice). Same-dataset T4 reference via
  `--dataset-path datasets/cube-single-play-v0` (Tasks 3/6).
- **Placeholder scan:** none — every code step is complete; the Task 3
  smoke note bounds the only env-dependent unknowns to `_jax_rollout`
  with an explicit fix locus, not a placeholder.
- **Type consistency:** parquet schemas consistent across producer
  (Task 3 `episodes_to_frames`/`main`), consumer (Task 4 `_load`/
  `aggregate`/`_comparison`), and paper (Task 5 reads
  `fb_vs_gciql.parquet`, `cmp_value_rho.png`, `comparison.md`). Verdict
  helpers (`_verdict_t1/_t4`,`_synthesis`) defined in Task 1
  `_profile_core` and imported identically in Tasks 4. `build_fragment`
  `gciql_cmp` kwarg defined Task 5 Step 3, used Task 5 Steps 1/4.
  `parse_flags`/`episodes_to_frames`/`build_argparser` names consistent
  between Tasks 2–3 and their tests.
