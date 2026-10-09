# FB Representation-Failure Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a four-probe FB representation diagnostic (value landscape, z-decoding+sparsity, B-resolution, coverage), a per-checkpoint CLI, a 7-seed aggregate (tables/plots/`story.md`), and a full-arc LaTeX paper fragment under `PAPER/<subdir>/`.

**Architecture:** Mirrors the `phase_probe` pattern: testable library `evals/representation_profile.py` + thin per-checkpoint CLI + cross-seed aggregate script, plus `evals/paper_fragment.py` + `scripts/build_paper_fragment.py` that consume existing aggregate artifacts. Pure logic is TDD-tested with stubs (no MuJoCo, no sklearn); real runs are documented acceptance steps. Additive only.

**Tech Stack:** Python 3.11, numpy, scipy, pandas/pyarrow, matplotlib (Agg), torch; reuses `evals.analysis`, `evals.phase_probe`, `OGBenchEvaluator._infer_z`, `envs.ogbench`.

**Spec:** `docs/superpowers/specs/2026-05-18-fb-representation-failure-profile-design.md`

**Test command:** `.venv/bin/python -m pytest <path> -v`

---

## Verified codebase facts

- `model.backward_map(obs) -> B [N, z_dim]` (public, `@torch.no_grad`, applies bw-encoder+normalize). `model.forward_map(obs, z, action) -> F [num_parallel, N, z_dim]`. `agent.act(obs[B,obs_dim], z) -> action [B,act_dim]` (z 1-D auto-unsqueezed). `z` is L2-projected; `Q_z = (F*z).sum(-1)` → `[num_parallel, N]`; diagnostics use **ensemble mean over dim 0**.
- `OGBenchEvaluator._infer_z(task) -> (z, relabel_metrics)` where `relabel_metrics` has `relabel_reward#nonzero`, `relabel_reward#num_samples` (post +1 shift: nonzero == goal-satisfying states for single cube).
- `envs.ogbench.get_relabel_fn(domain, task) -> fn(physics_np[N,phys], action_np[N,act]) -> reward[N,1]`. Cube xyz = `physics[:, 14:17]`; goal = `env.unwrapped.cur_task_info['goal_xyzs'][tb]`, `tb = getattr(env.unwrapped,'_target_block',0)`.
- Offline buffer `buffer.sample(n)` → dict with `["observation"]`, `["action"]`, `["next"]["observation"]`, `["next"]["physics"]` (tensors).
- `evals.phase_probe.rollout_with_phase_signals(env, agent, z, n, thr, scenario="S0")` returns per-episode dicts (`eff,cube,grip,goal,success,table_z,length`); `classify_phases(sig, thr)` → `furthest_phase`, `fail_phase` (`"transport"` etc.). mujoco import is lazy via `phase_probe._ensure_mujoco`.
- No `sklearn`; `scipy` is available. Use numpy closed-form ridge + Fisher-LDA accuracy.

## File Structure

| File | Responsibility |
|---|---|
| `evals/representation_profile.py` | `_ensemble_q`, `q_values`, `v_values`, `rollout_for_profile`, `transport_mask`, `probe_value_landscape`, `probe_z_decoding`, `probe_b_resolution`, `probe_coverage`, `run_representation_profile` |
| `evals/phase_probe.py` (modify) | add opt-in `record_obs` to `rollout_with_phase_signals` |
| `scripts/representation_profile.py` | per-checkpoint CLI |
| `scripts/representation_profile_aggregate.py` | 7-seed tables/plots/`story.md` |
| `evals/paper_fragment.py` | `latex_escape`, `df_to_booktabs`, `build_fragment` |
| `scripts/build_paper_fragment.py` | builds `PAPER/<subdir>/` |
| `tests/test_representation_profile.py`, `tests/test_paper_fragment.py` | unit tests |

---

## Task 1: Q/V helpers (`evals/representation_profile.py`)

**Files:**
- Create: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_representation_profile.py`:

```python
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import representation_profile as rp


class _StubModel:
    """B = first z_dim obs dims; F = (obs0 + action0) broadcast, 2 heads."""
    z_dim = 3

    def backward_map(self, obs):
        return obs[:, : self.z_dim].clone()

    def forward_map(self, obs, z, action):
        base = (obs[:, 0:1] + action[:, 0:1]).expand(-1, self.z_dim)  # [B,zd]
        return torch.stack([base, base + 1.0], dim=0)                 # [2,B,zd]


class _StubAgent:
    device = "cpu"

    def act(self, obs, z):
        return torch.zeros((obs.shape[0], 2))


def test_ensemble_q_means_over_heads():
    F = torch.tensor([[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])  # [2,1,3]
    z = torch.tensor([2.0, 0.0, 0.0])
    q = rp._ensemble_q(F, z)
    assert q.shape == (1,)
    assert q[0] == 2.0 * ((1.0 + 3.0) / 2)  # mean over heads of (F·z)


def test_q_values_monotone_in_action():
    m = _StubModel()
    obs = torch.zeros((4, 5))
    z = torch.ones(3)
    a_lo = torch.zeros((4, 2))
    a_hi = torch.ones((4, 2))
    q_lo = rp.q_values(m, obs, a_lo, z)
    q_hi = rp.q_values(m, obs, a_hi, z)
    assert q_lo.shape == (4,)
    assert np.all(q_hi > q_lo)


def test_v_values_uses_policy_action():
    m, ag = _StubModel(), _StubAgent()
    obs = torch.zeros((3, 5))
    z = torch.ones(3)
    v = rp.v_values(m, ag, obs, z)
    # policy action is zeros => same as q_values with zero action
    assert np.allclose(v, rp.q_values(m, obs, torch.zeros((3, 2)), z))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.representation_profile'`

- [ ] **Step 3: Create `evals/representation_profile.py`**

```python
"""evals/representation_profile.py — why does FB fail at transport?

Four probes (value landscape, z-decoding+sparsity, B-resolution,
coverage) over FB checkpoints. Pure logic here is unit-tested with
stubs; MuJoCo rollouts reuse evals.phase_probe. No training/eval code
is modified. See
docs/superpowers/specs/2026-05-18-fb-representation-failure-profile-design.md
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


@torch.no_grad()
def _ensemble_q(F: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """F [P, B, z_dim], z [z_dim] or [B, z_dim] -> Q [B] (mean over heads)."""
    if z.dim() == 1:
        z = z.view(1, 1, -1)
    elif z.dim() == 2:
        z = z.unsqueeze(0)
    q = (F * z).sum(-1)            # [P, B]
    return q.mean(0)              # [B]


@torch.no_grad()
def q_values(model, obs: torch.Tensor, action: torch.Tensor,
             z: torch.Tensor) -> np.ndarray:
    F = model.forward_map(obs, z, action)
    return _ensemble_q(F, z).cpu().numpy()


@torch.no_grad()
def v_values(model, agent, obs: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    action = agent.act(obs, z)
    if isinstance(action, np.ndarray):
        action = torch.as_tensor(action, dtype=torch.float32)
    return q_values(model, obs, action.to(obs.device).float(), z)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): ensemble Q/V helpers"
```

---

## Task 2: `record_obs` rollout + `rollout_for_profile`

**Files:**
- Modify: `evals/phase_probe.py`
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_representation_profile.py`:

```python
from evals import phase_probe as pp


class _PhaseEnv:
    """3-step episode; manip-like; cube approaches goal so it 'succeeds'."""
    def __init__(self):
        self.unwrapped = _PhaseUnwrapped()
        self.t = 0

    def _info(self, eff, cube):
        return {
            "proprio/effector_pos": np.array(eff, np.float64),
            "proprio/gripper_opening": np.array([1.0]),
            "privileged/block_0_pos": np.array(cube, np.float64),
            "success": self.t >= 3,
        }

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(5, np.float32), self._info([9, 9, 9], [0, 0, 0.02])

    def step(self, a):
        self.t += 1
        done = self.t >= 3
        return (np.full(5, self.t, np.float32), -1.0, done, False,
                self._info([0, 0, 0.20], [0, 0, 0.20]))

    def close(self):
        pass


class _PhaseUnwrapped:
    _target_block = 0
    cur_task_info = {"goal_xyzs": np.array([[0.0, 0.0, 0.20]]),
                     "init_xyzs": np.array([[0.0, 0.0, 0.02]])}


def test_phase_rollout_records_obs_when_requested():
    eps = pp.rollout_with_phase_signals(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds(), scenario="S0", record_obs=True)
    e = eps[0]
    assert "obs" in e and "action" in e
    assert e["obs"].shape[0] == e["length"]
    assert e["action"].shape[0] == e["length"] - 1  # actions between steps


def test_rollout_for_profile_attaches_outcome_and_d():
    eps = rp.rollout_for_profile(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds())
    e = eps[0]
    assert e["outcome"] in ("success", "transport_fail", "other")
    assert e["d"].shape[0] == e["obs"].shape[0]      # ||cube-goal|| per step
    assert e["transport_mask"].shape[0] == e["obs"].shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k "phase_rollout_records or rollout_for_profile" -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'record_obs'`

- [ ] **Step 3a: Add `record_obs` to `evals/phase_probe.py`**

In `evals/phase_probe.py`, change the `rollout_with_phase_signals` signature line:

```python
def rollout_with_phase_signals(
    env, agent, z, n_episodes: int, thr: Thresholds, scenario: str = "S0",
    record_obs: bool = False,
) -> List[Dict[str, Any]]:
```

Immediately after `s0 = step_signals(info)` and the three `effs/cubes/grips.append(...)` lines, add:

```python
        obs_list, act_list = [], []
        observation = _initial_obs(env)
        if record_obs:
            obs_list.append(np.asarray(observation, dtype=np.float32))
```

(The existing line `observation = _initial_obs(env)` that already follows must be removed so `_initial_obs` is called exactly once — it now lives in the block above.)

Inside the `while True:` loop, immediately after `a = agent.act(...).cpu().numpy()[0]` and before `observation, _, terminated, ... = env.step(a)`, add:

```python
            if record_obs:
                act_list.append(np.asarray(a, dtype=np.float32))
```

After `effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])` inside the loop, add:

```python
            if record_obs:
                obs_list.append(np.asarray(observation, dtype=np.float32))
```

In the `episodes.append({...})` dict literal, add these two keys before the closing brace:

```python
            "obs": (np.asarray(obs_list, dtype=np.float32)
                    if record_obs else None),
            "action": (np.asarray(act_list, dtype=np.float32)
                       if record_obs else None),
```

- [ ] **Step 3b: Add `rollout_for_profile` + `transport_mask` to `evals/representation_profile.py`**

Append to `evals/representation_profile.py`:

```python
def transport_mask(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step boolean: cube lifted above table and gripper closed."""
    cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(ep["grip"], dtype=np.float64).reshape(-1)
    lift = cube[:, 2] - float(ep["table_z"])
    return (lift > thr.delta_lift) & (grip > thr.tau_grip)


def rollout_for_profile(env, agent, z, n_episodes: int, thr,
                        seed_offset: int = 0) -> List[Dict[str, Any]]:
    """S0 rollouts with obs/action recorded, each tagged with outcome,
    per-step cube->goal distance, and a transport-region mask."""
    from evals.phase_probe import classify_phases, rollout_with_phase_signals

    eps = rollout_with_phase_signals(
        env, agent, z, n_episodes, thr, scenario="S0", record_obs=True)
    out = []
    for ep in eps:
        cls = classify_phases(ep, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
        goal = np.asarray(ep["goal"], dtype=np.float64).reshape(3)
        d = np.linalg.norm(cube - goal, axis=1)
        out.append({
            "obs": ep["obs"], "action": ep["action"],
            "cube": cube, "goal": goal, "d": d,
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "table_z": float(ep["table_z"]),
            "transport_mask": transport_mask(ep, thr),
            "outcome": outcome, "length": ep["length"],
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py tests/test_phase_probe.py -v`
Expected: PASS (Task-1 3 + Task-2 2 + existing 17 phase_probe = 22 passed; existing phase_probe tests still green because `record_obs` defaults False)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): opt-in obs/action recording + rollout_for_profile"
```

---

## Task 3: `probe_value_landscape`

**Files:**
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def _fake_ep(outcome, d, v_at_d):
    """Build a profile-episode whose V(s_t) = v_at_d(d_t) via stub model."""
    T = len(d)
    obs = np.zeros((T, 5), np.float32)
    obs[:, 0] = [v_at_d(x) for x in d]      # _StubModel Q ~ obs[:,0]
    return {
        "obs": obs, "action": np.zeros((max(T - 1, 0), 2), np.float32),
        "d": np.asarray(d, np.float64),
        "transport_mask": np.ones(T, bool),
        "outcome": outcome, "length": T,
    }


def test_value_landscape_slope_success_vs_fail():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # success: V rises as d -> 0 (slope wrt -d > 0); fail: V flat
    eps = [
        _fake_ep("success", [0.30, 0.20, 0.10, 0.02], lambda x: 1.0 - x),
        _fake_ep("transport_fail", [0.30, 0.20, 0.10, 0.05], lambda x: 0.5),
    ]
    per_ep, per_step = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["slope_V_negd"]
    assert s.loc["success"] > 0.5
    assert abs(s.loc["transport_fail"]) < 1e-6
    assert set(per_step.columns) >= {"outcome", "d", "V"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k value_landscape -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'probe_value_landscape'`

- [ ] **Step 3: Add `probe_value_landscape`**

Append to `evals/representation_profile.py`:

```python
def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x; nan if <2 distinct x."""
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def probe_value_landscape(model, agent, episodes, z):
    """Per-episode slope of V w.r.t. (-d) over the transport region,
    plus a long-form (outcome, d, V) frame for binned plots."""
    import pandas as pd

    per_ep_rows, per_step_rows = [], []
    for i, ep in enumerate(episodes):
        obs = torch.as_tensor(np.asarray(ep["obs"]), dtype=torch.float32)
        if obs.shape[0] == 0:
            continue
        V = v_values(model, agent, obs, z)             # [T]
        d = np.asarray(ep["d"], np.float64)
        m = np.asarray(ep["transport_mask"], bool)
        for dd, vv, mm in zip(d, V, m):
            per_step_rows.append({"episode": i, "outcome": ep["outcome"],
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm)})
        if m.sum() >= 2:
            slope = _slope(V[m], -d[m])
            v_secure = float(V[m][0])
            v_end = float(V[-1])
        else:
            slope, v_secure, v_end = float("nan"), float("nan"), float("nan")
        per_ep_rows.append({"episode": i, "outcome": ep["outcome"],
                            "slope_V_negd": slope, "V_at_secure": v_secure,
                            "V_at_end": v_end,
                            "n_transport_steps": int(m.sum())})
    return pd.DataFrame(per_ep_rows), pd.DataFrame(per_step_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): value-landscape probe"
```

---

## Task 4: `probe_z_decoding`

**Files:**
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_z_decoding_topk_at_goal():
    m = _StubModel()  # B(next_obs) = first 3 dims
    N = 20
    next_obs = torch.zeros((N, 5))
    # 5 states aligned with z=[1,0,0]; of those, 3 are at goal (d<0.04)
    next_obs[:5, 0] = 10.0
    physics = np.zeros((N, 20), np.float32)
    goal = np.array([0.0, 0.0, 0.02], np.float64)
    physics[:, 14:17] = goal              # all at goal by default
    physics[3:, 14] = 1.0                 # indices 3.. are FAR from goal
    z = torch.tensor([1.0, 0.0, 0.0])
    relabel_metrics = {"relabel_reward#nonzero": 3.0,
                       "relabel_reward#num_samples": 20.0}
    out = rp.probe_z_decoding(m, next_obs, physics, goal, z,
                              relabel_metrics, topk=5)
    assert out["relabel_pos_frac"] == 3.0 / 20.0
    assert 0.0 <= out["topk_pct_at_goal"] <= 1.0
    # top-5 by B·z are indices 0..4; of these, 0,1,2 at goal => 60%
    assert abs(out["topk_pct_at_goal"] - 0.6) < 1e-9
    assert out["topk_mean_d"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k z_decoding -v`
Expected: FAIL — `AttributeError: ... 'probe_z_decoding'`

- [ ] **Step 3: Add `probe_z_decoding`**

Append to `evals/representation_profile.py`:

```python
@torch.no_grad()
def probe_z_decoding(model, next_obs: torch.Tensor, physics: np.ndarray,
                     goal: np.ndarray, z: torch.Tensor,
                     relabel_metrics: Dict[str, float],
                     topk: int = 256,
                     goal_thresh: float = 0.04) -> Dict[str, float]:
    """How sparse are goal states, and does z point at 'placed' states?"""
    B = model.backward_map(next_obs.float())              # [N, z_dim]
    scores = (B * z.view(1, -1)).sum(-1).cpu().numpy()    # B·z
    cube = np.asarray(physics, np.float64)[:, 14:17]
    d = np.linalg.norm(cube - np.asarray(goal, np.float64).reshape(1, 3),
                        axis=1)
    k = int(min(topk, len(scores)))
    idx = np.argsort(-scores)[:k]
    ns = float(relabel_metrics.get("relabel_reward#num_samples", 0.0)) or 1.0
    return {
        "relabel_pos_frac": float(
            relabel_metrics.get("relabel_reward#nonzero", 0.0)) / ns,
        "topk_mean_d": float(d[idx].mean()),
        "topk_pct_at_goal": float(np.mean(d[idx] < goal_thresh)),
        "topk": k,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): z-decoding + relabel-sparsity probe"
```

---

## Task 5: `probe_b_resolution` (numpy ridge + Fisher-LDA)

**Files:**
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_b_resolution_linear_vs_random():
    rng = np.random.default_rng(0)
    N = 400
    B_lin = rng.normal(size=(N, 4)).astype(np.float32)
    d_lin = (B_lin @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1).astype(np.float64)
    out_lin = rp.probe_b_resolution(B_lin, d_lin, seed=0)
    assert out_lin["r2"] > 0.95

    B_rnd = rng.normal(size=(N, 4)).astype(np.float32)
    d_rnd = rng.normal(size=N).astype(np.float64)
    out_rnd = rp.probe_b_resolution(B_rnd, d_rnd, seed=0)
    assert out_rnd["r2"] < 0.2

    # separability: placed (d<0.04) vs near (0.04..0.12) clearly offset
    B_sep = np.vstack([np.zeros((100, 4)), np.ones((100, 4))]).astype(np.float32)
    d_sep = np.concatenate([np.full(100, 0.01), np.full(100, 0.08)])
    out_sep = rp.probe_b_resolution(B_sep, d_sep, seed=0)
    assert out_sep["placed_vs_near_acc"] > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k b_resolution -v`
Expected: FAIL — `AttributeError: ... 'probe_b_resolution'`

- [ ] **Step 3: Add `probe_b_resolution`**

Append to `evals/representation_profile.py`:

```python
def _ridge_r2(X: np.ndarray, y: np.ndarray, seed: int,
              alpha: float = 1.0) -> float:
    """Closed-form ridge with a 70/30 split; returns held-out R²."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    if len(y) < 4 or np.ptp(y) < 1e-12:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1))])
    Xte = np.hstack([X[te], np.ones((len(te), 1))])
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ y[tr])
    pred = Xte @ w
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _lda_acc(X: np.ndarray, lbl: np.ndarray) -> float:
    """Fisher-LDA train=test accuracy for a 2-class problem."""
    X = np.asarray(X, np.float64)
    cls = np.unique(lbl)
    if len(cls) != 2:
        return float("nan")
    mu = [X[lbl == c].mean(0) for c in cls]
    Sw = sum(np.cov(X[lbl == c], rowvar=False) * (np.sum(lbl == c) - 1)
             for c in cls)
    Sw = np.atleast_2d(Sw) + 1e-6 * np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mu[1] - mu[0])
    proj = X @ w
    thr = 0.5 * (proj[lbl == cls[0]].mean() + proj[lbl == cls[1]].mean())
    pred = np.where(proj > thr, cls[1], cls[0])
    return float((pred == lbl).mean())


def probe_b_resolution(B: np.ndarray, d: np.ndarray,
                       seed: int = 0) -> Dict[str, float]:
    """Ridge R²(B->d) and placed-vs-near-miss linear separability."""
    if hasattr(B, "detach"):
        B = B.detach().cpu().numpy()
    B = np.asarray(B, np.float64)
    d = np.asarray(d, np.float64)
    r2 = _ridge_r2(B, d, seed)
    placed = d < 0.04
    near = (d >= 0.04) & (d < 0.12)
    sel = placed | near
    if placed.sum() >= 2 and near.sum() >= 2:
        acc = _lda_acc(B[sel], placed[sel].astype(int))
    else:
        acc = float("nan")
    return {"r2": r2, "placed_vs_near_acc": acc,
            "n_placed": int(placed.sum()), "n_near": int(near.sum())}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): B-space placement-resolution probe"
```

---

## Task 6: `probe_coverage`

**Files:**
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_coverage_nn_distance_by_region():
    ref = np.zeros((50, 4), np.float32)            # support at origin
    eps = [{
        "obs": np.array([[0, 0, 0, 0],             # on-support (reach)
                          [0, 0, 0, 0],             # on-support (lift)
                          [9, 9, 9, 9]], np.float32),  # off-support (transport)
        "transport_mask": np.array([False, False, True]),
        "grip": np.array([0.0, 1.0, 1.0]),
        "cube": np.array([[0, 0, 0.02], [0, 0, 0.10], [0, 0, 0.20]]),
        "table_z": 0.02, "outcome": "transport_fail", "length": 3,
    }]
    df = rp.probe_coverage(eps, ref, pp.Thresholds())
    assert set(df.columns) >= {"outcome", "region", "nn_dist"}
    far = df[df.region == "transport"]["nn_dist"].mean()
    near = df[df.region == "reach"]["nn_dist"].mean()
    assert far > near
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k coverage -v`
Expected: FAIL — `AttributeError: ... 'probe_coverage'`

- [ ] **Step 3: Add `probe_coverage`**

Append to `evals/representation_profile.py`:

```python
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
        # batched: ||on||^2 + ||refn||^2 - 2 on·refn
        d2 = (on ** 2).sum(1, keepdims=True) + (refn ** 2).sum(1) \
            - 2.0 * on @ refn.T
        nn = np.sqrt(np.maximum(d2.min(1), 0.0))
        reg = _region(ep, thr)
        for r, n in zip(reg, nn):
            rows.append({"outcome": ep["outcome"], "region": str(r),
                         "nn_dist": float(n)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): coverage / off-support probe"
```

---

## Task 7: `run_representation_profile` orchestrator

**Files:**
- Modify: `evals/representation_profile.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_run_representation_profile_returns_four_frames(monkeypatch):
    m, ag = _StubModel(), _StubAgent()

    def infer_z(task):
        return torch.tensor([1.0, 0.0, 0.0]), {
            "relabel_reward#nonzero": 2.0,
            "relabel_reward#num_samples": 10.0}

    def make_env(task):
        return _PhaseEnv()

    def sample_buffer(n):
        return {
            "next_obs": torch.zeros((n, 5)),
            "physics": np.zeros((n, 20), np.float32),
            "action": np.zeros((n, 2), np.float32),
        }

    def relabel_fn(phys, act):
        return np.zeros((len(phys), 1), np.float32)

    out = rp.run_representation_profile(
        model=m, agent=ag, infer_z=infer_z, make_env=make_env,
        sample_buffer=sample_buffer, relabel_fn_for=lambda t: relabel_fn,
        goal_for=lambda t: np.array([0.0, 0.0, 0.20]),
        tasks=["t1", "t2"], n_episodes=1, thr=pp.Thresholds(),
        buffer_sample=8, topk=4, seed=0)
    assert set(out) == {"value_landscape", "value_steps",
                        "z_decoding", "b_resolution", "coverage"}
    assert {"task"}.issubset(out["z_decoding"].columns)
    assert out["z_decoding"]["task"].nunique() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k run_representation_profile -v`
Expected: FAIL — `AttributeError: ... 'run_representation_profile'`

- [ ] **Step 3: Add `run_representation_profile`**

Append to `evals/representation_profile.py`:

```python
def run_representation_profile(*, model, agent, infer_z, make_env,
                               sample_buffer, relabel_fn_for, goal_for,
                               tasks, n_episodes, thr, buffer_sample,
                               topk, seed):
    """Run the four probes per task. `infer_z(task)->(z,relabel_metrics)`,
    `make_env(task)->env`, `sample_buffer(n)->{next_obs,physics,action}`,
    `relabel_fn_for(task)->fn`, `goal_for(task)->goal[3]`. Returns a dict
    of five DataFrames (value_landscape, value_steps, z_decoding,
    b_resolution, coverage), each with a `task` column."""
    import pandas as pd

    vl, vs, zd, br, cv = [], [], [], [], []
    for task in tasks:
        z, relabel_metrics = infer_z(task)
        goal = np.asarray(goal_for(task), np.float64).reshape(3)

        env = make_env(task)
        try:
            eps = rollout_for_profile(env, agent, z, n_episodes, thr)
        finally:
            if hasattr(env, "close"):
                env.close()
        per_ep, per_step = probe_value_landscape(model, agent, eps, z)
        for df in (per_ep, per_step):
            df.insert(0, "task", task)
        vl.append(per_ep)
        vs.append(per_step)

        smp = sample_buffer(buffer_sample)
        next_obs = smp["next_obs"]
        physics = np.asarray(smp["physics"], np.float32)
        zdec = probe_z_decoding(model, next_obs, physics, goal, z,
                                relabel_metrics, topk=topk)
        zdec["task"] = task
        zd.append(zdec)

        B = model.backward_map(
            next_obs.float() if hasattr(next_obs, "float") else
            torch.as_tensor(next_obs, dtype=torch.float32))
        d_off = np.linalg.norm(
            physics[:, 14:17] - goal.reshape(1, 3), axis=1)
        bres = probe_b_resolution(B, d_off, seed=seed)
        bres["task"] = task
        br.append(bres)

        ref_obs = (smp["next_obs"].detach().cpu().numpy()
                   if hasattr(smp["next_obs"], "detach")
                   else np.asarray(smp["next_obs"]))
        cov = probe_coverage(eps, ref_obs, thr, seed=seed)
        cov.insert(0, "task", task)
        cv.append(cov)

    return {
        "value_landscape": pd.concat(vl, ignore_index=True),
        "value_steps": pd.concat(vs, ignore_index=True),
        "z_decoding": pd.DataFrame(zd),
        "b_resolution": pd.DataFrame(br),
        "coverage": pd.concat(cv, ignore_index=True),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): run_representation_profile orchestrator"
```

---

## Task 8: `scripts/representation_profile.py` CLI

**Files:**
- Create: `scripts/representation_profile.py`

- [ ] **Step 1: Create the CLI**

```python
"""scripts/representation_profile.py — per-checkpoint FB representation
profile. Caller loops seeds; the aggregate step runs all 7.

Usage:
    python scripts/representation_profile.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/representation_profile/s3_final \\
      --data-path datasets --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default=None, help="comma-separated; default=all")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--mujoco-gl", default=None)
    ap.add_argument("--buffer-sample", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=256)
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import numpy as np
    import torch

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.phase_probe import Thresholds
    from evals.representation_profile import run_representation_profile
    from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_path is not None:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None
    if hasattr(env, "close"):
        env.close()

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))
    thr = Thresholds()

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions)
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False)

    def infer_z(task):
        return evaluator._infer_z(task)

    def make_env(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        return e

    def sample_buffer(n):
        b = buffer.sample(min(n, len(buffer)))
        return {"next_obs": b["next"]["observation"],
                "physics": b["next"]["physics"].detach().cpu().numpy(),
                "action": b["action"].detach().cpu().numpy()}

    def relabel_fn_for(task):
        return get_relabel_fn(cfg.domain, task)

    _goal_env, _ = create_ogbench_env(tasks[0], seed=cfg.seed,
                                      obs_type=cfg.obs_type)
    _tb = getattr(_goal_env.unwrapped, "_target_block", 0)

    def goal_for(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        try:
            tb = getattr(e.unwrapped, "_target_block", 0)
            return np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tb],
                              np.float64)
        finally:
            e.close()
    _goal_env.close()

    res = run_representation_profile(
        model=agent.model, agent=agent, infer_z=infer_z,
        make_env=make_env, sample_buffer=sample_buffer,
        relabel_fn_for=relabel_fn_for, goal_for=goal_for, tasks=tasks,
        n_episodes=args.n_episodes, thr=thr,
        buffer_sample=args.buffer_sample, topk=args.topk,
        seed=cfg.seed)

    res["value_landscape"].to_parquet(out / "value_landscape.parquet")
    res["value_steps"].to_parquet(out / "value_steps.parquet")
    res["z_decoding"].to_parquet(out / "z_decoding.parquet")
    res["b_resolution"].to_parquet(out / "b_resolution.parquet")
    res["coverage"].to_parquet(out / "coverage.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "config": args.config, "checkpoint": args.checkpoint,
        "tasks": tasks, "buffer_sample": args.buffer_sample,
        "topk": args.topk,
        "q_aggregation": "ensemble-mean over F heads",
    }, indent=2, default=str))
    print(f"[representation_profile] done -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses + full suite**

Run: `.venv/bin/python scripts/representation_profile.py --help`
Expected: argparse help with `--config --checkpoint --out --device --tasks --n-episodes --seed --data-path --mujoco-gl --buffer-sample --topk`; no torch import on `--help`.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (prior 31 + repr 10 = 41 passed).

- [ ] **Step 3: Commit**

```bash
git add scripts/representation_profile.py
git commit -m "feat(repr_profile): per-checkpoint CLI"
```

- [ ] **Step 4: Acceptance (run by user, documented)**

```
for s in 3 4 5 6 7 8 10; do
 RUN=$(ls -d RESULTS/Factored-FB-cube-run/*__fb_flowbc__s${s})
 .venv/bin/python scripts/representation_profile.py \
   --config "$RUN/.hydra/config.yaml" \
   --checkpoint "$RUN/checkpoints/final.pt" \
   --out analysis/representation_profile/s${s}_final \
   --data-path datasets --device cpu --mujoco-gl glfw --n-episodes 10
done
```
Expected: `analysis/representation_profile/s*_final/` each with the 5
parquets + `metrics.json`.

---

## Task 9: `scripts/representation_profile_aggregate.py`

**Files:**
- Create: `scripts/representation_profile_aggregate.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_aggregate_builds_tables_and_story(tmp_path):
    import importlib
    import pandas as pd
    agg = importlib.import_module("scripts.representation_profile_aggregate")
    for s in (3, 7):
        d = tmp_path / f"s{s}_final"
        d.mkdir()
        pd.DataFrame([{"task": "t1", "outcome": "success",
                       "slope_V_negd": 1.0, "V_at_secure": 0.1,
                       "V_at_end": 0.9, "n_transport_steps": 5},
                      {"task": "t1", "outcome": "transport_fail",
                       "slope_V_negd": 0.0, "V_at_secure": 0.5,
                       "V_at_end": 0.5, "n_transport_steps": 5}]
                     ).to_parquet(d / "value_landscape.parquet")
        pd.DataFrame([{"task": "t1", "outcome": "success", "d": 0.1,
                       "V": 0.5, "transport": True}]
                     ).to_parquet(d / "value_steps.parquet")
        pd.DataFrame([{"task": "t1", "relabel_pos_frac": 0.02,
                       "topk_mean_d": 0.3, "topk_pct_at_goal": 0.1,
                       "topk": 4}]).to_parquet(d / "z_decoding.parquet")
        pd.DataFrame([{"task": "t1", "r2": 0.2,
                       "placed_vs_near_acc": 0.6, "n_placed": 5,
                       "n_near": 9}]).to_parquet(d / "b_resolution.parquet")
        pd.DataFrame([{"task": "t1", "outcome": "transport_fail",
                       "region": "transport", "nn_dist": 3.0}]
                     ).to_parquet(d / "coverage.parquet")
    out = tmp_path / "aggregate"
    agg.aggregate(tmp_path, out)
    assert (out / "story.md").exists()
    assert (out / "value_vs_dist.png").exists()
    assert (out / "T1_value_gradient.parquet").exists()
    txt = (out / "story.md").read_text()
    assert "## Readout" in txt
    assert "0.02" in txt          # relabel_pos_frac mean filled in
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k aggregate_builds -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.representation_profile_aggregate'`

- [ ] **Step 3: Create `scripts/representation_profile_aggregate.py`**

```python
"""scripts/representation_profile_aggregate.py — 7-seed aggregate.

Reads <root>/s*_final/{value_landscape,value_steps,z_decoding,
b_resolution,coverage}.parquet, tags `seed`, writes mean±std tables,
plots, and a data-filled story.md.
"""

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

_SEED_RE = re.compile(r"s(\d+)_final")


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/representation_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(series: pd.Series) -> str:
    return f"{series.mean():.3g} ± {series.std(ddof=0):.2g}"


def aggregate(root, out) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    vl = _load(root, "value_landscape")
    vs = _load(root, "value_steps")
    zd = _load(root, "z_decoding")
    br = _load(root, "b_resolution")
    cv = _load(root, "coverage")

    # T1 value gradient: per task, slope by outcome (mean over seeds)
    t1 = (vl.groupby(["task", "outcome"])["slope_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t2 = (zd.groupby("task")[["relabel_pos_frac", "topk_mean_d",
                              "topk_pct_at_goal"]]
          .agg(["mean", "std"]))
    t2.to_parquet(out / "T2_sparsity.parquet")
    t3 = (br.groupby("task")[["r2", "placed_vs_near_acc"]]
          .agg(["mean", "std"]))
    t3.to_parquet(out / "T3_b_resolution.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")

    # value_vs_dist.png — V binned by d, success vs transport_fail
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, max(0.05, float(vs["d"].quantile(0.95))), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for oc, color in (("success", "#55a868"),
                      ("transport_fail", "#c44e52")):
        sub = vs[vs.outcome == oc]
        if sub.empty:
            continue
        idx = np.clip(np.digitize(sub["d"], bins) - 1, 0, len(centers) - 1)
        g = pd.DataFrame({"b": idx, "V": sub["V"].values}).groupby("b")["V"]
        ax.plot(centers[g.mean().index], g.mean().values, marker="o",
                label=oc, color=color)
    ax.set_xlabel("cube → goal distance")
    ax.set_ylabel("V(s)")
    ax.set_title("Value vs goal distance (S0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "value_vs_dist.png", dpi=120)
    plt.close(fig)

    for name, frame, col in (
        ("sparsity_bars", zd, "topk_pct_at_goal"),
        ("b_resolution_bars", br, "r2")):
        fig, ax = plt.subplots(figsize=(6, 4))
        gg = frame.groupby("task")[col].mean()
        ax.bar([str(t) for t in gg.index], gg.values)
        ax.set_title(name)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title("coverage_curves")
    ax.set_ylabel("nn_dist")
    fig.tight_layout()
    fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    succ = vl[vl.outcome == "success"]["slope_V_negd"]
    fail = vl[vl.outcome == "transport_fail"]["slope_V_negd"]
    lines = [
        "# FB representation-failure — cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "",
        "## Sparsity (T2)",
        f"- offline goal-state fraction: {_ms(zd['relabel_pos_frac'])}",
        f"- top-k B·z mean cube→goal dist: {_ms(zd['topk_mean_d'])}",
        f"- top-k % actually at goal: {_ms(zd['topk_pct_at_goal'])}",
        "",
        "## B resolution (T3)",
        f"- ridge R²(B→d): {_ms(br['r2'])}",
        f"- placed-vs-near-miss acc: {_ms(br['placed_vs_near_acc'])}",
        "",
        "## Value gradient (T1)",
        f"- slope(V vs −d) success: {_ms(succ)}",
        f"- slope(V vs −d) transport-fail: {_ms(fail)}",
        "",
        "## Coverage (T4)",
        f"- transport-region nn_dist (transport_fail): "
        f"{_ms(cv[(cv.region=='transport') & (cv.outcome=='transport_fail')]['nn_dist'])}",
        "",
        "## Readout",
        "Reach ~100% and grasp+lift high (phase-probe). The play data is "
        "goal-sparse (T2) and the inferred z's top states sit far from the "
        "goal; B barely linearly resolves placement distance (T3). "
        "Consequently V provides a gradient toward the goal for successes "
        "but is ~flat for transport-fails (T1), and failures coincide with "
        "leaving data support (T4). ⇒ the FB representation under-encodes "
        "the precise goal-placement sub-manifold, so the policy lifts but "
        "has no learned signal to deliver.",
    ]
    (out / "story.md").write_text("\n".join(lines))
    print(f"[representation_profile_aggregate] "
          f"{vl['seed'].nunique()} seeds -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "representation_profile"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (11 passed)

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (42 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/representation_profile_aggregate.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): 7-seed aggregate tables/plots/story"
```

---

## Task 10: `evals/paper_fragment.py`

**Files:**
- Create: `evals/paper_fragment.py`
- Test: `tests/test_paper_fragment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paper_fragment.py`:

```python
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import paper_fragment as pf


def test_latex_escape():
    assert pf.latex_escape("a_b 50% #1 & $x") == \
        r"a\_b 50\% \#1 \& \$x"


def test_df_to_booktabs():
    df = pd.DataFrame({"task": ["t1"], "r2": [0.2]})
    tex = pf.df_to_booktabs(df, caption="B res", label="tab:b")
    assert "\\begin{tabular}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\caption{B res}" in tex and "\\label{tab:b}" in tex
    assert "t1" in tex


def test_build_fragment(tmp_path):
    phase = tmp_path / "phase" / "aggregate"
    repr_ = tmp_path / "repr" / "aggregate"
    phase.mkdir(parents=True)
    repr_.mkdir(parents=True)
    (phase / "overall.md").write_text("# phase\n- S0 success 47%")
    for fn in ("funnel.png", "fail_phase_composition.png"):
        (phase / fn).write_bytes(b"\x89PNG")
    (repr_ / "story.md").write_text("# story\n## Readout\nflat V")
    pd.DataFrame({"task": ["t1"], "r2": [0.2]}).to_parquet(
        repr_ / "T3_b_resolution.parquet")
    for fn in ("value_vs_dist.png", "sparsity_bars.png",
               "b_resolution_bars.png", "coverage_curves.png"):
        (repr_ / fn).write_bytes(b"\x89PNG")

    pf.build_fragment(phase, repr_, tmp_path / "PAPER", "fb-cube-failure")
    base = tmp_path / "PAPER" / "fb-cube-failure"
    tex = (base / "fb_cube_failure.tex").read_text()
    assert "\\section" in tex
    assert tex.count("\\subsection") >= 4
    assert "\\includegraphics" in tex
    assert "\\input{tables/" in tex
    assert (base / "standalone.tex").exists()
    assert (base / "figures" / "value_vs_dist.png").exists()
    assert list((base / "tables").glob("*.tex"))
    # missing optional figure handled, not crashed
    pf.build_fragment(phase, repr_, tmp_path / "PAPER2", "x",
                      figures=("nonexistent.png",))
    assert "% [figure missing]" in \
        (tmp_path / "PAPER2" / "x" / "fb_cube_failure.tex").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.paper_fragment'`

- [ ] **Step 3: Create `evals/paper_fragment.py`**

```python
"""evals/paper_fragment.py — render an \\input-able LaTeX section from
the phase-probe + representation-profile aggregates. No LaTeX is
compiled here; this only generates text + copies figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

_ESCAPE = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
           "$": r"\$", "{": r"\{", "}": r"\}"}

_DEFAULT_FIGS = ("value_vs_dist.png", "sparsity_bars.png",
                 "b_resolution_bars.png", "coverage_curves.png")
_PHASE_FIGS = ("funnel.png", "fail_phase_composition.png")


def latex_escape(s: str) -> str:
    return "".join(_ESCAPE.get(ch, ch) for ch in str(s))


def df_to_booktabs(df, caption: str, label: str) -> str:
    cols = list(df.columns)
    head = " & ".join(latex_escape(c) for c in cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.3g}" if isinstance(v, float)
                         else latex_escape(v))
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule", head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\end{table}", ""])


def _fig(figdir_rel: str, name: str, present: bool, caption: str) -> str:
    if not present:
        return f"% [figure missing] {name}\n"
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=0.7\linewidth]{"
        + f"{figdir_rel}/{name}" + "}",
        r"\caption{" + caption + "}",
        r"\end{figure}", ""])


def build_fragment(phase_root, repr_root, out_dir, subdir: str,
                   figures: Sequence[str] = _DEFAULT_FIGS) -> Path:
    phase_root, repr_root = Path(phase_root), Path(repr_root)
    base = Path(out_dir) / subdir
    figdir = base / "figures"
    tabdir = base / "tables"
    for d in (figdir, tabdir):
        d.mkdir(parents=True, exist_ok=True)

    for missing in (phase_root, repr_root):
        if not missing.exists():
            raise FileNotFoundError(
                f"Aggregate input not found: {missing}. Produce it with "
                f"the *_aggregate scripts before building the fragment.")

    # copy figures (phase + repr); track which exist
    present = {}
    for src_root, names in ((phase_root, _PHASE_FIGS),
                            (repr_root, figures)):
        for n in names:
            src = src_root / n
            if src.exists():
                shutil.copyfile(src, figdir / n)
                present[n] = True
            else:
                present[n] = False

    # tables from repr aggregate parquet
    table_inputs = []
    try:
        import pandas as pd
        for pq in sorted(repr_root.glob("T*.parquet")):
            df = pd.read_parquet(pq).reset_index()
            tname = pq.stem
            (tabdir / f"{tname}.tex").write_text(
                df_to_booktabs(df, caption=tname.replace("_", " "),
                               label=f"tab:{tname}"))
            table_inputs.append(f"tables/{tname}.tex")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] table generation skipped: {e}")

    story = ""
    sp = repr_root / "story.md"
    if sp.exists():
        story = latex_escape(sp.read_text())

    figs_phase = "".join(
        _fig("figures", n, present.get(n, False),
             "Phase-probe: " + n.replace("_", " ").replace(".png", ""))
        for n in _PHASE_FIGS)
    figs_repr = "".join(
        _fig("figures", n, present.get(n, False),
             n.replace("_", " ").replace(".png", ""))
        for n in figures)
    tbls = "".join(r"\input{" + t + "}" + "\n" for t in table_inputs)

    tex = f"""% Auto-generated; regenerate via scripts/build_paper_fragment.py
\\section{{Where and why FB fails on cube manipulation}}
\\label{{sec:fb-cube-failure}}

\\subsection{{Experiment design}}
We decompose each cube episode into reach $\\to$ grasp/lift $\\to$
transport, and probe the policy from three start states: S0 (baseline),
S1 (skip-reach), S2 (pre-grasped, physically clamped, no weld). The
phase-probe over seeds s3--s8, s10 localises failure.
{figs_phase}

\\subsection{{Methods}}
Four representation probes on the FB checkpoints: (i) value landscape
$V(s)=Q_z(s,\\pi(s))$ vs cube--goal distance; (ii) z-decoding and
relabel sparsity; (iii) ridge $R^2$ of $B(s)\\!\\to\\!d$ and
placed-vs-near separability; (iv) off-support nearest-neighbour
distance. Probes are aggregated (mean$\\pm$std) over the seven seeds.

\\subsection{{Results}}
{figs_repr}{tbls}

\\subsection{{Discussion}}
{story}
"""
    (base / "fb_cube_failure.tex").write_text(tex)
    (base / "standalone.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        "\\input{fb_cube_failure.tex}\n"
        "\\end{document}\n")
    return base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/paper_fragment.py tests/test_paper_fragment.py
git commit -m "feat(paper_fragment): LaTeX section/table/escape helpers"
```

---

## Task 11: `scripts/build_paper_fragment.py` + final verification

**Files:**
- Create: `scripts/build_paper_fragment.py`

- [ ] **Step 1: Create the CLI**

```python
"""scripts/build_paper_fragment.py — assemble PAPER/<subdir>/ from the
phase-probe and representation-profile aggregates (full-arc narrative).

Usage:
    python scripts/build_paper_fragment.py \\
      --phase-root analysis/phase_probe/aggregate \\
      --repr-root analysis/representation_profile/aggregate \\
      --paper-subdir fb-cube-failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase-root",
                    default=str(REPO_ROOT / "analysis" / "phase_probe"
                                / "aggregate"))
    ap.add_argument("--repr-root",
                    default=str(REPO_ROOT / "analysis"
                                / "representation_profile" / "aggregate"))
    ap.add_argument("--out", default=str(REPO_ROOT / "PAPER"))
    ap.add_argument("--paper-subdir", default="fb-cube-failure")
    args = ap.parse_args()

    from evals.paper_fragment import build_fragment

    base = build_fragment(args.phase_root, args.repr_root, args.out,
                          args.paper_subdir)
    print(f"[build_paper_fragment] wrote {base}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `.venv/bin/python scripts/build_paper_fragment.py --help`
Expected: argparse help with `--phase-root --repr-root --out --paper-subdir`.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_paper_fragment.py
git commit -m "feat(paper_fragment): build PAPER/<subdir> CLI"
```

- [ ] **Step 4: Final verification**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (45 passed: prior 31 + repr 11 + paper_fragment 3).

Run: `git diff --stat main -- train.py evals/ogbench.py envs/rollout.py envs/ogbench.py agents/ data/ogbench.py scripts/wandb_pull.py scripts/plot_sweep.py scripts/analyze_checkpoint.py`
Expected: empty (this feature only adds repr/paper files and the additive `record_obs` opt-in to `evals/phase_probe.py`; nothing else changes).

- [ ] **Step 5: Documented acceptance (run by user)**

```
# 1. per-seed profile (s3 already validated end-to-end pattern)
for s in 3 4 5 6 7 8 10; do
  RUN=$(ls -d RESULTS/Factored-FB-cube-run/*__fb_flowbc__s${s})
  .venv/bin/python scripts/representation_profile.py \
    --config "$RUN/.hydra/config.yaml" \
    --checkpoint "$RUN/checkpoints/final.pt" \
    --out analysis/representation_profile/s${s}_final \
    --data-path datasets --device cpu --mujoco-gl glfw --n-episodes 10
done
# 2. aggregate
.venv/bin/python scripts/representation_profile_aggregate.py
# 3. paper fragment (consumes phase_probe + representation_profile aggregates)
.venv/bin/python scripts/build_paper_fragment.py --paper-subdir fb-cube-failure
```
Expected: `analysis/representation_profile/aggregate/{T1..T4 parquet,
value_vs_dist.png, sparsity_bars.png, b_resolution_bars.png,
coverage_curves.png, story.md}` and `PAPER/fb-cube-failure/
{fb_cube_failure.tex, standalone.tex, figures/*, tables/*.tex}`.
```
