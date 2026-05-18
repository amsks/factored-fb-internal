# Cube Phase-Failure Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evals/phase_probe.py` + `scripts/phase_probe.py` to test whether FB's dominant cube failure is grasping/pick-up, via M1 (observational phase decomposition) and M2 (counterfactual S0/S1/S2 start states).

**Architecture:** One unit-tested library (`evals/phase_probe.py`) plus a thin Linux-runnable CLI (`scripts/phase_probe.py`) mirroring `scripts/analyze_checkpoint.py`. Pure phase-classification logic is TDD-tested; MuJoCo state surgery is contract-tested with a fake env; real-physics validation is a documented Linux acceptance run. Additive only — no training/eval code modified.

**Tech Stack:** Python 3.11, numpy, pandas/pyarrow, matplotlib (Agg), torch, mujoco (OGBench manipspace), pytest + monkeypatch.

**Spec:** `docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md`

**Test command:** `.venv/bin/python -m pytest <path> -v`

---

## File Structure

| File | Responsibility |
|---|---|
| `evals/phase_probe.py` | `Thresholds`, `PHASE_ORDER`, `N_SETTLE`; `classify_phases` (pure); `step_signals`/`episode_goal`/`episode_table_z`/`ensure_manip_env`; `rollout_with_phase_signals`; `apply_scenario`; `run_phase_probe`; `plot_phase_histogram`, `plot_scenario_success`, `write_summary_md` |
| `scripts/phase_probe.py` | CLI glue (load checkpoint, build evaluator for `_infer_z`, run probe, write outputs) |
| `tests/test_phase_probe.py` | Unit tests: classification, scenario mutation contract, aggregation, plots |

Reused unmodified: `evals.analysis` (`load_cfg`/`build_env_and_agent`/`load_checkpoint`), `evals.ogbench.OGBenchEvaluator._infer_z`, `data.ogbench.load_ogbench_dataset`, `envs.ogbench.create_ogbench_env`.

**Verified env facts (OGBench manipspace, mujoco 3.7):**
- `info` from `reset`/`step` contains `proprio/effector_pos` (3,), `proprio/gripper_opening` (1,), `privileged/block_0_pos` (3,), and `success` (bool).
- `info['privileged/target_block_pos']` is **None** in singletask — do NOT use.
- Goal: `env.unwrapped.cur_task_info['goal_xyzs'][tb]`; resting height: `env.unwrapped.cur_task_info['init_xyzs'][tb][2]`; `tb = getattr(env.unwrapped, '_target_block', 0)`.
- State surgery handles: `env.unwrapped._model`, `env.unwrapped._data`, `._pinch_site_id`, `._gripper_actuator_ids`; cube joint via `_data.joint('object_joint_0').qpos[:3]`; effector via `_data.site_xpos[_pinch_site_id]`. `gripper_opening ≈ 0` open, `→1` closed.

---

## Task 1: `Thresholds` + `classify_phases` (pure)

**Files:**
- Create: `evals/phase_probe.py`
- Test: `tests/test_phase_probe.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phase_probe.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import phase_probe as pp


def _sig(eff, cube, grip, success, goal=(1.0, 1.0, 0.0), table_z=0.02):
    return {
        "eff": np.asarray(eff, dtype=np.float32),
        "cube": np.asarray(cube, dtype=np.float32),
        "grip": np.asarray(grip, dtype=np.float32),
        "goal": np.asarray(goal, dtype=np.float32),
        "success": bool(success),
        "table_z": float(table_z),
    }


def test_full_success_episode():
    T = 8
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T)            # closed
    cube[:, 2] = 0.10            # lifted well above table_z=0.02
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"
    assert out["fail_phase"] == "none"
    assert out["length"] == T


def test_reach_but_no_secure():
    T = 6
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.zeros(T)           # never closes
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["reached"] and not out["secured"] and not out["success"]
    assert out["furthest_phase"] == "reached"
    assert out["fail_phase"] == "grasp"


def test_never_reach():
    T = 5
    eff = np.zeros((T, 3)); cube = np.full((T, 3), 9.0)
    grip = np.zeros(T)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert not out["reached"]
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"


def test_secure_but_no_success():
    T = 7
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T); cube[:, 2] = 0.10
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] and not out["success"]
    assert out["furthest_phase"] == "secured"
    assert out["fail_phase"] == "transport"


def test_monotone_forcing_success_implies_lower():
    T = 4
    eff = np.full((T, 3), 9.0); cube = np.zeros((T, 3))   # raw reach False
    grip = np.zeros(T)                                    # raw secure False
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached_raw"] is False and out["secured_raw"] is False
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"


def test_k_steps_window_required():
    T = 10
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3)); grip = np.ones(T)
    cube[:, 2] = 0.02            # at table; not lifted
    cube[3:5, 2] = 0.10          # lifted only 2 steps (< k=5)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] is False


def test_zero_length_episode():
    z = np.zeros((0, 3))
    out = pp.classify_phases(_sig(z, z, np.zeros(0), success=False),
                             pp.Thresholds())
    assert out["length"] == 0
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"
    assert np.isnan(out["min_eff_cube_dist"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.phase_probe'`

- [ ] **Step 3: Create `evals/phase_probe.py` with `Thresholds` + `classify_phases`**

Create `evals/phase_probe.py`:

```python
"""evals/phase_probe.py — test where FB fails in a cube episode.

M1: classify normal rollouts into reach/secure/success phases.
M2: counterfactual start states (S0 baseline / S1 skip-reach /
S2 pre-grasped) to isolate the grasp bottleneck.

See docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md.
Pure logic here is unit-tested; MuJoCo physics is validated by a
documented Linux acceptance run. No training/eval code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PHASE_ORDER = ["none", "reached", "secured", "success"]
N_SETTLE = 5  # MuJoCo steps to clamp the cube in S2 (module constant; YAGNI)


@dataclass
class Thresholds:
    eps_reach: float = 0.06   # m, hand-to-cube distance for "reached"
    delta_lift: float = 0.03  # m, cube height above table for "lifted"
    k_steps: int = 5          # consecutive steps the lift must hold
    tau_grip: float = 0.5     # gripper_opening > tau => "closed"


def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify_phases(signals: Dict[str, Any], thr: Thresholds) -> Dict[str, Any]:
    """Classify ONE episode's per-step signals into phase milestones.

    signals: eff[T,3], cube[T,3], grip[T], goal[3], success(bool),
             table_z(float).
    Returns the milestones, monotone furthest_phase/fail_phase, and
    diagnostics. Handles T == 0.
    """
    eff = np.asarray(signals["eff"], dtype=np.float64).reshape(-1, 3)
    cube = np.asarray(signals["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(signals["grip"], dtype=np.float64).reshape(-1)
    goal = np.asarray(signals["goal"], dtype=np.float64).reshape(3)
    table_z = float(signals["table_z"])
    success = bool(signals["success"])
    T = eff.shape[0]

    if T == 0:
        return {
            "reached": success, "secured": success, "success": success,
            "reached_raw": False, "secured_raw": False,
            "furthest_phase": "success" if success else "none",
            "fail_phase": "none" if success else "reach",
            "min_eff_cube_dist": float("nan"),
            "max_cube_lift": float("nan"),
            "final_cube_goal_dist": float("nan"),
            "length": 0,
        }

    eff_cube_dist = np.linalg.norm(eff - cube, axis=1)
    eff_cube_xy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - table_z

    reached_raw = bool(eff_cube_dist.min() < thr.eps_reach)
    secured_mask = (
        (lift > thr.delta_lift)
        & (grip > thr.tau_grip)
        & (eff_cube_xy < thr.eps_reach)
    )
    secured_raw = bool(_max_true_run(secured_mask) >= thr.k_steps)

    # Monotone ladder: success => secured => reached.
    secured = secured_raw or success
    reached = reached_raw or secured

    if success:
        furthest, fail = "success", "none"
    elif secured:
        furthest, fail = "secured", "transport"
    elif reached:
        furthest, fail = "reached", "grasp"
    else:
        furthest, fail = "none", "reach"

    return {
        "reached": reached, "secured": secured, "success": success,
        "reached_raw": reached_raw, "secured_raw": secured_raw,
        "furthest_phase": furthest, "fail_phase": fail,
        "min_eff_cube_dist": float(eff_cube_dist.min()),
        "max_cube_lift": float(lift.max()),
        "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal)),
        "length": int(T),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/test_phase_probe.py
git commit -m "feat(phase_probe): Thresholds + pure classify_phases"
```

---

## Task 2: signal extraction helpers

**Files:**
- Modify: `evals/phase_probe.py`
- Test: `tests/test_phase_probe.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_phase_probe.py`:

```python
import pytest


class _FakeUnwrapped:
    def __init__(self):
        self._data = object()
        self._pinch_site_id = 0
        self._target_block = 0
        self.cur_task_info = {
            "goal_xyzs": np.array([[0.4, -0.1, 0.02]]),
            "init_xyzs": np.array([[0.42, 0.10, 0.0199]]),
        }


class _FakeEnv:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


def test_step_signals_reads_info_keys():
    info = {
        "proprio/effector_pos": np.array([0.1, 0.2, 0.3]),
        "proprio/gripper_opening": np.array([0.9]),
        "privileged/block_0_pos": np.array([0.4, 0.5, 0.02]),
        "success": False,
    }
    s = pp.step_signals(info)
    assert np.allclose(s["eff"], [0.1, 0.2, 0.3])
    assert np.allclose(s["cube"], [0.4, 0.5, 0.02])
    assert s["grip"] == pytest.approx(0.9)


def test_episode_goal_and_table_z_from_cur_task_info():
    env = _FakeEnv(_FakeUnwrapped())
    assert np.allclose(pp.episode_goal(env), [0.4, -0.1, 0.02])
    assert pp.episode_table_z(env) == pytest.approx(0.0199)


def test_ensure_manip_env_rejects_non_manip():
    class Bare:
        pass

    with pytest.raises(RuntimeError, match="cube-only"):
        pp.ensure_manip_env(_FakeEnv(Bare()))
    # a manip-like env passes silently
    pp.ensure_manip_env(_FakeEnv(_FakeUnwrapped()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -k "step_signals or episode_goal or ensure_manip" -v`
Expected: FAIL — `AttributeError: module 'evals.phase_probe' has no attribute 'step_signals'`

- [ ] **Step 3: Add helpers to `evals/phase_probe.py`**

Append to `evals/phase_probe.py`:

```python
def step_signals(info: Dict[str, Any]) -> Dict[str, Any]:
    """Per-step signals from an OGBench manipspace `info` dict."""
    return {
        "eff": np.asarray(info["proprio/effector_pos"], dtype=np.float64),
        "cube": np.asarray(info["privileged/block_0_pos"], dtype=np.float64),
        "grip": float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]),
        "success": bool(info.get("success", False)),
    }


def _target_block(env) -> int:
    return int(getattr(env.unwrapped, "_target_block", 0) or 0)


def episode_goal(env) -> np.ndarray:
    tb = _target_block(env)
    return np.asarray(
        env.unwrapped.cur_task_info["goal_xyzs"][tb], dtype=np.float64
    )


def episode_table_z(env) -> float:
    tb = _target_block(env)
    return float(env.unwrapped.cur_task_info["init_xyzs"][tb][2])


def ensure_manip_env(env) -> None:
    """Raise a clear error if this is not an OGBench cube manip env."""
    u = getattr(env, "unwrapped", None)
    if u is None or not all(
        hasattr(u, a) for a in ("_data", "_pinch_site_id", "cur_task_info")
    ):
        raise RuntimeError(
            "phase_probe is cube-only: env.unwrapped lacks manipspace "
            "handles (_data/_pinch_site_id/cur_task_info)."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/test_phase_probe.py
git commit -m "feat(phase_probe): signal-extraction helpers"
```

---

## Task 3: `apply_scenario` (S0/S1/S2 state surgery)

**Files:**
- Modify: `evals/phase_probe.py`
- Test: `tests/test_phase_probe.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_phase_probe.py`:

```python
class _FakeJoint:
    def __init__(self):
        self.qpos = np.zeros(7)


class _FakeData:
    def __init__(self):
        self.site_xpos = np.array([[0.5, 0.6, 0.7]])  # pinch site id 0
        self.ctrl = np.zeros(8)
        self._joint = _FakeJoint()

    def joint(self, name):
        assert name == "object_joint_0"
        return self._joint


class _FakeManip:
    def __init__(self):
        self._model = object()
        self._data = _FakeData()
        self._pinch_site_id = 0
        self._gripper_actuator_ids = np.array([6, 7])
        self._target_block = 0
        self.cur_task_info = {
            "goal_xyzs": np.array([[0.4, -0.1, 0.02]]),
            "init_xyzs": np.array([[0.42, 0.10, 0.0199]]),
        }


class _FakeMujoco:
    def __init__(self):
        self.forward_calls = 0
        self.step_calls = 0

    def mj_forward(self, model, data):
        self.forward_calls += 1

    def mj_step(self, model, data):
        self.step_calls += 1


def test_apply_scenario_s0_is_noop(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    before = env.unwrapped._data._joint.qpos.copy()
    pp.apply_scenario(env, "S0")
    assert np.allclose(env.unwrapped._data._joint.qpos, before)
    assert fake_mj.step_calls == 0


def test_apply_scenario_s1_moves_cube_to_hand_xy(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    pp.apply_scenario(env, "S1")
    q = env.unwrapped._data._joint.qpos
    assert q[0] == pytest.approx(0.5) and q[1] == pytest.approx(0.6)
    assert q[2] == pytest.approx(0.0199)        # table_z, not hand z
    assert fake_mj.forward_calls == 1
    assert fake_mj.step_calls == 0


def test_apply_scenario_s2_grasps_and_settles(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    pp.apply_scenario(env, "S2")
    q = env.unwrapped._data._joint.qpos
    assert np.allclose(q[:3], [0.5, 0.6, 0.7])  # cube at effector
    assert np.all(env.unwrapped._data.ctrl[[6, 7]] == 255.0)
    assert fake_mj.step_calls == pp.N_SETTLE
    assert fake_mj.forward_calls >= 1


def test_apply_scenario_unknown_raises(monkeypatch):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())
    with pytest.raises(ValueError, match="unknown scenario"):
        pp.apply_scenario(_FakeEnv(_FakeManip()), "SX")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -k apply_scenario -v`
Expected: FAIL — `AttributeError: module 'evals.phase_probe' has no attribute 'apply_scenario'`

- [ ] **Step 3: Add `apply_scenario` to `evals/phase_probe.py`**

Add this import near the top of `evals/phase_probe.py` (after `import numpy as np`):

```python
import mujoco
```

Append to `evals/phase_probe.py`:

```python
SCENARIOS = ["S0", "S1", "S2"]


def apply_scenario(env, scenario: str) -> None:
    """Mutate post-reset MuJoCo state for a counterfactual start.

    S0: no-op. S1: cube under the hand at table height, gripper open.
    S2: cube at the effector, gripper closed, settle N_SETTLE steps so
    the fingers physically clamp it (no weld). The task goal/target
    mocap is never touched.
    """
    if scenario == "S0":
        return

    u = env.unwrapped
    model, data = u._model, u._data
    eff = np.asarray(data.site_xpos[u._pinch_site_id], dtype=np.float64).copy()
    joint = data.joint("object_joint_0")

    if scenario == "S1":
        table_z = episode_table_z(env)
        joint.qpos[0] = eff[0]
        joint.qpos[1] = eff[1]
        joint.qpos[2] = table_z
        mujoco.mj_forward(model, data)
        return

    if scenario == "S2":
        joint.qpos[:3] = eff
        mujoco.mj_forward(model, data)
        data.ctrl[u._gripper_actuator_ids] = 255.0
        for _ in range(N_SETTLE):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return

    raise ValueError(f"unknown scenario {scenario!r} (expected one of {SCENARIOS})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/test_phase_probe.py
git commit -m "feat(phase_probe): S0/S1/S2 counterfactual state surgery"
```

---

## Task 4: `rollout_with_phase_signals`

**Files:**
- Modify: `evals/phase_probe.py`
- Test: `tests/test_phase_probe.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_phase_probe.py`:

```python
import torch


class _RolloutEnv:
    """3-step episodes; cube fixed, effector reaches it; manip-like."""

    def __init__(self):
        self.unwrapped = _FakeManip()
        self.t = 0

    def _info(self, eff):
        return {
            "proprio/effector_pos": np.array(eff, dtype=np.float64),
            "proprio/gripper_opening": np.array([1.0]),
            "privileged/block_0_pos": np.array([0.0, 0.0, 0.10]),
            "success": self.t >= 3,
        }

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(4, dtype=np.float32), self._info([9.0, 9.0, 9.0])

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return (np.zeros(4, dtype=np.float32), -1.0, done, False,
                self._info([0.0, 0.0, 0.10]))

    def close(self):
        pass


class _RolloutAgent:
    device = "cpu"

    def act(self, obs=None, z=None):
        return torch.zeros((1, 4))


def test_rollout_with_phase_signals_shapes(monkeypatch):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())
    eps = pp.rollout_with_phase_signals(
        _RolloutEnv(), _RolloutAgent(), z=torch.zeros(4),
        n_episodes=2, thr=pp.Thresholds(), scenario="S0")
    assert len(eps) == 2
    e = eps[0]
    assert e["eff"].shape[1] == 3 and e["cube"].shape[1] == 3
    assert e["grip"].ndim == 1
    assert e["goal"].shape == (3,)
    assert e["table_z"] == pytest.approx(0.0199)
    assert e["success"] is True            # reaches t>=3
    assert e["length"] == e["eff"].shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -k rollout_with_phase -v`
Expected: FAIL — `AttributeError: module 'evals.phase_probe' has no attribute 'rollout_with_phase_signals'`

- [ ] **Step 3: Add `_initial_obs` + `rollout_with_phase_signals` to `evals/phase_probe.py`**

Append EXACTLY this to `evals/phase_probe.py` (the helper first, then the
main function — one `env.reset()` per episode; `_initial_obs` uses
`compute_observation()` so it never re-resets and re-randomizes):

```python
def _initial_obs(env):
    """Initial observation for the post-reset (post-scenario) state.

    The caller already did exactly one env.reset() this episode; calling
    env.reset() again would re-randomize the task. OGBench manip envs
    expose compute_observation() on the unwrapped env, so use that;
    fall back to a fresh reset's observation only if it is unavailable.
    """
    u = env.unwrapped
    if hasattr(u, "compute_observation"):
        return np.asarray(u.compute_observation(), dtype=np.float32)
    obs, _ = env.reset()
    return np.asarray(obs, dtype=np.float32)


def rollout_with_phase_signals(
    env, agent, z, n_episodes: int, thr: Thresholds, scenario: str = "S0",
) -> List[Dict[str, Any]]:
    """Roll out n_episodes under `scenario`, recording per-step signals.

    Each episode: one env.reset() -> apply_scenario -> run policy.
    `success` is OR-accumulated from per-step info (same convention as
    evals/ogbench.py). Returns one signals dict per episode suitable for
    classify_phases.
    """
    import torch

    ensure_manip_env(env)
    episodes: List[Dict[str, Any]] = []
    for _ in range(n_episodes):
        _, info = env.reset()
        apply_scenario(env, scenario)
        goal = episode_goal(env)
        table_z = episode_table_z(env)

        effs, cubes, grips = [], [], []
        s0 = step_signals(info)
        effs.append(s0["eff"]); cubes.append(s0["cube"]); grips.append(s0["grip"])
        ep_success = bool(s0["success"])

        observation = _initial_obs(env)
        while True:
            a = agent.act(
                obs=torch.tensor(np.asarray(observation), device=agent.device,
                                 dtype=torch.float32)[None],
                z=z,
            ).cpu().numpy()[0]
            observation, _, terminated, truncated, info = env.step(a)
            s = step_signals(info)
            effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])
            ep_success = ep_success or s["success"]
            if terminated or truncated:
                break

        episodes.append({
            "eff": np.asarray(effs, dtype=np.float64),
            "cube": np.asarray(cubes, dtype=np.float64),
            "grip": np.asarray(grips, dtype=np.float64),
            "goal": goal,
            "success": bool(ep_success),
            "table_z": table_z,
            "length": len(effs),
        })
    return episodes
```

- [ ] **Step 4: Run the full test file to verify pass**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/test_phase_probe.py
git commit -m "feat(phase_probe): rollout_with_phase_signals"
```

---

## Task 5: `run_phase_probe` + aggregation + plots

**Files:**
- Modify: `evals/phase_probe.py`
- Test: `tests/test_phase_probe.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_phase_probe.py`:

```python
def test_run_phase_probe_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())

    def infer_z(task):
        return torch.zeros(4)

    def make_env(task):
        return _RolloutEnv()

    per_ep, summary, hist = pp.run_phase_probe(
        agent=_RolloutAgent(), infer_z=infer_z, make_env=make_env,
        tasks=["t1", "t2"], scenarios=["S0", "S2"], n_episodes=3,
        thr=pp.Thresholds())
    assert set(per_ep.columns) >= {
        "task", "scenario", "episode", "reached", "secured", "success",
        "reached_raw", "secured_raw", "fail_phase", "min_eff_cube_dist",
        "max_cube_lift", "final_cube_goal_dist", "length"}
    assert len(per_ep) == 2 * 2 * 3
    row = summary[(summary.task == "t1") & (summary.scenario == "S0")].iloc[0]
    assert row["n"] == 3
    assert row["success_rate"] == pytest.approx(1.0)  # _RolloutEnv succeeds
    assert "t1" in hist


def test_plots_and_summary_md(tmp_path):
    import pandas as pd
    per_ep = pd.DataFrame([
        {"task": "t1", "scenario": "S0", "episode": 0, "furthest_phase": "reached"},
        {"task": "t1", "scenario": "S0", "episode": 1, "furthest_phase": "secured"},
    ])
    summary = pd.DataFrame([
        {"task": "t1", "scenario": "S0", "reached_rate": 1.0,
         "secured_rate": 0.5, "success_rate": 0.0, "n": 2},
        {"task": "t1", "scenario": "S2", "reached_rate": 1.0,
         "secured_rate": 1.0, "success_rate": 0.9, "n": 2},
    ])
    pp.plot_phase_histogram(per_ep, tmp_path / "hist.png")
    pp.plot_scenario_success(summary, tmp_path / "succ.png")
    pp.write_summary_md(summary, tmp_path / "summary.md")
    assert (tmp_path / "hist.png").exists()
    assert (tmp_path / "succ.png").exists()
    txt = (tmp_path / "summary.md").read_text()
    assert "Hypothesis" in txt and "t1" in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -k "run_phase_probe or plots_and_summary" -v`
Expected: FAIL — `AttributeError: module 'evals.phase_probe' has no attribute 'run_phase_probe'`

- [ ] **Step 3: Add aggregation + plots to `evals/phase_probe.py`**

Append to `evals/phase_probe.py`:

```python
def run_phase_probe(
    *, agent, infer_z, make_env, tasks: List[str], scenarios: List[str],
    n_episodes: int, thr: Thresholds,
):
    """Run M1+M2 over tasks x scenarios. Returns (per_ep_df, summary_df,
    hist) where hist[task] -> {furthest_phase: count} for scenario S0."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        z = infer_z(task)
        for scenario in scenarios:
            env = make_env(task)
            try:
                eps = rollout_with_phase_signals(
                    env, agent, z, n_episodes, thr, scenario=scenario)
            finally:
                if hasattr(env, "close"):
                    env.close()
            for i, sig in enumerate(eps):
                c = classify_phases(sig, thr)
                rows.append({
                    "task": task, "scenario": scenario, "episode": i,
                    "reached": c["reached"], "secured": c["secured"],
                    "success": c["success"], "reached_raw": c["reached_raw"],
                    "secured_raw": c["secured_raw"],
                    "furthest_phase": c["furthest_phase"],
                    "fail_phase": c["fail_phase"],
                    "min_eff_cube_dist": c["min_eff_cube_dist"],
                    "max_cube_lift": c["max_cube_lift"],
                    "final_cube_goal_dist": c["final_cube_goal_dist"],
                    "length": c["length"],
                })

    per_ep = pd.DataFrame(rows)
    summary = (
        per_ep.groupby(["task", "scenario"])
        .agg(reached_rate=("reached", "mean"),
             secured_rate=("secured", "mean"),
             success_rate=("success", "mean"),
             n=("episode", "count"))
        .reset_index()
    )
    hist: Dict[str, Dict[str, int]] = {}
    s0 = per_ep[per_ep.scenario == "S0"]
    for task, g in s0.groupby("task"):
        hist[task] = g["furthest_phase"].value_counts().to_dict()
    return per_ep, summary, hist


def plot_phase_histogram(per_ep, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s0 = per_ep[per_ep.scenario == "S0"] if "scenario" in per_ep else per_ep
    tasks = sorted(s0["task"].unique())
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(tasks)), 4))
    bottom = np.zeros(len(tasks))
    for phase in PHASE_ORDER:
        counts = [
            int((s0[s0.task == t]["furthest_phase"] == phase).sum())
            for t in tasks
        ]
        ax.bar(tasks, counts, bottom=bottom, label=phase)
        bottom += np.array(counts)
    ax.set_ylabel("episodes")
    ax.set_title("M1: furthest phase per task (S0)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scenario_success(summary, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(summary["task"].unique())
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(tasks))
    w = 0.8 / max(1, len(scenarios))
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(tasks)), 4))
    for j, sc in enumerate(scenarios):
        vals = [
            float(summary[(summary.task == t) & (summary.scenario == sc)]
                   ["success_rate"].mean())
            for t in tasks
        ]
        ax.bar(x + j * w, vals, w, label=sc)
    ax.set_xticks(x + w * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("success rate")
    ax.set_title("M2: success rate by scenario")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary_md(summary, path) -> None:
    lines = ["# Cube phase-failure probe — summary", ""]
    for _, r in summary.sort_values(["task", "scenario"]).iterrows():
        lines.append(
            f"- **{r['task']} / {r['scenario']}**: reached "
            f"{r['reached_rate']*100:.1f}%  secured "
            f"{r['secured_rate']*100:.1f}%  success "
            f"{r['success_rate']*100:.1f}%  (n={int(r['n'])})")
    lines += ["", "## Hypothesis readout", ""]
    lines.append(
        "Prediction: success(S0) ≈ success(S1) (grasp still required) and "
        "success(S2) ≫ success(S0) (transport easy once held); baseline "
        "reached-rate ≫ secured-rate ⇒ grasp/pick-up is the bottleneck.")
    for task in sorted(summary["task"].unique()):
        sub = summary[summary.task == task].set_index("scenario")
        def g(sc, col):
            return float(sub.loc[sc, col]) if sc in sub.index else float("nan")
        lines.append(
            f"- {task}: S0 reached={g('S0','reached_rate')*100:.0f}% "
            f"secured={g('S0','secured_rate')*100:.0f}% "
            f"success S0={g('S0','success_rate')*100:.0f}% "
            f"S1={g('S1','success_rate')*100:.0f}% "
            f"S2={g('S2','success_rate')*100:.0f}%")
    Path(path).write_text("\n".join(lines))
```

- [ ] **Step 4: Run the full test file to verify pass**

Run: `.venv/bin/python -m pytest tests/test_phase_probe.py -v`
Expected: PASS (17 passed)

Then full suite (regression):
Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all green (prior 14 + 17 phase_probe = 31 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/test_phase_probe.py
git commit -m "feat(phase_probe): run_phase_probe aggregation + plots + summary"
```

---

## Task 6: `scripts/phase_probe.py` CLI

**Files:**
- Create: `scripts/phase_probe.py`

- [ ] **Step 1: Create the CLI**

Create `scripts/phase_probe.py`:

```python
"""scripts/phase_probe.py — test where FB fails in cube episodes.

Loads one run's .hydra/config.yaml + a checkpoint, runs M1 (phase
decomposition) + M2 (S0/S1/S2 counterfactuals) over the cube tasks, and
writes per-episode parquet, a summary + hypothesis readout, and plots.
Caller loops seeds/checkpoints. Run on Linux with --mujoco-gl egl.

Usage:
    python scripts/phase_probe.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/phase_probe/s3_final --mujoco-gl egl
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
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--scenarios", default="S0,S1,S2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mujoco-gl", default=None,
                    help="osmesa|glfw|egl; sets MUJOCO_GL before env import")
    ap.add_argument("--eps-reach", type=float, default=0.06)
    ap.add_argument("--delta-lift", type=float, default=0.03)
    ap.add_argument("--k-steps", type=int, default=5)
    ap.add_argument("--tau-grip", type=float, default=0.5)
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.phase_probe import (
        Thresholds, plot_phase_histogram, plot_scenario_success,
        run_phase_probe, write_summary_md,
    )
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None
    # build_env_and_agent only needed env for obs/action dims; per-task
    # envs are created later. Release this one's renderer/physics resources.
    if hasattr(env, "close"):
        env.close()

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))
    scenarios = [s for s in args.scenarios.split(",") if s]
    thr = Thresholds(eps_reach=args.eps_reach, delta_lift=args.delta_lift,
                     k_steps=args.k_steps, tau_grip=args.tau_grip)

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions,
    )
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False,
    )

    def infer_z(task):
        z, _ = evaluator._infer_z(task)
        return z

    def make_env(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        return e

    per_ep, summary, hist = run_phase_probe(
        agent=agent, infer_z=infer_z, make_env=make_env, tasks=tasks,
        scenarios=scenarios, n_episodes=args.n_episodes, thr=thr)

    per_ep.to_parquet(out / "per_episode.parquet")
    summary.to_parquet(out / "summary.parquet")
    write_summary_md(summary, out / "summary.md")
    plot_phase_histogram(per_ep, out / "phase_histogram.png")
    plot_scenario_success(summary, out / "scenario_success.png")
    (out / "metrics.json").write_text(json.dumps({
        "config": args.config, "checkpoint": args.checkpoint,
        "tasks": tasks, "scenarios": scenarios,
        "thresholds": vars(thr), "n_episodes": args.n_episodes,
    }, indent=2, default=str))
    print(f"[phase_probe] done -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses (no torch/ogbench import on --help)**

Run: `.venv/bin/python scripts/phase_probe.py --help`
Expected: argparse help listing `--config --checkpoint --out --device --tasks --n-episodes --scenarios --seed --mujoco-gl --eps-reach --delta-lift --k-steps --tau-grip`. Fast (heavy imports are inside `main()` after `parse_args`).

Then full suite (no regression):
Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 31 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/phase_probe.py
git commit -m "feat(phase_probe): single-checkpoint phase-failure CLI"
```

- [ ] **Step 4: Phase-probe Linux acceptance (run by user, documented here)**

On a Linux machine with the OGBench cube dataset at `cfg.data_path`:
```
.venv/bin/python scripts/phase_probe.py \
  --config RESULTS/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3/.hydra/config.yaml \
  --checkpoint RESULTS/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3/checkpoints/final.pt \
  --out analysis/phase_probe/s3_final --n-episodes 20 --mujoco-gl egl
```
Expected: `analysis/phase_probe/s3_final/` with `per_episode.parquet`,
`summary.parquet`, `summary.md` (incl. hypothesis readout),
`phase_histogram.png`, `scenario_success.png`, `metrics.json`. Read
`summary.md`: the hypothesis holds if baseline reached-rate ≫ secured-rate
and success(S2) ≫ success(S0) ≈ success(S1). Loop over seeds s3–s8,s10 by
re-invoking with each run dir.

---

## Final verification

- [ ] `.venv/bin/python -m pytest tests/ -v` → 31 passed
- [ ] `git diff main -- train.py evals/ogbench.py envs/rollout.py envs/ogbench.py agents/ data/ogbench.py evals/analysis.py scripts/analyze_checkpoint.py scripts/wandb_pull.py scripts/plot_sweep.py` → empty (this feature is additive; it must not modify the prior pipeline)
