# TD-MPC2 → OGBench (state, goal-folded) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-party TD-MPC2 model-based agent to Factored-FB that trains on OGBench `cube-single` state data and plugs into the existing train/eval harness, to fill the `\dreamer{}`/MBRL row of the rlbrew paper.

**Architecture:** Copy TD-MPC2's internals (`world_model.py`, `layers.py`, `math.py`, `scale.py`, `init.py`, `tdmpc2.py`) into `agents/tdmpc2/` **verbatim**, then apply ONLY an enumerated set of mods: (1) remove `tensordict` — rewrite the Q-ensemble with `torch.func.stack_module_state`/`functional_call`/`vmap` and replace info `TensorDict`s with plain `dict`; (2) remove `torch.compile`/cudagraphs; (3) parameterize device. The TD-MPC2 *algorithm* (`_update`, `_plan`, `_estimate_value`, world-model forward math) stays line-for-line identical. All OGBench-specific logic (reward relabeling, goal-folding, batch reshape, planning eval hooks) lives in a thin `TDMPC2Agent(BaseAgent)` wrapper in `agents/tdmpc2/agent.py` that feeds `core` exactly the tensor shapes upstream already expects.

**Tech Stack:** PyTorch 2.12 (eager, no compile, no tensordict), Hydra/OmegaConf configs, existing `DictBuffer` horizon-window sampling, `pytest`.

**Fidelity contract:** After copying, every difference between `agents/tdmpc2/<file>.py` and the upstream original MUST be one of the mods listed in Task 2's diff table. Task 1 saves pristine originals as `*.orig` so `diff` can verify this; Task 11 removes them.

**Test command (this repo):** `uv` is broken — always use `.venv/bin/python -m pytest`.

**Key design decisions (settled):**
- **Goal = target cube xyz (3-dim)**, folded onto the observation: `obs' = concat(obs, goal)`. The world model sees `obs_dim+3` inputs; `task_dim=0`, `multitask=False`, `episodic=False`.
- **Training reward** is synthesized per step from `physics`: cube xyz is `physics[..., 14:17]`; `reward = 0.0 if ||cube_xyz - goal|| < 0.04 else -1.0` (matches `envs/ogbench.cube_reward_fn` for one cube).
- **Training goals** use hindsight + random mix: with prob `1-random_goal_ratio` the goal is the window's final achieved cube xyz; else a random other window's final cube xyz. `random_goal_ratio=0.5` default.
- **Eval goal** = `env.unwrapped.cur_task_info["goal_xyzs"][0]`, surfaced via `TDMPC2Agent.eval_context(env, domain, task)`; carried through the existing `ctx`/`z` channel and folded into obs at each step.
- **Buffer window length == TD-MPC2 `horizon`** (`horizon=3`): the adapter builds `H+1` states (`H` obs + final `next_obs`), `H` actions, `H` rewards, `H` terminated.
- **Episode length** for the discount heuristic: `episode_length=200` (OGBench cube-single).

---

## File Structure

| File | Responsibility |
|---|---|
| `agents/tdmpc2/__init__.py` | Package marker; exports `TDMPC2Agent`. |
| `agents/tdmpc2/init.py` | Copied verbatim from upstream `common/init.py` (weight init helpers). |
| `agents/tdmpc2/math.py` | Copied from `common/math.py`; only mod: `TensorDict`→`dict`. |
| `agents/tdmpc2/scale.py` | Copied from `common/scale.py`; only mod: device parameterized. |
| `agents/tdmpc2/layers.py` | Copied from `common/layers.py`; only mod: `Ensemble` rewritten with `torch.func`. |
| `agents/tdmpc2/world_model.py` | Copied from `common/world_model.py`; mods: ensemble/target-Q without tensordict, info dict, device. |
| `agents/tdmpc2/core.py` | Copied from `tdmpc2.py` (`TDMPC2` class → `TDMPC2Core`); mods: device, compile/cudagraph removal, info dict. The algorithm is otherwise untouched. |
| `agents/tdmpc2/config.py` | `build_tdmpc2_cfg(**kwargs) -> OmegaConf` — the cfg-shim replacing Hydra/parser. |
| `agents/tdmpc2/agent.py` | `TDMPC2Agent(BaseAgent)`: cfg-shim glue, `update` adapter (reshape + reward + goal-fold), `act`/`reset`/`eval_context`. |
| `configs/agent/tdmpc2.yaml` | Hydra agent config. |
| `evals/phase_probe.py` (modify) | Add latent→cube-state probe for the phase-transition-prediction diagnostic. |
| `scripts/agents/tdmpc2/state.sh` | Launcher mirroring `scripts/agents/rldp/state.sh`. |
| `tests/agents/tdmpc2/...` | Unit + integration tests. |
| `agents/base.py` (modify) | Add no-op `reset()` to `BaseAgent`. |
| `envs/rollout.py` (modify) | Call `agent.reset()` at each episode start. |
| `evals/ogbench.py` (modify) | Use `agent.eval_context(...)` when present. |
| `train.py` (modify) | Wire `cfg.agent == "tdmpc2"` into `make_agent`. |

---

## Task 0: Worktree & branch sanity

**Files:** none (setup only)

- [ ] **Step 1: Confirm clean state on the feature branch**

Run: `git status --short && git rev-parse --abbrev-ref HEAD`
Expected: branch `rldp-state-agent` (or a fresh `tdmpc2-agent` branch if you prefer isolation); no unexpected staged changes. If isolating, `git checkout -b tdmpc2-agent`.

- [ ] **Step 2: Confirm test runner works**

Run: `.venv/bin/python -m pytest tests/ -q -k "rldp or fb" 2>&1 | tail -5`
Expected: existing agent tests collect and pass (pandas-missing module errors are pre-existing and unrelated).

---

## Task 1: Copy TD-MPC2 internals verbatim

**Files:**
- Create: `agents/tdmpc2/__init__.py`
- Create (copy): `agents/tdmpc2/{init,math,scale,layers,world_model,core}.py`
- Create (pristine refs): `agents/tdmpc2/{init,math,scale,layers,world_model,core}.py.orig`

- [ ] **Step 1: Create the package and copy files**

```bash
SRC=../baselines/tdmpc2/tdmpc2
mkdir -p agents/tdmpc2 tests/agents/tdmpc2
touch agents/tdmpc2/__init__.py tests/agents/tdmpc2/__init__.py
cp "$SRC/common/init.py"        agents/tdmpc2/init.py
cp "$SRC/common/math.py"        agents/tdmpc2/math.py
cp "$SRC/common/scale.py"       agents/tdmpc2/scale.py
cp "$SRC/common/layers.py"      agents/tdmpc2/layers.py
cp "$SRC/common/world_model.py" agents/tdmpc2/world_model.py
cp "$SRC/tdmpc2.py"             agents/tdmpc2/core.py
# pristine references for the fidelity diff (deleted in Task 11)
for f in init math scale layers world_model core; do cp "agents/tdmpc2/$f.py" "agents/tdmpc2/$f.py.orig"; done
```

- [ ] **Step 2: Verify the copy is byte-identical to source**

Run:
```bash
diff agents/tdmpc2/core.py ../baselines/tdmpc2/tdmpc2/tdmpc2.py && echo "core OK"
diff agents/tdmpc2/world_model.py ../baselines/tdmpc2/tdmpc2/common/world_model.py && echo "wm OK"
```
Expected: no diff output, "core OK" and "wm OK" printed.

- [ ] **Step 3: Commit the verbatim copy (separately, so the diff is reviewable)**

```bash
git add agents/tdmpc2/
git commit -m "chore(tdmpc2): copy TD-MPC2 internals verbatim (pre-mod baseline)"
```

---

## Task 2: Apply the enumerated fidelity-preserving mods

This is the ONLY task that edits the copied algorithm files. Every edit below is one of the agreed mods. Nothing else changes.

**Files:**
- Modify: `agents/tdmpc2/init.py`, `math.py`, `scale.py`, `layers.py`, `world_model.py`, `core.py`

### Mod table (the complete, exhaustive list of allowed changes)

| # | File | Change | Reason |
|---|---|---|---|
| M1 | all copied | `from common import ...` / `from common.X import ...` → relative (`from . import ...`) | package relocation |
| M2 | math.py | `from tensordict import TensorDict` removed; `return TensorDict({...})` → `return {...}` | drop tensordict |
| M3 | scale.py | `torch.device('cuda:0')` (×2) → `torch.device(cfg.device)` | device param |
| M4 | layers.py | `Ensemble` rewritten with `torch.func` (Task 3) | drop tensordict |
| M5 | world_model.py | `from tensordict...` removed; `init()`/`soft_update_target_Q()`/`Q()`/`init.zero_` adapted to plain ensemble (Task 3) | drop tensordict |
| M6 | world_model.py | `info = TensorDict({...})` in `pi()` → `info = {...}`; indexing `info["mean"]` etc. unchanged (dict supports it) | drop tensordict |
| M7 | core.py | `from tensordict import TensorDict` removed; both `info = TensorDict({...})` → `info = {...}`; `.detach().mean()` on TensorDict → helper over dict (Task 6) | drop tensordict |
| M8 | core.py | `self.device = torch.device('cuda:0')` → `torch.device(cfg.device)`; `device='cuda:0'` in discount → `cfg.device` | device param |
| M9 | core.py | `capturable=True` (×2 Adam) → `capturable=(self.device.type=='cuda')` | cpu test support |
| M10 | core.py | `if cfg.compile:` block + `self.plan` compile branch left intact but **cfg.compile is always False**; `torch.compiler.cudagraph_mark_step_begin()` guarded by `if self.cfg.compile:` | drop compile/cudagraphs |
| M11 | core.py | class renamed `TDMPC2` → `TDMPC2Core` | avoid name clash, clarity |

- [ ] **Step 1: M1 — fix imports across copied files**

In `agents/tdmpc2/world_model.py` line 6: `from common import layers, math, init` → `from . import layers, math, init`.
In `agents/tdmpc2/layers.py`: remove `from tensordict import from_modules` (handled in Task 3).
In `agents/tdmpc2/core.py` lines 4-7:
```python
from . import math
from .scale import RunningScale
from .world_model import WorldModel
from .layers import api_model_conversion
```

- [ ] **Step 2: M2 — math.py drop tensordict**

Remove `from tensordict import TensorDict` (line 3). Change `termination_statistics` return (line 109) to:
```python
	return {'termination_rate': rate, 'termination_f1': f1}
```

- [ ] **Step 3: M3 — scale.py device**

`scale.py __init__` already receives `cfg`. Replace both `device=torch.device('cuda:0')` with `device=torch.device(cfg.device)`.

- [ ] **Step 4: M6 — world_model.py pi() info dict**

In `world_model.py` `pi()` (line 177): `info = TensorDict({...})` → `info = {...}`. Remove `from tensordict import TensorDict` / `from tensordict.nn import TensorDictParams` (lines 7-8) — the `TensorDictParams` usages are removed in Task 3.

- [ ] **Step 5: M7/M8/M9/M10/M11 — core.py**

- Remove `from tensordict import TensorDict` (line 8).
- Rename `class TDMPC2(torch.nn.Module):` → `class TDMPC2Core(torch.nn.Module):` (line 11).
- Line 21: `self.device = torch.device('cuda:0')` → `self.device = torch.device(cfg.device)`.
- Lines 23-32 Adam (`self.optim`): append `capturable=(self.device.type=='cuda')` (replace existing `capturable=True`).
- Line 32 (`self.pi_optim`): `capturable=True` → `capturable=(self.device.type=='cuda')`.
- Lines 36-38 discount: `device='cuda:0'` → `device=self.device`.
- Line 41 `_prev_mean`: `device=self.device` (already uses self.device — confirm).
- Lines 233-239 and 321-331: `info = TensorDict({...})` → `info = {...}`.
- Lines 332/`return info.detach().mean()`: replace with a dict reducer (added in Task 6 as `_reduce_info`). For now write `return self._reduce_info(info)`.
- Line 348 `torch.compiler.cudagraph_mark_step_begin()`: guard `if self.cfg.compile:\n\t\t\ttorch.compiler.cudagraph_mark_step_begin()`.

- [ ] **Step 6: Verify the diff contains only mod-table hunks**

Run: `diff agents/tdmpc2/core.py agents/tdmpc2/core.py.orig`
Expected: every hunk maps to a row in the mod table above (imports, device, capturable, compile guard, class rename, info dicts, `_reduce_info`). No algorithm/math lines changed. Repeat for `world_model.py`, `math.py`, `scale.py`.

- [ ] **Step 7: Commit**

```bash
git add agents/tdmpc2/
git commit -m "feat(tdmpc2): apply device + compile-removal + import mods (no algorithm change)"
```

---

## Task 3: Rewrite the Q-ensemble without tensordict (the only structural change)

Replace `tensordict`'s `from_modules`/`vmap`/`TensorDictParams` with stock `torch.func`. Behavior must be numerically identical to a naive per-member loop — proven by tests BEFORE wiring into the world model.

**Files:**
- Modify: `agents/tdmpc2/layers.py` (Ensemble)
- Modify: `agents/tdmpc2/world_model.py` (`init`, `soft_update_target_Q`, `Q`, last-layer zero-init)
- Test: `tests/agents/tdmpc2/test_ensemble.py`

- [ ] **Step 1: Write the failing ensemble equivalence test**

```python
# tests/agents/tdmpc2/test_ensemble.py
import torch
import torch.nn as nn
from agents.tdmpc2.layers import Ensemble


def _make_members(n, in_dim, out_dim):
    torch.manual_seed(0)
    return [nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim)) for _ in range(n)]


def test_vmap_ensemble_matches_naive_loop():
    n, in_dim, out_dim, B = 5, 6, 3, 4
    members = _make_members(n, in_dim, out_dim)
    ens = Ensemble(members)
    x = torch.randn(B, in_dim)
    out = ens(x)                                  # [n, B, out_dim]
    assert out.shape == (n, B, out_dim)
    ref = torch.stack([m(x) for m in members], 0)  # naive loop
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_ensemble_len():
    members = _make_members(3, 4, 2)
    assert len(Ensemble(members)) == 3
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_ensemble.py -v`
Expected: FAIL (current `Ensemble` imports `from_modules`/tensordict → ImportError or wrong behavior).

- [ ] **Step 3: Rewrite `Ensemble` in `layers.py`**

Replace the whole `Ensemble` class (lines 8-33) and the `from tensordict import from_modules` import with:
```python
import copy
from torch.func import stack_module_state, functional_call


class Ensemble(nn.Module):
    """Vectorized ensemble of identical modules (tensordict-free, torch.func based).

    Stacks member parameters along a leading dim and evaluates all members in a
    single ``torch.vmap`` call — numerically identical to looping over members.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        self._members = nn.ModuleList(modules)
        self._n = len(modules)
        self._repr = str(modules[0])
        self._base = copy.deepcopy(modules[0]).to("meta")

    def _stacked(self):
        params, buffers = stack_module_state(list(self._members))
        return params, buffers

    def __len__(self):
        return self._n

    def forward(self, *args, **kwargs):
        params, buffers = self._stacked()

        def fmodel(p, b, *a):
            return functional_call(self._base, (p, b), a, kwargs)

        return torch.vmap(fmodel, (0, 0, None), randomness="different")(params, buffers, *args)

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_ensemble.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Adapt `world_model.py` ensemble plumbing**

The upstream `init()` / `soft_update_target_Q()` and `_Qs.params[...]` indexing assumed tensordict-stacked params. Rewrite to plain-module ensembles + target copy.

In `world_model.py __init__`, replace last-layer zero-init (line 32):
```python
		# zero-init the reward head and each Q-member's final layer
		init.zero_([self._reward[-1].weight])
		for q in self._Qs._members:
			torch.nn.init.zeros_(q[-1].weight)
```

Replace `init()` (lines 38-53) with a plain deep-copied target + detached-forward helper:
```python
	def init(self):
		# Target ensemble is a plain deep copy; soft-updated via Polyak.
		self._target_Qs = deepcopy(self._Qs)
		for p in self._target_Qs.parameters():
			p.requires_grad_(False)
```

Replace `soft_update_target_Q()` (lines 82-86):
```python
	def soft_update_target_Q(self):
		with torch.no_grad():
			for sp, tp in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				tp.data.lerp_(sp.data, self.cfg.tau)
```

In `Q()` (lines 186-216) replace the `qnet` selection. The upstream `detach` path used a params-sharing `_detach_Qs`; reproduce "grad to input, not to weights" with a detached functional pass:
```python
		if target:
			out = self._target_Qs(z)
		elif detach:
			out = self._detach_forward(self._Qs, z)
		else:
			out = self._Qs(z)
```
And add the helper method to the class:
```python
	@staticmethod
	def _detach_forward(ensemble, z):
		# Forward through the online ensemble with parameters detached:
		# gradients flow to the input z (hence to pi) but NOT to Q weights.
		from torch.func import stack_module_state, functional_call
		params, buffers = stack_module_state(list(ensemble._members))
		params = {k: v.detach() for k, v in params.items()}
		buffers = {k: v.detach() for k, v in buffers.items()}

		def fmodel(p, b, x):
			return functional_call(ensemble._base, (p, b), (x,))

		return torch.vmap(fmodel, (0, 0, None))(params, buffers, z)
```
Remove the now-unused `_detach_Qs`/`_target_Qs_params`/`_detach_Qs_params` references and the `to()` override's `self.init()` call only if it errors (keep otherwise — it just rebuilds the target copy on device move, which is harmless).

- [ ] **Step 6: Write a world-model target/detach test**

```python
# tests/agents/tdmpc2/test_world_model_ensemble.py
import torch
from agents.tdmpc2.config import build_tdmpc2_cfg
from agents.tdmpc2.world_model import WorldModel


def _wm(obs_dim=12, act_dim=4):
    cfg = build_tdmpc2_cfg(obs_dim=obs_dim, action_dim=act_dim, device="cpu", horizon=3)
    return WorldModel(cfg).to("cpu"), cfg


def test_target_starts_equal_then_diverges_on_soft_update():
    wm, cfg = _wm()
    # perturb online Qs
    for p in wm._Qs.parameters():
        p.data.add_(torch.randn_like(p))
    before = [tp.clone() for tp in wm._target_Qs.parameters()]
    wm.soft_update_target_Q()
    after = list(wm._target_Qs.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # target moved


def test_detach_path_no_qweight_grad():
    wm, cfg = _wm()
    z = torch.randn(8, cfg.latent_dim, requires_grad=True)
    a = torch.randn(8, cfg.action_dim)
    q = wm.Q(z, a, task=None, return_type="avg", detach=True)
    q.sum().backward()
    assert z.grad is not None                       # grad reaches input
    assert all(p.grad is None for p in wm._Qs.parameters())  # not the weights
```
(Requires Task 4's `build_tdmpc2_cfg`; if executing strictly in order, write Task 4 first or stub the cfg. Recommended order: Task 4 Step 1 before this step.)

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_ensemble.py tests/agents/tdmpc2/test_world_model_ensemble.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add agents/tdmpc2/layers.py agents/tdmpc2/world_model.py tests/agents/tdmpc2/
git commit -m "feat(tdmpc2): tensordict-free torch.func Q-ensemble + plain target/detach (parity tested)"
```

---

## Task 4: cfg-shim (`build_tdmpc2_cfg`)

**Files:**
- Create: `agents/tdmpc2/config.py`
- Test: `tests/agents/tdmpc2/test_config.py`

- [ ] **Step 1: Write `build_tdmpc2_cfg`**

```python
# agents/tdmpc2/config.py
"""Construct the OmegaConf object TD-MPC2's internals expect, from flat kwargs.

Replaces upstream Hydra config.yaml + common/parser.py. Defaults mirror
../baselines/tdmpc2/tdmpc2/config.yaml exactly; only single-task / state /
non-episodic fields are specialized for OGBench cube-single.
"""
from __future__ import annotations

from omegaconf import OmegaConf

# Verbatim numeric defaults from upstream config.yaml (the values, not the Hydra wiring).
_DEFAULTS = dict(
    # training
    batch_size=256, reward_coef=0.1, value_coef=0.1, termination_coef=1.0,
    consistency_coef=20.0, rho=0.5, lr=3e-4, enc_lr_scale=0.3,
    grad_clip_norm=20.0, tau=0.01,
    discount_denom=5, discount_min=0.95, discount_max=0.995,
    # planning
    mpc=True, iterations=6, num_samples=512, num_elites=64, num_pi_trajs=24,
    horizon=3, min_std=0.05, max_std=2.0, temperature=0.5,
    # actor
    log_std_min=-10, log_std_max=2, entropy_coef=1e-4,
    # critic
    num_bins=101, vmin=-10, vmax=10,
    # architecture
    num_enc_layers=2, enc_dim=256, num_channels=32, mlp_dim=512,
    latent_dim=512, num_q=5, dropout=0.01, simnorm_dim=8,
    # misc
    episodic=False, multitask=False, compile=False, obs="state",
    episode_length=200,
)


def build_tdmpc2_cfg(*, obs_dim: int, action_dim: int, device: str = "cpu", **overrides):
    """obs_dim is the RAW obs dim; the agent folds a 3-dim goal on top, so the
    model's input dim is obs_dim+3 (see agent.GOAL_DIM)."""
    from agents.tdmpc2.agent import GOAL_DIM  # local import to avoid cycle

    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["device"] = device
    cfg["action_dim"] = int(action_dim)
    cfg["task_dim"] = 0            # single-task: no task embedding concatenated
    cfg["tasks"] = None
    cfg["obs_shape"] = {"state": (int(obs_dim) + GOAL_DIM,)}
    # derived (upstream computes these in common/parser.py)
    cfg["bin_size"] = (cfg["vmax"] - cfg["vmin"]) / (cfg["num_bins"] - 1)
    return OmegaConf.create(cfg)
```

- [ ] **Step 2: Write the config test**

```python
# tests/agents/tdmpc2/test_config.py
from agents.tdmpc2.config import build_tdmpc2_cfg


def test_cfg_folds_goal_into_obs_shape():
    cfg = build_tdmpc2_cfg(obs_dim=19, action_dim=5, device="cpu")
    assert cfg.obs_shape["state"][0] == 19 + 3   # GOAL_DIM
    assert cfg.task_dim == 0
    assert cfg.multitask is False
    assert abs(cfg.bin_size - (20 - (-10)) / 100) < 1e-9


def test_overrides_apply():
    cfg = build_tdmpc2_cfg(obs_dim=10, action_dim=4, device="cpu", horizon=5, num_q=3)
    assert cfg.horizon == 5 and cfg.num_q == 3
```

- [ ] **Step 3: Run (expect import error until agent.GOAL_DIM exists)**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_config.py -v`
Expected: FAIL importing `GOAL_DIM`. Proceed to Task 5 Step 1 (defines `GOAL_DIM`), then re-run — Expected: PASS.

- [ ] **Step 4: Commit (after Task 5 Step 1 lands GOAL_DIM)**

```bash
git add agents/tdmpc2/config.py tests/agents/tdmpc2/test_config.py
git commit -m "feat(tdmpc2): cfg-shim build_tdmpc2_cfg replacing Hydra/parser"
```

---

## Task 5: `TDMPC2Agent` skeleton + reward/goal/reshape adapter

**Files:**
- Create: `agents/tdmpc2/agent.py`
- Modify: `agents/tdmpc2/__init__.py`
- Test: `tests/agents/tdmpc2/test_adapter.py`

- [ ] **Step 1: Write `agent.py` (skeleton: GOAL_DIM, init, reset, state_dict, the reward/goal/reshape helpers)**

```python
# agents/tdmpc2/agent.py
"""TDMPC2Agent — BaseAgent wrapper bridging OGBench (offline, reward-free,
goal-conditioned) to TD-MPC2 (reward-driven MPC). All OGBench-specific logic
lives here; agents/tdmpc2/core.py stays algorithmically identical to upstream."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from agents.base import BaseAgent
from agents.tdmpc2.config import build_tdmpc2_cfg
from agents.tdmpc2.core import TDMPC2Core

GOAL_DIM = 3                 # cube target xyz
CUBE_SLICE = slice(14, 17)   # physics[..., 14:17] == cube xyz (see envs/ogbench.cube_reward_fn)
SUCCESS_THRESHOLD = 0.04


class TDMPC2Agent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        *,
        device: str = "cpu",
        horizon: int = 3,
        random_goal_ratio: float = 0.5,
        success_threshold: float = SUCCESS_THRESHOLD,
        **cfg_overrides,
    ):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.random_goal_ratio = float(random_goal_ratio)
        self.success_threshold = float(success_threshold)
        obs_dim = int(obs_space.shape[0])
        self.cfg = build_tdmpc2_cfg(
            obs_dim=obs_dim, action_dim=action_dim, device=str(device),
            horizon=horizon, **cfg_overrides,
        )
        self.horizon = int(self.cfg.horizon)
        self.core = TDMPC2Core(self.cfg)
        self._t0 = True

    # ---- goal / reward helpers ------------------------------------------- #
    def _cube_xyz(self, physics: Tensor) -> Tensor:
        return physics[..., CUBE_SLICE].to(self.device).float()

    def _sample_goals(self, next_phys_window: Tensor) -> Tensor:
        """next_phys_window: [B, H, phys]. Returns goal cube xyz [B, 3].
        Hindsight: window-final achieved cube xyz; with prob random_goal_ratio,
        swap for another window's final cube xyz (random goal relabeling)."""
        final_cube = self._cube_xyz(next_phys_window[:, -1, :])  # [B,3]
        B = final_cube.shape[0]
        use_random = torch.rand(B, device=self.device) < self.random_goal_ratio
        perm = torch.randperm(B, device=self.device)
        return torch.where(use_random.unsqueeze(-1), final_cube[perm], final_cube)

    def _reward(self, next_phys_window: Tensor, goal: Tensor) -> Tensor:
        """[B,H,phys], goal [B,3] -> reward [B,H,1]: 0 if within threshold else -1."""
        cube = self._cube_xyz(next_phys_window)               # [B,H,3]
        dist = torch.linalg.norm(cube - goal.unsqueeze(1), dim=-1)  # [B,H]
        return torch.where(dist < self.success_threshold, 0.0, -1.0).unsqueeze(-1)

    @staticmethod
    def _fold(obs: Tensor, goal: Tensor) -> Tensor:
        """obs [...,obs_dim], goal [B,3] broadcast over leading dims -> concat."""
        if obs.dim() == goal.dim():           # [B,obs] + [B,3]
            return torch.cat([obs, goal], dim=-1)
        # obs [T,B,obs] + goal [B,3] -> broadcast over T
        g = goal.unsqueeze(0).expand(obs.shape[0], *goal.shape)
        return torch.cat([obs, g], dim=-1)

    # ---- BaseAgent API (act/update in later steps) ----------------------- #
    def reset(self) -> None:
        self._t0 = True
        with torch.no_grad():
            self.core._prev_mean.zero_()

    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:  # unused for tdmpc2
        raise NotImplementedError("TDMPC2 uses eval_context, not infer_z")

    def state_dict(self) -> dict:
        return {"core": self.core.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.core.load_state_dict(state["core"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
```

- [ ] **Step 2: Add `__init__.py` export**

```python
# agents/tdmpc2/__init__.py
from agents.tdmpc2.agent import TDMPC2Agent

__all__ = ["TDMPC2Agent"]
```

- [ ] **Step 3: Write the reward/goal/reshape unit tests**

```python
# tests/agents/tdmpc2/test_adapter.py
import torch
import types
from agents.tdmpc2.agent import TDMPC2Agent, GOAL_DIM, CUBE_SLICE


def _agent(obs_dim=19, act_dim=5):
    obs_space = types.SimpleNamespace(shape=(obs_dim,))
    return TDMPC2Agent(obs_space, act_dim, device="cpu", horizon=3)


def test_reward_zero_at_goal_minus_one_far():
    a = _agent()
    B, H, P = 4, 3, 40
    phys = torch.zeros(B, H, P)
    phys[..., CUBE_SLICE] = 0.0
    goal = torch.zeros(B, GOAL_DIM)            # cube already at goal
    r = a._reward(phys, goal)
    assert r.shape == (B, H, 1)
    assert torch.all(r == 0.0)
    phys[..., CUBE_SLICE] = 1.0                # far
    assert torch.all(a._reward(phys, torch.zeros(B, GOAL_DIM)) == -1.0)


def test_fold_dims():
    a = _agent(obs_dim=19)
    obs2d = torch.randn(8, 19); goal = torch.randn(8, GOAL_DIM)
    assert a._fold(obs2d, goal).shape == (8, 19 + GOAL_DIM)
    obs3d = torch.randn(4, 8, 19)              # [T,B,obs]
    assert a._fold(obs3d, goal).shape == (4, 8, 19 + GOAL_DIM)


def test_sample_goals_shape():
    a = _agent()
    phys = torch.randn(6, 3, 40)
    assert a._sample_goals(phys).shape == (6, GOAL_DIM)
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_adapter.py tests/agents/tdmpc2/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/tdmpc2/agent.py agents/tdmpc2/__init__.py tests/agents/tdmpc2/test_adapter.py
git commit -m "feat(tdmpc2): TDMPC2Agent skeleton + reward/goal/fold adapters"
```

---

## Task 6: `update(batch, step)` — feed `core` the shapes it expects

**Files:**
- Modify: `agents/tdmpc2/agent.py` (add `update`), `agents/tdmpc2/core.py` (add `_reduce_info`, accept tensors instead of `buffer.sample()`)
- Test: `tests/agents/tdmpc2/test_update.py`

- [ ] **Step 1: Add `_reduce_info` + a tensor-input `update_step` to `core.py`**

Upstream `TDMPC2Core.update(buffer)` calls `buffer.sample()` itself. We instead pass tensors. Add a thin method that reuses `_update` unchanged:
```python
	@staticmethod
	def _reduce_info(info: dict) -> dict:
		out = {}
		for k, v in info.items():
			out[k] = v.detach().mean().item() if torch.is_tensor(v) else float(v)
		return out

	def update_step(self, obs, action, reward, terminated, task=None):
		"""Tensor-input entry point (replaces update(buffer)); shapes match _update."""
		kwargs = {"task": task} if task is not None else {}
		if self.cfg.compile:
			torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
```
Note `_update` already returns `self._reduce_info(info)` per Task 2 Step 5.

- [ ] **Step 2: Write the failing `update` integration test**

```python
# tests/agents/tdmpc2/test_update.py
import torch, types
from agents.tdmpc2.agent import TDMPC2Agent


def _agent(obs_dim=19, act_dim=5, H=3):
    return TDMPC2Agent(types.SimpleNamespace(shape=(obs_dim,)), act_dim, device="cpu", horizon=H)


def _windowed_batch(B=16, H=3, obs_dim=19, act_dim=5, P=40):
    rand = torch.randn
    return {
        "observation": rand(B, H, obs_dim),
        "action": rand(B, H, act_dim).clamp(-1, 1),
        "physics": rand(B, H, P),
        "next": {
            "observation": rand(B, H, obs_dim),
            "physics": rand(B, H, P),
            "terminated": torch.zeros(B, H, 1),
        },
    }


def test_update_runs_and_returns_float_metrics():
    a = _agent()
    metrics = a.update(_windowed_batch(), step=0)
    assert "consistency_loss" in metrics and "value_loss" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_update_changes_params():
    a = _agent()
    p0 = [p.clone() for p in a.core.model._dynamics.parameters()]
    for s in range(3):
        a.update(_windowed_batch(), step=s)
    p1 = list(a.core.model._dynamics.parameters())
    assert any(not torch.equal(b, x) for b, x in zip(p0, p1))
```

- [ ] **Step 3: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_update.py -v`
Expected: FAIL (`update` not defined on TDMPC2Agent).

- [ ] **Step 4: Implement `update` in `agent.py`**

```python
    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        # batch fields are [B, H, ...] windows from DictBuffer.sample(horizon=H).
        obs = batch["observation"].to(self.device).float()          # [B,H,obs]
        next_obs = batch["next"]["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()            # [B,H,act]
        next_phys = batch["next"]["physics"].to(self.device).float()  # [B,H,phys]
        term = batch["next"]["terminated"].to(self.device).float()  # [B,H,1] (or [B,H])
        if term.dim() == 2:
            term = term.unsqueeze(-1)

        goal = self._sample_goals(next_phys)                        # [B,3]
        reward = self._reward(next_phys, goal)                      # [B,H,1]

        # Build H+1 obs sequence: obs[:,0..H-1] then final next_obs[:,H-1].
        obs_seq = torch.cat([obs, next_obs[:, -1:, :]], dim=1)      # [B,H+1,obs]
        obs_seq = self._fold(obs_seq.transpose(0, 1), goal)         # -> [H+1,B,obs+3]
        action_t = action.transpose(0, 1)                           # [H,B,act]
        reward_t = reward.transpose(0, 1)                           # [H,B,1]
        term_t = term.transpose(0, 1)                               # [H,B,1]

        return self.core.update_step(obs_seq, action_t, reward_t, term_t, task=None)
```

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_update.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agents/tdmpc2/agent.py agents/tdmpc2/core.py tests/agents/tdmpc2/test_update.py
git commit -m "feat(tdmpc2): update() adapter feeds core H+1/H tensors with relabeled reward"
```

---

## Task 7: `act`, `eval_context`, and harness hooks

**Files:**
- Modify: `agents/tdmpc2/agent.py` (`act`, `eval_context`)
- Modify: `agents/base.py` (no-op `reset`)
- Modify: `envs/rollout.py` (call `agent.reset()` at episode starts)
- Modify: `evals/ogbench.py` (use `agent.eval_context` when present)
- Test: `tests/agents/tdmpc2/test_act.py`

- [ ] **Step 1: Add no-op `reset` to `BaseAgent`**

In `agents/base.py`, inside `BaseAgent`:
```python
    def reset(self) -> None:
        """Called at the start of each eval episode. Default: no-op."""
        return None
```

- [ ] **Step 2: rollout calls `reset()` at episode boundaries**

In `envs/rollout.py`:
- after line 19 (`observation, info = env.reset()`): add `agent.reset()`.
- after line 36 (the `env.reset()` inside the loop): add `agent.reset()`.

- [ ] **Step 3: evaluator uses `eval_context` when present**

In `evals/ogbench.py` `run()`, replace the `z, relabel_metrics = self._infer_z(task)` line (≈line 79) with:
```python
                if hasattr(self.agent, "eval_context"):
                    z, relabel_metrics = self.agent.eval_context(env=env, domain=self.domain, task=task)
                else:
                    z, relabel_metrics = self._infer_z(task)
```

- [ ] **Step 4: Implement `act` + `eval_context` in `agent.py`**

```python
    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        # z carries the goal cube xyz (shape [3] or [1,3]) from eval_context.
        obs = obs.to(self.device).float()              # [1, obs_dim]
        goal = z.to(self.device).float().reshape(GOAL_DIM)
        folded = torch.cat([obs.reshape(-1), goal], dim=-1)   # [obs_dim+3]
        action = self.core.act(folded, t0=self._t0, eval_mode=eval_mode)  # [act_dim]
        self._t0 = False
        return action.to(self.device).unsqueeze(0)     # [1, act_dim] for rollout

    @torch.no_grad()
    def eval_context(self, *, env, domain, task):
        goal_xyzs = env.unwrapped.cur_task_info["goal_xyzs"]   # [num_cubes, 3]
        goal = torch.as_tensor(goal_xyzs[0], dtype=torch.float32, device=self.device)
        return goal, {}
```

- [ ] **Step 5: Write `act` tests**

```python
# tests/agents/tdmpc2/test_act.py
import torch, types
from agents.tdmpc2.agent import TDMPC2Agent, GOAL_DIM


def _agent(obs_dim=19, act_dim=5):
    return TDMPC2Agent(types.SimpleNamespace(shape=(obs_dim,)), act_dim, device="cpu", horizon=3)


def test_act_shape_and_range():
    a = _agent(); a.reset()
    obs = torch.randn(1, 19); goal = torch.randn(GOAL_DIM)
    act = a.act(obs, goal, eval_mode=True)
    assert act.shape == (1, 5)
    assert torch.all(act.abs() <= 1.0 + 1e-5)


def test_reset_clears_t0_and_prev_mean():
    a = _agent()
    a.core._prev_mean.add_(1.0); a._t0 = False
    a.reset()
    assert a._t0 is True
    assert torch.all(a.core._prev_mean == 0.0)
```

- [ ] **Step 6: Run**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/ -v`
Expected: PASS (all tdmpc2 tests). Also run a quick FB regression:
`.venv/bin/python -m pytest tests/ -q -k "fb or rldp" 2>&1 | tail -5` — Expected: unchanged pass (the `reset`/`eval_context` hooks are no-ops for FB).

- [ ] **Step 7: Commit**

```bash
git add agents/tdmpc2/agent.py agents/base.py envs/rollout.py evals/ogbench.py tests/agents/tdmpc2/test_act.py
git commit -m "feat(tdmpc2): planner act + eval_context goal + rollout/evaluator hooks"
```

---

## Task 8: Hydra config + `make_agent` wiring

**Files:**
- Create: `configs/agent/tdmpc2.yaml`
- Modify: `train.py` (`make_agent`)
- Test: `tests/agents/tdmpc2/test_make_agent.py`

- [ ] **Step 1: Write the agent config**

```yaml
# configs/agent/tdmpc2.yaml
# @package _global_
defaults:
  - _self_

agent: tdmpc2
horizon: 3                 # buffer window length == TD-MPC2 planning horizon

tdmpc2:
  random_goal_ratio: 0.5
  success_threshold: 0.04
  # TD-MPC2 hyperparameters (override build_tdmpc2_cfg defaults here if needed)
  latent_dim: 512
  mlp_dim: 512
  num_q: 5
  horizon: 3
  num_samples: 512
  num_elites: 64
  num_pi_trajs: 24
  iterations: 6
  episode_length: 200
```

- [ ] **Step 2: Write the `make_agent` test**

```python
# tests/agents/tdmpc2/test_make_agent.py
import types
from omegaconf import OmegaConf
from train import make_agent


def test_make_agent_builds_tdmpc2():
    cfg = OmegaConf.create({
        "agent": "tdmpc2", "horizon": 3, "device": "cpu", "batch_size": 16,
        "tdmpc2": {"random_goal_ratio": 0.5, "success_threshold": 0.04,
                   "horizon": 3, "num_q": 5, "episode_length": 200},
    })
    obs_space = types.SimpleNamespace(shape=(19,))
    agent = make_agent(cfg, obs_space, action_dim=5)
    assert agent.__class__.__name__ == "TDMPC2Agent"
    assert agent.horizon == 3
```

- [ ] **Step 3: Run to confirm fail**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_make_agent.py -v`
Expected: FAIL (make_agent has no tdmpc2 branch).

- [ ] **Step 4: Wire into `train.py make_agent`**

Add near the top of `make_agent` (before the FB config construction that assumes `cfg.forward` etc., since tdmpc2 has no such groups):
```python
    if cfg.agent == "tdmpc2":
        from agents.tdmpc2 import TDMPC2Agent
        t = cfg.get("tdmpc2", {})
        return TDMPC2Agent(
            obs_space,
            action_dim,
            device=cfg.device,
            horizon=cfg.get("horizon", 3),
            random_goal_ratio=t.get("random_goal_ratio", 0.5),
            success_threshold=t.get("success_threshold", 0.04),
            **{k: v for k, v in t.items()
               if k not in ("random_goal_ratio", "success_threshold")},
        )
```
(Place this as the FIRST branch in `make_agent` so the FB-specific `ForwardArchiConfig(...)` lines never run for tdmpc2.)

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_make_agent.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/agent/tdmpc2.yaml train.py tests/agents/tdmpc2/test_make_agent.py
git commit -m "feat(tdmpc2): hydra config + make_agent wiring"
```

---

## Task 9: End-to-end smoke run on cube-single state

**Files:**
- Create: `scripts/agents/tdmpc2/state.sh`

- [ ] **Step 1: Write the launcher (mirror `scripts/agents/rldp/state.sh`)**

Read `scripts/agents/rldp/state.sh` and copy its structure, substituting `agent=tdmpc2` and dropping FB-only overrides. Keep `wandb_entity=amsks`. Example body:
```bash
#!/usr/bin/env bash
# scripts/agents/tdmpc2/state.sh — TD-MPC2 on OGBench cube-single (state).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SEEDS="${SEEDS:-0}"; GPUS="${GPUS:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
for seed in $SEEDS; do
  CUDA_VISIBLE_DEVICES="${GPUS%% *}" .venv/bin/python "$REPO/train.py" \
    domain=cube_single agent=tdmpc2 obs_type=state \
    num_train_steps="$TRAIN_STEPS" seed="$seed" device=cuda \
    data_path="$DATA_PATH" use_wandb=true wandb_entity=amsks \
    wandb_group="${RUN_GROUP:-tdmpc2_state}" &
done
wait
```

- [ ] **Step 2: Dry smoke — 2k steps, 1 eval, on GPU**

Run:
```bash
SEEDS=0 GPUS=0 TRAIN_STEPS=2000 RUN_GROUP=tdmpc2_smoke \
  bash scripts/agents/tdmpc2/state.sh 2>&1 | tee /dev/shm/tdmpc2_smoke.log
```
Expected: trains without error; prints `[step ...] consistency_loss=... value_loss=...`; reaches an `[eval]` block; no NaNs; `eval/reward/.../success` is a finite number in [0,1]. Success rate near 0 at 2k steps is fine — we are checking it RUNS, not that it solves the task.

- [ ] **Step 3: Inspect for sanity**

Run: `grep -E "consistency_loss|eval/reward/eval/success|Traceback|nan" /dev/shm/tdmpc2_smoke.log | tail -20`
Expected: losses present and finite; no `Traceback`; no `nan`.

- [ ] **Step 4: Commit the launcher**

```bash
git add scripts/agents/tdmpc2/state.sh
git commit -m "feat(scripts): tdmpc2 cube-single state launcher + smoke-verified"
```

---

## Task 10: Phase-transition-prediction probe (the rlbrew MBRL diagnostic)

Decoder-free TD-MPC2 cannot reconstruct observations, so we probe the latent: train a small `latent z -> (cube xyz, cube height, contact)` head on held-out states, then measure prediction error along **imagined** rollouts (`core.model.next`), stratified by phase (approach / grasp / maintain / transport). This produces the numbers for the `\dreamer{}` row and Fig. `f3_failure_composition`.

**Files:**
- Modify: `evals/phase_probe.py` (add `tdmpc2_latent_phase_probe`)
- Test: `tests/agents/tdmpc2/test_phase_probe.py`

- [ ] **Step 1: Read the existing probe to match conventions**

Read `evals/phase_probe.py` and `scripts/phase_probe_aggregate.py` to reuse the cube-height / contact extraction and the phase labels already defined for the FB `B`-embedding probe.

- [ ] **Step 2: Write the probe-fits test (shape + learnability on synthetic data)**

```python
# tests/agents/tdmpc2/test_phase_probe.py
import torch, types
from agents.tdmpc2.agent import TDMPC2Agent
from evals.phase_probe import tdmpc2_latent_phase_probe


def test_probe_returns_expected_keys():
    a = TDMPC2Agent(types.SimpleNamespace(shape=(19,)), 5, device="cpu", horizon=3)
    # synthetic: obs window + physics with cube xyz
    B, H, P = 32, 3, 40
    batch = {
        "observation": torch.randn(B, H, 19),
        "next": {"observation": torch.randn(B, H, 19),
                 "physics": torch.randn(B, H, P)},
    }
    goal = torch.zeros(3)
    out = tdmpc2_latent_phase_probe(a, batch, goal, rollout_len=2)
    for k in ("cube_xyz_mse", "open_loop_cube_mse_by_step"):
        assert k in out
```

- [ ] **Step 3: Implement `tdmpc2_latent_phase_probe`**

```python
def tdmpc2_latent_phase_probe(agent, batch, goal, *, rollout_len=3, ridge=1e-3):
    """Fit a linear probe z -> cube_xyz on encoded states, then measure how
    cube-xyz prediction error grows along an imagined rollout (model.next).
    Returns dict with single-step probe MSE and open-loop MSE per imagined step."""
    import torch
    dev = agent.device
    obs = batch["observation"].to(dev).float()             # [B,H,obs]
    next_phys = batch["next"]["physics"].to(dev).float()
    cube = next_phys[..., 14:17]                            # [B,H,3] targets
    g = goal.to(dev).float()

    with torch.no_grad():
        folded = agent._fold(obs[:, 0, :], g.expand(obs.shape[0], 3))  # [B,obs+3]
        z0 = agent.core.model.encode(folded, None)         # [B,latent]
    # closed-form ridge probe z0 -> cube[:,0]
    Z = z0; Y = cube[:, 0, :]
    W = torch.linalg.solve(Z.T @ Z + ridge * torch.eye(Z.shape[1], device=dev), Z.T @ Y)
    pred0 = Z @ W
    cube_xyz_mse = torch.mean((pred0 - Y) ** 2).item()

    # open-loop imagined rollout error
    open_loop = []
    with torch.no_grad():
        z = z0
        for t in range(min(rollout_len, cube.shape[1])):
            pred = z @ W
            open_loop.append(torch.mean((pred - cube[:, t, :]) ** 2).item())
            a_t = batch["action"][:, t, :].to(dev).float() if "action" in batch \
                else torch.zeros(z.shape[0], agent.action_dim, device=dev)
            z = agent.core.model.next(z, a_t, None)
    return {"cube_xyz_mse": cube_xyz_mse, "open_loop_cube_mse_by_step": open_loop}
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/test_phase_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/phase_probe.py tests/agents/tdmpc2/test_phase_probe.py
git commit -m "feat(tdmpc2): latent->cube phase-transition-prediction probe (rlbrew MBRL diagnostic)"
```

---

## Task 11: Cleanup & fidelity verification

**Files:**
- Delete: `agents/tdmpc2/*.py.orig`

- [ ] **Step 1: Final fidelity diff review**

Run: `for f in init math scale layers world_model core; do echo "== $f =="; diff agents/tdmpc2/$f.py agents/tdmpc2/$f.py.orig; done`
Expected: every hunk corresponds to a row in Task 2's mod table or Task 3's ensemble rewrite. If any hunk is NOT explained by the mod list, it is an accidental porting change — revert it to match upstream.

- [ ] **Step 2: Remove pristine references**

```bash
git rm agents/tdmpc2/*.py.orig 2>/dev/null || rm -f agents/tdmpc2/*.py.orig
```

- [ ] **Step 3: Full test pass**

Run: `.venv/bin/python -m pytest tests/agents/tdmpc2/ -v && .venv/bin/python -m pytest tests/ -q -k "fb or rldp" 2>&1 | tail -5`
Expected: all tdmpc2 tests PASS; FB/RLDP regression PASS.

- [ ] **Step 4: Commit**

```bash
git add -A agents/tdmpc2/
git commit -m "chore(tdmpc2): drop pristine .orig refs after fidelity review"
```

---

## Self-Review

**Spec coverage:**
- Copy TD-MPC2 first-party (not vendor) → Task 1 (verbatim) + Task 2 (mods).
- 100% fidelity except agreed mods → Task 1 `.orig` refs + Task 2 mod table + Task 11 diff review.
- Drop tensordict via torch.func (keep vmap speed) → Task 3.
- Drop compile/cudagraphs, parameterize device → Task 2 (M8–M10).
- Goal-folded, per-task → `GOAL_DIM`/`_fold` (Task 5), `obs_shape` fold (Task 4).
- Reward relabel from physics (distance) → `_reward`/`_sample_goals` (Task 5), used in `update` (Task 6).
- `[B,H,...]`→`[H+1,B,...]` reshape → `update` (Task 6).
- act + planner + per-episode `_prev_mean` reset → Task 7 (`act`/`reset` + rollout hook).
- Wire into train/eval harness → Task 7 (evaluator), Task 8 (make_agent/config).
- Phase-transition-prediction probe for the MBRL row → Task 10.

**Placeholder scan:** No TBD/TODO; all new modules have complete code; copied files have exact edit instructions + diff verification.

**Type/name consistency:** `GOAL_DIM=3` used consistently in config (obs_shape fold), agent (`_fold`/`act`/`eval_context`), tests. `TDMPC2Core` (renamed) used in agent + tests. `update_step`/`_reduce_info` defined in core (Task 6 Step 1), called from agent.update / _update. `eval_context(env, domain, task)` signature matches evaluator call site (Task 7 Step 3 ↔ Step 4). Buffer window length `horizon` == TD-MPC2 `horizon` asserted by config (both 3).

**Known risk (flagged, not a blocker):** sparse reward + short planning horizon (3) on long-horizon cube-single. The smoke run (Task 9) only verifies it RUNS; reaching paper-grade success is a tuning phase after this plan (knobs: `random_goal_ratio`, `horizon`, `consistency_coef`, `discount_*`, longer training). If smoke success is pathologically stuck, first suspects: goal distribution (too-near hindsight goals) and value bootstrap beyond horizon.
