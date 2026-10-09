# Goal-conditioned FB (V1 explicit-goal, V2 fixed-B/USFA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two flag-gated goal-conditioning variants to state-based FB+FlowBC on cube-single — V1 (feed the explicit goal observation into the forward map + actor) and V2 (replace the learned backward map with a fixed `cube_xyz` featurizer / USFA) — plus an `L_dim=256` capacity ablation, and a goal-aware eval path.

**Architecture:** All changes default off → byte-identical to current FB+FlowBC. V2 swaps `_backward_map`/`_bw_encoder` for a parameter-free `FixedCubeBackward` (z_dim becomes 3) and disables the orthonormality loss + backward optimizer. V1 widens `ForwardMap` and `NoiseConditionedActor` inputs by `goal_dim`, threads the (permuted) goal tensor through `sample_mixed_z`/`update_fb`/`update_actor`/`act`. Eval uses the explicit task goal via a new `eval_context` on the FB agent.

**Tech Stack:** PyTorch, Hydra configs, OGBench (mujoco), pytest. Tests run with `.venv/bin/python -m pytest` (NOT `uv run` — it's broken in this repo).

**Reference spec:** `docs/superpowers/specs/2026-06-02-goal-conditioned-fb-design.md`

**Key facts the executor must know (verified):**
- State cube obs layout: `obs[19:22]` = block_0 (cube) xyz, **scaled** as `(pos − [0.425,0,0]) · 10`; effector xyz at `obs[12:15]` (see `agents/tdmpc2/agent.py:16-22`).
- Eval goal: `env.unwrapped.cur_task_info["goal_xyzs"]` is `[num_cubes,3]` in **real metres** (see `agents/tdmpc2/agent.py:142`).
- FB+FlowBC actor: `NoiseConditionedActor.forward(obs, z, noise)`; forward map `ForwardMap.forward(obs, z, action)` (both in `nn_models.py`).
- `actor_encode_obs=false` for state ⇒ the actor's "obs" input is the **raw obs**, the forward map's input is `left_encoder(obs)`.
- The evaluator (`evals/ogbench.py:79`) calls `agent.eval_context(env, domain, task)` if present, else `_infer_z`; returns `(ctx, metrics)`; `ctx` is passed to `rollout(..., ctx=z)` → `agent.act(obs, ctx)`.
- Test env gotcha (memory `factored-fb-test-env`): use `.venv/bin/python -m pytest`; six unrelated modules fail on missing pandas — not regressions.

---

## File Structure

- **Create** `agents/fb/fixed_backward.py` — `FixedCubeBackward` module + a `make_fixed_backward` helper (V2). One responsibility: the parameter-free cube-xyz featurizer used as `B`.
- **Modify** `nn_models.py` — `ForwardMap` and `NoiseConditionedActor` gain an optional `goal_dim` input (V1).
- **Modify** `agents/fb/model.py` — `FBModel.__init__` builds fixed-B when requested, sets `z_dim=3`; `actor`/`forward_map`/`act` accept an optional goal.
- **Modify** `agents/fb/agent.py` — `__init__` (new flags), `setup_training` (skip backward optimizer + targets for fixed-B), `sample_mixed_z` (return the goal used), `update`/`update_fb` (thread goal, disable ortho for fixed-B), `eval_context` (new).
- **Modify** `agents/fb/flow_bc/agent.py` — `update_actor`, `sample_action_from_norm_obs`, `act` thread the goal.
- **Modify** `configs/agent/fb.yaml` + `configs/agent/fb_flowbc.yaml` — new flags `goal_cond`, `fixed_b`.
- **Modify** `evals/ogbench.py` — goal-obs sourcing helper (`g*`) for V1/baseline matched control.
- **Modify** `scripts/agents/fb/state.sh` — `GOAL_COND`, `FIXED_B`, `L_DIM` env-vars.
- **Create** tests under `tests/` per task.

---

## Phase 0 — Config & launcher scaffolding (default-off, byte-identical)

### Task 0.1: Add flags to FB config

**Files:**
- Modify: `configs/agent/fb.yaml`

- [ ] **Step 1: Add the two flags** (append after `onestep: false`):

```yaml
goal_cond: false        # V1: feed the explicit goal obs into forward map + actor
fixed_b: none           # V2: 'none' (learned B) | 'cube_xyz' (fixed USFA featurizer, z_dim->3)
```

- [ ] **Step 2: Verify it loads**

Run: `.venv/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/agent/fb.yaml'); print(c.goal_cond, c.fixed_b)"`
Expected: `False none`

- [ ] **Step 3: Commit**

```bash
git add configs/agent/fb.yaml
git commit -m "feat(fb): add goal_cond + fixed_b flags (default off)"
```

### Task 0.2: Plumb flags through make_agent

**Files:**
- Modify: `train.py` (the `shared = dict(...)` passed to FB agents, ~line 130-165)

- [ ] **Step 1: Read the current `shared` dict** in `train.py` `make_agent` (the block ending around `device=cfg.device,` before the `if cfg.agent == "fb_flowbc"`). Add two entries inside that dict literal:

```python
        goal_cond=cfg.get("goal_cond", False),
        fixed_b=cfg.get("fixed_b", "none"),
```

- [ ] **Step 2: Verify import path compiles**

Run: `.venv/bin/python -c "import ast; ast.parse(open('train.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat(fb): plumb goal_cond/fixed_b through make_agent"
```

### Task 0.3: Add agent ctor params (no behavior yet)

**Files:**
- Modify: `agents/fb/agent.py` (`FBAgent.__init__` signature + attribute storage)

- [ ] **Step 1: Add params** to `FBAgent.__init__` signature (next to `onestep: bool = False,`):

```python
        goal_cond: bool = False,
        fixed_b: str = "none",
```

- [ ] **Step 2: Store them** (right after `self.onestep = bool(onestep)`):

```python
        self.goal_cond = bool(goal_cond)
        self.fixed_b = str(fixed_b)
        # goal_dim: the slice of obs fed to F/actor in V1 (full obs by default).
        self.goal_dim = int(obs_space.shape[0]) if self.goal_cond else 0
        # Only the goal variants use eval_context; baseline keeps legacy reward-inference.
        self._use_eval_context = self.goal_cond or self.fixed_b == "cube_xyz"
        self._eval_goal = None
```

- [ ] **Step 3: Smoke-construct a state FB agent (defaults) and assert flags off**

Test: `tests/test_gc_fb_defaults.py`

```python
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _agent(**kw):
    obs_space = gym.spaces.Box(-1, 1, (40,))
    return FBFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)

def test_defaults_off():
    a = _agent()
    assert a.goal_cond is False and a.fixed_b == "none" and a.goal_dim == 0
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/test_gc_fb_defaults.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/fb/agent.py tests/test_gc_fb_defaults.py
git commit -m "feat(fb): accept goal_cond/fixed_b params (no behavior change)"
```

### Task 0.4: Launcher env-vars (L_DIM, GOAL_COND, FIXED_B)

**Files:**
- Modify: `scripts/agents/fb/state.sh`

- [ ] **Step 1: Add knobs** after `LR_B="${LR_B:-1e-4}"`:

```bash
L_DIM="${L_DIM:-}"            # empty = config default (50); set 256 for the capacity ablation
GOAL_COND="${GOAL_COND:-false}"
FIXED_B="${FIXED_B:-none}"    # none | cube_xyz
```

- [ ] **Step 2: Append to `extra`** after the existing `extra=(...)` line:

```bash
[[ -n "$L_DIM" ]] && extra+=("L_dim=$L_DIM")
[[ "$GOAL_COND" == "true" ]] && extra+=("goal_cond=true")
[[ "$FIXED_B" != "none" ]] && extra+=("fixed_b=$FIXED_B")
```

- [ ] **Step 3: Dry-run all four variants**

Run:
```bash
cd /home/adityamohan/git/Factored-FB
DRY_RUN=1 SEEDS='0' bash scripts/agents/fb/state.sh
DRY_RUN=1 SEEDS='0' L_DIM=256 bash scripts/agents/fb/state.sh
DRY_RUN=1 SEEDS='0' GOAL_COND=true bash scripts/agents/fb/state.sh
DRY_RUN=1 SEEDS='0' FIXED_B=cube_xyz bash scripts/agents/fb/state.sh
```
Expected: each prints a `train.py` command; the last three include `L_dim=256` / `goal_cond=true` / `fixed_b=cube_xyz` respectively.

- [ ] **Step 4: Commit**

```bash
git add scripts/agents/fb/state.sh
git commit -m "feat(fb): launcher env-vars for L_DIM/GOAL_COND/FIXED_B variants"
```

---

## Phase 1 — V2: fixed-B / USFA (`fixed_b=cube_xyz`)

### Task 1.1: `FixedCubeBackward` module

**Files:**
- Create: `agents/fb/fixed_backward.py`
- Test: `tests/test_fixed_backward.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from agents.fb.fixed_backward import FixedCubeBackward, CUBE_OBS_SLICE

def test_extracts_scaled_cube_xyz_and_has_no_params():
    b = FixedCubeBackward(cube_slice=CUBE_OBS_SLICE)
    obs = torch.zeros(4, 40)
    obs[:, CUBE_OBS_SLICE] = torch.tensor([1.0, 2.0, 3.0])
    out = b(obs)
    assert out.shape == (4, 3)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]).expand(4, 3))
    assert list(b.parameters()) == []   # parameter-free

def test_output_space_is_3():
    b = FixedCubeBackward(cube_slice=CUBE_OBS_SLICE)
    assert b.output_space.shape == (3,)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fixed_backward.py -v`
Expected: FAIL (`ModuleNotFoundError: agents.fb.fixed_backward`)

- [ ] **Step 3: Implement**

```python
"""agents/fb/fixed_backward.py — V2 (USFA): parameter-free backward map B(s) = cube xyz.

Replaces the learned FB backward map with the privileged goal variable (the cube
position), so B encodes the goal geometry by construction (R^2(B->d)=1). See
docs/superpowers/specs/2026-06-02-goal-conditioned-fb-design.md.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

# Verified state cube-single layout (agents/tdmpc2/agent.py:18-22): block_0 xyz,
# stored scaled as (pos - [0.425,0,0]) * 10.
CUBE_OBS_SLICE = slice(19, 22)
OBS_XYZ_OFFSET = (0.425, 0.0, 0.0)
OBS_XYZ_SCALE = 10.0


class FixedCubeBackward(nn.Module):
    """B(obs) = obs[cube_slice]  (the scaled cube xyz). No learnable parameters."""

    def __init__(self, cube_slice: slice = CUBE_OBS_SLICE) -> None:
        super().__init__()
        self.cube_slice = cube_slice
        self._output_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    @property
    def output_space(self):
        return self._output_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.cube_slice]


def scale_goal_xyz(goal_xyz: torch.Tensor) -> torch.Tensor:
    """Map a real-metre goal cube xyz into the scaled obs space FixedCubeBackward sees."""
    offset = torch.tensor(OBS_XYZ_OFFSET, dtype=goal_xyz.dtype, device=goal_xyz.device)
    return (goal_xyz - offset) * OBS_XYZ_SCALE
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fixed_backward.py -v`
Expected: PASS

> **Step 4a — VERIFY THE SLICE/SCALE against real data before trusting it** (do this once, record the result in the commit message):
> ```bash
> MUJOCO_GL=glfw .venv/bin/python - <<'PY'
> import numpy as np, glob
> f = sorted(glob.glob('datasets/cube-single-play-v0/buffer/episode_*.npz'))[0]
> ep = np.load(f)
> obs, phys = ep['observation'], ep['physics']
> cube_phys = phys[:, 14:17]                      # ground-truth cube xyz (metres)
> cube_obs  = obs[:, 19:22]                        # candidate scaled slice
> recon = cube_obs/10.0 + np.array([0.425,0,0])    # invert the documented transform
> print('max abs err (recon vs phys):', np.abs(recon - cube_phys).max())
> PY
> ```
> Expected: error ≈ 0 (≤1e-3). If not, fix `CUBE_OBS_SLICE`/offset/scale before proceeding.

- [ ] **Step 5: Commit**

```bash
git add agents/fb/fixed_backward.py tests/test_fixed_backward.py
git commit -m "feat(fb): FixedCubeBackward (V2 USFA featurizer) + slice/scale verified"
```

### Task 1.2: Wire fixed-B into the model

**Files:**
- Modify: `agents/fb/model.py` (`FBModel.__init__`)

- [ ] **Step 1: Add a `fixed_b` kwarg** to `FBModel.__init__` (default `"none"`), store `self.fixed_b = fixed_b`. Pass it from `FBFlowBCModel` (it already forwards `**kwargs`) — confirm `FBFlowBCModel.__init__` passes through.

- [ ] **Step 2: Replace the backward construction.** After the existing `self._bw_encoder = rgb_encoder_cfg.build(obs_space)` and where `self._backward_map = backward_cfg.build(...)` is built, branch:

```python
        if self.fixed_b == "cube_xyz":
            from agents.fb.fixed_backward import FixedCubeBackward
            self._bw_encoder = torch.nn.Identity()           # raw obs -> backward
            self._backward_map = FixedCubeBackward()         # B(obs) = scaled cube xyz, z_dim=3
            self.z_dim = 3
            z_dim = 3                                         # forward map output dim follows
        else:
            self._backward_map = backward_cfg.build(self._bw_encoder.output_space, z_dim)
```

> NOTE: `_bw_encoder` for state is already Identity; reassigning to `nn.Identity()` is safe. The `_forward_map` built next uses `z_dim` — ensure the `z_dim=3` reassignment happens BEFORE `self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim)`.

- [ ] **Step 3: Make `backward_map` tolerate a no-op encoder.** `FBModel.backward_map` currently calls `self._backward_map(self._bw_encoder(self._normalize(obs)))`. With Identity encoder this still works (Identity passes the normalized obs through). No change needed; just confirm by the test below.

- [ ] **Step 4: Test the model builds with z_dim=3 and B returns cube xyz**

Test: `tests/test_fixed_b_model.py`

```python
import torch, gymnasium as gym
from agents.fb.model import FBModel
from agents.fb.fixed_backward import CUBE_OBS_SLICE

def test_fixed_b_model_zdim3():
    obs_space = gym.spaces.Box(-1, 1, (40,))
    m = FBModel(obs_space=obs_space, action_dim=5, z_dim=50, L_dim=50, fixed_b="cube_xyz", device="cpu")
    assert m.z_dim == 3
    obs = torch.zeros(2, 40); obs[:, CUBE_OBS_SLICE] = torch.tensor([0.1, 0.2, 0.3])
    b = m.backward_map(obs)
    assert b.shape == (2, 3)
```

- [ ] **Step 5: Run it**

Run: `.venv/bin/python -m pytest tests/test_fixed_b_model.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agents/fb/model.py tests/test_fixed_b_model.py
git commit -m "feat(fb): build FixedCubeBackward in FBModel when fixed_b=cube_xyz (z_dim->3)"
```

### Task 1.3: Skip backward optimizer/targets + ortho loss for fixed-B

**Files:**
- Modify: `agents/fb/agent.py` (`setup_training`, `update_fb`)

- [ ] **Step 1: Guard the backward optimizer.** In `setup_training`, the `self.backward_optimizer = torch.optim.Adam(list(...bw params...))` will get an empty param list for fixed-B and raise. Wrap:

```python
        bw_params = list(self.model._backward_map.parameters()) + list(self.model._bw_encoder.parameters())
        self.backward_optimizer = (
            torch.optim.Adam(bw_params, lr=self.lr_b, weight_decay=self.weight_decay)
            if bw_params else None
        )
```

- [ ] **Step 2: Guard backward target prep.** `_prepare_for_train` (in `model.py`) deep-copies `_target_backward_map`. For fixed-B the target is the same parameter-free function — harmless, leave it. But in `update_fb`, guard the optimizer ops:

Replace `self.backward_optimizer.zero_grad(set_to_none=True)` with:
```python
        if self.backward_optimizer is not None:
            self.backward_optimizer.zero_grad(set_to_none=True)
```
and `self.backward_optimizer.step()` with:
```python
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()
```

- [ ] **Step 3: Disable ortho loss for fixed-B.** In `update_fb`, the ortho block adds `self.ortho_coef * orth_loss`. Gate it (fixed B should not be orthonormalized):

```python
            if self.fixed_b != "cube_xyz":
                Cov = ortho_cov(B, w)
                orth_loss_diag = -Cov.diag().mean()
                orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
                orth_loss = orth_loss_offdiag + orth_loss_diag
                fb_loss += self.ortho_coef * orth_loss
            else:
                orth_loss = orth_loss_diag = orth_loss_offdiag = torch.zeros(1, device=z.device, dtype=z.dtype)
```

(Keeps the metrics dict keys valid.)

- [ ] **Step 4: Test one update step runs for fixed-B (no crash, finite loss)**

Test: `tests/test_fixed_b_update.py`

```python
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _batch(n=8, obs=40, act=5):
    return {
        "observation": torch.randn(n, obs),
        "action": torch.rand(n, act) * 2 - 1,
        "next": {"observation": torch.randn(n, obs),
                 "action": torch.rand(n, act) * 2 - 1,
                 "physics": torch.randn(n, 45)},
        "terminated": torch.zeros(n, 1),
    }

def test_fixed_b_update_runs():
    a = FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", fixed_b="cube_xyz")
    assert a.backward_optimizer is None
    m = a.update(_batch(), step=0)
    assert torch.isfinite(m["fb_loss"]).all()
```

- [ ] **Step 5: Run it**

Run: `.venv/bin/python -m pytest tests/test_fixed_b_update.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agents/fb/agent.py tests/test_fixed_b_update.py
git commit -m "feat(fb): fixed-B skips backward optim + ortho loss; update step verified"
```

---

## Phase 2 — V1: explicit goal-conditioning (`goal_cond=true`)

### Task 2.1: `ForwardMap` optional goal input

**Files:**
- Modify: `nn_models.py` (`ForwardArchiConfig.build`, `ForwardMap`)
- Test: `tests/test_forwardmap_goal.py`

- [ ] **Step 1: Write the failing test**

```python
import torch, gymnasium as gym
from nn_models import ForwardArchiConfig

def test_forwardmap_accepts_goal():
    cfg = ForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=2)
    fm = cfg.build(gym.spaces.Box(-1, 1, (16,)), z_dim=50, action_dim=5, goal_dim=16)
    out = fm(torch.randn(4, 16), torch.randn(4, 50), torch.randn(4, 5), goal=torch.randn(4, 16))
    assert out.shape == (2, 4, 50)

def test_forwardmap_goal_none_default():
    cfg = ForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=2)
    fm = cfg.build(gym.spaces.Box(-1, 1, (16,)), z_dim=50, action_dim=5)  # goal_dim defaults 0
    out = fm(torch.randn(4, 16), torch.randn(4, 50), torch.randn(4, 5))
    assert out.shape == (2, 4, 50)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forwardmap_goal.py -v`
Expected: FAIL (`build() got an unexpected keyword argument 'goal_dim'`)

- [ ] **Step 3: Implement.** Thread `goal_dim` through `ForwardArchiConfig.build` and `_build_batch_forward` (add `goal_dim=0` params, pass to `ForwardMap`). In `ForwardMap.__init__`, accept `goal_dim: int = 0`, store `self.goal_dim`, and widen the `embed_z` input by `goal_dim`:

```python
        self.goal_dim = goal_dim
        self.embed_z = simple_embedding(obs_dim + z_dim + goal_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
```

And in `ForwardMap.forward`, add `goal: torch.Tensor | None = None`:

```python
    def forward(self, obs, z, action, goal=None):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
            if goal is not None:
                goal = goal.expand(self.num_parallel, -1, -1)
        z_in = torch.cat([obs, z], dim=-1) if goal is None else torch.cat([obs, z, goal], dim=-1)
        z_embedding = self.embed_z(z_in)
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forwardmap_goal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn_models.py tests/test_forwardmap_goal.py
git commit -m "feat(nn): ForwardMap optional goal input (goal_dim, default 0 = unchanged)"
```

### Task 2.2: `NoiseConditionedActor` optional goal input

**Files:**
- Modify: `nn_models.py` (`NoiseConditionedActorArchiConfig.build`, `NoiseConditionedActor`)
- Test: `tests/test_actor_goal.py`

- [ ] **Step 1: Write the failing test**

```python
import torch, gymnasium as gym
from nn_models import NoiseConditionedActorArchiConfig

def test_actor_accepts_goal():
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2)
    act = cfg.build(gym.spaces.Box(-1, 1, (40,)), z_dim=50, action_dim=5, goal_dim=40)
    a = act(torch.randn(4, 40), torch.randn(4, 50), torch.randn(4, 5), goal=torch.randn(4, 40))
    assert a.shape == (4, 5) and a.abs().max() <= 1.0

def test_actor_goal_none_default():
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2)
    act = cfg.build(gym.spaces.Box(-1, 1, (40,)), z_dim=50, action_dim=5)
    a = act(torch.randn(4, 40), torch.randn(4, 50), torch.randn(4, 5))
    assert a.shape == (4, 5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_actor_goal.py -v`
Expected: FAIL (`build() got an unexpected keyword argument 'goal_dim'`)

- [ ] **Step 3: Implement.** `NoiseConditionedActorArchiConfig.build` gains `goal_dim: int = 0` and passes it to `NoiseConditionedActor`. In `__init__`, store `self.goal_dim`, widen `embed_z`:

```python
        self.goal_dim = goal_dim
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim + goal_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
```

`forward` gains `goal=None`:

```python
    def forward(self, obs, z, noise, goal=None):
        z_in = [obs, z, noise] if goal is None else [obs, z, noise, goal]
        z_embedding = self.embed_z(torch.cat(z_in, dim=-1))
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))
        return torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_actor_goal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn_models.py tests/test_actor_goal.py
git commit -m "feat(nn): NoiseConditionedActor optional goal input (default 0 = unchanged)"
```

### Task 2.3: Build goal-conditioned maps in the model + pass goal_dim

**Files:**
- Modify: `train.py` (forward_cfg / actor build pass `goal_dim`), `agents/fb/model.py` (forward/actor build), `agents/fb/agent.py` (`self.goal_dim`)

- [ ] **Step 1: Pass `goal_dim` into the forward + actor builds.** In `agents/fb/model.py.__init__`, the model already knows `obs_space`. Add `goal_dim` kwarg (default 0), store `self.goal_dim`, and pass it to the two builds:

```python
        self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim, goal_dim=self.goal_dim) \
            if hasattr(forward_cfg, "build") else None
```

> Simpler: change the existing `self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim)` to append `, goal_dim=self.goal_dim`, and the actor build similarly. `ForwardArchiConfig.build`/`SimpleActorArchiConfig.build`/`NoiseConditionedActorArchiConfig.build` all now accept `goal_dim` (Tasks 2.1, 2.2; add the same default to `SimpleActorArchiConfig.build` signature even though state FlowBC uses noise-conditioned — keep signatures uniform).

- [ ] **Step 2: Pass `goal_dim` from agent → model.** In `FBAgent.__init__`, the `self._make_model(...)` call adds `goal_dim=self.goal_dim`.

- [ ] **Step 3: Verify model builds goal-conditioned**

Test: append to `tests/test_actor_goal.py`:

```python
def test_fb_model_goal_cond_builds():
    import gymnasium as gym
    from agents.fb.model import FBModel
    m = FBModel(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, z_dim=50, L_dim=50, goal_dim=40, device="cpu")
    assert m.goal_dim == 40
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/test_actor_goal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train.py agents/fb/model.py agents/fb/agent.py tests/test_actor_goal.py
git commit -m "feat(fb): thread goal_dim into forward map + actor construction"
```

### Task 2.4: Thread the goal through `sample_mixed_z` (return matched goal)

**Files:**
- Modify: `agents/fb/agent.py` (`sample_mixed_z`, `update`)

- [ ] **Step 1: Make `sample_mixed_z` also return the goal used per row.** The current method permutes `train_goal` and mixes `z=B(g_perm)` with random z under `mask`. For V1 the goal fed to F/π must be the SAME row-wise goal that produced z. Change the return to `(z, g_used)` where `g_used` is the permuted goal for goal-rows and a **zero placeholder** for random-z rows:

```python
    @torch.no_grad()
    def sample_mixed_z(self, train_goal=None, goal_weights=None, *args, **kwargs):
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            z = self.model.sample_z(self.batch_size, device=self.device)
            g_used = torch.zeros((self.batch_size, train_goal.shape[-1]), device=self.device) \
                if train_goal is not None else None
            if train_goal is not None:
                if goal_weights is None:
                    perm = torch.randperm(self.batch_size, device=self.device)
                else:
                    probs = goal_weights.float() / goal_weights.float().sum().clamp_min(1e-8)
                    perm = torch.multinomial(probs, self.batch_size, replacement=True)
                tg = train_goal[perm]
                goals = self.model.project_z(self.model._backward_map(tg))
                mask = torch.rand((self.batch_size, 1), device=self.device) < self.train_goal_ratio
                z = torch.where(mask, goals, z)
                g_used = torch.where(mask, tg, g_used)   # zero placeholder where z is random
        return z, g_used
```

> The mask flag is implicit: random-z rows have `g_used == 0`. F/π see a zero goal there, which is a valid fixed placeholder (the executor must NOT also feed `z=B(0)` — z stays random there).

- [ ] **Step 2: Update the caller in `update`.** Where `z = self.sample_mixed_z(train_goal=goal, ...).clone()` is called, change to:

```python
        z, g_used = self.sample_mixed_z(train_goal=goal,
                                        goal_weights=(w if self.weight_z else None))
        z = z.clone()
        self._g_used = g_used if self.goal_cond else None   # stashed for update_fb/update_actor
```

- [ ] **Step 3: Update existing callers/tests that unpack a single return.** Grep: `grep -rn "sample_mixed_z(" agents/ tests/`. Any other caller (e.g. tests) must unpack the tuple. Fix each.

- [ ] **Step 4: Verify the non-goal path is unchanged (z distribution identical seed-for-seed when goal_cond off)**

Test: `tests/test_sample_mixed_z_tuple.py`

```python
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _agent(**kw):
    return FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)

def test_returns_tuple_and_placeholder_zeros():
    a = _agent()
    g = torch.randn(8, 40)
    torch.manual_seed(0)
    z, g_used = a.sample_mixed_z(train_goal=g)
    assert z.shape == (8, 50) and g_used.shape == (8, 40)
    # rows where z came from random (not a goal) must have zero placeholder OR match a goal row
    is_zero = (g_used.abs().sum(-1) == 0)
    assert is_zero.any() or (~is_zero).all()
```

- [ ] **Step 5: Run it (and the full suite to catch unpack breaks)**

Run: `.venv/bin/python -m pytest tests/test_sample_mixed_z_tuple.py tests/test_fixed_b_update.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agents/fb/agent.py tests/test_sample_mixed_z_tuple.py
git commit -m "feat(fb): sample_mixed_z returns matched (z, goal); stash for goal-cond"
```

### Task 2.5: Feed goal into the FB loss + actor update

**Files:**
- Modify: `agents/fb/agent.py` (`update_fb` — pass goal to `_forward_map` calls), `agents/fb/flow_bc/agent.py` (`update_actor`, `sample_action_from_norm_obs`)

- [ ] **Step 1: Pass goal into `update_fb`'s forward calls.** `update_fb` already receives `goal` (the backward-map input = next_obs encoding). For V1 the *forward* map and the policy that picks `next_action` must also see the goal. Add a `goal_obs` param to `update_fb` (the raw `g_used`), default None, and use it:

  - target next action: `self.sample_action_from_norm_obs(actor_in, z, goal_obs)` (goal-aware policy) — see Step 2 for the signature.
  - `target_Fs = self.model._target_forward_map(next_left_enc, zf, next_action, goal=goal_obs)`
  - `Fs = self.model._forward_map(left_enc, zf, action, goal=goal_obs)`

  Wrap each `goal=...` so `goal_obs=None` (default / goal_cond off) reproduces the current call exactly:
```python
        fwd_kw = {} if goal_obs is None else {"goal": goal_obs}
        ...
        target_Fs = self.model._target_forward_map(next_left_enc, zf, next_action, **fwd_kw)
        ...
        Fs = self.model._forward_map(left_enc, zf, action, **fwd_kw)
```

- [ ] **Step 2: Make `sample_action_from_norm_obs` goal-aware** (FlowBC override in `flow_bc/agent.py` and the base in `agent.py`). FlowBC version:

```python
    def sample_action_from_norm_obs(self, obs, z, goal=None):
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        gk = {} if goal is None else {"goal": goal}
        return self.model._actor(obs, z, noises, **gk)
```

- [ ] **Step 3: Pass `self._g_used` from `update` into `update_fb`.** Where `update` calls `self.update_fb(...)`, add `goal_obs=self._g_used`.

- [ ] **Step 4: Feed goal into the FlowBC actor update.** In `flow_bc/agent.py.update_actor`, add a `goal=None` param; thread it:
  - `actor_actions = self.model._actor(actor_in, z, noises, **({} if goal is None else {"goal": goal}))`
  - `Fs = self.model._forward_map(left_enc, self._fb_z(z), actor_actions, **({} if goal is None else {"goal": goal}))`
  And in `update`'s actor call, pass `goal=self._g_used`.

> The BC flow vector field (`_actor_vf`) stays goal-UNconditioned (it's behavior cloning of the data action distribution; the goal enters only the Q-driven actor head and F). Document this in the commit message.

- [ ] **Step 5: Test a full V1 update step runs and matches baseline when goal_cond off**

Test: `tests/test_goal_cond_update.py`

```python
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _agent(**kw):
    return FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)

def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
            "next": {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
                     "physics": torch.randn(n, 45)}, "terminated": torch.zeros(n, 1)}

def test_v1_update_runs():
    a = _agent(goal_cond=True)
    m = a.update(_batch(), step=0)
    assert torch.isfinite(m["fb_loss"]).all()
```

- [ ] **Step 6: Run it**

Run: `.venv/bin/python -m pytest tests/test_goal_cond_update.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add agents/fb/agent.py agents/fb/flow_bc/agent.py tests/test_goal_cond_update.py
git commit -m "feat(fb): V1 threads goal into FB loss + FlowBC actor (vf stays uncond)"
```

### Task 2.6: Goal-aware `act` for eval

**Files:**
- Modify: `agents/fb/flow_bc/agent.py` (`act`), `agents/fb/model.py` (`act`/`actor`)

- [ ] **Step 1: Let `act` use a stashed eval goal.** The evaluator calls `agent.act(obs, ctx)` with `ctx=z`. For V1 the actor needs the goal. Store the eval goal on the agent in `eval_context` (Task 3.1) as `self._eval_goal`. In `flow_bc/agent.py.act`:

```python
    @torch.no_grad()
    def act(self, obs, z, *, eval_mode: bool = True):
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        goal = getattr(self, "_eval_goal", None) if self.goal_cond else None
        return self.model.act(obs, z, mean=True, goal=goal)
```

- [ ] **Step 2: Thread goal through `FBModel.act`/`actor`.** Add `goal=None` to `FBModel.act` and `FBModel.actor`; pass to `self._actor(obs, z, std/noise, goal=goal)` (the noise-conditioned actor for FlowBC takes noise; check the model's `act` path — for FlowBC the agent's `sample_action_from_norm_obs` is the real eval action sampler). If `FBModel.act` is unused by FlowBC at eval, route `act` to `self.sample_action_from_norm_obs(obs, z, goal)` in the FlowBC agent instead. Verify which path eval uses by Step 3.

- [ ] **Step 3: Test eval act runs with and without a stashed goal**

Test: append to `tests/test_goal_cond_update.py`:

```python
def test_v1_act_with_goal():
    a = _agent(goal_cond=True)
    a._eval_goal = torch.zeros(1, 40)
    out = a.act(torch.zeros(1, 40), torch.zeros(50))
    assert out.shape[-1] == 5
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/test_goal_cond_update.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/fb/flow_bc/agent.py agents/fb/model.py tests/test_goal_cond_update.py
git commit -m "feat(fb): goal-aware act() for V1 eval (uses stashed _eval_goal)"
```

---

## Phase 3 — Goal-aware eval (`eval_context`)

### Task 3.1: `eval_context` on the FB agent

**Files:**
- Modify: `agents/fb/agent.py` (add `eval_context`), `evals/ogbench.py` (goal-obs sourcing helper)

- [ ] **Step 1: VERIFY where a goal *observation* comes from.** Run, once:

```bash
MUJOCO_GL=glfw .venv/bin/python - <<'PY'
from envs.ogbench import create_ogbench_env, ALL_TASKS
task = ALL_TASKS["cube-single-play-v0"][0]
env, _ = create_ogbench_env(task, seed=0, obs_type="state")
info = env.unwrapped.cur_task_info
print("cur_task_info keys:", list(info.keys()))
obs, _ = env.reset()
print("obs shape:", obs.shape if hasattr(obs, "shape") else type(obs))
# look for a goal observation
for k in info:
    print(k, getattr(info[k], "shape", info[k]))
PY
```
Record what's available. Decision: if a goal **observation** exists (e.g. `goal` obs / `env.unwrapped.get_obs(goal=...)`), use it for `g*`. Else fall back to a **canonical near-goal data state**: from the relabel subsample (the reward-inference path), take the obs of the highest-reward state for this task.

- [ ] **Step 1b: Switch the evaluator dispatch off `hasattr`.** In `evals/ogbench.py:79`, `hasattr(self.agent, "eval_context")` would catch the baseline too (the method always exists on FBAgent). Change the guard to:

```python
                if getattr(self.agent, "_use_eval_context", False):
                    z, relabel_metrics = self.agent.eval_context(env=env, domain=self.domain, task=task)
                else:
                    z, relabel_metrics = self._infer_z(task)
```
This keeps baseline + L256 on the legacy reward-inference path and routes only V1/V2 through `eval_context`.

- [ ] **Step 2: Implement `eval_context`** in `FBAgent`:

```python
    @torch.no_grad()
    def eval_context(self, *, env, domain, task):
        """Goal-conditioned eval context. Returns (z, metrics). Also stashes the
        goal observation on self._eval_goal for goal_cond act()."""
        goal_xyzs = env.unwrapped.cur_task_info["goal_xyzs"]      # [num_cubes, 3], metres
        if self.fixed_b == "cube_xyz":
            from agents.fb.fixed_backward import scale_goal_xyz
            g = torch.as_tensor(goal_xyzs[0], dtype=torch.float32, device=self.device)
            z = self.model.project_z(scale_goal_xyz(g).unsqueeze(0)).squeeze(0)
            self._eval_goal = None
            return z, {"goal_source": 0.0}
        # V1 / baseline matched control: z = B(g*) from a goal OBSERVATION.
        g_obs = self._goal_observation(env, task)                # [obs_dim] tensor (Step 3)
        self._eval_goal = g_obs.unsqueeze(0) if self.goal_cond else None
        z = self.model.backward_map(g_obs.unsqueeze(0))
        z = self.model.project_z(z).squeeze(0)
        return z, {"goal_source": 1.0}
```

- [ ] **Step 3: Implement `_goal_observation`** per the Step-1 finding. If a goal obs is directly available:

```python
    def _goal_observation(self, env, task):
        # Preferred: env exposes a goal observation. Adjust attribute per Step-1 finding.
        info = env.unwrapped.cur_task_info
        if "goal_obs" in info and info["goal_obs"] is not None:
            return torch.as_tensor(info["goal_obs"], dtype=torch.float32, device=self.device)
        # Fallback: highest-reward state from the offline relabel subsample.
        raise NotImplementedError("set per Step-1 finding; see fallback in evals/ogbench.py")
```

> If the fallback is needed, implement the "highest-reward data state" sourcing in `evals/ogbench.py` (it already builds `relabel_fn` and samples the buffer in `_infer_z`) and pass `g*` into `eval_context` via an extra arg, OR give the agent a handle to the offline buffer. Decide based on Step-1; keep it in ONE place.

- [ ] **Step 4: Test eval_context returns a projected z of the right dim for both variants** (CPU, mock env):

Test: `tests/test_eval_context.py`

```python
import torch, types, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
from agents.fb.fixed_backward import CUBE_OBS_SLICE

def _agent(**kw):
    return FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)

def _fake_env():
    e = types.SimpleNamespace()
    e.unwrapped = types.SimpleNamespace(cur_task_info={"goal_xyzs": [[0.3, 0.0, 0.05]], "goal_obs": [0.0]*40})
    return e

def test_fixed_b_eval_context_zdim3():
    a = _agent(fixed_b="cube_xyz")
    z, m = a.eval_context(env=_fake_env(), domain="cube-single-play-v0", task="t1")
    assert z.shape == (3,)

def test_v1_eval_context_stashes_goal():
    a = _agent(goal_cond=True)
    z, m = a.eval_context(env=_fake_env(), domain="cube-single-play-v0", task="t1")
    assert z.shape == (50,) and a._eval_goal.shape == (1, 40)
```

- [ ] **Step 5: Run it**

Run: `.venv/bin/python -m pytest tests/test_eval_context.py -v`
Expected: PASS (adjust the `goal_obs` key to match the Step-1 finding)

- [ ] **Step 6: Commit**

```bash
git add agents/fb/agent.py evals/ogbench.py tests/test_eval_context.py
git commit -m "feat(fb): eval_context — z=phi(goal) for V2, z=B(g*)+stash for V1/baseline"
```

### Task 3.2: Reset clears the stashed eval goal

**Files:**
- Modify: `agents/fb/agent.py` (`reset` if present, else add)

- [ ] **Step 1: Ensure `reset()` clears `_eval_goal`** so it doesn't leak across tasks:

```python
    def reset(self) -> None:
        self._eval_goal = None
```

(If `FBAgent` already defines `reset`, add the line.)

- [ ] **Step 2: Commit**

```bash
git add agents/fb/agent.py
git commit -m "fix(fb): clear stashed eval goal on reset"
```

---

## Phase 4 — Integration smoke + launch

### Task 4.1: End-to-end tiny train (each variant, 50 steps, CPU)

**Files:**
- Test: `tests/test_gc_fb_e2e.py`

- [ ] **Step 1: Write a 50-step train smoke for all four configs** using a tiny synthetic `DictBuffer` (mirror `_batch` above repeatedly). Assert finite eval-less loop:

```python
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
import pytest

CFGS = [dict(), dict(L_dim=256), dict(goal_cond=True), dict(fixed_b="cube_xyz")]

def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
            "next": {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
                     "physics": torch.randn(n, 45)}, "terminated": torch.zeros(n, 1)}

@pytest.mark.parametrize("extra", CFGS)
def test_50_steps(extra):
    L = extra.pop("L_dim", 50)
    a = FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=L,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **extra)
    for s in range(50):
        m = a.update(_batch(), step=s)
    assert torch.isfinite(m["fb_loss"]).all()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_gc_fb_e2e.py -v`
Expected: 4 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_gc_fb_e2e.py
git commit -m "test(fb): 50-step e2e smoke for all four variants"
```

### Task 4.2: Full regression + 2k-step GPU sanity per variant

- [ ] **Step 1: Run the whole suite** (expect the 6 pre-existing pandas failures only — see memory `factored-fb-test-env`):

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all new tests PASS; only the known pandas-missing modules fail.

- [ ] **Step 2: 2k-step GPU run per variant** (real data, short) to confirm training + one eval works end-to-end:

```bash
cd /home/adityamohan/git/Factored-FB
for V in "" "L_DIM=256" "GOAL_COND=true" "FIXED_B=cube_xyz"; do
  env $V SEEDS='0' TRAIN_STEPS=2000 RUN_GROUP=gcfb_smoke bash scripts/agents/fb/state.sh
done
```
Expected: each completes, logs an `eval/success` (any value), no crash in `eval_context`/`act`.

- [ ] **Step 3: Commit any fixes**, then STOP for review before the full campaign.

### Task 4.3: Launch the 10-seed campaign (after review)

- [ ] **Step 1: Launch all four variants, 10 seeds @1M**, wandb entity `amsks` (memory `wandb-entity-amsks` — verify before launch):

```bash
cd /home/adityamohan/git/Factored-FB
SEEDS='0 1 2 3 4 5 6 7 8 9' TRAIN_STEPS=1000000 RUN_GROUP=fb_state_baseline  bash scripts/agents/fb/state.sh
SEEDS='0 1 2 3 4 5 6 7 8 9' TRAIN_STEPS=1000000 L_DIM=256 RUN_GROUP=fb_state_L256 bash scripts/agents/fb/state.sh
SEEDS='0 1 2 3 4 5 6 7 8 9' TRAIN_STEPS=1000000 GOAL_COND=true RUN_GROUP=fb_state_v1 bash scripts/agents/fb/state.sh
SEEDS='0 1 2 3 4 5 6 7 8 9' TRAIN_STEPS=1000000 FIXED_B=cube_xyz RUN_GROUP=fb_state_v2 bash scripts/agents/fb/state.sh
```

- [ ] **Step 2: Record run dirs/wandb groups** for the success-rate comparison (vs baseline-L50 and GCIQL's existing numbers).

---

## Self-Review notes
- **Spec coverage:** baseline (existing) ✓; L256 ablation (Task 0.4 + 4.3) ✓; V1 (Phase 2) ✓; V2 (Phase 1) ✓; eval matched-control z=B(g*) + V2 z=φ(goal) (Phase 3) ✓; legacy reward-inference eval preserved (baseline keeps `_infer_z`; eval_context only added — baseline runs WITHOUT goal_cond/fixed_b so it still hits `eval_context`? NOTE: adding `eval_context` to FBAgent means baseline now uses it too → it would switch baseline to z=B(g*). **This is intended as the matched control**, but to ALSO keep the legacy number, gate `eval_context` on `goal_cond or fixed_b` so plain baseline still falls through to `_infer_z`).
- **Placeholder scan:** the only deferred decision is the `g*` source (Task 3.1 Step 1), which is an explicit VERIFY step with a concrete fallback — not a silent TODO.
- **Type consistency:** `sample_mixed_z` now returns `(z, g_used)` everywhere (Task 2.4 Step 3 fixes all callers); `goal=`/`goal_dim=` names match across `ForwardMap`, `NoiseConditionedActor`, `FBModel`, agent calls.

## CRITICAL gate to fix during Task 3.1
Gate `eval_context` so it only activates for the goal variants, preserving baseline's legacy eval:
```python
    @torch.no_grad()
    def eval_context(self, *, env, domain, task):
        if not (self.goal_cond or self.fixed_b == "cube_xyz"):
            raise AttributeError("baseline uses reward-inference _infer_z")  # evaluator falls back
```
> `evals/ogbench.py:79` uses `hasattr(agent, "eval_context")`. Since the method always exists on FBAgent, change that guard to a try/except or to `getattr(agent, "_use_eval_context", False)`. Implement a `self._use_eval_context = self.goal_cond or self.fixed_b == "cube_xyz"` flag in `__init__` and switch the evaluator to check it. (Add this to Task 0.3 Step 2 and Task 3.1.)
