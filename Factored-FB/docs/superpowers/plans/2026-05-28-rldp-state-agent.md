# RLDP State Agent (sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `agents/rldp/` (plain RLDP + RLDP+FlowBC variants) for state-only cube-single, plumbed through configs/launchers/train.py.

**Architecture:** Adapt td_jepa's `agents/rldp/{model.py, agent.py, flow_bc/}` to our kwarg-based class pattern (the same recipe we used for FB and FB+FlowBC). RLDPModel subclasses FBModel and adds `_predictor`; RLDPAgent subclasses FBAgent and overrides `setup_training`+`update`+`update_fb` to add the self-predictive (SP) loss; RLDPFlowBC pair mirrors FBFlowBC. Buffer-windowed batches are sliced inline in `RLDPAgent.update()`. One conditional in `train.py` forwards `cfg.horizon` to `buffer.sample()`.

**Tech Stack:** PyTorch, numpy, Hydra, our existing FB infrastructure. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-28-rldp-state-agent-design.md` (with the spec's open-risk "literal copy vs adapt" resolved during plan writing — we adapt to our kwarg pattern, the same recipe as `agents/fb/flow_bc/`).

**Conventions:**
- Tests run via `.venv/bin/python -m pytest` (per `memory/factored-fb-test-env.md` — uv broken).
- Commits omit Co-Authored-By Claude trailer (per `memory/commit-no-claude-coauthor.md`).
- Adaptation pattern: same as `agents/fb/flow_bc/agent.py:1-10` header — `self.cfg.train.X -> self.X`, `self._model.X -> self.model.X`, `cudagraphs/compile -> dropped`.
- Source for adaptation: `/home/adityamohan/git/baselines/td_jepa/metamotivo/agents/rldp/`.

---

## File structure

**New files:**
- `agents/rldp/__init__.py` (empty)
- `agents/rldp/model.py` (~60 LOC — RLDPModel subclass)
- `agents/rldp/agent.py` (~180 LOC — RLDPAgent subclass with SP loss)
- `agents/rldp/flow_bc/__init__.py` (empty)
- `agents/rldp/flow_bc/model.py` (~50 LOC — RLDPFlowBCModel)
- `agents/rldp/flow_bc/agent.py` (~110 LOC — RLDPFlowBCAgent)
- `configs/agent/rldp.yaml`
- `configs/agent/rldp_flowbc.yaml`
- `scripts/agents/rldp/state.sh`
- `scripts/agents/rldp_flowbc/state.sh`
- `tests/test_rldp_model.py`
- `tests/test_rldp_agent_smoke.py`
- `tests/test_rldp_flowbc_model.py`
- `tests/test_rldp_flowbc_agent_smoke.py`

**Modified:**
- `train.py:278` — one conditional for `horizon`.
- `tests/scripts/test_agent_launchers.py` — append 2 new tests.

**Untouched:** `agents/fb/*`, `nn_models.py` (sub-project 1 already added VForwardMap), `buffers/transition.py` (sub-project 1 already added horizon kwarg).

---

## Task 1: `agents/rldp/model.py` + `__init__.py`

**Files:**
- Create: `agents/rldp/__init__.py`
- Create: `agents/rldp/model.py`
- Create: `tests/test_rldp_model.py`

- [ ] **Step 1: Create directory + empty __init__**

```bash
mkdir -p agents/rldp tests
: > agents/rldp/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_rldp_model.py`:

```python
"""Smoke tests for RLDPModel (subclass of FBModel that adds _predictor)."""
import gymnasium
import numpy as np
import torch

from agents.rldp.model import RLDPModel
from nn_models import VForwardArchiConfig


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_rldp_model_has_predictor_submodule():
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    assert hasattr(model, "_predictor"), "RLDPModel must register _predictor"
    assert isinstance(model._predictor, torch.nn.Module)
    # _predictor uses VForwardMap; should have num_parallel=2 by default
    assert model._predictor.num_parallel == 2


def test_rldp_model_inherits_fb_submodules():
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    for attr in ("_backward_map", "_forward_map", "_actor", "_bw_encoder", "_fw_encoder", "_left_encoder"):
        assert hasattr(model, attr), f"missing inherited submodule {attr}"


def test_rldp_model_predictor_forward():
    """Predictor maps (B, z_dim) + (B, action_dim) -> (num_parallel, B, z_dim)."""
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    B, z_dim = 8, model.z_dim
    z = torch.randn(B, z_dim)
    action = torch.randn(B, 4)
    out = model._predictor(z, action)
    # VForwardMap default num_parallel=2 -> leading parallel dim
    assert out.shape == (2, B, z_dim), out.shape


def test_rldp_model_custom_predictor_cfg():
    """Override predictor config kwarg."""
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2, num_parallel=1)
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4, predictor_cfg=cfg)
    assert model._predictor.hidden_dim == 64
    assert model._predictor.num_parallel == 1
```

- [ ] **Step 3: Run tests to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_model.py -v
```
Expected: ALL FAIL with `ModuleNotFoundError: No module named 'agents.rldp.model'`.

- [ ] **Step 4: Implement `agents/rldp/model.py`**

Create `agents/rldp/model.py`:

```python
"""agents/rldp/model.py — Adapted from td_jepa/.../rldp/model.py.

Same substitution scheme as agents/fb/flow_bc/model.py:
    self.cfg.archi.X     -> self.X / kwarg
    cfg.archi.predictor  -> predictor_cfg kwarg
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
import torch

from agents.fb.model import FBModel
from nn_models import VForwardArchiConfig


class RLDPModel(FBModel):
    """FB model with an added self-predictive (SP) `_predictor` head.

    `_predictor` maps a current z-embedding + an action to a *predicted next*
    z-embedding. Used by RLDPAgent.update_fb to add a self-predictive loss
    that regularizes the encoder via dynamics prediction (the 'DP' in RLDP).
    """

    def __init__(
        self,
        obs_space,
        action_dim: int,
        predictor_cfg: Optional[VForwardArchiConfig] = None,
        **fb_kwargs,
    ):
        # Stash predictor cfg BEFORE super().__init__ in case any FBModel hook needs it.
        self._predictor_cfg = predictor_cfg if predictor_cfg is not None else VForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, **fb_kwargs)

        # Build the predictor over (z, action) -> z_dim (state-only path; obs is the z-embedding here).
        z_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.z_dim,), dtype=np.float32)
        self._predictor = self._predictor_cfg.build(z_space, self.z_dim, output_dim=self.z_dim)

        # make sure the model is in eval mode and never computes gradients (matches FBModel)
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)
```

- [ ] **Step 5: Run tests to confirm pass**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_model.py -v
```
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```
git add agents/rldp/__init__.py agents/rldp/model.py tests/test_rldp_model.py
git commit -m "feat(rldp): RLDPModel subclass adds _predictor head (SP-loss target)"
```

---

## Task 2: `agents/rldp/agent.py` + smoke test

**Files:**
- Create: `agents/rldp/agent.py`
- Create: `tests/test_rldp_agent_smoke.py`

This is the largest task — the agent overrides `_make_model`, `setup_training`, `update`, and `update_fb`. The body of `update_fb` is the SP-loss addition (~25 lines new on top of FB's update_fb structure).

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_rldp_agent_smoke.py`:

```python
"""Smoke test for RLDPAgent — instantiate, run 10 update steps on synthetic data."""
import gymnasium
import numpy as np
import torch

from agents.rldp.agent import RLDPAgent
from buffers.transition import DictBuffer


def _make_synthetic_buffer(n_eps: int = 5, ep_len: int = 40, obs_dim: int = 20, act_dim: int = 4):
    total = n_eps * ep_len
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    next_obs = np.random.randn(total, obs_dim).astype(np.float32)
    action = np.random.randn(total, act_dim).astype(np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=bool)
    for i in range(n_eps):
        terminated[(i + 1) * ep_len - 1] = True
    timestep = np.concatenate([np.arange(ep_len) for _ in range(n_eps)]).astype(np.int64)
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_rldp_agent_constructs():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5)
    assert agent.horizon == 5
    assert hasattr(agent.model, "_predictor")


def test_rldp_agent_update_step_no_nan():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5,
                      ortho_coef=0.01, lr_b=1e-4, lr_f=1e-4, lr_actor=1e-4)
    buf = _make_synthetic_buffer()
    torch.manual_seed(0)
    for step in range(10):
        batch = buf.sample(16, horizon=5)
        metrics = agent.update(batch, step)
        for k, v in metrics.items():
            val = float(v.detach() if hasattr(v, "detach") else v)
            assert not np.isnan(val), f"NaN in metric {k!r} at step {step}: {val}"
            assert np.isfinite(val), f"non-finite metric {k!r} at step {step}: {val}"


def test_rldp_agent_backward_optimizer_includes_predictor():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5)
    # The optimizer should be over backward_map + bw_encoder + predictor params.
    opt_params = set()
    for pg in agent.backward_optimizer.param_groups:
        for p in pg["params"]:
            opt_params.add(id(p))
    for p in agent.model._predictor.parameters():
        assert id(p) in opt_params, "predictor param missing from backward_optimizer"
```

- [ ] **Step 2: Run to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_agent_smoke.py -v
```
Expected: ALL FAIL with `ModuleNotFoundError: No module named 'agents.rldp.agent'`.

- [ ] **Step 3: Implement `agents/rldp/agent.py`**

Create `agents/rldp/agent.py`:

```python
"""agents/rldp/agent.py — Adapted from td_jepa/.../rldp/agent.py.

Same substitution scheme as agents/fb/agent.py:
    self.cfg.train.X     -> self.X
    self._model.X        -> self.model.X
    cudagraphs/compile   -> dropped (eager mode)

RLDPAgent inherits from FBAgent. The novel pieces vs FB:
  - _predictor in self.model (added by RLDPModel)
  - backward_optimizer also updates _predictor params
  - update_fb adds a self-predictive (SP) loss:
      starting from B(obs), roll out _predictor(curr, action_t) for h steps,
      and L2-match each step to _target_backward_map(future_obs_t).
  - update() consumes windowed [B, horizon, ...] batches from
    DictBuffer.sample(batch_size, horizon=h) and slices inline.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from agents.fb.agent import FBAgent
from agents.rldp.model import RLDPModel
from nn_models import VForwardArchiConfig


class RLDPAgent(FBAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        horizon: int = 5,
        predictor_cfg: Optional[VForwardArchiConfig] = None,
        **fb_kwargs,
    ):
        # Stash RLDP-specific kwargs BEFORE super().__init__ so _make_model sees them.
        self.horizon = int(horizon)
        self._predictor_cfg = predictor_cfg if predictor_cfg is not None else VForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, **fb_kwargs)

    def _make_model(self, **kwargs):
        """Override FBAgent._make_model to build RLDPModel."""
        return RLDPModel(predictor_cfg=self._predictor_cfg, **kwargs)

    def setup_training(self) -> None:
        super().setup_training()
        # Extend backward_optimizer to include _predictor params (matches td_jepa rldp/agent.py).
        self.backward_optimizer = torch.optim.Adam(
            list(self.model._backward_map.parameters())
            + list(self.model._bw_encoder.parameters())
            + list(self.model._predictor.parameters()),
            lr=self.lr_b,
            weight_decay=self.weight_decay,
        )

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        """Consume a windowed batch (shape [B, horizon, ...] per key) and run one
        RLDP update. Inline-slices windowed batch into td_jepa-shaped intermediates.
        """
        # Inline adapter: windowed -> td_jepa shapes.
        obs        = batch["observation"][:, 0].to(self.device)
        action     = batch["action"][:, 0].to(self.device)
        next_obs   = batch["next"]["observation"][:, 0].to(self.device)
        terminated = batch["next"]["terminated"][:, 0].to(self.device)
        # future_obs covers steps 1..h-1 of the window (h-1 future targets for SP loss).
        # Shape: [B, h-1, obs_dim]. Flatten to [B*(h-1), obs_dim] for the target encoder pass.
        if self.horizon > 1:
            future_obs = batch["observation"][:, 1:].to(self.device)
            future_act = batch["action"][:, 1:].to(self.device)
            future_obs_flat = future_obs.reshape(-1, *future_obs.shape[2:])
        else:
            future_obs_flat = None
            future_act = None

        discount = self.discount * (~terminated.bool())

        # Update obs_normalizer running stats (matches td_jepa).
        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        from nn_models import eval_mode
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs = self.model._obs_normalizer(obs)
            next_obs = self.model._obs_normalizer(next_obs)
            if future_obs_flat is not None:
                future_obs_flat = self.model._obs_normalizer(future_obs_flat)

        # Augmentation + encoding (state-only: aug is identity, encoders pass through).
        obs       = self.model._augmentator(obs)
        next_obs  = self.model._augmentator(next_obs)
        if future_obs_flat is not None:
            future_obs_flat = self.model._augmentator(future_obs_flat)

        obs_fw  = self.model._fw_encoder(obs)
        obs_bw  = self.model._bw_encoder(obs)
        with torch.no_grad():
            goal = self.model._bw_encoder(next_obs)
            next_obs_fw = self.model._fw_encoder(next_obs)
            future_obs_bw = self.model._bw_encoder(future_obs_flat) if future_obs_flat is not None else None

        z = self.sample_mixed_z(train_goal=goal).clone()

        clip_grad_norm = self.clip_grad_norm if self.clip_grad_norm > 0 else None
        q_loss_coef = self.q_loss_coef if self.q_loss_coef > 0 else None

        metrics = self.update_fb(
            obs=obs_fw,
            obs_bw=obs_bw,
            action=action,
            future_act=future_act,
            discount=discount,
            next_obs=next_obs_fw,
            future_obs=future_obs_bw,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(self.update_actor(
            obs=obs_fw.detach(),
            action=action,
            z=z,
            clip_grad_norm=clip_grad_norm,
        ))

        # Soft-update targets (matches FBAgent post-update step).
        from nn_models import _soft_update_params
        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.f_target_tau)

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        obs_bw: torch.Tensor,
        action: torch.Tensor,
        future_act: Optional[torch.Tensor],
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: Optional[torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: Optional[float],
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                next_action = self.sample_action_from_norm_obs(actor_in, z)
                target_Fs = self.model._target_forward_map(next_left_enc, z, next_action)
                target_B = self.model._target_backward_map(goal)
                target_Ms = torch.matmul(target_Fs, target_B.T)
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.fb_pessimism_penalty)

            left_enc = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left_enc, z, action)
            B = self.model._backward_map(goal).detach()
            Ms = torch.matmul(Fs, B.T)

            diff = Ms - discount * target_M
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # ── Self-predictive (SP) loss — the RLDP-specific addition ──
            with torch.no_grad():
                targets = [self.model._target_backward_map(next_obs)]
                actions = [action]
                if future_obs is not None and future_act is not None:
                    future_phi = self.model._target_backward_map(future_obs)
                    future_phi = future_phi.reshape(*future_act.shape[:2], -1)  # [B, h-1, z_dim]
                    targets += [future_phi[:, i] for i in range(future_phi.shape[1])]
                    actions += [future_act[:, i] for i in range(future_act.shape[1])]
            B_pred = self.model._backward_map(obs_bw)
            curr = B_pred
            sp_loss = 0
            for act_t, target_t in zip(actions, targets):
                curr = self.model._predictor(curr, act_t)  # [num_parallel, B, z_dim]
                sp_loss = sp_loss + (curr - target_t.unsqueeze(0)).pow(2).sum(-1).mean()
            sp_loss = sp_loss / max(len(actions), 1)
            fb_loss = fb_loss + sp_loss

            # ── Orthonormality loss on B ──
            Cov = torch.matmul(B_pred, B_pred.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss = fb_loss + self.ortho_coef * orth_loss

            # ── Optional Q-MSE auxiliary loss ──
            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.fb_pessimism_penalty)
                    with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=False):
                        cov = torch.matmul(B_pred.T, B_pred) / B_pred.shape[0]
                    B_inv_conv = torch.linalg.solve(cov, B_pred, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss = fb_loss + q_loss_coef * q_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B_pred.mean(),
                "B_norm": torch.norm(B_pred, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "sp_loss": sp_loss,
                "q_loss": q_loss,
            }
        return output_metrics
```

- [ ] **Step 4: Run smoke tests**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_agent_smoke.py -v
```
Expected: 3 PASS. If `update_step_no_nan` fails with NaN, investigate which loss term blew up (check the `sp_loss` calculation first — it's the most error-prone new code).

- [ ] **Step 5: Run sub-project 1 + FB tests to confirm no regression**

Run:
```
.venv/bin/python -m pytest tests/test_nn_models_vforward.py tests/test_dict_buffer_horizon.py tests/parity/ tests/test_frame_stack_buffer.py tests/scripts/ -v 2>&1 | tail -10
```
Expected: all pre-RLDP tests still green (48 from sub-project 1 + foundation, plus whatever scripts pass).

- [ ] **Step 6: Commit**

```
git add agents/rldp/agent.py tests/test_rldp_agent_smoke.py
git commit -m "feat(rldp): RLDPAgent with self-predictive loss + inline windowed-batch adapter"
```

---

## Task 3: `agents/rldp/flow_bc/model.py` + `__init__.py`

**Files:**
- Create: `agents/rldp/flow_bc/__init__.py`
- Create: `agents/rldp/flow_bc/model.py`
- Create: `tests/test_rldp_flowbc_model.py`

- [ ] **Step 1: Create dir + empty init**

```bash
mkdir -p agents/rldp/flow_bc
: > agents/rldp/flow_bc/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_rldp_flowbc_model.py`:

```python
"""Smoke tests for RLDPFlowBCModel (subclass of RLDPModel that adds _actor_vf)."""
import gymnasium
import numpy as np
import torch

from agents.rldp.flow_bc.model import RLDPFlowBCModel


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_rldp_flowbc_model_has_actor_vf_and_predictor():
    model = RLDPFlowBCModel(obs_space=_obs_space(20), action_dim=4)
    for attr in ("_actor_vf", "_predictor", "_backward_map", "_forward_map"):
        assert hasattr(model, attr), f"missing submodule {attr}"


def test_rldp_flowbc_model_act_returns_actions():
    model = RLDPFlowBCModel(obs_space=_obs_space(20), action_dim=4)
    obs = torch.randn(8, 20)
    z = torch.randn(8, model.z_dim)
    actions = model.act(obs, z)
    assert actions.shape == (8, 4)
    assert torch.isfinite(actions).all()
```

- [ ] **Step 3: Run to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_flowbc_model.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.rldp.flow_bc.model'`.

- [ ] **Step 4: Implement `agents/rldp/flow_bc/model.py`**

Create `agents/rldp/flow_bc/model.py`:

```python
"""agents/rldp/flow_bc/model.py — Adapted from td_jepa/.../rldp/flow_bc/model.py.

Mirrors the FBFlowBCModel pattern in agents/fb/flow_bc/model.py: subclasses
RLDPModel and adds `_actor_vf` (the action-flow vector field) for flow-matching.
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
import torch
from torch.amp import autocast

from agents.rldp.model import RLDPModel
from nn_models import SimpleVectorFieldArchiConfig


class RLDPFlowBCModel(RLDPModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        **rldp_kwargs,
    ):
        super().__init__(obs_space=obs_space, action_dim=action_dim, **rldp_kwargs)

        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig()

        actor_vf_obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.L_dim,), dtype=np.float32)
            if self.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = actor_vf_cfg.build(actor_vf_obs_space, action_dim)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.actor_encode_obs else obs
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean
        return self.actor(obs, z)
```

- [ ] **Step 5: Run tests to confirm pass**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_flowbc_model.py -v
```
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```
git add agents/rldp/flow_bc/__init__.py agents/rldp/flow_bc/model.py tests/test_rldp_flowbc_model.py
git commit -m "feat(rldp): RLDPFlowBCModel subclass adds _actor_vf flow-matching head"
```

---

## Task 4: `agents/rldp/flow_bc/agent.py` + smoke test

**Files:**
- Create: `agents/rldp/flow_bc/agent.py`
- Create: `tests/test_rldp_flowbc_agent_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_rldp_flowbc_agent_smoke.py`:

```python
"""Smoke test for RLDPFlowBCAgent — instantiate, run 10 update steps."""
import gymnasium
import numpy as np
import torch

from agents.rldp.flow_bc.agent import RLDPFlowBCAgent
from buffers.transition import DictBuffer


def _make_synthetic_buffer(n_eps: int = 5, ep_len: int = 40, obs_dim: int = 20, act_dim: int = 4):
    total = n_eps * ep_len
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    next_obs = np.random.randn(total, obs_dim).astype(np.float32)
    action = np.tanh(np.random.randn(total, act_dim)).astype(np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=bool)
    for i in range(n_eps):
        terminated[(i + 1) * ep_len - 1] = True
    timestep = np.concatenate([np.arange(ep_len) for _ in range(n_eps)]).astype(np.int64)
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_rldp_flowbc_agent_constructs():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPFlowBCAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5)
    assert hasattr(agent, "actor_vf_optimizer")
    assert hasattr(agent.model, "_actor_vf")
    assert hasattr(agent.model, "_predictor")


def test_rldp_flowbc_agent_update_step_no_nan():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPFlowBCAgent(
        obs_space=obs_space, action_dim=4, batch_size=16, horizon=5,
        ortho_coef=0.01, lr_b=1e-4, lr_f=1e-4, lr_actor=1e-4, lr_actor_vf=3e-4,
        flow_steps=5, bc_coeff=0.3,
    )
    buf = _make_synthetic_buffer()
    torch.manual_seed(0)
    for step in range(10):
        batch = buf.sample(16, horizon=5)
        metrics = agent.update(batch, step)
        for k, v in metrics.items():
            val = float(v.detach() if hasattr(v, "detach") else v)
            assert not np.isnan(val), f"NaN in {k!r} at step {step}: {val}"
```

- [ ] **Step 2: Run to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_flowbc_agent_smoke.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.rldp.flow_bc.agent'`.

- [ ] **Step 3: Implement `agents/rldp/flow_bc/agent.py`**

Create `agents/rldp/flow_bc/agent.py`:

```python
"""agents/rldp/flow_bc/agent.py — Adapted from td_jepa/.../rldp/flow_bc/agent.py.

Mirrors the FBFlowBCAgent pattern in agents/fb/flow_bc/agent.py: subclasses
RLDPAgent and adds (a) actor_vf_optimizer for the action-flow vector field,
(b) flow-matching loss in update_actor, (c) compute_flow_actions Euler
integration helper used by the BC loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.amp import autocast

from agents.rldp.agent import RLDPAgent
from agents.rldp.flow_bc.model import RLDPFlowBCModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class RLDPFlowBCAgent(RLDPAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: Optional[NoiseConditionedActorArchiConfig] = None,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        **rldp_kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        self.flow_steps = int(flow_steps)
        self.lr_actor_vf = float(lr_actor_vf)
        self._actor_vf_cfg = actor_vf_cfg
        super().__init__(
            obs_space=obs_space, action_dim=action_dim,
            actor_cfg=actor_cfg, **rldp_kwargs,
        )

    def _make_model(self, **kwargs):
        return RLDPFlowBCModel(
            actor_vf_cfg=self._actor_vf_cfg,
            predictor_cfg=self._predictor_cfg,
            **kwargs,
        )

    def setup_training(self) -> None:
        super().setup_training()
        self.actor_vf_optimizer = torch.optim.Adam(
            self.model._actor_vf.parameters(),
            lr=self.lr_actor_vf,
            weight_decay=self.weight_decay,
        )

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        return self.model._actor(obs, z, noises)

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

            with torch.no_grad():
                left_enc = self.model._left_encoder(obs)
            actor_in = left_enc if self.actor_encode_obs else obs
            noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            actor_actions = self.model._actor(actor_in, z, noises)
            Fs = self.model._forward_map(left_enc, z, actor_actions)
            Qs = (Fs * z).sum(-1)
            _, _, Q = self.get_targets_uncertainty(Qs, self.actor_pessimism_penalty)
            actor_loss = -Q.mean()

            bc_loss = torch.tensor([0.0], device=action.device)
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                with torch.no_grad():
                    target_flow_actions = self.compute_flow_actions(obs, noises)
                bc_error = torch.pow(actor_actions - target_flow_actions, 2).mean()
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

            actor_loss = actor_loss + bc_flow_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()
        self.actor_vf_optimizer.step()

        return {
            "actor_loss": actor_loss.mean().detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "bc_error": bc_error.detach(),
            "q": Q.mean().detach(),
        }

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.flow_steps
            vels = self.model._actor_vf(obs, actions, t)
            actions = actions + vels / self.flow_steps
        return torch.clamp(actions, min=-1, max=1)

    def act(self, obs: torch.Tensor, z: torch.Tensor, *, eval_mode: bool = True) -> torch.Tensor:
        return self.model.act(obs, z, mean=eval_mode)
```

- [ ] **Step 4: Run smoke tests**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_flowbc_agent_smoke.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Run all RLDP tests to confirm both variants compose**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_model.py tests/test_rldp_agent_smoke.py tests/test_rldp_flowbc_model.py tests/test_rldp_flowbc_agent_smoke.py -v
```
Expected: 4 + 3 + 2 + 2 = 11 PASS.

- [ ] **Step 6: Commit**

```
git add agents/rldp/flow_bc/agent.py tests/test_rldp_flowbc_agent_smoke.py
git commit -m "feat(rldp): RLDPFlowBCAgent with flow-matching actor + BC loss"
```

---

## Task 5: `train.py` horizon kwarg + configs

**Files:**
- Modify: `train.py:278` (the `buffer.sample(...)` call)
- Create: `configs/agent/rldp.yaml`
- Create: `configs/agent/rldp_flowbc.yaml`

- [ ] **Step 1: Read the exact current `buffer.sample` call**

Run:
```
sed -n '275,282p' train.py
```
Expected: shows `batch = buffer.sample(cfg.batch_size)` on line 278.

- [ ] **Step 2: Edit `train.py`**

Use `Edit`:

Old:
```python
        # Update
        batch = buffer.sample(cfg.batch_size)
        metrics = agent.update(batch, step)
```

New:
```python
        # Update
        horizon = getattr(cfg, "horizon", 1)
        batch = buffer.sample(cfg.batch_size, horizon=horizon)
        metrics = agent.update(batch, step)
```

Behavior with `horizon` absent or 1: byte-identical to before (FB, FB+FlowBC use horizon=1 default).

- [ ] **Step 3: Create `configs/agent/rldp.yaml`**

```yaml
# @package _global_
defaults:
  - fb            # inherit forward/backward/actor/encoder + all training kwargs
  - _self_

agent: rldp
horizon: 5                   # RLDP samples horizon-5 windows for SP loss
ortho_coef: 0.01             # much lower than FB's 1000 — encoder absorbs B-regularization
predictor:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
```

- [ ] **Step 4: Create `configs/agent/rldp_flowbc.yaml`**

```yaml
# @package _global_
defaults:
  - fb_flowbc     # inherit fb + flow_bc machinery (noise actor + vector field)
  - _self_

agent: rldp_flowbc
horizon: 5
ortho_coef: 0.01
predictor:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
```

- [ ] **Step 5: Confirm FB still launches (regression)**

Run:
```
DRY_RUN=1 SEEDS=0 GPUS=0 bash scripts/agents/fb/state.sh 2>&1 | tail -3
```
Expected: same output as before — train.py still loads cleanly when `cfg.horizon` is absent.

Run the existing launcher test suite:
```
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k fb
```
Expected: 7 PASS (existing FB tests unchanged).

- [ ] **Step 6: Commit**

```
git add train.py configs/agent/rldp.yaml configs/agent/rldp_flowbc.yaml
git commit -m "feat(train+configs): horizon kwarg in train.py; rldp + rldp_flowbc agent configs"
```

---

## Task 6: Launchers + launcher tests

**Files:**
- Create: `scripts/agents/rldp/state.sh`
- Create: `scripts/agents/rldp_flowbc/state.sh`
- Modify: `tests/scripts/test_agent_launchers.py` (append 2 tests)

- [ ] **Step 1: Append failing tests to `tests/scripts/test_agent_launchers.py`**

Read the file to find the end:
```
wc -l tests/scripts/test_agent_launchers.py
```

Append at end:

```python


# ── rldp/state.sh ──────────────────────────────────────────────────────────
def test_rldp_state_dry_run():
    r = _dry_run("rldp/state.sh")
    assert r.returncode == 0, r.stderr
    assert "train.py" in r.stdout
    assert "agent=rldp" in r.stdout
    assert "domain=cube_single" in r.stdout
    assert "[DRY]" in r.stdout


# ── rldp_flowbc/state.sh ───────────────────────────────────────────────────
def test_rldp_flowbc_state_dry_run():
    r = _dry_run("rldp_flowbc/state.sh")
    assert r.returncode == 0, r.stderr
    assert "agent=rldp_flowbc" in r.stdout
```

- [ ] **Step 2: Run to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k rldp
```
Expected: 2 FAIL with "No such file or directory".

- [ ] **Step 3: Create the leaf scripts**

```bash
mkdir -p scripts/agents/rldp scripts/agents/rldp_flowbc
```

Create `scripts/agents/rldp/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/rldp/state.sh — plain RLDP on cube_single (state).
USAGE="usage: bash scripts/agents/rldp/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=rldp_state
    DATA_PATH=/dev/shm/factored-fb/datasets
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-rldp_state}"
_run_fb_sweep cube_single "$RUN_GROUP" agent=rldp
```

Create `scripts/agents/rldp_flowbc/state.sh`:

```bash
#!/usr/bin/env bash
# scripts/agents/rldp_flowbc/state.sh — RLDP + FlowBC on cube_single (state).
USAGE="usage: bash scripts/agents/rldp_flowbc/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=rldp_flowbc_state
    DATA_PATH=/dev/shm/factored-fb/datasets
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-rldp_flowbc_state}"
_run_fb_sweep cube_single "$RUN_GROUP" agent=rldp_flowbc
```

Make executable:
```bash
chmod +x scripts/agents/rldp/state.sh scripts/agents/rldp_flowbc/state.sh
```

- [ ] **Step 4: Run launcher tests**

Run:
```
.venv/bin/python -m pytest tests/scripts/test_agent_launchers.py -v -k rldp
```
Expected: 2 PASS.

- [ ] **Step 5: Run full launcher suite to confirm no regression**

Run:
```
.venv/bin/python -m pytest tests/scripts/ -v
```
Expected: pre-existing 25 PASS + 2 new = 27 PASS.

- [ ] **Step 6: Commit**

```
git add scripts/agents/rldp/state.sh scripts/agents/rldp_flowbc/state.sh tests/scripts/test_agent_launchers.py
git commit -m "feat(scripts): agents/rldp/{,_flowbc}/state.sh launchers + tests"
```

---

## Task 7: 5000-step end-to-end real smoke run

**Files:** none modified (real GPU run, verification only).

This is the only acceptance gate that touches real GPUs and real OGBench data.

- [ ] **Step 1: Confirm dataset exists**

Run:
```
ls -la /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/ 2>&1 | head -3
```
Expected: shows .npz files. If missing, fail fast — the user needs to extract the dataset first.

- [ ] **Step 2: Run a 5000-step single-seed RLDP smoke**

Run (this takes ~3-5 minutes on a single GPU):
```
TRAIN_STEPS=5000 SEEDS=0 GPUS=0 RUN_GROUP=rldp_smoke bash scripts/agents/rldp/state.sh 2>&1 | tee /dev/shm/rldp_smoke.log
```

Expected: completes with `[fb-sweep] ALL DONE`. No Python tracebacks. Per-seed log at `/dev/shm/rldp_smoke_<TS>/seed_0.log` shows training metric lines being printed periodically.

- [ ] **Step 3: Verify the smoke run produced finite metrics**

Run:
```
grep -E "fb_loss|sp_loss|orth_loss" /dev/shm/rldp_smoke_*/seed_0.log | tail -10
```
Expected: all printed loss values are finite numbers (no `nan`, no `inf`).

If NaN appears: investigate which loss term blew up (most likely `sp_loss` if the predictor is unstable, or `orth_loss` if `ortho_coef` is misconfigured). This is a sub-project 2 blocker.

- [ ] **Step 4: Run the same smoke for RLDP+FlowBC**

Run:
```
TRAIN_STEPS=5000 SEEDS=0 GPUS=0 RUN_GROUP=rldp_flowbc_smoke bash scripts/agents/rldp_flowbc/state.sh 2>&1 | tee /dev/shm/rldp_flowbc_smoke.log
```
Expected: completes without crash; loss metrics finite.

Verify:
```
grep -E "fb_loss|sp_loss|bc_flow_loss|actor_loss" /dev/shm/rldp_flowbc_smoke_*/seed_0.log | tail -10
```

- [ ] **Step 5: No commit needed (verification only)**

If both smokes pass: sub-project 2 is functionally complete.

---

## Task 8: Final acceptance — full suite + bounded diff

**Files:** none modified.

- [ ] **Step 1: Full test suite (excluding pre-existing pandas failures)**

Run:
```
.venv/bin/python -m pytest tests/test_rldp_*.py tests/test_nn_models_vforward.py tests/test_dict_buffer_horizon.py tests/parity/ tests/test_frame_stack_buffer.py tests/scripts/ -v 2>&1 | tail -15
```
Expected: all PASS. Specifically:
- 4 + 3 + 2 + 2 = 11 RLDP tests
- 5 VForward + 5 DictBuffer horizon + 7 parity = 17 sub-project 1 tests
- 6 frame_stack
- 27 launcher tests (25 original + 2 new)
- **Total: 61 PASS**

- [ ] **Step 2: Diff bounded check**

Run:
```
git diff main --stat
```
Expected files only:
- `agents/rldp/{__init__.py, model.py, agent.py}`
- `agents/rldp/flow_bc/{__init__.py, model.py, agent.py}`
- `configs/agent/rldp.yaml`
- `configs/agent/rldp_flowbc.yaml`
- `scripts/agents/rldp/state.sh`
- `scripts/agents/rldp_flowbc/state.sh`
- `tests/test_rldp_model.py`
- `tests/test_rldp_agent_smoke.py`
- `tests/test_rldp_flowbc_model.py`
- `tests/test_rldp_flowbc_agent_smoke.py`
- `tests/scripts/test_agent_launchers.py` (2-line addition)
- `train.py` (one conditional)

Plus sub-project 1's files if branched from before that merged. If the branch is from `rldp-foundation` then sub-project 1 files also appear; if branched from `main` then sub-project 1 files appear too (since `rldp-foundation` hasn't merged yet).

If unexpected paths appear: investigate before pushing.

- [ ] **Step 3: No commit needed (verification only)**

Sub-project 2 functionally complete. Hand off to `finishing-a-development-branch` skill for merge/PR decision.

---

## Self-Review

**1. Spec coverage:**
- Spec Section 2 (files created) → Tasks 1-6 cover all 14 new files + 2 modified.
- Spec Section 3 (buffer integration) → Task 2 implements the inline adapter in `RLDPAgent.update()`.
- Spec Section 4 (train.py edit) → Task 5 Step 2 makes the one-line change.
- Spec Section 4 (configs) → Task 5 Steps 3-4.
- Spec Section 4 (launchers) → Task 6.
- Spec Section 5 (tests) → Tasks 1, 2, 3, 4 each include both unit + smoke tests; Task 6 adds launcher tests.
- Spec acceptance criteria → Tasks 7 (end-to-end smoke) + 8 (full suite + bounded diff).
- Open risk "SimpleVectorFieldArchiConfig location" → resolved during recon (it's in our `nn_models.py:407`); Task 3 imports it from there.
- Open risk "base_model.load_model equivalence" → resolved by not porting RLDP's `load()` classmethods (we don't need disk-load for training; can add later if needed).
- Open risk "cudagraphs/compile cfg fields" → resolved by dropping (same as our FB port does — header comment of `agents/fb/agent.py:10`).

**2. Placeholder scan:**
- No TBD/TODO/"implement later" markers.
- Every code block contains the actual content.
- No "similar to Task N" — Task 3 (RLDPFlowBCModel) and Task 4 (RLDPFlowBCAgent) include their full file contents even though they mirror earlier patterns.

**3. Type consistency:**
- `RLDPModel.__init__` signature `(obs_space, action_dim, predictor_cfg=None, **fb_kwargs)` matches the test's instantiation in Task 1 Step 2.
- `RLDPAgent.__init__` `(obs_space, action_dim, horizon=5, predictor_cfg=None, **fb_kwargs)` matches Task 2 Step 1.
- `RLDPFlowBCModel.__init__` `(obs_space, action_dim, actor_vf_cfg=None, **rldp_kwargs)` matches Task 3.
- `RLDPFlowBCAgent.__init__` `(obs_space, action_dim, actor_cfg=None, actor_vf_cfg=None, flow_steps=10, lr_actor_vf=3e-4, **rldp_kwargs)` matches Task 4.
- `RLDPAgent.update_fb` signature consistent across definition + caller in `RLDPAgent.update()`.
- `_make_model` override returns the right class per task: Task 1 (RLDPModel via FBModel→RLDPModel chain), Task 4 (RLDPFlowBCModel).

No issues found.
