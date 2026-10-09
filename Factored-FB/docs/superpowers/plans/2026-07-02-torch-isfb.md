# Torch SG-ISFB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a faithful Scalar-Gated In-Sample Forward–Backward (SG-ISFB) agent on top of the canonical torch FB agent, whose ONLY additions over standard FB are the `PAPER/fb-modified/isfb.tex` pieces.

**Architecture:** New package `agents/fb/isfb/` mirroring `agents/fb/flow_bc/`. `ISFBModel(FBFlowBCModel)` adds one module — a value-forward map `F_V(s,z)` (a second `ForwardMap` fed a zeroed action). `ISFBAgent(FBFlowBCAgent)` keeps the flow actor unchanged and overrides only the critic update (`update_fb`) with the SG-ISFB loss (`L_V` scalar expectile → advantage → weighted vector regression `L_N` → measure TD `L_M` bootstrapped through `N⁻`, no `a'~π`). Registered in `train.py` under `cfg.agent == "isfb_flowbc"`.

**Tech Stack:** PyTorch, gymnasium spaces, existing `agents/fb/` FB port, Hydra configs, pytest (`.venv/bin/python -m pytest`).

## Global Constraints

- **Do not modify** `agents/fb/agent.py` or `agents/fb/model.py`. All new code lives under `agents/fb/isfb/` + `train.py` branch + a config yaml.
- Reuse standard-FB machinery verbatim: `expectile_loss`, `fb_successor_terms`, `ortho_cov` from `agents.fb.agent`; `sample_mixed_z`, `project_z`, target-net EMA, obs normalizer, `_value_scale`.
- Scalar readouts match the flow actor's Q exactly: `q = (Fs*z).sum(-1)`, `v = (Fv*z).sum(-1)` (no hardcoded `/√d`; `_value_scale` stays the FB default of 1.0).
- Measure terms stay unscaled (`M = Fs @ Bᵀ`), identical to canonical `update_fb`.
- `ortho_coef` stays **1.0** (isfb.tex: not 1000). `iql_expectile` (τ) is the expectile for both `L_V` and the `coord` path.
- Test command: `.venv/bin/python -m pytest`. `uv run` is broken in this repo.
- Commits: no `Co-Authored-By: Claude` trailer (repo hook strips it anyway).
- Exp1 scope only: exact-goal (`train_goal_ratio=1.0`). No Exp2/Exp3 machinery, no pixel path.

---

### Task 1: `ISFBModel` — the F_V value-forward leg

**Files:**
- Create: `agents/fb/isfb/__init__.py` (empty)
- Create: `agents/fb/isfb/model.py`
- Test: `tests/test_torch_isfb_model.py`

**Interfaces:**
- Consumes: `agents.fb.flow_bc.model.FBFlowBCModel`; `nn_models.ForwardArchiConfig`.
- Produces: `ISFBModel` with attributes `_value_forward_map` and (after `_prepare_for_train`) `_target_value_forward_map`. `_value_forward_map(left_enc, z, action)` returns `[num_parallel, batch, z_dim]`, same shape/arch as `_forward_map`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torch_isfb_model.py
import numpy as np
import torch


def _tiny_model():
    import gymnasium
    from agents.fb.isfb.model import ISFBModel
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return ISFBModel(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu",
    )


def test_value_forward_map_exists_and_shapes_match_forward():
    m = _tiny_model()
    B = 5
    left = torch.randn(B, m.L_dim)
    z = torch.randn(B, m.z_dim)
    a = torch.zeros(B, m.action_dim)
    fv = m._value_forward_map(left, z, a)
    f = m._forward_map(left, z, a)
    assert fv.shape == f.shape                      # [num_parallel, B, z_dim]
    assert fv.shape[-1] == m.z_dim


def test_prepare_for_train_builds_target_value_map():
    m = _tiny_model()
    m._prepare_for_train()
    assert hasattr(m, "_target_value_forward_map")
    # target starts as a copy of the online value map
    for p, tp in zip(m._value_forward_map.parameters(), m._target_value_forward_map.parameters()):
        assert torch.allclose(p, tp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.fb.isfb'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agents/fb/isfb/__init__.py
```
(empty file)

```python
# agents/fb/isfb/model.py
"""agents/fb/isfb/model.py — SG-ISFB model. Subclasses FBFlowBCModel (flow actor)
and adds ONE module: a value-forward map F_V(s,z), a second ForwardMap fed a zeroed
action. Everything else (F, B, actor, project_z, normalizer) is standard FB."""
from __future__ import annotations

import copy
from typing import Optional

import torch

from agents.fb.flow_bc.model import FBFlowBCModel
from nn_models import ForwardArchiConfig


class ISFBModel(FBFlowBCModel):
    def __init__(self, obs_space, action_dim: int, forward_cfg: Optional[ForwardArchiConfig] = None, **kwargs):
        # forward_cfg is consumed by FBModel to build _forward_map; capture it so we
        # can build F_V with the IDENTICAL architecture.
        self._isfb_forward_cfg = forward_cfg if forward_cfg is not None else ForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, forward_cfg=forward_cfg, **kwargs)
        self._value_forward_map = self._isfb_forward_cfg.build(
            self._left_encoder.output_space, self.z_dim, action_dim, goal_dim=self.goal_dim
        )
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_value_forward_map = copy.deepcopy(self._value_forward_map)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_model.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/fb/isfb/__init__.py agents/fb/isfb/model.py tests/test_torch_isfb_model.py
git commit -m "feat(isfb): ISFBModel with F_V value-forward leg (torch FB)"
```

---

### Task 2: `ISFBAgent` skeleton + optimizer/target wiring

**Files:**
- Create: `agents/fb/isfb/agent.py`
- Test: `tests/test_torch_isfb_agent.py`

**Interfaces:**
- Consumes: `agents.fb.flow_bc.agent.FBFlowBCAgent`; `agents.fb.isfb.model.ISFBModel`.
- Produces: `ISFBAgent(FBFlowBCAgent)` with config attrs `measure_insample` (default `"scalar_gate_hard"`), `alpha_n` (1.0), `alpha_m` (1.0), `awr_beta` (3.0), `awr_wmax` (100.0); F_V params added to `self.forward_optimizer`; paramlists `self._value_forward_map_paramlist`, `self._target_value_forward_map_paramlist`. `update_fb(...)` overridden in Task 3 (skeleton here returns `super().update_fb(...)` so build/optimizer tests pass first).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torch_isfb_agent.py
import numpy as np
import torch


def _tiny_isfb(**extra):
    import gymnasium
    from agents.fb.isfb.agent import ISFBAgent
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return ISFBAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def test_defaults_and_model_type():
    from agents.fb.isfb.model import ISFBModel
    a = _tiny_isfb()
    assert isinstance(a.model, ISFBModel)
    assert a.measure_insample == "scalar_gate_hard"
    assert a.alpha_n == 1.0 and a.alpha_m == 1.0
    assert a.critic_mode == "measure"          # ISFB uses the measure-branch update() path


def test_value_forward_params_in_forward_optimizer():
    a = _tiny_isfb()
    opt_params = {id(p) for g in a.forward_optimizer.param_groups for p in g["params"]}
    fv_params = {id(p) for p in a.model._value_forward_map.parameters()}
    assert fv_params.issubset(opt_params)
    assert len(a._target_value_forward_map_paramlist) == len(a._value_forward_map_paramlist) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.fb.isfb.agent'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agents/fb/isfb/agent.py
"""agents/fb/isfb/agent.py — SG-ISFB agent (PAPER/fb-modified/isfb.tex).

Subclasses FBFlowBCAgent (flow actor + z-cond inherited UNCHANGED). The ONLY
addition over standard FB is the critic update: the successor measure M is trained
with IQL's in-sample improvement operator (scalar expectile L_V on q=<F,z> vs
v=<F_V,z>; advantage-weighted vector regression L_N fitting N=<F_V,B> to sg M;
measure TD L_M bootstrapping through N^- at the next state, NO a'~pi)."""
from __future__ import annotations

from agents.fb.flow_bc.agent import FBFlowBCAgent
from agents.fb.isfb.model import ISFBModel


class ISFBAgent(FBFlowBCAgent):
    def __init__(self, *args, measure_insample: str = "scalar_gate_hard",
                 alpha_n: float = 1.0, alpha_m: float = 1.0,
                 awr_beta: float = 3.0, awr_wmax: float = 100.0, **kwargs):
        # Stash ISFB knobs BEFORE super().__init__ (which calls setup_training()).
        self.measure_insample = str(measure_insample)
        self.alpha_n = float(alpha_n)
        self.alpha_m = float(alpha_m)
        self.awr_beta = float(awr_beta)
        self.awr_wmax = float(awr_wmax)
        super().__init__(*args, **kwargs)

    def _make_model(self, **kwargs):
        return ISFBModel(actor_vf_cfg=self._actor_vf_cfg, **kwargs)

    def setup_training(self) -> None:
        super().setup_training()
        # F_V rides the forward optimizer (lr_f), matching FB's optimizer grouping.
        self.forward_optimizer.add_param_group(
            {"params": list(self.model._value_forward_map.parameters())}
        )
        self._value_forward_map_paramlist = tuple(self.model._value_forward_map.parameters())
        self._target_value_forward_map_paramlist = tuple(self.model._target_value_forward_map.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_agent.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/fb/isfb/agent.py tests/test_torch_isfb_agent.py
git commit -m "feat(isfb): ISFBAgent skeleton + F_V optimizer/target wiring"
```

---

### Task 3: `update_fb` override — the SG-ISFB loss

**Files:**
- Modify: `agents/fb/isfb/agent.py` (add `update_fb`, imports)
- Test: `tests/test_torch_isfb_agent.py` (extend)

**Interfaces:**
- Consumes: `expectile_loss`, `fb_successor_terms`, `ortho_cov` from `agents.fb.agent`; `_soft_update_params` from `nn_models`; the `update_fb(obs, action, discount, next_obs, goal, z, q_loss_coef, clip_grad_norm, w=None, weight_diag=False, next_action_override=None, goal_obs=None)` signature called by `FBAgent.update()`.
- Produces: `update_fb` returns a metrics dict containing `L_V`, `L_N`, `L_M`, `orth_loss`, `adv_mean`, `M_mean`. Bootstraps ONLY through `_target_value_forward_map` (never samples a policy action).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_torch_isfb_agent.py

def _batch(B=16):
    return {
        "observation": torch.randn(B, 4),
        "action": torch.rand(B, 2) * 2 - 1,
        "next": {"observation": torch.randn(B, 4), "terminated": torch.zeros(B, 1)},
    }


def _run(a, B=16, steps=1):
    m = None
    for s in range(steps):
        m = a.update(_batch(B), step=s)
    return m


def test_update_reports_isfb_losses_all_variants():
    for mode in ("coord", "scalar_gate_hard", "scalar_gate_awr"):
        a = _tiny_isfb(measure_insample=mode)
        m = _run(a)
        for k in ("L_N", "L_M", "orth_loss", "actor_loss"):
            assert k in m and np.isfinite(m[k]), (mode, k)
        if mode != "coord":
            assert "L_V" in m and np.isfinite(m["L_V"])


def test_coord_variant_has_no_scalar_value_loss():
    a = _tiny_isfb(measure_insample="coord")
    m = _run(a)
    assert m.get("L_V", 0.0) == 0.0            # coord omits L_V (isfb.tex Eq. coord_total)


def test_bootstrap_uses_target_fv_not_policy_action():
    # The SG-ISFB measure TD bootstraps through N^- = <target_F_V(s'), B>, NOT through a
    # policy next-action. Poison the policy-action sampler; the critic update must not touch it.
    a = _tiny_isfb(measure_insample="scalar_gate_hard")

    def _boom(*args, **kwargs):
        raise AssertionError("critic update sampled a policy action (a'~pi)")

    a.sample_action_from_norm_obs = _boom      # used only by the standard-FB measure bootstrap
    m = _run(a)                                # must succeed => no a'~pi in ISFB critic
    assert np.isfinite(m["L_M"])


def test_hard_weights_split_by_advantage_sign():
    # tau-hard weights: w = tau for A>0, (1-tau) for A<=0. Verify the helper directly.
    from agents.fb.isfb.agent import isfb_hard_weights
    A = torch.tensor([1.0, -1.0, 2.0, -0.5])
    tau = 0.9
    w = isfb_hard_weights(A, tau)
    assert torch.allclose(w, torch.tensor([tau, 1 - tau, tau, 1 - tau]))


def test_value_forward_params_change_after_update():
    a = _tiny_isfb(measure_insample="scalar_gate_hard")
    before = [p.detach().clone() for p in a.model._value_forward_map.parameters()]
    _run(a, steps=3)
    after = list(a.model._value_forward_map.parameters())
    assert any(not torch.allclose(b, x) for b, x in zip(before, after))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_agent.py -v`
Expected: FAIL — `ImportError: cannot import name 'isfb_hard_weights'` and the update tests error/NaN (default `update_fb` uses the measure loss and would call the poisoned sampler).

- [ ] **Step 3: Write minimal implementation**

Add to the top of `agents/fb/isfb/agent.py`:

```python
from typing import Dict

import torch
from torch.amp import autocast

from agents.fb.agent import expectile_loss, fb_successor_terms, ortho_cov
from nn_models import _soft_update_params


def isfb_hard_weights(A: torch.Tensor, tau: float) -> torch.Tensor:
    """tau-hard expectile weights: tau where advantage>0, (1-tau) elsewhere."""
    return torch.where(A > 0, torch.as_tensor(tau, dtype=A.dtype, device=A.device),
                       torch.as_tensor(1.0 - tau, dtype=A.dtype, device=A.device))
```

Add the method to `ISFBAgent`:

```python
    def update_fb(self, obs, action, discount, next_obs, goal, z,
                  q_loss_coef, clip_grad_norm, w=None, weight_diag=False,
                  next_action_override=None, goal_obs=None) -> Dict[str, torch.Tensor]:
        tau = self.iql_expectile
        zero_a = torch.zeros_like(action)
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            # target value successor measure N^- at the next state (NO a'~pi).
            with torch.no_grad():
                next_left = self.model._target_left_encoder(next_obs)
                tFv = self.model._target_value_forward_map(next_left, z, zero_a)  # [P,B,z]
                tB = self.model._target_backward_map(goal)                        # [B,z]
                tN = torch.matmul(tFv, tB.T)                                      # [P,B,B]
                _, _, tN_M = self.get_targets_uncertainty(tN, self.fb_pessimism_penalty)  # [B,B]

            left = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left, z, action)          # [P,B,z]
            Fv = self.model._value_forward_map(left, z, zero_a)    # [P,B,z]
            B = self.model._backward_map(goal)                     # [B,z]
            M = torch.matmul(Fs, B.T)                              # [P,B,B]
            N = torch.matmul(Fv, B.T)                              # [P,B,B]

            # scalar readouts, matching the flow actor's Q exactly (no /sqrt(d)).
            q = (Fs * z).sum(-1)                                   # [P,B]
            v = (Fv * z).sum(-1)                                   # [P,B]

            # (1) scalar gate: IQL expectile value loss (the only action-selection operator).
            L_V = expectile_loss(q.detach() - v, tau)
            A = (q - v).detach().mean(0)                           # head-mean advantage [B]

            # (2) improve the vector value measure N under the scalar gate.
            if self.measure_insample == "coord":
                L_N = expectile_loss(M.detach() - N, tau)          # elementwise expectile of M
            else:
                if self.measure_insample == "scalar_gate_awr":
                    wgt = torch.clamp(torch.exp(self.awr_beta * A), 0.0, self.awr_wmax)
                else:  # scalar_gate_hard
                    wgt = isfb_hard_weights(A, tau)
                wbar = wgt / (wgt.mean() + 1e-6)                   # [B]
                L_N = (wbar.view(1, -1, 1) * (M.detach() - N).pow(2)).mean()

            # (3) measure TD: M bootstraps through the target value measure at s' (no a'~pi).
            diff = M - discount * tN_M                             # [P,B,B]
            fb_offdiag, fb_diag = fb_successor_terms(diff, self.off_diag, self.off_diag_sum)
            L_M = fb_offdiag + fb_diag

            # (4) FB orthonormality on B (coef 1.0, unchanged from standard FB).
            Cov = ortho_cov(B, None)
            orth_loss = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum - Cov.diag().mean()

            loss = self.alpha_n * L_N + self.alpha_m * L_M + self.ortho_coef * orth_loss
            if self.measure_insample != "coord":
                loss = loss + L_V

        self.forward_optimizer.zero_grad(set_to_none=True)
        if self.backward_optimizer is not None:
            self.backward_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._value_forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()

        with torch.no_grad():
            _soft_update_params(self._value_forward_map_paramlist,
                                self._target_value_forward_map_paramlist, self.f_target_tau)

        return {
            "L_V": (L_V.detach() if self.measure_insample != "coord" else torch.zeros((), device=z.device)),
            "L_N": L_N.detach(), "L_M": L_M.detach(), "orth_loss": orth_loss.detach(),
            "adv_mean": A.mean().detach(), "M_mean": M.mean().detach(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_agent.py -v`
Expected: PASS (all tests). If a run NaNs, confirm the batch is random (not all-zeros — a rank-degenerate batch NaNs the measure objective; the JAX version hit this).

- [ ] **Step 5: Commit**

```bash
git add agents/fb/isfb/agent.py tests/test_torch_isfb_agent.py
git commit -m "feat(isfb): SG-ISFB update_fb (L_V/L_N/L_M, N^- bootstrap, no a'~pi)"
```

---

### Task 4: `train.py` wiring + agent config

**Files:**
- Modify: `train.py` (add `isfb_flowbc` branch + ISFB kwargs)
- Create: `configs/agent/isfb_flowbc.yaml`
- Test: `tests/test_torch_isfb_wiring.py`

**Interfaces:**
- Consumes: the existing `shared` kwargs dict and `cfg.agent` dispatch in `train.py` (see the `fb_flowbc` branch ~line 258); `agents.fb.isfb.agent.ISFBAgent`.
- Produces: `cfg.agent == "isfb_flowbc"` builds an `ISFBAgent`; `configs/agent/isfb_flowbc.yaml` carries `agent: isfb_flowbc` + the ISFB knobs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torch_isfb_wiring.py
import os
import numpy as np


def test_config_file_parses_and_selects_isfb():
    import yaml
    p = "configs/agent/isfb_flowbc.yaml"
    assert os.path.exists(p)
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert cfg["agent"] == "isfb_flowbc"
    assert cfg["measure_insample"] in ("coord", "scalar_gate_hard", "scalar_gate_awr")
    assert cfg["critic_mode"] == "measure"      # ISFB uses the measure-branch update() path
    assert float(cfg["ortho_coef"]) == 1.0
    assert float(cfg["train_goal_ratio"]) == 1.0  # Exp1 exact-goal


def test_train_has_isfb_branch():
    with open("train.py") as f:
        src = f.read()
    assert 'cfg.agent == "isfb_flowbc"' in src
    assert "ISFBAgent(" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_wiring.py -v`
Expected: FAIL — config file missing / branch absent.

- [ ] **Step 3: Write minimal implementation**

First read `configs/agent/fb.yaml` to mirror its keys, then create the ISFB config by copying it and setting/adding the keys below:

```yaml
# configs/agent/isfb_flowbc.yaml
# SG-ISFB (PAPER/fb-modified/isfb.tex) on the torch FB agent. Copy of the fb_flowbc
# agent config with the ISFB knobs added. Exp1 = exact-goal (train_goal_ratio=1.0).
agent: isfb_flowbc

# --- inherit the fb_flowbc cube settings (copy from configs/agent/fb.yaml) ---
# (z_dim, L_dim, actor{}, actor_vf{}, flow_steps, lr_*, bc_coeff=3.0, discount, etc.)

# --- standard-FB knobs pinned for ISFB Exp1 ---
critic_mode: measure       # ISFB overrides update_fb; stays on the measure branch of update()
ortho_coef: 1.0            # isfb.tex: harmless FB scale, NOT 1000
train_goal_ratio: 1.0      # Exp1 exact-goal: z = B(goal) always
iql_expectile: 0.9         # tau for L_V and the coord path

# --- ISFB additions ---
measure_insample: scalar_gate_hard   # coord | scalar_gate_hard | scalar_gate_awr
alpha_n: 1.0
alpha_m: 1.0
awr_beta: 3.0
awr_wmax: 100.0
```

In `train.py`, add the five ISFB keys to the `shared` dict is NOT needed (they are agent-branch-specific). Add this branch immediately after the `fb_flowbc` branch (after its `return`, ~line 274):

```python
    if cfg.agent == "isfb_flowbc":
        from agents.fb.isfb.agent import ISFBAgent
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return ISFBAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            measure_insample=cfg.get("measure_insample", "scalar_gate_hard"),
            alpha_n=cfg.get("alpha_n", 1.0),
            alpha_m=cfg.get("alpha_m", 1.0),
            awr_beta=cfg.get("awr_beta", 3.0),
            awr_wmax=cfg.get("awr_wmax", 100.0),
            **shared,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_wiring.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add train.py configs/agent/isfb_flowbc.yaml tests/test_torch_isfb_wiring.py
git commit -m "feat(isfb): wire isfb_flowbc into train.py + agent config"
```

---

### Task 5: Full-suite green + Exp1 launch playbook (no new code)

**Files:** none (verification + run).

- [ ] **Step 1: Run the ISFB test suite**

Run: `.venv/bin/python -m pytest tests/test_torch_isfb_model.py tests/test_torch_isfb_agent.py tests/test_torch_isfb_wiring.py -v`
Expected: all PASS.

- [ ] **Step 2: Confirm no regression in the FB baseline tests**

Run: `.venv/bin/python -m pytest tests/test_fb_iql.py -v`
Expected: all PASS (we did not touch `agents/fb/agent.py`).

- [ ] **Step 3: Launch Exp1 band on cube (state), in tmux, wandb entity `amsks`**

Establish the torch FB family band on this substrate. All in a detached tmux session (see memory: always-run-in-tmux, wandb-entity-amsks, host-identity-you-are-aws-gpu). Per seed:
- **C4-analog:** plain FB FlowBC (`configs/agent/fb.yaml`, d50).
- **B4-analog:** FB-IQL — `configs/agent/fb.yaml` with `critic_mode=iql train_goal_mode=traj` (run if no cube number exists yet).
- **SG-ISFB:** `configs/agent/isfb_flowbc.yaml` with `measure_insample ∈ {coord, scalar_gate_hard, scalar_gate_awr}`, `train_goal_ratio=1.0`.

- [ ] **Step 4: Collect `evaluation/overall_success`** for all cells and compare SG-ISFB (hard/awr) vs the B4-analog and vs coord/C4. Record in `PAPER/fb-modified/` per the isfb.tex "Expected outcomes" table.

---

## Self-Review

**Spec coverage:**
- F_V leg (spec §Architecture) → Task 1. ✅
- Agent subclass, flow actor unchanged, optimizer/target wiring (spec §Architecture) → Task 2. ✅
- `L_V`/advantage/weights/`L_N`/`L_M` N⁻ bootstrap, coord+hard+awr, ortho=1.0 (spec §update_isfb, §config) → Task 3. ✅
- Knobs `measure_insample`/`alpha_n`/`alpha_m`/`awr_beta`/`awr_wmax` (spec §4) → Task 2 ctor + Task 4 config. ✅
- Exact-goal via `train_goal_ratio=1.0` (spec §5) → Task 4 config. ✅
- Torch FB-family band incl. FB-IQL B4-analog (spec §5) → Task 5. ✅
- TDD test list incl. "N⁻ not a'~π" and random-batch smoke (spec §6) → Tasks 1–3. ✅
- Non-goals: no edits to `agents/fb/agent.py|model.py`; no Exp2/3; no pixel (spec §Non-goals) → honored (Global Constraints). ✅

**Placeholder scan:** the config yaml intentionally references "copy the inherited keys from `configs/agent/fb.yaml`" — this is a concrete instruction (read sibling, mirror), not a TODO; the ISFB-specific keys are all given verbatim. No other placeholders.

**Type consistency:** `measure_insample`/`alpha_n`/`alpha_m`/`awr_beta`/`awr_wmax` names identical across model/agent/config/train.py. `_value_forward_map` / `_target_value_forward_map` consistent between Task 1 (model) and Tasks 2–3 (agent). `isfb_hard_weights(A, tau)` signature matches its test. `update_fb` signature matches `FBAgent.update()`'s call site.
