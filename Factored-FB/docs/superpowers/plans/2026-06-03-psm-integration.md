# PSM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate PSM (Proto Successor Measure) into our codebase as a faithful baseline alongside FB, plugged into the state (FlowBC) and pixel (DrQ+FlowBC) pipelines, with a bit-exact reference-equivalence test as the correctness oracle.

**Architecture:** New `agents/psm/` package mirroring `agents/fb/` + `agents/rldp/` (Approach A). PSM's φ/ψ/actor nets are *vendored verbatim* from the reference (`/home/adityamohan/git/baselines/PSM/models/`) so the bit-exact test passes; PSM-specific logic (dual-ψ, binary/Gaussian z, proto behavior-sampler, reward-inference) lives in `agents/psm/{model,agent,proto_sampler}.py`. `PSMAgent` uses the TD3 actor (reference-equivalent); `PSMFlowBCAgent` swaps in our FlowBC actor. Pixel reuses the existing `build_pixel_cfgs` DrQ wiring. FB code is untouched.

**Tech Stack:** PyTorch, Hydra configs, OGBench (mujoco), pytest. Reference impl at `/home/adityamohan/git/baselines/PSM` (read-only). Tests run with `.venv/bin/python -m pytest` (NOT `uv run` — broken in this repo; 9 pre-existing pandas/script import failures are unrelated).

**Spec:** `docs/superpowers/specs/2026-06-03-psm-integration-design.md`

**Key verified facts the executor must know:**
- Reference nets (`models/psm_models.py`, `models/modules.py`, `models/parallel_modules.py`) are pure PyTorch and self-contained — vendor them verbatim (only changing the `TruncatedNormal` import).
- Two z dims: SF/task z is `z_dim` (default 128, Gaussian-normalized); proto z is `max_log_seed`-dim (default 16, binary). `phi` and BOTH ψ output `z_dim`; `psm_ψ` conditions on a 16-dim z, `sf_ψ` on a 128-dim z.
- `reward_inference`: `z = (reward.T @ phi(goal)) / N`, then `z = sqrt(z_dim)*normalize(z)` if `norm_z`. Matches FB; `evals/ogbench.py:_infer_z` calls `agent.model.reward_inference(next_obs, reward)` and works unchanged.
- Reference `next_observation_hash` is just the buffer ROW INDEX (`psm.py:255`; content-hash commented out). The proto-sampler seed = `(⟨z,2^i⟩ + row_index) % max_seed`, `max_seed = 2^max_log_seed + 20000`. Per-seed action = `(torch.rand(action_dim) - 1)*2` under `torch.random.manual_seed(seed_i)`.
- Update order per step (`psm.py:695–773`): sample binary z → `update_psm` (psm_ψ + φ, ortho) → soft-update psm_ψ,φ targets → sample Gaussian z (+ mix_ratio rows = φ(goal_perm)) → `update_sf` (sf_ψ, φ detached) → `update_actor` → soft-update sf_ψ target.
- Our conventions to mirror: `agents/fb/model.py` (FBModel: encoders, `_obs_normalizer`, `_augmentator`, autocast, target nets), `agents/fb/agent.py` (FBAgent: device, `update`, `state_dict`), `agents/fb/flow_bc/agent.py` (FlowBC actor + vf), `train.py:make_agent` + `shared`/`build_pixel_cfgs`, `evals/ogbench.py`, `scripts/agents/fb/state.sh`.
- Commits MUST NOT include a `Co-Authored-By: Claude` trailer (repo convention).

---

## File Structure

- **Create** `agents/psm/__init__.py` — exports `PSMAgent`, `PSMModel`.
- **Create** `agents/psm/psm_nets.py` — verbatim vendored reference nets: `mlp`, `parallel_mlp` (+ `DenseParallel`, `ParallelLayerNorm`, `Norm`, `_L2`, `weight_init`, `parallel_orthogonal_`), `PhiMap`, `PsiMap`, `SimpleActor`, `Actor`. One responsibility: the exact PSM network primitives.
- **Create** `agents/psm/proto_sampler.py` — `ProtoBehaviorSampler` (reference `SamplingSeedActor` port).
- **Create** `agents/psm/model.py` — `PSMModel(nn.Module)`: builds φ/sf_ψ/psm_ψ/actor + targets + encoders/normalizer; `sample_z`, `sample_z_psm`, `int_to_binary_array`, `get_targets_uncertainty`, `reward_inference`, `act`, normalize/encode helpers.
- **Create** `agents/psm/agent.py` — `PSMAgent(BaseAgent)`: `update` (dual-ψ + actor), `_update_psm`, `_update_sf`, `_update_actor` (TD3), `sample_mixed_z`, soft-updates, `act`, `infer_z`, `state_dict`/`load_state_dict`.
- **Create** `agents/psm/flow_bc/__init__.py`, `agents/psm/flow_bc/agent.py` — `PSMFlowBCAgent(PSMAgent)`: FlowBC actor (vector field + Q-driven head from sf_ψ·z), mirroring `agents/fb/flow_bc/agent.py`.
- **Create** `configs/agent/psm.yaml`, `configs/agent/psm_flowbc.yaml`.
- **Modify** `train.py` — `make_agent` gains `psm` / `psm_flowbc` branches; PSM-specific keys added to a `psm_shared` dict.
- **Modify** `data/` buffer loader — add optional `row_index` passthrough for the proto-sampler (default-off; see Task 4.1). [Executor: locate the loader via `grep -rn "next\"\]\[\"physics\|class .*Buffer\|def sample" data/ agents/` first.]
- **Create** `scripts/agents/psm/state.sh`, `scripts/agents/psm/pixel.sh`.
- **Create** tests under `tests/` per task.

---

## Phase 0 — Scaffolding (default-off, no behavior change)

### Task 0.1: Vendor the reference net primitives verbatim

**Files:**
- Create: `agents/psm/__init__.py` (empty for now)
- Create: `agents/psm/psm_nets.py`
- Test: `tests/test_psm_nets_build.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from agents.psm.psm_nets import PhiMap, PsiMap, SimpleActor, Actor, mlp, parallel_mlp


def test_phimap_shape_and_norm():
    phi = PhiMap(goal_dim=40, z_dim=128, hidden_dim=256, hidden_layers=2, norm=True)
    out = phi(torch.randn(8, 40))
    assert out.shape == (8, 128)
    # norm=True => Norm() last layer: rows have L2 norm sqrt(z_dim)
    import math
    assert torch.allclose(out.norm(dim=-1), torch.full((8,), math.sqrt(128)), atol=1e-4)


def test_psimap_parallel_output_dim():
    # sf-style: z input dim 128, output z_dim 128, num_parallel 2
    sf = PsiMap(obs_dim=40, z_dim=128, action_dim=5, hidden_dim=1024,
                hidden_layers=1, embedding_layers=2, num_parallel=2)
    out = sf(torch.randn(8, 40), torch.randn(8, 128), torch.randn(8, 5))
    assert out.shape == (2, 8, 128)
    # psm-style: z input dim 16 (binary), output_dim forced to 128
    psm = PsiMap(obs_dim=40, z_dim=16, action_dim=5, hidden_dim=1024,
                 hidden_layers=1, embedding_layers=2, num_parallel=2, output_dim=128)
    out2 = psm(torch.randn(8, 40), torch.randn(8, 16), torch.randn(8, 5))
    assert out2.shape == (2, 8, 128)


def test_actor_returns_truncated_normal():
    act = Actor(obs_dim=40, z_dim=128, action_dim=5, hidden_dim=1024,
                hidden_layers=1, embedding_layers=2)
    dist = act(torch.randn(8, 40), torch.randn(8, 128), 0.2)
    a = dist.sample(clip=0.3)
    assert a.shape == (8, 5) and a.abs().max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_psm_nets_build.py -v`
Expected: FAIL (`ModuleNotFoundError: agents.psm.psm_nets`)

- [ ] **Step 3: Implement `agents/psm/psm_nets.py` by transcribing the reference VERBATIM.**

Transcribe, in order, into a single file `agents/psm/psm_nets.py`:
1. From `/home/adityamohan/git/baselines/PSM/models/modules.py` (lines 11–67): `_L2`, `_nl`, `Norm`, `mlp`. Copy verbatim.
2. From `/home/adityamohan/git/baselines/PSM/models/parallel_modules.py` (lines 14–213): `weight_init`, `parallel_orthogonal_`, `DenseParallel`, `ParallelLayerNorm`, `_parallel_nl`, `parallel_mlp`. Copy verbatim.
3. From `/home/adityamohan/git/baselines/PSM/models/psm_models.py` (lines 14–128): `PhiMap`, `build_embedding`, `build_embedding_relu`, `PsiMap`, `SimpleActor`, `Actor`. Copy verbatim **except** change the `TruncatedNormal` import from `from ..utils import TruncatedNormal` to `from agents.psm._truncated_normal import TruncatedNormal` (created in Step 3a).

Top of file:
```python
"""agents/psm/psm_nets.py — VERBATIM port of the PSM reference networks.

Sources (copied unchanged so the bit-exact reference-equivalence test passes):
  models/modules.py        -> _L2, _nl, Norm, mlp
  models/parallel_modules.py-> weight_init, parallel_orthogonal_, DenseParallel,
                               ParallelLayerNorm, _parallel_nl, parallel_mlp
  models/psm_models.py     -> PhiMap, build_embedding(_relu), PsiMap, SimpleActor, Actor
Only change vs reference: TruncatedNormal import path. DO NOT "improve" anything here —
fidelity to op-order is what makes the equivalence test bit-exact.
"""
import math
import numbers
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

- [ ] **Step 3a: Vendor `TruncatedNormal` verbatim.**

Create `agents/psm/_truncated_normal.py` by copying `TruncatedNormal` (and any helper it needs) verbatim from `/home/adityamohan/git/baselines/PSM/utils.py:143–164`. (Read that range; it subclasses `torch.distributions.Normal` with a `sample(clip=...)` straight-through method.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_psm_nets_build.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/psm/__init__.py agents/psm/psm_nets.py agents/psm/_truncated_normal.py tests/test_psm_nets_build.py
git commit -m "feat(psm): vendor reference net primitives (mlp/parallel_mlp/PhiMap/PsiMap/Actor) verbatim"
```

### Task 0.2: Add PSM Hydra config (default values from the paper)

**Files:**
- Create: `configs/agent/psm.yaml`

- [ ] **Step 1: Write the config** (mirrors `configs/agent/fb.yaml` structure; values from reference `PSM_AgentConfig`/`ArchiConfig`/`OptimConfig`):

```yaml
# @package _global_
agent: psm

# representation
z_dim: 128
norm_z: true
max_log_seed: 16          # proto (binary) z dimension
phi_input: s              # s | as | sas | ss  (s => phi(next_obs))
norm_obs: false

batch_size: 1024
discount: 0.98            # match the domain default; overridden per-domain if needed
device: cuda

# phi (basis)
phi:
  hidden_dim: 256
  hidden_layers: 2
  norm: true
  batch_norm: false

# sf / psm psi (successor nets)
sf:
  hidden_dim: 1024
  hidden_layers: 1
  embedding_layers: 2
  num_parallel: 2

# TD3 actor (PSMAgent; replaced by FlowBC in psm_flowbc)
actor:
  hidden_dim: 1024
  hidden_layers: 1
  embedding_layers: 2

# optimization
lr_sf: 1.0e-4
lr_phi: 1.0e-4
lr_actor: 1.0e-4
weight_decay: 0.0
clip_grad_norm: 0.0
target_tau: 0.01
ortho_coef: 1.0
mix_ratio: 0.5
pessimism_penalty: 0.0
actor_pessimism_penalty: 0.5
stddev: 0.2               # fixed actor std (reference stddev_schedule="0.2")
stddev_clip: 0.3
amp: false
```

- [ ] **Step 2: Verify it loads**

Run: `.venv/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/agent/psm.yaml'); print(c.agent, c.z_dim, c.max_log_seed)"`
Expected: `psm 128 16`

- [ ] **Step 3: Commit**

```bash
git add configs/agent/psm.yaml
git commit -m "feat(psm): add psm agent config (paper defaults)"
```

---

## Phase 1 — Proto behavior-sampler (reference `SamplingSeedActor`)

### Task 1.1: `ProtoBehaviorSampler`

**Files:**
- Create: `agents/psm/proto_sampler.py`
- Test: `tests/test_psm_proto_sampler.py`

- [ ] **Step 1: Write the failing test** (asserts byte-equivalence to the reference table + indexing):

```python
import numpy as np
import torch
from agents.psm.proto_sampler import ProtoBehaviorSampler


def _reference_table(action_dim, max_log_seed):
    # Transcribed from PSM SamplingSeedActor.__init__ (psm.py:35-43)
    max_seed = 2 ** max_log_seed + 20000
    seed_to_action = []
    for i in range(max_seed):
        torch.random.manual_seed(i)
        action = (torch.rand(size=(action_dim,)).unsqueeze(0) - 1) * 2
        seed_to_action.append(action)
    return np.array(seed_to_action).squeeze(), max_seed


def test_table_matches_reference():
    s = ProtoBehaviorSampler(action_dim=5, max_log_seed=8, batch_size=4)  # small for speed
    ref_table, ref_max = _reference_table(5, 8)
    assert s.max_seed == ref_max
    assert np.allclose(s.seed_to_action, ref_table)


def test_forward_indexing_matches_reference():
    s = ProtoBehaviorSampler(action_dim=5, max_log_seed=8, batch_size=4)
    z = torch.randint(0, 2, (4, 8)).float()
    obs_hash = torch.tensor([0, 1, 2, 3])
    powers = torch.tensor([2 ** i for i in range(8)][::-1]).repeat(4, 1)
    seed_long = (z * powers).sum(1)
    final_seed = ((seed_long + obs_hash.reshape(-1)) % s.max_seed).cpu().numpy().astype(np.int32)
    expected = torch.FloatTensor(s.seed_to_action[final_seed])
    out = s(obs_hash, z)
    assert torch.equal(out, expected)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_psm_proto_sampler.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement** `agents/psm/proto_sampler.py` (port of `SamplingSeedActor`, `psm.py:29–53`, device-generalized — `.to(device)` instead of hardcoded `'cuda'`):

```python
"""agents/psm/proto_sampler.py — port of PSM's SamplingSeedActor (psm.py:29-53).

A deterministic pseudo-random behavior policy keyed by (row_index, binary z) used as
the target next-action when learning the proto successor net psm_psi. The reference
hardcodes 'cuda'; we take a device. Table/indexing reproduced byte-for-byte.
"""
import numpy as np
import torch
import torch.nn as nn


class ProtoBehaviorSampler(nn.Module):
    def __init__(self, action_dim: int, max_log_seed: int, batch_size: int, device: str = "cpu") -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.powers = torch.tensor([2 ** i for i in range(max_log_seed)][::-1]).to(device).repeat(batch_size, 1)
        self.max_seed = 2 ** max_log_seed + 20000
        seed_to_action = []
        for i in range(self.max_seed):
            torch.random.manual_seed(i)
            action = (torch.rand(size=(self.action_dim,)).unsqueeze(0) - 1) * 2
            seed_to_action.append(action)
        self.seed_to_action = np.array(seed_to_action).squeeze()

    def forward(self, obs_hash: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        seed_long = (z * self.powers).sum(1)
        final_seed = (seed_long + obs_hash.reshape(-1)) % self.max_seed
        actions = self.seed_to_action[final_seed.cpu().numpy().astype(np.int32)]
        return torch.FloatTensor(actions).to(self.device)
```

> NOTE: `self.powers` is built for a fixed `batch_size`; `forward` requires `z.shape[0] == batch_size`. The PSM update always uses full-batch z, so this holds. Add an assert in `forward`: `assert z.shape[0] == self.powers.shape[0]`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_psm_proto_sampler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/psm/proto_sampler.py tests/test_psm_proto_sampler.py
git commit -m "feat(psm): ProtoBehaviorSampler (SamplingSeedActor port), byte-equiv to reference"
```

---

## Phase 2 — PSMModel

### Task 2.1: `PSMModel` construction + z samplers + reward_inference

**Files:**
- Create: `agents/psm/model.py`
- Test: `tests/test_psm_model.py`

Mirror `agents/fb/model.py` for our-codebase conventions: read it first for the `_obs_normalizer`, `_fw_encoder`/`_bw_encoder` (Identity for state, DrQ for pixel), `_augmentator`, `_normalize`, autocast, and `project_z` patterns. For state, encoders are Identity; pixel reuses `rgb_encoder_cfg`/`augmentator_cfg`.

- [ ] **Step 1: Write the failing test**

```python
import math
import gymnasium as gym
import torch
from agents.psm.model import PSMModel


def _model(**kw):
    return PSMModel(obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5,
                    z_dim=128, max_log_seed=16, batch_size=8, device="cpu", **kw)


def test_builds_nets_and_targets():
    m = _model()
    assert m.phi is not None and m.sf_psi is not None and m.psm_psi is not None and m.actor is not None
    # targets exist for sf_psi, psm_psi, phi
    assert m.target_sf_psi is not None and m.target_psm_psi is not None and m.target_phi is not None


def test_sample_z_gaussian_normalized():
    m = _model(norm_z=True)
    z = m.sample_z(8, device="cpu")
    assert z.shape == (8, 128)
    assert torch.allclose(z.norm(dim=-1), torch.full((8,), math.sqrt(128)), atol=1e-4)


def test_sample_z_psm_binary_16d():
    m = _model()
    z = m.sample_z_psm(8, device="cpu")
    assert z.shape == (8, 16)
    assert set(torch.unique(z).tolist()) <= {0.0, 1.0}


def test_reward_inference_shape_and_norm():
    m = _model(norm_z=True)
    next_obs = torch.randn(32, 40)
    reward = torch.randn(32, 1)
    z = m.reward_inference(next_obs, reward)
    assert z.shape == (1, 128)
    assert torch.allclose(z.norm(dim=-1), torch.tensor([math.sqrt(128)]), atol=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_psm_model.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `agents/psm/model.py`.** Construction mirrors reference `psm.py:170–224` but uses our obs-normalizer/encoder configs (default Identity for state). Key methods:

```python
"""agents/psm/model.py — PSM representation model (PyTorch), our-codebase conventions.

Networks/targets/optim parameter lists and forward math mirror the reference
agent/psm.py:170-224 + models/psm_models.py. State: encoders are Identity. Pixel:
rgb_encoder_cfg/augmentator_cfg wire the DrQ front-end exactly as agents/fb/model.py.
"""
from __future__ import annotations
import copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from agents.psm.psm_nets import PhiMap, PsiMap, Actor, SimpleActor, weight_init
from agents.psm.proto_sampler import ProtoBehaviorSampler
from nn_models import eval_mode  # reuse our eval_mode context (see nn_models.py)


class PSMModel(nn.Module):
    def __init__(
        self, obs_space, action_dim: int, z_dim: int = 128, max_log_seed: int = 16,
        batch_size: int = 1024, norm_z: bool = True, phi_input: str = "s",
        phi_cfg=None, sf_cfg=None, actor_cfg=None, actor_kind: str = "td3",
        obs_normalizer_cfg=None, rgb_encoder_cfg=None, augmentator_cfg=None,
        num_parallel: int = 2, device: str = "cpu", amp: bool = False,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.max_log_seed = max_log_seed
        self.batch_size = batch_size
        self.norm_z = norm_z
        self.phi_input = phi_input
        self.num_parallel = num_parallel
        self.device = device
        self.device_type = str(device).split(":")[0]
        self.amp = amp
        self.amp_dtype = torch.bfloat16

        # --- encoders / normalizer (mirror agents/fb/model.py; Identity for state) ---
        # Executor: copy FBModel's construction of _obs_normalizer, _fw_encoder,
        # _bw_encoder, _augmentator from rgb_encoder_cfg/augmentator_cfg/obs_normalizer_cfg.
        # For state these are Identity; obs_dim below uses the encoder output dim.
        # ... (build self._obs_normalizer, self._fw_encoder, self._bw_encoder, self._augmentator)
        obs_dim = obs_space.shape[0]  # state; for pixels use encoder output_space dim

        # phi_input -> goal_dim (reference psm.py:172-187). Default "s" => phi(next_obs).
        goal_dim = {"s": obs_dim, "as": obs_dim + action_dim,
                    "sas": 2 * obs_dim + action_dim, "ss": 2 * obs_dim}[phi_input]
        self.goal_dim = goal_dim

        phi_cfg = phi_cfg or {}
        sf_cfg = sf_cfg or {}
        actor_cfg = actor_cfg or {}
        self.phi = PhiMap(goal_dim, z_dim, phi_cfg.get("hidden_dim", 256),
                          phi_cfg.get("hidden_layers", 2), phi_cfg.get("norm", True),
                          phi_cfg.get("batch_norm", False))
        self.sf_psi = PsiMap(obs_dim, z_dim, action_dim, sf_cfg.get("hidden_dim", 1024),
                             sf_cfg.get("hidden_layers", 1), sf_cfg.get("embedding_layers", 2),
                             num_parallel)
        self.psm_psi = PsiMap(obs_dim, max_log_seed, action_dim, sf_cfg.get("hidden_dim", 1024),
                              sf_cfg.get("hidden_layers", 1), sf_cfg.get("embedding_layers", 2),
                              num_parallel, output_dim=z_dim)
        # actor: TD3 Actor for PSMAgent (FlowBC variant builds its own).
        if actor_kind == "td3":
            self.actor = Actor(obs_dim, z_dim, action_dim, actor_cfg.get("hidden_dim", 1024),
                               actor_cfg.get("hidden_layers", 1), actor_cfg.get("embedding_layers", 2))
        else:
            self.actor = None  # FlowBC variant attaches its own actor + vf

        self.proto_sampler = ProtoBehaviorSampler(action_dim, max_log_seed, batch_size, device)

        nets = {"sf_psi": self.sf_psi, "psm_psi": self.psm_psi, "phi": self.phi}
        if self.actor is not None:
            nets["actor"] = self.actor
        for net in nets.values():
            net.apply(weight_init)
        self.target_phi = copy.deepcopy(self.phi)
        self.target_sf_psi = copy.deepcopy(self.sf_psi)
        self.target_psm_psi = copy.deepcopy(self.psm_psi)

        self.num_parallel_scaling = num_parallel ** 2 - num_parallel
        self.to(device)

    # ---- our-codebase helpers (mirror FBModel) ----
    def _normalize(self, obs):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    def project_z(self, z):
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def int_to_binary_array(self, int_vector, num_bits):
        return ((int_vector[:, None] & (1 << np.arange(num_bits))) > 0).astype(int)

    def sample_z(self, size, device="cpu"):
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def sample_z_psm(self, size, device="cpu"):
        z_np = np.random.randint(0, 2 ** self.max_log_seed, (size,))
        binary = self.int_to_binary_array(z_np, self.max_log_seed)
        return torch.FloatTensor(binary).to(device)

    def get_targets_uncertainty(self, preds, dim=0):
        preds_mean = preds.mean(dim=dim)
        d1 = preds.unsqueeze(dim=dim)
        d2 = preds.unsqueeze(dim=dim + 1)
        preds_unc = torch.abs(d1 - d2).sum(dim=(dim, dim + 1)) / self.num_parallel_scaling
        return preds_mean, preds_unc

    @torch.no_grad()
    def reward_inference(self, next_obs, reward, weight=None):
        # matches reference infer_context_from_reward (psm.py:780-792); phi_input="s" => phi(next_obs)
        next_obs = next_obs.to(self.device).float()
        reward = reward.to(self.device).float()
        with eval_mode(self):
            goal = self._goal_from_next_obs(next_obs)   # phi_input routing; "s" => next_obs (normalized)
            phi = self.phi(goal)
        z = torch.matmul(reward.T, phi) / phi.shape[0]
        return self.project_z(z)

    def _goal_from_next_obs(self, next_obs):
        # phi_input="s": goal = normalized next_obs. Other modes need obs/action; for eval
        # (reward inference) only next_obs is available -> "s" is the supported default.
        assert self.phi_input == "s", "reward_inference supports phi_input='s'"
        return self._normalize(next_obs)
```

> Executor: fill the encoder/normalizer construction block by copying the corresponding lines from `agents/fb/model.py.__init__` (the `_obs_normalizer`, `_fw_encoder`, `_bw_encoder`, `_augmentator` builds and `obs_dim = encoder.output_space.shape[0]`). Keep state = Identity. Add an `act(self, obs, z, mean=True, goal=None)` method mirroring `FBModel.act` once the actor exists (used by eval; for TD3 actor: `dist=self.actor(enc_obs, z, self.actor_std); return dist.mean/sample`).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_psm_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/psm/model.py tests/test_psm_model.py
git commit -m "feat(psm): PSMModel (nets+targets, z samplers, reward_inference) — state"
```

---

## Phase 3 — PSMAgent (TD3 actor; reference-equivalent)

### Task 3.1: Buffer row-index passthrough for the proto-sampler

**Files:**
- Modify: the offline buffer/loader (locate via `grep -rn "def sample" data/ agents/`).
- Test: `tests/test_psm_buffer_rowindex.py`

- [ ] **Step 1:** Read how `update(batch, step)` receives data (FBAgent.update consumes `batch["observation"]`, `batch["next"]["observation"]`, `batch["next"]["terminated"]`, etc.). The proto-sampler needs a per-row integer `next_obs_hash`. Reference uses the buffer ROW INDEX. **Plan:** `PSMAgent.update` will derive `next_obs_hash` from a `batch["index"]` field if present, else fall back to `torch.arange(batch_size)` (a documented approximation). Add an optional `index` field to the sampled batch behind a flag so FB/others are unaffected.

- [ ] **Step 2: Write the failing test** asserting the loader can attach row indices when asked, and is unchanged otherwise. [Executor: write against the actual buffer class found in Step 1; assert `batch.get("index")` is `None` by default and a `[B]` long tensor when `with_index=True`.]

- [ ] **Step 3–5:** Implement the minimal `with_index` passthrough, run, commit:

```bash
git commit -m "feat(psm): optional batch row-index passthrough for proto-sampler (default off)"
```

### Task 3.2: `PSMAgent.update` — dual-ψ + TD3 actor

**Files:**
- Create: `agents/psm/agent.py`
- Test: `tests/test_psm_update.py`

Transcribe the loss math from the reference verbatim (substituting our attribute names): `_update_psm` (`psm.py:422–477`), `_update_sf` (`psm.py:480–532`), `_update_td3_actor` (`psm.py:642–667`), and the `update` orchestration (`psm.py:734–763`). Mirror `agents/fb/agent.py` for device/autocast/`update` signature/`state_dict`.

- [ ] **Step 1: Write the failing test**

```python
import math
import gymnasium as gym
import torch
from agents.psm.agent import PSMAgent


def _agent(**kw):
    return PSMAgent(obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8,
                    z_dim=128, max_log_seed=16, device="cpu", **kw)


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1,
            "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_one_update_finite():
    a = _agent()
    m = a.update(_batch(), step=0)
    for k in ["psm_loss", "sf_loss", "actor_loss"]:
        assert math.isfinite(m[k]), k
```

- [ ] **Step 2: Run to verify it fails** → module missing.

- [ ] **Step 3: Implement `agents/psm/agent.py`.** Structure (full update orchestration; loss bodies transcribed from reference — shown abbreviated here, executor copies the exact diag/offdiag/ortho lines):

```python
from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from agents.base import BaseAgent
from agents.psm.model import PSMModel
from nn_models import _soft_update_params, eval_mode


class PSMAgent(BaseAgent):
    def __init__(self, obs_space, action_dim, batch_size=1024, z_dim=128, max_log_seed=16,
                 phi_cfg=None, sf_cfg=None, actor_cfg=None, norm_z=True, phi_input="s",
                 obs_normalizer_cfg=None, rgb_encoder_cfg=None, augmentator_cfg=None,
                 num_parallel=2, discount=0.98, lr_sf=1e-4, lr_phi=1e-4, lr_actor=1e-4,
                 weight_decay=0.0, clip_grad_norm=0.0, target_tau=0.01, ortho_coef=1.0,
                 mix_ratio=0.5, pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
                 actor_std=0.2, stddev_clip=0.3, amp=False, device="cpu", actor_kind="td3"):
        self.obs_space = obs_space; self.action_dim = action_dim; self.batch_size = batch_size
        self.z_dim = z_dim; self.max_log_seed = max_log_seed; self.discount = discount
        self.lr_sf = lr_sf; self.lr_phi = lr_phi; self.lr_actor = lr_actor
        self.weight_decay = weight_decay; self.clip_grad_norm = clip_grad_norm
        self.target_tau = target_tau; self.ortho_coef = ortho_coef; self.mix_ratio = mix_ratio
        self.pessimism_penalty = pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std; self.stddev_clip = stddev_clip
        self.num_parallel = num_parallel; self.amp = amp
        self.device = device; self.device_type = str(device).split(":")[0]; self.amp_dtype = torch.bfloat16

        self.model = PSMModel(obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
                              phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
                              obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
                              num_parallel, device, amp)
        self.setup_training()
        self.off_diag = 1 - torch.eye(batch_size, batch_size, device=device)
        self.off_diag_sum = self.off_diag.sum()

    def setup_training(self):
        m = self.model
        self.optim_phi = torch.optim.Adam(m.phi.parameters(), lr=self.lr_phi, weight_decay=self.weight_decay)
        self.optim_sf_psi = torch.optim.Adam(m.sf_psi.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        self.optim_psm_psi = torch.optim.Adam(m.psm_psi.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        if m.actor is not None:
            self.optim_actor = torch.optim.Adam(m.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)

    # _update_psm: transcribe psm.py:438-477 (target via proto_sampler; psm_loss = diag+offdiag+1*ortho;
    #   optimize psm_psi + phi). _update_sf: transcribe psm.py:495-532 (target via learned actor; phi detached;
    #   sf_loss = diag+offdiag; optimize sf_psi). _update_actor: transcribe psm.py:651-667 (Q=(sf_psi*z).sum,
    #   pessimistic; actor_loss=-Q.mean; optimize actor). Use self.model.get_targets_uncertainty.

    def sample_mixed_z(self, goal):
        # mirror reference update() z + mix (psm.py:744-756): Gaussian z, mix_ratio rows = phi(goal_perm)
        z = self.model.sample_z(self.batch_size, device=self.device)
        perm = torch.randperm(self.batch_size, device=self.device)
        phi_in = goal[perm]
        if self.mix_ratio > 0:
            mix_idxs = torch.where(torch.rand(self.batch_size, device=self.device) < self.mix_ratio)[0]
            with torch.no_grad(), eval_mode(self.model):
                mix_z = self.model.phi(phi_in[mix_idxs]).detach()
            mix_z = self.model.project_z(mix_z)
            z[mix_idxs] = mix_z
        return z

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device)
        idx = batch.get("index")
        next_obs_hash = (idx.to(self.device) if idx is not None
                         else torch.arange(self.batch_size, device=self.device))
        discount = self.discount * (~terminated if terminated.dtype == torch.bool else (1.0 - terminated))
        # normalize obs (mirror FBAgent.update) ; goal = phi_input routing ("s" => next_obs)
        obs_n, next_obs_n = self.model._normalize(obs), self.model._normalize(next_obs)
        goal = next_obs_n  # phi_input="s"

        metrics = {}
        z_psm = self.model.sample_z_psm(self.batch_size, device=self.device)
        metrics.update(self._update_psm(obs_n, action, discount, next_obs_n, next_obs_hash, goal, z_psm))
        _soft_update_params(self.model.psm_psi.parameters(), self.model.target_psm_psi.parameters(), self.target_tau)
        _soft_update_params(self.model.phi.parameters(), self.model.target_phi.parameters(), self.target_tau)

        z = self.sample_mixed_z(goal)
        metrics.update(self._update_sf(obs_n, action, discount, next_obs_n, goal, z))
        metrics.update(self._update_actor(obs_n, z))
        _soft_update_params(self.model.sf_psi.parameters(), self.model.target_sf_psi.parameters(), self.target_tau)
        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def act(self, obs, z, *, eval_mode: bool = True):
        obs = obs.to(self.device).float(); z = z.to(self.device).float()
        if z.dim() == 1: z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    def infer_z(self, obs, reward):
        return self.model.reward_inference(obs, reward)

    def state_dict(self):
        return {"model": self.model.state_dict(),
                "optim_phi": self.optim_phi.state_dict(),
                "optim_sf_psi": self.optim_sf_psi.state_dict(),
                "optim_psm_psi": self.optim_psm_psi.state_dict(),
                **({"optim_actor": self.optim_actor.state_dict()} if self.model.actor is not None else {})}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optim_phi.load_state_dict(state["optim_phi"])
        self.optim_sf_psi.load_state_dict(state["optim_sf_psi"])
        self.optim_psm_psi.load_state_dict(state["optim_psm_psi"])
        if self.model.actor is not None and "optim_actor" in state:
            self.optim_actor.load_state_dict(state["optim_actor"])
```

> Executor: write `_update_psm`/`_update_sf`/`_update_actor` by transcribing the cited reference lines exactly (the diag/offdiag/ortho/Q computations), wrapping the forward in `autocast(self.device_type, self.amp_dtype, enabled=self.amp)` like FBAgent, and returning the same metric keys (`psm_loss`, `psm_diag`, `psm_offdiag`, `orth_loss`, `sf_loss`, `sf_diag`, `sf_offdiag`, `actor_loss`, `q`). `_soft_update_params` takes param iterables (see `nn_models.py` signature; match FBAgent's usage).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_psm_update.py -v`
Expected: PASS (finite losses)

- [ ] **Step 5: Commit**

```bash
git add agents/psm/agent.py tests/test_psm_update.py
git commit -m "feat(psm): PSMAgent dual-psi + TD3 actor update (transcribed from reference)"
```

---

## Phase 4 — Bit-exact reference-equivalence test (correctness oracle)

### Task 4.1: `tests/test_psm_reference_equivalence.py`

**Files:**
- Test: `tests/test_psm_reference_equivalence.py`

This is the non-negotiable oracle: identical inputs → `torch.equal` (zero ULP) outputs vs reference.

- [ ] **Step 1: Add the reference repo to `sys.path` and import its net classes UNDER OUR TORCH.** At test top:

```python
import sys, math, copy
import numpy as np
import pytest
import torch

REF = "/home/adityamohan/git/baselines/PSM"
sys.path.insert(0, REF)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
DT = torch.float64
DEV = "cpu"

ref_nets = pytest.importorskip("models.psm_models",
    reason="reference PSM nets not importable under our torch; fall back to verbatim-transcription oracle")
from models.psm_models import PhiMap as RefPhi, PsiMap as RefPsi, Actor as RefActor
from agents.psm.psm_nets import PhiMap as OurPhi, PsiMap as OurPsi, Actor as OurActor
```

> If `pytest.importorskip` skips (torchrl/dmc import failure), implement the FALLBACK in Step 5.

- [ ] **Step 2: Build matched nets in float64, copy reference state_dict into ours, assert forward equality.**

```python
def _copy(ref, ours):
    ours.load_state_dict(ref.state_dict()); return ours

def test_phi_forward_bit_exact():
    g = 40
    ref = RefPhi(g, 128, 256, 2, True, False).to(DEV).double().eval()
    ours = _copy(ref, OurPhi(g, 128, 256, 2, True, False).to(DEV).double().eval())
    x = torch.randn(8, g, dtype=DT)
    assert torch.equal(ref(x), ours(x))

def test_psi_forward_bit_exact():
    ref = RefPsi(40, 128, 5, 1024, 1, 2, 2).to(DEV).double().eval()
    ours = _copy(ref, OurPsi(40, 128, 5, 1024, 1, 2, 2).to(DEV).double().eval())
    o, z, a = torch.randn(8, 40, dtype=DT), torch.randn(8, 128, dtype=DT), torch.randn(8, 5, dtype=DT)
    assert torch.equal(ref(o, z, a), ours(o, z, a))
```

(`load_state_dict` succeeds iff our module structure matches the reference exactly — which the verbatim vendoring guarantees. If it fails, the vendoring diverged: fix `psm_nets.py`.)

- [ ] **Step 3: Loss + gradient + post-step equivalence.** Replicate ONE `_update_psm` and ONE `_update_sf` step on both. Because the reference agent (`agent/psm.py`) pulls in torchrl, transcribe the *loss math* verbatim into a local `_ref_psm_loss(...)`/`_ref_sf_loss(...)` in the test (cite `psm.py:450-464` / `:505-519`), run it on the reference nets, and run `PSMAgent._update_psm/_update_sf` on ours with the SAME injected inputs (obs, action, next_obs, goal, z, **next_action injected** to bypass the sampler/actor). Assert:

```python
def test_psm_loss_grads_params_bit_exact():
    # build ref + our nets (float64), copy weights, identical Adam (lr, betas, eps)
    # inject: obs, action, next_obs, goal, z(binary), next_action, discount, off_diag
    # 1) assert torch.equal on each loss term (psm_diag, psm_offdiag, orth_*, psm_loss)
    # 2) backward both; assert torch.equal on every .grad for phi + psm_psi
    # 3) optim.step() both; assert torch.equal on every parameter
    ...
```

Provide the full body transcribing the reference loss lines verbatim (diag = `-torch.diagonal(diff,1,2).mean()*P`, offdiag = `0.5*(diff*off_diag).pow(2).sum()/off_diag_sum`, ortho on `phi@phi.T`). Use the SAME `off_diag`/`off_diag_sum`, the SAME target nets (deep-copied, no soft-update inside the step), and bypass `next_action` by passing a fixed tensor to both.

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_psm_reference_equivalence.py -v`
Expected: PASS with `torch.equal` (zero ULP). If any assert fails, FIX OP ORDER in `agents/psm/{psm_nets,agent}.py` until exact — do NOT relax to `allclose`.

- [ ] **Step 5 (fallback, only if Step 1 skipped):** If reference modules can't import, the test transcribes BOTH the reference nets' forward AND loss verbatim from source (with `psm.py`/`psm_models.py:LINE` citations) as the oracle, and compares our classes against the transcription. Mark clearly that this is the transcription-oracle path. The bit-exact assertions are identical.

- [ ] **Step 6: Commit**

```bash
git add tests/test_psm_reference_equivalence.py
git commit -m "test(psm): bit-exact reference-equivalence (forward+loss+grad+params, torch.equal)"
```

---

## Phase 5 — PSMFlowBCAgent (FlowBC actor)

### Task 5.1: `PSMFlowBCAgent`

**Files:**
- Create: `agents/psm/flow_bc/__init__.py`, `agents/psm/flow_bc/agent.py`
- Test: `tests/test_psm_flowbc.py`

Mirror `agents/fb/flow_bc/agent.py` exactly, replacing FB's `_forward_map` with PSM's `sf_psi` for the Q term. The BC flow vector field stays goal/z-unconditioned; the Q-driven head uses `Q = (sf_psi(obs, z, actor_actions) * z).sum(-1)` pessimistic.

- [ ] **Step 1: Write the failing test**

```python
import math
import gymnasium as gym
import torch
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    return PSMFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=128, max_log_seed=16,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1, "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_flowbc_update_and_act():
    a = _agent()
    m = a.update(_batch(), step=0)
    assert math.isfinite(m["sf_loss"]) and math.isfinite(m["actor_loss"])
    out = a.act(torch.zeros(1, 40), torch.zeros(128))
    assert out.shape[-1] == 5
```

- [ ] **Step 2: Run to verify it fails** → module missing.

- [ ] **Step 3: Implement `agents/psm/flow_bc/agent.py`.** Subclass `PSMAgent` with `actor_kind="flowbc"`; attach the `NoiseConditionedActor` + `SimpleVectorField` (build like `FBFlowBCAgent`), add `actor_vf_optimizer`, override `_update_actor` to the FlowBC actor update (copy `agents/fb/flow_bc/agent.py:update_actor`, swapping `self.model._forward_map(...)` → `self.model.sf_psi(...)` and `self._fb_z(z)` → `z`). Override `sample_action_from_norm_obs` (used as the SF target next-action) to the FlowBC noise-conditioned actor, mirroring FB. Override `act`/`model.act` to the flow actor (copy FB's flow_bc `act` + `compute_flow_actions`).

> Executor: read `agents/fb/flow_bc/agent.py` and `agents/fb/flow_bc/model.py` in full; the PSM version is the same code with `forward_map`→`sf_psi` and `_fb_z(z)`→`z`. Keep BC vf goal-unconditioned.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_psm_flowbc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/psm/flow_bc/ tests/test_psm_flowbc.py
git commit -m "feat(psm): PSMFlowBCAgent (FlowBC actor driven by sf_psi*z)"
```

### Task 5.2: `psm_flowbc.yaml` config

**Files:**
- Create: `configs/agent/psm_flowbc.yaml`

- [ ] **Step 1: Write** (mirrors `fb_flowbc.yaml`):

```yaml
# @package _global_
defaults:
  - psm
  - _self_

agent: psm_flowbc

bc_coeff: 3.0          # manipulation default (cube); antmaze would use 0.3
flow_steps: 10
lr_actor_vf: 3.0e-4

actor:
  name: noise_conditioned_actor
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2

actor_vf:
  hidden_dim: 512
  hidden_layers: 4
```

- [ ] **Step 2: Verify load** → `.venv/bin/python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/agent/psm_flowbc.yaml').agent)"` → `psm_flowbc`.
- [ ] **Step 3: Commit** → `git commit -m "feat(psm): psm_flowbc agent config"`

---

## Phase 6 — Registration, eval, pixel, launcher

### Task 6.1: Register `psm` / `psm_flowbc` in `make_agent`

**Files:**
- Modify: `train.py` (`make_agent`)
- Test: `tests/test_psm_make_agent.py`

- [ ] **Step 1:** Read `train.py:95–240`. Add, before the final `return FBAgent(...)`, branches that build a `psm_shared` dict from `cfg` (z_dim, max_log_seed, phi/sf/actor sub-cfgs, lrs, target_tau, ortho_coef, mix_ratio, pessimism, actor_std, stddev_clip, norm_z, phi_input, num_parallel, obs_normalizer/rgb_encoder/augmentator via `build_pixel_cfgs(cfg)`, device, amp) and return `PSMAgent(**psm_shared)` or `PSMFlowBCAgent(actor_cfg=..., actor_vf_cfg=..., flow_steps=..., lr_actor_vf=..., **psm_shared)`. Add imports at top.
- [ ] **Step 2: Write the failing test** — build both agents from a minimal `DictConfig` and assert types + that a single `update(_batch())` returns finite losses (state). 
- [ ] **Steps 3–5:** Implement, run (`.venv/bin/python -m pytest tests/test_psm_make_agent.py -v`), commit:

```bash
git commit -m "feat(psm): register psm + psm_flowbc in make_agent"
```

### Task 6.2: Eval reward-inference path

**Files:**
- Test: `tests/test_psm_eval_inference.py`

- [ ] **Step 1:** Confirm `evals/ogbench.py:_infer_z` calls `self.agent.model.reward_inference(next_obs=..., reward=...)` and that PSM's signature matches (it does — Task 2.1). No code change expected; if `_infer_z` uses a different call shape, adapt `PSMModel.reward_inference` to match FB's exactly (read `agents/fb/model.py.reward_inference` + `evals/ogbench.py:133–167`).
- [ ] **Step 2: Write a test** that builds a `PSMFlowBCAgent`, calls `agent.model.reward_inference(next_obs, reward)` with `[N,obs]`/`[N,1]`, asserts z shape `[1,z_dim]` and (norm_z) norm `sqrt(z_dim)`. 
- [ ] **Steps 3–5:** Run, commit:

```bash
git commit -m "test(psm): reward-inference eval path matches FB protocol"
```

### Task 6.3: Pixel (DrQ) wiring + e2e smoke

**Files:**
- Test: `tests/test_psm_pixel_smoke.py`

- [ ] **Step 1:** Verify `PSMModel` builds with `rgb_encoder_cfg`/`augmentator_cfg` (DrQ) by mirroring `FBModel`'s pixel construction (encoders become the DrQ CNN; `obs_dim` = encoder output dim; `_augmentator` applied in `update` like `FBAgent.aug`). Add the `aug`/`enc` mirror to `PSMAgent.update` (copy `FBAgent.aug`/`enc` shape).
- [ ] **Step 2: Write a pixel smoke test** building `PSMFlowBCAgent` with a DrQ `rgb_encoder_cfg`/`augmentator_cfg` (mirror how FB pixel tests construct them — grep `tests/` for an FB pixel test or `build_pixel_cfgs`), pixel obs space, and assert one `update` step yields finite losses. If no FB pixel unit test exists, construct the cfgs directly from `configs/domain/visual_cube_single.yaml`.
- [ ] **Steps 3–5:** Implement the pixel path in `PSMModel`/`PSMAgent`, run, commit:

```bash
git commit -m "feat(psm): pixel DrQ front-end wiring + pixel update smoke"
```

### Task 6.4: Launcher scripts

**Files:**
- Create: `scripts/agents/psm/state.sh`, `scripts/agents/psm/pixel.sh`

- [ ] **Step 1:** Copy `scripts/agents/fb/state.sh` → `scripts/agents/psm/state.sh`, change the agent to `psm_flowbc` and the default `RUN_GROUP` to `psm_state`; expose `Z_DIM`, `MAX_LOG_SEED`, `NUM_PARALLEL` env knobs (append to `extra` like FB's `L_DIM`). For `pixel.sh`, mirror the FB/CRL pixel launcher (`grep -rn "visual_cube_single\|encoder=drq" scripts/`) selecting the visual domain + DrQ.
- [ ] **Step 2: Dry-run** all variants:

```bash
DRY_RUN=1 SEEDS='0' bash scripts/agents/psm/state.sh
DRY_RUN=1 SEEDS='0' bash scripts/agents/psm/pixel.sh
```
Expected: each prints a `train.py agent=psm_flowbc ... ` command (state: cube_single; pixel: visual_cube_single + drq), wandb_entity=amsks.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(psm): launcher scripts for state + pixel PSM+FlowBC"
```

---

## Phase 7 — Integration validation

### Task 7.1: 50-step e2e smoke (state + pixel)

**Files:**
- Test: `tests/test_psm_e2e.py`

- [ ] **Step 1: Write** a parametrized 50-step train smoke for `PSMFlowBCAgent` (state) and (pixel) using synthetic batches (mirror `tests/test_gc_fb_e2e.py` fixture shape, adding `index` and a `bool` `terminated` under `next`). Assert finite `sf_loss`/`psm_loss`/`actor_loss` after 50 steps.
- [ ] **Step 2: Run** → `.venv/bin/python -m pytest tests/test_psm_e2e.py -v` → PASS.
- [ ] **Step 3: Commit** → `git commit -m "test(psm): 50-step e2e smoke (state + pixel)"`

### Task 7.2: Full regression + 2k-step GPU sanity

- [ ] **Step 1:** `.venv/bin/python -m pytest tests/ -q --continue-on-collection-errors` — all new PSM tests PASS; only the known pre-existing pandas/script failures remain. Confirm no FB regressions (FB tests still pass; FB code untouched).
- [ ] **Step 2:** 2k-step GPU sanity per variant (real data; wandb offline for the smoke):

```bash
cd /home/adityamohan/git/Factored-FB
WANDB_MODE=offline GPUS=0 SEEDS=0 TRAIN_STEPS=2000 RUN_GROUP=psm_smoke_state bash scripts/agents/psm/state.sh
WANDB_MODE=offline GPUS=1 SEEDS=0 TRAIN_STEPS=2000 RUN_GROUP=psm_smoke_pixel bash scripts/agents/psm/pixel.sh
```
Expected: both train 2k steps, run an eval (reward-inference, logs `eval/success`), and save `final.pt` with no crash. Inspect the seed logs for finite losses + a completed eval.

- [ ] **Step 3:** Commit any fixes, then STOP for review before any 10-seed campaign (campaign + capacity-matched ablation are out of scope here).

---

## Self-Review notes
- **Spec coverage:** faithful PSM nets (Task 0.1) ✓; dual-ψ + binary/Gaussian z + proto sampler (Tasks 1.1, 2.1, 3.2) ✓; FlowBC state (Task 5.1) ✓; pixel DrQ (Task 6.3) ✓; reward-inference eval matching FB (Tasks 2.1, 6.2) ✓; configs/registration/launcher (Tasks 0.2, 5.2, 6.1, 6.4) ✓; **bit-exact reference-equivalence on TD3-actor PSM (Task 4.1)** ✓; capacity knobs exposed (Z_DIM in 6.4, configurable) ✓.
- **Net-reuse exception honored:** PSM nets are vendored verbatim (Task 0.1), NOT reusing FB's ForwardMap — required for bit-exactness (spec §9).
- **Type consistency:** `PSMModel(obs_space, action_dim, z_dim, max_log_seed, batch_size, ...)`, `sf_psi`/`psm_psi`/`phi`/`target_*` names, `reward_inference(next_obs, reward)`, metric keys (`psm_loss`/`sf_loss`/`actor_loss`/`q`) consistent across Tasks 2–7.
- **Deferred/verify items flagged inline:** the encoder/normalizer construction block (copy from FBModel), `_update_*` loss bodies (transcribe cited reference lines), buffer row-index field (locate loader), pixel cfg construction — each is an explicit instruction with a source to copy, not a silent TODO.
- **Open risk:** if reference modules won't import under our torch (torchrl/dmc), Task 4.1 Step 5 fallback (verbatim-transcription oracle) keeps the bit-exact assertions intact.
