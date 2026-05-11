# Faithful FB on OGBench — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the FB and FlowBC agents in `Factored-FB/` to a faithful port of `td_jepa/metamotivo/`'s FB on state-based OGBench, fixing eight correctness bugs and aligning defaults with td_jepa's OGBench launch config.

**Architecture:** Single repo, Hydra YAML configs, `train.py` entry point. The cleanup keeps the file layout and config surface but replaces the math inside the FB agent / model / FlowBC agent, restores the dropped encoders (`_left_encoder`, `_obs_normalizer`, `_fw_encoder`, `_bw_encoder`), fixes the data loader, fixes the eval relabel source, and replaces the FlowBC mechanism with td_jepa's flow-matching velocity field + distillation BC.

**Tech Stack:** Python ≥ 3.11, PyTorch, Hydra/OmegaConf, Pydantic (for `BaseConfig`), OGBench, gymnasium.

**Reference spec:** `docs/superpowers/specs/2026-05-11-faithful-fb-ogbench-design.md`

**Reference implementation:** `/Users/adityamohan/git/Austin/td_jepa/metamotivo/agents/fb/` (`agent.py`, `model.py`, `flow_bc/`).

**Conventions used throughout this plan:**
- No TDD. The spec deletes the existing test suite (Task 1). Smoke checks use inline `python -c` invocations on synthetic Box spaces.
- `REPO=/Users/adityamohan/git/Austin/Factored-FB`. All bash commands assume `cd $REPO` first or use absolute paths.
- Commit messages follow the existing project style (lowercase prefix, imperative): `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.

---

## Task 1: Delete test suite, drop Factored-FB extension scaffolding, clean unused experiment configs

**Files:**
- Delete: `tests/` (entire directory)
- Modify: `README.md` (remove "Extending to Factored-FB" section and "Testing" section)
- Modify: `agents/base.py:50-52` (update `infer_z` docstring — no longer LS solve)
- Delete: `Maktefile` (already shown as deleted in git status — finalize)

- [ ] **Step 1: Inspect what we are deleting**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
ls tests/
git status -s
```

Expected: lists `test_*.py` files; `Maktefile` shows as `D`.

- [ ] **Step 2: Delete the test suite**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git rm -rf tests/
```

- [ ] **Step 3: Finalize Maktefile deletion**

`Maktefile` is already deleted on disk but not staged.

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git rm -f Maktefile 2>/dev/null || git add -u Maktefile
```

- [ ] **Step 4: Strip "Extending to Factored-FB" and "Testing" sections from README**

Open `README.md`. Remove the two sections entirely:
- Lines starting at `## Testing` through (and including) the table that ends with the `test_envs.py` row.
- Lines starting at `## Extending to Factored-FB` through (and including) the line `No changes needed to ...`.

Also change line 5's mention of "The FB baseline on top of which the Factored-FB variant is built" to just describe an FB baseline: replace that paragraph with:

```markdown
This repo is a faithful port of the FB / FlowBC agents from [facebookresearch/td_jepa](https://github.com/facebookresearch/td_jepa/tree/main/metamotivo) to state-based OGBench. DMC support, the `exca` launcher, and all non-FB agents are dropped.
```

Also remove the row of the agent table claiming `fb_flowbc` uses `NoiseConditionedActor + BC loss` — replace the table body with:

```markdown
| Agent | Actor | Use case |
|---|---|---|
| `fb` | TruncatedNormal (DDPG-style mean) | Antmaze / locomotion |
| `fb_flowbc` | Flow-matching velocity field + distilled noise actor | Cube / puzzle / scene manipulation |
```

- [ ] **Step 5: Fix `infer_z` docstring in `agents/base.py`**

The current docstring says "via the backward map least-squares solve" — wrong after Task 4. Replace `agents/base.py:50-52`:

```python
    @abstractmethod
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        """Infer goal embedding z from (obs, reward) pairs via the linear
        estimator z = project(sum(r * B(obs)))."""
```

- [ ] **Step 6: Verify nothing imports from `tests/`**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
grep -RIn "from tests" --include="*.py" . || echo "OK no imports"
grep -RIn "import tests" --include="*.py" . || echo "OK no imports"
```

Expected: `OK no imports` for both.

- [ ] **Step 7: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add -A
git commit -m "chore: drop test suite, Factored-FB extension scaffolding, stale Maktefile

The test suite asserted behavior of the (broken) FB implementation that
this branch replaces. See spec
docs/superpowers/specs/2026-05-11-faithful-fb-ogbench-design.md."
```

---

## Task 2: Fix data loader — rename function and move physics under `next/`

**Files:**
- Modify: `data/ogbench.py`

**Why:** `train.py:28` imports `load_ogbench_dataset` but the file exports `load_ogbench_data` — code does not run. Eval needs `batch["next"]["physics"]` (per td_jepa) but the loader puts `physics` at the top level. Reward functions read the *next-state* qpos, so the correct slice is `phys[1:]`.

- [ ] **Step 1: Rename the function and move physics**

Replace `data/ogbench.py` entirely with:

```python
"""
data/ogbench.py — Offline dataset loader for OGBench.

Episode .npz files are expected at:
    <data_path>/<domain>/buffer/ep*.npz

Each file stores one episode with keys:
    observation  [T, obs_dim]    float32
    action       [T, act_dim]    float32
    physics      [T, phys_dim]   float32  (used for reward relabelling at eval)
    discount     [T]             float32  (0 at terminal transitions)

Episodes of T timesteps yield T-1 transitions:
    (obs[t], action[t]) -> (obs[t+1], physics[t+1], terminated)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from buffers.transition import DictBuffer


def load_transitions(npz_files: List[Union[str, Path]]) -> Dict:
    """Load episode .npz files into a flat transition dict.

    Returns a dict with keys:
        observation                [N, obs_dim]
        action                     [N, act_dim]
        next.observation           [N, obs_dim]
        next.terminated            [N, 1]    bool
        next.physics               [N, phys_dim]   if present
    """
    obs_list, act_list, next_obs_list, term_list, next_phys_list = [], [], [], [], []
    has_physics = True

    for f in npz_files:
        ep = np.load(f)
        obs = ep["observation"].astype(np.float32)
        act = ep["action"].astype(np.float32)
        disc = ep["discount"].astype(np.float32) if "discount" in ep else np.ones(len(obs))

        obs_list.append(obs[:-1])
        act_list.append(act[:-1])
        next_obs_list.append(obs[1:])
        term_list.append((disc[1:] == 0.0).reshape(-1, 1))

        if has_physics and "physics" in ep:
            next_phys_list.append(ep["physics"].astype(np.float32)[1:])
        else:
            has_physics = False

    storage: Dict = {
        "observation": np.concatenate(obs_list, axis=0),
        "action": np.concatenate(act_list, axis=0),
        "next": {
            "observation": np.concatenate(next_obs_list, axis=0),
            "terminated": np.concatenate(term_list, axis=0),
        },
    }
    if has_physics and next_phys_list:
        storage["next"]["physics"] = np.concatenate(next_phys_list, axis=0)

    return storage


def load_ogbench_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
) -> DictBuffer:
    """Load offline OGBench data into a DictBuffer.

    Parameters
    ----------
    domain : e.g. "antmaze-medium-navigate-v0"
    data_path : root dir; episodes are at <data_path>/<domain>/buffer/*.npz
    load_n_episodes : max number of episodes to load
    device : device for sampled tensors
    n_transitions : if set, truncate the buffer to this many transitions
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    storage = load_transitions(files)

    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]

    n = len(storage["observation"])
    buffer = DictBuffer(capacity=n, device=device)
    buffer.extend(storage)
    return buffer
```

- [ ] **Step 2: Smoke-verify import and synthetic loader**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import numpy as np, tempfile, pathlib
from data.ogbench import load_transitions, load_ogbench_dataset

# Fake 2 episodes of 5 steps each
with tempfile.TemporaryDirectory() as d:
    buf = pathlib.Path(d) / 'fake-domain' / 'buffer'
    buf.mkdir(parents=True)
    for i in range(2):
        np.savez(buf / f'ep{i}.npz',
                 observation=np.zeros((5, 3), dtype=np.float32),
                 action=np.zeros((5, 2), dtype=np.float32),
                 physics=np.zeros((5, 4), dtype=np.float32),
                 discount=np.array([1,1,1,1,0], dtype=np.float32))
    buffer = load_ogbench_dataset('fake-domain', data_path=d, device='cpu')
    print('size:', len(buffer))
    batch = buffer.sample(3)
    print('keys:', sorted(batch.keys()))
    print('next.keys:', sorted(batch['next'].keys()))
    print('next.physics shape:', batch['next']['physics'].shape)
    print('next.terminated shape:', batch['next']['terminated'].shape)
"
```

Expected:
```
size: 8
keys: ['action', 'next', 'observation']
next.keys: ['observation', 'physics', 'terminated']
next.physics shape: torch.Size([3, 4])
next.terminated shape: torch.Size([3, 1])
```

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add data/ogbench.py
git commit -m "fix: rename load_ogbench_dataset, place physics under next/

train.py imports load_ogbench_dataset; the file previously exported
load_ogbench_data, so the entry point did not import. Also move physics
to batch['next']['physics'] (phys[1:] per episode) so eval relabels on
the next-state qpos, matching td_jepa's OGBench eval."
```

---

## Task 3: Add `IdentityNNConfig` in `nn_models.py`

**Files:**
- Modify: `nn_models.py` (append after the `VectorField` class, before the file ends)

**Why:** `FBModel` will wire `_obs_normalizer`, `_fw_encoder`, `_bw_encoder` slots that are Identity for state-based OGBench but must expose `output_space` so downstream modules can size their first linear layer. The existing `IdentityNormalizerConfig` returns a plain `nn.Identity`, which has no `output_space` — we need a config + module that does.

- [ ] **Step 1: Append `IdentityNNConfig` and `IdentityNN` to `nn_models.py`**

At the end of `nn_models.py`, append:

```python
##########################
# Identity passthrough (for encoder slots when running on state observations)
##########################


class IdentityNN(nn.Module):
    """Passthrough module that exposes an output_space attribute.

    Used as a default for _obs_normalizer/_fw_encoder/_bw_encoder/_left_encoder
    when no real encoder is needed. obs_space is passed through unchanged.
    """

    def __init__(self, obs_space) -> None:
        super().__init__()
        self._output_space = obs_space

    @property
    def output_space(self):
        return self._output_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityNNConfig(BaseConfig):
    name: tp.Literal["IdentityNN"] = "IdentityNN"

    def build(self, obs_space, *args, **kwargs) -> IdentityNN:
        return IdentityNN(obs_space)
```

- [ ] **Step 2: Also give `BackwardArchiConfig.build` an `output_space` property on the module it builds**

`BackwardMap` already produces `[B, z_dim]` outputs but does not expose `output_space`. We need this when `_left_encoder` is a `BackwardArchi` MLP. Edit `BackwardMap.__init__` in `nn_models.py:170-186`:

Find:
```python
class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
```

Replace with:
```python
class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        from gymnasium.spaces import Box
        import numpy as _np
        self._output_space = Box(
            low=-_np.inf, high=_np.inf, shape=(z_dim,), dtype=_np.float32
        )
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
```

And just before the `def forward` at line 185, add the property:

```python
    @property
    def output_space(self):
        return self._output_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 3: Smoke-verify imports**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import numpy as np
from gymnasium.spaces import Box
from nn_models import IdentityNN, IdentityNNConfig, BackwardArchiConfig
obs = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
nn_id = IdentityNNConfig().build(obs)
print('Identity output_space:', nn_id.output_space.shape)
b = BackwardArchiConfig(hidden_dim=64, hidden_layers=2, norm=True).build(obs, z_dim=50)
print('Backward output_space:', b.output_space.shape)
import torch
x = torch.randn(8, 10)
y = b(x)
print('B(x) shape:', tuple(y.shape), 'norm:', y.norm(dim=-1).mean().item())
"
```

Expected:
```
Identity output_space: (10,)
Backward output_space: (50,)
B(x) shape: (8, 50) norm: 7.07...
```
(7.07 ≈ √50, since `norm=True` rescales to ||·||=√z_dim.)

- [ ] **Step 4: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add nn_models.py
git commit -m "feat: add IdentityNN passthrough and BackwardMap.output_space

Encoders and normalizer slots on FBModel need an output_space attribute
so downstream modules can size their input layer. IdentityNN passes the
obs_space through; BackwardMap (used as _left_encoder) advertises a
Box(z_dim,) output."
```

---

## Task 4: Rewrite `FBModel` — wire encoders/normalizer, restore mixed_z and reward_inference

**Files:**
- Modify: `agents/fb/model.py` (full rewrite)

**Why:** Current `FBModel` skips `_obs_normalizer`, `_fw_encoder`, `_bw_encoder`, `_left_encoder`, and uses a regularized-LS `infer_z` with a zero-reward filter. Replace with a faithful port of td_jepa's `FBModel`.

- [ ] **Step 1: Rewrite `agents/fb/model.py`**

Replace the entire file with:

```python
"""
agents/fb/model.py — FBModel.

Network call chain (all encoders / normalizer are Identity for state OGBench):
    obs -> _obs_normalizer -> _fw_encoder -> _left_encoder -> _forward_map
                          \-> _bw_encoder ----> _backward_map
                          \-> (left_enc if actor_encode_obs else fw_enc) -> _actor

Target nets: _target_forward_map, _target_backward_map, _target_left_encoder.
All soft-updated.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    IdentityNN,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    _soft_update_params,
    weight_init,
)
from normalizers import IdentityNormalizerConfig


class FBModel(nn.Module):
    """Container for FB networks, encoders, normalizer, and target copies."""

    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        L_dim: int = 50,
        norm_z: bool = True,
        actor_encode_obs: bool = False,
        forward_cfg: Optional[ForwardArchiConfig] = None,
        backward_cfg: Optional[BackwardArchiConfig] = None,
        left_encoder_cfg: Optional[BackwardArchiConfig] = None,
        actor_cfg=None,
        obs_normalizer_cfg=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.L_dim = L_dim
        self.norm_z = norm_z
        self.actor_encode_obs = actor_encode_obs
        self.device = device

        # Defaults match td_jepa's OGBench launch
        if forward_cfg is None:
            forward_cfg = ForwardArchiConfig(hidden_dim=512, hidden_layers=2, num_parallel=2)
        if backward_cfg is None:
            backward_cfg = BackwardArchiConfig(hidden_dim=512, hidden_layers=4, norm=True)
        if left_encoder_cfg is None:
            left_encoder_cfg = BackwardArchiConfig(hidden_dim=512, hidden_layers=4, norm=True)
        if actor_cfg is None:
            actor_cfg = SimpleActorArchiConfig(hidden_dim=512, hidden_layers=2)
        if obs_normalizer_cfg is None:
            obs_normalizer_cfg = IdentityNormalizerConfig()

        # Encoders / normalizer (Identity passthroughs for state OGBench)
        self._obs_normalizer = obs_normalizer_cfg.build(obs_space)
        self._fw_encoder = IdentityNN(obs_space)
        self._bw_encoder = IdentityNN(obs_space)

        # Left encoder produces L_dim-shaped output from fw_encoder.output_space
        self._left_encoder = left_encoder_cfg.build(self._fw_encoder.output_space, L_dim)

        # FB maps
        self._backward_map = backward_cfg.build(self._bw_encoder.output_space, z_dim)
        self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim)

        # Actor input space depends on actor_encode_obs
        actor_in_space = (
            self._left_encoder.output_space if actor_encode_obs else self._fw_encoder.output_space
        )
        self._actor = actor_cfg.build(actor_in_space, z_dim, action_dim)

        # Init weights
        self.apply(weight_init)

        # Target nets
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)
        for p in self._target_forward_map.parameters():
            p.requires_grad_(False)
        for p in self._target_backward_map.parameters():
            p.requires_grad_(False)
        for p in self._target_left_encoder.parameters():
            p.requires_grad_(False)

        self.to(device)

    # ── Target updates ────────────────────────────────────────────────── #

    def soft_update_targets(self, f_tau: float, b_tau: float) -> None:
        _soft_update_params(
            list(self._forward_map.parameters()),
            list(self._target_forward_map.parameters()),
            f_tau,
        )
        _soft_update_params(
            list(self._backward_map.parameters()),
            list(self._target_backward_map.parameters()),
            b_tau,
        )
        _soft_update_params(
            list(self._left_encoder.parameters()),
            list(self._target_left_encoder.parameters()),
            f_tau,
        )

    # ── z utilities ──────────────────────────────────────────────────── #

    def project_z(self, z: Tensor) -> Tensor:
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def sample_z(self, n: int) -> Tensor:
        z = torch.randn(n, self.z_dim, device=self.device)
        return self.project_z(z)

    def mixed_z(self, next_obs_normed_encoded_bw: Tensor, train_goal_ratio: float) -> Tensor:
        """Sample z: with prob train_goal_ratio use B(next_obs[perm]), else random.

        `next_obs_normed_encoded_bw` is the output of `_bw_encoder(_normalize(next_obs))`,
        which the agent has already computed for the FB loss.
        """
        B = next_obs_normed_encoded_bw.shape[0]
        z = self.sample_z(B)
        perm = torch.randperm(B, device=self.device)
        with torch.no_grad():
            goals = self.project_z(
                self._backward_map(next_obs_normed_encoded_bw[perm])
            )
        mask = (torch.rand((B, 1), device=self.device) < train_goal_ratio)
        return torch.where(mask, goals, z)

    # ── Public API used at eval ───────────────────────────────────────── #

    def normalize(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self._obs_normalizer.eval()
            return self._obs_normalizer(x)

    @torch.no_grad()
    def backward_map(self, obs: Tensor) -> Tensor:
        """Public B(obs) including normalize + bw_encoder."""
        return self._backward_map(self._bw_encoder(self.normalize(obs)))

    @torch.no_grad()
    def reward_inference(self, next_obs: Tensor, reward: Tensor) -> Tensor:
        """Linear estimator: z = project(sum_i r_i * B(next_obs_i)).

        reward : [N, 1]
        next_obs : [N, obs_dim]
        Returns z of shape [z_dim].
        """
        B_mat = self.backward_map(next_obs)              # [N, z_dim]
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)                # [N, 1]
        z = (reward.T @ B_mat).squeeze(0)                # [z_dim]
        return self.project_z(z.unsqueeze(0)).squeeze(0)

    # Backwards-compatible alias used by FBAgent.infer_z
    infer_z = reward_inference
```

- [ ] **Step 2: Smoke-verify model construction and shapes**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from agents.fb.model import FBModel
obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
m = FBModel(obs_space=obs_space, action_dim=4, z_dim=50, L_dim=50, device='cpu')
obs = torch.randn(8, 15)
action = torch.randn(8, 4)
print('forward_map(left, z, a):', tuple(m._forward_map(m._left_encoder(obs), m.sample_z(8), action).shape))
print('backward_map(obs):', tuple(m.backward_map(obs).shape))
z_est = m.reward_inference(obs, torch.randn(8))
print('reward_inference shape:', tuple(z_est.shape), 'norm:', z_est.norm().item())
goal = m._bw_encoder(m.normalize(obs))
print('mixed_z shape:', tuple(m.mixed_z(goal, 0.5).shape))
"
```

Expected:
```
forward_map(left, z, a): (2, 8, 50)
backward_map(obs): (8, 50)
reward_inference shape: (50,) norm: 7.07...
mixed_z shape: (8, 50)
```

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add agents/fb/model.py
git commit -m "refactor: faithful FBModel — encoders, normalizer, linear reward inference

Wires _obs_normalizer / _fw_encoder / _bw_encoder / _left_encoder
(plus target_left_encoder) matching td_jepa. Replaces the regularised
least-squares z inference with the linear estimator
z = project(sum r_i * B(next_obs_i)). Restores the shuffled/masked
mixed_z sampler. actor_encode_obs switch defaults to False per
td_jepa's OGBench launch."
```

---

## Task 5: Rewrite `FBAgent` — M-matrix TD, contrastive ortho, F·z actor, pairwise-L1 pessimism

**Files:**
- Modify: `agents/fb/agent.py` (full rewrite)

**Why:** All four current loss formulations are wrong (see spec §3, bugs 1, 2, 3, 8). Replace with td_jepa's formulations from `td_jepa/metamotivo/agents/fb/agent.py`.

- [ ] **Step 1: Rewrite `agents/fb/agent.py`**

Replace the entire file with:

```python
"""
agents/fb/agent.py — FBAgent for state-based OGBench.

Implements the M-matrix TD loss with contrastive diagonal, contrastive
orthonormality on B, Q = F·z actor, pairwise-L1-distance pessimism, and
three optimizer groups (forward + left_encoder | backward | actor).

batch keys (from DictBuffer.sample):
    batch["observation"]            [B, obs_dim]
    batch["action"]                 [B, act_dim]
    batch["next"]["observation"]    [B, obs_dim]
    batch["next"]["terminated"]     [B, 1]   bool / float
    batch["next"]["physics"]        [B, phys_dim]    (eval only)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from agents.base import BaseAgent
from agents.fb.model import FBModel


def pairwise_l1_pessimism(
    preds: Tensor, penalty: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (mean, uncertainty, mean - penalty * uncertainty).

    preds : [E, ...]  E = num ensemble heads
    uncertainty = sum_{i,j} |p_i - p_j| / (E^2 - E)
    """
    mean = preds.mean(dim=0)
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()   # [E, E, ...]
    scaling = preds.shape[0] ** 2 - preds.shape[0]
    unc = diffs.sum(dim=(0, 1)) / max(scaling, 1)
    return mean, unc, mean - penalty * unc


class FBAgent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        batch_size: int = 256,
        z_dim: int = 50,
        L_dim: int = 50,
        actor_encode_obs: bool = False,
        forward_cfg=None,
        backward_cfg=None,
        left_encoder_cfg=None,
        actor_cfg=None,
        obs_normalizer_cfg=None,
        discount: float = 0.99,
        lr_f: float = 1e-4,
        lr_b: float = 1e-4,
        lr_actor: float = 1e-4,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 0.0,
        ortho_coef: float = 1.0,
        train_goal_ratio: float = 0.5,
        fb_pessimism_penalty: float = 0.0,
        actor_pessimism_penalty: float = 0.0,
        actor_std: float = 0.2,
        stddev_clip: float = 0.3,
        f_target_tau: float = 0.005,
        b_target_tau: float = 0.005,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.discount_scalar = discount
        self.lr_f = lr_f
        self.lr_b = lr_b
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.ortho_coef = ortho_coef
        self.train_goal_ratio = train_goal_ratio
        self.fb_pessimism_penalty = fb_pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.f_target_tau = f_target_tau
        self.b_target_tau = b_target_tau
        self.device = device
        self.actor_encode_obs = actor_encode_obs

        self.model = FBModel(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            L_dim=L_dim,
            actor_encode_obs=actor_encode_obs,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            left_encoder_cfg=left_encoder_cfg,
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=obs_normalizer_cfg,
            device=device,
        )

        self.forward_optimizer = Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters()),
            lr=lr_f,
            weight_decay=weight_decay,
        )
        self.backward_optimizer = Adam(
            self.model._backward_map.parameters(),
            lr=lr_b,
            weight_decay=weight_decay,
        )
        self.actor_optimizer = Adam(
            self.model._actor.parameters(),
            lr=lr_actor,
            weight_decay=weight_decay,
        )

        # Precompute off-diagonal mask of size [batch_size, batch_size].
        # Couples agent to a fixed batch size (same constraint as td_jepa).
        eye = torch.eye(batch_size, batch_size, device=device)
        self.off_diag = (1.0 - eye)
        self.off_diag_sum = self.off_diag.sum()

    # ── Main update ───────────────────────────────────────────────────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device).float()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)  # [B, 1]
        discount = self.discount_scalar * (1.0 - terminated)  # [B, 1]

        # Normalize + fw/bw encode (Identity for state OGBench but keeps the
        # call chain matching td_jepa)
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        next_obs_n = self.model._fw_encoder(self.model.normalize(next_obs))
        goal = self.model._bw_encoder(self.model.normalize(next_obs))

        # Sample mixed z (no grad)
        with torch.no_grad():
            z = self.model.mixed_z(goal, self.train_goal_ratio)

        fb_metrics = self._update_fb(obs_n, next_obs_n, action, goal, discount, z)
        actor_metrics = self._update_actor(obs_n.detach(), z)
        self.model.soft_update_targets(self.f_target_tau, self.b_target_tau)

        return {**fb_metrics, **actor_metrics}

    # ── FB loss (M-matrix TD + contrastive ortho) ─────────────────────── #

    def _update_fb(
        self,
        obs: Tensor,
        next_obs: Tensor,
        action: Tensor,
        goal: Tensor,
        discount: Tensor,
        z: Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_left = self.model._target_left_encoder(next_obs)
            actor_in = next_left if self.actor_encode_obs else next_obs
            next_dist = self.model._actor(actor_in, z, self.actor_std)
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_Fs = self.model._target_forward_map(next_left, z, next_action)  # [E, B, z_dim]
            target_B = self.model._target_backward_map(goal)                       # [B, z_dim]
            target_Ms = torch.matmul(target_Fs, target_B.T)                        # [E, B, B]
            _, _, target_M = pairwise_l1_pessimism(target_Ms, self.fb_pessimism_penalty)  # [B, B]

        left = self.model._left_encoder(obs)
        Fs = self.model._forward_map(left, z, action)                              # [E, B, z_dim]
        B_out = self.model._backward_map(goal)                                     # [B, z_dim]
        Ms = torch.matmul(Fs, B_out.T)                                             # [E, B, B]

        diff = Ms - discount * target_M                                            # [E, B, B]
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # Contrastive orthonormality on B (batch x batch)
        Cov = torch.matmul(B_out, B_out.T)
        orth_diag = -Cov.diag().mean()
        orth_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_offdiag + orth_diag
        fb_loss = fb_loss + self.ortho_coef * orth_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), self.clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        return {
            "loss/fb": fb_loss.item(),
            "loss/fb_diag": fb_diag.item(),
            "loss/fb_offdiag": fb_offdiag.item(),
            "loss/orth": orth_loss.item(),
            "loss/orth_diag": orth_diag.item(),
            "loss/orth_offdiag": orth_offdiag.item(),
            "train/M1": Ms[0].mean().item(),
            "train/B_norm": B_out.norm(dim=-1).mean().item(),
            "train/z_norm": z.norm(dim=-1).mean().item(),
        }

    # ── Actor loss (Q = F·z, pairwise-L1 pessimism, sampled action) ───── #

    def _update_actor(self, obs: Tensor, z: Tensor) -> Dict[str, float]:
        with torch.no_grad():
            left = self.model._left_encoder(obs)
        actor_in = left if self.actor_encode_obs else obs
        dist = self.model._actor(actor_in, z, self.actor_std)
        actor_action = dist.sample(clip=self.stddev_clip)
        Fs = self.model._forward_map(left, z, actor_action)
        Qs = (Fs * z).sum(-1)                                                      # [E, B]
        _, _, Q = pairwise_l1_pessimism(Qs, self.actor_pessimism_penalty)
        actor_loss = -Q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "loss/actor": actor_loss.item(),
            "train/Q_actor": Q.mean().item(),
        }

    # ── Eval API ──────────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        actor_in = self.model._left_encoder(obs_n) if self.actor_encode_obs else obs_n
        dist = self.model._actor(actor_in, z, std=0.0 if eval_mode else self.actor_std)
        # `mean` is the tanh'd policy output (deterministic)
        return dist.mean if hasattr(dist, "mean") else dist.loc

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.reward_inference(obs.to(self.device), reward.to(self.device))

    # ── Checkpointing ─────────────────────────────────────────────────── #

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.forward_optimizer.load_state_dict(state["forward_optimizer"])
        self.backward_optimizer.load_state_dict(state["backward_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
```

- [ ] **Step 2: Smoke-verify five update steps with synthetic data**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from agents.fb.agent import FBAgent

torch.manual_seed(0)
np.random.seed(0)
B = 64  # smaller batch for smoke
obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
agent = FBAgent(
    obs_space=obs_space, action_dim=4, batch_size=B, z_dim=32, L_dim=32,
    device='cpu',
)
def make_batch():
    return {
        'observation': torch.randn(B, 15),
        'action': torch.randn(B, 4).clamp(-1, 1),
        'next': {
            'observation': torch.randn(B, 15),
            'terminated': torch.zeros(B, 1),
        },
    }
last = None
for step in range(5):
    m = agent.update(make_batch(), step)
    last = m
print('final metrics keys:', sorted(last.keys()))
print('loss/fb finite:', not (last['loss/fb'] != last['loss/fb']))
print('loss/actor finite:', not (last['loss/actor'] != last['loss/actor']))
"
```

Expected: prints metrics keys and `loss/fb finite: True`, `loss/actor finite: True`.

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add agents/fb/agent.py
git commit -m "refactor: faithful FBAgent — M-matrix TD, F·z actor, pairwise-L1 LCB

Replaces the diagonal-only Q-TD loss with the full M-matrix TD plus
contrastive negative-diagonal term used by td_jepa, replaces the
B^T B -> I ortho with the contrastive B B^T form, switches the actor
target from Q = F·B(obs) to Q = F·z, switches pessimism from
ensemble-std to pairwise-L1 LCB, and splits the optimizer into three
groups (forward + left_encoder | backward | actor) matching td_jepa."
```

---

## Task 6: Configs — `train.yaml`, `configs/agent/fb.yaml`, antmaze domain configs

**Files:**
- Modify: `configs/train.yaml`
- Create: `configs/agent/fb.yaml`
- Create: `configs/domain/antmaze_medium.yaml`
- Create: `configs/domain/antmaze_large.yaml`
- Create: `configs/domain/antmaze_giant.yaml`

**Why:** `configs/train.yaml` references `defaults: [domain: antmaze_medium, agent: fb]` but those files do not exist — Hydra composition currently fails. Create them with td_jepa OGBench-launch defaults.

- [ ] **Step 1: Replace `configs/train.yaml`**

```yaml
defaults:
  - _self_
  - domain: antmaze_medium      # domain config selects its own agent via its defaults list
  - override hydra/launcher: basic

# Domain (overridden by domain/*.yaml)
domain: antmaze-medium-navigate-v0
obs_type: state
data_path: datasets
n_transitions: null         # null = load full dataset
load_n_episodes: 1000

# Training
num_train_steps: 1_000_000
batch_size: 256
eval_every: 50_000
log_every: 1_000
save_every: 200_000
seed: 42
device: cuda

# Eval
eval_n_episodes: 10
eval_relabel_size: 10_000
eval_shift_reward: 1.0

# Logging
use_wandb: false
wandb_project: factored-fb
wandb_run_name: null
save_dir: checkpoints/

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}__${domain}__${agent}__s${seed}
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}__s${seed}
```

- [ ] **Step 2: Create `configs/agent/fb.yaml`**

```yaml
# @package _global_
agent: fb
# core hyperparameters (override on CLI with `lr_b=1e-5` etc.)
z_dim: 50
L_dim: 50
actor_encode_obs: false
discount: 0.99
lr_f: 1.0e-4
lr_b: 1.0e-4
lr_actor: 1.0e-4
weight_decay: 0.0
clip_grad_norm: 0.0
ortho_coef: 1.0
train_goal_ratio: 0.5
fb_pessimism_penalty: 0.0
actor_pessimism_penalty: 0.0
actor_std: 0.2
stddev_clip: 0.3
f_target_tau: 0.005
b_target_tau: 0.005

# Per-network architecture (td_jepa OGBench launch defaults)
forward:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
backward:
  hidden_dim: 512
  hidden_layers: 4
  norm: true
left_encoder:
  hidden_dim: 512
  hidden_layers: 4
  norm: true
actor:
  name: simple
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
```

- [ ] **Step 3: Create `configs/domain/antmaze_medium.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb
  - _self_

domain: antmaze-medium-navigate-v0
agent: fb
# Per-domain sweepables (override on CLI):
ortho_coef: 1.0       # td_jepa sweep used 100 and 1000
lr_b: 1.0e-4          # td_jepa sweep used 1e-4 and 1e-5
```

- [ ] **Step 4: Create `configs/domain/antmaze_large.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb
  - _self_

domain: antmaze-large-navigate-v0
agent: fb
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 5: Create `configs/domain/antmaze_giant.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb
  - _self_

domain: antmaze-giant-navigate-v0
agent: fb
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 6: Smoke-verify config composition**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import hydra
from hydra import compose, initialize
with initialize(config_path='configs', version_base='1.3'):
    cfg = compose(config_name='train', overrides=['domain=antmaze_medium'])
    print('domain:', cfg.domain)
    print('agent:', cfg.agent)
    print('hidden:', cfg.forward.hidden_dim)
    print('z_dim:', cfg.z_dim, 'L_dim:', cfg.L_dim)
    print('actor_std:', cfg.actor_std)
"
```

Expected:
```
domain: antmaze-medium-navigate-v0
agent: fb
hidden: 512
z_dim: 50 L_dim: 50
actor_std: 0.2
```

- [ ] **Step 7: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add configs/
git commit -m "feat: configs — fb agent + antmaze domains, td_jepa OGBench defaults

Adds the missing configs/agent/fb.yaml and configs/domain/antmaze_*.yaml
files referenced from train.yaml's Hydra defaults. Values match
td_jepa's scripts/train/proprio/launch_fb_ogbench.py BASE_CFG."
```

---

## Task 7: Wire `train.py` to the new FBAgent, smoke-run on antmaze

**Files:**
- Modify: `train.py`

**Why:** Current `train.py` builds `ForwardArchiConfig(hidden_dim=cfg.hidden_dim)` (a field that no longer exists) and instantiates `FBAgent` with the old kwarg names (`forward_cfg`, `backward_cfg`, `pessimism`). Wire it to read the new config layout.

- [ ] **Step 1: Replace `train.py`**

```python
"""train.py — Faithful FB on state-based OGBench, Hydra-configured.

Examples
--------
    python train.py domain=antmaze_medium
    python train.py domain=antmaze_medium agent=fb_flowbc
    python train.py --multirun seed=1,2,3 domain=antmaze_medium
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.agent import FBFlowBCAgent
from data.ogbench import load_ogbench_dataset
from envs.ogbench import create_ogbench_env
from evals.ogbench import OGBenchEvaluator
from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import IdentityNormalizerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU")
        return "cpu"
    return requested


def make_agent(cfg: DictConfig, obs_space, action_dim: int):
    forward_cfg = ForwardArchiConfig(
        hidden_dim=cfg.forward.hidden_dim,
        hidden_layers=cfg.forward.hidden_layers,
        embedding_layers=cfg.forward.embedding_layers,
        num_parallel=cfg.forward.num_parallel,
    )
    backward_cfg = BackwardArchiConfig(
        hidden_dim=cfg.backward.hidden_dim,
        hidden_layers=cfg.backward.hidden_layers,
        norm=cfg.backward.norm,
    )
    left_encoder_cfg = BackwardArchiConfig(
        hidden_dim=cfg.left_encoder.hidden_dim,
        hidden_layers=cfg.left_encoder.hidden_layers,
        norm=cfg.left_encoder.norm,
    )

    shared = dict(
        obs_space=obs_space,
        action_dim=action_dim,
        batch_size=cfg.batch_size,
        z_dim=cfg.z_dim,
        L_dim=cfg.L_dim,
        actor_encode_obs=cfg.actor_encode_obs,
        forward_cfg=forward_cfg,
        backward_cfg=backward_cfg,
        left_encoder_cfg=left_encoder_cfg,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        discount=cfg.discount,
        lr_f=cfg.lr_f,
        lr_b=cfg.lr_b,
        lr_actor=cfg.lr_actor,
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
        ortho_coef=cfg.ortho_coef,
        train_goal_ratio=cfg.train_goal_ratio,
        fb_pessimism_penalty=cfg.fb_pessimism_penalty,
        actor_pessimism_penalty=cfg.actor_pessimism_penalty,
        actor_std=cfg.actor_std,
        stddev_clip=cfg.stddev_clip,
        f_target_tau=cfg.f_target_tau,
        b_target_tau=cfg.b_target_tau,
        device=cfg.device,
    )

    if cfg.agent == "fb_flowbc":
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return FBFlowBCAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            bc_coeff=cfg.bc_coeff,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            **shared,
        )

    actor_cfg = SimpleActorArchiConfig(
        hidden_dim=cfg.actor.hidden_dim,
        hidden_layers=cfg.actor.hidden_layers,
        embedding_layers=cfg.actor.embedding_layers,
    )
    return FBAgent(actor_cfg=actor_cfg, **shared)


def log_metrics(metrics: Dict[str, float], step: int, *, use_wandb: bool, prefix: str = "") -> None:
    tagged = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
    if use_wandb:
        import wandb
        wandb.log(tagged, step=step)
    kv = "  ".join(f"{k}={v:.4f}" for k, v in tagged.items())
    print(f"[step {step:>8d}]  {kv}")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    if cfg.use_wandb:
        import wandb
        run_name = cfg.wandb_run_name or f"{cfg.domain}__{cfg.agent}__s{cfg.seed}"
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.getcwd(),
        )

    print(f"[train] env: {cfg.domain}")
    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    print(f"[train] obs_space={obs_space.shape}  action_dim={action_dim}")

    print(f"[train] loading dataset for {cfg.domain}")
    buffer = load_ogbench_dataset(
        domain=cfg.domain,
        data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes,
        device=cfg.device,
        n_transitions=cfg.n_transitions,
    )
    print(f"[train] buffer size: {len(buffer):,}")

    agent = make_agent(cfg, obs_space, action_dim)
    print(f"[train] agent: {type(agent).__name__}")

    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        shift_reward=cfg.eval_shift_reward,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        device=cfg.device,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] {cfg.num_train_steps:,} steps")
    t0 = time.time()
    accum: Dict[str, float] = {}
    accum_n = 0

    for step in range(1, cfg.num_train_steps + 1):
        batch = buffer.sample(cfg.batch_size)
        metrics = agent.update(batch, step)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        accum_n += 1

        if step % cfg.log_every == 0:
            avg = {k: v / accum_n for k, v in accum.items()}
            avg["misc/steps_per_sec"] = cfg.log_every / (time.time() - t0)
            log_metrics(avg, step, use_wandb=cfg.use_wandb, prefix="train/")
            accum, accum_n = {}, 0
            t0 = time.time()

        if step % cfg.eval_every == 0:
            print(f"[eval] step={step}")
            log_metrics(evaluator.run(), step, use_wandb=cfg.use_wandb, prefix="eval/")

        if step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}.pt"
            torch.save(agent.state_dict(), ckpt)
            print(f"[ckpt] {ckpt}")

    log_metrics(evaluator.run(), cfg.num_train_steps, use_wandb=cfg.use_wandb, prefix="eval/")
    torch.save(agent.state_dict(), save_dir / "final.pt")
    print("[train] Done.")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-verify `make_agent` builds without needing OGBench data**

This avoids requiring the dataset for the smoke check.

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import hydra, numpy as np
from gymnasium.spaces import Box
from hydra import compose, initialize
from omegaconf import OmegaConf
from train import make_agent
with initialize(config_path='configs', version_base='1.3'):
    cfg = compose(config_name='train',
                  overrides=['domain=antmaze_medium', 'device=cpu', 'batch_size=32'])
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    import torch
    B = 32
    batch = {
        'observation': torch.randn(B, 15),
        'action': torch.randn(B, 4).clamp(-1, 1),
        'next': {'observation': torch.randn(B, 15),
                 'terminated': torch.zeros(B, 1)},
    }
    m = agent.update(batch, 0)
    print('built agent:', type(agent).__name__)
    print('loss/fb:', round(m['loss/fb'], 3))
    print('loss/actor:', round(m['loss/actor'], 3))
"
```

Expected:
```
built agent: FBAgent
loss/fb: <some finite float>
loss/actor: <some finite float>
```

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add train.py
git commit -m "refactor: train.py — wire new FBAgent kwargs and config layout

Reads forward / backward / left_encoder / actor sub-configs out of the
composed Hydra config; threads through actor_encode_obs, weight_decay,
clip_grad_norm, pessimism penalties, and batch_size to FBAgent. Adds
the fb_flowbc branch using SimpleVectorFieldArchiConfig (Task 9–10)."
```

---

## Task 8: Rewrite OGBench eval — physics relabel, shift_reward, linear estimator

**Files:**
- Modify: `evals/ogbench.py`

**Why:** Current eval (`evals/ogbench.py`) calls `relabel_fn(obs_np, act_np)` (uses observation not physics — wrong for cube/scene/puzzle), filters `|r|>1e-6` (destroys sparse reward signal), uses regularized LS (broken). Replace with td_jepa's procedure.

- [ ] **Step 1: Replace `evals/ogbench.py`**

```python
"""evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Sample a batch from the offline buffer.
  2. Relabel with `get_relabel_fn(domain, task)(next.physics, action)`.
  3. Shift rewards by `shift_reward` (default 1.0).
  4. Infer z = project(sum r_i * B(next_obs_i)).
  5. Roll out `n_episodes` episodes via envs/rollout.py.
  6. Report mean success and mean return per task.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
from envs.rollout import rollout


class OGBenchEvaluator:
    def __init__(
        self,
        domain: str,
        agent,
        offline_buffer,
        relabel_size: int = 10_000,
        n_episodes: int = 10,
        shift_reward: float = 1.0,
        obs_type: str = "state",
        seed: int = 0,
        device: str = "cpu",
    ):
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.device = device

        tasks = ALL_TASKS.get(domain, [])
        self.task_envs: Dict[str, object] = {}
        self.relabel_fns: Dict[str, object] = {}
        for task in tasks:
            env, _ = create_ogbench_env(task, seed=seed, obs_type=obs_type)
            self.task_envs[task] = env
            self.relabel_fns[task] = get_relabel_fn(domain, task)

    def run(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for task, env in self.task_envs.items():
            z = self._infer_z(task)
            stats, infos = rollout(env, self.agent, self.n_episodes, ctx=z)
            success = float(np.mean([
                any(step.get("success", False) for step in ep_info)
                for ep_info in infos
            ]))
            metrics[f"{task}/success"] = success
            metrics[f"{task}/reward"] = float(np.mean(stats["reward"]))
        return metrics

    def _infer_z(self, task: str) -> Tensor:
        relabel_fn = self.relabel_fns[task]
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return self.agent.model.sample_z(1).squeeze(0)

        batch = self.offline_buffer.sample(n)

        if "physics" not in batch["next"]:
            raise KeyError(
                "Buffer batch is missing batch['next']['physics']. "
                "Re-load the dataset with data.ogbench.load_ogbench_dataset "
                "(updated to place physics under next/)."
            )

        phys_np = batch["next"]["physics"].cpu().numpy()
        act_np = batch["action"].cpu().numpy()

        rewards_np = relabel_fn(phys_np, act_np)               # [N, 1]
        rewards_np = rewards_np + self.shift_reward
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)

        return self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device),
            reward=rewards,
        )
```

- [ ] **Step 2: Smoke-verify on a synthetic buffer (skip if OGBench not installed)**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from buffers.transition import DictBuffer
from agents.fb.model import FBModel

# Build a tiny model + synthetic buffer with the next/physics key.
obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
m = FBModel(obs_space=obs_space, action_dim=4, z_dim=32, L_dim=32, device='cpu')
buf = DictBuffer(capacity=200, device='cpu')
buf.extend({
    'observation': np.zeros((200, 15), dtype=np.float32),
    'action': np.zeros((200, 4), dtype=np.float32),
    'next': {
        'observation': np.random.randn(200, 15).astype(np.float32),
        'terminated': np.zeros((200, 1), dtype=bool),
        'physics': np.random.randn(200, 8).astype(np.float32),
    },
})
batch = buf.sample(50)
assert 'physics' in batch['next']
# Fake reward fn: constant zero
import numpy as _np
rewards = _np.zeros((50, 1), dtype=_np.float32) + 1.0  # after shift_reward=1
z = m.reward_inference(batch['next']['observation'].float(),
                       torch.as_tensor(rewards, dtype=torch.float32))
print('z shape:', tuple(z.shape), 'norm:', z.norm().item())
"
```

Expected: `z shape: (32,) norm: 5.65...` (≈ √32).

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add evals/ogbench.py
git commit -m "fix: OGBench eval — relabel on next.physics, shift rewards, linear z

Reads next.physics (full qpos) for the relabel — observation alone gives
wrong rewards for cube/scene/puzzle. Drops the |r|>1e-6 filter, applies
shift_reward (default 1.0) before inference, and calls the agent
model's linear reward_inference (no LS solve). Default relabel_size
bumped to 10_000."
```

---

## Task 9: FBFlowBCModel — adds `_actor_vf` velocity field

**Files:**
- Create: `agents/fb/flow_bc/model.py`

**Why:** FBFlowBCAgent (Task 10) trains a flow-matching velocity field separately from the noise-conditioned actor. Subclass `FBModel` to add `_actor_vf`.

- [ ] **Step 1: Create `agents/fb/flow_bc/model.py`**

```python
"""
agents/fb/flow_bc/model.py — FBFlowBCModel.

Extends FBModel with an unconditional flow-matching velocity field
`_actor_vf(obs, x_t, t) -> velocity`. The standard noise-conditioned
`_actor` is kept from FBModel; the velocity field is used during
training to define the BC distillation target.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from agents.fb.model import FBModel
from nn_models import SimpleVectorFieldArchiConfig, weight_init


class FBFlowBCModel(FBModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        **kwargs,
    ):
        super().__init__(obs_space=obs_space, action_dim=action_dim, **kwargs)
        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig(hidden_dim=512, hidden_layers=4)

        # Build the velocity field on the same input space as the noise-conditioned
        # actor (i.e. _fw_encoder.output_space when actor_encode_obs=False).
        actor_in_space = (
            self._left_encoder.output_space
            if self.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = actor_vf_cfg.build(actor_in_space, action_dim)
        self._actor_vf.apply(weight_init)
        self.to(self.device)
```

- [ ] **Step 2: Smoke-verify model construction**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from agents.fb.flow_bc.model import FBFlowBCModel
obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
m = FBFlowBCModel(obs_space=obs_space, action_dim=4, z_dim=32, L_dim=32, device='cpu')
B = 8
obs = torch.randn(B, 15)
x_t = torch.randn(B, 4)
t = torch.rand(B, 1)
vel = m._actor_vf(obs, x_t, t)
print('vf out shape:', tuple(vel.shape))
"
```

Expected: `vf out shape: (8, 4)`.

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add agents/fb/flow_bc/model.py
git commit -m "feat: FBFlowBCModel — add _actor_vf flow-matching velocity field

Subclass of FBModel that builds a SimpleVectorField (obs+action+t->vel)
alongside the existing noise-conditioned actor. The agent (next task)
trains _actor_vf with the FM loss and distills _actor against the
flow-matched action."
```

---

## Task 10: Rewrite `FBFlowBCAgent` — FM loss + RL + distillation BC

**Files:**
- Modify: `agents/fb/flow_bc/agent.py` (full rewrite)

**Why:** Current FlowBC uses `NoiseConditionedActor + MSE-BC` against dataset actions. Replace with td_jepa's flow-matching trainer (`td_jepa/.../fb/flow_bc/agent.py`).

- [ ] **Step 1: Replace `agents/fb/flow_bc/agent.py`**

```python
"""
agents/fb/flow_bc/agent.py — FBFlowBCAgent.

Three loss terms per actor update:
  1. Flow-matching loss   trains _actor_vf
  2. RL loss              trains _actor with Q = F·z
  3. Distillation BC      trains _actor against ODE rollout of _actor_vf

Inherits the FB loss and target-update logic from FBAgent.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from agents.fb.agent import FBAgent, pairwise_l1_pessimism
from agents.fb.flow_bc.model import FBFlowBCModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class FBFlowBCAgent(FBAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        bc_coeff: float = 0.3,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        **fb_kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig(
                hidden_dim=512, hidden_layers=2, embedding_layers=2
            )
        # Construct FBAgent with the standard FBModel build... but we need to
        # swap in FBFlowBCModel. We avoid super().__init__ creating an FBModel
        # by replicating the bookkeeping here.
        self.batch_size = fb_kwargs.get("batch_size", 256)
        self.discount_scalar = fb_kwargs.get("discount", 0.99)
        self.lr_f = fb_kwargs.get("lr_f", 1e-4)
        self.lr_b = fb_kwargs.get("lr_b", 1e-4)
        self.lr_actor = fb_kwargs.get("lr_actor", 1e-4)
        self.weight_decay = fb_kwargs.get("weight_decay", 0.0)
        self.clip_grad_norm = fb_kwargs.get("clip_grad_norm", 0.0)
        self.ortho_coef = fb_kwargs.get("ortho_coef", 1.0)
        self.train_goal_ratio = fb_kwargs.get("train_goal_ratio", 0.5)
        self.fb_pessimism_penalty = fb_kwargs.get("fb_pessimism_penalty", 0.0)
        self.actor_pessimism_penalty = fb_kwargs.get("actor_pessimism_penalty", 0.0)
        self.actor_std = fb_kwargs.get("actor_std", 0.2)
        self.stddev_clip = fb_kwargs.get("stddev_clip", 0.3)
        self.f_target_tau = fb_kwargs.get("f_target_tau", 0.005)
        self.b_target_tau = fb_kwargs.get("b_target_tau", 0.005)
        self.device = fb_kwargs.get("device", "cpu")
        self.actor_encode_obs = fb_kwargs.get("actor_encode_obs", False)

        self.bc_coeff = bc_coeff
        self.flow_steps = flow_steps
        self.action_dim = action_dim

        self.model = FBFlowBCModel(
            obs_space=obs_space,
            action_dim=action_dim,
            actor_vf_cfg=actor_vf_cfg,
            z_dim=fb_kwargs.get("z_dim", 50),
            L_dim=fb_kwargs.get("L_dim", 50),
            actor_encode_obs=self.actor_encode_obs,
            forward_cfg=fb_kwargs.get("forward_cfg"),
            backward_cfg=fb_kwargs.get("backward_cfg"),
            left_encoder_cfg=fb_kwargs.get("left_encoder_cfg"),
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=fb_kwargs.get("obs_normalizer_cfg"),
            device=self.device,
        )

        self.forward_optimizer = Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters()),
            lr=self.lr_f, weight_decay=self.weight_decay,
        )
        self.backward_optimizer = Adam(
            self.model._backward_map.parameters(),
            lr=self.lr_b, weight_decay=self.weight_decay,
        )
        self.actor_optimizer = Adam(
            self.model._actor.parameters(),
            lr=self.lr_actor, weight_decay=self.weight_decay,
        )
        self.actor_vf_optimizer = Adam(
            self.model._actor_vf.parameters(),
            lr=lr_actor_vf, weight_decay=self.weight_decay,
        )

        eye = torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag = (1.0 - eye)
        self.off_diag_sum = self.off_diag.sum()

    # ── Override actor update ─────────────────────────────────────────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device).float()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        discount = self.discount_scalar * (1.0 - terminated)

        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        next_obs_n = self.model._fw_encoder(self.model.normalize(next_obs))
        goal = self.model._bw_encoder(self.model.normalize(next_obs))

        with torch.no_grad():
            z = self.model.mixed_z(goal, self.train_goal_ratio)

        fb_metrics = self._update_fb(obs_n, next_obs_n, action, goal, discount, z)
        actor_metrics = self._update_actor_flow(obs_n.detach(), action, z)
        self.model.soft_update_targets(self.f_target_tau, self.b_target_tau)
        return {**fb_metrics, **actor_metrics}

    def _update_actor_flow(
        self, obs: Tensor, action: Tensor, z: Tensor
    ) -> Dict[str, float]:
        # 1. Flow-matching loss (trains _actor_vf)
        x_1 = action
        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0], 1), device=action.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.model._actor_vf(obs, x_t, t)
        bc_flow_loss = (pred - vel).pow(2).mean()

        # 2. RL loss (trains _actor)
        with torch.no_grad():
            left = self.model._left_encoder(obs)
        actor_in = left if self.actor_encode_obs else obs
        noises = torch.randn_like(x_1)
        a_actor = self.model._actor(actor_in, z, noises)
        Fs = self.model._forward_map(left, z, a_actor)
        Qs = (Fs * z).sum(-1)
        _, _, Q = pairwise_l1_pessimism(Qs, self.actor_pessimism_penalty)
        rl_loss = -Q.mean()

        # 3. Distillation BC (trains _actor)
        bc_error = torch.zeros((), device=action.device)
        if self.bc_coeff > 0:
            with torch.no_grad():
                target = self.compute_flow_actions(obs, noises)
            bc_error = (a_actor - target).pow(2).mean()
            bc_loss = self.bc_coeff * bc_error
            actor_loss = (rl_loss / Qs.abs().mean().detach().clamp(min=1e-6)) + bc_loss
        else:
            actor_loss = rl_loss

        total = actor_loss + bc_flow_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._actor_vf.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        self.actor_vf_optimizer.step()

        return {
            "loss/actor": actor_loss.item() if isinstance(actor_loss, Tensor) else float(actor_loss),
            "loss/actor_rl": rl_loss.item(),
            "loss/actor_bc": float(bc_error) if isinstance(bc_error, Tensor) else bc_error,
            "loss/bc_flow": bc_flow_loss.item(),
            "train/Q_actor": Q.mean().item(),
        }

    def compute_flow_actions(self, obs: Tensor, noises: Tensor) -> Tensor:
        """Euler-step the flow matching ODE from x=noises to t=1."""
        x = noises
        for i in range(self.flow_steps):
            t = torch.full((noises.shape[0], 1), float(i) / self.flow_steps, device=obs.device)
            vel = self.model._actor_vf(obs, x, t)
            x = x + vel / self.flow_steps
        return torch.clamp(x, -1.0, 1.0)

    # ── Eval ──────────────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        actor_in = self.model._left_encoder(obs_n) if self.actor_encode_obs else obs_n
        noises = torch.randn(obs.shape[0], self.action_dim, device=self.device)
        return self.model._actor(actor_in, z, noises)

    # ── Checkpointing ─────────────────────────────────────────────────── #

    def state_dict(self) -> dict:
        s = super().state_dict()
        s["actor_vf_optimizer"] = self.actor_vf_optimizer.state_dict()
        return s

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
```

- [ ] **Step 2: Smoke-verify five update steps with synthetic data**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from agents.fb.flow_bc.agent import FBFlowBCAgent

torch.manual_seed(0)
B = 64
obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
agent = FBFlowBCAgent(
    obs_space=obs_space, action_dim=4,
    batch_size=B, z_dim=32, L_dim=32, device='cpu',
    bc_coeff=0.3, flow_steps=5,
)
def make_batch():
    return {
        'observation': torch.randn(B, 15),
        'action': torch.randn(B, 4).clamp(-1, 1),
        'next': {'observation': torch.randn(B, 15),
                 'terminated': torch.zeros(B, 1)},
    }
last = None
for step in range(5):
    last = agent.update(make_batch(), step)
print('loss keys:', sorted(k for k in last if k.startswith('loss/')))
print('all finite:', all(np.isfinite(v) for v in last.values()))
"
```

Expected: prints `loss keys: ['loss/actor', 'loss/actor_bc', 'loss/actor_rl', 'loss/bc_flow', 'loss/fb', 'loss/fb_diag', 'loss/fb_offdiag', 'loss/orth', 'loss/orth_diag', 'loss/orth_offdiag']` and `all finite: True`.

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add agents/fb/flow_bc/agent.py
git commit -m "refactor: FBFlowBCAgent — flow matching + distillation BC

Replaces the NoiseConditionedActor + MSE-BC scheme with td_jepa's flow
matching trainer: _actor_vf learns the action vector field on dataset
actions; _actor is trained with normalised Q = F·z plus a BC term that
distills the FM ODE rollout. Drops the eval-time argmax-Q sampling
heuristic; eval is a single noise -> actor pass."
```

---

## Task 11: FlowBC config + manipulation domain configs

**Files:**
- Create: `configs/agent/fb_flowbc.yaml`
- Create: `configs/domain/cube_single.yaml`
- Create: `configs/domain/cube_double.yaml`
- Create: `configs/domain/scene.yaml`
- Create: `configs/domain/puzzle_3x3.yaml`

- [ ] **Step 1: Create `configs/agent/fb_flowbc.yaml`**

```yaml
# @package _global_
defaults:
  - fb           # inherit everything from configs/agent/fb.yaml
  - _self_

agent: fb_flowbc

# FlowBC-specific
bc_coeff: 0.3        # antmaze default; manipulation domains override to 3.0
flow_steps: 10
lr_actor_vf: 3.0e-4

# Actor becomes noise-conditioned
actor:
  name: noise_conditioned_actor
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2

# Velocity field for flow matching
actor_vf:
  hidden_dim: 512
  hidden_layers: 4
```

- [ ] **Step 2: Create `configs/domain/cube_single.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb_flowbc
  - _self_

domain: cube-single-play-v0
agent: fb_flowbc
bc_coeff: 3.0
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 3: Create `configs/domain/cube_double.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb_flowbc
  - _self_

domain: cube-double-play-v0
agent: fb_flowbc
bc_coeff: 3.0
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 4: Create `configs/domain/scene.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb_flowbc
  - _self_

domain: scene-play-v0
agent: fb_flowbc
bc_coeff: 3.0
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 5: Create `configs/domain/puzzle_3x3.yaml`**

```yaml
# @package _global_
defaults:
  - /agent: fb_flowbc
  - _self_

domain: puzzle-3x3-play-v0
agent: fb_flowbc
bc_coeff: 3.0
ortho_coef: 1.0
lr_b: 1.0e-4
```

- [ ] **Step 6: Smoke-verify FlowBC config composition + build**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import numpy as np
from gymnasium.spaces import Box
from hydra import compose, initialize
from train import make_agent
with initialize(config_path='configs', version_base='1.3'):
    cfg = compose(config_name='train',
                  overrides=['domain=cube_single', 'device=cpu', 'batch_size=32'])
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    print('agent type:', type(agent).__name__)
    print('bc_coeff:', cfg.bc_coeff, 'flow_steps:', cfg.flow_steps)
    print('actor name:', cfg.actor.name)
"
```

Expected:
```
agent type: FBFlowBCAgent
bc_coeff: 3.0 flow_steps: 10
actor name: noise_conditioned_actor
```

- [ ] **Step 7: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add configs/agent/fb_flowbc.yaml configs/domain/cube_single.yaml configs/domain/cube_double.yaml configs/domain/scene.yaml configs/domain/puzzle_3x3.yaml
git commit -m "feat: configs — fb_flowbc agent and cube/scene/puzzle domains

Inherits fb.yaml defaults and overrides the actor to noise_conditioned,
adds the actor_vf flow-matching network, sets bc_coeff=3.0 per
td_jepa's manipulation launch."
```

---

## Task 12: Update README

**Files:**
- Modify: `README.md`

**Why:** The README still references the broken behaviors (LS reward inference, MSE-BC FlowBC, the wrong defaults). Bring it in line with the new implementation.

- [ ] **Step 1: Rewrite the "Method", "Key hyperparameters", "Data layout", "Evaluation" sections**

Apply these edits to `README.md`:

**(a) Update the "Key hyperparameters" table** (around line 119-131). Replace its rows with:

```markdown
| Parameter | Default | Notes |
|---|---|---|
| `z_dim` | 50 | Latent goal dimension |
| `L_dim` | 50 | Left-encoder output dimension |
| `hidden_dim` (F, B, left_encoder, actor) | 512 | Width of MLP blocks |
| `forward.hidden_layers` | 2 | Forward map depth |
| `backward.hidden_layers` | 4 | Backward map depth |
| `left_encoder.hidden_layers` | 4 | Left encoder depth |
| `forward.num_parallel` | 2 | F ensemble heads |
| `ortho_coef` | 1.0 | B orthonormality weight (sweep 100/1000 for manipulation) |
| `actor_std` | 0.2 | TruncatedNormal std (FB only; ignored by FlowBC) |
| `stddev_clip` | 0.3 | Action sample clip |
| `train_goal_ratio` | 0.5 | Fraction of z drawn from B(next_obs[perm]) |
| `f_target_tau`, `b_target_tau` | 0.005 | Polyak soft-update rates |
| `bc_coeff` (FlowBC) | 0.3 antmaze / 3.0 manip | Distillation BC weight |
| `flow_steps` (FlowBC) | 10 | Euler steps for the action-flow ODE |
| `actor_encode_obs` | false | If true, actor sees the left-encoded representation |
```

**(b) Update the "Data layout" section** (around line 169-196) to reflect `next/physics`:

Replace the table after "Each `.npz` contains arrays of shape `[T, ·]` over one episode:" with:

```markdown
| Key | Shape | Description |
|---|---|---|
| `observation` | [T, obs_dim] | Proprioceptive state |
| `action` | [T, act_dim] | |
| `physics` | [T, phys_dim] | Full physics state; used for next-state reward relabelling |
| `discount` | [T] | 0 at terminal steps |
```

And below it, replace the "load_transitions converts ..." paragraph with:

```markdown
`load_ogbench_dataset` converts T-step episodes into T−1 transition tuples and
stores them in a `DictBuffer` under the layout:

```
batch["observation"]            [B, obs_dim]
batch["action"]                 [B, act_dim]
batch["next"]["observation"]    [B, obs_dim]
batch["next"]["physics"]        [B, phys_dim]
batch["next"]["terminated"]     [B, 1]
```
```

**(c) Update the "Evaluation" section** (around line 199-209). Replace its numbered list with:

```markdown
1. Sample `eval_relabel_size` transitions from the offline buffer.
2. Relabel rewards with `get_relabel_fn(domain, task)(batch["next"]["physics"], action)`.
3. Add `eval_shift_reward` (default 1.0) — shifts sparse rewards from `{−k, …, 0}` to `{1−k, …, 1}`.
4. Compute `z = project(Σ r_i · B(next_obs_i))` via `agent.model.reward_inference`.
5. Roll out `eval_n_episodes` episodes via `envs/rollout.py`.
6. Log `eval/{task}/success` and `eval/{task}/reward`.
```

**(d) Strip the old "Method" subsection** describing FlowBC's actor as `NoiseConditionedActor + BC loss` and replace with:

```markdown
- `fb_flowbc`: a noise-conditioned actor distilled from a flow-matching velocity field
  `v_θ(obs, x_t, t)`. The velocity field is trained on dataset actions; the actor is
  trained with `Q = F·z` plus a BC term that distills the ODE rollout of `v_θ`.
  Used for cube / scene / puzzle.
```

- [ ] **Step 2: Verify README renders without broken references**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
grep -nE "TBD|TODO|placeholder|Maktefile" README.md && echo "FOUND ISSUES" || echo "OK"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git add README.md
git commit -m "docs: README — align with faithful FB implementation

Updates hyperparameter table to td_jepa OGBench defaults, fixes the
data layout doc to show next.physics, rewrites the evaluation section
to describe the linear reward_inference path, and corrects FlowBC's
description (flow matching + distillation, not noise-conditioned BC)."
```

---

## Task 13: Final smoke — full pipeline import + 1 update with each agent

**Files:** none modified

**Why:** Catch any regressions where late tasks accidentally broke earlier ones.

- [ ] **Step 1: Run the full pipeline import + a single update for both agents**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
python -c "
import torch, numpy as np
from gymnasium.spaces import Box
from hydra import compose, initialize
from train import make_agent

torch.manual_seed(0)
np.random.seed(0)
B = 32

def smoke(domain, expected_type):
    with initialize(config_path='configs', version_base='1.3'):
        cfg = compose(config_name='train',
                      overrides=[f'domain={domain}', 'device=cpu', f'batch_size={B}'])
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    assert type(agent).__name__ == expected_type, (type(agent).__name__, expected_type)
    batch = {
        'observation': torch.randn(B, 15),
        'action': torch.randn(B, 4).clamp(-1, 1),
        'next': {'observation': torch.randn(B, 15),
                 'terminated': torch.zeros(B, 1)},
    }
    m = agent.update(batch, 0)
    print(domain, type(agent).__name__, 'loss/fb=', round(m['loss/fb'], 3), 'loss/actor=', round(m['loss/actor'], 3))

smoke('antmaze_medium', 'FBAgent')
smoke('cube_single', 'FBFlowBCAgent')
"
```

Expected: both lines print finite loss values.

- [ ] **Step 2: Verify git tree is clean and the spec is referenced from the README or plan**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB
git status -s
ls docs/superpowers/specs/ docs/superpowers/plans/
```

Expected: `git status -s` empty (or only untracked files unrelated to this work). Both `2026-05-11-*` files present.

- [ ] **Step 3: No commit needed — this is a verification task**

If any check failed, return to the corresponding task to fix.

---

## Coverage map (spec section → tasks)

| Spec section | Task(s) |
|---|---|
| §3 bug 1 (FB loss) | Task 5 |
| §3 bug 2 (actor Q) | Task 5 |
| §3 bug 3 (ortho) | Task 5 |
| §3 bug 4 (reward inference) | Task 4 |
| §3 bug 5 (physics relabel) | Tasks 2, 8 |
| §3 bug 6 (loader import) | Task 2 |
| §3 bug 7 (FlowBC mechanism) | Tasks 9, 10 |
| §3 bug 8 (mixed-z) | Task 4 |
| §4.1 additions (FBFlowBCModel, IdentityNNConfig, encoders) | Tasks 3, 4, 9 |
| §4.2 modifications | Tasks 2, 4, 5, 7, 8, 10 |
| §4.3 deletions (tests, README sections) | Tasks 1, 12 |
| §5 math (FB agent) | Task 5 |
| §6 architecture (call chain) | Task 4 |
| §7 config defaults | Tasks 6, 11 |
| §8 data + eval pipeline | Tasks 2, 8 |
| §9 FlowBC trainer | Task 10 |
| §10 out of scope | (negative; nothing to do) |
| §11 risks | called out inline in Tasks 4, 5, 10 |
| Success criterion (`expected_results.md`) | (not asserted in plan; ran by hand) |
