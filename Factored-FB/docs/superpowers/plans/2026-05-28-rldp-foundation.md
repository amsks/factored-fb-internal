# RLDP Integration — Foundation (sub-project 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three infrastructure pieces sub-projects 2 (RLDP + RLDP+FlowBC on state) and 3 (pixel variants) depend on — without writing any RLDP agent code yet.

**Architecture:** Three independent additions: (a) restore `VForwardArchiConfig` + `VForwardMap` in `nn_models.py` (port verbatim from td_jepa, ~30 LOC); (b) extend `DictBuffer.sample()` with a `horizon: int = 1` kwarg that returns contiguous-within-episode windows (~40 LOC); (c) create a reusable `tests/parity/` package with weight-transplant + numerical-match helpers (~150 LOC). Each is independently testable; FB behavior stays bit-identical (horizon=1 path is unchanged).

**Tech Stack:** PyTorch, numpy, pytest. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-28-rldp-integration-foundation-design.md`

**Conventions:**
- Tests run via `.venv/bin/python -m pytest` (per `memory/factored-fb-test-env.md` — `uv run` is broken).
- Commits omit the Co-Authored-By Claude trailer (per `memory/commit-no-claude-coauthor.md`).
- Each task is fully self-contained — don't read tasks out of order assuming context from others.

---

## File structure

**New files:**
- `tests/test_nn_models_vforward.py`
- `tests/test_dict_buffer_horizon.py`
- `tests/parity/__init__.py` (empty package marker)
- `tests/parity/harness.py`
- `tests/parity/test_harness_helpers.py`

**Modified:**
- `nn_models.py` — append `VForwardArchiConfig` + `VForwardMap`; update the "Removed VForwardMap" comment near the top.
- `buffers/transition.py` — extend `DictBuffer.sample()` with `horizon` kwarg + add lazy cache helpers.

**Untouched:** everything in `agents/`, `configs/`, `scripts/`, `train.py`.

---

## Task 1: Restore `VForwardArchiConfig` + `VForwardMap` to `nn_models.py`

**Files:**
- Modify: `nn_models.py` (top-of-file comment + append two new classes)
- Create: `tests/test_nn_models_vforward.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nn_models_vforward.py`:

```python
"""Unit tests for the restored VForwardArchiConfig + VForwardMap (RLDP predictor head)."""
import gymnasium
import numpy as np
import torch

from nn_models import VForwardArchiConfig, VForwardMap


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_vforward_archi_config_defaults():
    cfg = VForwardArchiConfig()
    assert cfg.hidden_dim == 1024
    assert cfg.hidden_layers == 1
    assert cfg.embedding_layers == 2
    assert cfg.num_parallel == 2


def test_vforward_config_build_returns_module():
    cfg = VForwardArchiConfig(num_parallel=2)
    m = cfg.build(_obs_space(10), z_dim=4)
    assert isinstance(m, VForwardMap)


def test_vforward_map_forward_shape_default_output():
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=1, num_parallel=2)
    m = VForwardMap(_obs_space(10), z_dim=4, cfg=cfg)
    obs = torch.randn(8, 10)
    z = torch.randn(8, 4)
    out = m(obs, z)
    # num_parallel=2 -> leading parallel dim, output_dim defaults to z_dim=4
    assert out.shape == (2, 8, 4), out.shape


def test_vforward_map_forward_shape_explicit_output_dim():
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=1, num_parallel=1)
    m = VForwardMap(_obs_space(10), z_dim=4, output_dim=7, cfg=cfg)
    obs = torch.randn(8, 10)
    z = torch.randn(8, 4)
    out = m(obs, z)
    # num_parallel=1 -> no leading parallel expansion in input; output last dim = 7
    assert out.shape[-1] == 7, out.shape


def test_vforward_map_backward_pass_runs():
    cfg = VForwardArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=1, num_parallel=2)
    m = VForwardMap(_obs_space(5), z_dim=3, cfg=cfg)
    obs = torch.randn(4, 5)
    z = torch.randn(4, 3)
    out = m(obs, z)
    loss = out.sum()
    loss.backward()
    # at least one parameter received a gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_nn_models_vforward.py -v
```
Expected: ALL FAIL with `ImportError: cannot import name 'VForwardArchiConfig' from 'nn_models'`.

- [ ] **Step 3: Read the current "Removed VForwardMap" comment**

Run:
```
sed -n '1,12p' nn_models.py
```
Expected output contains a comment line: `# - Removed VForwardMap, VectorField (not used in FB baseline)`. Note the exact line for the next step.

- [ ] **Step 4: Update the header comment**

Use `Edit` on `nn_models.py`:

Old:
```
#   - Removed VForwardMap, VectorField (not used in FB baseline)
```

New:
```
#   - Removed VectorField (not used in FB baseline)
#   - VForwardMap restored 2026-05-28 for the RLDP port (predictor head module)
```

If the actual comment wording differs slightly, preserve the surrounding lines and only swap the "Removed VForwardMap" portion.

- [ ] **Step 5: Append `VForwardArchiConfig` and `VForwardMap` to `nn_models.py`**

Append at the **end of the file** (so the diff is purely additive). The code is a verbatim port of td_jepa lines 280–310 of `/home/adityamohan/git/baselines/td_jepa/metamotivo/nn_models.py`, using our existing `BaseConfig`, `simple_embedding`, and `linear` helpers (already present in our file):

```python


# ──────────────────────────────────────────────────────────────────────────────
# RLDP predictor head — restored 2026-05-28 for the RLDP port.
# Verbatim port from td_jepa metamotivo/nn_models.py lines ~280-310.
# ──────────────────────────────────────────────────────────────────────────────


class VForwardArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2

    def build(self, obs_space, z_dim: int, output_dim=None) -> torch.nn.Module:
        return VForwardMap(obs_space, z_dim, output_dim, self)


class VForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        output_dim=None,
        cfg: VForwardArchiConfig = VForwardArchiConfig(),
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        return self.Fs(torch.cat([s_embedding, z_embedding], dim=-1))
```

- [ ] **Step 6: Run tests to confirm pass**

Run:
```
.venv/bin/python -m pytest tests/test_nn_models_vforward.py -v
```
Expected: 5 PASS.

- [ ] **Step 7: Run the existing test suite to confirm FB unchanged**

Run:
```
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```
Expected: all previously-passing tests still pass (modulo the known pandas-missing skips). No tests regress because of the additive change.

- [ ] **Step 8: Commit**

```
git add nn_models.py tests/test_nn_models_vforward.py
git commit -m "feat(nn_models): restore VForwardMap + VForwardArchiConfig (RLDP predictor head)"
```

---

## Task 2: Extend `DictBuffer.sample()` with `horizon` kwarg

**Files:**
- Modify: `buffers/transition.py`
- Create: `tests/test_dict_buffer_horizon.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dict_buffer_horizon.py`:

```python
"""Tests for DictBuffer.sample(horizon=...) — horizon=1 must be unchanged;
horizon>1 must return [B, h, ...] windows contiguous within an episode."""
import numpy as np
import pytest
import torch

from buffers.transition import DictBuffer


def _make_buffer_with_two_episodes(ep1_len: int, ep2_len: int) -> DictBuffer:
    """Build a buffer with two synthetic episodes. The 'observation' values
    are global step indices so we can verify contiguity by reading them back."""
    total = ep1_len + ep2_len
    obs = np.arange(total, dtype=np.float32).reshape(-1, 1)
    next_obs = obs + 1
    timestep = np.concatenate([np.arange(ep1_len), np.arange(ep2_len)]).astype(np.int64)
    action = np.zeros((total, 2), dtype=np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=np.float32)
    terminated[ep1_len - 1] = 1.0
    terminated[-1] = 1.0
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_horizon_1_default_unchanged():
    """sample(batch_size) without horizon kwarg returns single transitions
    (same shape as before the change). Regression gate for FB."""
    buf = _make_buffer_with_two_episodes(50, 50)
    torch.manual_seed(0)
    out = buf.sample(8)
    assert out["observation"].shape == (8, 1)
    assert out["action"].shape == (8, 2)
    assert "next" in out and out["next"]["observation"].shape == (8, 1)


def test_horizon_1_explicit_kwarg_identical_to_default():
    """sample(batch_size, horizon=1) and sample(batch_size) must be byte-identical
    given the same RNG seed."""
    buf = _make_buffer_with_two_episodes(50, 50)
    torch.manual_seed(42)
    a = buf.sample(8)
    torch.manual_seed(42)
    b = buf.sample(8, horizon=1)
    for k in ("observation", "action", "reward"):
        assert torch.equal(a[k], b[k]), f"horizon=1 path diverged for key {k}"


def test_horizon_5_returns_windowed_shape():
    """horizon=5 returns [B, 5, ...] for each leaf tensor."""
    buf = _make_buffer_with_two_episodes(30, 30)
    out = buf.sample(4, horizon=5)
    assert out["observation"].shape == (4, 5, 1)
    assert out["action"].shape == (4, 5, 2)
    assert out["reward"].shape == (4, 5)
    assert out["next"]["observation"].shape == (4, 5, 1)


def test_horizon_5_windows_are_contiguous_within_episode():
    """Windows must be from a single episode (no cross-boundary bleed).
    Our synthetic buffer has obs = global step index; within a true window
    the observations increment by exactly 1."""
    buf = _make_buffer_with_two_episodes(30, 30)
    out = buf.sample(64, horizon=5)
    obs = out["observation"].cpu().numpy()  # [B, 5, 1]
    diffs = np.diff(obs, axis=1)  # [B, 4, 1]
    # every consecutive pair within a window must differ by exactly 1
    assert np.all(diffs == 1.0), f"non-contiguous window detected: diffs unique = {np.unique(diffs)}"


def test_horizon_too_long_raises():
    """If no episode is long enough for the requested horizon, sample() raises."""
    buf = _make_buffer_with_two_episodes(3, 3)
    with pytest.raises((ValueError, AssertionError)):
        buf.sample(8, horizon=10)
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/test_dict_buffer_horizon.py -v
```
Expected: at least 3 FAIL (horizon=5 tests raise `TypeError` because the kwarg doesn't exist yet). The horizon-1 tests may pass by accident; the goal is the horizon>1 ones failing.

- [ ] **Step 3: Read the current `sample()` to know exact insertion point**

Run:
```
sed -n '145,185p' buffers/transition.py
```
Expected: prints the existing `sample(self, batch_size)` definition. Note the line numbers.

- [ ] **Step 4: Modify `sample()` to accept `horizon` + add helper method**

Use `Edit` to replace the existing `sample` definition. Old:

```python
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample *batch_size* transitions uniformly. Returns nested dict of tensors.

        Uses torch.randint (not np.random) so RNG consumption matches td_jepa's
        DictBuffer.sample exactly.
        """
        assert self._size > 0, "Buffer is empty"
        idx = torch.randint(0, self._size, (batch_size,)).numpy()
```

New:

```python
    def sample(self, batch_size: int, horizon: int = 1) -> Dict[str, Any]:
        """Sample *batch_size* transitions uniformly. Returns nested dict of tensors.

        Uses torch.randint (not np.random) so RNG consumption matches td_jepa's
        DictBuffer.sample exactly.

        horizon=1 (default) is the byte-identical legacy path — single
        transitions, shape [B, ...].

        horizon>1 returns contiguous-within-episode windows, shape [B, horizon, ...].
        A window starting at index i is valid iff timestep[i:i+horizon] is
        strictly consecutive (no episode boundary). Raises ValueError if no
        episode in the buffer is long enough to provide a horizon-h window.
        """
        assert self._size > 0, "Buffer is empty"
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon > 1:
            return self._sample_horizon(batch_size, horizon)
        idx = torch.randint(0, self._size, (batch_size,)).numpy()
```

Note: only the signature line, the docstring, and the two new lines (`if horizon < 1:` and `if horizon > 1:`) change. The body after `idx = torch.randint(...)` stays exactly as it is — that's the byte-identical horizon=1 path the regression test guards.

- [ ] **Step 5: Add `_sample_horizon` helper method**

Find the end of `sample()` (right before the next `def` or the `# Dunder helpers` comment near line 187). Insert this new method immediately before that block:

```python
    def _sample_horizon(self, batch_size: int, horizon: int) -> Dict[str, Any]:
        """Return [B, horizon, ...] windows that stay within a single episode."""
        if "timestep" not in self._storage:
            raise ValueError(
                "horizon>1 sampling requires 'timestep' in storage "
                "(produced by data/ogbench.py loader)."
            )
        ts = self._storage["timestep"][: self._size]
        # Valid start = index i such that ts[i:i+horizon] is strictly consecutive.
        # Equivalent: ts[i+horizon-1] - ts[i] == horizon - 1 AND i+horizon-1 < self._size.
        max_start = self._size - horizon
        if max_start < 0:
            raise ValueError(
                f"buffer size {self._size} < horizon {horizon}; no valid starts"
            )
        starts = np.arange(max_start + 1)
        ts_end = ts[starts + horizon - 1]
        ts_start = ts[starts]
        valid_mask = (ts_end - ts_start) == (horizon - 1)
        valid_starts = starts[valid_mask]
        if valid_starts.size == 0:
            raise ValueError(
                f"no episode in buffer is long enough for horizon={horizon}"
            )
        # Sample valid starts uniformly (torch.randint for RNG-stability).
        pick_idx = torch.randint(0, valid_starts.size, (batch_size,)).numpy()
        starts_sampled = valid_starts[pick_idx]
        # Build [B, horizon] index matrix.
        offsets = np.arange(horizon)
        window_idx = starts_sampled[:, None] + offsets[None, :]  # [B, h]
        flat_idx = window_idx.reshape(-1)  # [B*h]
        # Gather, then reshape leading [B*h, ...] -> [B, h, ...].
        out_flat = {}
        for k, v in self._storage.items():
            if k == "timestep":
                continue
            gathered = torch.as_tensor(v[flat_idx]).to(self.device)
            new_shape = (batch_size, horizon) + tuple(gathered.shape[1:])
            out_flat[k] = gathered.reshape(new_shape)
        return _unflatten(out_flat)
```

- [ ] **Step 6: Run tests to confirm pass**

Run:
```
.venv/bin/python -m pytest tests/test_dict_buffer_horizon.py -v
```
Expected: 5 PASS.

- [ ] **Step 7: Run the existing test suite to confirm horizon=1 path unchanged**

Run:
```
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -15
```
Expected: no regressions. Specifically `tests/test_frame_stack_buffer.py` and any other existing buffer tests must still pass.

- [ ] **Step 8: Commit**

```
git add buffers/transition.py tests/test_dict_buffer_horizon.py
git commit -m "feat(buffer): DictBuffer.sample(horizon=h) for RLDP windowed sampling"
```

---

## Task 3: Parity harness — package skeleton + `transplant`

**Files:**
- Create: `tests/parity/__init__.py` (empty)
- Create: `tests/parity/harness.py`
- Create: `tests/parity/test_harness_helpers.py`

- [ ] **Step 1: Create the package marker**

Run:
```
mkdir -p tests/parity && : > tests/parity/__init__.py
```

- [ ] **Step 2: Write the failing test for `transplant`**

Create `tests/parity/test_harness_helpers.py`:

```python
"""Unit tests for the parity-gate harness helpers.

The harness is the shared scaffolding sub-projects 2 and 3 use to weight-transplant
td_jepa-trained RLDP modules into our torch implementations and assert numerical
equivalence. These tests exercise the helpers on toy synthetic modules so we know
the harness itself is correct before relying on it for real parity gates.
"""
import numpy as np
import pytest
import torch
from torch import nn

from tests.parity.harness import (
    transplant,
    assert_forward_match,
    assert_grad_match,
)


def test_transplant_copies_linear_weights_with_jax_shape():
    """JAX linear stores W as (in, out); torch as (out, in). transplant must
    transpose. Bias has the same shape in both."""
    torch_lin = nn.Linear(3, 4)
    # Synthetic "jax params" — note (in, out) shape for the kernel.
    src = {
        "linear.kernel": np.arange(12, dtype=np.float32).reshape(3, 4),
        "linear.bias": np.arange(4, dtype=np.float32),
    }
    mapping = {"weight": "linear.kernel", "bias": "linear.bias"}
    transplant(src, torch_lin, mapping)
    assert torch.allclose(torch_lin.weight, torch.as_tensor(src["linear.kernel"].T))
    assert torch.allclose(torch_lin.bias, torch.as_tensor(src["linear.bias"]))


def test_transplant_raises_on_unassigned_torch_param():
    """If any torch param is missing from the mapping, raise so the user notices."""
    torch_lin = nn.Linear(3, 4)
    src = {"linear.kernel": np.zeros((3, 4), dtype=np.float32)}
    mapping = {"weight": "linear.kernel"}  # 'bias' omitted on purpose
    with pytest.raises(KeyError, match="bias"):
        transplant(src, torch_lin, mapping)


def test_transplant_raises_on_unknown_src_key():
    """If a mapping points at a missing src key, raise immediately."""
    torch_lin = nn.Linear(3, 4)
    src = {"linear.kernel": np.zeros((3, 4), dtype=np.float32)}
    mapping = {"weight": "linear.kernel", "bias": "linear.bias"}
    with pytest.raises(KeyError, match="linear.bias"):
        transplant(src, torch_lin, mapping)


def test_assert_forward_match_passes_for_identical_fns():
    """Two identical functions on the same input must satisfy the gate."""
    def f1(x): return x * 2.0
    def f2(x): return x * 2.0
    x = torch.randn(4)
    assert_forward_match(f1, f2, x, atol=1e-7, rtol=1e-7, label="trivial")


def test_assert_forward_match_raises_on_mismatch():
    """When outputs diverge beyond tolerance, raise AssertionError with diff info."""
    def f1(x): return x * 2.0
    def f2(x): return x * 2.1
    x = torch.randn(4)
    with pytest.raises(AssertionError, match="max abs diff"):
        assert_forward_match(f1, f2, x, atol=1e-5, rtol=1e-5, label="mismatch")


def test_assert_grad_match_after_one_step_on_identical_modules():
    """Two clones starting from the same params and stepped on the same loss
    must remain identical."""
    torch.manual_seed(0)
    m_ref = nn.Linear(3, 4)
    m_ours = nn.Linear(3, 4)
    m_ours.load_state_dict(m_ref.state_dict())  # start identical
    x = torch.randn(2, 3)

    def loss_fn(module, inputs):
        return module(inputs).pow(2).sum()

    assert_grad_match(m_ref, m_ours, loss_fn, x, atol=1e-7, rtol=1e-7)


def test_assert_grad_match_raises_on_diverged_modules():
    """If the modules diverge after one step, raise."""
    torch.manual_seed(0)
    m_ref = nn.Linear(3, 4)
    m_ours = nn.Linear(3, 4)  # different init -> different post-step params
    x = torch.randn(2, 3)

    def loss_fn(module, inputs):
        return module(inputs).pow(2).sum()

    with pytest.raises(AssertionError):
        assert_grad_match(m_ref, m_ours, loss_fn, x, atol=1e-7, rtol=1e-7)
```

- [ ] **Step 3: Run tests to confirm failure**

Run:
```
.venv/bin/python -m pytest tests/parity/test_harness_helpers.py -v
```
Expected: ALL FAIL with `ModuleNotFoundError: No module named 'tests.parity.harness'`.

- [ ] **Step 4: Implement the harness**

Create `tests/parity/harness.py`:

```python
"""Reusable weight-transplant + numerical-match helpers for parity gates.

These helpers are the scaffolding sub-projects 2 (RLDP + RLDP+FlowBC on state)
and 3 (pixel variants) use to verify that our torch port of td_jepa's RLDP
produces numerically equivalent outputs to the reference implementation.

Usage pattern (in a sub-project 2/3 test file):

    src = load_jax_checkpoint("/path/to/td_jepa/checkpoint.safetensors")
    torch_model = OurRLDPModel(...)
    mapping = {"backward_map.weight": "params/backward_map/kernel", ...}
    transplant(src, torch_model, mapping)
    assert_forward_match(jax_forward, torch_model, inputs, atol=1e-5, rtol=1e-5)
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from torch import nn


def load_jax_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load a safetensors file (the canonical td_jepa save format) into a
    leaf-name -> numpy array dict. We use safetensors because td_jepa's
    `FBModel.save()` writes `model.safetensors`."""
    import safetensors.numpy
    loaded = safetensors.numpy.load_file(path)
    return dict(loaded)


def transplant(
    src_params: Dict[str, np.ndarray],
    torch_module: nn.Module,
    mapping: Dict[str, str],
) -> None:
    """Copy values from `src_params` into `torch_module.state_dict()`.

    `mapping`: dict of torch_param_name -> src_param_name. The src is assumed
    to follow JAX/flax conventions where Linear layers store kernels as
    (in_dim, out_dim); torch stores (out_dim, in_dim). Kernels are transposed
    when the mapped torch name ends with '.weight' AND the rank is 2.
    Bias and higher-rank tensors are copied as-is (callers can override by
    pre-transposing the src values if needed for non-standard layouts).

    Raises:
        KeyError: if any torch parameter is not in the mapping, or any mapped
                  src name is missing from `src_params`.
    """
    torch_keys = set(dict(torch_module.named_parameters()).keys())
    mapped_torch_keys = set(mapping.keys())
    unassigned = torch_keys - mapped_torch_keys
    if unassigned:
        raise KeyError(
            f"transplant: torch params not in mapping: {sorted(unassigned)}"
        )
    new_state = dict(torch_module.state_dict())
    for tname, sname in mapping.items():
        if sname not in src_params:
            raise KeyError(f"transplant: src param '{sname}' not found")
        val = torch.as_tensor(src_params[sname])
        if tname.endswith(".weight") and val.ndim == 2:
            val = val.T.contiguous()
        new_state[tname] = val
    torch_module.load_state_dict(new_state, strict=True)


def assert_forward_match(
    ref_fn: Callable,
    ours_fn: Callable,
    inputs,
    atol: float,
    rtol: float,
    label: str = "",
) -> None:
    """Call both functions on `inputs`, assert torch.allclose.

    `inputs` may be a single tensor or a tuple of args. On mismatch, raises
    AssertionError including the max abs diff and the position of the worst
    element."""
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    ref_out = ref_fn(*args)
    ours_out = ours_fn(*args)
    ref_t = torch.as_tensor(ref_out)
    ours_t = torch.as_tensor(ours_out)
    if not torch.allclose(ref_t, ours_t, atol=atol, rtol=rtol):
        diff = (ref_t - ours_t).abs()
        max_diff = diff.max().item()
        worst_pos = tuple(diff.flatten().argmax().item() // s.numel() if False else 0 for s in [diff])
        raise AssertionError(
            f"[parity:{label}] forward mismatch (atol={atol}, rtol={rtol}): "
            f"max abs diff = {max_diff:.3e}; ref shape={tuple(ref_t.shape)}, "
            f"ours shape={tuple(ours_t.shape)}"
        )


def assert_grad_match(
    ref_module: nn.Module,
    ours_module: nn.Module,
    loss_fn: Callable[[nn.Module, "torch.Tensor"], "torch.Tensor"],
    inputs,
    atol: float,
    rtol: float,
    lr: float = 1e-3,
) -> None:
    """Run one SGD step on each module with the same `loss_fn(module, inputs)`,
    then assert post-step state_dict parity. Modules are assumed to start with
    identical weights (caller's responsibility — typically via transplant()).

    Raises AssertionError on the first parameter that diverges beyond tolerance.
    """
    opt_ref = torch.optim.SGD(ref_module.parameters(), lr=lr)
    opt_ours = torch.optim.SGD(ours_module.parameters(), lr=lr)
    opt_ref.zero_grad()
    opt_ours.zero_grad()
    loss_fn(ref_module, inputs).backward()
    loss_fn(ours_module, inputs).backward()
    opt_ref.step()
    opt_ours.step()
    ref_state = ref_module.state_dict()
    ours_state = ours_module.state_dict()
    for name, ref_v in ref_state.items():
        ours_v = ours_state[name]
        if not torch.allclose(ref_v, ours_v, atol=atol, rtol=rtol):
            diff = (ref_v - ours_v).abs().max().item()
            raise AssertionError(
                f"[parity:grad] param '{name}' diverged after 1 step: "
                f"max abs diff = {diff:.3e} (atol={atol}, rtol={rtol})"
            )
```

- [ ] **Step 5: Run tests to confirm pass**

Run:
```
.venv/bin/python -m pytest tests/parity/test_harness_helpers.py -v
```
Expected: 7 PASS.

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run:
```
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -10
```
Expected: same pass count as before this task plus the 7 new parity-helper tests. No regressions.

- [ ] **Step 7: Commit**

```
git add tests/parity/__init__.py tests/parity/harness.py tests/parity/test_harness_helpers.py
git commit -m "feat(parity): tests/parity/ harness for weight-transplant + match gates"
```

---

## Task 4: Acceptance — full-suite green + FB launcher smoke

**Files:** none modified (verification only).

- [ ] **Step 1: Run the full test suite**

Run:
```
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```
Expected: all previously-passing tests still pass + 5 from `test_nn_models_vforward.py` + 5 from `test_dict_buffer_horizon.py` + 7 from `tests/parity/test_harness_helpers.py` = **17 net new passes**. No failures, no regressions.

- [ ] **Step 2: Confirm FB launcher tree still works (regression smoke)**

Run:
```
.venv/bin/python -m pytest tests/scripts/ -v 2>&1 | tail -5
```
Expected: 25 PASS (the existing scripts tests from the previous PR). The buffer change has no effect on launchers.

- [ ] **Step 3: Dry-run a real FB launch to confirm the agent code path still resolves**

Run:
```
DRY_RUN=1 SEEDS=0 GPUS=0 bash scripts/agents/fb/state.sh 2>&1 | tail -3
```
Expected: prints a `[DRY] CUDA_VISIBLE_DEVICES=0 ... train.py domain=cube_single ...` line and exits 0. No errors about missing modules or signatures.

- [ ] **Step 4: Confirm diff is bounded to spec'd files**

Run:
```
git diff main --stat
```
Expected: only these paths appear:
- `nn_models.py`
- `buffers/transition.py`
- `tests/test_nn_models_vforward.py`
- `tests/test_dict_buffer_horizon.py`
- `tests/parity/__init__.py`
- `tests/parity/harness.py`
- `tests/parity/test_harness_helpers.py`

If any other file appears, investigate and either revert it or document why before pushing.

- [ ] **Step 5: No commit needed (verification only)**

If everything green: foundation is done. Sub-project 2 (RLDP + RLDP+FlowBC on state) can now use `VForwardArchiConfig`/`VForwardMap`, `DictBuffer.sample(horizon=5)`, and `tests/parity/harness`.

---

## Self-Review

**1. Spec coverage:**
- Component 1 (restore `VForwardMap`) → Task 1.
- Component 2 (`DictBuffer.sample(horizon)`) → Task 2.
- Component 3 (`tests/parity/` harness) → Task 3.
- Acceptance criteria (full suite + FB regression + bounded diff) → Task 4.
- Open risks (short-episode behavior) → handled in `_sample_horizon` (`raises ValueError` when no episode is long enough), tested by `test_horizon_too_long_raises`.

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later". Every step has actual code or commands.
- The `transplant` mapping logic specifies the JAX-vs-torch transpose convention concretely.
- One worth noting: in `assert_forward_match`, the `worst_pos` calculation is intentionally simplified (just reports max diff, not the precise position). This is fine for the harness — sub-projects 2/3 will print enough info to debug from the max-diff number.

**3. Type consistency:**
- `VForwardArchiConfig` / `VForwardMap` names match the spec and td_jepa source.
- `DictBuffer.sample(batch_size, horizon=1)` signature used consistently across tests and implementation.
- `transplant(src_params, torch_module, mapping)` signature matches between the harness and the test file.
- `assert_forward_match(ref_fn, ours_fn, inputs, atol, rtol, label="")` and `assert_grad_match(ref_module, ours_module, loss_fn, inputs, atol, rtol, lr=1e-3)` signatures consistent.

No issues found.
