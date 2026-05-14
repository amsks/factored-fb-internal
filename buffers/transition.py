"""
buffers/transition.py — Generic nested-dict replay buffer.

DictBuffer accepts arbitrary nested-dict data structures (e.g. the
``{"observation": ..., "next": {"observation": ..., "terminated": ...}}``
layout used by OGBench episodes) and returns the same structure from
``sample()``.

Internally all arrays are stored flat (``"next/observation"`` etc.) for
O(1) random-access; the nesting is reconstructed on sample.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Nested-dict helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flatten(d: Dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """Recursively flatten nested dict into ``{"a/b/c": array}`` form."""
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        else:
            out[key] = np.asarray(v)
    return out


def _unflatten(flat: Dict[str, Any]) -> Dict:
    """Reconstruct nested dict from flat ``{"a/b/c": value}`` form."""
    result: Dict = {}
    for dotkey, v in flat.items():
        parts = dotkey.split("/")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DictBuffer
# ──────────────────────────────────────────────────────────────────────────────

class DictBuffer:
    """Circular replay buffer that stores and returns arbitrary nested dicts.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to hold.
    device : str
        Device for tensors returned by ``sample()``.

    Usage
    -----
    >>> buf = DictBuffer(capacity=10_000, device="cuda")
    >>> buf.extend({"observation": obs_arr, "action": act_arr,
    ...             "next": {"observation": nobs_arr, "terminated": term_arr}})
    >>> batch = buf.sample(256)   # same nested structure, as tensors
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self._size: int = 0
        self._ptr: int = 0
        self._storage: Dict[str, np.ndarray] = {}  # flat storage

    # ------------------------------------------------------------------ #
    # Adding data
    # ------------------------------------------------------------------ #

    def extend(self, data: Dict) -> None:
        """Add a batch of transitions.  ``data`` may contain nested dicts."""
        flat = _flatten(data)
        n = len(next(iter(flat.values())))

        if n > self.capacity:
            # Keep only the most recent `capacity` entries
            flat = {k: v[-self.capacity:] for k, v in flat.items()}
            n = self.capacity

        # Lazy initialisation on first extend
        if not self._storage:
            for k, v in flat.items():
                self._storage[k] = np.zeros(
                    (self.capacity, *v.shape[1:]), dtype=v.dtype
                )

        # Ring-buffer write
        end = self._ptr + n
        if end <= self.capacity:
            for k, v in flat.items():
                self._storage[k][self._ptr:end] = v
        else:
            first = self.capacity - self._ptr
            for k, v in flat.items():
                self._storage[k][self._ptr:] = v[:first]
                self._storage[k][: n - first] = v[first:]

        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample *batch_size* transitions uniformly.  Returns nested dict of tensors.

        Uses torch.randint (not np.random) so RNG consumption matches td_jepa's
        DictBuffer.sample exactly. With the same torch.manual_seed(seed), this
        produces the same sequence of batch indices as td_jepa.
        """
        assert self._size > 0, "Buffer is empty"
        idx = torch.randint(0, self._size, (batch_size,)).numpy()
        flat_tensors = {
            k: torch.as_tensor(v[idx]).to(self.device)
            for k, v in self._storage.items()
        }
        return _unflatten(flat_tensors)

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity