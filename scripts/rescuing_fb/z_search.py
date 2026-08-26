"""Spec 31 — zero-order z-search core for Density-FB.

Pure numpy at module level (importable from the torch test venv); jax is
imported lazily inside make_weights_fn only. The min-of-softmax-heads
weight computation here is THE shared implementation: the g293 oracle and
the spec-31 harness both call it, so their numerics agree by construction.
"""
from __future__ import annotations

import numpy as np


def sphere_project(z: np.ndarray) -> np.ndarray:
    """Scale rows to the standing ||z|| = sqrt(d) invariant (d = last axis)."""
    z = np.asarray(z, np.float64)
    d = z.shape[-1]
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return np.sqrt(d) * z / np.maximum(n, 1e-12)


def sphere_samples(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return sphere_project(rng.standard_normal((n, d)))


def min_head_weights(f_emb: np.ndarray, b: np.ndarray, temp: float,
                     renorm: bool = False) -> np.ndarray:
    """f_emb [H, S, d], b [M, d] -> [S, M]: per-head softmax over M, then
    elementwise MIN over heads (the actor/eval convention — sub-normalized
    whenever heads disagree, spec 30 Item 4). renorm=True re-normalizes rows
    (the convention the affine-invariance assertion is exact under)."""
    logits = np.einsum("hbd,md->hbm", np.asarray(f_emb, np.float64),
                       np.asarray(b, np.float64)) / temp
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(-1, keepdims=True)
    w = w.min(0)
    if renorm:
        w = w / w.sum(-1, keepdims=True)
    return w
