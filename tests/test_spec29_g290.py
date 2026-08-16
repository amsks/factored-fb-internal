"""Tests for spec 29 G29-0 pure logic (scripts/rescuing_fb/eval_spec29_g290.py).

Jax-free: no checkpoint, no mujoco, no GPU (the runner's heavy imports all live
inside `main`). Run with `.venv/bin/python -m pytest tests/test_spec29_g290.py`.

The point of the gate is a spread verdict, so these tests pin (a) that the
batched dual index IS the committed spec-28 `ridge_index` up to the sqrt(d)
renormalization, and (b) that the second spread statistic reads 1 on a collapsed
set and ~d on an isotropic one -- both ends of the instrument, the standing rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from eval_spec29_g290 import (  # noqa: E402
    dual_index_batch,
    participation_ratio,
    set_spread,
    sphere_reference,
)
from spec28_encodings import ridge_index, ridge_lambdas  # noqa: E402


def _pool(rng, P=512, d=8):
    return rng.normal(size=(P, d))


# ------------------------------------------------------- the dual index --

def test_dual_index_batch_matches_committed_ridge_index():
    """Row-for-row equality with spec 28's `ridge_index`, up to the sqrt(d)
    renormalization spec 29 sec.1 re-imposes."""
    rng = np.random.default_rng(0)
    b = _pool(rng)
    r = rng.random(size=(5, len(b)))
    lam = ridge_lambdas(b)[1e-2]
    got = dual_index_batch(b, r, lam)
    for i in range(len(r)):
        ref = ridge_index(b, r[i], lam)
        ref = ref * np.sqrt(b.shape[1]) / (np.linalg.norm(ref) + 1e-6)
        assert np.max(np.abs(ref - got[i])) < 1e-10


def test_dual_index_batch_norm_is_sqrt_d():
    rng = np.random.default_rng(1)
    b = _pool(rng, d=16)
    z = dual_index_batch(b, rng.random(size=(7, len(b))), 0.1)
    assert np.allclose(np.linalg.norm(z, axis=1), np.sqrt(16.0), atol=1e-4)


def test_dual_index_whitened_pool_agrees_with_primal_direction():
    """Under a whitened Sigma_B = I the dual and the primal moment point the
    same way -- the mechanism spec 28 sec.0 names, and the reason the collapse
    verdict can only differ when B is NOT whitened."""
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    b = q * np.sqrt(6.0)                     # Sigma_B = B^T B / P = I
    r = rng.random(size=(3, 6))
    z = dual_index_batch(b, r, 1e-9)
    m = r @ b / len(b)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    # tolerance is set by `_sphere`'s own +1e-6 denominator epsilon, not by the solve
    assert np.all(np.sum(m * z / np.sqrt(6.0), axis=1) > 1 - 1e-5)


def test_dual_index_rank1_pool_collapses():
    """The failure mode the gate is looking for: if Sigma_B is (near) rank one
    and lambda is large enough to dominate its null space, every reward row's
    dual coefficient is pulled onto the same ray -- so a de-collapsed reading is
    informative, not automatic."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=8)
    b = np.outer(rng.random(256) + 1.0, v)   # exactly rank 1
    r = rng.random(size=(64, 256))
    z = dual_index_batch(b, r, 1e-3)
    assert set_spread(z)["pr_second_moment"] == pytest.approx(1.0, abs=1e-6)


def test_dual_index_shape_guard():
    b = np.random.default_rng(4).normal(size=(32, 4))
    with pytest.raises(AssertionError):
        dual_index_batch(b, np.zeros((2, 31)), 0.1)


# ------------------------------------------------- the spread statistic --

def test_participation_ratio_flat_and_rank1():
    assert participation_ratio(np.ones(50)) == pytest.approx(50.0)
    assert participation_ratio(np.array([3.0] + [0.0] * 49)) == pytest.approx(1.0)


def test_set_spread_collapsed_set_reads_one():
    """All indices on one ray -> PR(second moment) = 1 even though the cosine
    statistic is also saturated: the two agree at the collapsed end."""
    z = np.outer(np.linspace(0.5, 1.5, 128), np.ones(20))
    s = set_spread(z)
    assert s["pr_second_moment"] == pytest.approx(1.0, abs=1e-9)
    assert s["top_eig_share_second_moment"] == pytest.approx(1.0, abs=1e-9)


def test_set_spread_sphere_reference_reads_near_d():
    d = 50
    z = sphere_reference(4096, d, np.random.default_rng(5))
    s = set_spread(z)
    assert s["norm_median"] == pytest.approx(np.sqrt(d), rel=1e-6)
    assert s["pr_second_moment"] > 0.9 * d
    assert s["pr_covariance"] > 0.9 * d


def test_set_spread_is_scale_free():
    """A global rescaling of the set must not move either PR -- the statistic is
    about direction spread, like the cosine it accompanies."""
    z = np.random.default_rng(6).normal(size=(256, 12))
    a, b = set_spread(z), set_spread(17.0 * z)
    assert a["pr_second_moment"] == pytest.approx(b["pr_second_moment"], rel=1e-9)
    assert a["pr_covariance"] == pytest.approx(b["pr_covariance"], rel=1e-9)


def test_set_spread_interpolates_between_the_ends():
    """A set spanning k directions reads PR ~ k, so intermediate (marginal)
    outcomes are legible rather than saturating."""
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.normal(size=(30, 30)))[0][:, :5]
    z = rng.normal(size=(2048, 5)) @ basis.T
    assert set_spread(z)["pr_second_moment"] == pytest.approx(5.0, rel=0.15)
