"""Spec 31 z-search core tests — pure numpy, torch venv, no jax/ckpt."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from z_search import min_head_weights, sphere_project, sphere_samples  # noqa: E402


def test_sphere_samples_norm_sqrt_d():
    rng = np.random.default_rng(0)
    z = sphere_samples(rng, 100, 50)
    assert z.shape == (100, 50)
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1),
                               np.sqrt(50.0), rtol=1e-6)


def test_sphere_project_idempotent_and_norm():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((7, 16)) * 3.0
    p = sphere_project(z)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1),
                               np.sqrt(16.0), rtol=1e-6)
    np.testing.assert_allclose(sphere_project(p), p, rtol=1e-6)


def test_min_head_weights_matches_manual_softmax():
    rng = np.random.default_rng(2)
    f = rng.standard_normal((2, 5, 8))
    b = rng.standard_normal((30, 8))
    temp = 0.7
    w = min_head_weights(f, b, temp)
    # manual twin (the g293 oracle inline computation)
    logits = np.einsum("hbd,md->hbm", f, b) / temp
    e = np.exp(logits - logits.max(-1, keepdims=True))
    sm = e / e.sum(-1, keepdims=True)
    np.testing.assert_allclose(w, sm.min(0), rtol=1e-10)
    # min of simplex rows is SUB-normalized when heads disagree
    assert (w.sum(-1) <= 1.0 + 1e-9).all()
    # renorm puts rows back on the simplex
    wr = min_head_weights(f, b, temp, renorm=True)
    np.testing.assert_allclose(wr.sum(-1), 1.0, rtol=1e-9)
