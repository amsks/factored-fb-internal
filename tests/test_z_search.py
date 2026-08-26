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


from z_search import cem_search  # noqa: E402


def _planted(d=16, seed=3):
    rng = np.random.default_rng(seed)
    z_true = sphere_samples(rng, 1, d)[0]
    weights_of = lambda Z: np.asarray(Z, np.float64)      # W == Z
    score_fn = lambda W: W @ z_true                       # max on sphere at z_true
    return z_true, weights_of, score_fn, rng


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_cem_recovers_planted_argmax():
    z_true, weights_of, score_fn, rng = _planted()
    z_star, diag = cem_search(score_fn, weights_of, 16, rng,
                              n_init=256, n_round=128, rounds=3)
    assert _cos(z_star, z_true) > 0.9
    assert diag["round_max"][-1] >= diag["round_max"][0]
    assert diag["J_star"] == max(diag["round_max"])


def test_seed_z_is_round0_candidate_zero():
    z_true, weights_of, score_fn, rng = _planted(seed=4)
    seed_z = sphere_samples(rng, 1, 16)[0]
    z_star, diag = cem_search(score_fn, weights_of, 16, rng, seed_z=seed_z,
                              n_init=64, n_round=32, rounds=1)
    np.testing.assert_allclose(diag["Z0"][0], seed_z, rtol=1e-12)
    assert diag["J_seed"] == diag["J0"][0]
    assert diag["cos_seed_star"] is not None
    # z* can never score below the seed (argmax over all evaluated)
    assert diag["J_star"] >= diag["J_seed"]


def test_all_rounds_stay_on_sphere():
    d = 16
    z_true, weights_of, _, rng = _planted(seed=5)
    seen = []
    def spy_weights(Z):
        seen.append(np.asarray(Z))
        return np.asarray(Z, np.float64)
    score_fn = lambda W: W @ z_true
    cem_search(score_fn, spy_weights, d, rng, n_init=64, n_round=32, rounds=2)
    assert len(seen) == 3                                 # round 0 + 2 refits
    for Z in seen:
        np.testing.assert_allclose(np.linalg.norm(Z, axis=-1),
                                   np.sqrt(float(d)), rtol=1e-6)


from z_search import affine_argmax_check  # noqa: E402


def test_affine_argmax_exact_under_renorm():
    rng = np.random.default_rng(6)
    K, M = 40, 25
    # sub-normalized rows with VARYING row sums (the min-convention regime)
    W = rng.random((K, M))
    W /= W.sum(-1, keepdims=True)
    W *= rng.uniform(0.5, 1.0, size=(K, 1))
    r = rng.standard_normal(M)
    out = affine_argmax_check(W, r, [(2.0, 0.0), (1.0, 3.0), (0.5, -7.0)])
    # renormalized: argmax_z (W@(a r + b)) == argmax_z (W@r) EXACTLY, a > 0
    assert out["renorm_all_match"] is True
    for cell in out["renorm"].values():
        assert cell["match"] is True
    # min convention: reported (bool present), not asserted true
    assert set(out["min_convention"]) == set(out["renorm"])
    assert isinstance(out["min_all_match"], bool)


def test_affine_argmax_base_is_plain_reward_argmax():
    rng = np.random.default_rng(7)
    W = rng.random((10, 6))
    r = rng.standard_normal(6)
    out = affine_argmax_check(W, r, [(1.0, 1.0)])
    Wn = W / W.sum(-1, keepdims=True)
    assert out["argmax_base"] == int(np.argmax(Wn @ r))
