"""Tests for spec 28 P0b/P0c pure logic (scripts/rescuing_fb/spec28_encodings.py).

Jax-free: no checkpoint, no mujoco, no GPU. Run with
`.venv/bin/python -m pytest tests/test_spec28_encodings.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from spec28_encodings import (  # noqa: E402
    auc_hit,
    coord_percentile,
    family_mean_aggregate,
    hard_negatives,
    margin_std,
    moment_index,
    reconstruct,
    ridge_index,
    ridge_lambdas,
    score_encoding,
    spearman,
    topk_precision,
)


# --------------------------------------------------------------- rank stats --

def test_spearman_monotone_and_antitone():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_matches_scipy_like_reference_with_ties():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=200).astype(float)   # heavy ties
    b = a + rng.normal(size=200)
    # Pearson on average ranks is the definition Spearman-with-ties uses.
    ra = np.argsort(np.argsort(a)).astype(float)
    del ra  # (argsort ranks break ties arbitrarily; only the module's is correct)
    rho = spearman(a, b)
    assert 0.5 < rho < 1.0


def test_auc_perfect_and_chance():
    hit = np.array([1, 1, 0, 0], bool)
    auc, n = auc_hit(np.array([9.0, 8.0, 1.0, 0.0]), hit)
    assert (auc, n) == (1.0, 2)
    auc, _ = auc_hit(np.array([0.0, 1.0, 9.0, 8.0]), hit)
    assert auc == 0.0
    auc, _ = auc_hit(np.ones(4), hit)          # all ties -> exactly 0.5
    assert auc == pytest.approx(0.5)


def test_auc_none_when_a_class_is_empty():
    auc, n = auc_hit(np.arange(5.0), np.zeros(5, bool))
    assert auc is None and n == 0


def test_topk_precision_bounds():
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert topk_precision(s, s, 3) == 1.0
    assert topk_precision(s, -s, 2) == 0.0


def test_hard_negatives_picks_nearest_non_hits():
    dist = np.array([0.0, 0.1, 0.5, 0.9, 2.0])
    hit = np.array([1, 1, 0, 0, 0], bool)
    m = hard_negatives(dist, hit, 2)
    assert m.tolist() == [False, False, True, True, False]


def test_margin_std_scale_invariant():
    rng = np.random.default_rng(1)
    s = rng.normal(size=50)
    hit = np.zeros(50, bool); hit[:5] = True
    hard = np.zeros(50, bool); hard[5:15] = True
    assert margin_std(s, hit, hard) == pytest.approx(margin_std(7.3 * s, hit, hard))


# ---------------------------------------------------------------- encodings --

def test_moment_index_is_the_reward_weighted_pool_mean():
    b = np.array([[1.0, 0.0], [0.0, 2.0]])
    r = np.array([1.0, 3.0])
    assert moment_index(b, r) == pytest.approx(np.array([0.5, 3.0]))


def test_ridge_index_matches_the_explicit_solve():
    rng = np.random.default_rng(2)
    b = rng.normal(size=(64, 5))
    r = rng.normal(size=64)
    lam = 0.37
    sigma = b.T @ b / 64
    want = np.linalg.solve(sigma + lam * np.eye(5), moment_index(b, r))
    assert ridge_index(b, r, lam) == pytest.approx(want)


def test_ridge_reduces_to_the_moment_direction_under_a_whitened_basis():
    """The mechanism claim, stated as a test: when Sigma_B = I (the whitened /
    orthonormal geometry FB's ortho loss enforces) the dual coefficient IS the
    primal moment up to a positive scalar -- so primal-vs-dual can only matter
    for a NON-white B, which is exactly the regime spec 28 is probing."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    b = q[:8] * np.sqrt(8)                       # Sigma_B = B^T B / 8 = I
    assert b.T @ b / 8 == pytest.approx(np.eye(8), abs=1e-9)
    r = rng.normal(size=8)
    m = moment_index(b, r)
    c = ridge_index(b, r, 1e-8)
    cos = float(m @ c / (np.linalg.norm(m) * np.linalg.norm(c)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_ridge_lambdas_scale_with_mean_eigenvalue():
    b = np.eye(4) * 3.0
    lams = ridge_lambdas(b, mults=(1e-2,))
    # Sigma = 9/4 * I -> tr/d = 9/4
    assert lams[1e-2] == pytest.approx(1e-2 * 9.0 / 4.0)


def test_normalized_and_unnormalized_moment_score_identically():
    """Every P0b metric is invariant to a positive rescaling of z, so the
    spec's "un-normalized variant" is a CONTROL, not a new arm: any difference
    between point/moment and ridge cannot be attributed to normalization."""
    rng = np.random.default_rng(4)
    b = rng.normal(size=(300, 6))
    kg = rng.random(300)
    hit = kg > np.quantile(kg, 0.9)
    dist = 1.0 - kg
    m = moment_index(b, kg)
    m_norm = m * np.sqrt(6) / np.linalg.norm(m)
    a = score_encoding(b, m, kg, hit, dist)
    c = score_encoding(b, m_norm, kg, hit, dist)
    for key in ("auc", "spearman_kg", "margin_std"):
        assert a[key] == pytest.approx(c[key])
    assert a["topk_prec"] == c["topk_prec"]
    assert a["z_norm"] != pytest.approx(c["z_norm"])


def test_ridge_beats_the_moment_on_a_deliberately_anisotropic_basis():
    """End-to-end sanity for the repair hypothesis on synthetic data with a
    KNOWN answer: r is exactly linear in B, but B has a badly conditioned
    covariance. The primal moment then reconstructs r poorly and the dual
    solve recovers it."""
    rng = np.random.default_rng(5)
    d, P = 8, 4000
    scale = np.geomspace(1.0, 40.0, d)
    b = rng.normal(size=(P, d)) * scale[None]
    w_true = rng.normal(size=d)
    r = b @ w_true
    r = (r - r.min()) / (r.max() - r.min())      # kg-like, non-negative
    hit = r > np.quantile(r, 0.98)
    dist = 1.0 - r
    m = score_encoding(b, moment_index(b, r), r, hit, dist)
    lam = ridge_lambdas(b, (1e-3,))[1e-3]
    g = score_encoding(b, ridge_index(b, r, lam), r, hit, dist)
    assert g["spearman_kg"] > m["spearman_kg"] + 0.05
    assert g["auc"] > m["auc"]


def test_reconstruct_is_the_plain_bilinear_readout():
    b = np.arange(6.0).reshape(3, 2)
    z = np.array([1.0, -1.0])
    assert reconstruct(b, z) == pytest.approx(b @ z)


# ------------------------------------------------------- coordinate percentile --

def test_coord_percentile_best_chance_worst():
    coords = np.array([[0.0], [1.0], [2.0], [3.0]])
    ctr = np.array([0.0])
    assert coord_percentile(coords, ctr, 0) == 0.0
    assert coord_percentile(coords, ctr, 2) == 0.5
    assert coord_percentile(coords, ctr, 3) == 0.75


def test_coord_percentile_random_selection_is_chance_in_expectation():
    rng = np.random.default_rng(6)
    coords = rng.normal(size=(2000, 3))
    ctr = np.zeros(3)
    vals = [coord_percentile(coords, ctr, int(i)) for i in rng.integers(0, 2000, 400)]
    assert abs(float(np.mean(vals)) - 0.5) < 0.05


def test_family_mean_aggregate_is_mean_of_family_means_not_of_cells():
    per_cell = {"dirac": {"t1": 0.0, "t2": 0.0, "t3": 0.0, "t4": 0.0},
                "region": {"c0": 1.0}}
    assert family_mean_aggregate(per_cell) == pytest.approx(0.5)
    assert family_mean_aggregate(per_cell) != pytest.approx(0.2)


def test_family_mean_aggregate_reproduces_the_archived_spec27_percentiles():
    """The archived per-cell percentiles of results/spec27_q2/e2b_z25_cs_actor
    .json and e2b_z25_cs_full.json (recomputed from their `top1_idx` against a
    from-scratch pool draw) aggregate to the quoted 0.35 / 0.47 under THIS
    aggregate and to 0.343 / 0.469 under a flat cell mean -- which is how the
    metric definition was pinned before any spec-28 number was read."""
    actor = {
        "dirac": {"t1": 0.1450, "t2": 0.4909, "t3": 0.8807, "t4": 0.0308, "t5": 0.3196},
        "dense": {"t1": 0.1450, "t2": 0.7498, "t3": 0.0747, "t4": 0.0308, "t5": 0.3196},
        "region": {"c0": 0.2686, "c1": 0.6709, "c2": 0.1750, "c3": 0.4968},
    }
    full = {
        "dirac": {"t1": 0.7449, "t2": 0.2769, "t3": 0.1780, "t4": 0.2522, "t5": 0.4229},
        "dense": {"t1": 0.5308, "t2": 0.9927, "t3": 0.3840, "t4": 0.2522, "t5": 0.4229},
        "region": {"c0": 0.7712, "c1": 0.5679, "c2": 0.1213, "c3": 0.6541},
    }
    assert round(family_mean_aggregate(actor), 2) == 0.35
    assert round(family_mean_aggregate(full), 2) == 0.47
    flat_actor = float(np.mean([v for f in actor.values() for v in f.values()]))
    assert round(flat_actor, 2) == 0.34


# ------------------------------------------------------------ P0c control gate --

def test_gate_top1_exact_passes_only_on_identical_indices():
    from eval_spec28_p0c import gate_top1_exact

    arch = {"dirac/t1": 5, "dense/t1": 9}
    ok, rows = gate_top1_exact({"dirac/t1": 5, "dense/t1": 9}, arch)
    assert ok and all(r["ok"] for r in rows) and len(rows) == 2
    ok, rows = gate_top1_exact({"dirac/t1": 5, "dense/t1": 8}, arch)
    assert not ok
    assert {r["cell"]: r["ok"] for r in rows} == {"dirac/t1": True, "dense/t1": False}


def test_gate_top1_exact_refuses_a_disjoint_comparison():
    from eval_spec28_p0c import gate_top1_exact

    with pytest.raises(AssertionError):
        gate_top1_exact({"dirac/t1": 5}, {"region/c0": 5})


# ----------------------------------------------------- P0b instrument control --

def test_oracle_and_random_bracket_the_scale_through_the_same_path():
    """The P0b instrument gate, as a unit test: `score_rhat` fed the TARGET must
    saturate every metric, and fed noise must read chance. Any encoding verdict
    lives strictly between these two readings."""
    from spec28_encodings import score_rhat

    rng = np.random.default_rng(7)
    P = 3000
    dist = rng.random(P) * 5
    kg = np.exp(-dist)
    kg = kg / kg.sum()
    hit = dist < 0.4
    orc = score_rhat(kg, kg, hit, dist)
    assert orc["auc"] == pytest.approx(1.0)
    assert orc["spearman_kg"] == pytest.approx(1.0)
    assert orc["topk_prec"]["64"] == 1.0
    assert orc["margin_std"] > 0.0
    noise = score_rhat(rng.normal(size=P), kg, hit, dist)
    assert abs(noise["auc"] - 0.5) < 0.1
    assert abs(noise["spearman_kg"]) < 0.15
    assert noise["topk_prec"]["64"] < 0.05


def test_score_encoding_delegates_to_score_rhat():
    from spec28_encodings import reconstruct, score_encoding, score_rhat

    rng = np.random.default_rng(8)
    b = rng.normal(size=(200, 4))
    z = rng.normal(size=4)
    kg = rng.random(200)
    hit = kg > 0.95
    dist = 1 - kg
    a = score_encoding(b, z, kg, hit, dist)
    c = score_rhat(reconstruct(b, z), kg, hit, dist)
    assert a["auc"] == c["auc"] and a["spearman_kg"] == c["spearman_kg"]
    assert a["z_norm"] == pytest.approx(float(np.linalg.norm(z)))
