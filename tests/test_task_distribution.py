"""Unit tests for evals/task_distribution.py pure logic (synthetic data)."""

import numpy as np
import pytest

from evals.task_distribution import (
    INTERACTION_BUCKETS,
    bucket_histogram,
    coverage_score,
    estimate_table_z,
    label_interaction,
    participation_ratio,
    separability_auc,
)


def test_estimate_table_z_robust_to_lifts():
    # cube mostly at z=0.02 (resting), a few lifted high; table estimate ~0.02
    cube_z = np.concatenate([np.full(990, 0.02), np.linspace(0.1, 0.5, 10)])
    assert estimate_table_z(cube_z) == pytest.approx(0.02, abs=2e-3)


def test_label_interaction_ladder():
    eff = np.array([
        [0.0, 0.0, 0.0],   # hand far (dist 1.0 > reach)         -> no_contact
        [0.3, 0.0, 0.0],   # within reach_radius, beyond grasp   -> reaching
        [0.0, 0.0, 0.0],   # hand at cube + contact              -> grasped
        [0.0, 0.0, 0.0],   # cube lifted                         -> lifted_or_transport
    ])
    cube = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    lift = np.array([0.0, 0.0, 0.0, 0.10])
    contact = np.array([0.0, 0.0, 1.0, 1.0])
    lbl = label_interaction(eff, cube, lift, contact)
    assert list(lbl) == [0, 1, 2, 3]
    assert INTERACTION_BUCKETS[lbl[3]] == "lifted_or_transport"


def test_label_grasped_requires_contact():
    # hand at cube but no contact -> not grasped
    eff = np.array([[0.0, 0.0, 0.0]])
    cube = np.array([[0.0, 0.0, 0.0]])
    lift = np.array([0.0])
    assert label_interaction(eff, cube, lift, np.array([0.0]))[0] != 2
    assert label_interaction(eff, cube, lift, np.array([1.0]))[0] == 2


def test_bucket_histogram_sums_to_one():
    labels = np.array([0, 0, 1, 2, 3, 3])
    h = bucket_histogram(labels)
    assert set(h) == set(INTERACTION_BUCKETS)
    assert sum(h.values()) == pytest.approx(1.0)
    assert h["no_contact"] == pytest.approx(2 / 6)


def test_participation_ratio_isotropic_vs_rank1():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((5000, 10))
    assert participation_ratio(iso) == pytest.approx(10.0, rel=0.15)
    # rank-1: all variance in one direction
    rank1 = np.outer(rng.standard_normal(5000), np.ones(10))
    assert participation_ratio(rank1) == pytest.approx(1.0, abs=0.1)


def test_separability_auc_separated_vs_chance():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((500, 8)) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
    b = rng.standard_normal((500, 8)) - np.array([5, 0, 0, 0, 0, 0, 0, 0])
    Z = np.vstack([a, b])
    pos = np.array([True] * 500 + [False] * 500)
    assert separability_auc(Z, pos) > 0.99
    # identical distributions -> chance
    Z2 = rng.standard_normal((1000, 8))
    pos2 = rng.random(1000) < 0.5
    assert separability_auc(Z2, pos2) == pytest.approx(0.5, abs=0.08)


def test_separability_auc_empty_class():
    Z = np.random.default_rng(2).standard_normal((100, 4))
    assert separability_auc(Z, np.zeros(100, dtype=bool)) == 0.5


def test_coverage_score_in_vs_out():
    rng = np.random.default_rng(3)
    train = rng.standard_normal((2000, 6))
    in_q = rng.standard_normal((5, 6))            # same distribution
    out_q = np.full((5, 6), 20.0)                 # far OOD
    cov_in = coverage_score(train, in_q)
    cov_out = coverage_score(train, out_q)
    assert cov_out["mahalanobis"].mean() > cov_in["mahalanobis"].mean()
    # a near-duplicate of a training point has cos_nn ~ 1
    dup = train[:3] + 1e-6
    assert coverage_score(train, dup)["cos_nn"].min() > 0.99
