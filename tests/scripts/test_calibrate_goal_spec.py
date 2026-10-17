"""Goal-spec success radius calibration from the effective support of the softmax-normalised goal reward."""

import numpy as np
import pytest

from scripts.calibrate_goal_spec import calibrate_support, effective_support


def _blob(n, d, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_support_limits():
    x = _blob(4096, 8)
    # Huge radius: every candidate gets 1/1024.
    assert effective_support(
        x, eps=1e6, n_cand=1024, n_goals=16, seed=0
    ) == pytest.approx(1024.0, rel=1e-3)
    # Tiny radius: all mass on the nearest candidate.
    assert effective_support(
        x, eps=1e-6, n_cand=1024, n_goals=16, seed=0
    ) == pytest.approx(1.0, abs=1e-6)


def test_support_monotone_in_radius():
    x = _blob(4096, 8)
    s = [
        effective_support(x, eps=e, n_cand=1024, n_goals=32, seed=0)
        for e in (0.3, 1.0, 3.0)
    ]
    assert s[0] < s[1] < s[2]


def test_calibrate_hits_target():
    x = _blob(4096, 8)
    target = 60.0
    eps = calibrate_support(x, target, n_cand=1024, n_goals=32, seed=0)
    assert effective_support(x, eps, n_cand=1024, n_goals=32, seed=0) == pytest.approx(
        target, rel=0.02
    )
