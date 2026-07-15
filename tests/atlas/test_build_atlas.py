"""center_and_standardize is pure numpy (runs under .venv OR jax venv)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.build_atlas import center_and_standardize


def test_centering_zeroes_mu_mean_and_frobenius_is_one():
    rng = np.random.RandomState(0)
    Q = rng.randn(5, 3).astype(np.float64)        # n_a=5 actions, n_g=3 goals
    Qmu = rng.randn(9, 3)                          # 9 mu-center samples
    A = center_and_standardize(Q, Qmu)
    assert A.shape == (5, 3)
    assert abs(np.linalg.norm(A) - 1.0) < 1e-9


def test_centering_subtracts_per_goal_mu_mean():
    Q = np.array([[2.0, 5.0]])                     # 1 action, 2 goals
    Qmu = np.array([[1.0, 4.0], [1.0, 4.0]])       # mu mean = [1, 4]
    A = center_and_standardize(Q, Qmu)             # pre-norm = [[1, 1]]
    assert np.allclose(A, np.array([[1.0, 1.0]]) / np.sqrt(2))
