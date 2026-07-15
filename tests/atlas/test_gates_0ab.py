import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.gates import whitened_dist


def test_whitened_dist_near_zero_on_cloud_point():
    rng = np.random.RandomState(0)
    cloud = rng.randn(50, 5)
    d = whitened_dist(cloud[0], cloud, lam=0.1, eps=1e-4)
    assert d >= 0.0 and d < 1e-6


def test_whitened_dist_grows_off_cloud():
    rng = np.random.RandomState(0)
    cloud = rng.randn(50, 5)
    far = np.ones(5) * 10.0
    assert whitened_dist(far, cloud, 0.1, 1e-4) > whitened_dist(cloud[3], cloud, 0.1, 1e-4)
