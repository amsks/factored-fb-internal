"""Run under the jax venv: /dev/shm/.venv-jax/bin/python -m pytest tests/atlas/test_mu_flow.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.jax_load import CRITICS
from scripts.atlas.mu_flow import assert_q_independent, load_mu

CKPT, STEP = CRITICS['bilinear_flow']


def test_mu_is_goal_independent_and_shaped():
    mu = load_mu(CKPT, STEP)
    obs = np.zeros((3, 28), np.float32)
    a = mu.sample(obs, n=16, seed=0)
    assert a.shape == (3, 16, 5)
    assert np.abs(a).max() <= 1.0 + 1e-5


def test_mu_sample_is_deterministic_given_seed():
    mu = load_mu(CKPT, STEP)
    obs = np.zeros((2, 28), np.float32)
    a1 = mu.sample(obs, n=8, seed=7)
    a2 = mu.sample(obs, n=8, seed=7)
    assert np.array_equal(a1, a2)


def test_actor_vf_q_independent():
    assert assert_q_independent(CKPT, STEP) is True
