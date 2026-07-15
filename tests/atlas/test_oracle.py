"""Run under the jax venv (run_jax_tests.py). Small pool/H for speed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
pytest.importorskip('jax')  # jax-venv only; skipped under repo .venv


from scripts.atlas.banks import load_banks
from scripts.atlas.jax_load import CRITICS
from scripts.atlas.mu_flow import load_mu
from scripts.atlas.oracle import goal_cube_xyz, make_pool, oracle_J
from scripts.atlas.prefix_replay import PrefixState


def _ps_from_banks(b, p, k):
    L = int(b['state_prefix_len'][p, k])
    return PrefixState(seed=int(p * 32 + k) % 5000, task_id=1,
                       prefix=b['state_prefix_actions'][p, k, :L])


def test_goal_cube_extraction_roundtrips():
    # obs[19:22] = (cube - center)*10 ; recover cube
    cube = np.array([0.5, -0.2, 0.06], np.float32)
    obs = np.zeros(28, np.float32)
    obs[19:22] = (cube - np.array([0.425, 0.0, 0.0])) * 10.0
    assert np.allclose(goal_cube_xyz(obs), cube, atol=1e-6)


def test_oracle_scalar_and_deterministic():
    b = load_banks('analysis/atlas/banks.npz')
    mu = load_mu(*CRITICS['bilinear_flow'])
    pool = make_pool(6)
    # use the P0 reset state (empty prefix) so replay is just a reset
    ps = PrefixState(seed=0, task_id=1, prefix=b['state_prefix_actions'][0, 0, :0])
    a0 = mu.sample(b['state_obs'][0, 0][None], n=1, seed=1)[0, 0]
    goal = b['goals_eval'][0]
    j1 = oracle_J(pool, mu, ps, a0, goal, replicates=2, H=8)
    j2 = oracle_J(pool, mu, ps, a0, goal, replicates=2, H=8)
    assert np.isfinite(j1)
    assert j1 == j2, f"oracle must be deterministic: {j1} vs {j2}"
    assert j1 <= 0.0, "discounted pre-hit cost is non-positive"
