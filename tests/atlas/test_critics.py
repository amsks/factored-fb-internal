"""jax critics run under the jax venv (run_jax_tests.py); the FB critic runs under the
repo .venv separately (test_fb_critic, invoked by a torch-venv runner)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
pytest.importorskip('jax')  # jax-venv only; skipped under repo .venv


from scripts.atlas.critics import load_critic


def test_gciql_Q_shape():
    c = load_critic("bilinear_flow")
    S, na, ng = 2, 7, 3
    obs = np.zeros((S, 28), np.float32)
    acts = np.zeros((S, na, 5), np.float32)
    goals = np.zeros((ng, 28), np.float32)
    q = c.Q(obs, acts, goals)
    assert q.shape == (S, na, ng)
    assert np.isfinite(q).all()


def test_gciql_Q_varies_with_action():
    c = load_critic("bilinear_flow")
    obs = np.zeros((1, 28), np.float32)
    goals = np.zeros((1, 28), np.float32)
    acts = np.stack([np.zeros(5), np.ones(5) * 0.5])[None].astype(np.float32)  # [1,2,5]
    q = c.Q(obs, acts, goals)
    assert abs(q[0, 0, 0] - q[0, 1, 0]) > 1e-6, "Q should depend on the action"
