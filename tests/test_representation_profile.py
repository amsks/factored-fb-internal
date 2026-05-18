import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import representation_profile as rp


class _StubModel:
    """B = first z_dim obs dims; F = (obs0 + action0) broadcast, 2 heads."""
    z_dim = 3

    def backward_map(self, obs):
        return obs[:, : self.z_dim].clone()

    def forward_map(self, obs, z, action):
        base = (obs[:, 0:1] + action[:, 0:1]).expand(-1, self.z_dim)  # [B,zd]
        return torch.stack([base, base + 1.0], dim=0)                 # [2,B,zd]


class _StubAgent:
    device = "cpu"

    def act(self, obs, z):
        return torch.zeros((obs.shape[0], 2))


def test_ensemble_q_means_over_heads():
    F = torch.tensor([[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])  # [2,1,3]
    z = torch.tensor([2.0, 0.0, 0.0])
    q = rp._ensemble_q(F, z)
    assert q.shape == (1,)
    assert q[0] == 2.0 * ((1.0 + 3.0) / 2)  # mean over heads of (F·z)


def test_q_values_monotone_in_action():
    m = _StubModel()
    obs = torch.zeros((4, 5))
    z = torch.ones(3)
    a_lo = torch.zeros((4, 2))
    a_hi = torch.ones((4, 2))
    q_lo = rp.q_values(m, obs, a_lo, z)
    q_hi = rp.q_values(m, obs, a_hi, z)
    assert q_lo.shape == (4,)
    assert np.all(q_hi > q_lo)


def test_v_values_uses_policy_action():
    m, ag = _StubModel(), _StubAgent()
    obs = torch.zeros((3, 5))
    z = torch.ones(3)
    v = rp.v_values(m, ag, obs, z)
    # policy action is zeros => same as q_values with zero action
    assert np.allclose(v, rp.q_values(m, obs, torch.zeros((3, 2)), z))
