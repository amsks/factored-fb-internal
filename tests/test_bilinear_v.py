"""bilinear_v: replace the FB-IQL critic's MLP value head V=MLP([s,z]) with a bilinear
V=<phi_V(s), z>/sqrt(d), so V matches the bilinear Q's form (consistency for the IQL
bootstrap). Used on the learned-B / decondition / d_norm bridge runs."""
import numpy as np
import torch
from gymnasium.spaces import Box

from agents.fb.agent import FBAgent


def _agent(**kw):
    obs_space = Box(low=-1, high=1, shape=(12,), dtype=np.float32)
    return FBAgent(obs_space=obs_space, action_dim=4, batch_size=8,
                   z_dim=16, L_dim=16, device="cpu", critic_mode="iql", **kw)


def test_bilinear_v_is_inner_product():
    ag = _agent(bilinear_v=True, d_norm=True)
    obs, z = torch.randn(8, 12), torch.randn(8, 16)
    expected = (ag._value_phi(obs) * z).sum(-1) * ag._value_scale
    assert torch.allclose(ag._V(obs, z), expected)


def test_mlp_v_is_default():
    ag = _agent(bilinear_v=False)
    obs, z = torch.randn(8, 12), torch.randn(8, 16)
    expected = ag._value_mlp(torch.cat([obs, z], dim=-1)).squeeze(-1)
    assert torch.allclose(ag._V(obs, z), expected)
