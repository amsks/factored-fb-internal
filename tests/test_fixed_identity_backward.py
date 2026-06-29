import torch
from gymnasium.spaces import Box
import numpy as np

from agents.fb.fixed_backward import FixedIdentityBackward
from agents.fb.model import FBModel


def test_identity_passes_obs_through_and_has_no_params():
    b = FixedIdentityBackward(obs_dim=29)
    obs = torch.randn(4, 29)
    out = b(obs)
    assert out.shape == (4, 29)
    assert torch.allclose(out, obs)            # B(obs) = obs exactly
    assert list(b.parameters()) == []          # parameter-free
    assert b.output_space.shape == (29,)


def test_model_identity_sets_zdim_to_obsdim_and_disables_norm_z():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
    m = FBModel(obs_space=obs_space, action_dim=5, z_dim=50, fixed_b="identity")
    assert m.z_dim == 29                        # z_dim follows obs_dim, not the configured 50
    assert m.norm_z is False                    # z = g literally (no sphere projection)
    assert list(m._backward_map.parameters()) == []
    # project_z must be a no-op so z == g
    g = torch.randn(8, 29)
    assert torch.allclose(m.project_z(g), g)


def test_model_identity_norm_keeps_sphere_projection():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
    m = FBModel(obs_space=obs_space, action_dim=5, z_dim=50, fixed_b="identity_norm")
    assert m.z_dim == 29                        # still B=I => z_dim follows obs_dim
    assert m.norm_z is True                     # but z = sqrt(d)*g/||g|| (bounded)
    assert list(m._backward_map.parameters()) == []
    g = torch.randn(8, 29)
    zp = m.project_z(g)
    import math
    assert torch.allclose(zp.norm(dim=-1), torch.full((8,), math.sqrt(29)), atol=1e-4)
