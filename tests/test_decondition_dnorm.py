"""decondition_F: feed zeros to the forward map's task slot so F(s,a,z) -> phi(s,a)
(task-agnostic encoder; bilinear-GCIQL direction). d_norm: scale the bilinear value
F.z by 1/sqrt(d) so its magnitude is latent-dim-independent (GCIQL's phi.psi/sqrt(d))."""
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from agents.fb.agent import FBAgent


def _agent(**kw):
    obs_space = Box(low=-1, high=1, shape=(12,), dtype=np.float32)
    return FBAgent(obs_space=obs_space, action_dim=4, batch_size=8,
                   z_dim=16, L_dim=16, device="cpu", **kw)


# ---- I3: decondition_F ----
def test_decondition_F_feeds_zeros_to_forward_map():
    ag = _agent(decondition_F=True)
    z = torch.randn(8, 16)
    assert torch.allclose(ag._fb_z(z), torch.zeros_like(z))


def test_default_passes_z_to_forward_map():
    ag = _agent(decondition_F=False)
    z = torch.randn(8, 16)
    assert torch.allclose(ag._fb_z(z), z)


# ---- I4: d_norm ----
def test_d_norm_scales_value_by_inv_sqrt_d():
    ag = _agent(d_norm=True)            # z_dim=16 -> sqrt(16)=4
    assert ag._value_scale == pytest.approx(1.0 / 4.0)


def test_no_d_norm_scale_is_one():
    ag = _agent(d_norm=False)
    assert ag._value_scale == 1.0


def test_d_norm_scales_bilinear_Q_heads_by_inv_sqrt_d():
    ag_off = _agent(d_norm=False)
    ag_on = _agent(d_norm=True)
    ag_on.model.load_state_dict(ag_off.model.state_dict())  # only the scale differs
    obs, a, z = torch.randn(8, 12), torch.randn(8, 4), torch.randn(8, 16)
    q_off = ag_off._fb_Q_heads(ag_off.model._left_encoder(obs), z, a, ag_off.model._forward_map)
    q_on = ag_on._fb_Q_heads(ag_on.model._left_encoder(obs), z, a, ag_on.model._forward_map)
    assert torch.allclose(q_on, q_off / 4.0, atol=1e-5)


def test_decondition_F_zeros_the_forward_map_task_input_in_Q_heads():
    # With decondition_F, Q must not depend on the value of z fed to F (only via the readout).
    ag = _agent(decondition_F=True)
    obs, a = torch.randn(8, 12), torch.randn(8, 4)
    z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
    left = ag.model._left_encoder(obs)
    # F sees zeros either way => F(s,a) identical; only the explicit readout z differs.
    f1 = ag.model._forward_map(left, ag._fb_z(z1), a)
    f2 = ag.model._forward_map(left, ag._fb_z(z2), a)
    assert torch.allclose(f1, f2)
