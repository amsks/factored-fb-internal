"""Coverage-balanced FB: consistent goal-side rho' reweight.
See docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md.

The same goal-agnostic measure rho' (inverse-density over offline cube xyz, never
eval/task goals) is applied to EVERY goal-side term — successor off-diag + diag AND
the orthonormality whitening — so the model stays a valid FB in L2(rho'). Default-off
(reweight_alpha=0 -> w=None) is byte-identical to current FB.
"""
import inspect

import numpy as np
import torch

from agents.fb.agent import coverage_weights, fb_successor_terms, ortho_cov


def _density():
    """2x2x2 cube-xyz density: cell (0,0,0) common, (1,1,1) rare."""
    H = torch.zeros(2, 2, 2)
    H[0, 0, 0] = 0.9
    H[1, 1, 1] = 0.001
    H = H / H.sum()
    edges = [torch.tensor([0.0, 0.5, 1.0]) for _ in range(3)]
    return (H, edges)


# --- (1) BYTE-IDENTICAL: w=None == w=ones for every reweighted term ---------------

def test_successor_terms_ones_equal_none():
    torch.manual_seed(0)
    diff = torch.randn(3, 5, 5)
    off_diag = 1.0 - torch.eye(5)
    off_diag_sum = off_diag.sum()
    o0, d0 = fb_successor_terms(diff, off_diag, off_diag_sum, None)
    o1, d1 = fb_successor_terms(diff, off_diag, off_diag_sum, torch.ones(5))
    assert torch.allclose(o0, o1) and torch.allclose(d0, d1)


def test_ortho_cov_ones_equal_none():
    torch.manual_seed(0)
    B = torch.randn(6, 4)
    assert torch.allclose(ortho_cov(B, None), ortho_cov(B, torch.ones(6)))


# --- (2) WEIGHT VALIDITY: rare>common, mean~=1, finite, clip binds ----------------

def test_coverage_weights_rare_gets_more_weight():
    cube = torch.tensor([[0.25, 0.25, 0.25],   # common cell
                         [0.75, 0.75, 0.75]])  # rare cell
    w = coverage_weights(cube, _density(), alpha=1.0, clip=10.0)
    assert torch.isfinite(w).all()
    assert w[1] > w[0]
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_coverage_weights_alpha_zero_is_uniform():
    cube = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    w = coverage_weights(cube, _density(), alpha=0.0, clip=10.0)
    assert torch.allclose(w, torch.ones_like(w))


def test_coverage_weights_clip_bounds_rare_advantage():
    common, rare = [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]
    cube = torch.tensor([common, common, common, common, rare])  # majority-common -> clip binds
    tight = coverage_weights(cube, _density(), alpha=1.0, clip=1.5)
    loose = coverage_weights(cube, _density(), alpha=1.0, clip=1e6)
    assert (tight[-1] / tight[0]) < (loose[-1] / loose[0])


# --- (3) ORTHONORMALITY IS REWEIGHTED (the centerpiece) ---------------------------

def test_ortho_cov_is_weighted_second_moment():
    """ortho_cov(B,w) == (sqrt(w)B)(sqrt(w)B)^T, so its trace term is the rho'-weighted
    second moment, and it differs from the unweighted Cov when w != 1."""
    torch.manual_seed(1)
    B = torch.randn(6, 4)
    w = torch.rand(6) + 0.5
    w = w / w.mean()
    Cov = ortho_cov(B, w)
    assert torch.allclose(Cov.diag().mean(), (w * (B * B).sum(-1)).mean(), atol=1e-6)
    assert not torch.allclose(Cov, ortho_cov(B, None))


def test_successor_diag_reweights_when_structured():
    torch.manual_seed(2)
    diff = torch.randn(2, 4, 4)
    off_diag = 1.0 - torch.eye(4)
    s = off_diag.sum()
    w = torch.tensor([4.0, 0.0, 0.0, 0.0])  # mean 1, all weight on goal 0
    _, d_w = fb_successor_terms(diff, off_diag, s, w)
    _, d_none = fb_successor_terms(diff, off_diag, s, None)
    assert not torch.allclose(d_w, d_none)


# --- (4) GOAL-AGNOSTIC -------------------------------------------------------------

def test_coverage_weights_is_goal_agnostic():
    params = set(inspect.signature(coverage_weights).parameters)
    assert not (params & {"goal", "goals", "task", "z", "reward"})


# --- __init__ + method wiring (default-off; reads only physics+density) ------------

def _tiny_state_agent(**extra):
    import gymnasium
    from agents.fb.agent import FBAgent
    from nn_models import IdentityNNConfig
    from normalizers import IdentityNormalizerConfig
    return FBAgent(
        obs_space=gymnasium.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu",
        **extra,
    )


def test_agent_reweight_default_off():
    agent = _tiny_state_agent()
    assert agent.reweight_alpha == 0.0
    assert agent._rho is None
    assert agent._coverage_weights({"next": {"physics": torch.rand(4, 20)}}) is None


def test_agent_coverage_weights_reads_physics_when_alpha_positive(tmp_path):
    p = tmp_path / "cube_density.npz"
    H, edges = _density()
    np.savez(p, H=H.numpy(), ex=edges[0].numpy(), ey=edges[1].numpy(), ez=edges[2].numpy())
    agent = _tiny_state_agent(reweight_alpha=1.0, reweight_density_path=str(p))
    assert agent._rho is not None
    batch = {"next": {"physics": torch.rand(8, 20)}}  # cube xyz at [14:17]
    w = agent._coverage_weights(batch)
    assert w is not None and w.shape == (8,)
    assert torch.isfinite(w).all() and abs(w.mean().item() - 1.0) < 1e-5
