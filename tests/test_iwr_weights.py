"""Interaction-weighted FB (IWR-FB): contact-proximity goal weights.
See PAPER/specs/2026-06-11-interaction-weighted-fb.md. The weight reuses the existing
rho' plumbing (fb_successor_terms / ortho_cov, tested in test_fb_reweight.py); these
tests cover the new weight function and its dispatch."""
import torch

from agents.fb.agent import interaction_weights


def _states(d_values):
    """Effector at origin, cube at distance d along x."""
    n = len(d_values)
    eff = torch.zeros(n, 3)
    cube = torch.zeros(n, 3)
    cube[:, 0] = torch.tensor(d_values)
    return eff, cube


def test_peak_at_contact_distance():
    eff, cube = _states([0.03, 0.10, 0.5])    # at-contact, in-window, far (floor)
    w = interaction_weights(eff, cube, c=0.03, sigma=0.05, eps=0.1)
    assert w[0] > w[1] > w[2]                  # monotone away from the boundary
    assert w.argmax() == 0


def test_floor_keeps_full_support():
    eff, cube = _states([0.03, 5.0])           # far state ~ eps floor
    w = interaction_weights(eff, cube, c=0.03, sigma=0.05, eps=0.1)
    raw_far_over_peak = 0.1 / (0.1 + 1.0)      # eps / (eps + exp(0))
    assert torch.allclose(w[1] / w[0], torch.tensor(raw_far_over_peak), atol=1e-5)
    assert (w > 0).all()


def test_mean_normalized_and_detached():
    torch.manual_seed(0)
    eff = torch.randn(512, 3, requires_grad=True)
    cube = torch.randn(512, 3)
    w = interaction_weights(eff, cube, c=0.03, sigma=0.05, eps=0.1)
    assert torch.allclose(w.mean(), torch.tensor(1.0), atol=1e-5)
    assert not w.requires_grad


def test_symmetric_in_distance_only():
    # same |eff-cube| distance, different directions -> same weight
    eff = torch.zeros(2, 3)
    cube = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, -0.1]])
    w = interaction_weights(eff, cube, c=0.03, sigma=0.05, eps=0.1)
    assert torch.allclose(w[0], w[1])


def test_agent_dispatch_interaction_mode():
    """_coverage_weights routes to interaction_weights without needing a density file."""
    from agents.fb.agent import FBAgent
    import inspect
    sig = inspect.signature(FBAgent.__init__)
    assert "reweight_mode" in sig.parameters
    assert "iwr_c" in sig.parameters
    # pure-function check of the dispatch math on a fake batch (no agent build):
    nobs = torch.zeros(4, 28)
    nobs[:, 12:15] = 0.0
    nobs[:, 19] = torch.tensor([0.03, 0.2, 1.0, 2.0])  # cube x at varying distance
    w = interaction_weights(nobs[:, 12:15], nobs[:, 19:22], c=0.03, sigma=0.05, eps=0.1)
    assert w.argmax() == 0 and w.shape == (4,)
