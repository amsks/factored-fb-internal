"""Synthetic checks for evals/z_inference.py (the B3 inference ladder)."""
import math

import pytest
import torch

from evals.z_inference import infer_l0, infer_l1, infer_l2, project_sphere, value_auc


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))


def test_l0_closed_form():
    torch.manual_seed(0)
    B = torch.randn(100, 8)
    r = torch.randn(100, 1)
    assert torch.allclose(infer_l0(B, r), (r * B).sum(0), atol=1e-6)


def test_l1_equals_l0_direction_when_orthonormal():
    torch.manual_seed(0)
    # B with identity second moment: iid N(0, I) columns, large N
    B = torch.randn(200_000, 6)
    r = torch.randn(200_000, 1)
    z0 = infer_l0(B, r)
    z1 = infer_l1(B, r, ridge=1e-6)
    assert _cos(z0, z1) > 0.995


def test_l1_recovers_true_z_under_skewed_covariance():
    torch.manual_seed(1)
    d, n = 6, 100_000
    # skewed feature covariance C != I
    A = torch.randn(d, d)
    L = torch.linalg.cholesky(A @ A.T / d + 0.5 * torch.eye(d))
    B = torch.randn(n, d) @ L.T
    z_true = torch.randn(d)
    r = (B @ z_true).reshape(-1, 1)  # reward exactly linear in features
    z0 = infer_l0(B, r)
    z1 = infer_l1(B, r, ridge=1e-6)
    assert _cos(z1, z_true) > 0.999
    assert _cos(z1, z_true) > _cos(z0, z_true) + 0.01  # L0 is biased by C


def test_l2_improves_bellman_residual_and_matches_lstsq():
    torch.manual_seed(2)
    d, n = 8, 5000
    Phi = torch.randn(n, d)          # F(s,a) fixed features (z-independent F)
    Phi_n = torch.randn(n, d)        # F(s',a')
    z_true = project_sphere(torch.randn(d))
    gamma = 0.95 * torch.ones(n)
    # reward consistent with Q = Phi @ z_true
    r = Phi @ z_true - gamma * (Phi_n @ z_true)

    def q_pair(z):
        return Phi @ z, Phi_n @ z

    z0 = project_sphere(torch.randn(d))
    z2, info = infer_l2(z0, q_pair, r, gamma, steps=400, lr=0.1)
    assert info["residual_final"] < info["residual_init"]
    assert _cos(z2, z_true) > 0.99
    assert abs(float(z2.norm()) - math.sqrt(d)) < 1e-3  # stayed on the sphere


def test_value_auc_extremes():
    V = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pos = torch.tensor([0, 0, 1, 1])
    assert value_auc(V, pos) == pytest.approx(1.0)
    assert value_auc(-V, pos) == pytest.approx(0.0)
