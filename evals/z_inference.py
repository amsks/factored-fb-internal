"""evals/z_inference.py — task-inference rules for FB/PSM-family models.

The standard FB rule  z = E_rho[r * B(s')]  is the L2(rho)-projection of the
reward onto span(B) *assuming* E[B B^T] = I and m = F^T B exactly. This module
implements the ladder of corrections studied in
PAPER/specs/2026-06-11-fb-psm-crl-gap-analysis-and-program.md (bottleneck B3):

  L0  z ∝ E_n[r B]                       — current rule (matches model.reward_inference)
  L1  z ∝ (E_n[B B^T] + λI)^{-1} E_n[rB] — GLS: exact L2-projection coefficients of r
                                            on span(B) under the empirical batch measure
  L2  z = argmin_z  E_n[(Q(s,a,z) − r − γ(1−T)·Q(s',a',z))²]  s.t. ||z|| = √d
                                          — self-consistent ("Bellman") inference:
                                            pick the task vector whose induced value is
                                            Bellman-consistent with the relabeled reward.

All functions are torch-only and side-effect free. L2 takes a closure
`q_pair_fn(z) -> (Q, Q_next)` so the caller owns the method-specific Q and the
(semi-gradient) next-action refresh policy.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch


def project_sphere(z: torch.Tensor) -> torch.Tensor:
    """Scale to the ||z|| = sqrt(d) sphere (mirrors model.project_z with norm_z=True)."""
    return math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1)


def infer_l0(B: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """z ∝ Σ_i r_i B_i (unnormalized; caller projects). B [N,d], r [N,1] or [N]."""
    r = r.reshape(-1, 1)
    return (r * B).sum(0)


def infer_l1(B: torch.Tensor, r: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """GLS coefficients of the L2-projection of r onto span(B): C^{-1} E[rB].

    Reduces to (a scalar multiple of) L0 when the empirical second moment
    C = E_n[B B^T] is the identity — i.e. when B is exactly orthonormal under
    the batch measure. `ridge` regularizes the solve (C is d x d, N >> d).
    """
    r = r.reshape(-1, 1)
    n, d = B.shape
    C = B.T @ B / n + ridge * torch.eye(d, dtype=B.dtype, device=B.device)
    b = (B * r).mean(0)
    return torch.linalg.solve(C, b)


def infer_l2(
    z0: torch.Tensor,
    q_pair_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    r: torch.Tensor,
    discount: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    project: bool = True,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Minimize the empirical Bellman residual over the single task vector z.

    q_pair_fn(z) must return (Q(s,a,z), Q(s',a',z)) as [N] tensors differentiable
    w.r.t. z; the caller decides how a' is produced (typically actor(s', z0),
    semi-gradient — Q_next should be a function of z but its actions fixed).
    discount: [N] per-sample γ(1−terminated). Projects z to the √d sphere each
    step when `project` (the trained actor only ever saw sphere z's).
    """
    r = r.reshape(-1)
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    res0 = res = None
    for _ in range(steps):
        Q, Qn = q_pair_fn(z)
        # residual-gradient flavor: z enters both sides; semi-gradient on actions only
        sq = (Q - (r + discount * Qn)).pow(2)
        res = (sq * sample_weight).mean() if sample_weight is not None else sq.mean()
        if res0 is None:
            res0 = float(res.detach())
        opt.zero_grad()
        res.backward()
        opt.step()
        if project:
            with torch.no_grad():
                z.copy_(project_sphere(z))
    return z.detach(), {"residual_init": float(res0), "residual_final": float(res.detach())}


def value_auc(V: torch.Tensor, positive: torch.Tensor) -> float:
    """Rank-AUC of values V for positive (r>0) vs negative states. 0.5 = chance."""
    V = V.reshape(-1)
    pos = positive.reshape(-1).bool()
    if pos.all() or (~pos).any() is False or pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    vp, vn = V[pos], V[~pos]
    # Mann-Whitney U via pairwise comparison in chunks (N small enough here)
    gt = (vp.unsqueeze(1) > vn.unsqueeze(0)).float().mean()
    eq = (vp.unsqueeze(1) == vn.unsqueeze(0)).float().mean()
    return float(gt + 0.5 * eq)
