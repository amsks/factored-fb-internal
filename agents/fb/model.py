"""
agents/fb/model.py — FBModel

Constructs all FB networks via their ArchiConfig builders, keeping target copies
for F and B. Uses _soft_update_params from nn_models for efficient Polyak updates.

ForwardMap output shape: [num_parallel, B, z_dim]  (when num_parallel > 1)
BackwardMap output shape: [B, z_dim]  (already L2-normalised by Norm layer)
Actor output: TruncatedNormal(loc=[B, act_dim], scale=std)
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleActorArchiConfig,
    _soft_update_params,
    weight_init,
)


class FBModel(nn.Module):
    """Container for all FB networks + target copies.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Box  (1-D state space)
    action_dim : int
    z_dim : int
    forward_cfg : ForwardArchiConfig
    backward_cfg : BackwardArchiConfig
    actor_cfg : SimpleActorArchiConfig | NoiseConditionedActorArchiConfig
    f_target_tau, b_target_tau : float
    device : str
    """

    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        forward_cfg: ForwardArchiConfig | None = None,
        backward_cfg: BackwardArchiConfig | None = None,
        actor_cfg: SimpleActorArchiConfig | None = None,
        f_target_tau: float = 0.005,
        b_target_tau: float = 0.005,
        device: str = "cpu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.f_target_tau = f_target_tau
        self.b_target_tau = b_target_tau
        self.device = device

        # ── Defaults matching OGBench paper configs ──────────────────────── #
        if forward_cfg is None:
            forward_cfg = ForwardArchiConfig()       # hidden_dim=1024, num_parallel=2
        if backward_cfg is None:
            backward_cfg = BackwardArchiConfig()     # hidden_dim=256, norm=True
        if actor_cfg is None:
            actor_cfg = SimpleActorArchiConfig()     # hidden_dim=1024

        # ── Build networks ───────────────────────────────────────────────── #
        self.forward_map = forward_cfg.build(obs_space, z_dim, action_dim)
        self.backward_map = backward_cfg.build(obs_space, z_dim)
        self.actor = actor_cfg.build(obs_space, z_dim, action_dim)

        # ── Apply orthogonal init ─────────────────────────────────────────── #
        self.forward_map.apply(weight_init)
        self.backward_map.apply(weight_init)
        self.actor.apply(weight_init)

        # ── Target networks (F and B only) ────────────────────────────────── #
        self.forward_map_target = copy.deepcopy(self.forward_map)
        self.backward_map_target = copy.deepcopy(self.backward_map)
        for p in self.forward_map_target.parameters():
            p.requires_grad_(False)
        for p in self.backward_map_target.parameters():
            p.requires_grad_(False)

        self.to(device)

    # ── Target updates ───────────────────────────────────────────────────── #

    def soft_update_targets(self) -> None:
        _soft_update_params(
            list(self.forward_map.parameters()),
            list(self.forward_map_target.parameters()),
            self.f_target_tau,
        )
        _soft_update_params(
            list(self.backward_map.parameters()),
            list(self.backward_map_target.parameters()),
            self.b_target_tau,
        )

    # ── z utilities ──────────────────────────────────────────────────────── #

    def project_z(self, z: Tensor) -> Tensor:
        """Re-normalise to ||z|| = sqrt(z_dim) (matches Norm layer convention)."""
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8) * math.sqrt(self.z_dim)

    def sample_z(self, n: int) -> Tensor:
        return self.project_z(torch.randn(n, self.z_dim, device=self.device))

    def mixed_z(self, next_obs: Tensor, train_goal_ratio: float = 0.5) -> Tensor:
        """50 % from B(next_obs), 50 % random Gaussian."""
        B = next_obs.shape[0]
        n_goal = int(B * train_goal_ratio)
        with torch.no_grad():
            z_goal = self.backward_map(next_obs[:n_goal])   # already normalised by Norm
        z_rand = self.sample_z(B - n_goal)
        return torch.cat([z_goal, z_rand], dim=0)

    # ── Reward inference ─────────────────────────────────────────────────── #

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        """Closed-form least-squares: w* = (B^T B + λI)^{-1} B^T r → normalised z.

        Parameters
        ----------
        obs    : [N, obs_dim]
        reward : [N] or [N, 1]
        Returns
        -------
        z : [z_dim]   (normalised)
        """
        r = reward.reshape(-1, 1).float().to(self.device)
        B_mat = self.backward_map(obs.to(self.device))          # [N, z_dim]
        reg = 1e-4 * torch.eye(self.z_dim, device=self.device)
        w = torch.linalg.solve(B_mat.T @ B_mat + reg, B_mat.T @ r).squeeze(-1)
        return self.project_z(w.unsqueeze(0)).squeeze(0)

    # ── Convenience: pessimistic Q from ensemble ─────────────────────────── #

    @staticmethod
    def pessimistic_q(F_out: Tensor, B_out: Tensor, pessimism: float = 1.0) -> Tensor:
        """Compute Q_mean − pessimism·Q_std from ensemble.

        Parameters
        ----------
        F_out : [E, B, z_dim]  or  [B, z_dim]
        B_out : [B, z_dim]
        Returns
        -------
        Q : [B]
        """
        if F_out.dim() == 3:
            # [E, B, z_dim] · [B, z_dim] → [E, B]
            Qs = (F_out * B_out.unsqueeze(0)).sum(-1)
            return Qs.mean(0) - pessimism * Qs.std(0)
        # Single head
        return (F_out * B_out).sum(-1)