"""
agents/fb/model.py — FBModel.

Network call chain (all encoders / normalizer are Identity for state OGBench):
    obs -> _obs_normalizer -> _fw_encoder -> _left_encoder -> _forward_map
                          \-> _bw_encoder ----> _backward_map
                          \-> (left_enc if actor_encode_obs else fw_enc) -> _actor

Target nets: _target_forward_map, _target_backward_map, _target_left_encoder.
All soft-updated.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    IdentityNN,
    SimpleActorArchiConfig,
    _soft_update_params,
    weight_init,
)
from normalizers import IdentityNormalizerConfig


class FBModel(nn.Module):
    """Container for FB networks, encoders, normalizer, and target copies."""

    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        L_dim: int = 50,
        norm_z: bool = True,
        actor_encode_obs: bool = False,
        forward_cfg: Optional[ForwardArchiConfig] = None,
        backward_cfg: Optional[BackwardArchiConfig] = None,
        left_encoder_cfg: Optional[BackwardArchiConfig] = None,
        actor_cfg=None,
        obs_normalizer_cfg=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.L_dim = L_dim
        self.norm_z = norm_z
        self.actor_encode_obs = actor_encode_obs
        self.device = device

        # Defaults match td_jepa's OGBench launch
        if forward_cfg is None:
            forward_cfg = ForwardArchiConfig(hidden_dim=512, hidden_layers=2, num_parallel=2)
        if backward_cfg is None:
            backward_cfg = BackwardArchiConfig(hidden_dim=512, hidden_layers=4, norm=True)
        if left_encoder_cfg is None:
            left_encoder_cfg = BackwardArchiConfig(hidden_dim=512, hidden_layers=4, norm=True)
        if actor_cfg is None:
            actor_cfg = SimpleActorArchiConfig(hidden_dim=512, hidden_layers=2)
        if obs_normalizer_cfg is None:
            obs_normalizer_cfg = IdentityNormalizerConfig()

        # Encoders / normalizer (Identity passthroughs for state OGBench)
        self._obs_normalizer = obs_normalizer_cfg.build(obs_space)
        self._fw_encoder = IdentityNN(obs_space)
        self._bw_encoder = IdentityNN(obs_space)

        # Left encoder produces L_dim-shaped output from fw_encoder.output_space
        self._left_encoder = left_encoder_cfg.build(self._fw_encoder.output_space, L_dim)

        # FB maps
        self._backward_map = backward_cfg.build(self._bw_encoder.output_space, z_dim)
        self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim)

        # Actor input space depends on actor_encode_obs
        actor_in_space = (
            self._left_encoder.output_space if actor_encode_obs else self._fw_encoder.output_space
        )
        self._actor = actor_cfg.build(actor_in_space, z_dim, action_dim)

        # Init weights
        self.apply(weight_init)

        # Target nets
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)
        for p in self._target_forward_map.parameters():
            p.requires_grad_(False)
        for p in self._target_backward_map.parameters():
            p.requires_grad_(False)
        for p in self._target_left_encoder.parameters():
            p.requires_grad_(False)

        self.to(device)

    # ── Target updates ────────────────────────────────────────────────── #

    def soft_update_targets(self, f_tau: float, b_tau: float) -> None:
        _soft_update_params(
            list(self._forward_map.parameters()),
            list(self._target_forward_map.parameters()),
            f_tau,
        )
        _soft_update_params(
            list(self._backward_map.parameters()),
            list(self._target_backward_map.parameters()),
            b_tau,
        )
        _soft_update_params(
            list(self._left_encoder.parameters()),
            list(self._target_left_encoder.parameters()),
            f_tau,
        )

    # ── z utilities ──────────────────────────────────────────────────── #

    def project_z(self, z: Tensor) -> Tensor:
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def sample_z(self, n: int) -> Tensor:
        z = torch.randn(n, self.z_dim, device=self.device)
        return self.project_z(z)

    def mixed_z(self, next_obs_normed_encoded_bw: Tensor, train_goal_ratio: float) -> Tensor:
        """Sample z: with prob train_goal_ratio use B(next_obs[perm]), else random.

        `next_obs_normed_encoded_bw` is the output of `_bw_encoder(_normalize(next_obs))`,
        which the agent has already computed for the FB loss.
        """
        B = next_obs_normed_encoded_bw.shape[0]
        z = self.sample_z(B)
        perm = torch.randperm(B, device=self.device)
        with torch.no_grad():
            goals = self.project_z(
                self._backward_map(next_obs_normed_encoded_bw[perm])
            )
        mask = (torch.rand((B, 1), device=self.device) < train_goal_ratio)
        return torch.where(mask, goals, z)

    # ── Public API used at eval ───────────────────────────────────────── #

    def normalize(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self._obs_normalizer.eval()
            return self._obs_normalizer(x)

    @torch.no_grad()
    def backward_map(self, obs: Tensor) -> Tensor:
        """Public B(obs) including normalize + bw_encoder."""
        return self._backward_map(self._bw_encoder(self.normalize(obs)))

    @torch.no_grad()
    def reward_inference(self, next_obs: Tensor, reward: Tensor) -> Tensor:
        """Linear estimator: z = project(sum_i r_i * B(next_obs_i)).

        reward : [N, 1] or [N]
        next_obs : [N, obs_dim]
        Returns z of shape [z_dim].
        """
        B_mat = self.backward_map(next_obs)              # [N, z_dim]
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)                # [N, 1]
        z = (reward.T @ B_mat).squeeze(0)                # [z_dim]
        return self.project_z(z.unsqueeze(0)).squeeze(0)

    # Backwards-compatible alias used by FBAgent.infer_z
    infer_z = reward_inference
