"""
agents/fb/agent.py — FBAgent for state-based OGBench tasks.

Batch layout (from DictBuffer.sample):
    batch["observation"]            [B, obs_dim]
    batch["action"]                 [B, act_dim]
    batch["next"]["observation"]    [B, obs_dim]
    batch["next"]["terminated"]     [B]          bool / float

act() accepts keyword args to match envs/rollout.py:
    agent.act(obs=obs_t, z=z_t)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from agents.base import BaseAgent
from agents.fb.model import FBModel
from nn_models import BackwardArchiConfig, ForwardArchiConfig, SimpleActorArchiConfig


class FBAgent(BaseAgent):

    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        forward_cfg: ForwardArchiConfig | None = None,
        backward_cfg: BackwardArchiConfig | None = None,
        actor_cfg: SimpleActorArchiConfig | None = None,
        discount: float = 0.99,
        lr_f: float = 1e-4,
        lr_b: float = 1e-4,
        lr_actor: float = 1e-4,
        ortho_coef: float = 1.0,
        pessimism: float = 1.0,
        train_goal_ratio: float = 0.5,
        actor_std: float = 0.1,
        actor_std_clip: float = 0.3,
        f_target_tau: float = 0.005,
        b_target_tau: float = 0.005,
        device: str = "cpu",
    ):
        self.discount = discount
        self.ortho_coef = ortho_coef
        self.pessimism = pessimism
        self.train_goal_ratio = train_goal_ratio
        self.actor_std = actor_std
        self.actor_std_clip = actor_std_clip
        self.device = device

        self.model = FBModel(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            actor_cfg=actor_cfg,
            f_target_tau=f_target_tau,
            b_target_tau=b_target_tau,
            device=device,
        )

        self.opt_fb = Adam(
            list(self.model.forward_map.parameters())
            + list(self.model.backward_map.parameters()),
            lr=lr_f,
        )
        self.opt_actor = Adam(self.model.actor.parameters(), lr=lr_actor)

    # ── Main update ─────────────────────────────────────────────────────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        # Unpack nested dict layout from DictBuffer
        obs      = batch["observation"].to(self.device)
        actions  = batch["action"].to(self.device)
        next_obs = batch["next"]["observation"].to(self.device)
        dones    = batch["next"]["terminated"].float().to(self.device)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)   # [B, 1]

        with torch.no_grad():
            z = self.model.mixed_z(next_obs, self.train_goal_ratio)

        # ── FB + ortho ──────────────────────────────────────────────────── #
        fb_loss, metrics = self._fb_loss(obs, next_obs, actions, dones, z)
        ortho_loss = self._ortho_loss(obs)
        total = fb_loss + self.ortho_coef * ortho_loss

        self.opt_fb.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            list(self.model.forward_map.parameters())
            + list(self.model.backward_map.parameters()),
            max_norm=10.0,
        )
        self.opt_fb.step()

        metrics["loss/ortho"] = ortho_loss.item()
        metrics["loss/fb_total"] = total.item()

        # ── Actor ───────────────────────────────────────────────────────── #
        actor_loss, actor_metrics = self._actor_loss(obs, z)
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=10.0)
        self.opt_actor.step()
        metrics.update(actor_metrics)

        self.model.soft_update_targets()
        return metrics

    # ── FB contrastive loss ──────────────────────────────────────────────── #

    def _fb_loss(
        self,
        obs: Tensor,
        next_obs: Tensor,
        actions: Tensor,
        dones: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        B = obs.shape[0]

        with torch.no_grad():
            next_dist = self.model.actor(next_obs, z, self.actor_std)
            next_actions = next_dist.sample(clip=self.actor_std_clip)

            next_F = self.model.forward_map_target(next_obs, z, next_actions)
            next_B = self.model.backward_map_target(next_obs)
            Q_target = self.model.pessimistic_q(next_F, next_B, self.pessimism)   # [B]
            diag_target = (1.0 - dones.squeeze(-1)) * self.discount * Q_target    # [B]

        F_out = self.model.forward_map(obs, z, actions)   # [E, B, z_dim] or [B, z_dim]
        B_out = self.model.backward_map(next_obs)         # [B, z_dim]

        if F_out.dim() == 3:
            Q_online = (F_out * B_out.unsqueeze(0)).sum(-1)          # [E, B]
        else:
            Q_online = (F_out * B_out).sum(-1).unsqueeze(0)          # [1, B]

        diag_loss = F.mse_loss(Q_online, diag_target.unsqueeze(0).expand_as(Q_online))

        # Off-diagonal orthogonality (first ensemble head)
        F0 = F_out[0] if F_out.dim() == 3 else F_out
        M = F0 @ B_out.T                                             # [B, B]
        off_mask = ~torch.eye(B, dtype=torch.bool, device=obs.device)
        off_diag_loss = M[off_mask].pow(2).mean()

        fb_loss = diag_loss + off_diag_loss
        return fb_loss, {
            "loss/fb_diag": diag_loss.item(),
            "loss/fb_offdiag": off_diag_loss.item(),
            "loss/fb": fb_loss.item(),
            "train/Q_target": Q_target.mean().item(),
        }

    # ── B ortho regulariser ──────────────────────────────────────────────── #

    def _ortho_loss(self, obs: Tensor) -> Tensor:
        B_out = self.model.backward_map(obs)
        cov = (B_out.T @ B_out) / B_out.shape[0]
        eye = torch.eye(cov.shape[0], device=cov.device)
        return (cov * (1.0 - eye)).pow(2).mean() + ((cov * eye) - 1.0).pow(2).mean()

    # ── Actor loss ───────────────────────────────────────────────────────── #

    def _actor_loss(self, obs: Tensor, z: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        dist = self.model.actor(obs, z, self.actor_std)
        action_mu = dist.loc  # tanh mean — differentiable, no rsample needed

        with torch.no_grad():
            B_sg = self.model.backward_map(obs)

        F_pi = self.model.forward_map(obs, z, action_mu)
        Q_pi = self.model.pessimistic_q(F_pi, B_sg, self.pessimism)
        actor_loss = -Q_pi.mean()

        return actor_loss, {
            "loss/actor": actor_loss.item(),
            "train/Q_actor": Q_pi.mean().item(),
        }

    # ── Inference API ────────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        """Matches rollout.py: agent.act(obs=..., z=...)"""
        obs = obs.to(self.device)
        z = z.to(self.device)
        dist = self.model.actor(obs, z, std=0.0 if eval_mode else self.actor_std)
        return dist.loc

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.infer_z(obs, reward)

    # ── Checkpointing ────────────────────────────────────────────────────── #

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "opt_fb": self.opt_fb.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.opt_fb.load_state_dict(state["opt_fb"])
        self.opt_actor.load_state_dict(state["opt_actor"])