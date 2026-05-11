"""
agents/fb/agent.py — FBAgent for state-based OGBench.

Implements the M-matrix TD loss with contrastive diagonal, contrastive
orthonormality on B, Q = F·z actor, pairwise-L1-distance pessimism, and
three optimizer groups (forward + left_encoder | backward | actor).

batch keys (from DictBuffer.sample):
    batch["observation"]            [B, obs_dim]
    batch["action"]                 [B, act_dim]
    batch["next"]["observation"]    [B, obs_dim]
    batch["next"]["terminated"]     [B, 1]   bool / float
    batch["next"]["physics"]        [B, phys_dim]    (eval only)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from agents.base import BaseAgent
from agents.fb.model import FBModel


def pairwise_l1_pessimism(
    preds: Tensor, penalty: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (mean, uncertainty, mean - penalty * uncertainty).

    preds : [E, ...]  E = num ensemble heads
    uncertainty = sum_{i,j} |p_i - p_j| / (E^2 - E)
    """
    mean = preds.mean(dim=0)
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()   # [E, E, ...]
    scaling = preds.shape[0] ** 2 - preds.shape[0]
    unc = diffs.sum(dim=(0, 1)) / max(scaling, 1)
    return mean, unc, mean - penalty * unc


class FBAgent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        batch_size: int = 256,
        z_dim: int = 50,
        L_dim: int = 50,
        actor_encode_obs: bool = False,
        forward_cfg=None,
        backward_cfg=None,
        left_encoder_cfg=None,
        actor_cfg=None,
        obs_normalizer_cfg=None,
        discount: float = 0.99,
        lr_f: float = 1e-4,
        lr_b: float = 1e-4,
        lr_actor: float = 1e-4,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 0.0,
        ortho_coef: float = 1.0,
        train_goal_ratio: float = 0.5,
        fb_pessimism_penalty: float = 0.0,
        actor_pessimism_penalty: float = 0.0,
        actor_std: float = 0.2,
        stddev_clip: float = 0.3,
        f_target_tau: float = 0.005,
        b_target_tau: float = 0.005,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.discount_scalar = discount
        self.lr_f = lr_f
        self.lr_b = lr_b
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.ortho_coef = ortho_coef
        self.train_goal_ratio = train_goal_ratio
        self.fb_pessimism_penalty = fb_pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.f_target_tau = f_target_tau
        self.b_target_tau = b_target_tau
        self.device = device
        self.actor_encode_obs = actor_encode_obs

        self.model = FBModel(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            L_dim=L_dim,
            actor_encode_obs=actor_encode_obs,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            left_encoder_cfg=left_encoder_cfg,
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=obs_normalizer_cfg,
            device=device,
        )

        self.forward_optimizer = Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters()),
            lr=lr_f,
            weight_decay=weight_decay,
        )
        self.backward_optimizer = Adam(
            self.model._backward_map.parameters(),
            lr=lr_b,
            weight_decay=weight_decay,
        )
        self.actor_optimizer = Adam(
            self.model._actor.parameters(),
            lr=lr_actor,
            weight_decay=weight_decay,
        )

        # Precompute off-diagonal mask of size [batch_size, batch_size].
        # Couples agent to a fixed batch size (same constraint as td_jepa).
        eye = torch.eye(batch_size, batch_size, device=device)
        self.off_diag = (1.0 - eye)
        self.off_diag_sum = self.off_diag.sum()

    # ── Main update ───────────────────────────────────────────────────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device).float()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)  # [B, 1]
        discount = self.discount_scalar * (1.0 - terminated)  # [B, 1]

        # Normalize + fw/bw encode (Identity for state OGBench but keeps the
        # call chain matching td_jepa)
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        next_obs_n = self.model._fw_encoder(self.model.normalize(next_obs))
        goal = self.model._bw_encoder(self.model.normalize(next_obs))

        # Sample mixed z (no grad)
        with torch.no_grad():
            z = self.model.mixed_z(goal, self.train_goal_ratio)

        fb_metrics = self._update_fb(obs_n, next_obs_n, action, goal, discount, z)
        actor_metrics = self._update_actor(obs_n.detach(), z)
        self.model.soft_update_targets(self.f_target_tau, self.b_target_tau)

        return {**fb_metrics, **actor_metrics}

    # ── FB loss (M-matrix TD + contrastive ortho) ─────────────────────── #

    def _update_fb(
        self,
        obs: Tensor,
        next_obs: Tensor,
        action: Tensor,
        goal: Tensor,
        discount: Tensor,
        z: Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_left = self.model._target_left_encoder(next_obs)
            actor_in = next_left if self.actor_encode_obs else next_obs
            next_dist = self.model._actor(actor_in, z, self.actor_std)
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_Fs = self.model._target_forward_map(next_left, z, next_action)  # [E, B, z_dim]
            target_B = self.model._target_backward_map(goal)                       # [B, z_dim]
            target_Ms = torch.matmul(target_Fs, target_B.T)                        # [E, B, B]
            _, _, target_M = pairwise_l1_pessimism(target_Ms, self.fb_pessimism_penalty)  # [B, B]

        left = self.model._left_encoder(obs)
        Fs = self.model._forward_map(left, z, action)                              # [E, B, z_dim]
        B_out = self.model._backward_map(goal)                                     # [B, z_dim]
        Ms = torch.matmul(Fs, B_out.T)                                             # [E, B, B]

        diff = Ms - discount * target_M                                            # [E, B, B]
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # Contrastive orthonormality on B (batch x batch)
        Cov = torch.matmul(B_out, B_out.T)
        orth_diag = -Cov.diag().mean()
        orth_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_offdiag + orth_diag
        fb_loss = fb_loss + self.ortho_coef * orth_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), self.clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        return {
            "loss/fb": fb_loss.item(),
            "loss/fb_diag": fb_diag.item(),
            "loss/fb_offdiag": fb_offdiag.item(),
            "loss/orth": orth_loss.item(),
            "loss/orth_diag": orth_diag.item(),
            "loss/orth_offdiag": orth_offdiag.item(),
            "train/M1": Ms[0].mean().item(),
            "train/B_norm": B_out.norm(dim=-1).mean().item(),
            "train/z_norm": z.norm(dim=-1).mean().item(),
        }

    # ── Actor loss (Q = F·z, pairwise-L1 pessimism, sampled action) ───── #

    def _update_actor(self, obs: Tensor, z: Tensor) -> Dict[str, float]:
        with torch.no_grad():
            left = self.model._left_encoder(obs)
        actor_in = left if self.actor_encode_obs else obs
        dist = self.model._actor(actor_in, z, self.actor_std)
        actor_action = dist.sample(clip=self.stddev_clip)
        Fs = self.model._forward_map(left, z, actor_action)
        Qs = (Fs * z).sum(-1)                                                      # [E, B]
        _, _, Q = pairwise_l1_pessimism(Qs, self.actor_pessimism_penalty)
        actor_loss = -Q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "loss/actor": actor_loss.item(),
            "train/Q_actor": Q.mean().item(),
        }

    # ── Eval API ──────────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        actor_in = self.model._left_encoder(obs_n) if self.actor_encode_obs else obs_n
        dist = self.model._actor(actor_in, z, std=0.0 if eval_mode else self.actor_std)
        # `mean` is the tanh'd policy output (deterministic)
        return dist.mean if hasattr(dist, "mean") else dist.loc

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.reward_inference(obs.to(self.device), reward.to(self.device))

    # ── Checkpointing ─────────────────────────────────────────────────── #

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.forward_optimizer.load_state_dict(state["forward_optimizer"])
        self.backward_optimizer.load_state_dict(state["backward_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
