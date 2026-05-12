"""
agents/fb/flow_bc/agent.py — FBFlowBCAgent.

Three loss terms per actor update:
  1. Flow-matching loss   trains _actor_vf
  2. RL loss              trains _actor with Q = F·z
  3. Distillation BC      trains _actor against ODE rollout of _actor_vf

Inherits the FB loss and target-update logic from FBAgent.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from agents.fb.agent import FBAgent, pairwise_l1_pessimism
from agents.fb.flow_bc.model import FBFlowBCModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class FBFlowBCAgent(FBAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        bc_coeff: float = 0.3,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        **fb_kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig(
                hidden_dim=512, hidden_layers=2, embedding_layers=2
            )
        # Construct FBAgent with the standard FBModel build... but we need to
        # swap in FBFlowBCModel. We avoid super().__init__ creating an FBModel
        # by replicating the bookkeeping here.
        self.batch_size = fb_kwargs.get("batch_size", 256)
        self.discount_scalar = fb_kwargs.get("discount", 0.99)
        self.lr_f = fb_kwargs.get("lr_f", 1e-4)
        self.lr_b = fb_kwargs.get("lr_b", 1e-4)
        self.lr_actor = fb_kwargs.get("lr_actor", 1e-4)
        self.weight_decay = fb_kwargs.get("weight_decay", 0.0)
        self.clip_grad_norm = fb_kwargs.get("clip_grad_norm", 0.0)
        self.ortho_coef = fb_kwargs.get("ortho_coef", 1.0)
        self.train_goal_ratio = fb_kwargs.get("train_goal_ratio", 0.5)
        self.fb_pessimism_penalty = fb_kwargs.get("fb_pessimism_penalty", 0.0)
        self.actor_pessimism_penalty = fb_kwargs.get("actor_pessimism_penalty", 0.0)
        self.actor_std = fb_kwargs.get("actor_std", 0.2)
        self.stddev_clip = fb_kwargs.get("stddev_clip", 0.3)
        self.f_target_tau = fb_kwargs.get("f_target_tau", 0.005)
        self.b_target_tau = fb_kwargs.get("b_target_tau", 0.005)
        self.device = fb_kwargs.get("device", "cpu")
        self.actor_encode_obs = fb_kwargs.get("actor_encode_obs", False)

        self.bc_coeff = bc_coeff
        self.flow_steps = flow_steps
        self.action_dim = action_dim

        self.model = FBFlowBCModel(
            obs_space=obs_space,
            action_dim=action_dim,
            actor_vf_cfg=actor_vf_cfg,
            z_dim=fb_kwargs.get("z_dim", 50),
            L_dim=fb_kwargs.get("L_dim", 50),
            actor_encode_obs=self.actor_encode_obs,
            forward_cfg=fb_kwargs.get("forward_cfg"),
            backward_cfg=fb_kwargs.get("backward_cfg"),
            left_encoder_cfg=fb_kwargs.get("left_encoder_cfg"),
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=fb_kwargs.get("obs_normalizer_cfg"),
            device=self.device,
        )

        self.forward_optimizer = Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters()),
            lr=self.lr_f, weight_decay=self.weight_decay,
        )
        self.backward_optimizer = Adam(
            self.model._backward_map.parameters(),
            lr=self.lr_b, weight_decay=self.weight_decay,
        )
        self.actor_optimizer = Adam(
            self.model._actor.parameters(),
            lr=self.lr_actor, weight_decay=self.weight_decay,
        )
        self.actor_vf_optimizer = Adam(
            self.model._actor_vf.parameters(),
            lr=lr_actor_vf, weight_decay=self.weight_decay,
        )

        eye = torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag = (1.0 - eye)
        self.off_diag_sum = self.off_diag.sum()

    # ── Target action sampling (noise-conditioned) ─────────────────────── #

    def _sample_target_action(self, actor_in: Tensor, z: Tensor) -> Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        return self.model._actor(actor_in, z, noises)

    # ── Override actor update ─────────────────────────────────────────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device).float()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        discount = self.discount_scalar * (1.0 - terminated)

        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        next_obs_n = self.model._fw_encoder(self.model.normalize(next_obs))
        goal = self.model._bw_encoder(self.model.normalize(next_obs))

        with torch.no_grad():
            z = self.model.mixed_z(goal, self.train_goal_ratio)

        fb_metrics = self._update_fb(obs_n, next_obs_n, action, goal, discount, z)
        actor_metrics = self._update_actor_flow(obs_n.detach(), action, z)
        self.model.soft_update_targets(self.f_target_tau, self.b_target_tau)
        return {**fb_metrics, **actor_metrics}

    def _update_actor_flow(
        self, obs: Tensor, action: Tensor, z: Tensor
    ) -> Dict[str, float]:
        # 1. Flow-matching loss (trains _actor_vf)
        x_1 = action
        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0], 1), device=action.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.model._actor_vf(obs, x_t, t)
        bc_flow_loss = (pred - vel).pow(2).mean()

        # 2. RL loss (trains _actor)
        with torch.no_grad():
            left = self.model._left_encoder(obs)
        actor_in = left if self.actor_encode_obs else obs
        noises = torch.randn_like(x_1)
        a_actor = self.model._actor(actor_in, z, noises)
        Fs = self.model._forward_map(left, z, a_actor)
        Qs = (Fs * z).sum(-1)
        _, _, Q = pairwise_l1_pessimism(Qs, self.actor_pessimism_penalty)
        rl_loss = -Q.mean()

        # 3. Distillation BC (trains _actor)
        bc_error = torch.zeros((), device=action.device)
        if self.bc_coeff > 0:
            with torch.no_grad():
                target = self.compute_flow_actions(obs, noises)
            bc_error = (a_actor - target).pow(2).mean()
            bc_loss = self.bc_coeff * bc_error
            actor_loss = (rl_loss / Qs.abs().mean().detach().clamp(min=1e-6)) + bc_loss
        else:
            actor_loss = rl_loss

        total = actor_loss + bc_flow_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.model._actor_vf.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        self.actor_vf_optimizer.step()

        return {
            "loss/actor": float(actor_loss.detach()) if isinstance(actor_loss, Tensor) else float(actor_loss),
            "loss/actor_rl": rl_loss.item(),
            "loss/actor_bc": float(bc_error.detach()) if isinstance(bc_error, Tensor) else float(bc_error),
            "loss/bc_flow": bc_flow_loss.item(),
            "train/Q_actor": Q.mean().item(),
        }

    def compute_flow_actions(self, obs: Tensor, noises: Tensor) -> Tensor:
        """Euler-step the flow matching ODE from x=noises to t=1."""
        x = noises
        for i in range(self.flow_steps):
            t = torch.full((noises.shape[0], 1), float(i) / self.flow_steps, device=obs.device)
            vel = self.model._actor_vf(obs, x, t)
            x = x + vel / self.flow_steps
        return torch.clamp(x, -1.0, 1.0)

    # ── Eval ──────────────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        obs_n = self.model._fw_encoder(self.model.normalize(obs))
        actor_in = self.model._left_encoder(obs_n) if self.actor_encode_obs else obs_n
        noises = torch.randn(obs.shape[0], self.action_dim, device=self.device)
        return self.model._actor(actor_in, z, noises)

    # ── Checkpointing ─────────────────────────────────────────────────── #

    def state_dict(self) -> dict:
        s = super().state_dict()
        s["actor_vf_optimizer"] = self.actor_vf_optimizer.state_dict()
        return s

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
