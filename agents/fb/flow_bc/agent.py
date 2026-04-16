"""
agents/fb/flow_bc/agent.py — FBFlowBCAgent

Extends FBAgent with NoiseConditionedActor and a BC loss:
    L_bc = MSE(actor(obs, z, ε), a_data)

At eval, picks the highest-Q action from N noise samples.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from agents.fb.agent import FBAgent
from nn_models import BackwardArchiConfig, ForwardArchiConfig, NoiseConditionedActorArchiConfig


class FBFlowBCAgent(FBAgent):

    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        forward_cfg: ForwardArchiConfig | None = None,
        backward_cfg: BackwardArchiConfig | None = None,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        bc_coeff: float = 0.3,
        n_action_samples: int = 10,
        **kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        super().__init__(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            actor_cfg=actor_cfg,
            **kwargs,
        )
        self.bc_coeff = bc_coeff
        self.n_action_samples = n_action_samples
        self.action_dim = action_dim

    # ── Cache data actions for _actor_loss then run parent update ─────── #

    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        self._data_actions = batch["action"].to(self.device)
        return super().update(batch, step)

    # ── Actor loss: RL + BC ──────────────────────────────────────────────── #

    def _actor_loss(self, obs: Tensor, z: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        B = obs.shape[0]

        # RL: maximise Q with a single noise sample
        noise_rl = torch.randn(B, self.action_dim, device=self.device)
        action_rl = self.model.actor(obs, z, noise_rl)
        with torch.no_grad():
            B_sg = self.model.backward_map(obs)
        F_rl = self.model.forward_map(obs, z, action_rl)
        Q_rl = self.model.pessimistic_q(F_rl, B_sg, self.pessimism)
        rl_loss = -Q_rl.mean()

        # BC: reproduce data action for a random noise seed
        noise_bc = torch.randn_like(self._data_actions)
        pred = self.model.actor(obs.detach(), z.detach(), noise_bc)
        bc_loss = F.mse_loss(pred, self._data_actions)

        actor_loss = rl_loss + self.bc_coeff * bc_loss
        return actor_loss, {
            "loss/actor_rl": rl_loss.item(),
            "loss/actor_bc": bc_loss.item(),
            "loss/actor": actor_loss.item(),
            "train/Q_actor": Q_rl.mean().item(),
        }

    # ── Eval: argmax-Q over N noise samples ──────────────────────────────── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device)
        z = z.to(self.device)
        B = obs.shape[0]

        if not eval_mode or self.n_action_samples <= 1:
            return self.model.actor(obs, z, torch.randn(B, self.action_dim, device=self.device))

        N = self.n_action_samples
        obs_r = obs.repeat_interleave(N, dim=0)
        z_r   = z.repeat_interleave(N, dim=0)
        noise = torch.randn(B * N, self.action_dim, device=self.device)

        actions = self.model.actor(obs_r, z_r, noise)                 # [B*N, act_dim]
        B_out   = self.model.backward_map(obs_r)
        F_out   = self.model.forward_map(obs_r, z_r, actions)
        Q       = self.model.pessimistic_q(F_out, B_out, pessimism=0.0)

        best = Q.view(B, N).argmax(dim=1)                             # [B]
        return actions.view(B, N, self.action_dim)[
            torch.arange(B, device=self.device), best
        ]