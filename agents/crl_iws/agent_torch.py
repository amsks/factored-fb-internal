"""PyTorch CRL agent — self-contained port of Unsupervised_Manipulation's PyTorchSGCRLAgent.

Architecture matches UM exactly:
  Actor:  MLP → tanh(x/10)*10 mean, softplus scale; tanh-squashed Normal
  Critic: two separate MLP encoders → dot-product logit matrix
  Loss:   CPC cross-entropy + logsumexp² regularization; actor: -diag(logits)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: tuple[int, ...], min_scale: float = 1e-6):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.loc_layer = nn.Linear(prev, action_dim)
        self.scale_layer = nn.Linear(prev, action_dim)
        self.min_scale = float(min_scale)

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = 10.0 * torch.tanh(self.loc_layer(h) / 10.0)
        scale = F.softplus(self.scale_layer(h)) + self.min_scale
        dist = torch.distributions.Normal(mean, scale)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_prob = torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        else:
            log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class _Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        repr_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        score_type: str = "dot",
    ):
        super().__init__()
        self.score_type = str(score_type)

        def _mlp(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, repr_dim))
            return nn.Sequential(*layers)

        self.sa_encoder = _mlp(state_dim + action_dim)
        self.goal_encoder = _mlp(goal_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.sa_encoder(torch.cat([state, action], dim=-1))
        psi = self.goal_encoder(goal)
        if self.score_type == "cosine":
            phi = F.normalize(phi, dim=-1, eps=1e-6)
            psi = F.normalize(psi, dim=-1, eps=1e-6)
        logits = phi @ psi.t()
        return logits, phi, psi


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CRLIWSAgentTorch:
    """Goal-conditioned CRL agent with IWS-compatible update interface."""

    def __init__(
        self,
        *,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device | str = "cpu",
        hidden_dims: tuple[int, ...] = (256, 256),
        repr_dim: int = 64,
        actor_lr: float = 6e-4,
        critic_lr: float = 3e-4,
        entropy_coeff: float = 0.0,
        logsumexp_coeff: float = 0.1,
        random_goals: float = 0.5,
        use_action_entropy: bool = True,
        critic_score_type: str = "dot",
    ) -> None:
        self.device = torch.device(device)
        self.entropy_coeff = float(entropy_coeff)
        self.logsumexp_coeff = float(logsumexp_coeff)
        self.random_goals = float(random_goals)
        self.use_action_entropy = bool(use_action_entropy)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        self.actor = _Actor(state_dim + goal_dim, action_dim, hidden_dims).to(self.device)
        self.critic = _Critic(
            state_dim, action_dim, goal_dim, repr_dim, hidden_dims,
            score_type=critic_score_type,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.reshape(1, -1)
        high = self.action_high.reshape(1, -1)
        return low + 0.5 * (action + 1.0) * (high - low)

    def _mix_goals(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.random_goals == 0.5:
            return (
                torch.cat([state, state], dim=0),
                torch.cat([goal, torch.roll(goal, 1, dims=0)], dim=0),
            )
        if self.random_goals == 1.0:
            return state, torch.roll(goal, 1, dims=0)
        return state, goal

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        state = self._to_tensor(batch["states"])
        action = self._to_tensor(batch["actions"])
        goal = self._to_tensor(batch["goals"])

        # ---- Critic ----
        logits, _, _ = self.critic(state, action, goal)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)

        logsumexp = torch.logsumexp(logits, dim=1).pow(2).mean()
        critic_loss = F.cross_entropy(logits, labels) + self.logsumexp_coeff * logsumexp

        logits_pos = logits[pos_mask].mean()
        logits_neg = logits[~pos_mask].mean()
        cat_acc = (logits.argmax(dim=1) == labels).float().mean()
        bin_acc = ((logits > 0) == pos_mask).float().mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor (critic frozen) ----
        for p in self.critic.parameters():
            p.requires_grad_(False)

        actor_state, actor_goal = self._mix_goals(state, goal)
        action_sample, log_prob = self.actor.sample(
            torch.cat([actor_state, actor_goal], dim=-1)
        )
        actor_logits, _, _ = self.critic(actor_state, self._scale_action(action_sample), actor_goal)

        actor_loss_vec = -actor_logits.diag()
        entropy = -log_prob.mean()
        if self.use_action_entropy:
            actor_loss_vec = actor_loss_vec + self.entropy_coeff * log_prob.squeeze(-1)
        actor_loss = actor_loss_vec.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        return {
            "train/critic_loss": float(critic_loss.item()),
            "train/actor_loss": float(actor_loss.item()),
            "train/logits_pos": float(logits_pos.item()),
            "train/logits_neg": float(logits_neg.item()),
            "train/categorical_accuracy": float(cat_acc.item()),
            "train/binary_accuracy": float(bin_acc.item()),
            "train/policy_entropy": float(entropy.item()),
            "train/logsumexp": float(logsumexp.item()),
        }

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        states: np.ndarray,
        goals: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs = torch.cat([self._to_tensor(states), self._to_tensor(goals)], dim=-1)
        action, _ = self.actor.sample(obs, deterministic=deterministic)
        return self._scale_action(action).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.actor_optimizer.load_state_dict(d["actor_optimizer"])
        self.critic_optimizer.load_state_dict(d["critic_optimizer"])
