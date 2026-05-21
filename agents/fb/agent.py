"""agents/fb/agent.py — Literal port of td_jepa/metamotivo/agents/fb/agent.py.

Goal: byte-equivalent algorithm to td_jepa, only differing where their config
infrastructure is replaced by our Hydra/kwarg-driven instance attributes.

Substitutions vs upstream:
    self.cfg.train.X      -> self.X         (hyperparams live on the agent)
    self.cfg.model.X      -> self.X         (model-side hyperparams too)
    self._model.X         -> self.model.X   (single-underscore convention)
    cudagraphs / compile  -> dropped (eager mode)
    safetensors / pickle  -> dropped (use torch.save state_dict)

Everything else — math, gradient paths, autocast wraps, optimizer parameter
lists, off-diagonal mask precomputation, soft-update wrapping — is preserved
line for line.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from agents.base import BaseAgent
from agents.fb.model import FBModel
from nn_models import _soft_update_params, eval_mode, weight_init


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
        rgb_encoder_cfg=None,
        augmentator_cfg=None,
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
        bc_coeff: float = 0.0,
        q_loss_coef: float = 0.0,
        amp: bool = False,
        device: str = "cpu",
    ):
        # Store every hyperparam as an instance attribute — direct stand-in
        # for td_jepa's self.cfg.train.X / self.cfg.model.X.
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.L_dim = L_dim
        self.actor_encode_obs = actor_encode_obs
        self.discount = discount
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
        self.bc_coeff = bc_coeff
        self.q_loss_coef = q_loss_coef
        self.amp = amp
        self.device = device

        # Build the model via a hook so subclasses (FBFlowBCAgent) can return
        # FBFlowBCModel directly — matches td_jepa's `cfg.model.build(...)`
        # one-shot construction. Building twice (parent first, child second)
        # would consume RNG and shift weight_init's offset.
        self.model = self._make_model(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            L_dim=L_dim,
            actor_encode_obs=actor_encode_obs,
            amp=amp,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            left_encoder_cfg=left_encoder_cfg,
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=obs_normalizer_cfg,
            rgb_encoder_cfg=rgb_encoder_cfg,
            augmentator_cfg=augmentator_cfg,
            device=device,
        )
        self.setup_training()
        self.model.to(self.device)

    def _make_model(self, **kwargs):
        """Construct the model. Subclasses override to return their variant."""
        return FBModel(**kwargs)

    @property
    def amp_dtype(self):
        return self.model.amp_dtype

    def setup_training(self) -> None:
        self.model.train(True)
        self.model.requires_grad_(True)
        self.model.apply(weight_init)
        self.model._prepare_for_train()  # ensure target nets are initialized AFTER applying the weights

        self.backward_optimizer = torch.optim.Adam(
            list(self.model._backward_map.parameters()) + list(self.model._bw_encoder.parameters()),
            lr=self.lr_b,
            weight_decay=self.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters())
            + list(self.model._fw_encoder.parameters()),
            lr=self.lr_f,
            weight_decay=self.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model._actor.parameters(),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )

        # prepare parameter list (used by soft_update_params via torch._foreach_*)
        self._forward_map_paramlist = tuple(x for x in self.model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self.model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self.model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self.model._target_backward_map.parameters())
        self._left_encoder_paramlist = tuple(x for x in self.model._left_encoder.parameters())
        self._target_left_encoder_paramlist = tuple(x for x in self.model._target_left_encoder.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            z = self.model.sample_z(self.batch_size, device=self.device)
            if train_goal is not None:
                perm = torch.randperm(self.batch_size, device=self.device)
                train_goal = train_goal[perm]
                goals = self.model._backward_map(train_goal)
                goals = self.model.project_z(goals)
                mask = torch.rand((self.batch_size, 1), device=self.device) < self.train_goal_ratio
                z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def aug(self, obs, next_obs):
        """Augments observations when training from pixels, does nothing otherwise."""
        return self.model._augmentator(obs), self.model._augmentator(next_obs)

    def enc(self, obs, next_obs):
        """Encodes observations when training from pixels, does nothing otherwise."""
        obs = self.model._fw_encoder(obs)
        goal = self.model._bw_encoder(next_obs)
        with torch.no_grad():
            next_obs = self.model._fw_encoder(next_obs)
        return obs, next_obs, goal

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device).float(),
            batch["action"].to(self.device).float(),
            batch["next"]["observation"].to(self.device).float(),
            batch["next"]["terminated"].to(self.device),
        )
        if terminated.dtype == torch.bool:
            discount = self.discount * ~terminated
        else:
            discount = self.discount * (1.0 - terminated)

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs, next_obs = self.model._obs_normalizer(obs), self.model._obs_normalizer(next_obs)

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs, goal = self.enc(obs, next_obs)

        z = self.sample_mixed_z(train_goal=goal).clone()

        q_loss_coef = self.q_loss_coef if self.q_loss_coef > 0 else None
        clip_grad_norm = self.clip_grad_norm if self.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self.update_actor(
                obs=obs.detach(),
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.f_target_tau)

        # convert tensor metrics to floats for downstream logging (train.py expects floats)
        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            dist = self.model._actor(obs, z, self.actor_std)
            action = dist.sample(clip=self.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)  # batch x L_dim
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                next_action = self.sample_action_from_norm_obs(actor_in, z)
                target_Fs = self.model._target_forward_map(next_left_enc, z, next_action)  # num_parallel x batch x z_dim
                target_B = self.model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            left_enc = self.model._left_encoder(obs)  # batch x L_dim
            Fs = self.model._forward_map(left_enc, z, action)  # num_parallel x batch x z_dim
            B = self.model._backward_map(goal)  # batch x z_dim
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # compute orthonormality loss for backward embedding
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss += self.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.fb_pessimism_penalty)  # batch
                    # we disable autocast here to make sure B and cov have the same dtype (otherwise torch.linalg.solve fails)
                    with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, action=action, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                left_enc = self.model._left_encoder(obs)
            actor_in = left_enc if self.actor_encode_obs else obs
            dist = self.model._actor(actor_in, z, self.actor_std)
            actor_action = dist.sample(clip=self.stddev_clip)
            Fs = self.model._forward_map(left_enc, z, actor_action)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "bc_error": bc_error.detach() if isinstance(bc_error, torch.Tensor) else torch.tensor(bc_error),
            "q": Q.mean().detach(),
        }

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    # ── Rollout / eval API (kept from previous impl; not in td_jepa's agent.py
    #     because their rollout calls model.act directly via extract_model) ── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.reward_inference(obs.to(self.device), reward.to(self.device))

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
