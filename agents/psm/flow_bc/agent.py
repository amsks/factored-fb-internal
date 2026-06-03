"""agents/psm/flow_bc/agent.py — PSMFlowBCAgent.

PSM representation (proto + SF psi, phi basis) with the TD3 actor REPLACED by
our FlowBC actor: a noise-conditioned flow vector field (behavior cloning of
dataset actions) plus a Q-driven actor head, where Q = sf_psi(obs, z, a) · z.

This is the FBFlowBCAgent pattern (agents/fb/flow_bc/agent.py) ported onto the
PSM base. The deltas vs FBFlowBCAgent:
  - base class FBAgent -> PSMAgent
  - the Q term uses PSM's sf_psi (PsiMap, signature (obs, z, action)) in place
    of FB's _forward_map
  - the real z is used (PSM has no one-step zeroing; there is NO _fb_z)
  - PSM's get_targets_uncertainty returns (mean, unc) (2-tuple), so the
    pessimistic Q is Q_mean - actor_pessimism_penalty * Q_unc (matches PSMAgent
    ._update_actor exactly)
  - the SF target next-action comes from the flow actor (sample_action_from_norm_obs)

The base PSMAgent TD3 update math (_update_psm / _update_sf / _update_td3_actor)
is NOT modified; PSMFlowBCAgent only overrides the actor path. actor_kind is
forced to "flowbc" so PSMModel builds no TD3 actor; PSMFlowBCModel attaches the
flow actor (_actor) + vector field (_actor_vf).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.psm.agent import PSMAgent
from agents.psm.flow_bc.model import PSMFlowBCModel
from nn_models import (
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
    _soft_update_params,
    eval_mode,
)


class PSMFlowBCAgent(PSMAgent):
    def __init__(
        self,
        obs_space,
        action_dim,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        bc_coeff: float = 0.0,
        batch_size=1024,
        z_dim=128,
        max_log_seed=16,
        phi_cfg=None,
        sf_cfg=None,
        norm_z=True,
        phi_input="s",
        obs_normalizer_cfg=None,
        rgb_encoder_cfg=None,
        augmentator_cfg=None,
        num_parallel=2,
        discount=0.98,
        lr_sf=1e-4,
        lr_phi=1e-4,
        lr_actor=1e-4,
        weight_decay=0.0,
        clip_grad_norm=0.0,
        target_tau=0.01,
        ortho_coef=1.0,
        mix_ratio=0.5,
        pessimism_penalty=0.0,
        actor_pessimism_penalty=0.5,
        actor_std=0.2,
        stddev_clip=0.3,
        amp=False,
        device="cpu",
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()

        # store plain attributes (mirror PSMAgent.__init__)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.max_log_seed = max_log_seed
        self.discount = discount
        self.lr_sf = lr_sf
        self.lr_phi = lr_phi
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.target_tau = target_tau
        self.ortho_coef = ortho_coef
        self.mix_ratio = mix_ratio
        self.pessimism_penalty = pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.num_parallel = num_parallel
        self.amp = amp
        self.device = device
        self.device_type = str(device).split(":")[0]

        # FlowBC-specific
        self.flow_steps = flow_steps
        self.lr_actor_vf = lr_actor_vf
        self.bc_coeff = bc_coeff

        # actor_kind="flowbc" => PSMModel builds NO TD3 actor; PSMFlowBCModel
        # attaches _actor (NoiseConditionedActor) + _actor_vf (SimpleVectorField).
        self.model = PSMFlowBCModel(
            obs_space, action_dim, actor_cfg=actor_cfg, actor_vf_cfg=actor_vf_cfg,
            z_dim=z_dim, max_log_seed=max_log_seed, batch_size=batch_size,
            norm_z=norm_z, phi_input=phi_input, phi_cfg=phi_cfg, sf_cfg=sf_cfg,
            actor_kind="flowbc", obs_normalizer_cfg=obs_normalizer_cfg, rgb_encoder_cfg=rgb_encoder_cfg,
            augmentator_cfg=augmentator_cfg, num_parallel=num_parallel, device=device, amp=amp,
        )
        self.model.actor_std = actor_std
        self.model.stddev_clip = stddev_clip
        self.setup_training()
        self.model.to(self.device)

    def setup_training(self) -> None:
        m = self.model
        m.train(True)
        m.requires_grad_(True)
        self.optim_phi = torch.optim.Adam(m.phi.parameters(), lr=self.lr_phi, weight_decay=self.weight_decay)
        self.optim_sf_psi = torch.optim.Adam(m.sf_psi.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        self.optim_psm_psi = torch.optim.Adam(m.psm_psi.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        # actor = the noise-conditioned flow actor head (NOT a TD3 actor)
        self.optim_actor = torch.optim.Adam(m._actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)
        self.actor_vf_optimizer = torch.optim.Adam(
            m._actor_vf.parameters(), lr=self.lr_actor_vf, weight_decay=self.weight_decay
        )

        self._phi_paramlist = tuple(m.phi.parameters())
        self._target_phi_paramlist = tuple(m.target_phi.parameters())
        self._sf_psi_paramlist = tuple(m.sf_psi.parameters())
        self._target_sf_psi_paramlist = tuple(m.target_sf_psi.parameters())
        self._psm_psi_paramlist = tuple(m.psm_psi.parameters())
        self._target_psm_psi_paramlist = tuple(m.target_psm_psi.parameters())

        self.off_diag = 1 - torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    # ── flow actor sampler (replaces the TD3 actor's stochastic sample) ── #
    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        action = self.model._actor(obs, z, noises)
        return action

    # ── FlowBC actor update: flow-matching BC + Q-driven head (Q from sf_psi·z) ── #
    def update_actor(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            # flow matching l2 loss (goal/z-UNconditioned — pure BC of dataset actions)
            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

            # Q loss. actor_in = obs (already fw-encoded; Identity for state).
            actor_in = obs
            noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            actor_actions = self.model._actor(actor_in, z, noises)
            # Q via PSM's sf_psi (PsiMap signature (obs, z, action)); dots with the real z.
            Fs = self.model.sf_psi(actor_in, z, actor_actions)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            Q_mean, Q_unc = self.model.get_targets_uncertainty(Qs)  # batch
            Q = Q_mean - self.actor_pessimism_penalty * Q_unc  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_loss = torch.tensor([0.0], device=action.device)
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                with torch.no_grad():
                    target_flow_actions = self.compute_flow_actions(actor_in, noises)
                bc_error = torch.pow(actor_actions - target_flow_actions, 2).mean()
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

            actor_loss = actor_loss + bc_flow_loss

        self.optim_actor.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
        self.optim_actor.step()
        self.actor_vf_optimizer.step()

        return {
            "actor_loss": actor_loss.mean().detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "bc_error": bc_error.detach() if isinstance(bc_error, torch.Tensor) else torch.tensor(bc_error),
            "q": Q.mean().detach(),
        }

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.flow_steps
            vels = self.model._actor_vf(obs, actions, t)
            actions = actions + vels / self.flow_steps
        actions = torch.clamp(actions, min=-1, max=1)
        return actions

    # ── update orchestration (mirror PSMAgent.update; actor path swapped) ── #
    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device)
        idx = batch.get("index")
        next_obs_hash = (idx.to(self.device) if idx is not None
                         else torch.arange(self.batch_size, device=self.device))

        if terminated.dtype == torch.bool:
            discount = self.discount * ~terminated
        else:
            discount = self.discount * (1.0 - terminated)
        discount = discount.reshape(-1, 1)

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs_n = self.model._obs_normalizer(obs)
            next_obs_n = self.model._obs_normalizer(next_obs)

        obs_enc = self._enc_obs(obs_n)
        next_obs_enc = self._enc_obs(next_obs_n)
        goal = self._enc_goal(next_obs_n)

        metrics: Dict = {}

        # 1) proto-successor branch (unchanged from PSMAgent)
        z_psm = self.model.sample_z_psm(self.batch_size, device=self.device)
        metrics.update(self._update_psm(obs_enc, action, discount, next_obs_enc, next_obs_hash, goal, z_psm))
        with torch.no_grad():
            _soft_update_params(self._psm_psi_paramlist, self._target_psm_psi_paramlist, self.target_tau)
            _soft_update_params(self._phi_paramlist, self._target_phi_paramlist, self.target_tau)

        # 2) SF + FlowBC-actor branch: mixed Gaussian z.
        z = self.sample_mixed_z(goal)
        # SF target next-action comes from the FLOW actor (base _update_sf would
        # otherwise call self.model.actor(...), which is None for flowbc).
        with torch.no_grad():
            sf_next_action = self.sample_action_from_norm_obs(next_obs_enc, z)
        metrics.update(self._update_sf(obs_enc, action, discount, next_obs_enc, goal, z,
                                       next_action=sf_next_action))
        metrics.update(self.update_actor(obs_enc, action, z))
        with torch.no_grad():
            _soft_update_params(self._sf_psi_paramlist, self._target_sf_psi_paramlist, self.target_tau)

        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def act(self, obs, z, *, eval_mode: bool = True):
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    def state_dict(self) -> dict:
        state = {
            "model": self.model.state_dict(),
            "optim_phi": self.optim_phi.state_dict(),
            "optim_sf_psi": self.optim_sf_psi.state_dict(),
            "optim_psm_psi": self.optim_psm_psi.state_dict(),
            "optim_actor": self.optim_actor.state_dict(),
            "actor_vf_optimizer": self.actor_vf_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optim_phi.load_state_dict(state["optim_phi"])
        self.optim_sf_psi.load_state_dict(state["optim_sf_psi"])
        self.optim_psm_psi.load_state_dict(state["optim_psm_psi"])
        self.optim_actor.load_state_dict(state["optim_actor"])
        self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
