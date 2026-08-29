"""agents/dfb/agent.py — Density-FB as a standalone torch agent."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from agents.base import BaseAgent, soft_update
from agents.dfb.hit_spec import delta_phi, parse_hit_spec, raw_block_cols, scaled_block_cols
from agents.dfb.model import DFBModel
from nn_models import weight_init



# Zero-shot readout (numpy, mirrors eval_density_zeroshot.py exactly)

def density_weights(f_emb: np.ndarray, b_pool: np.ndarray, temp: float,
                    double_q: bool) -> np.ndarray:
    """Per-head softmax over the pool, then (double_q) elementwise MIN over the
    two probability vectors. f_emb [H, d]; b_pool [P, d] -> [P].
    NOTE: the min of two simplex vectors is SUB-normalized whenever the heads
    disagree (spec 30 Item 4 finding) — kept because it is the shipped eval
    convention, and quantified in the tests rather than silently renormalized."""
    logits = f_emb @ b_pool.T / temp                       # [H, P]
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / w.sum(axis=-1, keepdims=True)                  # [H, P] simplex rows
    return w.min(axis=0) if double_q else w[0]


def compute_density_primary(r_x0: float, w: np.ndarray, kappa: np.ndarray,
                            r_pool: np.ndarray, gamma: float) -> float:
    """PRIMARY de-biased readout: ( r(x0) + (w - kappa)^T r(U_P) ) / (1 - gamma).
    The r(x0) / -kappa^T r pair is the finite-pool smoothing correction whose
    halves must never be dropped separately (the G4 false-null postmortem)."""
    return (float(r_x0) + float((w - kappa) @ r_pool)) / (1.0 - gamma)


class DFBAgent(BaseAgent):
    """Density-FB: normalized successor-density measure + reward-integrating actor."""

    def __init__(
        self,
        obs_space,
        action_dim: int,
        measure_hit_spec: str = "19:22:0.4",
        measure_density_dim: int = 50,
        measure_density_temp: float = 1.0,
        measure_density_hard: bool = True,      # fdefaults to hard variant
        hidden_dims=(512, 512, 512),
        actor_kind: str = "flow",                # 'flow' | 'ddpgbc'
        actor_cfg=None,
        actor_vf_cfg=None,
        discount: float = 0.99,
        lr_f: float = 3e-4,
        lr_b: float = 3e-4,
        lr_actor: float = 3e-4,
        lr_actor_vf: float = 3e-4,
        flow_steps: int = 10,
        bc_coeff: float = 3.0,                   # manipulation default
        alpha: float = 0.3,                      # ddpgbc BC coefficient
        actor_std: float = 0.2,
        const_std: bool = True,
        double_q: bool = True,
        actor_q_center: bool = True,
        target_tau: float = 0.005,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.blocks = parse_hit_spec(measure_hit_spec)
        self.query_dim = sum(b - a for (a, b, e) in self.blocks)
        self.density_dim = measure_density_dim
        self.temp = measure_density_temp
        self.hard = bool(measure_density_hard)
        self.actor_kind = actor_kind
        self.discount = discount
        self.flow_steps = flow_steps
        self.bc_coeff = bc_coeff
        self.alpha = alpha
        self.const_std = const_std
        self.double_q = double_q
        self.actor_q_center = actor_q_center
        self.target_tau = target_tau
        self.device = device

        self.model = DFBModel(
            obs_space, action_dim, query_dim=self.query_dim,
            density_dim=measure_density_dim, hidden_dims=tuple(hidden_dims),
            actor_kind=actor_kind, actor_cfg=actor_cfg, actor_vf_cfg=actor_vf_cfg,
            actor_std=actor_std, device=device,
        )
        self.model.apply(weight_init)
        self.model._prepare_for_train()

        self.f_optimizer = torch.optim.Adam(self.model.density_f.parameters(),
                                            lr=lr_f, weight_decay=weight_decay)
        self.b_optimizer = torch.optim.Adam(self.model.density_b.parameters(),
                                            lr=lr_b, weight_decay=weight_decay)
        self.actor_optimizer = torch.optim.Adam(self.model._actor.parameters(),
                                                lr=lr_actor, weight_decay=weight_decay)
        self.actor_vf_optimizer = (
            torch.optim.Adam(self.model._actor_vf.parameters(),
                             lr=lr_actor_vf, weight_decay=weight_decay)
            if self.model._actor_vf is not None else None)

    # ── measure ──────────────────────────────────────────────────────────────

    def density_targets(self, obs: Tensor, nobs: Tensor, goals: Tensor,
                        u_raw: Tensor, u_sc: Tensor) -> Tensor:
        """t = sg[(1-gamma) k + gamma w_bar]  [2, B, M], rows on the simplex.
        Frozen target towers; bootstrap action a+ ~ pi(s', g), stop-gradded."""
        with torch.no_grad():
            if self.actor_kind == "flow":
                noise = torch.randn((nobs.shape[0], self.action_dim),
                                    device=nobs.device, dtype=nobs.dtype)
                a_plus = self.model.one_shot_action(nobs, goals, noise)
            else:  # ddpgbc: mode if const_std else a sample 
                dist = self.model.gaussian_dist(nobs, goals)
                a_plus = dist.mean if self.const_std else dist.sample()
                a_plus = torch.clamp(a_plus, -1.0, 1.0)
            tb = self.model.target_density_b(u_sc)                     # [M, d]
            tf = self.model.target_density_f(
                self.model.f_input(nobs, a_plus, goals))               # [2, B, d]
            tgt_logits = torch.einsum("hbd,md->hbm", tf, tb) / self.temp
            w_bar = torch.softmax(tgt_logits, dim=-1)                  # [2, B, M]
            if self.hard:
                k = torch.eye(nobs.shape[0], device=nobs.device, dtype=nobs.dtype)
            else:
                k = delta_phi(nobs, u_raw, self.blocks)                # [B, M]
            t = (1.0 - self.discount) * k[None] + self.discount * w_bar
        return t

    def density_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict]:
        obs, nobs = batch["observations"], batch["next_observations"]
        acts, goals = batch["actions"], batch["value_goals"]

        u_raw = raw_block_cols(nobs, self.blocks)                      # [M, d_c] RAW
        u_sc = scaled_block_cols(nobs, self.blocks)                    # [M, d_c] scaled
        b_emb = self.model.density_b(u_sc)                             # [M, d]
        f_emb = self.model.density_f(self.model.f_input(obs, acts, goals))  # [2, B, d]
        logits = torch.einsum("hbd,md->hbm", f_emb, b_emb) / self.temp
        logp = F.log_softmax(logits, dim=-1)                           # [2, B, M]

        t = self.density_targets(obs, nobs, goals, u_raw, u_sc)        # sg [2, B, M]
        ce = -(t * logp).sum(dim=-1).mean()

        with torch.no_grad():
            w = torch.softmax(logits, dim=-1)
            ess_rows = 1.0 / (w ** 2).sum(dim=-1)                      # [2, B]
            info = {
                "density_ce": float(ce.detach()),
                "density_ess": float(ess_rows.mean()),
                "density_ess_median": float(ess_rows.median()),
                "density_w_entropy": float(-(w * logp).sum(-1).mean()),
                "density_tgt_entropy": float(-(t * torch.log(t + 1e-8)).sum(-1).mean()),
                "density_logit_max": float(logits.max(dim=-1).values.mean()),
            }
        return ce, info

    # ── actor ────────────────────────────────────────────────────────────────

    def density_actor_q(self, batch: Dict[str, Tensor], obs: Tensor,
                        actions: Tensor, goals: Tensor) -> tuple[Tensor, Tensor]:
        """(q1, q2) = sum_j w_h(s,a,g,u_j) * kernel(goal, u_j). Towers evaluated
        normally (their optimizers never step on the actor loss — the torch twin
        of jax's frozen-params call); kg stop-gradded (the kernel is fixed, not
        trained through the actor)."""
        nobs = batch["next_observations"]
        u_raw = raw_block_cols(nobs, self.blocks)
        u_sc = scaled_block_cols(nobs, self.blocks)
        with torch.no_grad():
            kg = delta_phi(goals, u_raw, self.blocks)                  # [B, M] sg
        b_emb = self.model.density_b(u_sc)                             # [M, d]
        f_emb = self.model.density_f(self.model.f_input(obs, actions, goals))
        w = torch.softmax(torch.einsum("hbd,md->hbm", f_emb, b_emb) / self.temp, dim=-1)
        q = (w * kg[None]).sum(dim=-1)                                 # [2, B]
        return q[0], q[1]

    def _flow_rollout(self, obs: Tensor, noise: Tensor) -> Tensor:
        """Euler-integrate the (goal-agnostic) BC velocity field, frozen (no grad)."""
        with torch.no_grad():
            actions = noise
            for i in range(self.flow_steps):
                t = torch.full((noise.shape[0], 1), i / self.flow_steps,
                               device=noise.device, dtype=noise.dtype)
                actions = actions + self.model._actor_vf(obs, actions, t) / self.flow_steps
            return torch.clamp(actions, -1.0, 1.0)

    def actor_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict]:
        obs, goals, actions = batch["observations"], batch["actor_goals"], batch["actions"]

        if self.actor_kind == "ddpgbc":
            dist = self.model.gaussian_dist(obs, goals)
            a_pi = torch.clamp(dist.mean if self.const_std else dist.rsample(), -1.0, 1.0)
            q1, q2 = self.density_actor_q(batch, obs, a_pi, goals)
            q = torch.minimum(q1, q2) if self.double_q else q1
            q_scale = (q - q.mean()).abs().mean()                      # actor_q_center
            q_loss = -q.mean() / (q_scale.detach() + 1e-6)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            bc_loss = -(self.alpha * log_prob).mean()
            loss = q_loss + bc_loss
            return loss, {"actor_loss": float(loss.detach()), "q_loss": float(q_loss.detach()),
                          "bc_loss": float(bc_loss.detach()), "q_mean": float(q.mean().detach()),
                          "mse": float(((dist.mean - actions) ** 2).mean().detach())}

        # flow: 1) flow-matching BC on the goal-agnostic vf
        x1 = actions
        x0 = torch.randn_like(x1)
        tt = torch.rand((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
        x_t = (1 - tt) * x0 + tt * x1
        vel = x1 - x0
        pred = self.model._actor_vf(obs, x_t, tt)
        bc_flow_loss = ((pred - vel) ** 2).mean()

        # 2) Q-max on the one-shot actor's actions (towers not stepped here)
        noise = torch.randn_like(actions)
        actor_actions = self.model.one_shot_action(obs, goals, noise)
        q1, q2 = self.density_actor_q(batch, obs, actor_actions, goals)
        q = torch.minimum(q1, q2) if self.double_q else q1
        loss = -q.mean()

        # 3) FQL distillation to the frozen flow rollout + centered q-scale
        distill_error = torch.zeros((), device=obs.device)
        if self.bc_coeff > 0:
            target = self._flow_rollout(obs, noise)                    # no grad
            distill_error = ((actor_actions - target) ** 2).mean()
            bc_loss = self.bc_coeff * distill_error
            q_scale = ((q - q.mean()).abs().mean() if self.actor_q_center
                       else q.abs().mean())
            loss = loss / (q_scale.detach() + 1e-6) + bc_loss
        loss = loss + bc_flow_loss

        return loss, {"actor_loss": float(loss.detach()),
                      "bc_flow_loss": float(bc_flow_loss.detach()),
                      "distill_loss": float(distill_error.detach()),
                      "q_mean": float(q.mean().detach()),
                      "q_abs_mean": float(q.abs().mean().detach())}

    # ── update / act / readout ───────────────────────────────────────────────

    def update(self, batch: Dict[str, Tensor], step: int) -> Dict[str, float]:
        dev = self.device
        batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        ce, m_info = self.density_loss(batch)
        self.f_optimizer.zero_grad(set_to_none=True)
        self.b_optimizer.zero_grad(set_to_none=True)
        ce.backward()
        self.f_optimizer.step()
        self.b_optimizer.step()

        a_loss, a_info = self.actor_loss(batch)
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.actor_vf_optimizer is not None:
            self.actor_vf_optimizer.zero_grad(set_to_none=True)
        a_loss.backward()
        self.actor_optimizer.step()
        if self.actor_vf_optimizer is not None:
            self.actor_vf_optimizer.step()

        soft_update(self.model.density_f, self.model.target_density_f, self.target_tau)
        soft_update(self.model.density_b, self.model.target_density_b, self.target_tau)
        return {**m_info, **a_info}

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        """z slot carries the raw goal observation (actor_cond='goal')."""
        obs = obs.to(self.device).float()
        goal = z.to(self.device).float()
        if obs.dim() == 1:
            obs, goal = obs[None], goal[None]
            return self.model.act(obs, goal)[0]
        return self.model.act(obs, goal)

    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        raise NotImplementedError(
            "Density-FB has no linear z_r = E[r B] inverse (the softmax removes "
            "FB's linear identity — spec 28); use zero_shot_q with reward samples.")

    # zero-shot reward evaluation ------------------------------------------------
    @torch.no_grad()
    def zero_shot_q(self, obs: Tensor, action: Tensor, goal: Tensor,
                    pool_obs: Tensor, r_pool: np.ndarray, r_x0: float) -> float:
        """De-biased zero-shot readout at a single (s, a, g). pool_obs are FULL
        observation rows for the pool (raw); kappa is evaluated at the CURRENT
        state obs against the pool, exactly as eval_density_zeroshot.py."""
        u_raw = raw_block_cols(pool_obs, self.blocks)
        u_sc = scaled_block_cols(pool_obs, self.blocks)
        b_pool = self.model.density_b(u_sc).cpu().numpy()              # [P, d]
        f_emb = self.model.density_f(
            self.model.f_input(obs[None], action[None], goal[None]))[:, 0]  # [2, d]
        w = density_weights(f_emb.cpu().numpy(), b_pool, self.temp, self.double_q)
        kappa = delta_phi(obs[None], u_raw, self.blocks)[0].cpu().numpy()
        return compute_density_primary(r_x0, w, kappa, np.asarray(r_pool), self.discount)

    # persistence ----------------------------------------------------------------
    def state_dict(self) -> dict:
        s = {"model": self.model.state_dict(),
             "f_optimizer": self.f_optimizer.state_dict(),
             "b_optimizer": self.b_optimizer.state_dict(),
             "actor_optimizer": self.actor_optimizer.state_dict()}
        if self.actor_vf_optimizer is not None:
            s["actor_vf_optimizer"] = self.actor_vf_optimizer.state_dict()
        return s

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.f_optimizer.load_state_dict(state["f_optimizer"])
        self.b_optimizer.load_state_dict(state["b_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if self.actor_vf_optimizer is not None and "actor_vf_optimizer" in state:
            self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
