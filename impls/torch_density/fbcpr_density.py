"""DensityFBcprAgent — Meta Motivo's FB-CPR trainer with ONLY the objective swapped.

WHAT THIS IS (2026-08-27 user directive: "run exactly what metamotivo ran, just with the
density objective"). A subclass of `metamotivo.fb_cpr.agent.FBcprAgent` that overrides two
methods and nothing else:

  * `update_fb` -> the density successor-measure loss (impls/critics/density_fb.py
    stage_loss, hard mode): per F head, softmax over the batch's next-state queries,
    cross-entropy against the stop-gradded TD target
        t = (1 - gamma_row) * I  +  gamma_row * softmax(F^-(s', z, a+) B^-(u)^T / tau)
    with a+ their own actor sample (TruncatedNormal, clip stddev_clip). Replaces their
    contrastive M loss + orthonormality + aux Q. Optimizer scaffolding copied verbatim.
  * `update_actor` -> theirs verbatim except Q_fb: instead of F^T z it is the density
    readout q_h = sum_u softmax(F B(u)^T / tau) * r_z(u), r_z(u) = B^-(u)^T z / sqrt(d),
    min over the two heads (== their pessimism penalty 0.5 at num_parallel=2). The
    discriminator term, reg_coeff, scale_reg weighting, optimizer and clipping are theirs.

EVERYTHING ELSE IS INHERITED UNCHANGED: the discriminator + WGAN gradient penalty, AMASS
expert buffer and its B-encoding (`encode_expert`), the z mix (train_goal_ratio /
expert_asm_ratio / uniform), relabel_ratio, the CPR critic, obs normalizer, target updates,
architectures, and the online training loop in examples/fbcpr_train_humenv.py.

DECLARED DIFFERENCES FROM OUR JAX HUMENV ARM (impls/critics/density_fb.py):
  * z channel is THEIRS — B(goal) encodings + expert-trajectory encodings + uniform sphere.
    The jax arm's kernel-index z (measure_hit_spec, zenc/zstats) is NOT used; in hard mode
    the density target is the identity over batch queries, so no hit radius exists here.
  * No centered actor-q normalizer — their actor loss normalization (scale_reg on the
    discriminator term) is kept instead.
  * B input is the normalized full observation (their _backward_map), not eps-scaled
    object coordinates.

Torch port of the loss verified by tests/impls/test_torch_density_parity.py against a
numpy transcription of the jax stage_loss math (run in .venv-humenv; that venv has no
jax, so the reference is numpy mirroring impls/critics/density_fb.py stage_loss
line-for-line — it checks the torch axes/broadcasting, which is where a port breaks).

Knobs (env vars, so their config dataclasses stay untouched):
  FB_DENSITY_TEMP    softmax temperature tau (default 1.0 — the humenv jax arm's value)
"""
import os

import torch
import torch.nn.functional as F  # noqa: N812

from metamotivo.fb_cpr.agent import FBcprAgent


def density_ce_loss(Fs, B_emb, target_Fs, target_B, discount, tau):
    """The density measure loss on plain tensors (shared with the parity test).

    Fs, target_Fs: [P, B, d]; B_emb, target_B: [M, d] (M == B, the batch's own next
    states); discount: [B, 1] (gamma * ~terminated, row-wise). -> (ce, info dict)."""
    with torch.no_grad():
        w_bar = torch.softmax(torch.matmul(target_Fs, target_B.T) / tau, dim=-1)  # P x B x M
        eye = torch.eye(Fs.shape[1], device=Fs.device, dtype=Fs.dtype)
        t = (1.0 - discount) * eye + discount * w_bar                              # P x B x M
    logits = torch.matmul(Fs, B_emb.T) / tau
    logp = torch.log_softmax(logits, dim=-1)
    ce = -(t * logp).sum(-1).mean()
    with torch.no_grad():
        w = torch.softmax(logits, dim=-1)
        ess_rows = 1.0 / (w**2).sum(-1)                                            # P x B
    return ce, {
        "density_ce": ce.detach(),
        "density_ess": ess_rows.mean(),
        "density_ess_median": ess_rows.median(),
    }


class DensityFBcprAgent(FBcprAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.density_temp = float(os.environ.get("FB_DENSITY_TEMP", "1.0"))

    def update_fb(self, obs, action, discount, next_obs, goal, z, q_loss_coef, clip_grad_norm):
        # a+ from THEIR actor sampling law (TruncatedNormal around the mean, clipped).
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # P x B x d
            target_B = self._model._target_backward_map(goal)                      # B x d
        Fs = self._model._forward_map(obs, z, action)                              # P x B x d
        B_emb = self._model._backward_map(goal)                                    # B x d
        ce, info = density_ce_loss(Fs, B_emb, target_Fs, target_B, discount, self.density_temp)

        # Their optimizer scaffolding, verbatim (q_loss_coef is FB's aux loss — the density
        # objective has no counterpart, so the knob is deliberately ignored).
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        ce.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        # Queries for update_actor's density readout: this update's own (normalized) next
        # states. update() calls update_fb before update_actor, so the stash is always
        # fresh; detached — the actor step must not backprop into the query tensor's
        # producer.
        self._density_queries = next_obs.detach()
        return {k: v for k, v in info.items()}

    def update_actor(self, obs, action, z, clip_grad_norm):
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)

        # Discriminator term: theirs verbatim.
        Qs_discriminator = self._model._critic(obs, z, action)
        _, _, Q_discriminator = self.get_targets_uncertainty(
            Qs_discriminator, self.cfg.train.actor_pessimism_penalty)

        # Density readout in place of F^T z. r_z on the TARGET backward tower (the jax
        # q_fn's frozen-params discipline); min over heads == their pessimism 0.5 at P=2.
        u = self._density_queries                                                  # M x obs
        tau = self.density_temp
        with torch.no_grad():
            tb = self._model._target_backward_map(u)                               # M x d
            r_z = torch.matmul(z, tb.T) / (float(z.shape[-1]) ** 0.5)              # B x M
        Fs = self._model._forward_map(obs, z, action)                              # P x B x d
        B_u = self._model._backward_map(u).detach()                                # M x d
        w = torch.softmax(torch.matmul(Fs, B_u.T) / tau, dim=-1)                   # P x B x M
        Q_fb = (w * r_z.unsqueeze(0)).sum(-1).min(dim=0).values                    # B

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            return {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
