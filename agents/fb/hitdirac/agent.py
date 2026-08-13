"""agents/fb/hitdirac/agent.py — the DECONFOUNDING CONTROL for the torch-FB
bounded-measure result (2026-07-28).

The phase-1 cell that scored ~0.93 on cube-single changed TWO things at once
relative to stock FB:

  (A) the REWARD SUPPORT: FB's Dirac on the exact next state (the identity
      diagonal of the [B,B] goal matrix) became a privileged cube-xyz BALL
      (hit[i,j] = 1[||cube(s'_i) - cube(s'_j)|| <= eps]) whose radius matches
      the eval reward's 4cm threshold (envs/ogbench.py cube_reward_fn);
  (B) the LOSS FORM: FB's quadratic-offdiag + LINEAR-diag split (+ ortho,
      unbounded fixed point) became an every-entry-squared bounded residual
      with NO ortho.

Only (B) is a novel claim. This agent isolates (A): **FB's exact loss, with the
reward support set generalized from I to the hit ball.** Ortho stays ON, the
diagonal term stays LINEAR, the fixed point stays unbounded — everything that
makes FB "FB" is retained; only the set on which the Dirac reward is placed
changes.

Reading the 3-arm comparison (all cube-single, same eval):
    stock FB (hit = I)          0.42   <- results/summary_agents.csv anchor
    THIS control (hit = ball)   ?
    bounded/no-ortho + ball     0.93
  - control ~= 0.9  => the privileged ball did the work; the bounded objective
    is not the story, and the "successor-measure-that-works" claim collapses.
  - control ~= 0.42 => the ball alone is insufficient; the bounded/no-ortho
    objective is doing real work and the claim survives.

EXACT REDUCTION: with hit = I this agent's fb_loss is algebraically identical to
FBFlowBCAgent's (see tests/test_fb_hitdirac.py) --
    offdiag: nonhit = 1-I = off_diag, nonhit.sum() = off_diag_sum
    diag:    -(diff*I).sum()/I.sum() = -sum(diag)/batch
             = -diagonal.mean() * n_parallel   (the parent's explicit factor)
so the control is a strict generalization along one axis, not a reimplementation.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.fb.agent import ortho_cov
from agents.fb.flow_bc.agent import FBFlowBCAgent


class FBHitDiracFlowBCAgent(FBFlowBCAgent):
    def __init__(self, *args, hit_eps: float = 0.4, hit_slice=(19, 22),
                 hit_spec: str = "", **kwargs):
        self._init_hit_spec(hit_spec, hit_slice=hit_slice, hit_eps=hit_eps)
        self._raw_next_obs = None
        super().__init__(*args, **kwargs)

    def _init_hit_spec(self, hit_spec: str = "", *, hit_slice=(19, 22),
                       hit_eps: float = 0.4) -> None:
        """Resolve the reward support: either the historical single ball, or the
        conjunctive `hit_spec` the multi-object envs need.

        Same grammar and same conjunction semantics as
        FBMeasureFlowBCAgent.__init__ ("a:b:eps,..."), because the loss-form x
        support 2x2 is only interpretable when both arms query the IDENTICAL
        region. Empty spec is byte-identical to the previous behaviour.
        """
        self.hit_eps = float(hit_eps)
        self.hit_slice = tuple(int(x) for x in hit_slice)
        assert len(self.hit_slice) == 2 and 0 <= self.hit_slice[0] < self.hit_slice[1]
        self.hit_blocks = None
        if str(hit_spec).strip():
            blocks = []
            for part in str(hit_spec).split(","):
                part = part.strip()
                if not part:
                    continue
                a, b, e = part.split(":")
                a, b, e = int(a), int(b), float(e)
                assert 0 <= a < b and e > 0.0, f"bad hit_spec block {part!r}"
                blocks.append((a, b, e))
            assert blocks, f"empty hit_spec {hit_spec!r}"
            self.hit_blocks = tuple(blocks)

    def hit_matrix(self, raw_next_obs: torch.Tensor) -> torch.Tensor:
        """[B,B] hit[i,j] = 1[||c(s'_i) - c(s'_j)|| <= eps] over the RAW obs
        slice — identical to FBMeasureFlowBCAgent.hit_matrix so the two arms
        share one reward-support definition. Diagonal is always 1.

        With `hit_spec` set, the hit is the CONJUNCTION over blocks: every object
        must be within its own radius, matching the manipspace success criteria.
        """
        if self.hit_blocks is not None:
            hit = None
            for a, b, e in self.hit_blocks:
                x = raw_next_obs[:, a:b]
                h = (torch.cdist(x, x) <= e)
                hit = h if hit is None else (hit & h)
            return hit.float()
        a, b = self.hit_slice
        cube = raw_next_obs[:, a:b]
        return (torch.cdist(cube, cube) <= self.hit_eps).float()

    def update(self, batch, step):
        # Stash RAW next_obs — the ball radius lives in the obs frame while
        # update_fb receives normalized tensors (the measure-agent pattern).
        self._raw_next_obs = batch["next"]["observation"].to(self.device).float()
        return super().update(batch, step)

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
        w: torch.Tensor | None = None,
        weight_diag: bool = False,
        next_action_override: torch.Tensor | None = None,
        goal_obs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        # Arms this control deliberately does not carry (fail loud, ISFB discipline).
        assert w is None, "fb_hitdirac ignores reweighting (reweight_alpha must be 0)"
        assert goal_obs is None, "fb_hitdirac ignores goal_cond (goal_cond must be False)"
        assert q_loss_coef is None, "fb_hitdirac has no q_loss (q_loss_coef must be 0)"
        assert self._raw_next_obs is not None, "update_fb called outside update()"
        with torch.no_grad():
            hit = self.hit_matrix(self._raw_next_obs).to(obs.dtype)   # [B,B]
            nonhit = 1.0 - hit
            hit_sum = hit.sum().clamp_min(1.0)
            nonhit_sum = nonhit.sum().clamp_min(1.0)

        zf = self._fb_z(z)
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                next_action = (next_action_override if next_action_override is not None
                               else self.sample_action_from_norm_obs(actor_in, z))
                target_Fs = self.model._target_forward_map(next_left_enc, zf, next_action)
                target_B = self.model._target_backward_map(goal)
                target_Ms = torch.matmul(target_Fs, target_B.T)
                _, _, target_M = self.get_targets_uncertainty(
                    target_Ms, self.fb_pessimism_penalty)             # [B,B]

            left_enc = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left_enc, zf, action)        # [P,B,z]
            B = self.model._backward_map(goal)                        # [B,z]
            Ms = torch.matmul(Fs, B.T)                                # [P,B,B]

            # ---- FB's OWN loss, with the Dirac support I -> hit ----------------
            diff = Ms - discount * target_M
            fb_offdiag = 0.5 * (diff * nonhit).pow(2).sum() / nonhit_sum
            # LINEAR in diff on the reward support (FB's -E[F^T B(s')] term). The
            # sum already runs over the P heads, so it carries the parent's
            # explicit *n_parallel factor implicitly.
            fb_diag = -(diff * hit).sum() / hit_sum
            fb_loss = fb_offdiag + fb_diag

            # ortho stays ON — this is stock FB behavior (fixed_b=='none' branch).
            if self.fixed_b == "none":
                Cov = ortho_cov(B, None)
                orth_loss_diag = -Cov.diag().mean()
                orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
                orth_loss = orth_loss_offdiag + orth_loss_diag
                fb_loss = fb_loss + self.ortho_coef * orth_loss
            else:
                orth_loss = orth_loss_diag = orth_loss_offdiag = torch.zeros(
                    1, device=z.device, dtype=z.dtype)

        self.forward_optimizer.zero_grad(set_to_none=True)
        if self.backward_optimizer is not None:
            self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()

        with torch.no_grad():
            return {
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "hit_rate": hit.mean(),
                "ms_abs_mean": Ms.abs().mean(),   # unbounded-fixed-point tell
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
            }
