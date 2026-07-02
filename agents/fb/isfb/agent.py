"""agents/fb/isfb/agent.py — SG-ISFB agent (PAPER/fb-modified/isfb.tex).

Subclasses FBFlowBCAgent (flow actor + z-cond inherited UNCHANGED). The ONLY
addition over standard FB is the critic update: the successor measure M is trained
with IQL's in-sample improvement operator (scalar expectile L_V on q=<F,z> vs
v=<F_V,z>; advantage-weighted vector regression L_N fitting N=<F_V,B> to sg M;
measure TD L_M bootstrapping through N^- at the next state, NO a'~pi)."""
from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.fb.agent import expectile_loss, fb_successor_terms, ortho_cov
from agents.fb.flow_bc.agent import FBFlowBCAgent
from agents.fb.isfb.model import ISFBModel
from nn_models import _soft_update_params


def isfb_hard_weights(A: torch.Tensor, tau: float) -> torch.Tensor:
    """tau-hard expectile weights: tau where advantage>0, (1-tau) elsewhere."""
    return torch.where(A > 0, torch.as_tensor(tau, dtype=A.dtype, device=A.device),
                       torch.as_tensor(1.0 - tau, dtype=A.dtype, device=A.device))


class ISFBAgent(FBFlowBCAgent):
    def __init__(self, *args, measure_insample: str = "scalar_gate_hard",
                 alpha_n: float = 1.0, alpha_m: float = 1.0,
                 awr_beta: float = 3.0, awr_wmax: float = 100.0, **kwargs):
        # Stash ISFB knobs BEFORE super().__init__ (which calls setup_training()).
        self.measure_insample = str(measure_insample)
        self.alpha_n = float(alpha_n)
        self.alpha_m = float(alpha_m)
        self.awr_beta = float(awr_beta)
        self.awr_wmax = float(awr_wmax)
        super().__init__(*args, **kwargs)

    def _make_model(self, **kwargs):
        return ISFBModel(actor_vf_cfg=self._actor_vf_cfg, **kwargs)

    def setup_training(self) -> None:
        super().setup_training()
        # F_V rides the forward optimizer (lr_f), matching FB's optimizer grouping.
        self.forward_optimizer.add_param_group(
            {"params": list(self.model._value_forward_map.parameters())}
        )
        self._value_forward_map_paramlist = tuple(self.model._value_forward_map.parameters())
        self._target_value_forward_map_paramlist = tuple(self.model._target_value_forward_map.parameters())

    def update_fb(self, obs, action, discount, next_obs, goal, z,
                  q_loss_coef, clip_grad_norm, w=None, weight_diag=False,
                  next_action_override=None, goal_obs=None) -> Dict[str, torch.Tensor]:
        tau = self.iql_expectile
        zero_a = torch.zeros_like(action)
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            # target value successor measure N^- at the next state (NO a'~pi).
            with torch.no_grad():
                next_left = self.model._target_left_encoder(next_obs)
                tFv = self.model._target_value_forward_map(next_left, z, zero_a)  # [P,B,z]
                tB = self.model._target_backward_map(goal)                        # [B,z]
                tN = torch.matmul(tFv, tB.T)                                      # [P,B,B]
                _, _, tN_M = self.get_targets_uncertainty(tN, self.fb_pessimism_penalty)  # [B,B]

            left = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left, z, action)          # [P,B,z]
            Fv = self.model._value_forward_map(left, z, zero_a)    # [P,B,z]
            B = self.model._backward_map(goal)                     # [B,z]
            M = torch.matmul(Fs, B.T)                              # [P,B,B]
            N = torch.matmul(Fv, B.T)                              # [P,B,B]

            # scalar readouts, matching the flow actor's Q exactly (no /sqrt(d)).
            q = (Fs * z).sum(-1)                                   # [P,B]
            v = (Fv * z).sum(-1)                                   # [P,B]

            # (1) scalar gate: IQL expectile value loss (the only action-selection operator).
            L_V = expectile_loss(q.detach() - v, tau)
            A = (q - v).detach().mean(0)                           # head-mean advantage [B]

            # (2) improve the vector value measure N under the scalar gate.
            if self.measure_insample == "coord":
                L_N = expectile_loss(M.detach() - N, tau)          # elementwise expectile of M
            else:
                if self.measure_insample == "scalar_gate_awr":
                    wgt = torch.clamp(torch.exp(self.awr_beta * A), 0.0, self.awr_wmax)
                else:  # scalar_gate_hard
                    wgt = isfb_hard_weights(A, tau)
                wbar = wgt / (wgt.mean() + 1e-6)                   # [B]
                L_N = (wbar.view(1, -1, 1) * (M.detach() - N).pow(2)).mean()

            # (3) measure TD: M bootstraps through the target value measure at s' (no a'~pi).
            diff = M - discount * tN_M                             # [P,B,B]
            fb_offdiag, fb_diag = fb_successor_terms(diff, self.off_diag, self.off_diag_sum)
            L_M = fb_offdiag + fb_diag

            # (4) FB orthonormality on B (coef 1.0, unchanged from standard FB).
            Cov = ortho_cov(B, None)
            orth_loss = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum - Cov.diag().mean()

            loss = self.alpha_n * L_N + self.alpha_m * L_M + self.ortho_coef * orth_loss
            if self.measure_insample != "coord":
                loss = loss + L_V

        self.forward_optimizer.zero_grad(set_to_none=True)
        if self.backward_optimizer is not None:
            self.backward_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._value_forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()

        with torch.no_grad():
            _soft_update_params(self._value_forward_map_paramlist,
                                self._target_value_forward_map_paramlist, self.f_target_tau)

        return {
            "L_V": (L_V.detach() if self.measure_insample != "coord" else torch.zeros((), device=z.device)),
            "L_N": L_N.detach(), "L_M": L_M.detach(), "orth_loss": orth_loss.detach(),
            "adv_mean": A.mean().detach(), "M_mean": M.mean().detach(),
        }
