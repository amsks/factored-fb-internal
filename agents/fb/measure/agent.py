"""agents/fb/measure/agent.py — the two-phase BOUNDED successor-measure objective
in FB's own house (torch port of the jax gciql_flow recipe that produced the
FP-NoOrtho 0.967 / phase-2 ball-visitation cells).

Subclasses FBFlowBCAgent (flow actor, z-sampling, target nets, soft updates all
inherited UNCHANGED — the actor stays greedy on <F,z>, which IS the actor_z
objective natively). The ONLY change is update_fb: the density residual
diff = Ms - discount*target_M with FB's quadratic-offdiag + LINEAR-diag split is
replaced by the bounded hit-referenced residual with EVERY entry squared:

    first_passage (phase 1):  diff = Ms - hit - discount*(1-hit)*sg[target_M]
    visitation    (phase 2):  diff = Ms - hit - discount*sg[target_M]
    loss = mean(diff^2)  [/ sg[mean(Ms^2)]+1e-6 with scale_norm]

where hit[i,j] = 1[||cube(next_obs_i) - cube(next_obs_j)|| <= hit_eps] over the
RAW (un-normalized) obs cube slice — the goal columns ARE the batch next_obs
(FB's native n x n structure), and eps lives in obs units, so the raw tensor is
sliced (the encoder is Identity for state runs; slicing raw is explicit and
pixel-safe-by-refusal). hit is a REPRESENTABLE 0/1 reward: the diagonal is a
plain regressed residual (no linear attraction => no unbounded diagonal pull),
the per-column fixed point is bounded ([0,1] for first-passage,
<= 1/(1-gamma) for visitation). NO ortho term (ortho was the toxin in the jax
cells); NO q_loss.

Phase-2 knobs:
    freeze_b       requires_grad_(False) on _backward_map + _bw_encoder and
                   backward_optimizer=None (the parent code is None-tolerant)
                   — B frozen at the warm-started phase-1 geometry.
    nhat_head      phase 2b: N-hat(g) = softplus(MLP 2x256(B(g).detach()))
                   regressed onto sg[hit column mean] (unbiased MC of the goal
                   ball's data mass). The head RIDES the forward optimizer as
                   an extra param group (lr_f — the ISFB add_param_group
                   pattern; documented choice: one optimizer step path, no
                   second Adam to checkpoint). Gradient-isolated (input and
                   target both detached).
    warm_start_ckpt  path to a stock-FB / phase-1 step_*.pt: MODEL weights
                   loaded strict=False (missing-local-modules refused, extra
                   checkpoint modules ignored — the evals/analysis.py
                   pattern); optimizers stay FRESH; the nhat head (which lives
                   outside self.model) is always fresh.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.agent import FBFlowBCAgent


class RegionMeasureMixin:
    """The region-occupancy critic, independent of the actor.

    Everything below overrides the CRITIC side only (`update_fb`, `hit_matrix`,
    plus init / checkpoint plumbing); the actor side (`update_actor`, `act`,
    `_make_model`) is left to whatever class this is mixed in front of. The two
    concrete arms at the bottom of the file compose it with the flow-BC actor
    (the historical stack — MRO puts this mixin exactly where the subclass body
    used to be, so resolution is unchanged) and with FBAgent's native TD3+BC
    actor (2026-08-05, to test whether the flow actor helps or hurts).

    Every `super()` below is cooperative and resolves through the MRO, so the
    same code serves both compositions.
    """

    def __init__(self, *args, measure_mode: str = "first_passage",
                 hit_eps: float = 0.4, hit_slice=(19, 22),
                 hit_identity: bool = False,
                 scale_norm: bool = True, freeze_b: bool = False,
                 nhat_head: bool = False, warm_start_ckpt: str = "",
                 nonneg_coef: float = 0.0, mass_coef: float = 0.0,
                 hit_kernel: str = "ball", hit_spec: str = "", **kwargs):
        assert measure_mode in ("first_passage", "visitation"), measure_mode
        # Stash knobs BEFORE super().__init__ (which calls setup_training()).
        self.measure_mode = str(measure_mode)
        self.hit_eps = float(hit_eps)
        self.hit_slice = tuple(int(x) for x in hit_slice)
        assert len(self.hit_slice) == 2 and 0 <= self.hit_slice[0] < self.hit_slice[1]
        # hit_identity: the bounded objective on FB's OWN Dirac support (hit = I),
        # i.e. the reward event is "the query state IS the observed successor".
        # This is the missing cell of the loss-form x reward-support 2x2 and the
        # test of whether the bounded objective needs a privileged region at all.
        # NOTE hit_eps -> 0 does NOT approximate this: the cube is stationary for
        # long stretches of play, so exact cube-position duplicates are common
        # (6.6% of off-diagonal pairs), and float32 cdist leaves ~7% of the
        # diagonal above a zero threshold. The identity is constructed, not
        # approached (tests/test_hit_identity.py).
        self.hit_identity = bool(hit_identity)
        self.scale_norm = bool(scale_norm)
        self.freeze_b = bool(freeze_b)
        self.use_nhat_head = bool(nhat_head)
        self.warm_start_ckpt = str(warm_start_ckpt or "")
        self.nonneg_coef = float(nonneg_coef)
        assert self.nonneg_coef >= 0.0, "nonneg_coef must be >= 0"
        # mass_coef (2026-07-29, ledger §9.15): total-mass conservation anchor.
        # The hinge alone fixed sign but inflated set mass 2.5-3.6x (§9.9)
        # because nothing conserves total mass. For the VISITATION fixed point
        # E_{g~rho}[M(s,g)/rho(B_eps(g))] = sum_t gamma^t = 1/(1-gamma) exactly
        # (the cube is always somewhere); the in-batch estimator uses the hit
        # column mean as N-hat, clamped at 2/B so a lone-diagonal column cannot
        # dominate the importance weight. Default 0.0 is bit-identical.
        self.mass_coef = float(mass_coef)
        assert self.mass_coef >= 0.0, "mass_coef must be >= 0"
        # hit_kernel (2026-07-29, ledger §9.15): 'ball' = the historical 0/1
        # kernel; 'quartic' = biweight (1-(d/h)^2)^2 with support h=hit_eps.
        # D1 showed the negativity is truncated-basis RINGING of a rank-d fit
        # to a boxcar; a C^1 kernel has less high-frequency content at the
        # same rank. The hit column mean stays the (soft) kernel mass, so
        # N-hat, the nhat_head and every readout generalize unchanged.
        self.hit_kernel = str(hit_kernel)
        assert self.hit_kernel in ("ball", "quartic"), self.hit_kernel
        assert not (self.hit_identity and self.hit_kernel != "ball"), \
            "hit_identity and a soft hit_kernel are mutually exclusive"
        # hit_spec (2026-07-30): multi-env hit indicator, mirroring the jax
        # `measure_hit_spec` (spec 08 E7). "a:b:eps,..." observation blocks,
        # combined as a CONJUNCTION (all blocks within their radii), because
        # manipspace success criteria are conjunctions:
        #   cube-double '19:22:0.4,28:31:0.4'
        #   scene       '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'
        # Empty string (default) = the historical single (hit_slice, hit_eps)
        # ball, byte-identical. A STRING, not a list (hydra/CLI friendliness,
        # same lesson as the jax knob).
        self.hit_blocks = None
        if str(hit_spec).strip():
            assert not self.hit_identity, "hit_spec and hit_identity are exclusive"
            assert self.hit_kernel == "ball", \
                "hit_spec currently supports the ball kernel only"
            blocks = []
            for part in str(hit_spec).split(","):
                a, b, e = part.strip().split(":")
                a, b, e = int(a), int(b), float(e)
                assert 0 <= a < b and e > 0.0, f"bad hit_spec block {part!r}"
                blocks.append((a, b, e))
            assert blocks
            self.hit_blocks = tuple(blocks)
        self._raw_next_obs = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------- setup ----
    def setup_training(self) -> None:
        super().setup_training()
        if self.use_nhat_head:
            zd = self.z_dim
            self._nhat_head = torch.nn.Sequential(
                torch.nn.Linear(zd, 256), torch.nn.GELU(),
                torch.nn.Linear(256, 256), torch.nn.GELU(),
                torch.nn.Linear(256, 1), torch.nn.Softplus(),
            ).to(self.device)
            # ride the forward optimizer (lr_f) as its own param group — the
            # ISFB _value_forward_map pattern; no second Adam to checkpoint.
            self.forward_optimizer.add_param_group(
                {"params": list(self._nhat_head.parameters())})
        if self.warm_start_ckpt:
            self._load_warm_start(self.warm_start_ckpt)
        if self.freeze_b:
            # Phase 2: B (and its encoder) frozen at the warm-started geometry.
            # The parent's optimizer/step/soft-update code is None-tolerant.
            self.model._backward_map.requires_grad_(False)
            self.model._bw_encoder.requires_grad_(False)
            self.backward_optimizer = None
            print("[fb_measure] freeze_b: _backward_map + _bw_encoder frozen, "
                  "backward_optimizer=None", flush=True)

    def _load_warm_start(self, path: str) -> None:
        """Model-weights-only warm start (fresh optimizers), strict=False in the
        evals/analysis.load_checkpoint style: extra checkpoint modules are
        ignored, but every parameter THIS model needs must be present."""
        state = torch.load(str(path), map_location=self.device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state \
            else state
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        assert not missing, (f"warm_start_ckpt does not match the model: "
                             f"{len(missing)} missing, e.g. {missing[:5]}")
        print(f"[fb_measure] warm-started model from {path}; ignored "
              f"{len(unexpected)} extra checkpoint key(s); optimizers FRESH; "
              f"nhat_head {'fresh-init' if self.use_nhat_head else 'absent'}",
              flush=True)

    # ------------------------------------------------------- checkpointing ----
    def state_dict(self) -> dict:
        s = super().state_dict()
        if self.use_nhat_head:
            s["nhat_head"] = self._nhat_head.state_dict()
        return s

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if self.use_nhat_head and state.get("nhat_head") is not None:
            self._nhat_head.load_state_dict(state["nhat_head"])

    # ------------------------------------------------------------- update ----
    def update(self, batch, step):
        # Stash the RAW next_obs (obs units) — the hit ball radius lives in the
        # obs frame, while update_fb receives normalized/encoded tensors (the
        # ISFB raw-batch-stash pattern).
        self._raw_next_obs = batch["next"]["observation"].to(self.device).float()
        return super().update(batch, step)

    def hit_matrix(self, raw_next_obs: torch.Tensor) -> torch.Tensor:
        """[B,B] hit[i,j] = 1[||cube(s'_i) - cube(s'_j)|| <= eps] over the raw
        obs cube slice. Columns j ARE the batch next_obs (FB's native goal
        columns), so the diagonal is always 1.

        hit_identity=True returns exactly I instead: FB's Dirac support, in which
        only the observed successor counts as its own arrival."""
        if self.hit_identity:
            n = raw_next_obs.shape[0]
            return torch.eye(n, device=raw_next_obs.device, dtype=torch.float32)
        if self.hit_blocks is not None:
            # conjunction over blocks: hit iff EVERY block is within its radius
            hit = None
            for a, b, e in self.hit_blocks:
                x = raw_next_obs[:, a:b]
                h = (torch.cdist(x, x) <= e)
                hit = h if hit is None else (hit & h)
            return hit.float()
        a, b = self.hit_slice
        cube = raw_next_obs[:, a:b]
        if self.hit_kernel == "quartic":
            u2 = (torch.cdist(cube, cube) / self.hit_eps).pow(2)
            return (1.0 - u2).clamp_min(0.0).pow(2).float()
        return (torch.cdist(cube, cube) <= self.hit_eps).float()

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
        # The measure objective intentionally supports none of these arms; fail
        # loud rather than silently drop them (the ISFB discipline).
        assert w is None, "fb_measure ignores reweighting (reweight_alpha must be 0)"
        assert goal_obs is None, "fb_measure ignores goal_cond (goal_cond must be False)"
        assert q_loss_coef is None, "fb_measure has no q_loss (q_loss_coef must be 0)"
        assert self._raw_next_obs is not None, "update_fb called outside update()"
        with torch.no_grad():
            hit = self.hit_matrix(self._raw_next_obs).to(obs.dtype)          # [B,B]

        zf = self._fb_z(z)
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                next_action = (next_action_override if next_action_override is not None
                               else self.sample_action_from_norm_obs(actor_in, z))
                target_Fs = self.model._target_forward_map(next_left_enc, zf, next_action)
                target_B = self.model._target_backward_map(goal)             # [B,z]
                target_Ms = torch.matmul(target_Fs, target_B.T)              # [P,B,B]
                # keep the parent's P-head pessimism/target-min machinery
                _, _, target_M = self.get_targets_uncertainty(
                    target_Ms, self.fb_pessimism_penalty)                    # [B,B]

            left_enc = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left_enc, zf, action)               # [P,B,z]
            B = self.model._backward_map(goal)                               # [B,z]
            Ms = torch.matmul(Fs, B.T)                                       # [P,B,B]

            if self.measure_mode == "first_passage":
                # absorbing: a hit terminates the recursion; fixed point
                # E[gamma^T_hit] in [0,1] per column.
                diff = Ms - hit - discount * (1.0 - hit) * target_M
            else:  # visitation: revisits counted; fixed point <= 1/(1-gamma)
                diff = Ms - hit - discount * target_M
            # EVERY entry squared — no linear diagonal term, no ortho.
            measure_loss = diff.pow(2).mean()
            # One divisor shared by every term so their ratio is scale-free
            # (the jax measure_scale_norm mirror); 1.0 when scale_norm is off.
            denom = (Ms.detach().pow(2).mean() + 1e-6) if self.scale_norm else 1.0
            if self.scale_norm:
                measure_loss = measure_loss / denom
            fb_loss = measure_loss

            # --- non-negativity (2026-07-28) ------------------------------------
            # A successor measure is non-negative by definition, but nothing in the
            # residual enforces it: the bounded target bounds the FIXED POINT, while
            # the bilinear F^T psi is free to go negative wherever the regression is
            # not pinned by data. Measured on the phase-2 checkpoints, ~45% of M is
            # negative, which is why the set-mass readout is 65-215x off with rank
            # correlation ~0.2 (spec 10 L1). A one-sided squared hinge penalizes only
            # the negative part, so it adds NOTHING where the model is already valid
            # and leaves the BILINEAR form intact -- the zero-shot readout
            # Q_r = <F, z_r> with z_r = E[r psi] depends on M being linear in psi, so
            # squashing M (softplus/square) would break the FB API. Default 0.0 is
            # bit-identical to the pre-existing objective.
            nonneg_loss = torch.relu(-Ms).pow(2).mean()
            if self.scale_norm:
                nonneg_loss = nonneg_loss / denom
            if self.nonneg_coef > 0.0:
                fb_loss = fb_loss + self.nonneg_coef * nonneg_loss

            # --- total-mass conservation (2026-07-29, ledger §9.15) --------------
            # Visitation identity: E_{g~rho}[M(s,a,g)/rho(B_eps(g))] = 1/(1-gamma).
            # In-batch: N-hat_j = hit column mean (clamped 2/B), row mass
            # R_i = mean_j M_ij/N-hat_j, penalty ((1-gamma) R - 1)^2 -- already
            # dimensionless, so NOT divided by the scale_norm denom. This is the
            # anchor the nonneg hinge lacked: sign and scale stop trading off.
            n_cols = hit.shape[-1]
            nhat_cols = hit.mean(dim=0).clamp_min(2.0 / n_cols)               # [B]
            row_mass = (Ms / nhat_cols).mean(dim=-1)                          # [P,B]
            mass_loss = ((1.0 - self.discount) * row_mass - 1.0).pow(2).mean()
            if self.mass_coef > 0.0:
                assert self.measure_mode == "visitation", \
                    "mass_coef anchors the visitation fixed point; first_passage " \
                    "has a different total-mass identity (E[gamma^T] per column)"
                fb_loss = fb_loss + self.mass_coef * mass_loss

            nhat_loss = nhat_pred_mean = None
            if self.use_nhat_head:
                # phase 2b: N-hat(g) regressed onto the hit-matrix column mean
                # (unbiased MC of rho(B_eps(g))). Input AND target detached —
                # gradient reaches only the head; coefficient 1.0 (in fb_loss,
                # so it is genuinely optimized — the jax lesson).
                nhat_pred = self._nhat_head(B.detach()).squeeze(-1)          # [B]
                nhat_tgt = hit.mean(dim=0).detach()                          # [B]
                nhat_loss = (nhat_pred - nhat_tgt).pow(2).mean()
                nhat_pred_mean = nhat_pred.mean()
                fb_loss = fb_loss + nhat_loss

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
            metrics = {
                "fb_loss": fb_loss,
                "measure_loss": measure_loss,
                "ms_abs_mean": Ms.abs().mean(),      # the scale trace — divergence tell
                # logged ALWAYS, even at coef 0: this is the defect spec 10 L1
                # measured post hoc, and it should be visible during training.
                "neg_frac": (Ms < 0).float().mean(),
                "nonneg_loss": nonneg_loss,
                # logged ALWAYS (like neg_frac): the set-scale trace. 1.0 = the
                # visitation total mass is right; the hinge-only runs sat at 2.5-3.6.
                "mass_ratio": ((1.0 - self.discount) * row_mass).mean(),
                "mass_loss": mass_loss,
                "hit_rate": hit.mean(),
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
            }
            if nhat_loss is not None:
                metrics["nhat_loss"] = nhat_loss
                metrics["nhat_pred_mean"] = nhat_pred_mean
        return metrics


# ---------------------------------------------------------------- arms ------
# Two compositions of ONE critic. Keep the mixin first in both bases: for the
# flow arm that reproduces the pre-2026-08-05 single-class MRO exactly, so the
# paper's torch stack is untouched.

class FBMeasureFlowBCAgent(RegionMeasureMixin, FBFlowBCAgent):
    """Region measure + FB flow-matching BC actor. The historical stack."""


class FBMeasureAgent(RegionMeasureMixin, FBAgent):
    """Region measure + FBAgent's native TD3+BC actor — the no-flow control.

    Same critic, same region, same data; the only changed axis is the actor
    class. Answers whether the two-phase method's multi-object behaviour is a
    property of the objective or of the flow actor it has always been paired
    with.
    """
