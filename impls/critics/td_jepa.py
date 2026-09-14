"""TDJEPACritic — TD-JEPA (arXiv:2510.00739) on the factored seam.

Transcribed from the reference `~/git/baselines/td_jepa/metamotivo/agents/td_jepa/
{agent,model}.py` @ 840a745 (`TDJEPAAgent.update_tdjepa_asym`, `TDJEPAAgent.update_actor`,
`TDJEPAModel.__init__` / `reward_inference`). Nothing is imported from that tree; the math
below is a line-by-line transcription restructured onto the seam's per-group TrainState
params, the same way `impls/critics/rldp.py` and `impls/critics/hilp.py` were.

The algorithm
-------------
Two encoders on observations and two predictors that roll them one step forward,
conditioned on the task vector z and the action:

    phi(s) in R^{d_phi}          psi(s) in R^{d_psi}          z in R^{d_psi}
    Pred_phi(phi(s), z, a) -> R^{d_psi}
    Pred_psi(psi(s), z, a) -> R^{d_phi}

Each predictor regresses a TD target built from the OTHER encoder at the next state:

    a'             = pi(s', z)
    target_phi     = psi_tgt(s') + gamma * Pred_phi_tgt(phi_tgt(s'), z, a')
    target_psi     = phi_tgt(s') + gamma * Pred_psi_tgt(psi_tgt(s'), z, a')
    L              = || Pred_phi(phi(s), z, a) - sg[target_phi] ||^2
                   + || Pred_psi(psi(s), z, a) - sg[target_psi] ||^2
                   + c_phi * orth(phi(s)) + c_psi * orth(psi(s))

So the prediction targets are the encoders' own outputs one step ahead, bootstrapped
through the predictors -- a latent successor feature, learned by a squared TD residual in
latent space. The orthonormality term is what stops the encoders collapsing; it is the
same batch-Gram form as FB's and RLDP's (`_ortho` below).

The value readout is the inner product of the phi-predictor with the task vector:

    Q(s, a, z) = < Pred_phi(phi(s), z, a), z >

and the task vector for a reward is the least-squares solve of the reward samples on the
psi features, sqrt(d)-normalized (`TDJEPAModel.reward_inference`) -- HILP's readout, not
FB's reward moment.

Divergences from the reference, both recorded rather than hidden
---------------------------------------------------------------
1. **The actor consumes the raw observation, not phi(s).** The reference's
   `actor_use_full_encoder` defaults True, so its policy net reads `phi(s)`. Every actor in
   this tree reads the observation plus `cond_vec` and the eval loop has no encoder to call,
   so this port runs the reference's `actor_use_full_encoder=False` branch, in which
   `actor_in` is the (identity-encoded) observation. That is a configuration their own code
   supports; the critic, its losses, its z distribution and its readout are unchanged by it.
   `tdjepa_actor_full_encoder` is reserved for the other branch; it raises until the
   actor-side plumbing exists, so nobody selects it by accident.
2. **No terminal masking.** The reference multiplies the discount by `~terminated`. Every
   dataset in this tree carries `terminals=0` ("always-gamma bootstrap", see
   impls/utils/datasets.py), which is what `impls/critics/{fb,rldp}.py` already assume, so
   the plain discount is the same number here.

KNOWN UPSTREAM BUG: this critic cannot be traced under `jax_enable_x64`
-----------------------------------------------------------------------
Under jax/jaxlib 0.10.0 the jitted update raises

    FAILED_PRECONDITION: A cycle is detected while visiting instruction
    %get-tuple-element... metadata={op_name="jit(update)/jvp()/add"}

i.e. XLA builds a cyclic HLO. It is a compiler bug, not a defect in the math: every piece
of the loss differentiates and jits cleanly on its own under x64 (phi branch, psi branch,
both orthonormality terms, q_fn), and the failure needs all three of

    jax_enable_x64  AND  d_phi == d_psi  AND  num_parallel == 2

together -- the two prediction branches are then bit-identical subgraphs. Breaking ANY of
the three makes it compile: tdjepa_phi_dim=64 compiles, num_parallel=1 compiles, and f32
compiles at every batch size tried (8, 32, 256). Emitting the branches sequentially rather
than interleaved does not help, and no available XLA flag avoids it
(--xla_gpu_enable_command_buffer=, latency-hiding scheduler off, triton gemm off all still
fail; the horizontal-fusion flags do not exist in this version).

EVERY REAL RUN IS UNAFFECTED: training runs in f32. Only the test session trips it, because
four test modules call `jax.config.update("jax_enable_x64", True)` at import and that is
process-global, so it is on for everything collected after them. tests/impls/
test_phase0_gate.py therefore skips this critic under x64 with that reason, and
tests/impls/test_td_jepa.py::test_update_runs_in_f32 pins the path that actually ships.
Re-check on the next jax upgrade and delete the skip if it compiles.

Symmetric mode (`TDJEPAModelConfig.symmetric=True`, one encoder, one predictor) is NOT
ported: no config in the reference's OGBench or DMC sweeps selects it, and an unexercised
branch in a loss is a liability. `tdjepa_symmetric` raises if set.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


@dataclasses.dataclass(frozen=True)
class TDJEPACritic:
    phi_enc: Any
    psi_enc: Any
    phi_pred: Any
    psi_pred: Any
    z_dim: int           # = psi_dim; the dimension z and the psi features live in
    phi_dim: int
    num_parallel: int
    discount: float
    phi_ortho_coef: float
    psi_ortho_coef: float
    train_goal_ratio: float
    norm_z: bool
    enc_tau: float       # encoder_target_tau
    pred_tau: float      # predictor_target_tau
    # Actor maximizes mean - coef * uncertainty over the predictor ensemble
    # (TDJEPAAgentTrainConfig.actor_pessimism_penalty, default 0.0 = plain mean).
    actor_pessimism: float = 0.0
    # TD target reduction over the predictor ensemble
    # (predictor_pessimism_penalty, default 0.0 = plain mean).
    pred_pessimism: float = 0.0

    GROUPS = ("phi_enc", "psi_enc", "phi_pred", "psi_pred")
    STAGES = ("tdjepa",)
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"z"})
    # The z readout is solved on psi(s'), and sample_cond mixes in psi(next_observations),
    # so the reward->z solve must see the SAME rows paired with that transition's reward.
    INFER_COND_MARGINAL = "next_observations"

    @property
    def TARGETS(self):
        # The reference polyaks all four: encoders at encoder_target_tau, predictors at
        # predictor_target_tau (`TDJEPAAgent.update`, both 0.005 by default).
        return {"phi_enc": self.enc_tau, "psi_enc": self.enc_tau,
                "phi_pred": self.pred_tau, "psi_pred": self.pred_tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def map_eval_cond(self, net, cond):
        """Eval-time hook: turn the env's goal frame into the z the actor was trained on,
        z = project_z(psi(goal)) -- the same embedding sample_cond mixes in. Without it
        eval runs on eval_cond0's zero z, where Q(s,a;0) = <Pred, 0> == 0 and the policy is
        goal-blind (the 2026-08-17 FB-zero bug, and the Route-A goal-blindness that made
        eight HILP cells meaningless on 2026-09-08)."""
        if "goal" not in cond:
            return cond
        z = project_z(self._psi(net["psi_enc"].params, cond["goal"]), self.norm_z)
        return {**cond, "z": z}

    def stage_groups(self, stage):
        assert stage == "tdjepa"
        return self.GROUPS

    # --- towers ---
    def _phi(self, p, obs):
        return self.phi_enc.apply({"params": p}, obs)

    def _psi(self, p, obs):
        return self.psi_enc.apply({"params": p}, obs)

    def _pred_phi(self, p, phi, z, action):
        return self.phi_pred.apply({"params": p}, phi, z, action)

    def _pred_psi(self, p, psi, z, action):
        return self.psi_pred.apply({"params": p}, psi, z, action)

    def _ortho(self, enc, off, off_sum):
        """`TDJEPAAgent._orth_loss`: Cov = enc @ enc.T over the BATCH axis, pull the
        diagonal up and whiten the off-diagonal. Identical in form to RLDP's, applied to
        each encoder's own output."""
        cov = enc @ enc.T
        off_term = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        diag = -jnp.mean(jnp.diagonal(cov))
        return off_term + diag, off_term, diag

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "tdjepa"
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        z, next_action = cond["z"], boot["next_action"]
        disc, P = self.discount, self.num_parallel
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        # ---- TD targets. The reference computes this whole block under
        # `with torch.no_grad()` (update_tdjepa_asym), so the stop_gradient goes on the
        # TOWER OUTPUTS, not just on the two sums at the end. That is not cosmetic: the
        # target predictors' ensemble spread (targets_uncertainty) is a pairwise tensor,
        # and leaving its tangent subgraph in the trace only to zero it at the end made
        # XLA build a cyclic HLO under jax_enable_x64 -- which four test modules set
        # process-globally, so it surfaced only inside the test session
        # ("FAILED_PRECONDITION: A cycle is detected", jit(update)/jvp()/add).
        sg = jax.lax.stop_gradient
        nphi = sg(self._phi(target["phi_enc"], next_obs))                # [B, d_phi]
        npsi = sg(self._psi(target["psi_enc"], next_obs))                # [B, d_psi]
        tphi_preds = sg(self._pred_phi(target["phi_pred"], nphi, z, next_action))  # [P, B, d_psi]
        tphi_mean, tphi_unc = targets_uncertainty(tphi_preds, P)
        td_target_phi = sg(npsi + disc * (tphi_mean - self.pred_pessimism * tphi_unc))
        tpsi_preds = sg(self._pred_psi(target["psi_pred"], npsi, z, next_action))  # [P, B, d_phi]
        tpsi_mean, tpsi_unc = targets_uncertainty(tpsi_preds, P)
        td_target_psi = sg(nphi + disc * (tpsi_mean - self.pred_pessimism * tpsi_unc))

        # ---- online predictions. The encoders are trained HERE (and only here): the
        # actor's q_fn stop-grads them, matching update_actor's no_grad on phi_enc.
        phi_enc = self._phi(params["phi_enc"], obs)                      # [B, d_phi]
        psi_enc = self._psi(params["psi_enc"], obs)                      # [B, d_psi]
        phi_preds = self._pred_phi(params["phi_pred"], phi_enc, z, action)     # [P, B, d_psi]
        psi_preds = self._pred_psi(params["psi_pred"], psi_enc, z, action)     # [P, B, d_phi]

        # `(preds - target).pow(2).sum(-1).mean()`: sum over the feature axis, mean over
        # the ensemble and batch axes together.
        phi_loss = jnp.mean(jnp.sum((phi_preds - td_target_phi[None]) ** 2, axis=-1))
        psi_loss = jnp.mean(jnp.sum((psi_preds - td_target_psi[None]) ** 2, axis=-1))
        tdjepa_loss = phi_loss + psi_loss

        phi_orth, phi_orth_off, phi_orth_diag = self._ortho(phi_enc, off, off_sum)
        psi_orth, psi_orth_off, psi_orth_diag = self._ortho(psi_enc, off, off_sum)

        loss = (tdjepa_loss
                + self.phi_ortho_coef * phi_orth
                + self.psi_ortho_coef * psi_orth)
        return loss, {
            "tdjepa_loss": tdjepa_loss, "phi_tdjepa_loss": phi_loss,
            "psi_tdjepa_loss": psi_loss, "total_loss": loss,
            "phi_orth_loss": phi_orth, "phi_orth_offdiag": phi_orth_off,
            "phi_orth_diag": phi_orth_diag,
            "psi_orth_loss": psi_orth, "psi_orth_offdiag": psi_orth_off,
            "psi_orth_diag": psi_orth_diag,
            "phi_encoder_norm": jnp.mean(jnp.linalg.norm(phi_enc, axis=-1)),
            "psi_encoder_norm": jnp.mean(jnp.linalg.norm(psi_enc, axis=-1)),
            "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1)),
            "td_target_phi": jnp.mean(td_target_phi),
        }

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Q = <Pred_phi(phi(s), z, a), z>, mean over the predictor ensemble.

        phi is ALWAYS stop-gradded, matching `update_actor`'s `with torch.no_grad():
        phi_enc = self._model._phi_mlp_encoder(phi_obs)`. The encoders are trained by
        stage_loss alone, so `stop_encoder_grad` cannot select anything else."""
        del stop_encoder_grad
        z = cond["z"]
        phi = jax.lax.stop_gradient(self._phi(params["phi_enc"], obs))
        preds = self._pred_phi(params["phi_pred"], phi, z, actions)      # [P, B, d_psi]
        Qs = (preds * z).sum(-1)                                         # [P, B]
        qmean, qunc = targets_uncertainty(Qs, self.num_parallel)
        if self.actor_pessimism:
            return qmean - self.actor_pessimism * qunc, Qs
        return qmean, Qs

    def cond_vec(self, cond):
        return cond["z"]

    def sample_cond(self, net, batch, rng):
        """`sample_mixed_z`: a sqrt(d)-sphere Gaussian, with a `train_goal_ratio` fraction
        of rows replaced by psi(goal) of a PERMUTED goal, also projected. The reference's
        `train_goal` is the buffer's goal frame; on this seam that is next_observations,
        the same rows RLDP's sample_cond embeds.

        `scale_train_goals` (whiten the goal embedding by the running z covariance) is not
        carried: it is False in the reference's defaults and in every sweep config."""
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        r_z, r_mask, r_perm = jax.random.split(rng, 3)
        z_gauss = project_z(jax.random.normal(r_z, (B, self.z_dim)), self.norm_z)
        goals = project_z(
            self._psi(net["psi_enc"].params, next_obs)[jax.random.permutation(r_perm, B)],
            self.norm_z,
        )
        mix_mask = (jax.random.uniform(r_mask, (B,)) < self.train_goal_ratio)[:, None]
        return {"z": jnp.where(mix_mask, goals, z_gauss)}

    def infer_cond(self, net, obs, rewards):
        """`TDJEPAModel.reward_inference`: z = lstsq(psi(s'), r), then sqrt(d)-normalized.
        A least-squares SOLVE against the feature matrix, like HILP's, NOT FB's reward
        moment E[r B]."""
        psi = self._psi(net["psi_enc"].params, jnp.asarray(obs))
        r = jnp.asarray(rewards).reshape(-1, 1)
        z = jnp.linalg.lstsq(psi, r, rcond=None)[0].reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr

        if bool(config.get("tdjepa_symmetric", False)):
            raise NotImplementedError(
                "tdjepa_symmetric selects the reference's one-encoder branch "
                "(update_tdjepa_sym), which is not ported: no reference sweep config "
                "sets it. Port update_tdjepa_sym before enabling.")
        if bool(config.get("tdjepa_actor_full_encoder", False)):
            raise NotImplementedError(
                "tdjepa_actor_full_encoder selects the reference's default actor input, "
                "phi(s) rather than the observation. The seam's actors read the raw "
                "observation and the eval loop has no encoder to call, so the critic "
                "would train and evaluate the policy on different inputs. See divergence "
                "1 in this module's docstring.")

        # z lives in psi's space (asymmetric: TDJEPAModel sets z_dim = psi_dim).
        z_dim = int(config["z_dim"])
        phi_dim = int(config.get("tdjepa_phi_dim", z_dim))
        P = int(config["num_parallel"])

        # phi_mlp_encoder / psi_mlp_encoder are BackwardArchiConfig in the reference, the
        # same module family this tree builds as PhiMap for FB's and RLDP's backward map.
        phi_enc = PhiMap(z_dim=phi_dim, hidden_dim=config["backward"]["hidden_dim"],
                         hidden_layers=config["backward"]["hidden_layers"],
                         norm=config["backward"]["norm"])
        psi_enc = PhiMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                         hidden_layers=config["backward"]["hidden_layers"],
                         norm=config["backward"]["norm"])
        # Both predictors are ForwardArchiConfig builds: (feature, z, action) -> output.
        # Pred_phi maps phi-space to PSI's width and Pred_psi maps psi-space to PHI's --
        # the crossed widths are the asymmetric variant's whole point, since each
        # predictor's target is the other encoder.
        phi_pred = ForwardMap(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"],
                              num_parallel=P,
                              hidden_layers=config["forward"]["hidden_layers"],
                              embedding_layers=config["forward"]["embedding_layers"])
        psi_pred = ForwardMap(z_dim=phi_dim, hidden_dim=config["forward"]["hidden_dim"],
                              num_parallel=P,
                              hidden_layers=config["forward"]["hidden_layers"],
                              embedding_layers=config["forward"]["embedding_layers"])

        r_phi, r_psi, r_pp, r_sp = jr.split(jr.PRNGKey(config["seed"]), 4)
        n = ex_obs.shape[0]
        ex_z = jnp.zeros((n, z_dim))
        params = {
            "phi_enc": phi_enc.init(r_phi, ex_obs)["params"],
            "psi_enc": psi_enc.init(r_psi, ex_obs)["params"],
            "phi_pred": phi_pred.init(r_pp, jnp.zeros((n, phi_dim)), ex_z, ex_actions)["params"],
            "psi_pred": psi_pred.init(r_sp, jnp.zeros((n, z_dim)), ex_z, ex_actions)["params"],
        }
        ortho = float(config["ortho_coef"])
        critic = cls(
            phi_enc=phi_enc, psi_enc=psi_enc, phi_pred=phi_pred, psi_pred=psi_pred,
            z_dim=z_dim, phi_dim=phi_dim, num_parallel=P,
            discount=float(config["discount"]),
            phi_ortho_coef=float(config.get("tdjepa_phi_ortho_coef", ortho)),
            psi_ortho_coef=float(config.get("tdjepa_psi_ortho_coef", ortho)),
            train_goal_ratio=float(config["train_goal_ratio"]),
            norm_z=bool(config["norm_z"]),
            enc_tau=float(config.get("tdjepa_encoder_target_tau", config["b_target_tau"])),
            pred_tau=float(config.get("tdjepa_predictor_target_tau", config["f_target_tau"])),
            actor_pessimism=float(config.get("tdjepa_actor_pessimism", 0.0)),
            pred_pessimism=float(config.get("tdjepa_predictor_pessimism", 0.0)),
        )
        targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
        return critic, params, targets
