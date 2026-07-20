"""FbForwardBilinearCritic — the z-INDEXED forward family on the factored seam.

Where IqlBilinearCritic's forward leg is z-BLIND (phi(s,a)), this critic feeds a
stop-gradded, head-mean task vector z = psi(z_goals) INTO the forward map, restoring FB's
defining z-indexed family M(s,a,z) = <F(s,a,z), B(g)>/sqrt(d). It is the highest-value and
highest-risk of the Phase 1 bridge critics.

THE S1 FIX (what makes B2/B4 constructible). In the bridge reference (tools/wandb_mode_shim/
gciql_flow.py) the forward architecture is chosen by a MUTUALLY EXCLUSIVE if/elif chain
(sf_feature_td -> fb_forward_z -> fwd_two_tower), so `fwd_two_tower=True` alongside
`fb_forward_z=True` is silently swallowed: you get GCFBValue and the two-tower flag has no
effect (audit S1). B2 (two-tower + z-forward) and B4 (the "payoff cell": + sphere + ortho)
are therefore UNREACHABLE there. Here the forward architecture is a `phi_module` CHOICE on a
critic that is ALWAYS z-indexed, so two_tower COMPOSES with the z-index by construction. An
unknown phi_module raises at build; nothing is ever silently substituted.

    phi_module ∈ {flat, two_tower, isfb_twin_leg}   (GCFbForwardValue in networks.py)
    objective  ∈ {iql, measure, mix, isfb}
    psi        ALWAYS learned (freezing the goal leg is IqlBilinear's axis, not this one)

VERIFICATION HONESTY (read .superpowers/sdd/phase1-fbforward-report.md):
  * `fb_forward_z` — the z task-index itself — has NO external ground truth (not vendored,
    not torch). It is the Phase 1 analogue of Phase 0's FQL actor. The iql/mix z-indexed
    backup is therefore locked ONLY as a transcription against the first-party bridge
    (tests/impls/test_fb_forward_bilinear_static.py), never claimed correct against an
    upstream reference; and its structure/decomposition/frozen-leg gradients are tested.
  * The measure objective's DEFAULT path (measure_legs='critic', scale_norm=False, test
    rho/full) reproduces FB's own successor-measure TD — the math traces to torch
    agents/fb/agent.py — and IS per-term locked (with the bootstrap action held fixed, so the
    lock isolates the measure math from the actor's sampling). scale_norm / test_states /
    test_rank / anneal are UNVERIFIED and tested structurally only.
  * objective='isfb' (SG-ISFB on the bilinear twin-leg, GCISFBValue-style) is STRUCTURAL
    ONLY. The verified torch-faithful SG-ISFB lives in ISFBCritic (a different architecture:
    ForwardMap towers, not bilinear MLP legs). Here it is constructible and smoke-trainable
    with its loss decomposed and the coord L_V=0 invariant checked, but NOT numerically
    locked. Do not read a green suite as validating the bilinear ISFB numerics.

Deviations from the bridge reference, each deliberate (all shared with IqlBilinear):
  * project_z is SYMMETRIC across the value and critic nets (audit S3), not critic-only.
  * No silent knob coercion; every combination the reference silently overwrites or ignores
    raises FbForwardBilinearConfigError at build time (see `_validate`).
  * The actor's conditioning z is produced by sample_cond OUTSIDE value_and_grad, so it
    carries no gradient into the representation (the reference trains psi through actor_loss).
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCBilinearPsiValue, GCFbForwardValue, PHI_MODULES, sphere_project
from impls.utils.fb_math import targets_uncertainty


class FbForwardBilinearConfigError(ValueError):
    """Raised for an FbForwardBilinear knob combination that would be a silent no-op or a
    silent override in the bridge reference. Named (not a bare assert) so callers can
    distinguish it from ComboError."""


OBJECTIVES = ("iql", "measure", "mix", "isfb")
MEASURE_LEGS = ("critic", "value")
MEASURE_TEST_STATES = ("rho", "goals")
MEASURE_TEST_RANK = ("full", "task")
MEASURE_INSAMPLE = ("scalar_gate_hard", "scalar_gate_awr", "coord")
ACTOR_CONDS = ("goal", "z")


@dataclasses.dataclass(frozen=True)
class FbForwardBilinearCritic:
    value: Any            # z-BLIND GCBilinearPsiValue (V), ensemble=False
    critic: Any           # z-INDEXED GCFbForwardValue (Q), ensemble=True
    phi_module: str
    objective: str
    discount: float
    expectile: float
    tau: float
    obs_dim: int
    latent_dim: int
    action_dim: int
    project_z: bool
    double_q: bool
    actor_cond: str
    actor_z_from_critic: bool
    actor_z_infer: bool
    # measure family
    measure_coef: float
    measure_legs: str
    measure_scale_norm: bool
    measure_stopgrad_phi: bool
    measure_ortho_coef: float
    measure_test_states: str
    measure_test_rank: str
    iql_ortho_coef: float
    # isfb family (structural-only)
    measure_insample: str
    alpha_n: float
    alpha_m: float
    alpha_v: float
    awr_beta: float
    awr_wmax: float
    awr_normalize_adv: bool
    fb_pessimism_penalty: float
    goal_mask: bool

    GROUPS = ("value", "critic")
    STAGES = ("fb_forward",)
    PROVIDES_V = True
    # z is emitted by sample_cond only under actor_cond='z'/actor_z_infer, so — exactly as
    # IqlBilinear — it cannot be promised at class level. FlowBCActor hard-requires "z"
    # unconditionally, so fb_forward_bilinear x flowbc correctly raises ComboError.
    PROVIDES_COND = frozenset({"value_goal", "goal"})
    NUM_HEADS = 2         # the ensembled critic has exactly 2 heads (P for the ISFB scale)

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def _z_cond(self):
        return self.actor_cond == "z" or self.actor_z_infer

    @property
    def cvec_dim(self):
        return self.latent_dim if self._z_cond else self.obs_dim

    def stage_groups(self, stage):
        assert stage == "fb_forward"
        # measure/isfb train only the critic legs (the value net inits but never trains,
        # exactly like the reference's measure_only). iql/mix train both.
        return ("value", "critic") if self.objective in ("iql", "mix") else ("critic",)

    # --- module apply helpers ---
    def _V(self, p, obs, goal):
        return self.value.apply({"params": p}, obs, goal)

    def _Q(self, p, obs, goal, actions):
        # z_goals defaults to goal: Q(s,a,g) = <F(s,a,psi(g)), psi(g)>/sqrt(d).
        return self.critic.apply({"params": p}, obs, goal, actions)

    def _psi_value(self, p, goals):
        return self.value.apply({"params": p}, goals, method=GCBilinearPsiValue.psi_of)

    def _psi_critic(self, p, goals):
        psi = self.critic.apply({"params": p}, goals, method=GCFbForwardValue.psi_of)  # [2,B,d]
        return psi.mean(axis=0)

    def _actor_z(self, params, goals):
        if self.actor_z_from_critic:
            return self._psi_critic(params["critic"], goals)
        return self._psi_value(params["value"], goals)

    def _infer_z(self, psi, rng):
        B = psi.shape[0]
        mask = (jax.random.uniform(rng, (B, B)) < 0.1).astype(psi.dtype) + jnp.eye(B, dtype=psi.dtype)
        return sphere_project(mask @ psi, self.latent_dim)

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        # `>= 0` tie-break, matching OGBench/GCIQL (the iql/mix objective). NOT ISFB's `> 0`.
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    @staticmethod
    def _isfb_expectile(diff, expectile):
        # strict `> 0` tie-break, matching ISFBCritic / torch agents/fb/agent.py.
        weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff ** 2).mean()

    # ------------------------------------------------------------------ iql legs
    def _iql_losses(self, params, target, batch, cond):
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        q1, q2 = self._Q(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        cq1, cq2 = self._Q(params["critic"], obs, vg, action)
        critic_loss = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()
        info = {"value_loss": value_loss, "critic_loss": critic_loss,
                "v_mean": v.mean(), "q_target_mean": tq.mean()}
        return value_loss + critic_loss, info

    def _ortho_only(self, p, goals):
        """FB orthonormality on the critic's backward leg psi(value_goals), no measure — the
        'bilinear GCIQL + ortho only' cell. Per-head then head-mean (reference ortho_loss)."""
        psi = self.critic.apply({"params": p}, goals, method=GCFbForwardValue.psi_of)  # [2,B,d]
        B = psi.shape[1]
        off = 1.0 - jnp.eye(B)
        cov = jnp.einsum("hid,hjd->hij", psi, psi)
        o_off = 0.5 * jnp.sum((cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(cov, axis1=1, axis2=2), axis=1)
        return jnp.mean(o_off + o_diag)

    # ------------------------------------------------------------------ measure legs
    @staticmethod
    def _measure_terms(M, tM, gamma, scale_norm, eps=1e-6):
        """Reduce M, stop-gradded tM into FB (offdiag, diag) TD terms over the last two axes.
        Verbatim from gciql_flow.py::_measure_terms."""
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        diff = M - gamma * tM
        offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
        diagt = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
        if scale_norm:
            msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
            offdiag = offdiag / (msq + eps)
            diagt = diagt / (jnp.sqrt(msq) + eps)
        return offdiag, diagt

    def _measure_value_legs(self, params, batch):
        """State-only successor measure on the z-BLIND value legs. Fully deterministic (no
        actor bootstrap): online value net serves as its own target."""
        obs, nobs = batch["observations"], batch["next_observations"]
        d = self.latent_dim
        sqd = jnp.sqrt(d)
        gamma = self.discount
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        _, phi, psi = self.value.apply({"params": params["value"]}, obs, nobs, info=True)
        if self.measure_stopgrad_phi:
            phi = jax.lax.stop_gradient(phi)
        M = (phi @ psi.T) / sqd
        _, tphi, tpsi = self.value.apply({"params": params["value"]}, nobs, nobs, info=True)
        tM = jax.lax.stop_gradient((tphi @ tpsi.T) / sqd)
        offdiag, diagt = self._measure_terms(M, tM, gamma, self.measure_scale_norm)
        cov = psi @ psi.T
        orth = 0.5 * jnp.sum((cov * off) ** 2) / jnp.sum(off) - jnp.mean(jnp.diag(cov))
        loss = offdiag + diagt + self.measure_ortho_coef * orth
        return loss, {"measure_loss": loss, "measure_offdiag": offdiag,
                      "measure_diag": diagt, "measure_orth": orth}

    def _measure_critic_legs(self, params, target, batch, cond, boot):
        """State-ACTION successor measure on the z-indexed critic legs (FB-faithful). The
        bootstrap a+ ~ pi(s+) is boot['next_action'] (drawn by the seam actor, exactly as
        FBCritic). z=sg(psi(value_goals)); B=psi(next_obs). DEFAULT (rho/full, scale_norm off)
        is the per-term-locked path; test_states/test_rank/scale_norm are structural only."""
        obs, nobs, acts = batch["observations"], batch["next_observations"], batch["actions"]
        vg = cond["value_goal"]
        a_next = jax.lax.stop_gradient(boot["next_action"])
        d = self.latent_dim
        sqd = jnp.sqrt(d)
        gamma = self.discount
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        scale_norm = self.measure_scale_norm

        _, phi, psi = self.critic.apply({"params": params["critic"]}, obs, nobs, acts,
                                        z_goals=vg, info=True)             # phi=F [2,B,d], psi=B(s')
        if self.measure_stopgrad_phi:
            phi = jax.lax.stop_gradient(phi)
        M = jnp.einsum("hid,hjd->hij", phi, psi) / sqd                     # [2,B,B]
        _, tphi, tpsi = self.critic.apply({"params": target["critic"]}, nobs, nobs, a_next,
                                          z_goals=vg, info=True)
        tM = jax.lax.stop_gradient(jnp.einsum("hid,hjd->hij", tphi, tpsi) / sqd)

        qz_mean = None
        if self.measure_test_states == "rho" and self.measure_test_rank == "full":
            offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)  # [2]
        else:
            # Task norm (UNVERIFIED). Test set moves to nu=psi(value_goals); Dirac stays on
            # psi(s'). Verbatim structure from gciql_flow.py measure_loss L819-864.
            _, _, psi_t = self.critic.apply({"params": params["critic"]}, obs, vg, acts,
                                            z_goals=vg, info=True)
            _, _, tpsi_t = self.critic.apply({"params": target["critic"]}, nobs, vg, a_next,
                                             z_goals=vg, info=True)
            M_t = jnp.einsum("hid,hjd->hij", phi, psi_t) / sqd
            tM_t = jax.lax.stop_gradient(jnp.einsum("hid,hjd->hij", tphi, tpsi_t) / sqd)
            diff_t = M_t - gamma * tM_t
            if self.measure_test_rank == "full":
                offdiag = 0.5 * jnp.sum((diff_t * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            else:
                qz = jnp.diagonal(diff_t, axis1=1, axis2=2)                 # [2,B]
                offdiag = 0.5 * jnp.mean(qz ** 2, axis=1)                   # [2]
                qz_mean = jnp.mean(qz)
            dir_i = jnp.diagonal(M - gamma * tM, axis1=1, axis2=2)          # [2,B]
            diagt = -jnp.mean(dir_i, axis=1)                               # [2]
            if scale_norm:
                msq_t = jax.lax.stop_gradient(jnp.mean(M_t ** 2))
                msq_o = jax.lax.stop_gradient(jnp.mean(M ** 2))
                offdiag = offdiag / (msq_t + 1e-6)
                diagt = diagt / (jnp.sqrt(msq_o) + 1e-6)

        cov = jnp.einsum("hid,hjd->hij", psi, psi)                         # ortho on B:=psi(s')
        o_off = 0.5 * jnp.sum((cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(cov, axis1=1, axis2=2), axis=1)
        orth = o_off + o_diag                                             # [2]
        lam = self.measure_ortho_coef
        loss = jnp.mean(offdiag + diagt + lam * orth)                     # per-head, then average
        offdiag, diagt, orth = jnp.mean(offdiag), jnp.mean(diagt), jnp.mean(orth)
        info = {"measure_loss": loss, "measure_offdiag": offdiag,
                "measure_diag": diagt, "measure_orth": orth}
        if qz_mean is not None:
            info["measure_qz_mean"] = qz_mean
        return loss, info

    def _measure_loss(self, params, target, batch, cond, boot):
        if self.measure_legs == "value":
            return self._measure_value_legs(params, batch)
        return self._measure_critic_legs(params, target, batch, cond, boot)

    # ------------------------------------------------------------------ isfb legs (structural)
    def _isfb_loss(self, params, target, batch, cond, boot):
        """SG-ISFB on the bilinear twin-leg (GCISFBValue-style). STRUCTURAL ONLY — the
        verified torch-faithful SG-ISFB is ISFBCritic (ForwardMap towers). Loss SHAPE mirrors
        ISFBCritic.stage_loss; here M/N/q/v carry the bilinear head's /sqrt(d). P = 2 heads."""
        del boot                                    # ISFB never samples a policy action
        obs, nobs, acts = batch["observations"], batch["next_observations"], batch["actions"]
        vg = cond["value_goal"]
        d = self.latent_dim
        sqd = jnp.sqrt(d)
        P = self.NUM_HEADS
        tau = self.expectile
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        _, phi, psi, fv, z = self.critic.apply({"params": params["critic"]}, obs, nobs, acts,
                                               z_goals=vg, info=True, return_fv=True)
        M = jnp.einsum("hid,hjd->hij", phi, psi) / sqd                     # [P,B,B]
        N = jnp.einsum("hid,hjd->hij", fv, psi) / sqd                      # [P,B,B]
        q = (phi * z[None]).sum(-1) / sqd                                 # [P,B]
        v = (fv * z[None]).sum(-1) / sqd                                  # [P,B]

        _, _, tpsi, tfv, _ = self.critic.apply({"params": target["critic"]}, nobs, nobs, acts,
                                               z_goals=vg, info=True, return_fv=True)
        tN = jnp.einsum("hid,hjd->hij", tfv, tpsi) / sqd                   # [P,B,B]
        tmean, tunc = targets_uncertainty(tN, P)
        tN_M = jax.lax.stop_gradient(tmean - self.fb_pessimism_penalty * tunc)  # [B,B]

        L_V = P * self._isfb_expectile(jax.lax.stop_gradient(q) - v, tau)
        A = jax.lax.stop_gradient(q - v).mean(0)                           # [B]
        Msg = jax.lax.stop_gradient(M)
        if self.measure_insample == "coord":
            L_N = P * self._isfb_expectile(Msg - N, tau)
        else:
            if self.measure_insample == "scalar_gate_awr":
                An = A / (jnp.std(A, ddof=1) + 1e-6) if self.awr_normalize_adv else A
                wgt = jnp.clip(jnp.exp(self.awr_beta * An), 0.0, self.awr_wmax)
            else:
                wgt = jnp.where(A > 0, tau, 1.0 - tau)
            wbar = wgt / (wgt.mean() + 1e-6)
            L_N = P * (wbar.reshape(1, -1, 1) * (Msg - N) ** 2).mean()

        disc = self.discount * batch["masks"][:, None] if self.goal_mask else self.discount
        diff = M - disc * tN_M
        fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        L_M = fb_off + fb_diag

        # single (head-mean) backward map for the ortho, matching FB's non-ensembled B ortho.
        Bm = psi.mean(0)
        cov = Bm @ Bm.T
        orth = 0.5 * jnp.sum((cov * off) ** 2) / off_sum - jnp.mean(jnp.diagonal(cov))

        loss = self.alpha_n * L_N + self.alpha_m * L_M + self.measure_ortho_coef * orth
        if self.measure_insample != "coord":
            loss = loss + self.alpha_v * L_V
        else:
            L_V = jnp.zeros((), jnp.float64)
        return loss, {"isfb_loss": loss, "isfb_L_V": L_V, "isfb_L_N": L_N, "isfb_L_M": L_M,
                      "isfb_offdiag": fb_off, "isfb_diag": fb_diag, "ortho_loss": orth,
                      "adv_mean": A.mean(), "M_mean": M.mean()}

    # ------------------------------------------------------------------ stage
    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "fb_forward"
        if self.objective == "isfb":
            return self._isfb_loss(params, target, batch, cond, boot)
        if self.objective == "measure":
            m_loss, m_info = self._measure_loss(params, target, batch, cond, boot)
            loss = self.measure_coef * m_loss
            return loss, {**{f"measure/{k}": v for k, v in m_info.items()}, "loss": loss}

        loss, info = self._iql_losses(params, target, batch, cond)
        if self.objective == "mix" and self.measure_coef > 0.0:
            m_loss, m_info = self._measure_loss(params, target, batch, cond, boot)
            loss = loss + self.measure_coef * m_loss
            info.update({f"measure/{k}": v for k, v in m_info.items()})
        if self.iql_ortho_coef > 0.0:
            orth = self._ortho_only(params["critic"], cond["value_goal"])
            loss = loss + self.iql_ortho_coef * orth
            info["ortho_loss"] = orth
        return loss, info

    # ------------------------------------------------------------------ readouts / seam
    def v_fn(self, params, obs, cond):
        if self.objective == "isfb":
            # ISFB scalar V = <F_V(s,z), psi(g)>/sqrt(d), reduced like q_fn.
            zero_a = jnp.zeros(obs.shape[:-1] + (self.action_dim,))
            _, _, psi, fv, z = self.critic.apply(
                {"params": params["critic"]}, obs, cond["goal"], zero_a,
                z_goals=cond["goal"], info=True, return_fv=True)
            vh = jnp.einsum("hid,hid->hi", fv, psi) / jnp.sqrt(self.latent_dim)
            return jnp.minimum(vh[0], vh[1]) if self.double_q else vh[0]
        return self._V(params["value"], obs, cond["goal"])

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        q1, q2 = self._Q(params["critic"], obs, cond["goal"], actions)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        return cond["z"] if self._z_cond else cond["goal"]

    def sample_cond(self, net, batch, rng):
        cond = {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}
        if self._z_cond:
            params = {"value": net["value"].params, "critic": net["critic"].params}
            z = self._actor_z(params, batch["actor_goals"])
            if self.actor_z_infer:
                z = self._infer_z(z, rng)
            cond["z"] = z
        return cond

    def map_eval_cond(self, net, cond):
        if not self._z_cond or "goal" not in cond:
            return cond
        params = {"value": net["value"].params, "critic": net["critic"].params}
        return {**cond, "z": self._actor_z(params, cond["goal"])}

    def eval_cond0(self, ex_obs, config):
        cond = {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        if self._z_cond:
            cond["z"] = jnp.zeros((self.latent_dim,), jnp.float64)
        return cond

    def infer_cond(self, net, obs, rewards):
        return self.eval_cond0(obs, None)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _validate(config, goal_dim):
        phi_module = str(config["phi_module"])
        objective = str(config["objective"])
        actor_cond = str(config["actor_cond"])
        legs = str(config["measure_legs"])
        test_states = str(config["measure_test_states"])
        test_rank = str(config["measure_test_rank"])
        insample = str(config["measure_insample"])
        d = int(config["value_latent_dim"])

        if phi_module not in PHI_MODULES:
            raise FbForwardBilinearConfigError(
                f"phi_module must be one of {PHI_MODULES}, got {phi_module!r}")
        if objective not in OBJECTIVES:
            raise FbForwardBilinearConfigError(
                f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if actor_cond not in ACTOR_CONDS:
            raise FbForwardBilinearConfigError(
                f"actor_cond must be one of {ACTOR_CONDS}, got {actor_cond!r}")
        if legs not in MEASURE_LEGS:
            raise FbForwardBilinearConfigError(
                f"measure_legs must be one of {MEASURE_LEGS}, got {legs!r}")
        if test_states not in MEASURE_TEST_STATES:
            raise FbForwardBilinearConfigError(
                f"measure_test_states must be one of {MEASURE_TEST_STATES}, got {test_states!r}")
        if test_rank not in MEASURE_TEST_RANK:
            raise FbForwardBilinearConfigError(
                f"measure_test_rank must be one of {MEASURE_TEST_RANK}, got {test_rank!r}")
        if insample not in MEASURE_INSAMPLE:
            raise FbForwardBilinearConfigError(
                f"measure_insample must be one of {MEASURE_INSAMPLE}, got {insample!r}")

        # S8-analogue: the isfb objective needs the value-measure leg F_V; refuse any other
        # phi_module (would read a non-existent fv submodule at trace time).
        if objective == "isfb" and phi_module != "isfb_twin_leg":
            raise FbForwardBilinearConfigError(
                "objective='isfb' requires phi_module='isfb_twin_leg' (the SG-ISFB value "
                "measure N=<F_V,B> needs the F_V leg); got phi_module=" + repr(phi_module))
        # Conversely the twin leg is dead weight under any non-isfb objective; refuse it
        # rather than build an untrained F_V and silently ignore it (S10/S11).
        if phi_module == "isfb_twin_leg" and objective != "isfb":
            raise FbForwardBilinearConfigError(
                "phi_module='isfb_twin_leg' only trains under objective='isfb'; its F_V leg "
                "would be built but never used. Got objective=" + repr(objective))
        # measure_coef is only consulted by measure/mix. Under iql it is a silent no-op; under
        # measure it must be positive or the whole objective is zeroed.
        mc = float(config["measure_coef"])
        if objective == "iql" and mc != 0.0:
            raise FbForwardBilinearConfigError(
                "objective='iql' ignores measure_coef (no measure term); set measure_coef=0 "
                "or use objective='mix'. Got measure_coef=" + repr(mc))
        if objective in ("measure", "mix") and mc <= 0.0:
            raise FbForwardBilinearConfigError(
                f"objective={objective!r} needs measure_coef>0 (else the measure term, its "
                f"whole point, is zeroed). Got measure_coef={mc!r}")
        # test_rank='task' requires test_states='goals' (reference L1383).
        if test_rank == "task" and test_states != "goals":
            raise FbForwardBilinearConfigError(
                "measure_test_rank='task' requires measure_test_states='goals' (the rank-1 "
                "projection is onto the task set nu=psi(goals)).")
        # measure_test_* only apply to the critic-leg measure (reference L1389).
        if (test_states != "rho" or test_rank != "full") and \
                (objective not in ("measure", "mix") or legs != "critic"):
            raise FbForwardBilinearConfigError(
                "measure_test_states/measure_test_rank only apply to the critic-leg measure "
                "(objective measure/mix, measure_legs='critic').")
        # S6: actor_z_from_critic is a silent no-op under actor_cond='goal'.
        if bool(config["actor_z_from_critic"]) and actor_cond != "z":
            raise FbForwardBilinearConfigError(
                "actor_z_from_critic=True is a silent no-op unless actor_cond='z' (audit S6).")
        # iql_ortho_coef only lives on the iql/mix scalar path (reference threads it in
        # total_loss). Under measure/isfb the measure_ortho_coef term owns the ortho.
        if float(config["iql_ortho_coef"]) > 0.0 and objective not in ("iql", "mix"):
            raise FbForwardBilinearConfigError(
                "iql_ortho_coef>0 only applies under objective in (iql, mix); measure/isfb own "
                "their ortho via measure_ortho_coef.")

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        goal_dim = int(ex_obs.shape[-1])
        cls._validate(config, goal_dim)

        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])
        phi_module = str(config["phi_module"])
        project_z = bool(config["project_z"])

        # value net: z-BLIND learned bilinear head (the IQL V leg). SYMMETRIC project_z (S3).
        value = GCBilinearPsiValue(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=False,
                                   psi_base="learned", project_z=project_z)
        # critic net: z-INDEXED forward family.
        critic = GCFbForwardValue(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=True,
                                  phi_module=phi_module, project_z=project_z)

        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc = jr.split(jr.PRNGKey(config["seed"] + 7000), 2)
        # isfb_twin_leg's F_V submodule is only traced (and hence only gets params) on the
        # return_fv path, so init MUST take that path or the leg would be missing at apply.
        crit_init_kw = dict(info=True, return_fv=True) if phi_module == "isfb_twin_leg" else {}
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "critic": critic.init(rc, ex_obs, ex_goal, ex_actions, **crit_init_kw)["params"],
        }
        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(
            value=value, critic=critic, phi_module=phi_module, objective=str(config["objective"]),
            discount=float(config["discount"]), expectile=float(config["expectile"]),
            tau=float(config["tau"]), obs_dim=goal_dim, latent_dim=d,
            action_dim=int(ex_actions.shape[-1]), project_z=project_z,
            double_q=bool(config["double_q"]), actor_cond=str(config["actor_cond"]),
            actor_z_from_critic=bool(config["actor_z_from_critic"]),
            actor_z_infer=bool(config["actor_z_infer"]),
            measure_coef=float(config["measure_coef"]), measure_legs=str(config["measure_legs"]),
            measure_scale_norm=bool(config["measure_scale_norm"]),
            measure_stopgrad_phi=bool(config["measure_stopgrad_phi"]),
            measure_ortho_coef=float(config["measure_ortho_coef"]),
            measure_test_states=str(config["measure_test_states"]),
            measure_test_rank=str(config["measure_test_rank"]),
            iql_ortho_coef=float(config["iql_ortho_coef"]),
            measure_insample=str(config["measure_insample"]),
            alpha_n=float(config["alpha_n"]), alpha_m=float(config["alpha_m"]),
            alpha_v=float(config["alpha_v"]), awr_beta=float(config["awr_beta"]),
            awr_wmax=float(config["awr_wmax"]),
            awr_normalize_adv=bool(config["awr_normalize_adv"]),
            fb_pessimism_penalty=float(config["fb_pessimism_penalty"]),
            goal_mask=bool(config["goal_mask"]),
        )
        return obj, params, targets
