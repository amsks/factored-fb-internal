"""PSMDensityCritic — PSM's proto representation learner with a Density-FB SF branch.

PSM-Density-SF (design note: the Density-FB -> PSM/RLDP extension, 2026-08-24). The proto
branch — PSM's distinctive contribution, the successor measures of a family of fixed
pseudo-random policies, and the ONLY place phi and its orthonormality term are trained —
is inherited UNCHANGED (this file's "psm" stage literally delegates to PSMCritic). What
changes is the branch the deployed actor actually uses: the continuous SF head's squared
Bellman residual on M = psi phi^T becomes the categorical Bellman cross-entropy on the
normalized successor density

    p_h(j | s_i, z_i, a_i) = softmax_j( psi_sf_h(s_i, z_i, a_i)^T sg[phi(s'_j)] / tau )

with headwise target rows

    t_h(i, .) = (1 - d_i) one_hot(i)
                + d_i * sg[ softmax_j( psī_sf_h(s'_i, z_i, a'_i)^T sg[phi(s'_j)] / tau ) ],

d_i = gamma * (1 - terminal_i) (the parent's row discount; a terminal row collapses to its
own one-hot so every target row stays on the simplex). Both sides read the ONLINE phi the
"psm" stage just stepped, detached — exactly the parent SF stage's convention (reference
_update_sf uses online phi, not target_phi, and detaches it). Keeping phi out of the
density loss also keeps the softmax's additive gauge (phi -> phi + b changes no softmax)
away from the reward moment E[r phi] that `infer_cond` still inverts.

What must ALSO change once the score is a logit (design note sec. 4): psi_sf^T z is no
longer a value. `q_fn` becomes the reward integration

    q_h(s, a; z) = sum_j p_h(j | s, z, a) * ( phi(u_j)^T z ),      u_j = batch next states,

r_z(u) = phi(u)^T z being the PSM/BFM reward semantics infer_cond inverts. The pool cannot
reach q_fn through its signature, so `sample_cond` stashes it on cond ("_pool_phi") — the
density_fb.py seam workaround. The parent's actor pessimism is PRESERVED but moved to
where it is well-defined: applied to the per-head SCALARS after integration
(mean_h - actor_pessimism_penalty * uncertainty_h), never to the target distributions —
so the parent's `pessimism_penalty` on the SF-stage target measure is lattice-refused
(must be 0.0; the proto stage never read it at cube launch values, where it is 0.0 too).

`actor_q_center` (default true, the bridge spec-19a lattice rule for density actors): the
SECOND q_fn return becomes the centered vector so FlowBC/FQL's |Qs|.mean() normalizer
measures the controllable spread of q, not its offset (see density_fb.py::q_fn).

Inherited unchanged: proto stage + ortho, sample_cond's mixed z and proto code/action
draw (extended only by the pool stash), map_eval_cond (z = project_z(phi(g))), infer_cond
(z ∝ E[r phi]), eval_cond0, GROUPS / STAGES / stage_groups / TARGETS / _LR_KEY rows.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics.psm import PSMCritic, PsmConfigError
from impls.utils.fb_math import targets_uncertainty


@dataclasses.dataclass(frozen=True)
class PSMDensityCritic(PSMCritic):
    temp: float = 1.0
    actor_temp: Any = None          # None => temp (bit-identical readout temperature)
    actor_q_center: bool = True

    def stage_loss(self, stage, params, target, batch, cond, boot):
        if stage == "psm":
            # The proto branch IS PSM — untouched (a parity test pins the two classes to
            # the same proto-stage loss value at identical params).
            return super().stage_loss(stage, params, target, batch, cond, boot)

        assert stage == "sf", stage
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs                     # phi_input="s"; bw encoder is Identity for state
        B = obs.shape[0]
        d = self._row_discount(batch, B)
        z, next_action = cond["z"], boot["next_action"]

        # ONLINE phi, detached, on BOTH sides — the parent SF stage's convention.
        phi = jax.lax.stop_gradient(self._phi(params["phi"], goal))        # [B, z]

        logits = jnp.einsum("pbz,cz->pbc",
                            self._sf(params["sf_psi"], obs, z, action),
                            phi) / self.temp                               # [P, B, B]
        logp = jax.nn.log_softmax(logits, -1)

        # Headwise targets — no mean-minus-uncertainty on distributions (module docstring).
        w_bar = jax.nn.softmax(
            jnp.einsum("pbz,cz->pbc",
                       self._sf(target["sf_psi"], next_obs, z, next_action),
                       phi) / self.temp, -1)

        k = jnp.eye(B)
        t = jax.lax.stop_gradient((1.0 - d) * k + d * w_bar)               # [P, B, B]
        ce = -(t * logp).sum(-1).mean()

        w = jnp.exp(logp)
        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [P, B]
        return ce, {
            "sf_loss": ce,
            "sf_density_ess": ess_rows.mean(),
            "sf_density_ess_median": jnp.median(ess_rows),
            "sf_density_w_entropy": -jnp.mean(jnp.sum(w * logp, axis=-1)),
            "sf_density_tgt_entropy": -jnp.mean(jnp.sum(t * jnp.log(t + 1e-8), axis=-1)),
            "sf_density_logit_max": jnp.max(logits, axis=-1).mean(),
            "sf_phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
        }

    def sample_cond(self, net, batch, rng):
        cond = super().sample_cond(net, batch, rng)
        # The candidate pool for q_fn's reward integration: the batch's own next states
        # through the ONLINE phi, stop-gradded — q_fn cannot reach the batch through its
        # signature (the density_fb.py seam workaround).
        pool = jax.lax.stop_gradient(
            self._phi(net["phi"].params, batch["next_observations"]))
        return {**cond, "_pool_phi": pool}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Reward integration through the normalized successor density, at FROZEN critic
        params — the measure trains only through stage_loss (density_fb.py discipline).
        Gradients flow through `actions` only. Pessimism on scalars, after integration."""
        del stop_encoder_grad  # towers hard-frozen below; nothing further to cut
        pool = cond["_pool_phi"]                                           # [C, z], sg'd
        z = cond["z"]
        Fs = self._sf(jax.lax.stop_gradient(params["sf_psi"]), obs, z, actions)  # [P, B, z]
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum("pbz,cz->pbc", Fs, pool) / a_tau, -1)  # [P, B, C]
        kg = jax.lax.stop_gradient(jnp.einsum("cz,bz->bc", pool, z))         # [B, C]
        Qs = (w * kg[None]).sum(-1)                                          # [P, B]
        qmean, qunc = targets_uncertainty(Qs, self.num_parallel)
        q = qmean - self.actor_pessimism_penalty * qunc
        if self.actor_q_center:
            return q, (q - q.mean())[None]
        return q, Qs

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        if float(config.get("pessimism_penalty", 0.0)) != 0.0:
            raise PsmConfigError(
                "psm_density refuses pessimism_penalty != 0: mean-minus-uncertainty on "
                "target DISTRIBUTIONS leaves the simplex. The proto stage at cube launch "
                "values runs it at 0.0 anyway; actor pessimism lives in q_fn, on scalars "
                "(actor_pessimism_penalty), after reward integration.")
        temp = float(config.get("temp", 1.0))   # density_fb's default temperature
        if temp <= 0.0:
            raise PsmConfigError(f"temp must be > 0, got {temp}")
        critic, params, targets = super().build(ex_obs, ex_actions, config)
        at = config.get("actor_temp", None)
        critic = dataclasses.replace(
            critic, temp=temp,
            actor_temp=(None if at is None else float(at)),
            actor_q_center=bool(config.get("actor_q_center", True)),
        )
        return critic, params, targets
