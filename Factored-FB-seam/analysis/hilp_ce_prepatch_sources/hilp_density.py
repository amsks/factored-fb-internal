"""HILPDensityCritic — HILP's Hilbert representation with Density-FB's successor estimator.

The Density-FB -> RLDP extension, applied a second time. HILP qualifies as a substrate for
the same structural reason RLDP does: its feature map phi is learned ENTIRELY by the
Hilbert/IQL value loss and the successor objective never touches it. `stage_groups("sf")`
returns ("hilp_sf",) alone, and the SF loss reads phi through stop_gradient. So replacing
the successor estimator changes exactly ONE thing, and phi's training signal is
bit-identical between hilp_tdj and hilp_density_tdj.

The swap, with psi in F's role and phi in B's:

    p_h(j | s_i, a_i, z) = softmax_j( psi_h(s_i, a_i, z)^T sg[phi(s'_j)] / temp )

    t_h(i, .) = (1 - d_i) one_hot(i) + d_i * sg[ softmax_j( psibar_h(s'_i, a'_i, z)^T
                                                            sg[phibar(s'_j)] / temp ) ]

d_i = gamma * (1 - terminal_i), so a terminal row puts all its mass on its own successor
and every target row stays on the simplex. This REPLACES the vector regression
`target_SF = phi(s') + discount * psi(s'', a'')` (and, under hilp_q_loss, its scalar
variant); everything else -- the phi value loss, sample_cond's mixed-z draw, infer_cond,
the target schedule -- is inherited untouched.

WHAT ALSO CHANGES, for the same reason as rldp_density: once psi^T phi is a LOGIT,
<psi, z> is no longer a value. q_fn becomes the reward integration

    q_h(s, a; z) = sum_j p_h(j | s, a, z) * ( phi(u_j)^T z ),   u_j = the batch next states,

and r(u) = phi(u)^T z is exactly HILP's own reward convention -- the thing infer_cond's
least squares inverts -- so the readout is consistent with how z is produced at eval. The
pool cannot reach q_fn through its signature, so sample_cond stashes it on cond
("_pool_phi"), the seam workaround density_fb.py and rldp_density.py both use.

Ensemble reduction happens AFTER integration, on scalars. The per-head target
distributions are never collapsed by a mean-minus-uncertainty, which can leave the simplex.

td_jepa parity is INHERITED, not re-decided: this subclasses the corrected HILPCritic, and
critic/hilp_density_tdj.yaml extends critic/hilp_tdj.yaml, so hilp_head_reduce=mean,
hilp_actor_head_reduce=mean and hilp_sf_loss_scale carry over. hilp_q_loss is IGNORED here
and refused if true -- it selects between vector and scalar regression, and this critic
performs neither.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics.hilp import HILPCritic
from impls.critics._measure_common import parse_hit_spec
from impls.critics.density_fb import _delta_phi, _raw_block_cols
from impls.utils.fb_math import project_z, targets_uncertainty


class HilpDensityConfigError(ValueError):
    """Raised at build time for a structurally invalid Density-HILP config."""


@dataclasses.dataclass(frozen=True)
class HILPDensityCritic(HILPCritic):
    temp: float = 1.0
    actor_temp: Any = None
    actor_q_center: bool = True
    # goal_kernel (2026-09-06, mirrors rldp_density's Exp A): every row is conditioned on
    # the WHITENED index of a real hindsight goal (no random-direction mix) and the actor's
    # reward is that goal's object-coordinate kernel over the pool -- density_fb's goal
    # interface on the HILP substrate. Default False = the pre-existing critic bit-for-bit.
    goal_kernel: bool = False
    kernel_blocks: tuple = ()

    def _row_discount(self, batch, B):
        """gamma * (1 - terminated), shaped [B, 1].

        The parent needs no row discount (scalar gamma inside a squared residual); the
        categorical target DOES -- a terminal row must collapse to its own one-hot or the
        target leaves the simplex."""
        if "terminals" in batch:
            return self.discount * (1.0 - jnp.asarray(batch["terminals"]).reshape(B, 1))
        if "masks" in batch:
            return self.discount * jnp.asarray(batch["masks"]).reshape(B, 1)
        return self.discount

    def stage_loss(self, stage, params, target, batch, cond, boot):
        # The phi stage is the parent's, verbatim -- that is the whole isolation argument.
        if stage == "phi":
            return super().stage_loss(stage, params, target, batch, cond, boot)
        assert stage == "sf", stage
        obs, action = batch["observations"], batch["actions"]
        nobs, z = batch["next_observations"], cond["z"]
        Bn = obs.shape[0]
        d = self._row_discount(batch, Bn)

        # Online logits. phi is DETACHED, exactly as the parent's SF loss detaches it, so
        # the CE trains hilp_sf only and phi's gradient path is unchanged.
        # Through `_le`, like the parent: with td_jepa's left encoder on, the SF tower is
        # built on the ENCODED observation (sf/model.py:82) and applying it to a raw row
        # would be a shape error at best and a silently different tower at worst.
        psi = self._sf(params["hilp_sf"], self._le(params, obs), z, action)   # [P, B, d]
        phi_pool = jax.lax.stop_gradient(
            self._feature_net(params["hilp_phi1"], params["hilp_stats"], nobs))  # [B, d]
        logits = jnp.einsum("pbd,cd->pbc", psi, phi_pool) / self.temp         # [P, B, B]
        logp = jax.nn.log_softmax(logits, -1)

        # Target rows, per-head. Target psi at the bootstrap action, target phi for the
        # pool -- the parent reads the ONLINE phi for its regression target (its TRAP 4),
        # but here the pool defines the OUTCOME SPACE of a distribution, and mixing an
        # online pool into a target softmax makes the target move with the online net
        # within a step. Both nets are stop_gradded either way.
        t_psi = self._sf(target["hilp_sf"], self._le(target, nobs), z,
                         boot["next_action"])                                 # [P, B, d]
        t_phi = jax.lax.stop_gradient(
            self._feature_net(target["hilp_phi1"], target["hilp_stats"], nobs))
        w_bar = jax.nn.softmax(jnp.einsum("pbd,cd->pbc", t_psi, t_phi) / self.temp, -1)

        k = jnp.eye(Bn)
        t = jax.lax.stop_gradient((1.0 - d) * k + d * w_bar)                  # [P, B, B]
        ce = -(t * logp).sum(-1).mean()
        sf_loss = self.sf_loss_scale * ce

        w = jnp.exp(logp)
        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)
        return sf_loss, {
            "sf_loss": sf_loss, "density_ce": ce,
            "density_ess": ess_rows.mean(),
            "density_ess_median": jnp.median(ess_rows),
            "density_w_entropy": -jnp.mean(jnp.sum(w * logp, axis=-1)),
            "density_tgt_entropy": -jnp.mean(jnp.sum(t * jnp.log(t + 1e-8), axis=-1)),
            "density_logit_max": jnp.max(logits, axis=-1).mean(),
            "phi_norm": jnp.mean(jnp.linalg.norm(phi_pool, axis=-1)),
            "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1)),
        }

    def sample_cond(self, net, batch, rng):
        nobs = batch["next_observations"]
        # q_fn's candidate pool: the batch's own next states through the ONLINE phi,
        # stop-gradded. q_fn cannot reach the batch through its signature.
        pool = jax.lax.stop_gradient(self._feature_net(
            net["hilp_phi1"].params, net["hilp_stats"].params, nobs))
        if not self.goal_kernel:
            cond = super().sample_cond(net, batch, rng)
            return {**cond, "_pool_phi": pool}
        # goal_kernel: z = HILP's own whitened goal index, sqrt(d)*normalize(phi(g) @ pinv(cov)),
        # for every row (the parent's mix_ratio encoding, applied to real goals only).
        goals = batch["actor_goals"] if "actor_goals" in batch else \
            nobs[jax.random.permutation(rng, nobs.shape[0])]
        phi_g = jax.lax.stop_gradient(self._feature_net(
            net["hilp_phi1"].params, net["hilp_stats"].params, goals))
        cov = (pool.T @ pool) / pool.shape[0]
        z = jax.lax.stop_gradient(project_z(phi_g @ jnp.linalg.pinv(cov), self.norm_z))
        return {"z": z, "_pool_phi": pool,
                "_pool_u_raw": jax.lax.stop_gradient(_raw_block_cols(nobs, self.kernel_blocks)),
                "_goal_obs": jax.lax.stop_gradient(goals)}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Reward integration through the normalized successor density, at FROZEN critic
        params -- the measure trains only through stage_loss, never through the actor's
        -Q.mean() term. Gradients flow through `actions` only."""
        del stop_encoder_grad
        pool, z = cond["_pool_phi"], cond["z"]                                # [C, d], [B, d]
        psi = self._sf(jax.lax.stop_gradient(params["hilp_sf"]),
                       self._le(jax.lax.stop_gradient(params), obs), z, actions)
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum("pbd,cd->pbc", psi, pool) / a_tau, -1)  # [P, B, C]
        if self.goal_kernel:
            # the goal's query-set-normalized object-coordinate kernel over the pool
            kg = jax.lax.stop_gradient(
                _delta_phi(cond["_goal_obs"], cond["_pool_u_raw"], self.kernel_blocks))  # [B, C]
        else:
            # r(u) = phi(u)^T z is HILP's own reward convention, the one infer_cond inverts.
            kg = jax.lax.stop_gradient(jnp.einsum("cd,bd->bc", pool, z))      # [B, C]
        Qs = (w * kg[None]).sum(-1)                                           # [P, B]
        qmean, _ = targets_uncertainty(Qs, self.num_parallel)
        if self.actor_q_center:
            return qmean, (qmean - qmean.mean())[None]
        return qmean, Qs


def _hilp_density_build(cls, ex_obs, ex_actions, config):
    if bool(config.get("hilp_q_loss", True)):
        raise HilpDensityConfigError(
            "hilp_density refuses hilp_q_loss=true: that flag selects between regressing "
            "the SF vector and the scalar <psi,z>, and this critic does neither -- it "
            "fits a categorical distribution. Set hilp_q_loss: false.")
    critic, params, targets = HILPCritic.build.__func__(cls, ex_obs, ex_actions, config)
    goal_kernel = bool(config.get("goal_kernel", False))
    kernel_blocks = ()
    if goal_kernel:
        kernel_blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        need = max(b for (_a, b, _e) in kernel_blocks)
        if int(ex_obs.shape[-1]) < need:
            raise HilpDensityConfigError(
                f"goal_kernel=true needs observations at least {need} wide for every "
                f"measure_hit_spec block; got {int(ex_obs.shape[-1])}.")
    critic = dataclasses.replace(
        critic,
        temp=float(config.get("temp", 1.0)),
        actor_temp=config.get("actor_temp", None),
        actor_q_center=bool(config.get("actor_q_center", True)),
        goal_kernel=goal_kernel, kernel_blocks=kernel_blocks,
    )
    return critic, params, targets


HILPDensityCritic.build = classmethod(_hilp_density_build)
