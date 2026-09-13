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

THE REGRESSION TASK INDEX (2026-09-13; `hilp_density_task_index`, default 'feature' =
bit-identical to the above). The port of Density-RLDP's LEVER 3 (rldp_density.py) to this
substrate, the same lever under the same helper functions:

  'feature'     the shipped interface: training z = whitened phi(permuted next state) mixed
                50/50 with sphere draws (hilp.py::sample_cond), actor reward phi(u)^T z,
                infer_cond = lstsq of the rewards on the centered phi, map_eval_cond =
                HILP's normalize(phi(g) - phi(s)).
  'regression'  density_fb's: for each row the goal is the dataset's hindsight goal
                (batch["actor_goals"]; HILPFutureDataset serves them under
                `hilp_goal_sampling: true` with the GC goal-sampling keys), its reward on
                the batch's candidate next states is the query-set-normalized kernel
                `_delta_phi` over the hit-spec object columns (the SAME parsed spec
                goal_kernel reads, `kernel_blocks`), and z is the ridge regression of that
                reward on the CENTERED features of a SLOW EMA copy of phi1 (group
                "hilp_phi_zenc", rate z_enc_tau) at the stored running moments mu_hat /
                C_hat (group "hilp_zstats", EMA rate z_sigma_tau, plus the frozen eval
                pool), through density_fb's own `_z_centered_moment` / `_z_centered_coef` /
                `_z_centered_index` (reused as function objects, not copied: ridge lam =
                0.01 tr(C)/d, eigenvalue floor max(1e-6 tr(C)/d, 1e-12), constant-reward
                fallback, sqrt(d) normalization). A `hilp_mix_ratio` fraction of the rows
                keeps that index (td_jepa's train_goal_ratio, 0.5 on the campaign
                substrate), the rest is a uniform sqrt(d)-sphere draw -- the parent's own
                mix with its whitened-feature branch swapped for the regressed index. The
                actor reward is density_fb's 'latent_centered': r_z(u) = (phi_slow(u) -
                mu_hat)^T z / sqrt(d) on the batch candidates. The SF tower conditions on
                -- and the bootstrap action is drawn at -- the same map applied to
                batch["value_goals"] (cond["_cvec_vgoal"], boot["next_action_vgoal"]), as
                density_fb does. `infer_cond` and `map_eval_cond` (env goal -> kernel
                reward on the frozen pool of next states) go through the SAME solve on the
                same slow tower and moments, so the training index and the test index are
                one function (`_z_index`). `update_stats` writes the slow tower, the moments
                and the pool AFTER the parent's own phi running-mean EMA; those two groups
                take no gradient step and no polyak.

The index basis is the RAW slow phi1 (through the `_phi_in` choke point), not the
running-mean-centered `_feature_net`: the solve centers at mu_hat and the actor reward is
centered at mu_hat too, so HILP's running mean would cancel exactly in both places. The
Hilbert loss, the expectile, the temperature, the phi stage and q_fn's softmax over the
ONLINE centered feature pool are all unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics.hilp import HILPCritic
from impls.critics._measure_common import parse_hit_spec
from impls.critics.density_fb import (DensityFBCritic, _ZStatsState, _delta_phi,
                                      _raw_block_cols)
from impls.utils.fb_math import project_z

# The regression index's two state groups: the slow EMA index tower (a copy of phi1) and
# the running moments + frozen eval pool. Neither is in any stage_groups nor in TARGETS;
# `update_stats` is their sole writer (rldp_density's backward_zenc / rldp_zstats, under
# this critic's `hilp_*` namespace -- _LR_KEY is keyed by group name across every critic).
_REG_STATE_GROUPS = ("hilp_phi_zenc", "hilp_zstats")


class HilpDensityConfigError(ValueError):
    """Raised at build time for a structurally invalid Density-HILP config."""


def _temperature_setting(config, canonical, legacy, default):
    """Accept campaign and legacy temperature names without silently picking a conflict."""
    if canonical in config and legacy in config and config[canonical] != config[legacy]:
        raise HilpDensityConfigError(
            f"Conflicting Density-HILP temperatures: {canonical}={config[canonical]!r} "
            f"and {legacy}={config[legacy]!r}. Set one name or give both the same value.")
    return config.get(canonical, config.get(legacy, default))


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
    # ---- the regression task index (module docstring). Default 'feature' = bit-identical.
    task_index: str = "feature"          # 'feature' | 'regression'
    # knobs, density_fb's names and round-1 values (read only under 'regression')
    z_ridge_lambda_mult: float = 0.01    # lam = mult * tr(C_hat)/d
    z_cov_eig_floor: float = 1e-6        # eigenvalue floor, a multiple of tr(C_hat)/d
    z_const_eps: float = 1e-6            # constant-reward fallback threshold on ||c||
    z_enc_tau: float = 0.0005            # EMA rate of the slow index tower
    z_sigma_tau: float = 0.005           # EMA rate of mu_hat / C_hat
    z_eval_pool_m: int = 256             # M of the frozen eval pool (map_eval_cond)
    # modules, named after their groups (Agent.create does getattr(critic, g))
    hilp_phi_zenc: Any = None            # the phi1 module def; params are the slow copy
    hilp_zstats: Any = None              # _ZStatsState

    # density_fb's centered ridge index, REUSED (the same function objects; they read
    # z_dim / z_ridge_lambda_mult / z_cov_eig_floor / z_const_eps and `_B`, all carried
    # here under the same names). `_B` takes FULL observation rows: `_phi` applies the
    # `_phi_in` choke point itself, so the "u_sc" argument of the moment is a full-row pool.
    _z_centered_moment = DensityFBCritic._z_centered_moment
    _z_centered_coef = DensityFBCritic._z_centered_coef
    _z_centered_index = DensityFBCritic._z_centered_index

    @property
    def regression(self):
        return self.task_index == "regression"

    def _B(self, p, x):
        """The index basis: the RAW phi1 tower at params `p` (the slow copy, when called
        from the solve) on full observation rows, through `_phi_in`. See the docstring on
        why the running mean is not subtracted here."""
        return self._phi(p, x)

    def _z_index(self, zenc_params, zstats, rows, r):
        """THE one solve: reward samples `r` [B, M] (or [M]) on the candidate rows `rows`
        [M, obs_dim] -> index [B, d], at the STORED slow tower and moments. Training rows,
        the value leg, infer_cond and map_eval_cond all go through here."""
        m_c = self._z_centered_moment(zenc_params, rows, zstats["mu"], r)
        return self._z_centered_index(m_c, zstats["cov"])

    def _z_mix(self, z_g, rng):
        """Keep the regressed index with probability mix_ratio (td_jepa's train_goal_ratio,
        the parent's own mix_ratio), else a fresh uniform sqrt(d)-sphere draw."""
        n = z_g.shape[0]
        r_z, r_mask = jax.random.split(rng)
        z_rand = project_z(jax.random.normal(r_z, (n, self.z_dim)), self.norm_z)
        keep = (jax.random.uniform(r_mask, (n,)) < self.mix_ratio)[:, None]
        return jax.lax.stop_gradient(jnp.where(keep, z_g, z_rand))

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
        nobs = batch["next_observations"]
        if self.regression:
            # The SF tower conditions on the VALUE-leg index and the bootstrap action is
            # the actor's draw at that same index (density_fb's discipline);
            # boot["next_action"] is the ACTOR-leg draw, a different goal distribution.
            z = cond["_cvec_vgoal"]
            if "next_action_vgoal" not in boot:
                raise KeyError(
                    "hilp_density_task_index='regression' needs boot['next_action_vgoal'] "
                    "-- the bootstrap action drawn at cond['_cvec_vgoal'] (FlowBC/DDPGBC "
                    "actors emit it whenever the cond carries '_cvec_vgoal'). "
                    "boot['next_action'] is the ACTOR-leg action and is NOT a valid "
                    "bootstrap for this target.")
            next_action = boot["next_action_vgoal"]
        else:
            z, next_action = cond["z"], boot["next_action"]
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
                         next_action)                                         # [P, B, d]
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
        if self.regression:
            return self._sample_cond_regression(net, batch, rng, pool)
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

    def _sample_cond_regression(self, net, batch, rng, pool):
        """The regression index's conditioning: TWO indices from the SAME solve, the actor
        leg at actor_goals and the value leg at value_goals (what the SF tower conditions
        on and where the bootstrap action is drawn), each mixed with sphere rows; plus the
        operands q_fn cannot reach through its signature."""
        if "actor_goals" not in batch or "value_goals" not in batch:
            raise KeyError(
                "hilp_density_task_index='regression' needs batch['actor_goals'] and "
                "batch['value_goals'] (the hindsight goals its training tasks are built "
                "from). HILPFutureDataset serves them under `hilp_goal_sampling: true` "
                "with the GC goal-sampling keys (critic/hilp_density_reg.yaml); a batch "
                "without them cannot run this mode -- no permuted-next-state fallback.")
        nobs = batch["next_observations"]
        zenc_p, st = net["hilp_phi_zenc"].params, net["hilp_zstats"].params
        u_raw = jax.lax.stop_gradient(_raw_block_cols(nobs, self.kernel_blocks))   # [M, d_c]
        r_a, r_v = jax.random.split(rng)

        def _z_at(goals, r_mix):
            r = jax.lax.stop_gradient(_delta_phi(goals, u_raw, self.kernel_blocks))  # [B, M]
            return self._z_mix(self._z_index(zenc_p, st, nobs, r), r_mix)

        return {
            "z": _z_at(batch["actor_goals"], r_a),
            "_cvec_vgoal": _z_at(batch["value_goals"], r_v),
            "_pool_phi": pool,
            # the CENTERED slow-tower embedding of the candidate set: the basis the index
            # was solved against, and q_fn's reward basis ('latent_centered').
            "_pool_zenc_c": jax.lax.stop_gradient(self._B(zenc_p, nobs) - st["mu"][None]),
        }

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
        elif self.regression:
            # r_z(u) = (phi_slow(u) - mu_hat)^T z / sqrt(d) -- the reward the index was
            # regressed from, on the same centered slow-tower basis.
            kg = jax.lax.stop_gradient(
                jnp.einsum("cd,bd->bc", cond["_pool_zenc_c"], z)) / jnp.sqrt(float(self.z_dim))
        else:
            # r(u) = phi(u)^T z is HILP's own reward convention, the one infer_cond inverts.
            kg = jax.lax.stop_gradient(jnp.einsum("cd,bd->bc", pool, z))      # [B, C]
        Qs = (w * kg[None]).sum(-1)                                           # [P, B]
        q = jnp.mean(Qs, axis=0) if self.actor_head_reduce == "mean" else jnp.min(Qs, axis=0)
        if self.actor_q_center:
            return q, (q - q.mean())[None]
        return q, Qs

    def infer_cond(self, net, obs, rewards):
        """'feature': HILP's lstsq on the centered phi (the parent's). 'regression': the
        reward samples go through the training index's own solve."""
        if not self.regression:
            return super().infer_cond(net, obs, rewards)
        r = jnp.asarray(rewards)
        z = self._z_index(net["hilp_phi_zenc"].params, net["hilp_zstats"].params,
                          jnp.asarray(obs), r)
        return {"z": z[0] if r.ndim == 1 else z}

    def map_eval_cond(self, net, cond, observations=None):
        """Eval-time goal -> index. 'feature': HILP's z = normalize(phi(g) - phi(s)) (the
        parent's, joint in goal and current observation). 'regression': the goal becomes
        kernel reward samples on the FROZEN pool of next states and goes through the same
        solve as every training row (density_fb.py::map_eval_cond); `observations` is
        accepted for the harness's calling convention and unused."""
        if not self.regression or "goal" not in cond:
            return super().map_eval_cond(net, cond, observations=observations)
        g = cond["goal"]
        unbatched = g.ndim == 1
        g2 = g[None] if unbatched else g
        st = net["hilp_zstats"].params
        pool = st["pool"]                                                  # [M, obs_dim]
        kg = jax.lax.stop_gradient(_delta_phi(g2, _raw_block_cols(pool, self.kernel_blocks),
                                              self.kernel_blocks))         # [B, M]
        z = self._z_index(net["hilp_phi_zenc"].params, st, pool, kg)
        return {**cond, "z": z[0] if unbatched else z}

    def update_stats(self, net, pre_net, batch):
        """The parent's phi running-mean EMA, then -- under 'regression' only -- the index
        state, written once per update by Agent.apply_update AFTER the gradient step and
        the polyak targets, from the PRE-STEP online net (density_fb.py::update_stats's
        order): (1) the slow index tower EMA'd toward the pre-step online phi1, (2) mu_hat
        / C_hat EMA'd over this batch's next states at the OLD slow tower, (3) the eval
        pool frozen on the first update. A Python-level no-op under 'feature'."""
        net = super().update_stats(net, pre_net, batch)
        if not self.regression:
            return net
        nobs = batch["next_observations"]
        zenc_old = net["hilp_phi_zenc"].params
        zenc_new = jax.tree_util.tree_map(
            lambda p, ep: p * self.z_enc_tau + ep * (1.0 - self.z_enc_tau),
            pre_net["hilp_phi1"].params, zenc_old)
        b = jax.lax.stop_gradient(self._B(zenc_old, nobs))                 # [M, d]
        mu_b = jnp.mean(b, axis=0)
        db = b - mu_b[None]
        cov_b = db.T @ db / b.shape[0]
        st = net["hilp_zstats"].params
        tau = self.z_sigma_tau
        mu_new = st["mu"] * (1.0 - tau) + mu_b * tau
        cov_new = st["cov"] * (1.0 - tau) + cov_b * tau
        cov_new = 0.5 * (cov_new + cov_new.T)
        if nobs.shape[0] < self.z_eval_pool_m:
            raise ValueError(
                f"hilp_density_z_eval_pool_m={self.z_eval_pool_m} needs a batch of at least "
                f"that many next states to draw the frozen eval pool from, got "
                f"{nobs.shape[0]}.")
        pool_new = jnp.where(st["filled"] > 0.5, st["pool"], nobs[:self.z_eval_pool_m])
        cast = lambda new, old: jnp.asarray(new, old.dtype)
        net = dict(net)
        net["hilp_phi_zenc"] = net["hilp_phi_zenc"].replace(params=zenc_new)
        net["hilp_zstats"] = net["hilp_zstats"].replace(params={
            "mu": cast(mu_new, st["mu"]), "cov": cast(cov_new, st["cov"]),
            "pool": cast(pool_new, st["pool"]), "filled": jnp.ones_like(st["filled"])})
        return net


def _hilp_density_build(cls, ex_obs, ex_actions, config):
    if bool(config.get("hilp_q_loss", True)):
        raise HilpDensityConfigError(
            "hilp_density refuses hilp_q_loss=true: that flag selects between regressing "
            "the SF vector and the scalar <psi,z>, and this critic does neither -- it "
            "fits a categorical distribution. Set hilp_q_loss: false.")
    temp = _temperature_setting(config, "measure_density_temp", "temp", 1.0)
    actor_temp = _temperature_setting(config, "measure_density_actor_temp", "actor_temp", None)
    task_index = str(config.get("hilp_density_task_index", "feature"))
    if task_index not in ("feature", "regression"):
        raise HilpDensityConfigError(
            f"hilp_density_task_index must be 'feature' or 'regression', got {task_index!r}")
    regression = task_index == "regression"
    goal_kernel = bool(config.get("goal_kernel", False))
    if regression and goal_kernel:
        raise HilpDensityConfigError(
            "hilp_density_task_index='regression' and goal_kernel=true are two task "
            "interfaces for one critic (kernel reward at whitened goal embeddings vs the "
            "regressed index); pick one.")
    critic, params, targets = HILPCritic.build.__func__(cls, ex_obs, ex_actions, config)
    kernel_blocks = ()
    if goal_kernel or regression:
        # The kernel is DEFINED BY COLUMN SLICES of the observation, so the dataset must
        # actually carry those columns -- refuse at build rather than let jax slice short
        # rows into a silently different reward. One parse serves both interfaces.
        kernel_blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        need = max(b for (_a, b, _e) in kernel_blocks)
        if int(ex_obs.shape[-1]) < need:
            raise HilpDensityConfigError(
                f"{'goal_kernel=true' if goal_kernel else 'hilp_density_task_index=regression'}"
                f" needs observations at least {need} wide for every measure_hit_spec "
                f"block; got {int(ex_obs.shape[-1])}.")
    extra = {}
    if regression:
        z_lam = float(config.get("hilp_density_z_ridge_lambda_mult", 0.01))
        z_floor = float(config.get("hilp_density_z_cov_eig_floor", 1e-6))
        z_eps = float(config.get("hilp_density_z_const_eps", 1e-6))
        z_enc_tau = float(config.get("hilp_density_z_enc_tau", 0.0005))
        z_sigma_tau = float(config.get("hilp_density_z_sigma_tau", 0.005))
        z_pool_m = int(config.get("hilp_density_z_eval_pool_m", 256))
        if not (z_lam > 0.0 and z_floor > 0.0 and z_eps > 0.0):
            raise HilpDensityConfigError(
                "hilp_density_z_ridge_lambda_mult, hilp_density_z_cov_eig_floor and "
                "hilp_density_z_const_eps must all be > 0 (density_fb's contract: an "
                "unregularized/unfloored solve, or a constant reward encoded as noise)")
        if not (0.0 < z_enc_tau <= 0.1 and 0.0 < z_sigma_tau <= 1.0):
            raise HilpDensityConfigError(
                "hilp_density_z_enc_tau must be in (0, 0.1] and hilp_density_z_sigma_tau "
                f"in (0, 1]; got {z_enc_tau}, {z_sigma_tau}")
        bs = int(config.get("batch_size", z_pool_m))
        if not (1 < z_pool_m <= bs):
            raise HilpDensityConfigError(
                f"hilp_density_z_eval_pool_m={z_pool_m} must be > 1 and <= batch_size={bs}: "
                f"the frozen pool is drawn from one batch's next states and the centered "
                f"moment is identically zero at M=1")
        import copy
        import jax.random as jr
        # The slow tower starts as a COPY of phi1; the moments at mu=0, C=I (well-posed
        # first solve); the pool holds FULL observation rows (`_phi` slices them itself
        # under backward_input='hit_spec', `_delta_phi` reads the raw block columns).
        params["hilp_phi_zenc"] = copy.deepcopy(params["hilp_phi1"])
        zstats_mod = _ZStatsState(dim=int(critic.z_dim), pool_m=z_pool_m,
                                  d_c=int(ex_obs.shape[-1]),
                                  dtype=jnp.zeros((), ex_obs.dtype).dtype)
        params["hilp_zstats"] = zstats_mod.init(jr.PRNGKey(int(config["seed"]) + 7000))["params"]
        extra = dict(z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                     z_enc_tau=z_enc_tau, z_sigma_tau=z_sigma_tau, z_eval_pool_m=z_pool_m,
                     hilp_phi_zenc=critic.hilp_phi1, hilp_zstats=zstats_mod,
                     GROUPS=tuple(critic.GROUPS) + _REG_STATE_GROUPS)
    critic = dataclasses.replace(
        critic,
        temp=float(temp),
        actor_temp=actor_temp,
        actor_q_center=bool(config.get("actor_q_center", True)),
        goal_kernel=goal_kernel, kernel_blocks=kernel_blocks,
        task_index=task_index, **extra,
    )
    return critic, params, targets


HILPDensityCritic.build = classmethod(_hilp_density_build)
