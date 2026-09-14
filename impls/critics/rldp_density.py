"""RLDPDensityCritic — RLDP's representation learner with Density-FB's successor estimator.

Density-RLDP (design note: the Density-FB -> PSM/RLDP extension, 2026-08-24). RLDP's own
structure is the cleanest control for the density objective: the backward map B is learned
ENTIRELY by the self-predictive (SP) rollout and the orthonormality term — the successor
loss never touches it (rldp.py divergence 1). Swapping the successor loss therefore changes
exactly one thing, the estimator: squared Bellman residual on m = F^T B  becomes the
categorical Bellman cross-entropy on the NORMALIZED successor density

    p_h(j | s_i, a_i, z_i) = softmax_j( F_h(L(s_i), z_i, a_i)^T sg[B(s'_j)] / tau )

with target rows

    t_h(i, .) = (1 - d_i) one_hot(i) + d_i * sg[ softmax_j( F̄_h(L̄(s'_i), z_i, a'_i)^T
                                                           sg[B̄(s'_j)] / tau ) ],

d_i = gamma * (1 - terminal_i) — a terminal row puts ALL its mass on the observed
successor and every target row stays on the simplex (the plain (1 - gamma) coefficient
would not). a'_i = pi(s'_i, z_i) is the seam's stop-gradded bootstrap draw
(boot["next_action"]), identical to the parent's.

Inherited UNCHANGED from RLDPCritic (the point of the control):
  * the SP rollout `_sp_loss` (with the corrected h-step alignment, step k -> B(s_{t+k});
    see rldp.py::_sp_pairs),
  * the orthonormality term at B(obs),
  * B detached inside the successor term — the density CE trains forward + left_encoder
    only; a gradient-isolation test pins this,
  * sample_cond's mixed-z draw, map_eval_cond (z = project_z(B(g))),
    the target-network schedule, GROUPS / STAGES / _LR_KEY rows.

What must ALSO change once the score is a logit (design note sec. 4): F^T z is no longer
a value. `q_fn` becomes the reward integration

    q_h(s, a; z) = sum_j p_h(j | s, a, z) * ( B(u_j)^T z ),        u_j = batch next states,

r_z(u) = B(u)^T z is the latent reward convention. Reward inference solves
least squares against B, then applies the existing task-vector normalization.
The parent's reward moment is equivalent only for whitened backward features.
The goal-kernel variant keeps its inherited moment-based encoding, because its
actor is trained on goal embeddings instead of linear reward coefficients. The candidate
pool cannot reach q_fn through its signature, so `sample_cond` stashes it on cond
("_pool_B"), the same seam workaround density_fb.py uses ("_density_u_raw"/"_density_u_sc").
Ensemble reduction happens AFTER integration, on scalars (mean over heads, with the
parent's optional actor pessimism); the per-head target distributions are NEVER collapsed by
mean-minus-uncertainty — that operation can leave the simplex, so `fb_pessimism_penalty`
is lattice-refused (must be 0.0) rather than silently ignored.

`actor_q_center` (default true, the bridge spec-19a lattice rule for density actors):
the SECOND q_fn return becomes the centered vector, so FlowBC/FQL's |Qs|.mean() normalizer
measures the CONTROLLABLE spread of q rather than its offset. The first return is
untouched (centering -Q.mean() would zero the greedy signal — see density_fb.py::q_fn).

Exp A: `goal_kernel` (default FALSE = bit-identical to the above)
-----------------------------------------------------------------
The discriminating variant for the density_fb (0.97) vs rldp_density (0.24) gap: keep the
LEARNED-B outcome space (the softmax over B(next_obs) is untouched) but give the critic
density_fb's TASK INTERFACE — the query-set-normalized object-coordinate kernel reward at
REAL goals. Two changes, both config-gated:

  * `sample_cond`: z = project_z(B(hindsight goals)) with NO random mix — the batch's
    `actor_goals` when the dataset serves them, else permuted next_observations (the
    hindsight fallback; WindowedDataset, this critic's dataset_class, has no goal
    sampling). The goal ROWS are stashed on cond ("_goal_obs"), together with the RAW
    hit-spec block columns of the pool's next states ("_pool_u_raw"), because q_fn cannot
    reach the batch through its signature.
  * `q_fn`: w stays the softmax over the learned-B pool exactly as above, but
    kg = pool·z is replaced by kg = _delta_phi(goal_obs, _pool_u_raw, blocks) — density_fb's
    exact reward semantics (softmax over the query axis, eps-scaled internally),
    stop-gradded, at the SAME goals whose embeddings conditioned the actor.

stage_loss and map_eval_cond are untouched: cond["z"] now carries the hindsight-goal
embedding (sample_cond does the work; the bootstrap action is drawn by the actor at the
same cond["z"], so F and the bootstrap policy stay conditioned on one thing), and eval
still maps the env goal through z = project_z(B(goal)). The kernel helpers are imported
from density_fb.py / _measure_common.py rather than duplicated.

The three LEVERS (2026-09-12; each default-OFF = bit-identical to the above)
----------------------------------------------------------------------------
The three ways this retrofit still differs from Density-FB, made switchable so the gap
can be attributed:

  LEVER 1  `rldp_density_train_b` (bool). The density CE also trains the backward tower:
           the stop_gradient on the ONLINE B of the candidate pool inside `stage_loss` is
           dropped (the target-side B stays inside the stop-gradded target as before), so
           params["backward"] receives the CE gradient on top of the parent's SP +
           orthonormality gradients, which are unchanged. "backward" is already in
           `stage_groups` in the joint phase, so the optimizer steps it; the two-phase
           schedule (B frozen in phase 2) is refused at build.
  LEVER 2  `backward_input: hit_spec` -- the parent's object-coordinate B input
           (rldp.py::_b_in). Nothing here; critic/rldp_density_oc.yaml is that config.
  LEVER 3  `rldp_density_task_index` ('feature' | 'regression'). 'feature' is the shipped
           interface above: training z = project_z(B(random next state)) mixed with
           sphere draws, actor reward pool.z, infer_cond = lstsq on B. 'regression' is
           density_fb's: for each row the goal is the dataset's hindsight goal
           (batch["actor_goals"]; WindowedDataset serves them under
           `windowed_goal_sampling: true`), its reward on the batch's candidate next
           states is the query-set-normalized kernel `_delta_phi` over the hit-spec object
           columns, and z is the ridge regression of that reward on the CENTERED features
           of a SLOW EMA copy of B (group "backward_zenc", rate z_enc_tau) at the stored
           running moments mu_hat / C_hat (group "rldp_zstats", EMA rate z_sigma_tau),
           through density_fb's own `_z_centered_moment` / `_z_centered_coef` /
           `_z_centered_index` (reused, not copied: ridge lam = 0.01 tr(C)/d, eigenvalue
           floor, constant-reward fallback, sqrt(d) normalization). A `train_goal_ratio`
           fraction of the rows keeps that index, the rest is a uniform sqrt(d)-sphere
           draw (the same 0.5 mix as 'feature'). The actor reward is density_fb's
           'latent_centered': r_z(u) = (B_slow(u) - mu_hat)^T z / sqrt(d). F conditions
           on -- and the bootstrap action is drawn at -- the same map applied to
           batch["value_goals"] (cond["_cvec_vgoal"], boot["next_action_vgoal"]), as
           density_fb does. `infer_cond` and `map_eval_cond` (env goal -> kernel reward on
           a frozen pool of next states) go through the SAME solve on the same slow tower
           and moments, so the training index and the test index are one function
           (`_z_index`). `update_stats` (the Agent hook density_fb uses) writes the slow
           tower, the moments and the pool; those two groups take no gradient step.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics._measure_common import parse_hit_spec
from impls.critics.density_fb import (DensityFBCritic, _ZStatsState, _delta_phi,
                                      _raw_block_cols)
from impls.critics.rldp import RLDPCritic
from impls.utils.fb_math import project_z, targets_uncertainty

# LEVER 3's two state groups: the slow EMA index tower and the running moments + frozen
# eval pool. Neither is in any stage_groups nor in TARGETS; `update_stats` is their sole
# writer (density_fb's density_b_zenc / density_zstats, under this critic's names).
_REG_STATE_GROUPS = ("backward_zenc", "rldp_zstats")


class RldpDensityConfigError(ValueError):
    """Raised at build time for a structurally invalid Density-RLDP config."""


@dataclasses.dataclass(frozen=True)
class RLDPDensityCritic(RLDPCritic):
    temp: float = 1.0
    actor_temp: Any = None          # None => temp (bit-identical readout temperature)
    actor_q_center: bool = True
    # Exp A (module docstring): density_fb's task interface over the learned-B outcome
    # space. Default OFF = bit-identical to the shipped rldp_density. The parsed
    # measure_hit_spec lives on the PARENT's `blocks` field (rldp.py), which is read by
    # goal_kernel's kernel reward here AND by the parent's backward_input="hit_spec"
    # input map; both parse the same config key, so one field serves both.
    goal_kernel: bool = False
    # ---- the three levers (module docstring). All default-off. ----
    train_b: bool = False                # LEVER 1: the CE trains B
    task_index: str = "feature"          # LEVER 3: 'feature' | 'regression'
    loss_kind: str = "ce"                # 'ce' | 'sq' (squared error on the softmax probabilities; ablation)
    # LEVER 3 knobs, density_fb's names and round-1 values (read only under 'regression').
    z_ridge_lambda_mult: float = 0.01    # lam = mult * tr(C_hat)/d
    z_cov_eig_floor: float = 1e-6        # eigenvalue floor, a multiple of tr(C_hat)/d
    z_const_eps: float = 1e-6            # constant-reward fallback threshold on ||c||
    z_enc_tau: float = 0.0005            # EMA rate of the slow index tower
    z_sigma_tau: float = 0.005           # EMA rate of mu_hat / C_hat
    z_eval_pool_m: int = 256             # M of the frozen eval pool (map_eval_cond)
    # LEVER 3 modules, named after their groups (Agent.create does getattr(critic, g)).
    backward_zenc: Any = None            # the backward module def; params are the slow copy
    rldp_zstats: Any = None              # _ZStatsState
    # GROUPS becomes an instance field so 'regression' can append the two state groups;
    # the default is the parent's tuple, so every other build is unchanged.
    GROUPS: tuple = RLDPCritic.GROUPS

    # density_fb's centered ridge index, REUSED (the same function objects; they read
    # z_dim / z_ridge_lambda_mult / z_cov_eig_floor / z_const_eps and `_B`, all of which
    # this class carries under the same names). `_B` here takes FULL observation rows and
    # applies `_b_in` itself, so the "u_sc" argument of the moment is a full-row pool.
    _z_centered_moment = DensityFBCritic._z_centered_moment
    _z_centered_coef = DensityFBCritic._z_centered_coef
    _z_centered_index = DensityFBCritic._z_centered_index

    @property
    def regression(self):
        return self.task_index == "regression"

    def stage_groups(self, stage):
        # The state groups are written by update_stats only; never stepped.
        return tuple(g for g in super().stage_groups(stage) if g not in _REG_STATE_GROUPS)

    def _z_index(self, zenc_params, zstats, rows, r):
        """THE one solve (LEVER 3): reward samples `r` [B, M] (or [M]) on the candidate
        rows `rows` [M, obs_dim] -> index [B, d], at the STORED slow tower and moments.
        Training rows, the value leg, infer_cond and map_eval_cond all go through here."""
        m_c = self._z_centered_moment(zenc_params, rows, zstats["mu"], r)
        return self._z_centered_index(m_c, zstats["cov"])

    def _z_mix(self, z_g, rng):
        """Keep the encoded index with probability train_goal_ratio, else a fresh uniform
        sqrt(d)-sphere draw -- the parent's mix (sample_cond) with the goal branch swapped
        for the regression index."""
        n = z_g.shape[0]
        r_z, r_mask = jax.random.split(rng)
        z_rand = project_z(jax.random.normal(r_z, (n, self.z_dim)), self.norm_z)
        keep = (jax.random.uniform(r_mask, (n,)) < self.train_goal_ratio)[:, None]
        return jax.lax.stop_gradient(jnp.where(keep, z_g, z_rand))

    def _row_discount(self, batch, B):
        """gamma * (1 - terminated), shaped [B, 1] (same convention as PSMCritic).
        The parent has no row discount (scalar gamma inside a squared residual); the
        categorical target NEEDS it — a terminal row must collapse to its own one-hot
        for the target to stay a distribution."""
        if "terminals" in batch:
            return self.discount * (1.0 - jnp.asarray(batch["terminals"]).reshape(B, 1))
        if "masks" in batch:
            return self.discount * jnp.asarray(batch["masks"]).reshape(B, 1)
        return self.discount

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "fb"
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs  # bw_encoder = Identity for state, as in the parent
        if self.regression:
            # LEVER 3: F conditions on the VALUE-leg index and the bootstrap action is the
            # actor's draw at that same index (density_fb's discipline); boot["next_action"]
            # is the ACTOR-leg draw, a different goal distribution.
            z = cond["_cvec_vgoal"]
            if "next_action_vgoal" not in boot:
                raise KeyError(
                    "rldp_density_task_index='regression' needs boot['next_action_vgoal'] "
                    "-- the bootstrap action drawn at cond['_cvec_vgoal'] (FlowBC/DDPGBC "
                    "actors emit it whenever the cond carries '_cvec_vgoal'). "
                    "boot['next_action'] is the ACTOR-leg action and is NOT a valid "
                    "bootstrap for this target.")
            next_action = boot["next_action_vgoal"]
        else:
            z, next_action = cond["z"], boot["next_action"]
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()
        d = self._row_discount(batch, B)

        # --- density successor term (replaces fb_off + fb_diag) ---
        # Online logits. Divergence 1 is PRESERVED by default: B(goal) detached, the CE
        # trains forward + left_encoder only. LEVER 1 (train_b) drops that stop_gradient.
        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)                     # [P, B, z]
        Bg = self._B(params["backward"], goal)                             # [B, z]
        if not self.train_b:
            Bg = jax.lax.stop_gradient(Bg)
        logits = jnp.einsum("pbz,cz->pbc", Fs, Bg) / self.temp            # [P, B, B]
        logp = jax.nn.log_softmax(logits, -1)

        # Target logits: per-head, with the parent's choice of backward snapshot.
        # No mean-minus-uncertainty on distributions; see module docstring.
        nle = self._le(target["left_encoder"], next_obs)
        tFs = self._F(target["forward"], nle, z, next_action)
        tB = self._B(params["backward"] if self.fb_target_online_b else target["backward"],
                     goal)
        w_bar = jax.nn.softmax(jnp.einsum("pbz,cz->pbc", tFs, tB) / self.temp, -1)

        k = jnp.eye(B)
        t = jax.lax.stop_gradient((1.0 - d) * k + d * w_bar)               # [P, B, B]
        ce = -(t * logp).sum(-1).mean()
        sq = 0.5 * ((jnp.exp(logp) - t) ** 2).sum(-1).mean()
        fit = ce if self.loss_kind == "ce" else sq

        # --- SP head + orthonormality, verbatim from the parent ---
        sp, B_pred = self._sp_loss(params, target, batch)
        orth, _, _ = self._ortho(B_pred, off, off_sum)

        loss = fit + sp + self.ortho_coef * orth

        w = jnp.exp(logp)
        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [P, B]
        return loss, {
            "fb_loss": loss, "sp_loss": sp, "ortho_loss": orth,
            "density_ce": ce,
            "density_sq": sq,
            "density_ess": ess_rows.mean(),
            "density_ess_median": jnp.median(ess_rows),
            "density_w_entropy": -jnp.mean(jnp.sum(w * logp, axis=-1)),
            "density_tgt_entropy": -jnp.mean(jnp.sum(t * jnp.log(t + 1e-8), axis=-1)),
            "density_logit_max": jnp.max(logits, axis=-1).mean(),
        }

    def sample_cond(self, net, batch, rng):
        nobs = batch["next_observations"]
        # The candidate pool for q_fn's reward integration: the batch's own next states
        # through the ONLINE backward map, stop-gradded — q_fn cannot reach the batch
        # through its signature (the density_fb.py seam workaround).
        pool = jax.lax.stop_gradient(self._B(net["backward"].params, nobs))
        if self.regression:
            return self._sample_cond_regression(net, batch, rng, pool)
        if not self.goal_kernel:
            cond = super().sample_cond(net, batch, rng)
            return {**cond, "_pool_B": pool}
        # Exp A: z = project_z(B(hindsight goals)), NO random mix (density_fb's task
        # interface conditions every row on a real goal). `actor_goals` when the dataset
        # serves them (GCDataset); WindowedDataset — this critic's dataset_class — has no
        # goal sampling, so the fallback is the standard hindsight surrogate: permuted
        # next_observations.
        if "actor_goals" in batch:
            goals = batch["actor_goals"]
        else:
            goals = nobs[jax.random.permutation(rng, nobs.shape[0])]
        z = jax.lax.stop_gradient(
            project_z(self._B(net["backward"].params, goals), self.norm_z))
        return {
            "z": z, "_pool_B": pool,
            # q_fn's kernel operands, stashed because its signature cannot reach the
            # batch: the RAW hit-spec block columns of the pool's next states, and the
            # SAME goal rows whose embeddings conditioned the actor (module docstring).
            "_pool_u_raw": jax.lax.stop_gradient(_raw_block_cols(nobs, self.blocks)),
            "_goal_obs": jax.lax.stop_gradient(goals),
        }

    def _sample_cond_regression(self, net, batch, rng, pool):
        """LEVER 3's conditioning: TWO indices from the SAME solve, the actor leg at
        actor_goals and the value leg at value_goals (what F conditions on and where the
        bootstrap action is drawn), each mixed with sphere rows; plus the operands q_fn
        cannot reach through its signature."""
        if "actor_goals" not in batch or "value_goals" not in batch:
            raise KeyError(
                "rldp_density_task_index='regression' needs batch['actor_goals'] and "
                "batch['value_goals'] (the hindsight goals its training tasks are built "
                "from). WindowedDataset serves them under `windowed_goal_sampling: true` "
                "with the GC goal-sampling keys (critic/rldp_density_reg.yaml); a batch "
                "without them cannot run this mode -- no permuted-next-state fallback.")
        nobs = batch["next_observations"]
        zenc_p, st = net["backward_zenc"].params, net["rldp_zstats"].params
        u_raw = jax.lax.stop_gradient(_raw_block_cols(nobs, self.blocks))      # [M, d_c]
        r_a, r_v = jax.random.split(rng)

        def _z_at(goals, r_mix):
            r = jax.lax.stop_gradient(_delta_phi(goals, u_raw, self.blocks))   # [B, M]
            return self._z_mix(self._z_index(zenc_p, st, nobs, r), r_mix)

        return {
            "z": _z_at(batch["actor_goals"], r_a),
            "_cvec_vgoal": _z_at(batch["value_goals"], r_v),
            "_pool_B": pool,
            # the CENTERED slow-tower embedding of the candidate set: the basis the index
            # was solved against, and q_fn's reward basis ('latent_centered').
            "_pool_zenc_c": jax.lax.stop_gradient(self._B(zenc_p, nobs) - st["mu"][None]),
        }

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Reward integration through the normalized successor density, at FROZEN critic
        params — the measure trains only through stage_loss, never through the actor's
        -Q.mean() term (density_fb.py discipline). Gradients flow through `actions` only."""
        del stop_encoder_grad  # towers hard-frozen below; nothing further to cut
        pool = cond["_pool_B"]                                             # [C, z], sg'd
        z = cond["z"]
        le = self._le(jax.lax.stop_gradient(params["left_encoder"]), obs)
        Fs = self._F(jax.lax.stop_gradient(params["forward"]), le, z, actions)  # [P, B, z]
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum("pbz,cz->pbc", Fs, pool) / a_tau, -1)  # [P, B, C]
        if self.goal_kernel:
            # Exp A: density_fb's exact reward semantics over the learned measure — the
            # query-set-normalized object-coordinate kernel at the REAL goal rows that
            # conditioned the actor, in place of the latent reward pool·z. _delta_phi
            # eps-scales internally and softmaxes over the query (pool) axis, so each kg
            # row is on the simplex.
            kg = jax.lax.stop_gradient(
                _delta_phi(cond["_goal_obs"], cond["_pool_u_raw"], self.blocks))  # [B, C]
        elif self.regression:
            # LEVER 3: r_z(u) = (B_slow(u) - mu_hat)^T z / sqrt(d) -- the reward the index
            # was regressed from, on the same centered slow-tower basis.
            kg = jax.lax.stop_gradient(
                jnp.einsum("cz,bz->bc", cond["_pool_zenc_c"], z)) / jnp.sqrt(float(self.z_dim))
        else:
            kg = jax.lax.stop_gradient(jnp.einsum("cz,bz->bc", pool, z))     # [B, C]
        Qs = (w * kg[None]).sum(-1)                                          # [P, B]
        qmean, qunc = targets_uncertainty(Qs, self.num_parallel)
        if self.actor_pessimism:
            qmean = qmean - self.actor_pessimism * qunc
        if self.actor_q_center:
            # Centered magnitude for the FlowBC/FQL |Qs|.mean() normalizer; the FIRST
            # return stays uncentered (see density_fb.py::q_fn on why).
            return qmean, (qmean - qmean.mean())[None]
        return qmean, Qs

    def infer_cond(self, net, obs, rewards):
        """Fit the latent reward B(u)^T z; preserve the goal-kernel task encoding. Under
        LEVER 3 the reward samples go through the training index's own solve."""
        if self.regression:
            r = jnp.asarray(rewards)
            if r.ndim == 2 and r.shape[-1] == 1:
                # Dataset.sample hands rewards as [M, 1] (impls/utils/datasets.py, reshape(B, -1)):
                # ONE task with M samples, not M tasks of one sample -- the centered moment
                # refuses M=1. Flatten so the solve sees [M]. (2026-09-14, walker block)
                r = r.reshape(-1)
            z = self._z_index(net["backward_zenc"].params, net["rldp_zstats"].params,
                              jnp.asarray(obs), r)
            return {"z": z[0] if r.ndim == 1 else z}
        if self.goal_kernel:
            return super().infer_cond(net, obs, rewards)
        Bm = self._B(net["backward"].params, jnp.asarray(obs))
        r = jnp.asarray(rewards).reshape(-1, 1)
        # Unlike the parent's linear successor-measure head, categorical reward
        # integration does not implicitly supply an inverse feature covariance.
        # lstsq also gives a finite minimum-norm solution for deficient rank.
        z = jnp.linalg.lstsq(Bm, r, rcond=None)[0].reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    def map_eval_cond(self, net, cond):
        """Eval-time goal -> index. 'feature': the parent's z = project_z(B(goal)). LEVER 3:
        the goal becomes kernel reward samples on the FROZEN pool of next states and goes
        through the same solve as every training row (density_fb.py::map_eval_cond)."""
        if not self.regression or "goal" not in cond:
            return super().map_eval_cond(net, cond)
        g = cond["goal"]
        unbatched = g.ndim == 1
        g2 = g[None] if unbatched else g
        st = net["rldp_zstats"].params
        pool = st["pool"]                                                  # [M, obs_dim]
        kg = jax.lax.stop_gradient(_delta_phi(g2, _raw_block_cols(pool, self.blocks),
                                              self.blocks))                # [B, M]
        z = self._z_index(net["backward_zenc"].params, st, pool, kg)
        return {**cond, "z": z[0] if unbatched else z}

    def update_stats(self, net, pre_net, batch):
        """LEVER 3's persistent state, written once per update by Agent.apply_update AFTER
        the gradient step and the polyak targets, from the PRE-STEP online net -- the
        order density_fb.py::update_stats uses: (1) the slow index tower EMA'd toward the
        pre-step online B, (2) mu_hat / C_hat EMA'd over this batch's next states at the
        OLD slow tower, (3) the eval pool frozen on the first update. A Python-level
        no-op under 'feature', so nothing extra is traced there."""
        if not self.regression:
            return net
        nobs = batch["next_observations"]
        zenc_old = net["backward_zenc"].params
        zenc_new = jax.tree_util.tree_map(
            lambda p, ep: p * self.z_enc_tau + ep * (1.0 - self.z_enc_tau),
            pre_net["backward"].params, zenc_old)
        b = jax.lax.stop_gradient(self._B(zenc_old, nobs))                 # [M, d]
        mu_b = jnp.mean(b, axis=0)
        db = b - mu_b[None]
        cov_b = db.T @ db / b.shape[0]
        st = net["rldp_zstats"].params
        tau = self.z_sigma_tau
        mu_new = st["mu"] * (1.0 - tau) + mu_b * tau
        cov_new = st["cov"] * (1.0 - tau) + cov_b * tau
        cov_new = 0.5 * (cov_new + cov_new.T)
        if nobs.shape[0] < self.z_eval_pool_m:
            raise ValueError(
                f"rldp_density_z_eval_pool_m={self.z_eval_pool_m} needs a batch of at least "
                f"that many next states to draw the frozen eval pool from, got "
                f"{nobs.shape[0]}.")
        pool_new = jnp.where(st["filled"] > 0.5, st["pool"], nobs[:self.z_eval_pool_m])
        cast = lambda new, old: jnp.asarray(new, old.dtype)
        net = dict(net)
        net["backward_zenc"] = net["backward_zenc"].replace(params=zenc_new)
        net["rldp_zstats"] = net["rldp_zstats"].replace(params={
            "mu": cast(mu_new, st["mu"]), "cov": cast(cov_new, st["cov"]),
            "pool": cast(pool_new, st["pool"]), "filled": jnp.ones_like(st["filled"])})
        return net

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        if float(config.get("fb_pessimism_penalty", 0.0)) != 0.0:
            raise RldpDensityConfigError(
                "rldp_density refuses fb_pessimism_penalty != 0: mean-minus-uncertainty "
                "on target DISTRIBUTIONS leaves the simplex (can go negative / "
                "sub-normalized). Pessimism belongs after reward integration, on scalars.")
        temp = float(config.get("temp", 1.0))   # density_fb's default temperature
        if temp <= 0.0:
            raise RldpDensityConfigError(f"temp must be > 0, got {temp}")
        goal_kernel = bool(config.get("goal_kernel", False))
        train_b = bool(config.get("rldp_density_train_b", False))
        task_index = str(config.get("rldp_density_task_index", "feature"))
        if task_index not in ("feature", "regression"):
            raise RldpDensityConfigError(
                f"rldp_density_task_index must be 'feature' or 'regression', got "
                f"{task_index!r}")
        regression = task_index == "regression"
        if train_b and bool(config.get("rldp_two_phase", False)):
            raise RldpDensityConfigError(
                "rldp_density_train_b=true contradicts rldp_two_phase: the official phase 2 "
                "FREEZES the backward group (no optimizer step), so the CE gradient into B "
                "would be computed and dropped. Pick one.")
        if regression and goal_kernel:
            raise RldpDensityConfigError(
                "rldp_density_task_index='regression' and goal_kernel=true are two task "
                "interfaces for one critic (kernel reward at goal embeddings vs the "
                "regressed index); pick one.")
        blocks = ()
        if goal_kernel or regression:
            # The kernel is DEFINED BY COLUMN SLICES of the observation, so the dataset
            # must actually carry those columns — refuse at build rather than let jax
            # slice short rows into a silently different reward.
            blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
            need = max(b for (_a, b, _e) in blocks)
            obs_dim = int(ex_obs.shape[-1])
            if obs_dim < need:
                raise RldpDensityConfigError(
                    f"{'goal_kernel=true' if goal_kernel else 'rldp_density_task_index=regression'}"
                    f" needs observations wide enough for every measure_hit_spec block: the "
                    f"spec reaches column {need} but the observations are only {obs_dim} "
                    f"wide. The object-coordinate kernel reward is a literal column slice; "
                    f"on this dataset it is undefined.")
        critic, params, targets = super().build(ex_obs, ex_actions, config)
        at = config.get("actor_temp", None)
        extra = {}
        if regression:
            z_lam = float(config.get("rldp_density_z_ridge_lambda_mult", 0.01))
            z_floor = float(config.get("rldp_density_z_cov_eig_floor", 1e-6))
            z_eps = float(config.get("rldp_density_z_const_eps", 1e-6))
            z_enc_tau = float(config.get("rldp_density_z_enc_tau", 0.0005))
            z_sigma_tau = float(config.get("rldp_density_z_sigma_tau", 0.005))
            z_pool_m = int(config.get("rldp_density_z_eval_pool_m", 256))
            if not (z_lam > 0.0 and z_floor > 0.0 and z_eps > 0.0):
                raise RldpDensityConfigError(
                    "rldp_density_z_ridge_lambda_mult, rldp_density_z_cov_eig_floor and "
                    "rldp_density_z_const_eps must all be > 0 (density_fb's contract: an "
                    "unregularized/unfloored solve, or a constant reward encoded as noise)")
            if not (0.0 < z_enc_tau <= 0.1 and 0.0 < z_sigma_tau <= 1.0):
                raise RldpDensityConfigError(
                    "rldp_density_z_enc_tau must be in (0, 0.1] and rldp_density_z_sigma_tau "
                    f"in (0, 1]; got {z_enc_tau}, {z_sigma_tau}")
            bs = int(config.get("batch_size", z_pool_m))
            if not (1 < z_pool_m <= bs):
                raise RldpDensityConfigError(
                    f"rldp_density_z_eval_pool_m={z_pool_m} must be > 1 and <= batch_size="
                    f"{bs}: the frozen pool is drawn from one batch's next states and the "
                    f"centered moment is identically zero at M=1")
            import copy
            import jax.random as jr
            # The slow tower starts as a COPY of B; the moments at mu=0, C=I (well-posed
            # first solve); the pool holds FULL observation rows (`_B` slices them itself
            # under backward_input='hit_spec', `_delta_phi` reads the raw block columns).
            params["backward_zenc"] = copy.deepcopy(params["backward"])
            zstats_mod = _ZStatsState(dim=int(critic.z_dim), pool_m=z_pool_m,
                                      d_c=int(ex_obs.shape[-1]),
                                      dtype=jnp.zeros((), ex_obs.dtype).dtype)
            params["rldp_zstats"] = zstats_mod.init(jr.PRNGKey(int(config["seed"]) + 7000))["params"]
            extra = dict(z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                         z_enc_tau=z_enc_tau, z_sigma_tau=z_sigma_tau, z_eval_pool_m=z_pool_m,
                         backward_zenc=critic.backward, rldp_zstats=zstats_mod,
                         GROUPS=tuple(critic.GROUPS) + _REG_STATE_GROUPS)
        critic = dataclasses.replace(
            critic, temp=temp,
            actor_temp=(None if at is None else float(at)),
            actor_q_center=bool(config.get("actor_q_center", True)),
            goal_kernel=goal_kernel, train_b=train_b, task_index=task_index,
            loss_kind=str(config.get("rldp_density_loss", "ce")),
            # goal_kernel / regression and the parent's backward_input="hit_spec" parse
            # the SAME measure_hit_spec key, so the values agree when both are on; when
            # neither lever is on, KEEP the parent's parse — overwriting with () here
            # would sever the hit_spec input map (the 2026-08-25 near-miss).
            blocks=(blocks if (goal_kernel or regression) else critic.blocks),
            **extra,
        )
        return critic, params, targets
