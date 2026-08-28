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
  * sample_cond's mixed-z draw, map_eval_cond (z = project_z(B(g))), infer_cond
    (z ∝ E[r B]), the target-network schedule, GROUPS / STAGES / _LR_KEY rows.

What must ALSO change once the score is a logit (design note sec. 4): F^T z is no longer
a value. `q_fn` becomes the reward integration

    q_h(s, a; z) = sum_j p_h(j | s, a, z) * ( B(u_j)^T z ),        u_j = batch next states,

r_z(u) = B(u)^T z being exactly the reward convention infer_cond inverts. The candidate
pool cannot reach q_fn through its signature, so `sample_cond` stashes it on cond
("_pool_B"), the same seam workaround density_fb.py uses ("_density_u_raw"/"_density_u_sc").
Ensemble reduction happens AFTER integration, on scalars (mean over heads, matching the
parent's q_fn reduction); the per-head target distributions are NEVER collapsed by
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
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics._measure_common import parse_hit_spec
from impls.critics.density_fb import _delta_phi, _raw_block_cols
from impls.critics.rldp import RLDPCritic
from impls.utils.fb_math import project_z, targets_uncertainty


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
        z, next_action = cond["z"], boot["next_action"]
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()
        d = self._row_discount(batch, B)

        # --- density successor term (replaces fb_off + fb_diag) ---
        # Online logits. Divergence 1 is PRESERVED: B(goal) detached, the CE trains
        # forward + left_encoder only.
        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)                     # [P, B, z]
        Bg = jax.lax.stop_gradient(self._B(params["backward"], goal))     # [B, z]
        logits = jnp.einsum("pbz,cz->pbc", Fs, Bg) / self.temp            # [P, B, B]
        logp = jax.nn.log_softmax(logits, -1)

        # Target logits: full target nets, per-head (HEADWISE targets — no
        # mean-minus-uncertainty on distributions; see module docstring).
        nle = self._le(target["left_encoder"], next_obs)
        tFs = self._F(target["forward"], nle, z, next_action)
        tB = self._B(target["backward"], goal)
        w_bar = jax.nn.softmax(jnp.einsum("pbz,cz->pbc", tFs, tB) / self.temp, -1)

        k = jnp.eye(B)
        t = jax.lax.stop_gradient((1.0 - d) * k + d * w_bar)               # [P, B, B]
        ce = -(t * logp).sum(-1).mean()

        # --- SP head + orthonormality, verbatim from the parent ---
        sp, B_pred = self._sp_loss(params, target, batch)
        cov = B_pred @ B_pred.T
        orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        orth_diag = -jnp.mean(jnp.diagonal(cov))
        orth = orth_off + orth_diag

        loss = ce + sp + self.ortho_coef * orth

        w = jnp.exp(logp)
        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [P, B]
        return loss, {
            "fb_loss": loss, "sp_loss": sp, "ortho_loss": orth,
            "density_ce": ce,
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
        else:
            kg = jax.lax.stop_gradient(jnp.einsum("cz,bz->bc", pool, z))     # [B, C]
        Qs = (w * kg[None]).sum(-1)                                          # [P, B]
        qmean, _ = targets_uncertainty(Qs, self.num_parallel)
        if self.actor_q_center:
            # Centered magnitude for the FlowBC/FQL |Qs|.mean() normalizer; the FIRST
            # return stays uncentered (see density_fb.py::q_fn on why).
            return qmean, (qmean - qmean.mean())[None]
        return qmean, Qs

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
        blocks = ()
        if goal_kernel:
            # Exp A lattice: the kernel is DEFINED BY COLUMN SLICES of the observation, so
            # the dataset must actually carry those columns — refuse at build rather than
            # let jax slice short rows into a silently different reward.
            blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
            need = max(b for (_a, b, _e) in blocks)
            obs_dim = int(ex_obs.shape[-1])
            if obs_dim < need:
                raise RldpDensityConfigError(
                    f"goal_kernel=true needs observations wide enough for every "
                    f"measure_hit_spec block: the spec reaches column {need} but the "
                    f"observations are only {obs_dim} wide. The object-coordinate kernel "
                    f"reward is a literal column slice; on this dataset it is undefined.")
        critic, params, targets = super().build(ex_obs, ex_actions, config)
        at = config.get("actor_temp", None)
        critic = dataclasses.replace(
            critic, temp=temp,
            actor_temp=(None if at is None else float(at)),
            actor_q_center=bool(config.get("actor_q_center", True)),
            goal_kernel=goal_kernel,
            # Both goal_kernel and the parent's backward_input="hit_spec" parse the SAME
            # measure_hit_spec key, so the values agree when both are on; when goal_kernel
            # is off, KEEP the parent's parse — overwriting with () here would sever the
            # hit_spec input map (the 2026-08-25 near-miss).
            blocks=(blocks if goal_kernel else critic.blocks),
        )
        return critic, params, targets
