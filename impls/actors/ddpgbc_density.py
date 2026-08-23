"""DDPGBCDensityActor — the bridge's `actor_kind='ddpgbc'` extraction for density_fb.

Ported from the bridge's `tools/wandb_mode_shim/gciql_flow.py` (bridge spec 22). Measured
hard-ddpgbc 0.920/0.868/0.844 against hard-flow 0.772/0.580/0.656.

Two things separate it from the stock `DDPGBCActor` it subclasses:

1. Centered q normalizer. The bridge hardcodes `q_scale = |q - q.mean()|.mean()` here where
   stock DDPG+BC divides by `|q|.mean()`. The normalizer is taken from the critic's SECOND
   `q_fn` return — the same seam FlowBCActor normalizes through — so both actor kinds share
   one definition of centering instead of each hardcoding one.
2. Bootstrap at the VALUE goal. `density_fb` requires `boot["next_action_vgoal"]`: its
   forward tower takes the goal as an input, so its row is g-indexed and the TD target must
   bootstrap at that same g. Stock DDPG+BC emits only `next_action`, at the ACTOR goal,
   which `density_fb` rejects.

Everything else is the stock branch, inherited unchanged; `DDPGBCActor` is untouched, so
existing gciql x ddpgbc cells stay bit-identical.
"""
import dataclasses

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import DDPGBCActor, _cvec, _obs_feat, _nobs_feat


@dataclasses.dataclass(frozen=True)
class DDPGBCDensityActor(DDPGBCActor):
    GROUPS = ("actor",)
    NEEDS_V = False
    REQUIRES_COND = frozenset()

    def _q_actions(self, dist, boot):
        """The action the q-term is taken at: `dist.mode()` under `const_std`, else a sample.

        `loss` gets no rng from the seam, so the key is stashed in `boot` by
        `sample_bootstrap` — still one draw, still no extra split.
        """
        if self.const_std:
            return jnp.clip(dist.mode(), -1.0, 1.0)
        rng = boot.get("actor_sample_rng")
        if rng is None:
            raise KeyError(
                "const_std=false needs boot['actor_sample_rng'], which this actor's "
                "sample_bootstrap emits; a critic that rebuilds `boot` by hand must carry it.")
        return jnp.clip(dist.sample(seed=rng), -1.0, 1.0)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        # Policy on FEATURES, critic on the raw observation; identity on state.
        dist = self._dist(params["actor"], _obs_feat(cond, obs), cvec)
        q_actions = self._q_actions(dist, boot)
        q, qs = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        # `qs` is the critic's normalizer leg, centered under density_fb's actor_q_center.
        # Only the DENOMINATOR is centered: centering the numerator would make -q.mean() the
        # batch mean of a batch-centered quantity, identically ~0, i.e. a pure-BC actor.
        q_scale = jnp.abs(qs).mean()
        q_loss = -q.mean() / jax.lax.stop_gradient(q_scale + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {
            "actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
            "q_mean": q.mean(), "q_scale": q_scale, "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - action) ** 2),
            # Mirrors FlowBCActor._policy_drift, so the actor kinds stay comparable.
            "action_gap_l2": jnp.linalg.norm(q_actions - action, axis=-1).mean(),
        }

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = _nobs_feat(cond, batch["next_observations"])
        p = net["actor"].params
        r_boot, r_sample = jax.random.split(rng)
        dist = self._dist(p, next_obs, _cvec(cond))
        boot = {"next_action": jnp.clip(dist.mode() if self.const_std
                                        else dist.sample(seed=r_boot), -1.0, 1.0),
                "actor_sample_rng": r_sample}
        if "_cvec_vgoal" in cond:
            # a+ ~ pi(.|s', g_value) at FROZEN params: density_fb's forward tower F(s,a,g)
            # makes its row g-indexed, and it refuses the ACTOR-goal `next_action` above.
            # The vector comes from the CRITIC as cond["_cvec_vgoal"], NOT cond["value_goal"]
            # (impls/actors/ddpgbc.py). On the goal-conditioned state arm the two keys hold
            # the same array, so reading the wrong one fails SILENTLY there.
            dv = self._dist(p, next_obs, cond["_cvec_vgoal"])
            boot["next_action_vgoal"] = jnp.clip(
                dv.mode() if self.const_std else dv.sample(seed=r_boot), -1.0, 1.0)
        return boot
