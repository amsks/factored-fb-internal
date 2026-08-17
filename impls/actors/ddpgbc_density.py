"""DDPGBCDensityActor — the bridge's `actor_kind='ddpgbc'` extraction for density_fb.

Ported from `tools/wandb_mode_shim/gciql_flow.py` @ `origin/gciql-fb-bridge` 59f5072
(bridge spec 22): the `ddpgbc` branch of `actor_loss` (~1305) and the matching bootstrap
branch of `density_loss` (~2518). It is the only bridge mechanism that closes an
antmaze-sized gap — hard-ddpgbc 0.920/0.868/0.844 against hard-flow 0.772/0.580/0.656 — and
it is mechanistically the interesting arm for the mw-pick-place collapse: R1 blames a
feedback loop between a flow actor ascending a spiky kernel readout and a critic target that
bootstraps at that actor's actions, and this swaps BOTH ends of it for the stock gaussian
DDPG+BC policy.

Two things separate this from the stock `DDPGBCActor` it subclasses, and they are exactly
the two the bridge branch spells out:

1. **Centered q normalizer.** The bridge hardcodes `q_scale = |q - q.mean()|.mean()` in this
   branch ("actor_q_center, lattice-required"). Stock DDPG+BC divides by `|q|.mean()`. Here
   the normalizer is taken from the critic's SECOND `q_fn` return, which is where
   `density_fb.q_fn` puts the centered quantity when `actor_q_center: true` — the same seam
   FlowBCActor normalizes through, so the two actor kinds share one definition of centering
   instead of each hardcoding one.
2. **Bootstrap at the VALUE goal.** `density_fb` requires `boot["next_action_vgoal"]`: its
   forward tower takes the goal as an input, so its row is g-indexed and the TD target must
   bootstrap at that same g (see the block comment in `flowbc.py::sample_bootstrap`). Stock
   DDPG+BC emits only `next_action`, drawn at the ACTOR goal, which `density_fb` explicitly
   rejects. This adds the value-goal draw at frozen params.

Everything else — the gaussian actor, `alpha`-weighted log-prob BC term, `dist.mode()` under
`const_std` — is the stock branch, inherited unchanged. `DDPGBCActor` itself is untouched, so
the existing gciql x ddpgbc cells stay bit-identical (bridge knob discipline: new branch,
default-off).
"""
import dataclasses

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import DDPGBCActor, _cvec


@dataclasses.dataclass(frozen=True)
class DDPGBCDensityActor(DDPGBCActor):
    GROUPS = ("actor",)
    NEEDS_V = False
    REQUIRES_COND = frozenset()

    def _q_actions(self, dist, boot):
        """The action the q-term is taken at: `dist.mode()` under `const_std`, else a sample.

        The bridge draws that sample off the same rng the flow branch would have spent on
        noise. Our seam gives `loss` no rng, so the key is stashed in `boot` by
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
        dist = self._dist(params["actor"], obs, cvec)
        q_actions = self._q_actions(dist, boot)
        q, qs = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        # `qs` is the critic's normalizer leg: centered under density_fb's actor_q_center,
        # which is the bridge's `|q - q.mean()|.mean()`. Only the DENOMINATOR is centered --
        # centering the numerator would make -q.mean() the batch mean of a batch-centered
        # quantity, identically ~0, i.e. a pure-BC actor (the failed first anchor generation;
        # see density_fb.py::q_fn).
        q_scale = jnp.abs(qs).mean()
        q_loss = -q.mean() / jax.lax.stop_gradient(q_scale + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {
            "actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
            "q_mean": q.mean(), "q_scale": q_scale, "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - action) ** 2),
            # R1 instrumentation, mirroring FlowBCActor._policy_drift so the two actor kinds
            # are comparable on the collapse diagnostics.
            "action_gap_l2": jnp.linalg.norm(q_actions - action, axis=-1).mean(),
        }

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        r_boot, r_sample = jax.random.split(rng)
        dist = self._dist(p, next_obs, _cvec(cond))
        boot = {"next_action": jnp.clip(dist.mode() if self.const_std
                                        else dist.sample(seed=r_boot), -1.0, 1.0),
                "actor_sample_rng": r_sample}
        if "value_goal" in cond:
            # a+ ~ pi(.|s', g_value) at FROZEN params. density_fb's forward tower is
            # F(s, a, g), so the target must bootstrap at the goal its row is indexed by;
            # `next_action` above is at the ACTOR goal and density_fb refuses it outright.
            dv = self._dist(p, next_obs, cond["value_goal"])
            boot["next_action_vgoal"] = jnp.clip(
                dv.mode() if self.const_std else dv.sample(seed=r_boot), -1.0, 1.0)
        return boot
