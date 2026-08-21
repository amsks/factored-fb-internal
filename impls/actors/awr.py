"""AWR actor — transcribed from OGBench gciql.py:actor_loss (awr branch).

Advantage baseline: when the critic provides a value function (PROVIDES_V), use it exactly
as vendored (A = Q(s,a) - V(s)). For value-less critics (FB, SF-feature, centered-bilinear)
fall back to a POLICY baseline A = Q(s,a) - Q(s, pi(s)), so AWR composes with every critic.
The advantage is a stop-gradient weight either way, so the V-path is byte-identical."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec


@dataclasses.dataclass(frozen=True)
class AWRActor:
    actor: Any
    alpha: float

    GROUPS = ("actor",)
    NEEDS_V = False   # value-less critics use a policy baseline (see module docstring)
    REQUIRES_COND = frozenset()  # _cvec accepts either cond["z"] or cond["goal"]

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        q, _ = critic.q_fn(params, obs, cond, action, stop_encoder_grad=True)
        if getattr(critic, "PROVIDES_V", False):
            v = critic.v_fn(params, obs, cond)
        else:
            # No V: baseline with Q at the policy's mode action (stop-grad below keeps the
            # advantage a pure weight, so pi(s) here does not leak gradient into the actor).
            a_pi = self._dist(params["actor"], obs, cvec).mode()
            v, _ = critic.q_fn(params, obs, cond, jnp.clip(a_pi, -1.0, 1.0), stop_encoder_grad=True)
        adv = jax.lax.stop_gradient(q - v)
        exp_a = jnp.minimum(jnp.exp(adv * self.alpha), 100.0)
        dist = self._dist(params["actor"], obs, cvec)
        log_prob = dist.log_prob(action)
        loss = -(exp_a * log_prob).mean()
        return loss, {"actor_loss": loss, "adv": adv.mean(), "bc_log_prob": log_prob.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        boot = {"next_action": jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1)}
        if "value_goal" in cond:
            # See the block comment in impls/actors/ddpgbc.py::sample_bootstrap: a+ drawn at
            # the VALUE goal, for critics declaring BOOT_COND == "value_goal". Separate key,
            # deterministic draw, no randomness spent, next_action untouched.
            boot["next_action_vgoal"] = jnp.clip(
                self._dist(p, next_obs, cond["value_goal"]).mode(), -1, 1)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        dist = self._dist(net["actor"].params, obs, _cvec(cond), temperature)
        return jnp.clip(dist.mode(), -1, 1)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import Actor
        a_cfg = config["actor"]
        # tanh_squash=False matches vendored GCIQL's GCActor (gciql.py:248).
        actor = Actor(hidden_dims=tuple(a_cfg["hidden_dims"]), action_dim=ex_actions.shape[-1],
                      layer_norm=a_cfg.get("layer_norm", False), tanh_squash=False,
                      const_std=a_cfg.get("const_std", True))
        ex_in = jnp.zeros((ex_obs.shape[0], ex_obs.shape[-1] + config.get("cvec_dim", config.get("z_dim"))))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 3000), ex_in)["params"]}
        return cls(actor=actor, alpha=float(a_cfg["alpha"])), params
