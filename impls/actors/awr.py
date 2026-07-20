"""AWR actor — transcribed from OGBench gciql.py:actor_loss (awr branch). NEEDS_V=True,
so it pairs only with a V-providing critic (e.g. GCIQL, PR3); awr×fb raises ComboError."""
import dataclasses
from typing import Any

import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec


@dataclasses.dataclass(frozen=True)
class AWRActor:
    actor: Any
    alpha: float

    GROUPS = ("actor",)
    NEEDS_V = True

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        v = critic.v_fn(params, obs, cond)
        q, _ = critic.q_fn(params, obs, cond, action, stop_encoder_grad=True)
        adv = q - v
        exp_a = jnp.minimum(jnp.exp(adv * self.alpha), 100.0)
        dist = self._dist(params["actor"], obs, cvec)
        log_prob = dist.log_prob(action)
        loss = -(exp_a * log_prob).mean()
        return loss, {"actor_loss": loss, "adv": adv.mean(), "bc_log_prob": log_prob.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        dist = self._dist(net["actor"].params, batch["next_observations"], _cvec(cond))
        return {"next_action": jnp.clip(dist.mode(), -1, 1)}

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
