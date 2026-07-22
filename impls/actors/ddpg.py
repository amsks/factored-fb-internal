"""Plain DDPG actor — deterministic policy gradient, NO behavior-cloning term.

Identical to DDPGBCActor except the loss is just -Q(s, pi(s)) (normalized). This is the
"no BC" control: it isolates what the critic's value signal alone buys, with nothing
anchoring the policy to the data. On near-expert offline data it is expected to be unstable
(the whole reason DDPG+BC exists) -- that is the point of having it in the grid.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec


@dataclasses.dataclass(frozen=True)
class DDPGActor:
    actor: Any
    const_std: bool

    GROUPS = ("actor",)
    NEEDS_V = False
    REQUIRES_COND = frozenset()   # conditions on z or goal via _cvec

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs = batch["observations"]
        cvec = critic.cond_vec(cond)
        dist = self._dist(params["actor"], obs, cvec)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        return q_loss, {"actor_loss": q_loss, "q_loss": q_loss, "q_mean": q.mean()}

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
        actor = Actor(hidden_dims=tuple(a_cfg["hidden_dims"]), action_dim=ex_actions.shape[-1],
                      layer_norm=a_cfg.get("layer_norm", False), tanh_squash=False,
                      const_std=a_cfg.get("const_std", True))
        ex_in = jnp.zeros((ex_obs.shape[0], ex_obs.shape[-1] + config.get("cvec_dim", config.get("z_dim"))))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 2000), ex_in)["params"]}
        return cls(actor=actor, const_std=bool(a_cfg.get("const_std", True))), params
