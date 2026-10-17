"""Deterministic TD3-style actor.

Maximizes Q(s, pi(s, c), c), optionally scaled by the batch |Q|. Clipped Gaussian noise
is added to the actor-loss action and to the bootstrap action (target policy smoothing).
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.base import cond_vector


def _smooth(a, rng, std, clip):
    """Add clipped Gaussian noise and clip to [-1, 1]."""
    eps = jnp.clip(std * jax.random.normal(rng, a.shape), -clip, clip)
    return jnp.clip(a + eps, -1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DDPGBCActor:
    actor: Any
    boot_noise_std: float = 0.2
    boot_noise_clip: float = 0.3
    q_noise_std: float = 0.2
    q_noise_clip: float = 0.3
    # Divide the Q term by sg(mean |Q|)
    q_normalize: bool = True

    GROUPS = ("actor",)
    REQUIRES_COND = frozenset()

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs = batch["observations"]
        cvec = critic.cond_vec(cond)
        dist = self._dist(params["actor"], obs, cvec)
        q_actions = _smooth(jnp.clip(dist.mode(), -1, 1), boot["actor_q_rng"],
                            self.q_noise_std, self.q_noise_clip)
        q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_loss = -q.mean()
        if self.q_normalize:
            q_loss = q_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        return q_loss, {"actor_loss": q_loss, "q_loss": q_loss, "q_mean": q.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        boot = {"next_action": _smooth(
            jnp.clip(self._dist(p, next_obs, cond_vector(cond)).mode(), -1, 1),
            rng, self.boot_noise_std, self.boot_noise_clip),
                "actor_q_rng": jax.random.fold_in(rng, 7)}
        # Bootstrap action at the TD-target goal, if the critic uses a separate one
        if "_cvec_vgoal" in cond:
            boot["next_action_vgoal"] = _smooth(
                jnp.clip(self._dist(p, next_obs, cond["_cvec_vgoal"]).mode(), -1, 1),
                rng, self.boot_noise_std, self.boot_noise_clip)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        dist = self._dist(net["actor"].params, obs, cond_vector(cond), temperature)
        return jnp.clip(dist.mode(), -1, 1)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.actors._motivo_nets import MotivoPolicy
        a_cfg = config["actor"]
        odim = int(ex_obs.shape[-1])
        # hidden_dims gives width and number of trunk layers; entries must be equal.
        hd = tuple(a_cfg["hidden_dims"])
        if len(set(hd)) != 1:
            raise ValueError(f"actor.hidden_dims is read as (width, n_layers) and needs "
                             f"every entry equal; got {list(hd)}.")
        actor = MotivoPolicy(obs_dim=odim, action_dim=ex_actions.shape[-1],
                             hidden_dim=hd[0], trunk_layers=len(hd), embed_layers=2)
        ex_in = jnp.zeros((ex_obs.shape[0], odim + config["cvec_dim"]))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 2000), ex_in)["params"]}
        return cls(actor=actor,
                   boot_noise_std=float(a_cfg["boot_noise_std"]),
                   boot_noise_clip=float(a_cfg["boot_noise_clip"]),
                   q_noise_std=float(a_cfg["q_noise_std"]),
                   q_noise_clip=float(a_cfg["q_noise_clip"]),
                   q_normalize=bool(a_cfg["q_normalize"])), params
