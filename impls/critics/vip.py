"""VIPCritic — state-based Value-Implicit Pre-training (Ma et al., arXiv:2210.00030).

Hard-codes V(s; g) = -||phi(s) - phi(g)||_2 (raw L2 — no normalization, no temperature)
and trains the encoder phi with VIP's Fenchel-dual objective (their Eq. 6):

    L = (1-gamma) E[ ||phi(o0) - phi(g)|| ]
      + log E[ exp( ||phi(o_k) - phi(g)|| - r + gamma ||phi(o_{k+1}) - phi(g)|| * (-1) ) ]

transcribed from facebookresearch/vip vip/trainer.py (log-mean-exp over the batch, sparse
reward r = -1 until the sub-trajectory's final frame). Needs VIPDataset batches
(initial_observations / value_goals = trajectory-final frames).

Representation-only: pair with bcflow. V is exposed; q_fn ignores actions (logged only).
No target networks.
"""
import dataclasses
from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from impls.utils.networks import MLP


class VIPEncoder(nn.Module):
    hidden_dims: tuple
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs):
        return MLP((*self.hidden_dims, self.latent_dim),
                   activate_final=False, layer_norm=self.layer_norm)(obs)


def _dist(a, b):
    return jnp.sqrt(((a - b) ** 2).sum(-1) + 1e-12)


@dataclasses.dataclass(frozen=True)
class VIPCritic:
    value: Any
    discount: float
    obs_dim: int

    GROUPS = ("value",)
    STAGES = ("vip",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    @property
    def TARGETS(self):
        return {}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "vip"
        return self.GROUPS

    def _phi(self, p, obs):
        return self.value.apply({"params": p}, obs)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "vip"
        p = params["value"]
        g = self._phi(p, cond["value_goal"])
        d0 = _dist(self._phi(p, batch["initial_observations"]), g)
        dk = _dist(self._phi(p, batch["observations"]), g)
        dn = _dist(self._phi(p, batch["next_observations"]), g)

        attract = (1.0 - self.discount) * d0.mean()
        # exp(||phi(o_k)-phi(g)|| - r - gamma*||phi(o_k+1)-phi(g)||), log-mean-exp, stabilized
        arg = dk - batch["rewards"] - self.discount * dn
        m = jnp.max(arg)
        repulse = jnp.log(jnp.mean(jnp.exp(arg - m)) + 1e-6) + m
        loss = attract + repulse
        return loss, {"vip_loss": loss, "attract": attract, "repulse": repulse,
                      "d_goal_mean": dk.mean()}

    def v_fn(self, params, obs, cond):
        p = params["value"]
        return -_dist(self._phi(p, obs), self._phi(p, cond["goal"]))

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # No action model: Q := V(s,g). Logged only under bcflow (q_coeff=0).
        v = self.v_fn(params, obs, cond)
        return v, jnp.stack([v, v], 0)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        enc = VIPEncoder(hidden_dims=tuple(config["value_hidden_dims"]),
                         latent_dim=int(config["latent_dim"]),
                         layer_norm=bool(config["layer_norm"]))
        params = {"value": enc.init(jr.PRNGKey(config["seed"] + 9000), ex_obs)["params"]}
        obj = cls(value=enc, discount=float(config["discount"]), obs_dim=int(ex_obs.shape[-1]))
        return obj, params, {}
