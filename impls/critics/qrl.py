"""QRLCritic — Quasimetric RL critic on the factored seam.

Learns a goal-conditioned quasimetric distance d(s, g) (IQE or MRN) whose negative is the
value: V(s, g) = -d(s, g). The Q for an action goes through a learned LATENT DYNAMICS model:

    Q(s, a, g) = - d( phi(s) + f(phi(s), a),  phi(g) )

i.e. how close the predicted next latent lands to the goal latent. Value loss is the QRL dual
objective (push negatives far via a softplus, keep one-step positives <= 1 via a Lagrangian
`lam`) plus a latent one-step dynamics loss. Math transcribed from OGBench qrl.py.

PROVIDES_V. No target networks. Uses GCIQEValue/GCMRNValue/LogParam (copied into
impls.utils.networks so impls stays self-contained JAX).
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCIQEValue, GCMRNValue, LogParam, MLP


@dataclasses.dataclass(frozen=True)
class QRLCritic:
    value: Any
    lam: Any
    dynamics: Any
    eps: float
    obs_dim: int

    GROUPS = ("value", "lam", "dynamics")
    STAGES = ("qrl",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    @property
    def TARGETS(self):
        return {}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "qrl"
        return self.GROUPS

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "qrl"
        obs, nobs, act = batch["observations"], batch["next_observations"], batch["actions"]
        vg = cond["value_goal"]
        # QRL dual value loss: negatives (random goals) pushed far; one-step positives kept <=1.
        d_neg = self.value.apply({"params": params["value"]}, obs, vg)
        d_pos = self.value.apply({"params": params["value"]}, obs, nobs)
        lam = self.lam.apply({"params": params["lam"]})
        d_neg_loss = (100.0 * jax.nn.softplus(5.0 - d_neg / 100.0)).mean()
        d_pos_loss = (jax.nn.relu(d_pos - 1.0) ** 2).mean()
        value_loss = d_neg_loss + d_pos_loss * jax.lax.stop_gradient(lam)
        lam_loss = lam * (self.eps - jax.lax.stop_gradient(d_pos_loss))
        # latent one-step dynamics: predicted next latent should sit at the true next latent.
        _, ob_reps, next_ob_reps = self.value.apply({"params": params["value"]}, obs, nobs, info=True)
        pred = ob_reps + self.dynamics.apply(
            {"params": params["dynamics"]}, jnp.concatenate([ob_reps, act], -1))
        dist1 = self.value.apply({"params": params["value"]}, next_ob_reps, pred, is_phi=True)
        dist2 = self.value.apply({"params": params["value"]}, pred, next_ob_reps, is_phi=True)
        dyn_loss = (dist1 + dist2).mean() / 2.0
        loss = value_loss + lam_loss + dyn_loss
        return loss, {"value_loss": value_loss, "lam_loss": lam_loss, "dynamics_loss": dyn_loss,
                      "lam": lam, "d_neg_mean": d_neg.mean(), "d_pos_mean": d_pos.mean()}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        _, ob_reps, goal_reps = self.value.apply({"params": params["value"]}, obs, cond["goal"], info=True)
        pred = ob_reps + self.dynamics.apply(
            {"params": params["dynamics"]}, jnp.concatenate([ob_reps, actions], -1))
        q = -self.value.apply({"params": params["value"]}, pred, goal_reps, is_phi=True)
        return q, q[None]

    def v_fn(self, params, obs, cond):
        return -self.value.apply({"params": params["value"]}, obs, cond["goal"])

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
        hd = tuple(config["value_hidden_dims"])
        d = int(config["latent_dim"])
        ln = bool(config["layer_norm"])
        if config.get("quasimetric_type", "iqe") == "mrn":
            value = GCMRNValue(hidden_dims=hd, latent_dim=d, layer_norm=ln)
        else:
            value = GCIQEValue(hidden_dims=hd, latent_dim=d,
                               dim_per_component=int(config["dim_per_component"]), layer_norm=ln)
        dynamics = MLP((*hd, d), activate_final=False, layer_norm=ln)
        lam = LogParam()
        ex_goal = jnp.zeros_like(ex_obs)
        ex_lat = jnp.zeros((ex_obs.shape[0], d))
        rv, rd = jr.split(jr.PRNGKey(config["seed"] + 9000), 2)
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "lam": lam.init(jr.PRNGKey(config["seed"] + 9001))["params"],
            "dynamics": dynamics.init(rd, jnp.concatenate([ex_lat, ex_actions], -1))["params"],
        }
        obj = cls(value=value, lam=lam, dynamics=dynamics, eps=float(config["eps"]),
                  obs_dim=int(ex_obs.shape[-1]))
        return obj, params, {}
