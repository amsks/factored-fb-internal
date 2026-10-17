"""One-step policy distilled from a flow-matching behavior policy (FQL-style).

The policy maximizes Q / mean |Q| plus bc_coeff times the squared distance to the flow's
action for the same noise.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.base import cond_vector


@dataclasses.dataclass(frozen=True)
class FlowBCActor:
    actor: Any
    actor_vf: Any
    flow_steps: int
    bc_coeff: float

    GROUPS = ("actor", "actor_vf")
    REQUIRES_COND = frozenset()

    def _actor(self, p, obs, z, noise):
        return self.actor.apply({"params": p}, obs, z, noise)

    def _vf(self, p, obs, action, t):
        return self.actor_vf.apply({"params": p}, obs, action, t)

    def _rollout(self, vf_p, obs, noise):
        """Euler-integrate the flow from noise to an action."""
        a = noise
        for i in range(self.flow_steps):
            ti = jnp.full((obs.shape[0], 1), i / self.flow_steps)
            a = a + self._vf(vf_p, obs, a, ti) / self.flow_steps
        return jnp.clip(a, -1.0, 1.0)

    @staticmethod
    def _policy_drift(params, obs, cond, critic, a_actor, a_data, q_actor):
        """Logging: action gap to the data and Q(s, a_data) - Q(s, a_actor)."""
        q_data, _ = critic.q_fn(params, obs, cond, a_data, stop_encoder_grad=True)
        return jax.lax.stop_gradient({
            "action_gap_l2": jnp.linalg.norm(a_actor - a_data, axis=-1).mean(),
            "q_gap_data_minus_actor": q_data.mean() - q_actor.mean(),
        })

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        c = critic.cond_vec(cond)
        x0, t, noise = boot["flow_x0"], boot["flow_t"], boot["actor_noise"]

        # Flow matching
        x1 = action
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        pred = self._vf(params["actor_vf"], obs, xt, t)
        bc_flow = jnp.mean((pred - vel) ** 2)

        # One-step policy
        a = self._actor(params["actor"], obs, c, noise)
        Q, Qs = critic.q_fn(params, obs, cond, a, stop_encoder_grad=True)
        loss = -Q.mean()
        info = {"q": Q.mean(), "bc_flow_loss": bc_flow}
        info.update(self._policy_drift(params, obs, cond, critic, a, action, Q))
        tgt = jax.lax.stop_gradient(self._rollout(params["actor_vf"], obs, noise))
        distill = jnp.mean((a - tgt) ** 2)
        q_scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(Qs).mean(), 1e-6))
        loss = loss / q_scale + self.bc_coeff * distill
        info["bc_error"] = distill
        loss = loss + bc_flow
        info["actor_loss"] = loss
        return loss, info

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        adim = batch["actions"].shape[-1]
        c = cond_vector(cond)
        r_na, r_x0, r_t, r_noise = jax.random.split(rng, 4)
        na_noise = jax.random.normal(r_na, (B, adim))
        na = self._actor(net["actor"].params, next_obs, c, na_noise)
        boot = {"next_action": na,
                "flow_x0": jax.random.normal(r_x0, (B, adim)),
                "flow_t": jax.random.uniform(r_t, (B, 1)),
                "actor_noise": jax.random.normal(r_noise, (B, adim))}
        # Bootstrap action at the TD-target goal, if the critic uses a separate one
        if "_cvec_vgoal" in cond:
            boot["next_action_vgoal"] = self._actor(
                net["actor"].params, next_obs, cond["_cvec_vgoal"], na_noise)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        c = cond_vector(cond)
        noise = jax.random.normal(rng, (*obs.shape[:-1], self.actor.action_dim))
        a = self._actor(net["actor"].params, obs, c, noise)
        return jnp.clip(a, -1.0, 1.0)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import NoiseConditionedActor, FlowVectorField
        adim = ex_actions.shape[-1]
        cdim = config["cvec_dim"]
        a_cfg = config["actor"]
        actor = NoiseConditionedActor(action_dim=adim, hidden_dim=a_cfg["hidden_dim"],
                                      hidden_layers=a_cfg["hidden_layers"],
                                      embedding_layers=a_cfg["embedding_layers"])
        vf = FlowVectorField(action_dim=adim, hidden_dim=a_cfg["flow_vf_hidden_dim"],
                             hidden_layers=a_cfg["flow_vf_hidden_layers"])
        ract, rvf = jr.split(jr.PRNGKey(config["seed"] + 1000), 2)
        ex_z = jnp.zeros((ex_obs.shape[0], cdim))
        params = {
            "actor": actor.init(ract, ex_obs, ex_z, ex_actions)["params"],
            "actor_vf": vf.init(rvf, ex_obs, ex_actions, ex_actions[..., :1])["params"],
        }
        obj = cls(actor=actor, actor_vf=vf, flow_steps=int(a_cfg["flow_steps"]),
                  bc_coeff=float(a_cfg["bc_coeff"]))
        return obj, params
