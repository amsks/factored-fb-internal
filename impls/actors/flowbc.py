"""FlowBCActor — FQL-style flow-matching actor on the factored seam.

Math transcribed verbatim from PSMFlows agents/fb.py:_actor_loss_fn (flow branch),
_draw_injection (flow), and sample_actions.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec


@dataclasses.dataclass(frozen=True)
class FlowBCActor:
    actor: Any
    actor_vf: Any
    flow_steps: int
    bc_coeff: float
    actor_pessimism_penalty: float
    # Weight on the critic's value term in the ONE-STEP actor objective. 1.0 is FlowBC as
    # published. 0.0 makes this a pure behaviour-cloning flow policy — the control that
    # answers "does the critic buy anything over BC?" — with architecture, optimiser,
    # conditioning and data identical to the agents it is the control for.
    q_coeff: float = 1.0

    GROUPS = ("actor", "actor_vf")
    NEEDS_V = False
    # Conditions on the critic's conditioning vector (critic.cond_vec / _cvec): for FB/ISFB
    # that IS cond["z"] (byte-identical to the original z-only path); for goal critics it is
    # the goal, making this a goal-conditioned flow-BC actor. No hard cond-key requirement.
    REQUIRES_COND = frozenset()

    def _actor(self, p, obs, z, noise):
        return self.actor.apply({"params": p}, obs, z, noise)

    def _vf(self, p, obs, action, t):
        return self.actor_vf.apply({"params": p}, obs, action, t)

    def _rollout(self, vf_p, obs, noise):
        a = noise
        for i in range(self.flow_steps):
            ti = jnp.full((obs.shape[0], 1), i / self.flow_steps)
            a = a + self._vf(vf_p, obs, a, ti) / self.flow_steps
        return jnp.clip(a, -1.0, 1.0)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        c = critic.cond_vec(cond)
        x0, t, noise = boot["flow_x0"], boot["flow_t"], boot["actor_noise"]

        x1 = action
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        pred = self._vf(params["actor_vf"], obs, xt, t)
        bc_flow = jnp.mean((pred - vel) ** 2)

        a = self._actor(params["actor"], obs, c, noise)
        Q, Qs = critic.q_fn(params, obs, cond, a, stop_encoder_grad=True)
        loss = -self.q_coeff * Q.mean()
        info = {"q": Q.mean(), "bc_flow_loss": bc_flow}
        if self.bc_coeff > 0:
            tgt = jax.lax.stop_gradient(self._rollout(params["actor_vf"], obs, noise))
            distill = jnp.mean((a - tgt) ** 2)
            if self.q_coeff == 0.0:
                # Pure BC. Q is still computed and logged (so the control observes the same
                # critic as its treatment arms) but never reaches the gradient. The
                # Q-normalisation is dropped rather than multiplied by zero: |Qs| can be ~0
                # early in training, and 0/0 is nan.
                loss = self.bc_coeff * distill
            else:
                loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + self.bc_coeff * distill
            info["bc_error"] = distill
        loss = loss + bc_flow
        info["actor_loss"] = loss
        return loss, info

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        adim = batch["actions"].shape[-1]
        c = _cvec(cond)
        r_na, r_x0, r_t, r_noise = jax.random.split(rng, 4)
        na = self._actor(net["actor"].params, next_obs, c, jax.random.normal(r_na, (B, adim)))
        return {"next_action": na,
                "flow_x0": jax.random.normal(r_x0, (B, adim)),
                "flow_t": jax.random.uniform(r_t, (B, 1)),
                "actor_noise": jax.random.normal(r_noise, (B, adim))}

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        c = _cvec(cond)
        noise = jax.random.normal(rng, (*obs.shape[:-1], self.actor.action_dim))
        a = self._actor(net["actor"].params, obs, c, noise)
        return jnp.clip(a, -1.0, 1.0)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import NoiseConditionedActor, FlowVectorField
        adim = ex_actions.shape[-1]
        # cvec_dim is set by Agent.create from the critic (= z_dim for FB/ISFB, obs_dim for
        # goal critics). Fall back to z_dim for the direct-build parity fixtures.
        cdim = config.get("cvec_dim", config.get("z_dim"))
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
                  bc_coeff=float(a_cfg["bc_coeff"]), actor_pessimism_penalty=0.0,
                  q_coeff=float(a_cfg.get("q_coeff", 1.0)))
        return obj, params
