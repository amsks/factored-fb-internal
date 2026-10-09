"""FlowBCActor — FQL-style flow-matching actor on the factored seam.

Math transcribed verbatim from PSMFlows agents/fb.py:_actor_loss_fn (flow branch),
_draw_injection (flow), and sample_actions.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec, _obs_feat, _nobs_feat


@dataclasses.dataclass(frozen=True)
class FlowBCActor:
    actor: Any
    actor_vf: Any
    flow_steps: int
    bc_coeff: float
    actor_pessimism_penalty: float
    # Weight on the critic's value term in the ONE-STEP actor objective. 1.0 is FlowBC as
    # published; 0.0 is a pure behaviour-cloning flow policy, the BC control.
    q_coeff: float = 1.0

    GROUPS = ("actor", "actor_vf")
    NEEDS_V = False
    # Conditions on the critic's conditioning vector (critic.cond_vec / _cvec): cond["z"]
    # for FB, the goal for goal critics. No hard cond-key requirement.
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

    @staticmethod
    def _policy_drift(params, obs, cond, critic, a_actor, a_data, q_actor):
        """Diagnostics for the actor--critic bootstrap loop: how far the policy has walked
        off the behaviour support (mean ||a_actor - a_data||) and how much the critic prefers
        where it walked to (mean Q(s, a_data) - Q(s, a_actor)).

        Reporting only: every value is stop-gradient'd and nothing here reaches `loss`.
        """
        q_data, _ = critic.q_fn(params, obs, cond, a_data, stop_encoder_grad=True)
        return jax.lax.stop_gradient({
            "action_gap_l2": jnp.linalg.norm(a_actor - a_data, axis=-1).mean(),
            "q_gap_data_minus_actor": q_data.mean() - q_actor.mean(),
        })

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        # Policy nets take FEATURES; the critic gets the raw observation and encodes it
        # itself behind `stop_encoder_grad`. On state `pobs is obs`.
        pobs = _obs_feat(cond, obs)
        c = critic.cond_vec(cond)
        x0, t, noise = boot["flow_x0"], boot["flow_t"], boot["actor_noise"]

        x1 = action
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        pred = self._vf(params["actor_vf"], pobs, xt, t)
        bc_flow = jnp.mean((pred - vel) ** 2)

        a = self._actor(params["actor"], pobs, c, noise)
        Q, Qs = critic.q_fn(params, obs, cond, a, stop_encoder_grad=True)
        loss = -self.q_coeff * Q.mean()
        info = {"q": Q.mean(), "bc_flow_loss": bc_flow}
        info.update(self._policy_drift(params, obs, cond, critic, a, action, Q))
        if self.bc_coeff > 0:
            tgt = jax.lax.stop_gradient(self._rollout(params["actor_vf"], pobs, noise))
            distill = jnp.mean((a - tgt) ** 2)
            if self.q_coeff == 0.0:
                # Pure BC: Q is still computed and logged but never reaches the gradient.
                # The Q-normalisation is DROPPED rather than multiplied by zero -- |Qs| can
                # be ~0 early in training, and 0/0 is nan.
                loss = self.bc_coeff * distill
            else:
                loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + self.bc_coeff * distill
            info["bc_error"] = distill
        loss = loss + bc_flow
        info["actor_loss"] = loss
        return loss, info

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = _nobs_feat(cond, batch["next_observations"])
        B = next_obs.shape[0]
        adim = batch["actions"].shape[-1]
        c = _cvec(cond)
        r_na, r_x0, r_t, r_noise = jax.random.split(rng, 4)
        na_noise = jax.random.normal(r_na, (B, adim))
        na = self._actor(net["actor"].params, next_obs, c, na_noise)
        boot = {"next_action": na,
                "flow_x0": jax.random.normal(r_x0, (B, adim)),
                "flow_t": jax.random.uniform(r_t, (B, 1)),
                "actor_noise": jax.random.normal(r_noise, (B, adim))}
        # The cond["_cvec_vgoal"] contract: see impls/actors/ddpgbc.py::sample_bootstrap.
        if "_cvec_vgoal" in cond:
            # Reuses na_noise, so critics reading `next_action` stay bit-identical. NOT
            # universal: right for density_fb, whose forward tower F(s,a,g) makes its row
            # g-indexed; td_fb's forward map is phi(s,a) with no goal leg and reads
            # `next_action` instead -- measured 0.920 actor-goal against 0.076 value-goal on
            # the cube-single phase-1 anchor.
            boot["next_action_vgoal"] = self._actor(
                net["actor"].params, next_obs, cond["_cvec_vgoal"], na_noise)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        c = _cvec(cond)
        # Noise shape from the FEATURE array, not the raw observation: on pixels
        # `obs.shape[:-1]` is (64, 64), which would draw a (64, 64, adim) noise tensor.
        pobs = _obs_feat(cond, obs)
        noise = jax.random.normal(rng, (*pobs.shape[:-1], self.actor.action_dim))
        a = self._actor(net["actor"].params, pobs, c, noise)
        return jnp.clip(a, -1.0, 1.0)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import NoiseConditionedActor, FlowVectorField
        adim = ex_actions.shape[-1]
        # cvec_dim comes from the critic via Agent.create; z_dim is the fixture fallback.
        cdim = config.get("cvec_dim", config.get("z_dim"))
        a_cfg = config["actor"]
        actor = NoiseConditionedActor(action_dim=adim, hidden_dim=a_cfg["hidden_dim"],
                                      hidden_layers=a_cfg["hidden_layers"],
                                      embedding_layers=a_cfg["embedding_layers"])
        vf = FlowVectorField(action_dim=adim, hidden_dim=a_cfg["flow_vf_hidden_dim"],
                             hidden_layers=a_cfg["flow_vf_hidden_layers"])
        ract, rvf = jr.split(jr.PRNGKey(config["seed"] + 1000), 2)
        ex_z = jnp.zeros((ex_obs.shape[0], cdim))
        # `actor_obs_dim` is published when the actor sees something other than the raw
        # observation (density_fb on pixels); absent on state.
        odim = config.get("actor_obs_dim")
        ex_pobs = ex_obs if odim is None else jnp.zeros((ex_obs.shape[0], int(odim)))
        params = {
            "actor": actor.init(ract, ex_pobs, ex_z, ex_actions)["params"],
            "actor_vf": vf.init(rvf, ex_pobs, ex_actions, ex_actions[..., :1])["params"],
        }
        obj = cls(actor=actor, actor_vf=vf, flow_steps=int(a_cfg["flow_steps"]),
                  bc_coeff=float(a_cfg["bc_coeff"]), actor_pessimism_penalty=0.0,
                  q_coeff=float(a_cfg.get("q_coeff", 1.0)))
        return obj, params
