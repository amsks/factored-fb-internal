"""Flow Q-Learning actor — transcribed from PSMFlows fql.py:actor_loss, conditioned on
the critic's cond_vec. Distinct from flowbc: Park's bc-flow + one-step distillation with
alpha-weighted distill (not the FB-reference NoiseConditionedActor)."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _cvec, _obs_feat, _nobs_feat


@dataclasses.dataclass(frozen=True)
class FQLActor:
    actor_bc_flow: Any
    actor_onestep: Any
    flow_steps: int
    alpha: float
    normalize_q_loss: bool

    GROUPS = ("actor_bc_flow", "actor_onestep")
    NEEDS_V = False
    REQUIRES_COND = frozenset()  # _cvec accepts either cond["z"] or cond["goal"]

    def _bc(self, p, obs_c, x, t):
        return self.actor_bc_flow.apply({"params": p}, obs_c, x, t)

    def _one(self, p, obs_c, noise):
        return self.actor_onestep.apply({"params": p}, obs_c, noise)

    def _rollout(self, p, obs_c, noise):
        a = noise
        for i in range(self.flow_steps):
            t = jnp.full((*obs_c.shape[:-1], 1), i / self.flow_steps)
            a = a + self._bc(p, obs_c, a, t) / self.flow_steps
        return jnp.clip(a, -1.0, 1.0)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        # Both flow nets take FEATURES; the critic below still gets the raw observation and
        # encodes it itself behind `stop_encoder_grad`. Identity off pixels, so every state
        # cell stays bit-identical -- see impls/actors/ddpgbc.py::_obs_feat.
        obs_c = jnp.concatenate([_obs_feat(cond, obs), cvec], -1)
        x0, t, noise = boot["flow_x0"], boot["flow_t"], boot["actor_noise"]
        xt = (1 - t) * x0 + t * action
        vel = action - x0
        pred = self._bc(params["actor_bc_flow"], obs_c, xt, t)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        tgt = jax.lax.stop_gradient(self._rollout(params["actor_bc_flow"], obs_c, noise))
        a1 = self._one(params["actor_onestep"], obs_c, noise)
        distill = jnp.mean((a1 - tgt) ** 2)
        q, _ = critic.q_fn(params, obs, cond, jnp.clip(a1, -1, 1), stop_encoder_grad=True)
        q_loss = -q.mean()
        if self.normalize_q_loss:
            q_loss = jax.lax.stop_gradient(1.0 / jnp.abs(q).mean()) * q_loss
        loss = bc_flow_loss + self.alpha * distill + q_loss
        return loss, {"actor_loss": loss, "bc_flow_loss": bc_flow_loss,
                      "distill_loss": distill, "q_loss": q_loss, "q": q.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = _nobs_feat(cond, batch["next_observations"])
        obs_c = jnp.concatenate([next_obs, _cvec(cond)], -1)
        B, A = batch["actions"].shape
        r_na, r_x0, r_t, r_n = jax.random.split(rng, 4)
        # hoisted out of the call below so the value-goal draw can REUSE it
        na_noise = jax.random.normal(r_na, (B, A))
        na = jnp.clip(self._one(net["actor_onestep"].params, obs_c, na_noise), -1, 1)
        boot = {"next_action": na, "flow_x0": jax.random.normal(r_x0, (B, A)),
                "flow_t": jax.random.uniform(r_t, (B, 1)), "actor_noise": jax.random.normal(r_n, (B, A))}
        # The CRITIC decides what the actor's conditioning vector at the value goal is, and
        # publishes it as cond["_cvec_vgoal"]. Actors must not infer it from cond["value_goal"]:
        # that raw goal is only the right vector when the actor rides the goal itself, and
        # guessing it broke twice -- iql_bilinear(actor_cond="z") wanted a latent, hiql wanted
        # a subgoal rep, and both got an obs-shaped vector against a differently-shaped net.
        # Absent key => this critic does not bootstrap at the value goal, so no draw.
        if "_cvec_vgoal" in cond:
            # See impls/actors/ddpgbc.py::sample_bootstrap. Unlike ddpgbc/awr this actor is
            # stochastic, so the draw REUSES na_noise -- one draw, no extra split, the same
            # four-way split as before. Splitting again would move `next_action` and every
            # existing x fql cell with it.
            obs_v = jnp.concatenate([next_obs, cond["_cvec_vgoal"]], -1)
            boot["next_action_vgoal"] = jnp.clip(
                self._one(net["actor_onestep"].params, obs_v, na_noise), -1, 1)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        # The noise shape is taken from the FEATURE array, not the raw observation: on pixels
        # `obs.shape[:-1]` is (H, W), which would draw an (H, W, adim) noise tensor.
        pobs = _obs_feat(cond, obs)
        obs_c = jnp.concatenate([pobs, _cvec(cond)], -1)
        noise = jax.random.normal(rng, (*pobs.shape[:-1], self.actor_onestep.action_dim))
        return jnp.clip(self._one(net["actor_onestep"].params, obs_c, noise), -1, 1)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import ActorVectorField
        a_cfg = config["actor"]
        A = ex_actions.shape[-1]
        hd = tuple(a_cfg["hidden_dims"])
        bc = ActorVectorField(hidden_dims=hd, action_dim=A, layer_norm=a_cfg.get("layer_norm", False))
        one = ActorVectorField(hidden_dims=hd, action_dim=A, layer_norm=a_cfg.get("layer_norm", False))
        # `actor_obs_dim` is published by Agent.create from the critic when the actor's
        # observation leg is NOT the raw observation width (density_fb on pixels: encoder
        # features). Absent elsewhere, and then this is ex_obs.shape[-1] exactly as before.
        odim = int(config.get("actor_obs_dim") or ex_obs.shape[-1])
        obs_c = jnp.zeros((ex_obs.shape[0], odim + config.get("cvec_dim", config.get("z_dim"))))
        rbc, ron = jr.split(jr.PRNGKey(config["seed"] + 4000))
        params = {
            "actor_bc_flow": bc.init(rbc, obs_c, ex_actions, ex_actions[..., :1])["params"],
            "actor_onestep": one.init(ron, obs_c, ex_actions, None)["params"],
        }
        return cls(actor_bc_flow=bc, actor_onestep=one, flow_steps=int(a_cfg["flow_steps"]),
                   alpha=float(a_cfg["alpha"]),
                   normalize_q_loss=bool(a_cfg.get("normalize_q_loss", False))), params
