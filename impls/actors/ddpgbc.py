"""DDPG+BC actor — transcribed from OGBench gciql.py:actor_loss (ddpgbc branch)."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


def _cvec(cond):
    """Conditioning vector for actors that don't receive the critic (sample_* paths)."""
    return cond["z"] if "z" in cond else cond["goal"]


@dataclasses.dataclass(frozen=True)
class DDPGBCActor:
    actor: Any
    alpha: float
    const_std: bool

    GROUPS = ("actor",)
    NEEDS_V = False
    REQUIRES_COND = frozenset()  # _cvec accepts either cond["z"] or cond["goal"]

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        dist = self._dist(params["actor"], obs, cvec)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {"actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
                      "q_mean": q.mean(), "bc_log_prob": log_prob.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        boot = {"next_action": jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1)}
        # `_cvec` conditions the actor on cond["z"] when the critic supplies one, and on
        # cond["goal"] otherwise. The value-goal draw substitutes cond["value_goal"] for that
        # vector, which is only well-defined in the goal branch: when the actor is conditioned
        # on a latent z there is no "z at the value goal" in cond at all (the critic derives z
        # from actor_goals), and feeding the raw goal instead is a width mismatch -- 
        # iql_bilinear with actor_cond="z" builds actor_net for obs+latent and gets obs+obs.
        if "value_goal" in cond and "z" not in cond:
            # a+ ~ pi(.|s', g_value) at frozen params, for critics whose TD target is only
            # valid at the VALUE goal (Critic.BOOT_COND == "value_goal"; density_fb's forward
            # tower is F(s,a,g), so its row is g-indexed). Emitted as a SEPARATE key -- every
            # critic reading `next_action` stays bit-identical. Mirrors
            # FlowBCActor.sample_bootstrap; this actor is deterministic (dist.mode()), so the
            # extra draw spends no randomness at all.
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
        # tanh_squash=False matches vendored GCIQL's GCActor (gciql.py:248 leaves the
        # GCActor default). Actions are clipped at the call sites instead.
        actor = Actor(hidden_dims=tuple(a_cfg["hidden_dims"]), action_dim=ex_actions.shape[-1],
                      layer_norm=a_cfg.get("layer_norm", False), tanh_squash=False,
                      const_std=a_cfg.get("const_std", True))
        ex_in = jnp.zeros((ex_obs.shape[0], ex_obs.shape[-1] + config.get("cvec_dim", config.get("z_dim"))))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 2000), ex_in)["params"]}
        return cls(actor=actor, alpha=float(a_cfg["alpha"]),
                   const_std=bool(a_cfg.get("const_std", True))), params
