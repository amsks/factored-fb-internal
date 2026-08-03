"""GCIQLCritic — goal-conditioned IQL critic on the factored seam.

Expectile value + twin-Q TD, transcribed from OGBench gciql.py (continuous branch).
First PROVIDES_V critic — unblocks AWRActor. One joint `iql` stage matches vendored's
single total_loss grad; stop_gradient on next_v decouples value/critic exactly as vendored
(which reads the online value there). Nothing vendored is imported.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCValue


@dataclasses.dataclass(frozen=True)
class GCIQLCritic:
    value: Any
    critic: Any
    discount: float
    expectile: float
    tau: float
    obs_dim: int
    q_bound: float

    GROUPS = ("value", "critic")
    STAGES = ("iql",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})  # sample_cond() produces these

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "iql"
        return self.GROUPS

    # --- module apply helpers ---
    def _squash(self, x):
        # q_bound > 0 restricts the hypothesis class to |V|,|Q| <= q_bound via
        # kappa*tanh(x/kappa) — the bounded-monolithic control for the bilinear critic,
        # whose norm-concentrated head satisfies |Q| <= ||phi||*||psi||/sqrt(d) by
        # construction (phenomenon.tex, Pred. mech.1). Python-float condition: static
        # under jit, so q_bound=0 (the default) compiles to the identity.
        return self.q_bound * jnp.tanh(x / self.q_bound) if self.q_bound > 0 else x

    def _V(self, p, obs, goal):
        return self._squash(self.value.apply({"params": p}, obs, goal))            # (B,)

    def _Q(self, p, obs, goal, actions):
        return self._squash(self.critic.apply({"params": p}, obs, goal, actions))  # (2, B)

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "iql"
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        # value loss (expectile): grad -> value; twin-Q from the frozen target.
        q1, q2 = self._Q(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2)
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        # critic loss (TD): grad -> critic; next_v uses stop-grad online value (vendored decoupling).
        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        cq1, cq2 = self._Q(params["critic"], obs, vg, action)
        critic_loss = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()

        loss = value_loss + critic_loss
        return loss, {"value_loss": value_loss, "critic_loss": critic_loss,
                      "v_mean": v.mean(), "q_target_mean": tq.mean()}

    def v_fn(self, params, obs, cond):
        return self._V(params["value"], obs, cond["goal"])

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad is accepted for seam uniformity; GCIQL has no separate encoder.
        q1, q2 = self._Q(params["critic"], obs, cond["goal"], actions)
        return jnp.minimum(q1, q2), jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        # GC agents condition on the env's task goal (set via sample_actions(goals=)), not an
        # inferred latent; return a zero placeholder that the eval goal-path overrides.
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        value = GCValue(hidden_dims=hd, layer_norm=ln, ensemble=False)
        critic = GCValue(hidden_dims=hd, layer_norm=ln, ensemble=True)
        obs_dim = ex_obs.shape[-1]
        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc = jr.split(jr.PRNGKey(config["seed"] + 5000), 2)
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "critic": critic.init(rc, ex_obs, ex_goal, ex_actions)["params"],
        }
        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(value=value, critic=critic, discount=float(config["discount"]),
                  expectile=float(config["expectile"]), tau=float(config["tau"]),
                  obs_dim=int(obs_dim), q_bound=float(config["q_bound"]))
        return obj, params, targets
