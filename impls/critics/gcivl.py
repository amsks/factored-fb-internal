"""GCIVLCritic — goal-conditioned implicit V-learning on the factored seam.

Transcribed from OGBench gcivl.py (continuous branch). Nothing vendored is imported.
GCIVL is GCIQL without the Q functions: a twin-head V trained with an expectile loss
whose *advantage* leg reads the frozen target and whose *difference* leg reads the
online value (vendored's double-DQN-style split, gcivl.py:29-56).

Two seam details that are not cosmetic:

  * `q_fn` has no action model, so it returns V(s,g) and ignores `actions`, matching the
    convention qrl/vip already set here. That is fine for flowbc/fql/bcflow, which only
    ascend it, but it makes the seam's default AWR advantage `q_fn - v_fn` identically
    ZERO -- which silently turns AWR into plain behaviour cloning. Measured: qrl x awr
    and vip x awr both report adv = 0 today.
  * so this critic supplies the optional `adv_fn` hook, and AWRActor prefers it. Vendored
    GCIVL's AWR advantage is the NEXT-STATE one, adv = mean(V(s',g)) - mean(V(s,g))
    (gcivl.py:65-69), which is what `adv_fn` returns. Without the hook the GCIVL column of
    a comparison table would be a BC run wearing a GCIVL label.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCValue


@dataclasses.dataclass(frozen=True)
class GCIVLCritic:
    value: Any
    discount: float
    expectile: float
    tau: float
    obs_dim: int

    GROUPS = ("value",)
    STAGES = ("ivl",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    @property
    def TARGETS(self):
        return {"value": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "ivl"
        return self.GROUPS

    def _V(self, p, obs, goal):
        return self.value.apply({"params": p}, obs, goal)        # (2, B)

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "ivl"
        obs, next_obs = batch["observations"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        # advantage leg: frozen target on both sides (vendored's overestimation guard)
        next_v1_t, next_v2_t = self._V(target["value"], next_obs, vg)
        q = rewards + self.discount * masks * jnp.minimum(next_v1_t, next_v2_t)
        v1_t, v2_t = self._V(target["value"], obs, vg)
        adv = q - (v1_t + v2_t) / 2

        # difference leg: online value, per-head targets
        q1 = rewards + self.discount * masks * next_v1_t
        q2 = rewards + self.discount * masks * next_v2_t
        v1, v2 = self._V(params["value"], obs, vg)
        v = (v1 + v2) / 2
        value_loss = (self._expectile_loss(adv, q1 - v1, self.expectile).mean()
                      + self._expectile_loss(adv, q2 - v2, self.expectile).mean())

        return value_loss, {"value_loss": value_loss, "v_mean": v.mean(),
                            "v_max": v.max(), "v_min": v.min(), "adv_value": adv.mean()}

    def v_fn(self, params, obs, cond):
        v1, v2 = self._V(params["value"], obs, cond["goal"])
        return (v1 + v2) / 2

    def adv_fn(self, params, batch, cond):
        """Vendored GCIVL's AWR advantage: mean(V(s',g)) - mean(V(s,g)) at the ACTOR goal
        (gcivl.py:65-69). Stop-gradient is applied by AWRActor, which uses this only as a
        weight, exactly as vendored does."""
        nv1, nv2 = self._V(params["value"], batch["next_observations"], cond["goal"])
        v1, v2 = self._V(params["value"], batch["observations"], cond["goal"])
        return (nv1 + nv2) / 2 - (v1 + v2) / 2

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # No action model: Q := V(s,g), ignoring actions -- the qrl/vip convention. AWR does
        # NOT use this (see adv_fn above); flowbc/fql ascend it.
        v = self.v_fn(params, obs, cond)
        return v, jnp.stack([v, v], 0)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        # GC agent: conditions on the env's task goal, not an inferred latent.
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        value = GCValue(hidden_dims=tuple(config["value_hidden_dims"]),
                        layer_norm=bool(config["layer_norm"]), ensemble=True)
        ex_goal = jnp.zeros_like(ex_obs)
        params = {"value": value.init(jr.PRNGKey(config["seed"] + 7000), ex_obs, ex_goal)["params"]}
        targets = {"value": copy.deepcopy(params["value"])}
        obj = cls(value=value, discount=float(config["discount"]),
                  expectile=float(config["expectile"]), tau=float(config["tau"]),
                  obs_dim=int(ex_obs.shape[-1]))
        return obj, params, targets
