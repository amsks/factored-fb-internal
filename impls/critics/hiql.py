"""HIQLCritic — hierarchical implicit Q-learning on the factored seam.

Transcribed from OGBench hiql.py (continuous, state-only). Nothing vendored is imported.

HIQL is not a flat critic, so the seam mapping is the load-bearing design decision
(docs/specs/2026-08-21-density-fb-agent-branch.md, risk 1):

    critic owns  goal_rep, the IVL value head, and the HIGH-level actor
    seam actor IS HIQL's low-level actor, conditioned on the rep the critic hands it

That keeps HIQL composable with all four actor families instead of stranding it on a
bespoke hierarchical actor. Three consequences worth stating:

  * `cond` carries BOTH `goal` (the rep phi([s; g_low]), what the actor conditions on) and
    `low_goal` (the RAW low-level goal). adv_fn needs the raw one: vendored evaluates
    V(s', g_low), whose rep is phi([s'; g_low]) -- a DIFFERENT vector from phi([s; g_low]).
    Collapsing the two silently changes the advantage.
  * the low-level AWR advantage rides the `adv_fn` hook (added with GCIVL). Without it the
    seam's default q_fn - v_fn is zero for a V-only critic and AWR degenerates to BC.
  * `low_actor_rep_grad=True` is NOT supported. The rep is computed in sample_cond, outside
    any grad, so the low-level actor never back-propagates into goal_rep. That matches
    vendored's DEFAULT (low_actor_rep_grad=False); build() rejects the other setting rather
    than silently ignoring it.

Eval: `map_eval_cond` runs the high-level policy to turn the raw env goal into a subgoal
rep. It needs `observations` because HIQL's conditioning is a joint function of state and
goal -- which is why the seam's hook signature was widened.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import MLP, Actor, GCValue


def _length_normalize(x):
    """vendored utils/networks.py:LengthNormalize -- scale to length sqrt(dim)."""
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


@dataclasses.dataclass(frozen=True)
class HIQLCritic:
    goal_rep: Any
    value: Any
    high_actor: Any
    discount: float
    expectile: float
    tau: float
    high_alpha: float
    obs_dim: int
    rep_dim: int

    GROUPS = ("goal_rep", "value", "high_actor")
    STAGES = ("ivl", "high_actor")
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"goal", "low_goal"})
    # infer_cond here is a fixed zero placeholder that the eval goal path overrides, so it
    # never reads `obs`. Declared rather than defaulted -- see base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "unused"

    @property
    def TARGETS(self):
        # ONLY the value head is polyak-tracked. Vendored *looks* like it builds a second
        # goal_rep (target_value_encoder_def is a fresh GCEncoder wrapping goal_rep_def,
        # hiql.py:306), but flax binds the shared goal_rep_def instance ONCE inside the
        # ModuleDict, so `modules_target_value` contains only `value_net` -- its concat
        # encoder resolves to `modules_goal_rep`, the ONLINE rep. Measured on the vendored
        # agent: the param tree has exactly one goal_rep and target_value has no encoder
        # params. Polyak-tracking a second copy here made the TD target read a lagged rep
        # and moved the IVL loss (see test_ivl_target_legs_read_the_online_goal_rep).
        return {"value": self.tau}

    @property
    def cvec_dim(self):
        return self.rep_dim

    def stage_groups(self, stage):
        return ("goal_rep", "value") if stage == "ivl" else ("high_actor",)

    # --- modules -------------------------------------------------------------
    def _rep(self, p_rep, obs, goal):
        """phi([s; g]), length-normalised."""
        h = self.goal_rep.apply({"params": p_rep}, jnp.concatenate([obs, goal], -1))
        return _length_normalize(h)

    def _V_rep(self, p_val, obs, rep):
        return self.value.apply({"params": p_val}, obs, rep)                # (2, B)

    def _V_raw(self, p_rep, p_val, obs, goal):
        """V(s, g) with the rep computed from THIS state and that goal, as vendored's
        GCEncoder(concat_encoder=goal_rep) does."""
        v1, v2 = self._V_rep(p_val, obs, self._rep(p_rep, obs, goal))
        return (v1 + v2) / 2

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    # --- stages --------------------------------------------------------------
    def _ivl(self, params, target, batch, vg):
        obs, next_obs = batch["observations"], batch["next_observations"]
        rewards, masks = batch["rewards"], batch["masks"]
        # The TD-target legs read the ONLINE goal_rep (vendored's target_value shares
        # `modules_goal_rep`; see TARGETS), detached because vendored calls target_value
        # without `params=grad_params`, which stops the gradient through the whole module
        # -- encoder included (flax_utils.py:90-114).
        tr, tv = jax.lax.stop_gradient(params["goal_rep"]), target["value"]
        orr, ov = params["goal_rep"], params["value"]

        nv1_t, nv2_t = self._V_rep(tv, next_obs, self._rep(tr, next_obs, vg))
        q = rewards + self.discount * masks * jnp.minimum(nv1_t, nv2_t)
        v1_t, v2_t = self._V_rep(tv, obs, self._rep(tr, obs, vg))
        adv = q - (v1_t + v2_t) / 2

        q1 = rewards + self.discount * masks * nv1_t
        q2 = rewards + self.discount * masks * nv2_t
        v1, v2 = self._V_rep(ov, obs, self._rep(orr, obs, vg))
        v = (v1 + v2) / 2
        loss = (self._expectile_loss(adv, q1 - v1, self.expectile).mean()
                + self._expectile_loss(adv, q2 - v2, self.expectile).mean())
        return loss, {"value_loss": loss, "v_mean": v.mean(), "adv_value": adv.mean()}

    def _high_actor(self, params, batch):
        """vendored hiql.py:141-166. AWR over subgoal REPS: the high policy regresses the
        rep of the subgoal it should aim for, weighted by that subgoal's advantage."""
        obs = batch["observations"]
        hg, ht = batch["high_actor_goals"], batch["high_actor_targets"]
        # value and goal_rep are frozen for this stage (vendored reads them without grad_params)
        rp = jax.lax.stop_gradient(params["goal_rep"])
        vp = jax.lax.stop_gradient(params["value"])
        v = self._V_raw(rp, vp, obs, hg)
        nv = self._V_raw(rp, vp, ht, hg)
        adv = nv - v
        exp_a = jnp.minimum(jnp.exp(adv * self.high_alpha), 100.0)

        dist = self.high_actor.apply({"params": params["high_actor"]},
                                     jnp.concatenate([obs, hg], -1))
        target = jax.lax.stop_gradient(self._rep(rp, obs, ht))
        log_prob = dist.log_prob(target)
        loss = -(exp_a * log_prob).mean()
        return loss, {"high_actor_loss": loss, "high_adv": adv.mean(),
                      "high_bc_log_prob": log_prob.mean()}

    def stage_loss(self, stage, params, target, batch, cond, boot):
        if stage == "ivl":
            return self._ivl(params, target, batch, cond["_value_goal"])
        assert stage == "high_actor"
        return self._high_actor(params, batch)

    # --- seam API ------------------------------------------------------------
    def v_fn(self, params, obs, cond):
        v1, v2 = self._V_rep(params["value"], obs, cond["goal"])
        return (v1 + v2) / 2

    def adv_fn(self, params, batch, cond):
        """vendored hiql.py:104-108: adv = mean(V(s',g_low)) - mean(V(s,g_low)), at the RAW
        low-level goal, so each side gets its own state-dependent rep."""
        rp, vp = params["goal_rep"], params["value"]
        g = cond["low_goal"]
        return (self._V_raw(rp, vp, batch["next_observations"], g)
                - self._V_raw(rp, vp, batch["observations"], g))

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # No action model: Q := V(s, rep), ignoring actions (the qrl/vip/gcivl convention).
        # AWR does NOT use this -- see adv_fn. flowbc/fql ascend it.
        v = self.v_fn(params, obs, cond)
        return v, jnp.stack([v, v], 0)

    def cond_vec(self, cond):
        return cond["goal"]                                   # the subgoal REP

    def sample_cond(self, net, batch, rng):
        rep = jax.lax.stop_gradient(
            self._rep(net["goal_rep"].params, batch["observations"], batch["low_actor_goals"]))
        # `_value_goal`, not `value_goal`: the underscore keeps it PRIVATE to this critic,
        # matching density_fb's `_density_u_*` convention. Every actor emits its extra
        # value-goal bootstrap draw on `"value_goal" in cond`, conditioning the actor on the
        # RAW goal -- but HIQL's actor rides a subgoal REP of width rep_dim, so that draw
        # would hand a rep-shaped network a goal-shaped input. Only HIQL's own stage_loss
        # reads this, and its BOOT_COND is the default "actor_goal", so nothing is lost.
        return {"_value_goal": batch["value_goals"], "goal": rep,
                "low_goal": batch["low_actor_goals"]}

    def map_eval_cond(self, net, cond, observations=None):
        """Run the HIGH-level policy: raw env goal -> subgoal rep the low-level actor rides.
        Needs `observations` because phi([s; g]) is joint in state and goal."""
        if observations is None:
            raise ValueError(
                "HIQLCritic.map_eval_cond needs observations: its conditioning is "
                "goal_rep([s; g]), a joint function of state and goal, so a subgoal rep "
                "cannot be produced from the goal alone.")
        goal = cond["goal"]
        goal = jnp.broadcast_to(goal, observations.shape) if goal.ndim == 1 else goal
        dist = self.high_actor.apply({"params": net["high_actor"].params},
                                     jnp.concatenate([observations, goal], -1))
        return {**cond, "goal": _length_normalize(dist.mode()), "low_goal": goal}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        if config.get("low_actor_rep_grad", False):
            raise ValueError(
                "hiql: low_actor_rep_grad=true is not supported on this seam. The subgoal "
                "rep is computed in sample_cond, outside any grad, so the low-level actor "
                "cannot back-propagate into goal_rep. Vendored's default is false.")
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        rep_dim = int(config["rep_dim"])
        obs_dim = int(ex_obs.shape[-1])

        goal_rep = MLP((*hd, rep_dim), activate_final=False, layer_norm=ln)
        value = GCValue(hidden_dims=hd, layer_norm=ln, ensemble=True)
        # The high-level policy is the CRITIC's, so it takes its width from the critic
        # config -- not from config["actor"], which belongs to whichever seam actor is
        # playing HIQL's low-level role and whose schema differs per actor family.
        high_actor = Actor(hidden_dims=tuple(config["high_actor_hidden_dims"]),
                           action_dim=rep_dim, layer_norm=False, tanh_squash=False,
                           const_std=bool(config.get("high_const_std", True)))

        rg, rv, rh = jr.split(jr.PRNGKey(config["seed"] + 8000), 3)
        ex_pair = jnp.concatenate([ex_obs, jnp.zeros_like(ex_obs)], -1)
        ex_rep = jnp.zeros((ex_obs.shape[0], rep_dim))
        params = {
            "goal_rep": goal_rep.init(rg, ex_pair)["params"],
            "value": value.init(rv, ex_obs, ex_rep)["params"],
            "high_actor": high_actor.init(rh, ex_pair)["params"],
        }
        targets = {"value": copy.deepcopy(params["value"])}
        obj = cls(goal_rep=goal_rep, value=value, high_actor=high_actor,
                  discount=float(config["discount"]), expectile=float(config["expectile"]),
                  tau=float(config["tau"]), high_alpha=float(config["high_alpha"]),
                  obs_dim=obs_dim, rep_dim=rep_dim)
        return obj, params, targets
