"""DDPG+BC actor — transcribed from OGBench gciql.py:actor_loss (ddpgbc branch)."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


def _cvec(cond):
    """Conditioning vector for actors that don't receive the critic (sample_* paths).

    A critic whose conditioning is neither the raw goal nor a latent -- density_fb on
    PIXELS, where it is the encoder's detached embedding of the goal IMAGE -- publishes it
    explicitly as cond["_cvec"], the same convention as the existing cond["_cvec_vgoal"].
    Checked FIRST so no actor ever has to guess: cond["goal"] is a (64, 64, 3) tensor there,
    and concatenating it onto an observation would fail loudly at best."""
    if "_cvec" in cond:
        return cond["_cvec"]
    return cond["z"] if "z" in cond else cond["goal"]


def _obs_feat(cond, obs):
    """The array the actor's POLICY nets consume in place of `obs`.

    Identity everywhere except pixels, where the critic publishes DETACHED encoder features
    as cond["_obs_feat"] (and cond["_obs_feat_next"] for s'). The actor owns no encoder:
    that is FB's own split (`actor_optimizer` holds `_actor` only, and the actor's update
    wraps the encode in `with torch.no_grad():`), and it is why these arrive through `cond`
    rather than being computed here -- `sample_bootstrap`/`sample_actions` are handed `net`
    and `cond`, never the critic."""
    return cond["_obs_feat"] if "_obs_feat" in cond else obs


def _nobs_feat(cond, next_obs):
    """`_obs_feat`'s twin for the next state (the bootstrap draw's input)."""
    return cond["_obs_feat_next"] if "_obs_feat_next" in cond else next_obs


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
        # The POLICY takes features; the CRITIC takes the raw observation and encodes it
        # itself under `stop_encoder_grad`. Both legs are detached, so this stage moves no
        # encoder parameter.
        dist = self._dist(params["actor"], _obs_feat(cond, obs), cvec)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {"actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
                      "q_mean": q.mean(), "bc_log_prob": log_prob.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = _nobs_feat(cond, batch["next_observations"])
        p = net["actor"].params
        boot = {"next_action": jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1)}
        # The CRITIC decides what the actor's conditioning vector at the value goal is, and
        # publishes it as cond["_cvec_vgoal"]. Actors must not infer it from cond["value_goal"]:
        # that raw goal is only the right vector when the actor rides the goal itself, and
        # guessing it broke twice -- iql_bilinear(actor_cond="z") wanted a latent, hiql wanted
        # a subgoal rep, and both got an obs-shaped vector against a differently-shaped net.
        # Absent key => this critic does not bootstrap at the value goal, so no draw.
        if "_cvec_vgoal" in cond:
            # a+ ~ pi(.|s', g_value) at frozen params, for critics whose TD target is only
            # valid at the VALUE goal (Critic.BOOT_COND == "value_goal"; density_fb's forward
            # tower is F(s,a,g), so its row is g-indexed). Emitted as a SEPARATE key -- every
            # critic reading `next_action` stays bit-identical. Mirrors
            # FlowBCActor.sample_bootstrap; this actor is deterministic (dist.mode()), so the
            # extra draw spends no randomness at all.
            boot["next_action_vgoal"] = jnp.clip(
                self._dist(p, next_obs, cond["_cvec_vgoal"]).mode(), -1, 1)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        dist = self._dist(net["actor"].params, _obs_feat(cond, obs), _cvec(cond), temperature)
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
        # `actor_obs_dim` is published by Agent.create from the critic when the actor's
        # observation leg is NOT the raw observation width -- density_fb on pixels, where the
        # policy sees encoder features (width enc_dim) and never the (64, 64, 3) image.
        odim = int(config.get("actor_obs_dim") or ex_obs.shape[-1])
        ex_in = jnp.zeros((ex_obs.shape[0], odim + config.get("cvec_dim", config.get("z_dim"))))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 2000), ex_in)["params"]}
        return cls(actor=actor, alpha=float(a_cfg["alpha"]),
                   const_std=bool(a_cfg.get("const_std", True))), params
