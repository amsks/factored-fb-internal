"""DDPG+BC actor — transcribed from OGBench gciql.py:actor_loss (ddpgbc branch)."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


def _cvec(cond):
    """Conditioning vector for actors that don't receive the critic (sample_* paths).

    A critic whose conditioning is neither the raw goal nor a latent -- density_fb on
    PIXELS, where it is the encoder's detached embedding of the goal IMAGE -- publishes it
    as cond["_cvec"]. Checked FIRST so no actor has to guess: cond["goal"] is a (64, 64, 3)
    tensor there, and concatenating it onto an observation would fail at best."""
    if "_cvec" in cond:
        return cond["_cvec"]
    return cond["z"] if "z" in cond else cond["goal"]


def _obs_feat(cond, obs):
    """The array the actor's POLICY nets consume in place of `obs`.

    Identity everywhere except pixels, where the critic publishes DETACHED encoder features
    as cond["_obs_feat"] (and cond["_obs_feat_next"] for s'). The actor owns no encoder, so
    these arrive through `cond`: `sample_bootstrap`/`sample_actions` are handed `net` and
    `cond`, never the critic."""
    return cond["_obs_feat"] if "_obs_feat" in cond else obs


def _nobs_feat(cond, next_obs):
    """`_obs_feat`'s twin for the next state (the bootstrap draw's input)."""
    return cond["_obs_feat_next"] if "_obs_feat_next" in cond else next_obs


def _smooth(a, rng, std, clip):
    """TD3-style TARGET POLICY SMOOTHING on a bootstrap action.

    `a + eps`, eps ~ N(0, std) truncated at +-clip, re-clipped to the action range. This is
    the published FB recipe, not an embellishment: Touati et al. Appendix G.4 evaluates the
    TD target at `F(s', pi(s') + eps_i, z)` with eps from a truncated centered Gaussian, and
    Table 1 pins stddev 0.2 / truncation 0.3. Without it a deterministic actor bootstraps at
    its own argmax and is free to ride sharp maxima of an unconverged critic -- which is the
    shape of the FB divergence measured on ExoRL (q_mean 77 -> 436 on walker).

    std=0.0 short-circuits and draws NO randomness, so every pre-existing run stays
    bit-identical.
    """
    if std <= 0.0:
        return a
    eps = jnp.clip(std * jax.random.normal(rng, a.shape), -clip, clip)
    return jnp.clip(a + eps, -1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DDPGBCActor:
    actor: Any
    alpha: float
    const_std: bool
    boot_noise_std: float = 0.0      # target policy smoothing; 0.0 = off (bit-identical)
    boot_noise_clip: float = 0.3
    # Touati parity (2026-08-27): the reference's ACTOR loss also samples the action —
    # dist.sample(clip=stddev_clip) around the mode (controllable_agent fb_ddpg.py; paper
    # eq. 43's eps inside the policy objective). 0.0 = the noiseless mode every prior cell
    # ran. The rng arrives as boot["actor_q_rng"], a fold_in that consumes nothing from the
    # existing draw stream.
    q_noise_std: float = 0.0
    q_noise_clip: float = 0.3
    # Divide the Q term by sg[|q|.mean()] before adding the BC term. True (default) is the
    # shipped behavior: it fixes the scale of the value term relative to BC, which matters
    # when the two are summed. RLDP's official actor is a PURE TD3 objective with no BC
    # term at all -- `actor_loss = -Q.mean()` (baselines/rldp agent.py::update_td3_actor) --
    # so there is nothing to balance against and the normalizer would only act as an
    # adaptive, q-magnitude-dependent learning rate the reference does not have. Set false
    # for that arm; every existing cell keeps the normalizer.
    q_normalize: bool = True

    GROUPS = ("actor",)
    NEEDS_V = False
    REQUIRES_COND = frozenset()  # _cvec accepts either cond["z"] or cond["goal"]

    def _dist(self, p, obs, cvec, temperature=1.0):
        return self.actor.apply({"params": p}, jnp.concatenate([obs, cvec], -1), temperature=temperature)

    def loss(self, params, target, batch, critic, cond, boot):
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        # The POLICY takes features, the CRITIC the raw observation, which it encodes under
        # `stop_encoder_grad`. Both legs detached, so no encoder param moves.
        dist = self._dist(params["actor"], _obs_feat(cond, obs), cvec)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        if self.q_noise_std > 0.0:
            q_actions = _smooth(q_actions, boot["actor_q_rng"],
                                self.q_noise_std, self.q_noise_clip)
        q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_loss = -q.mean()
        if self.q_normalize:
            q_loss = q_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {"actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
                      "q_mean": q.mean(), "bc_log_prob": log_prob.mean()}

    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = _nobs_feat(cond, batch["next_observations"])
        p = net["actor"].params
        # ONE eps for both bootstrap keys, following flowbc's reused-noise discipline: the
        # two keys are the same policy evaluated at two conditionings, and spending a second
        # draw here would shift every downstream rng without changing what is smoothed.
        boot = {"next_action": _smooth(
            jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1),
            rng, self.boot_noise_std, self.boot_noise_clip),
                # For loss()'s q-term noise (q_noise_std > 0). fold_in derives a fresh key
                # WITHOUT consuming the stream, so every existing draw is bit-identical.
                "actor_q_rng": jax.random.fold_in(rng, 7)}
        # CONTRACT (canonical statement; the other actors point here). The CRITIC decides
        # the actor's conditioning vector at the value goal and publishes it as
        # cond["_cvec_vgoal"]. Actors must NOT infer it from cond["value_goal"], which is the
        # right vector only when the actor rides the goal itself -- a latent- or
        # subgoal-conditioned net gets an obs-shaped vector against a differently-shaped net.
        # Absent key => this critic does not bootstrap at the value goal, so no draw.
        if "_cvec_vgoal" in cond:
            # a+ ~ pi(.|s', g_value) at frozen params, for critics whose TD target is only
            # valid at the VALUE goal (Critic.BOOT_COND == "value_goal"). A SEPARATE key, so
            # critics reading `next_action` stay bit-identical.
            boot["next_action_vgoal"] = _smooth(
                jnp.clip(self._dist(p, next_obs, cond["_cvec_vgoal"]).mode(), -1, 1),
                rng, self.boot_noise_std, self.boot_noise_clip)
        return boot

    def sample_actions(self, net, obs, cond, rng, temperature=1.0):
        dist = self._dist(net["actor"].params, _obs_feat(cond, obs), _cvec(cond), temperature)
        return jnp.clip(dist.mode(), -1, 1)

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        from impls.utils.networks import Actor
        a_cfg = config["actor"]
        # `actor_obs_dim` is published by Agent.create when the actor's observation leg is
        # NOT the raw observation width -- density_fb on pixels sees encoder features.
        odim = int(config.get("actor_obs_dim") or ex_obs.shape[-1])
        if a_cfg.get("motivo_archi", False):
            # metamotivo/nn_models.py::Actor -- two embedding branches, over (obs, cond) and
            # over obs alone, each narrowing to hidden_dim // 2 and concatenated back to
            # hidden_dim, then a trunk. `hidden_dims` is read as their (hidden_dim,
            # hidden_layers) pair, so [1024, 1024] is width 1024 with 2 trunk layers.
            from impls.actors._motivo_nets import MotivoPolicy
            hd = tuple(a_cfg["hidden_dims"])
            if len(set(hd)) != 1:
                raise ValueError(
                    f"actor.motivo_archi reads hidden_dims as (width, n_layers) and needs "
                    f"every entry equal; got {list(hd)}.")
            actor = MotivoPolicy(obs_dim=odim, action_dim=ex_actions.shape[-1],
                                 hidden_dim=hd[0], trunk_layers=len(hd), embed_layers=2,
                                 const_std=bool(a_cfg.get("const_std", True)))
        else:
            # tanh_squash=False matches vendored GCIQL's GCActor; actions are clipped at the
            # call sites instead.
            actor = Actor(hidden_dims=tuple(a_cfg["hidden_dims"]), action_dim=ex_actions.shape[-1],
                          layer_norm=a_cfg.get("layer_norm", False), tanh_squash=False,
                          const_std=a_cfg.get("const_std", True))
        ex_in = jnp.zeros((ex_obs.shape[0], odim + config.get("cvec_dim", config.get("z_dim"))))
        params = {"actor": actor.init(jr.PRNGKey(config["seed"] + 2000), ex_in)["params"]}
        # boot_noise_* default to OFF so every existing cell is bit-identical; the ExoRL
        # arms pass the published FB values (0.2 / 0.3, Touati et al. Table 1). One builder
        # serves DDPGBCDensityActor too -- it subclasses this actor and inherits the fields.
        return cls(actor=actor, alpha=float(a_cfg["alpha"]),
                   const_std=bool(a_cfg.get("const_std", True)),
                   boot_noise_std=float(a_cfg.get("boot_noise_std", 0.0)),
                   boot_noise_clip=float(a_cfg.get("boot_noise_clip", 0.3)),
                   q_noise_std=float(a_cfg.get("q_noise_std", 0.0)),
                   q_noise_clip=float(a_cfg.get("q_noise_clip", 0.3)),
                   q_normalize=bool(a_cfg.get("q_normalize", True))), params
