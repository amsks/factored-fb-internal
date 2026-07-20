"""SimpleSfCritic — Chua et al.'s Simple Successor-Features head on the factored seam.

GCIQL's expectile-V + twin-Q TD (identical objective to GCIQLCritic / IqlBilinearCritic),
but the flat MLP head is replaced by a SINGLE trunk that emits two DIFFERENT vectors off a
shared body:

    z         = psi(g)                          # task vector (sqrt(d)-sphere if project_z)
    h(s,a,z)  = trunk([s, a, sg[z]])
    b(s,a,z)  = h / ||h||                       # reward BASIS, zero params of its own
    Psi_i     = head_i(h)                       # twin critic heads
    Q_i       = <Psi_i, z_used> / sqrt(d)

The design has ONE distinguishing knob, `ground_psi` — the only stop_gradient-psi anywhere
in the factored family. It selects between two rungs, and NOTHING in between is a coordinate
here (the reference isolates this architecture behind 8 asserts; audit critic #6 says keep
it isolated):

  * rung 2  (ground_psi=False, reward_coef=0): plain IQL on the SF head. psi is trained by
    the TD through the readout `z_used = z`. b is trained by the same TD through the trunk.
  * rung 3  (ground_psi=True,  reward_coef>0): the paper's cell. The readout uses
    `z_used = sg[z]`, so the IQL TD gives psi EXACTLY ZERO gradient; the asymmetric reward
    regression r_hat = <sg[b], psi(g)>/sqrt(d) (see `_reward_loss`) is then psi's ONLY
    gradient path. Applied to BOTH nets: grounding only the critic's psi would leak goal
    information back through V(s',g) into the critic target.

The exact-zero-gradient property of the psi leg under ground_psi is the defining, and the
single most important, testable fact about this critic — it has no external numerical
reference (Chua et al. is ported only through the jax shim; the parity scout lists no torch
or vendored analogue), so structure + invariants + that zero-grad property are what is
locked, and no numerical reference is fabricated.

Deviations from the bridge reference (tools/wandb_mode_shim/gciql_flow.py), each deliberate
and each turning one of the reference's silent behaviours into a build-time refusal
(audit S11 / "no silent no-ops"):
  * `ground_psi=True` with `reward_coef==0` FREEZES psi at init (zero grad from everything).
    The reference would run it silently; here it raises SimpleSfConfigError.
  * `reward_coef>0` with `ground_psi=False` is inert: the reference only adds the reward loss
    under `simplesf_ground_psi`, so a lone reward_coef does nothing. Here it raises.
  * `actor_cond` is PINNED to 'goal' (reference assert L1417, S11). A supplied 'z' raises
    rather than being silently accepted then mis-sized.
  * the curriculum fade (`ground_fade`, the reference's anneal_simplesf_ground) is a
    time-varying training SCHEDULE, not a coordinate (audit §6). The static factored seam
    threads no step counter into stage_loss, so a fade cannot be honoured here — it raises at
    build time instead of silently running rung 2.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCSimpleSFValue


class SimpleSfConfigError(ValueError):
    """Raised for a SimpleSf knob combination that would be a silent no-op, a silent
    override, or a schedule the static seam cannot honour. Named (not a bare assert) so
    callers can distinguish it from ComboError."""


@dataclasses.dataclass(frozen=True)
class SimpleSfCritic:
    value: Any
    critic: Any
    discount: float
    expectile: float
    tau: float
    obs_dim: int
    latent_dim: int
    ground_psi: bool
    reward_coef: float
    project_z: bool
    double_q: bool

    GROUPS = ("value", "critic")
    STAGES = ("iql",)
    PROVIDES_V = True
    # actor_cond is pinned to 'goal' (see _validate), so — exactly like GCIQLCritic — the
    # actor conditions on the raw goal and never on a psi readout. No "z" is ever produced.
    PROVIDES_COND = frozenset({"value_goal", "goal"})

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
    def _V(self, p, obs, goal):
        return self.value.apply({"params": p}, obs, goal)             # (B,)

    def _Q(self, p, obs, goal, actions):
        return self.critic.apply({"params": p}, obs, goal, actions)   # (2, B)

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        # `>= 0` tie-break, matching OGBench/GCIQLCritic (the SF head reuses IQL's objective).
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    def _iql_loss(self, params, target, batch, cond):
        """The pure IQL leg: expectile V + twin-Q TD on the SF head, with NO grounding term.
        Exposed as its own method precisely so the zero-gradient test can grad IT (rather than
        the combined stage_loss) w.r.t. psi: this is the leg whose gradient to psi is exactly
        zero under ground_psi, independent of whatever reward-regression policy sits on top."""
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        q1, q2 = self._Q(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        cq1, cq2 = self._Q(params["critic"], obs, vg, action)
        critic_loss = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()

        loss = value_loss + critic_loss
        return loss, {"value_loss": value_loss, "critic_loss": critic_loss,
                      "v_mean": v.mean(), "q_target_mean": tq.mean()}

    def _reward_loss(self, params, batch, cond):
        """Simple SF's reward-prediction loss (sf_simple.py), psi's ONLY gradient path under
        ground_psi:

            r_hat(s,a,g) = <sg[b(s,a,z)], psi(g)>/sqrt(d)   ->  regress on batch['rewards']

        b is stop_gradded (the reference computes it under torch.no_grad()); the whole
        gradient of this term therefore flows into psi (via z=psi(g)) and NOWHERE ELSE. No
        reward shift is needed: the model is ASYMMETRIC (b is a separate object from psi), so
        it has no psi=0 saddle. Applied to BOTH nets — grounding only the critic's psi would
        leak goal information back through V(s',g) into the critic target."""
        obs, acts = batch["observations"], batch["actions"]
        vg, r = cond["value_goal"], batch["rewards"]
        d = self.latent_dim
        _, _, z_q, b_q = self.critic.apply({"params": params["critic"]}, obs, vg, acts, info=True)
        _, _, z_v, b_v = self.value.apply({"params": params["value"]}, obs, vg, info=True)
        # critic Psi/z are ensembled [2,B,d]; value's are [B,d]. b is detached in both.
        rhat_q = (jax.lax.stop_gradient(b_q) * z_q).sum(axis=-1) / jnp.sqrt(d)    # [2,B]
        rhat_v = (jax.lax.stop_gradient(b_v) * z_v).sum(axis=-1) / jnp.sqrt(d)    # [B]
        loss = jnp.mean((rhat_q - r) ** 2) + jnp.mean((rhat_v - r) ** 2)
        return loss, {"reward_loss": loss, "rhat_mean": jnp.mean(rhat_q),
                      "b_norm_mean": jnp.mean(jnp.linalg.norm(b_q, axis=-1))}

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "iql"
        del boot                            # scalar IQL never bootstraps through an actor sample
        loss, info = self._iql_loss(params, target, batch, cond)
        # reward_coef is a Python float, so this branch is static under jit. It is >0 iff
        # ground_psi (enforced in _validate), so this is exactly rung 3.
        if self.reward_coef > 0.0:
            rl, rinfo = self._reward_loss(params, batch, cond)
            loss = loss + self.reward_coef * rl
            info.update({f"simplesf_{k}": v for k, v in rinfo.items()})
        return loss, info

    def v_fn(self, params, obs, cond):
        return self._V(params["value"], obs, cond["goal"])

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad accepted for seam uniformity; there is no separate encoder here.
        q1, q2 = self._Q(params["critic"], obs, cond["goal"], actions)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        return q, jnp.stack([q1, q2], 0)

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

    # ------------------------------------------------------------------ build
    @staticmethod
    def _validate(config):
        """Every check here is a place the bridge reference silently overrides, silently
        ignores, or cannot express on the static seam. Raising is the deliverable."""
        ground_psi = bool(config["ground_psi"])
        reward_coef = float(config["reward_coef"])
        actor_cond = str(config["actor_cond"])

        if bool(config["ground_fade"]):
            raise SimpleSfConfigError(
                "ground_fade=True (the reference's anneal_simplesf_ground curriculum) is a "
                "time-varying training SCHEDULE, not a static coordinate (audit §6). The "
                "factored seam threads no step counter into stage_loss, so the fade cannot be "
                "honoured here; it would silently run as rung 2. Use ground_psi + reward_coef "
                "for the static rungs, or add a TrainingSchedule object.")

        if actor_cond != "goal":
            raise SimpleSfConfigError(
                f"value_arch='simplesf' pins actor_cond='goal' (audit S11 / reference L1417); "
                f"got actor_cond={actor_cond!r}. The SF head exposes no psi readout for the "
                f"actor to condition on.")

        if ground_psi and reward_coef <= 0.0:
            raise SimpleSfConfigError(
                "ground_psi=True with reward_coef<=0 FREEZES psi at init: the readout stop-grads "
                "z so the IQL TD gives psi zero gradient, and with no reward regression psi has "
                "no gradient at all (a silent no-op in the reference). Set reward_coef>0 (rung 3).")

        if reward_coef > 0.0 and not ground_psi:
            raise SimpleSfConfigError(
                "reward_coef>0 with ground_psi=False is inert: the reward-regression grounding "
                "is only added under ground_psi (the reference gates it on simplesf_ground_psi), "
                "so reward_coef would silently do nothing. Set ground_psi=True (rung 3) or "
                "reward_coef=0 (rung 2).")

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        cls._validate(config)

        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])
        ground_psi = bool(config["ground_psi"])
        project_z = bool(config["project_z"])
        obs_dim = int(ex_obs.shape[-1])          # state-based GC: goals are observations

        # SYMMETRIC head kwargs: value and critic differ ONLY in `ensemble` (audit S3 lesson —
        # both nets are grounded, so both must be sphered/grounded identically).
        head_kw = dict(hidden_dims=hd, latent_dim=d, layer_norm=ln,
                       ground_psi=ground_psi, project_z=project_z)
        value = GCSimpleSFValue(ensemble=False, **head_kw)
        critic = GCSimpleSFValue(ensemble=True, **head_kw)

        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc = jr.split(jr.PRNGKey(config["seed"] + 5000), 2)
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "critic": critic.init(rc, ex_obs, ex_goal, ex_actions)["params"],
        }
        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(value=value, critic=critic,
                  discount=float(config["discount"]), expectile=float(config["expectile"]),
                  tau=float(config["tau"]), obs_dim=obs_dim, latent_dim=d,
                  ground_psi=ground_psi, reward_coef=float(config["reward_coef"]),
                  project_z=project_z, double_q=bool(config["double_q"]))
        return obj, params, targets
