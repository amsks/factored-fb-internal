"""IqlBilinearCritic — scalar expectile TD on a bilinear Q head, on the factored seam.

This is GCIQLCritic's objective (expectile V + twin-Q TD) with GCIQL's flat MLP head
replaced by the low-rank bilinear head

    V(s, g) = <phi(s),    psi(g)> / sqrt(d)
    Q(s,a,g) = <phi(s,a), psi(g)> / sqrt(d)

which is the "~90 family" of the bridge sweeps. The head is `GCBilinearPsiValue`, whose
psi_base='learned', project_z=False path is numerically identical to the vendored OGBench
`GCBilinearValue` locked at 1e-10 in tests/impls/test_gcbilinear.py.

The forward map is z-BLIND: phi never sees the task vector. (The z-indexed forward family
is a separate critic, Phase 1 #3.)

**This critic makes B3 reachable.** In the bridge reference (tools/wandb_mode_shim/
gciql_flow.py) `project_z` is a MODULE attribute that plain `GCBilinearValue` does not
declare, so the "hard sphere-Norm(z,B) + ortho on the 90 bilinear critic" cell (B3 of
untried-design-space.md) silently ran WITHOUT the sphere projection and could therefore
never actually be run (audit S2). Here project_z is a first-class field of the head module
and reaches both nets, so B3 is expressible.

Deviations from the bridge reference, each deliberate:
  * **project_z is SYMMETRIC** (audit S3). The reference passes project_z only to
    `critic_def`; `value_def` is built without it. Under actor_cond='z' with
    actor_z_from_critic=False that trains a sphered critic psi while the actor conditions
    on an UNSPHERED value psi — a silent train/condition mismatch. Ours applies it to both
    nets or to neither.
  * **No silent knob coercion.** Every combination the reference silently overwrites or
    ignores raises `IqlBilinearConfigError` at build time instead (see `_validate`).
  * **The actor's z carries no gradient.** The reference computes psi(goal) inside
    actor_loss under grad_params, so the actor's flow/BC loss trains the representation.
    On this seam the conditioning vector is produced by `sample_cond`, i.e. OUTSIDE
    value_and_grad, so psi stays purely IQL-trained. That matches the reference's own
    stated intent for `actor_vf_cond='z'` (which stop_gradients the z) and removes an
    otherwise-undeclared coupling between the actor loss and the critic representation.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCBilinearPsiValue, sphere_project


class IqlBilinearConfigError(ValueError):
    """Raised for an IqlBilinear knob combination that would be a silent no-op or a silent
    override. Named (not a bare assert) so callers can distinguish it from ComboError."""


PSI_BASES = ("learned", "frozen", "identity")
ACTOR_CONDS = ("goal", "z")


@dataclasses.dataclass(frozen=True)
class IqlBilinearCritic:
    value: Any
    critic: Any
    discount: float
    expectile: float
    tau: float
    obs_dim: int
    latent_dim: int
    psi_base: str
    actor_cond: str
    actor_z_from_critic: bool
    actor_z_infer: bool
    iql_ortho_coef: float
    project_z: bool
    double_q: bool

    GROUPS = ("value", "critic")
    STAGES = ("iql",)
    PROVIDES_V = True
    # Declared contract. Note that sample_cond ALSO emits "z" when actor_cond='z' (or
    # actor_z_infer) — but only then, so it cannot be promised at class level and is
    # deliberately not advertised. FlowBCActor hard-requires "z" unconditionally (it needs a
    # latent it can bootstrap and infer, not a per-batch psi(goal) readout), so
    # iql_bilinear x flowbc correctly raises ComboError, exactly as gciql x flowbc does.
    PROVIDES_COND = frozenset({"value_goal", "goal"})
    # infer_cond here is a fixed zero placeholder that the eval goal path overrides, so it
    # never reads `obs`. Declared rather than defaulted -- see base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "unused"

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def _z_cond(self):
        """Whether the actor conditions on the task vector psi(g) instead of the raw goal.
        actor_z_infer implies it: the inferred z lives in psi-space (reference L1513)."""
        return self.actor_cond == "z" or self.actor_z_infer

    @property
    def cvec_dim(self):
        return self.latent_dim if self._z_cond else self.obs_dim

    def stage_groups(self, stage):
        assert stage == "iql"
        return self.GROUPS

    def _V(self, p, obs, goal):
        return self.value.apply({"params": p}, obs, goal)             # (B,)

    def _Q(self, p, obs, goal, actions):
        return self.critic.apply({"params": p}, obs, goal, actions)   # (2, B)

    def _psi_value(self, p, goals):
        # psi depends only on the goal argument; calling psi_of directly avoids fabricating
        # a phi placeholder (which the reference has to do, at goals-as-observations).
        return self.value.apply({"params": p}, goals, method=GCBilinearPsiValue.psi_of)

    def _psi_critic(self, p, goals):
        psi = self.critic.apply({"params": p}, goals, method=GCBilinearPsiValue.psi_of)
        if self.psi_base == "identity":
            return psi                      # param-free, unensembled already
        # critic is ensembled [2,B,d] -> head-mean [B,d]. Under project_z the two heads are
        # each on the sqrt(d) sphere but their MEAN is inside it; the reference does the
        # same, and it is recorded in the seam test rather than silently corrected.
        return psi.mean(axis=0)

    def _actor_z(self, params, goals):
        """The task vector the actor conditions on. D1 (`actor_z_from_critic`) reads the
        CRITIC's psi — the leg the TD (and the ortho penalty) actually train — instead of
        the value net's."""
        if self.actor_z_from_critic:
            return self._psi_critic(params["critic"], goals)
        return self._psi_value(params["value"], goals)

    def _infer_z(self, psi, rng):
        """exp4: the INFERRED task vector z = E[r psi(s')] rather than the exact psi(g).
        Random sparse reward over the batch's goal embeddings (own goal + ~10% of the
        others), then sphere-Norm to sqrt(d). Verbatim from the reference's _infer_actor_z."""
        B = psi.shape[0]
        mask = (jax.random.uniform(rng, (B, B)) < 0.1).astype(psi.dtype) + jnp.eye(B, dtype=psi.dtype)
        return sphere_project(mask @ psi, self.latent_dim)

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        # `>= 0` tie-break, matching OGBench/GCIQLCritic (NOT ISFB's strict `> 0`).
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    def _ortho(self, p, goals):
        """FB orthonormality ||E[psi psi^T] - I|| on the CRITIC's goal leg, with no measure
        bootstrap — the "bilinear GCIQL + ortho only, nothing else from FB" cell. Per-head
        formula, then head-mean, matching the reference's ortho_loss exactly."""
        psi = self.critic.apply({"params": p}, goals, method=GCBilinearPsiValue.psi_of)  # [2,B,d]
        B = psi.shape[1]
        off = 1.0 - jnp.eye(B)
        cov = jnp.einsum("hid,hjd->hij", psi, psi)                                       # [2,B,B]
        o_off = 0.5 * jnp.sum((cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(cov, axis1=1, axis2=2), axis=1)
        return jnp.mean(o_off + o_diag)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "iql"
        del boot                            # scalar IQL never bootstraps through an actor sample
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        # value loss (expectile): grad -> value; twin-Q from the frozen target.
        # double_q=False ablates the clipped double-Q to the single head q1.
        q1, q2 = self._Q(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        # critic loss (TD): grad -> critic; next_v uses stop-grad online value (as GCIQL).
        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        cq1, cq2 = self._Q(params["critic"], obs, vg, action)
        critic_loss = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()

        loss = value_loss + critic_loss
        info = {"value_loss": value_loss, "critic_loss": critic_loss,
                "v_mean": v.mean(), "q_target_mean": tq.mean()}

        # iql_ortho_coef is a Python float, so this branch is static under jit.
        if self.iql_ortho_coef > 0.0:
            orth = self._ortho(params["critic"], vg)
            loss = loss + self.iql_ortho_coef * orth
            info["ortho_loss"] = orth
        return loss, info

    def v_fn(self, params, obs, cond):
        return self._V(params["value"], obs, cond["goal"])

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad accepted for seam uniformity; there is no separate encoder here.
        q1, q2 = self._Q(params["critic"], obs, cond["goal"], actions)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        return cond["z"] if self._z_cond else cond["goal"]

    def sample_cond(self, net, batch, rng):
        cond = {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}
        if self._z_cond:
            params = {"value": net["value"].params, "critic": net["critic"].params}
            z = self._actor_z(params, batch["actor_goals"])
            if self.actor_z_infer:
                z = self._infer_z(z, rng)
            cond["z"] = z                   # constant w.r.t. every loss (see module docstring)
        return cond

    def map_eval_cond(self, net, cond):
        """Eval-time hook: turn the env's raw task goal into the conditioning vector the
        actor was TRAINED on. Under actor_cond='goal' this is the identity; under
        actor_cond='z' the actor's input is psi(goal), which cannot be recovered from `cond`
        alone (it needs the critic/value params), hence the hook.

        actor_z_infer is deliberately NOT applied here: the inferred z is a training-time
        noise model over a batch of goal embeddings; at eval the exact embedding is the
        right conditioning (that is what the inference experiment measures against)."""
        if not self._z_cond or "goal" not in cond:
            return cond
        params = {"value": net["value"].params, "critic": net["critic"].params}
        return {**cond, "z": self._actor_z(params, cond["goal"])}

    def eval_cond0(self, ex_obs, config):
        cond = {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        if self._z_cond:
            cond["z"] = jnp.zeros((self.latent_dim,), jnp.float64)
        return cond

    def infer_cond(self, net, obs, rewards):
        # GC agents condition on the env's task goal (set via sample_actions(goals=)), not
        # an inferred latent; return the zero placeholder the eval goal-path overrides.
        return self.eval_cond0(obs, None)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _validate(config, goal_dim):
        """Every check here is a place the bridge reference silently overrides or silently
        ignores a knob. Raising is the deliverable."""
        psi_base = str(config["psi_base"])
        actor_cond = str(config["actor_cond"])
        d = int(config["value_latent_dim"])
        project_z = bool(config["project_z"])
        ortho = float(config["iql_ortho_coef"])

        if psi_base not in PSI_BASES:
            raise IqlBilinearConfigError(
                f"psi_base must be one of {PSI_BASES}, got {psi_base!r}")
        if actor_cond not in ACTOR_CONDS:
            raise IqlBilinearConfigError(
                f"actor_cond must be one of {ACTOR_CONDS}, got {actor_cond!r}")

        # S12: psi(g)=g is param-free, so the rank is PINNED to the goal dim. The reference
        # asserts this too; we keep it because silently overriding d would make a d-sweep
        # over the identity base secretly a sweep over nothing.
        if psi_base == "identity" and d != goal_dim:
            raise IqlBilinearConfigError(
                f"psi_base='identity' pins value_latent_dim to the goal dim ({goal_dim}); "
                f"got value_latent_dim={d}. d is not a free rank on the identity base (audit S12).")

        # S6: in the reference actor_z_from_critic is only ever consulted from the z paths,
        # so under actor_cond='goal' it is a SILENT no-op — the config comment says it needs
        # actor_cond='z' but nothing enforces it, and sweeps over it did nothing.
        if bool(config["actor_z_from_critic"]) and actor_cond != "z":
            raise IqlBilinearConfigError(
                "actor_z_from_critic=True is a silent no-op unless actor_cond='z' "
                "(audit S6): with actor_cond='goal' the actor conditions on the raw goal and "
                "never reads any psi. Set actor_cond='z' or actor_z_from_critic=False.")

        # S2-adjacent: the identity base has no parameters on the goal leg, so both the
        # sphere projection and the orthonormality penalty would be silently inert.
        if project_z and psi_base == "identity":
            raise IqlBilinearConfigError(
                "project_z=True is inert with psi_base='identity': psi(g)=g is param-free, so "
                "sphering it changes the readout scale without any trainable leg to shape "
                "(and the reference's GCIdentityBValue does not even declare the attribute).")
        if ortho > 0.0 and psi_base == "identity":
            raise IqlBilinearConfigError(
                "iql_ortho_coef>0 is a silent no-op with psi_base='identity': the penalty is "
                "computed on psi(g)=g, which carries no parameters, so its gradient is zero.")
        if ortho > 0.0 and psi_base == "frozen":
            raise IqlBilinearConfigError(
                "iql_ortho_coef>0 is a silent no-op with psi_base='frozen': psi is "
                "stop_gradient'd, so the orthonormality penalty produces no gradient at all.")

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        goal_dim = int(ex_obs.shape[-1])          # state-based GC: goals are observations
        cls._validate(config, goal_dim)

        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])
        psi_base = str(config["psi_base"])
        project_z = bool(config["project_z"])
        # SYMMETRIC head kwargs (audit S3): value and critic differ ONLY in `ensemble`.
        head_kw = dict(hidden_dims=hd, latent_dim=d, layer_norm=ln,
                       psi_base=psi_base, project_z=project_z,
                       psi_layer_norm=config["psi_layer_norm"],
                       output_scale=bool(config.get("output_scale", False)))
        value = GCBilinearPsiValue(ensemble=False, **head_kw)
        critic = GCBilinearPsiValue(ensemble=True, **head_kw)

        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc = jr.split(jr.PRNGKey(config["seed"] + 5000), 2)
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "critic": critic.init(rc, ex_obs, ex_goal, ex_actions)["params"],
        }
        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(value=value, critic=critic,
                  discount=float(config["discount"]), expectile=float(config["expectile"]),
                  tau=float(config["tau"]), obs_dim=goal_dim, latent_dim=d,
                  psi_base=psi_base, actor_cond=str(config["actor_cond"]),
                  actor_z_from_critic=bool(config["actor_z_from_critic"]),
                  actor_z_infer=bool(config["actor_z_infer"]),
                  iql_ortho_coef=float(config["iql_ortho_coef"]),
                  project_z=project_z, double_q=bool(config["double_q"]))
        return obj, params, targets
