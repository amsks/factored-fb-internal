"""SfFeatureCritic — per-sample successor-FEATURE vector TD on the factored seam.

This is the SF objective of raymondchua/simple_successor_features, NOT the successor-MEASURE
loss (the B×B density matrix M[i,j]=<phi_i, psi_j> that FB/ISFB train). phi(s,a) is the ONLY
learned leg; the goal leg is the SF *base* b, and the readout Q(s,a,g)=<phi(s,a), psi(g)>/√d
drives a goal-conditioned greedy actor. phi is trained by a per-sample vector regression:

    phi(s,a)  <-  b(s) + gamma * mask * phi_target(s', a'),   a' ~ pi(s', g)
    L_SF      =  mean_i || phi(s_i,a_i) - sg[ b(s_i) + gamma m_i phi_target(s'_i,a'_i) ] ||^2

The DECISIVE structural signature (locked in the seam tests): at gamma=0 the target collapses
to b(s), so L_SF == mean((phi - b(s))^2) — a per-sample regression a B×B measure matrix could
never equal. Math transcribed from tools/wandb_mode_shim/gciql_flow.py::sf_loss (the E3 grid).

VERIFICATION STANCE. sf_feature_td is first-party research code: there is NO external
(torch / vendored) reference for the vector-TD loss, so this critic is locked STRUCTURALLY —
invariants (the gamma=0 decomposition, the exact-zero-gradient of a fixed base, the
reward-regression additive split, per-term isolation) — not against a fabricated number. This
is the Phase 1 analogue of Phase 0's FQL. It is the honest way to ship an unreferenced loss.

RESEARCH CONTEXT. The E3 / sf_feature_td campaign's published reading was RETRACTED: the
identity/frozen-base ~0.09 was NOT a representation collapse — it is over-determined, because a
sparse goal indicator is simply not linearly expressible in identity/frozen features (r=<s,g>
is linear in s, whose argmax is not the goal). So this critic reproduces a retraction, and the
point is structural fidelity, not a headline number. The reward path (`sf_reward_regress`,
`GCLearnedBValue`) is exactly the corrected cell that makes the goal indicator linearly
readable in the base; the identity/frozen behaviour is the retracted story. Both are kept legible.

DEFECT NOT REPRODUCED — this critic is where it lives:
  * S4: the bridge does `config['actor_cond'] = 'goal'` (gciql_flow.py:1460), SILENTLY
    overwriting a user-supplied actor_cond='z'. Here actor conditioning is pinned to 'goal' by
    RAISING SfFeatureConfigError on any conflicting request — never by silent coercion. This is
    the single most important behavioural correction in this critic.
  * S11: sf_base (and the other load-bearing "defaults") are build-time asserts, not silent
    fallbacks — an out-of-range value or an unsupported combination raises here.

Deviations from the bridge, each deliberate and documented:
  * The bootstrap action a'~pi(s',g) arrives via the seam's `boot["next_action"]`, produced by
    the actor's sample_bootstrap conditioned on cond["goal"] (=actor_goals). The bridge instead
    computes a' inside sf_loss conditioned on value_goals. On the factored seam the critic never
    holds the actor, so it consumes the seam-standard bootstrap; the conditioning goal is
    actor_goals rather than value_goals. Structural-only, so this convention difference is noted
    rather than reconciled against a nonexistent numerical reference.
  * project_z is intentionally NOT exposed (it is not in the E3 knob set); keep the surface tight.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import GCSfBilinearValue


class SfFeatureConfigError(ValueError):
    """Raised for an SfFeature knob combination the bridge silently coerces or silently leaves
    degenerate. Named (not a bare assert) so callers can distinguish it from ComboError."""


SF_BASES = ("identity", "frozen", "learned")
SF_MASKS = ("goal", "none")
SF_BASE_INDICES = ("current", "next")


@dataclasses.dataclass(frozen=True)
class SfFeatureCritic:
    critic: Any
    discount: float
    tau: float
    obs_dim: int
    latent_dim: int
    sf_base: str
    sf_mask: str
    sf_base_index: str
    sf_joint_psi: bool
    sf_reward_regress: bool
    sf_reward_coef: float
    sf_reward_shift: float
    sf_ortho_coef: float
    sf_psi_ortho_coef: float
    sf_coef: float
    fb_forward_z: bool

    GROUPS = ("critic",)
    STAGES = ("sf",)
    PROVIDES_V = False          # SF-only: there is no value function, just the bilinear Q.
    # Actor conditioning is PINNED to 'goal' (S4). Like GCIQL/IqlBilinear the actor conditions
    # on the raw goal, so this critic never produces a "z" key — FlowBCActor (REQUIRES_COND={"z"})
    # correctly raises ComboError, and AWRActor (NEEDS_V) raises too (PROVIDES_V is False).
    PROVIDES_COND = frozenset({"value_goal", "goal"})
    # infer_cond here is a fixed zero placeholder that the eval goal path overrides, so it
    # never reads `obs`. Declared rather than defaulted -- see base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "unused"

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim         # actor conditions on the raw goal (pinned)

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def stage_groups(self, stage):
        assert stage == "sf"
        return self.GROUPS

    # --- module apply helper ---
    def _critic(self, p, obs, goals, actions, z_goals):
        """(v, phi, psi) from the SF bilinear head. z_goals is threaded only under fb_forward_z
        (None otherwise); the head ignores it when fb_forward_z is False."""
        return self.critic.apply({"params": p}, obs, goals, actions, z_goals, info=True)

    def _base(self, params, obs, next_obs, acts, zarg):
        """The immediate SF feature b(s) (or b(s') under sf_base_index='next').

        identity -> b=s (param-free), shape [1,B,d] to broadcast against phi's [2,B,d].
        frozen/learned -> b=psi(s): a second critic call reading only the psi leg. Stop-gradded
        UNLESS sf_joint_psi, in which case the TD residual trains psi through the base (held
        non-degenerate by sf_psi_ortho_coef; without it phi=0,psi=0 is a global min of the TD).
        """
        base_src = next_obs if self.sf_base_index == "next" else obs
        if self.sf_base == "identity":
            return base_src[None]                                   # [1,B,d], param-free
        _, _, b = self._critic(params["critic"], obs, base_src, acts, zarg)   # psi(base_src) [2,B,d]
        return b if self.sf_joint_psi else jax.lax.stop_gradient(b)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "sf"
        obs, acts, nobs = batch["observations"], batch["actions"], batch["next_observations"]
        vgoals = cond["value_goal"]
        gamma = self.discount
        # sf_mask='goal' -> batch['masks']=1[s!=g], an ABSORBING goal (Q=gamma^T, a first-hitting
        # value). 'none' -> m==1, no absorption (Q=sum_t gamma^t 1[s_t=g], a visitation DENSITY).
        mask = batch["masks"] if self.sf_mask == "goal" else jnp.ones_like(batch["masks"])
        # fb_forward_z: task-condition phi([s,a,z=psi(g)]); same task g on both sides of the backup.
        zarg = vgoals if self.fb_forward_z else None

        _, phi, _ = self._critic(params["critic"], obs, vgoals, acts, zarg)      # [2,B,d]
        b = self._base(params, obs, nobs, acts, zarg)

        # a' ~ pi(s', g): supplied by the seam's actor bootstrap (see module docstring). The
        # bootstrap is ALWAYS semi-gradient (target net + stop_gradient), matching FB.
        a_next = jax.lax.stop_gradient(jnp.clip(boot["next_action"], -1.0, 1.0))
        _, tphi, _ = self._critic(target["critic"], nobs, vgoals, a_next, zarg)  # [2,B,d]
        boot_term = gamma * mask[None, :, None] * jax.lax.stop_gradient(tphi)     # [2,B,d]
        # b carries gradient iff sf_joint_psi; boot is always sg. With joint=False this is the
        # plain sg[b + gamma*m*tphi] target.
        target_phi = b + boot_term
        td = jnp.mean((phi - target_phi) ** 2)
        loss = td
        info = {"sf_loss": td, "phi_absmax": jnp.max(jnp.abs(phi)),
                "target_absmax": jnp.max(jnp.abs(jax.lax.stop_gradient(target_phi)))}

        if self.sf_ortho_coef > 0.0:
            # anti-collapse aux reg: orthonormality ||E[phi phi^T]-I|| on the SF vector phi,
            # per-head then averaged. Same form as the measure/ISFB ortho.
            Bn = obs.shape[0]
            off = 1.0 - jnp.eye(Bn)
            Cov = jnp.einsum("hid,hjd->hij", phi, phi)                            # [2,B,B]
            o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
            orth = jnp.mean(o_off + o_diag)
            loss = loss + self.sf_ortho_coef * orth
            info["orth"] = orth

        if self.sf_psi_ortho_coef > 0.0:
            # FB orthonormality ||E_rho[psi psi^T]-I|| on the BASE/goal leg psi (B:=psi(s'), the
            # rho marginal). This keeps a jointly-trained psi non-degenerate: un-stop-gradding
            # the base makes phi=0,psi=0 a GLOBAL min of the TD, and -tr(C) turns the origin into
            # a strict local maximum (repelling). Distinct from sf_ortho_coef (which hits phi).
            _, _, psi_o = self._critic(params["critic"], obs, nobs, acts, zarg)   # psi(s') [2,B,d]
            Bn = obs.shape[0]
            off_p = 1.0 - jnp.eye(Bn)
            CovP = jnp.einsum("hid,hjd->hij", psi_o, psi_o)                        # [2,B,B]
            p_off = 0.5 * jnp.sum((CovP * off_p[None]) ** 2, axis=(1, 2)) / jnp.sum(off_p)
            p_diag = -jnp.mean(jnp.diagonal(CovP, axis1=1, axis2=2), axis=1)
            orth_psi = jnp.mean(p_off + p_diag)
            loss = loss + self.sf_psi_ortho_coef * orth_psi
            info["psi_ortho"] = orth_psi
            info["psi_norm_mean"] = jnp.mean(jnp.linalg.norm(psi_o, axis=-1))

        if self.sf_reward_regress:
            # E3 corrected cell: ground the base in the ACTUAL reward. Symmetric bilinear reward
            # model r_hat(s,g)=<psi(s), psi(g)>/sqrt(d) regressed on batch rewards — psi's ONLY
            # grad path (the TD target stop-grads b, the actor Q-term takes no critic grads). This
            # is what makes the goal indicator linearly readable in the base. CURRENT-state
            # convention (forced by GCDataset: successes = idxs==value_goal_idxs ties rewards AND
            # masks to the current index). sf_reward_shift escapes the psi=0 saddle of the
            # symmetric bilinear form (raw {-1,0} rewards vanish there; shift=1 gives {0,1}=the
            # goal INDICATOR, argmax-invariant to the raw ordering).
            _, _, psi_s = self._critic(params["critic"], obs, obs, acts, zarg)     # psi(s)  [2,B,d]
            _, _, psi_g = self._critic(params["critic"], obs, vgoals, acts, zarg)  # psi(g)  [2,B,d]
            r_hat = (psi_s * psi_g).sum(axis=-1) / jnp.sqrt(psi_s.shape[-1])       # [2,B]
            r_tgt = batch["rewards"] + self.sf_reward_shift
            rew = jnp.mean((r_hat - r_tgt[None]) ** 2)
            loss = loss + self.sf_reward_coef * rew
            info["sf_reward_loss"] = rew
            info["sf_rhat_mean"] = jnp.mean(r_hat)

        loss = self.sf_coef * loss
        info["sf_total"] = loss
        return loss, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad accepted for seam uniformity; there is no separate encoder here.
        goal = cond["goal"]
        zarg = goal if self.fb_forward_z else None        # z=psi(actor goal), matching the readout
        v, _, _ = self._critic(params["critic"], obs, goal, actions, zarg)   # v = Q per head [2,B]
        q1, q2 = v[0], v[1]
        return jnp.minimum(q1, q2), v                     # bridge readout: min over the 2 heads

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def infer_cond(self, net, obs, rewards):
        # GC agents condition on the env's task goal (set via sample_actions(goals=)), not an
        # inferred latent; return the zero placeholder the eval goal-path overrides.
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    # ------------------------------------------------------------------ build
    @staticmethod
    def _validate(config, goal_dim):
        """Each check is a place the bridge silently coerces (S4) or silently leaves a knob
        degenerate/unsignalled (S11 + the sf_feature_td guard block, gciql_flow.py:1419-1459).
        Raising a named error is the deliverable."""
        sf_base = str(config["sf_base"])
        sf_mask = str(config["sf_mask"])
        sf_base_index = str(config["sf_base_index"])
        actor_cond = str(config["actor_cond"])
        d = int(config["value_latent_dim"])
        joint = bool(config["sf_joint_psi"])
        regress = bool(config["sf_reward_regress"])
        psi_ortho = float(config["sf_psi_ortho_coef"])
        fb_forward_z = bool(config["fb_forward_z"])

        if sf_base not in SF_BASES:
            raise SfFeatureConfigError(f"sf_base must be one of {SF_BASES}, got {sf_base!r}")
        if sf_mask not in SF_MASKS:
            raise SfFeatureConfigError(f"sf_mask must be one of {SF_MASKS}, got {sf_mask!r}")
        if sf_base_index not in SF_BASE_INDICES:
            raise SfFeatureConfigError(
                f"sf_base_index must be one of {SF_BASE_INDICES}, got {sf_base_index!r}")

        # S4: the bridge FORCE-SETS actor_cond='goal' (gciql_flow.py:1460), silently discarding a
        # user-supplied 'z'. We pin by RAISING instead — never by coercion.
        if actor_cond != "goal":
            raise SfFeatureConfigError(
                f"SfFeatureCritic pins actor conditioning to 'goal'; got actor_cond={actor_cond!r}. "
                "The bridge reference silently overwrites a conflicting actor_cond (audit S4); this "
                "critic refuses instead. Remove the override or set actor_cond='goal'.")

        # S12: b(g)=g is param-free, so the rank is PINNED to the goal dim under the identity base.
        if sf_base == "identity" and d != goal_dim:
            raise SfFeatureConfigError(
                f"sf_base='identity' pins value_latent_dim to the goal dim ({goal_dim}); got "
                f"value_latent_dim={d}. d is not a free rank on the identity base (audit S12).")

        # A linear-in-s reward proxy on a FIXED base cannot express the goal indicator — the exact
        # confound the retraction is about. Refuse it rather than let a sweep run the broken cell.
        if regress and sf_base != "learned":
            raise SfFeatureConfigError(
                f"sf_reward_regress requires sf_base='learned' (a fixed {sf_base!r} base can't "
                "express the goal indicator: r=<s,g> is linear in s, argmax != goal).")

        # A learned base with no grad path is the psi=0 saddle: refuse a base that never trains.
        if sf_base == "learned" and not (regress or joint):
            raise SfFeatureConfigError(
                "sf_base='learned' requires sf_reward_regress=True or sf_joint_psi=True (else psi "
                "has no training signal and collapses to the psi=0 saddle).")

        if joint:
            if sf_base != "learned":
                raise SfFeatureConfigError(
                    f"sf_joint_psi requires sf_base='learned' (a fixed {sf_base!r} psi has no "
                    "gradient to receive through the base).")
            if regress:
                raise SfFeatureConfigError(
                    "sf_joint_psi and sf_reward_regress are mutually exclusive (two grad paths "
                    "into psi).")
            if psi_ortho <= 0.0:
                raise SfFeatureConfigError(
                    "sf_joint_psi requires sf_psi_ortho_coef>0 (else phi=0,psi=0 is a global "
                    "minimum of the SF vector TD and the base collapses).")
            if sf_mask == "goal" and not fb_forward_z:
                raise SfFeatureConfigError(
                    "sf_joint_psi with sf_mask='goal' requires fb_forward_z (a goal-dependent "
                    "mask needs a goal-conditioned phi).")

        # Mixing an arriving-state base with a current-index absorbing mask crosses conventions.
        if sf_base_index == "next" and sf_mask == "goal":
            raise SfFeatureConfigError(
                "sf_base_index='next' requires sf_mask='none' (arriving-state base + current-index "
                "absorbing mask mixes reward/mask/base conventions).")

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        goal_dim = int(ex_obs.shape[-1])          # state-based GC: goals are observations
        cls._validate(config, goal_dim)

        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])
        sf_base = str(config["sf_base"])
        fb_forward_z = bool(config["fb_forward_z"])
        # sf_base -> psi_base on the shared SF head (identity/frozen/learned name the same axis).
        critic = GCSfBilinearValue(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=True,
                                   psi_base=sf_base, fb_forward_z=fb_forward_z)

        ex_goal = jnp.zeros_like(ex_obs)
        ex_z = ex_goal if fb_forward_z else None
        rc = jr.PRNGKey(config["seed"] + 6000)
        params = {"critic": critic.init(rc, ex_obs, ex_goal, ex_actions, ex_z)["params"]}
        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(
            critic=critic, discount=float(config["discount"]), tau=float(config["tau"]),
            obs_dim=goal_dim, latent_dim=d, sf_base=sf_base,
            sf_mask=str(config["sf_mask"]), sf_base_index=str(config["sf_base_index"]),
            sf_joint_psi=bool(config["sf_joint_psi"]),
            sf_reward_regress=bool(config["sf_reward_regress"]),
            sf_reward_coef=float(config["sf_reward_coef"]),
            sf_reward_shift=float(config["sf_reward_shift"]),
            sf_ortho_coef=float(config["sf_ortho_coef"]),
            sf_psi_ortho_coef=float(config["sf_psi_ortho_coef"]),
            sf_coef=float(config["sf_coef"]), fb_forward_z=fb_forward_z)
        return obj, params, targets
