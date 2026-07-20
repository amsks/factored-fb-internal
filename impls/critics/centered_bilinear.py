"""CenteredBilinearCritic — behaviour-relative advantage on the factored seam.

Two training programs that share ONE concept — subtract a behaviour-mean baseline b(s) so
the actor maximises only the part of the value it can move through the action:

  mode='dueling'      : IN-ARCHITECTURE. Q(s,a,g) = U(s,g) + A_bil(s,a,g), with
                        A_bil = (phi(s,a) - b(s))·psi(g)/sqrt(d) (GCDuelingValue). Trained by
                        the SAME expectile-V + twin-Q TD as the plain bilinear critic, plus a
                        scale-normed regression L_b = ||b(s) - sg[phi(s,a_data)]||^2. The
                        actor's q-term is A_bil ALONE (not U — a state-goal cardinal the actor
                        cannot influence). This is the historical `value_arch='dueling'` cell.

  mode='centered_aux' : T10-a. Stock scalar IQL (expectile-V + twin-Q TD on a PLAIN bilinear
                        critic) grounds Q_full = phi·psi, and a centered-measure ΔM TD rides
                        on top as an auxiliary:
                            ΔM = (phi(s,a) - sg[b(s)])·psi(s')/sqrt(d),
                        an FB-style B×B measure residual with a π bootstrap (a'~π at s').

  mode='centered_only': T10-b. NO scalar TD anywhere. A behaviour-anchored BASE measure leg
                        (SARSA bootstrap on the dataset next action) grounds M = phi·psi,
                        while the ΔM leg keeps its π bootstrap — the "split bootstrap" design.

The dueling/centered losses are FIRST-PARTY research code (T10 spec / the dueling cell); no
external numerical reference exists (parity scout: Axis-3 at best). This port therefore locks
STRUCTURE + INVARIANTS, not numbers — chiefly the baseline-centering decomposition
Q = U + A_bil with A_bil built from (phi - b), which is an exact algebraic property.

PINNED (audit S11 — load-bearing "defaults" encoded as build-time asserts that RAISE, never
silent coercion): flow-family goal actor, actor conditioning on the raw goal (no z anywhere),
no vector-field conditioning, state observations. This critic has no z leg at all, so it
pairs with the goal-conditioned flow actor `fql` (and `ddpgbc`); it CANNOT pair with `flowbc`
(hard-requires cond['z']) or `awr` (needs a V function this critic does not expose) — both
raise ComboError.

DELIBERATE DEVIATIONS from tools/wandb_mode_shim/gciql_flow.py (each documented, none silent):

  * b(s) regresses to the DATASET behaviour action's feature sg[phi(s,a_data)] — a single,
    genuinely goal-agnostic sample of the behaviour policy μ — for BOTH the dueling and the
    centered modes. This is exactly what the reference's dueling `critic_loss` does (L558).
    The reference's centered_measure_loss instead averages K=center_b_samples samples of the
    *goal-agnostic BC flow* μ; that estimator is not reproducible on the factored seam, where
    the critic cannot roll the actor's flow and the seam's BC flow is goal-CONDITIONED. So
    `center_b_samples` is validated and PINNED to 1 (a value != 1 RAISES, rather than silently
    estimating something different).

  * S13 (out-of-band env coupling → explicit config key). The reference's centered_only +
    split-bootstrap SARSA base leg reads batch['next_actions'], available only if the run was
    launched with the env var OGBENCH_NEXT_ACTIONS=1. Here it is the CONFIG key
    `use_next_actions`: build() RAISES if the mode needs next-actions and the config doesn't
    grant them, and stage_loss RAISES loudly if the granted batch does not actually carry
    them. Never a silent dependency on a global env var.

  * `centered_scale_anchors` is part of the scale-control cluster the audit blames for 18/18
    crashed runs; the finite-loss tests exercise it on and off in every centered mode.
"""
import copy
import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp

from impls.utils.networks import (GCValue, GCBilinearValue, GCDuelingValue, MLP,
                                   ensemblize, sphere_project)


class CenteredBilinearConfigError(ValueError):
    """Raised for a CenteredBilinear knob combination the bridge reference would silently
    override, silently ignore, or couple to an out-of-band env var. Named (not a bare assert)
    so callers can distinguish it from ComboError."""


MODES = ("dueling", "centered_aux", "centered_only")


@dataclasses.dataclass(frozen=True)
class CenteredBilinearCritic:
    value: Optional[Any]        # GCValue expectile V (dueling / centered_aux); None for 'only'
    critic: Any                 # GCDuelingValue (dueling) or GCBilinearValue (centered_*)
    center_b: Optional[Any]     # ensembled MLP baseline b(s); None for 'dueling' (b is in critic)
    mode: str
    discount: float
    expectile: float
    tau: float
    obs_dim: int
    latent_dim: int
    double_q: bool
    dueling_b_coef: float
    centered_measure_coef: float
    center_b_coef: float
    centered_split_bootstrap: bool
    centered_scale_anchors: bool
    use_next_actions: bool
    GROUPS: tuple

    STAGES = ("centered",)
    PROVIDES_V = False          # the value net (when present) is an internal IQL leg, NOT an
                                # actor-facing V — exposing it would invite the AWR A_bil - V
                                # trap; awr is correctly rejected instead.
    PROVIDES_COND = frozenset({"value_goal", "goal"})   # goal-conditioned, no z (S11 pin)

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim     # actor conditions on the raw goal (no z leg exists)

    def stage_groups(self, stage):
        assert stage == "centered"
        return self.GROUPS

    # --- module apply helpers -------------------------------------------------
    def _V(self, p, obs, goal):
        return self.value.apply({"params": p}, obs, goal)                 # (B,)

    def _Q_dueling(self, p, obs, goal, actions, full_info=False):
        return self.critic.apply({"params": p}, obs, goal, actions, full_info=full_info)

    def _Q_bilinear(self, p, obs, goal, actions, info=False):
        return self.critic.apply({"params": p}, obs, goal, actions, info=info)  # (2,B)[,phi,psi]

    def _b(self, p, obs):
        return self.center_b.apply({"params": p}, obs)                    # [2,B,d]

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        # `>= 0` tie-break, matching OGBench/GCIQL (the scalar-TD convention here).
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff ** 2

    @staticmethod
    def _measure_terms(M, tM, gamma, scale_norm, eps=1e-6):
        """FB (offdiag, diag) TD terms over the last two (B×B) axes, head axes preserved.
        With scale_norm, divide by a stop-gradded measure scale so the gradient is
        dimensionless — the audit-20260706 fix that stops the unbounded <F,z> inflation from
        blowing up the shared critic. Transcribed from gciql_flow.py::_measure_terms."""
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        diff = M - gamma * tM
        offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
        diagt = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
        if scale_norm:
            msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
            offdiag = offdiag / (msq + eps)
            diagt = diagt / (jnp.sqrt(msq) + eps)
        return offdiag, diagt

    # --- losses ---------------------------------------------------------------
    def _dueling_loss(self, params, target, batch, cond):
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        # value loss (expectile): grad -> value; twin-Q Q=U+A_bil from the frozen target.
        q1, q2 = self._Q_dueling(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        # critic loss (TD on Q=U+A_bil) + the scale-normed baseline regression L_b. b is
        # trained to track sg[phi(s,a_data)] (the behaviour action's feature); the TD and L_b
        # share one optimizer, exactly as gciql_flow.py::critic_loss does under 'dueling'.
        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        out = self._Q_dueling(params["critic"], obs, vg, action, full_info=True)
        cq1, cq2 = out["Q"]
        critic_td = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()
        b_scale = jax.lax.stop_gradient(jnp.mean(out["phi"] ** 2)) + 1e-6
        b_loss = jnp.mean((out["bs"] - jax.lax.stop_gradient(out["phi"])) ** 2) / b_scale

        loss = value_loss + critic_td + self.dueling_b_coef * b_loss
        info = {"value_loss": value_loss, "critic_loss": critic_td,
                "dueling_b_loss": b_loss, "dueling_b_scale": b_scale,
                "dueling_A_mean": out["A_bil"].mean(), "dueling_U_mean": out["U"].mean(),
                "v_mean": v.mean(), "q_target_mean": tq.mean()}
        return loss, info

    def _iql_loss(self, params, target, batch, cond):
        """Stock expectile-V + twin-Q TD on the PLAIN bilinear critic (the centered_aux
        grounding). Identical structure to IqlBilinearCritic's iql stage."""
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        vg = cond["value_goal"]
        rewards, masks = batch["rewards"], batch["masks"]

        q1, q2 = self._Q_bilinear(target["critic"], obs, vg, action)
        q = jnp.minimum(q1, q2) if self.double_q else q1
        v = self._V(params["value"], obs, vg)
        value_loss = self._expectile_loss(q - v, q - v, self.expectile).mean()

        next_v = self._V(jax.lax.stop_gradient(params["value"]), next_obs, vg)
        tq = rewards + self.discount * masks * next_v
        cq1, cq2 = self._Q_bilinear(params["critic"], obs, vg, action)
        critic_loss = ((cq1 - tq) ** 2 + (cq2 - tq) ** 2).mean()
        return value_loss + critic_loss, {"value_loss": value_loss, "critic_loss": critic_loss,
                                          "v_mean": v.mean(), "q_target_mean": tq.mean()}

    def _centered_loss(self, params, target, batch, cond, boot):
        """The centered-measure ΔM leg (both aux and only), the baseline regression, and — in
        'only' — the behaviour-anchored BASE leg. b(s) is trained ONLY by its regression: it is
        stop-gradded on both sides of ΔM, so the measure TD sends it no gradient (matching the
        reference's split-gradient design)."""
        obs, acts, nobs = batch["observations"], batch["actions"], batch["next_observations"]
        d = self.latent_dim
        sqd = jnp.sqrt(d)
        gamma = self.discount
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        anchors = self.centered_scale_anchors

        # online critic legs: phi(s,a), psi(s') — test states = ρ (next-observations).
        _, phi, psi = self._Q_bilinear(params["critic"], obs, nobs, acts, info=True)   # [2,B,d]
        if anchors:
            psi = sphere_project(psi, d)          # hard ||psi||=sqrt(d) anchor (scale control)

        # b(s): behaviour-mean feature, regression to sg[phi(s,a_data)]. Scale-normed by
        # sg[mean(phi^2)] so the raw ||b-phi||^2 (which inflates as bilinear features grow)
        # stays a dimensionless O(1) relative error — the dueling-v1 lesson.
        phi_mu = jax.lax.stop_gradient(phi)                                            # [2,B,d]
        bs = self._b(params["center_b"], obs)                                          # [2,B,d]
        b_scale = jax.lax.stop_gradient(jnp.mean(phi_mu ** 2)) + 1e-6
        b_loss = jnp.mean((bs - phi_mu) ** 2) / b_scale
        b_fid = (jnp.mean(jnp.linalg.norm(phi_mu - jax.lax.stop_gradient(bs), axis=-1))
                 / (jnp.mean(jnp.linalg.norm(phi_mu, axis=-1)) + 1e-6))

        # ΔM leg: centered measure TD, π bootstrap a'~π at s' (the actor's one-shot sample).
        a_pi = boot["next_action"]
        _, tphi, tpsi = self._Q_bilinear(target["critic"], nobs, nobs, a_pi, info=True)
        if anchors:
            tpsi = sphere_project(tpsi, d)
        tb = jax.lax.stop_gradient(self._b(params["center_b"], nobs))    # online b at s' (no target b)
        dphi = phi - jax.lax.stop_gradient(bs)          # b trained ONLY by its regression
        dM = jnp.einsum("hid,hjd->hij", dphi, psi) / sqd                                # [2,B,B]
        tdM = jax.lax.stop_gradient(jnp.einsum("hid,hjd->hij", tphi - tb, tpsi) / sqd)
        dm_off, dm_diag = self._measure_terms(dM, tdM, gamma, scale_norm=True)
        dm_loss = jnp.mean(dm_off + dm_diag)

        info = {"b_loss": b_loss, "b_scale": b_scale, "b_fidelity": b_fid,
                "dm_loss": dm_loss, "dm_offdiag": jnp.mean(dm_off), "dm_diag": jnp.mean(dm_diag),
                "dm_msq": jnp.mean(dM ** 2),
                "phi_norm_sq": jnp.mean(jnp.sum(phi ** 2, axis=-1)),
                "psi_norm_sq": jnp.mean(jnp.sum(psi ** 2, axis=-1))}

        base_loss = jnp.zeros((), phi.dtype)
        if self.mode == "centered_only":
            if self.centered_split_bootstrap:
                # S13: the SARSA base leg reads the dataset next action. use_next_actions
                # granted it at build time; the batch must actually carry it — fail loudly if
                # not (a structural check on the batch dict, evaluated at trace time).
                assert "next_actions" in batch, (
                    "centered_only + centered_split_bootstrap needs batch['next_actions'] "
                    "(the dataset SARSA bootstrap on the base leg). use_next_actions=True was "
                    "set, but this batch does not carry next_actions — the dataset must emit "
                    "them (the factored seam's GCDataset does not by default). This is the "
                    "explicit config-keyed replacement for the reference's OGBENCH_NEXT_ACTIONS "
                    "env var (audit S13).")
                a_base = batch["next_actions"]
            else:   # non-contraction control: π bootstrap on the base leg too
                a_base = a_pi
            _, tphi_b, tpsi_b = self._Q_bilinear(target["critic"], nobs, nobs, a_base, info=True)
            if anchors:
                tpsi_b = sphere_project(tpsi_b, d)
            M = jnp.einsum("hid,hjd->hij", phi, psi) / sqd
            tM = jax.lax.stop_gradient(jnp.einsum("hid,hjd->hij", tphi_b, tpsi_b) / sqd)
            # anchors: RAW regression on the contractive SARSA leg (scale_norm off) — the one
            # place an absolute scale can be pinned without a scale-invariance runaway.
            base_off, base_diag = self._measure_terms(M, tM, gamma, scale_norm=not anchors)
            base_loss = jnp.mean(base_off + base_diag)
            info.update({"base_loss": base_loss, "base_offdiag": jnp.mean(base_off),
                         "base_diag": jnp.mean(base_diag), "base_msq": jnp.mean(M ** 2)})

        loss = base_loss + self.centered_measure_coef * dm_loss + self.center_b_coef * b_loss
        info["centered_loss"] = loss
        return loss, info

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "centered"
        if self.mode == "dueling":
            del boot                    # dueling bootstraps through the value net, never π
            return self._dueling_loss(params, target, batch, cond)
        if self.mode == "centered_aux":
            iql_loss, iql_info = self._iql_loss(params, target, batch, cond)
            cm_loss, cm_info = self._centered_loss(params, target, batch, cond, boot)
            return iql_loss + cm_loss, {**iql_info, **cm_info}
        # centered_only: NO scalar TD; the centered legs are the whole objective.
        return self._centered_loss(params, target, batch, cond, boot)

    # --- seam surface ---------------------------------------------------------
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad accepted for seam uniformity; there is no separate encoder here.
        goal = cond["goal"]
        if self.mode == "dueling":
            # the actor's q-term is A_bil ALONE (the behaviour-relative advantage), not the
            # full Q=U+A_bil: U is a state-goal cardinal the actor cannot move through a.
            out = self._Q_dueling(params["critic"], obs, goal, actions, full_info=True)
            q1, q2 = out["A_bil"]
        else:
            q1, q2 = self._Q_bilinear(params["critic"], obs, goal, actions)   # full Q=phi·psi
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
        # inferred latent; return the zero placeholder the eval goal-path overrides.
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    # --- build ----------------------------------------------------------------
    @staticmethod
    def _validate(config, goal_dim):
        """Every check raises where the bridge reference silently overrides / ignores a knob
        or smuggles an out-of-band dependency (audit S11/S13)."""
        mode = str(config["mode"])
        if mode not in MODES:
            raise CenteredBilinearConfigError(f"mode must be one of {MODES}, got {mode!r}")

        # S11 pins — RAISE on a conflicting request, never silently coerce.
        if str(config.get("actor_cond", "goal")) != "goal":
            raise CenteredBilinearConfigError(
                "CenteredBilinear pins actor_cond='goal' (no z leg exists — under 'only' the "
                f"value net, the only psi source, never trains). Got {config.get('actor_cond')!r}.")
        if str(config.get("actor_vf_cond", "none")) != "none":
            raise CenteredBilinearConfigError(
                "CenteredBilinear pins actor_vf_cond='none' (b(s) regresses on the "
                "goal-agnostic behaviour action; a goal/z-conditioned vf would mismatch).")
        if bool(config.get("actor_z_infer", False)):
            raise CenteredBilinearConfigError(
                "CenteredBilinear has no task vector z; actor_z_infer is incompatible.")

        # center_b_samples is PINNED to 1: this seam estimates E_{a~mu}[phi] from the single
        # dataset behaviour action, not K samples of the goal-agnostic BC flow (not reproducible
        # here). A value != 1 would silently estimate something different — so it raises.
        k = int(config["center_b_samples"])
        if k != 1:
            raise CenteredBilinearConfigError(
                f"center_b_samples must be 1 on the factored seam (got {k}). The behaviour-mean "
                "feature b(s) is estimated from the dataset action; the reference's K-flow-sample "
                "estimator needs the goal-agnostic BC flow, which the critic cannot roll here.")

        use_na = bool(config["use_next_actions"])
        split = bool(config["centered_split_bootstrap"])
        if mode == "dueling":
            # centered-only knobs would be silently ignored under dueling — reject non-defaults.
            if bool(config.get("centered_scale_anchors", False)):
                raise CenteredBilinearConfigError(
                    "centered_scale_anchors is a centered-measure knob and is inert under "
                    "mode='dueling' (which has its own always-on b-scale-norm).")
            if use_na:
                raise CenteredBilinearConfigError(
                    "use_next_actions is a centered_only knob and is inert under mode='dueling'.")
        else:
            # S13: the SARSA base leg (only + split) is the sole consumer of next_actions.
            needs_na = (mode == "centered_only" and split)
            if needs_na and not use_na:
                raise CenteredBilinearConfigError(
                    "mode='centered_only' with centered_split_bootstrap=True needs the dataset "
                    "SARSA next action on the base leg. Set use_next_actions=True AND ensure the "
                    "dataset emits batch['next_actions'] (explicit config key replacing the "
                    "reference's OGBENCH_NEXT_ACTIONS env var, audit S13); or set "
                    "centered_split_bootstrap=False for the π-bootstrap control.")
            if use_na and not needs_na:
                raise CenteredBilinearConfigError(
                    "use_next_actions=True is inert unless mode='centered_only' AND "
                    "centered_split_bootstrap=True (nothing else reads batch['next_actions']).")

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        goal_dim = int(ex_obs.shape[-1])          # state-based GC: goals are observations
        cls._validate(config, goal_dim)

        mode = str(config["mode"])
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])

        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc, rb = jr.split(jr.PRNGKey(config["seed"] + 5000), 3)
        params = {}

        if mode == "dueling":
            value = GCValue(hidden_dims=hd, layer_norm=ln, ensemble=False)
            critic = GCDuelingValue(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=True)
            center_b = None
            groups = ("value", "critic")
            params["value"] = value.init(rv, ex_obs, ex_goal)["params"]
            params["critic"] = critic.init(rc, ex_obs, ex_goal, ex_actions)["params"]
        else:
            critic = GCBilinearValue(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=True)
            center_b = ensemblize(MLP, 2)((*hd, d), activate_final=False, layer_norm=ln)
            params["critic"] = critic.init(rc, ex_obs, ex_goal, ex_actions)["params"]
            params["center_b"] = center_b.init(rb, ex_obs)["params"]
            if mode == "centered_aux":
                value = GCValue(hidden_dims=hd, layer_norm=ln, ensemble=False)
                groups = ("value", "critic", "center_b")
                params["value"] = value.init(rv, ex_obs, ex_goal)["params"]
            else:  # centered_only — no scalar TD, so no value net at all.
                value = None
                groups = ("critic", "center_b")

        targets = {"critic": copy.deepcopy(params["critic"])}
        obj = cls(
            value=value, critic=critic, center_b=center_b, mode=mode,
            discount=float(config["discount"]), expectile=float(config["expectile"]),
            tau=float(config["tau"]), obs_dim=goal_dim, latent_dim=d,
            double_q=bool(config["double_q"]),
            dueling_b_coef=float(config["dueling_b_coef"]),
            centered_measure_coef=float(config["centered_measure_coef"]),
            center_b_coef=float(config["center_b_coef"]),
            centered_split_bootstrap=bool(config["centered_split_bootstrap"]),
            centered_scale_anchors=bool(config["centered_scale_anchors"]),
            use_next_actions=bool(config["use_next_actions"]),
            GROUPS=groups,
        )
        return obj, params, targets
