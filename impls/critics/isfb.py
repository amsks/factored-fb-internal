"""ISFBCritic — SG-ISFB (in-sample successor-measure FB) on the factored seam.

Standard FB trains the successor measure M with a policy-evaluation backup that bootstraps
through an out-of-distribution a' ~ pi_z(.|s'). ISFB replaces that with IQL's in-sample
improvement operator, applied AT THE MEASURE LEVEL, by adding exactly one module: a value
forward map F_V(s, z) -> R^z_dim (no action argument) giving

    N_ij = <F_V(s_i, z_i), B(s'_j)>     (value successor measure; SHARES B with M)
    V_z(s) = <F_V(s,z), z>              (scalar state value)

so F_V : N :: F : M, exactly as V : Q in IQL, and the backup becomes
    M(s,a,.;z)  <-  delta_{s'} + gamma N^-(s', .; z)      (no a'~pi anywhere).

Math transcribed from the torch reference agents/fb/isfb/agent.py::update_fb, which the
repo treats as upstream ground truth for FB-family agents. Deliberate deviations, one only:
the bootstrap mask (see `goal_mask` below) follows OGBench/GCIQL semantics.

Sharp edges preserved from torch (each is a documented disagreement with the jax shim in
tools/wandb_mode_shim/isfb_flow.py; torch wins all but the mask):
  * NO 1/sqrt(d) on M, N, q, v — keeps ISFB's q on the same scale as FBCritic's q, so
    alpha_n/alpha_m/ortho_coef mean the same thing across the FB family.
  * L_M is head-SUMMED (off-diag summed over all P heads / |off|; diag x P), identical to
    FBCritic — hence the explicit `* P` on L_V and L_N, which exists solely to put them on
    that same head-summed scale. Dropping it silently reweights the objective by P.
  * The target measure is reduced across heads by targets_uncertainty (mean minus
    fb_pessimism_penalty * mean pairwise head spread) into a SINGLE shared [B,B] target.
  * AWR weights use the STD-NORMALIZED advantage (torch's fix for <F,z> inflating over
    training and re-saturating a fixed beta); `awr_normalize_adv=False` restores the raw
    paper form.
  * `coord` mode DROPS L_V entirely (isfb.tex's boxed L_Coord-ISFB has no L_V term) and
    reports it as 0.0. The jax shim always adds it — that is a bug for the coord baseline.
  * The expectile weight uses a STRICT `> 0` tie-break (torch), NOT gciql.py's `>= 0`.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


class IsfbConfigError(ValueError):
    """Raised for an invalid ISFB knob at BUILD time (matching the *ConfigError discipline in
    iql_bilinear.py / sf_feature.py), so a bad config fails at Agent.create rather than dying
    on a bare ValueError deep inside stage_loss at trace time."""


MEASURE_INSAMPLE = ("scalar_gate_hard", "scalar_gate_awr", "coord")


def _expectile_loss(diff, expectile):
    """Asymmetric L2. Strict `> 0` tie-break, matching agents/fb/agent.py::expectile_loss.
    (Do NOT reuse GCIQLCritic._expectile_loss: it uses `>= 0`.)"""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff ** 2).mean()


@dataclasses.dataclass(frozen=True)
class ISFBCritic:
    forward: Any
    value_forward: Any
    backward: Any
    left_encoder: Any
    z_dim: int
    action_dim: int
    num_parallel: int
    discount: float
    ortho_coef: float
    fb_pessimism_penalty: float
    train_goal_ratio: float
    norm_z: bool
    f_tau: float
    b_tau: float
    expectile: float
    measure_insample: str
    alpha_n: float
    alpha_m: float
    alpha_v: float
    awr_beta: float
    awr_wmax: float
    awr_normalize_adv: bool
    goal_mask: bool

    GROUPS = ("forward", "value_forward", "backward", "left_encoder")
    # ONE joint stage: all four losses come from a single forward pass over the same online
    # params and are summed into one scalar (torch takes exactly one loss.backward()).
    # Splitting into stages would CHANGE THE MATH — later stages would see already-stepped
    # params. Decoupling is achieved purely by stop_gradient.
    STAGES = ("isfb",)
    PROVIDES_V = True          # v = <F_V, z>, on exactly the same scale/units as q_fn
    PROVIDES_COND = frozenset({"z"})   # identical to FBCritic: ISFB generates z

    @property
    def TARGETS(self):
        # F_V rides the forward group's tau (torch soft-updates _target_value_forward_map
        # with f_target_tau and puts F_V's params in the forward optimizer => lr_f).
        return {"forward": self.f_tau, "value_forward": self.f_tau,
                "left_encoder": self.f_tau, "backward": self.b_tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def stage_groups(self, stage):
        assert stage == "isfb"
        return self.GROUPS

    # --- module apply helpers ---
    def _F(self, p, obs_feat, z, action):
        return self.forward.apply({"params": p}, obs_feat, z, action)

    def _Fv(self, p, obs_feat, z, zero_action):
        # F_V(s, z): the SAME ForwardMap architecture fed a zeroed action, so F_V is
        # action-independent by construction while sharing F's tower shape and left encoder.
        return self.value_forward.apply({"params": p}, obs_feat, z, zero_action)

    def _B(self, p, x):
        return self.backward.apply({"params": p}, x)

    def _le(self, p, x):
        return self.left_encoder.apply({"params": p}, x)

    def _terms(self, params, target, batch, cond):
        """Every intermediate the ISFB losses are built from. Shared by stage_loss and by
        debug_terms (which the parity test uses to localize a mismatch to one tensor)."""
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs                       # bw_encoder = Identity for state observations
        z = cond["z"]
        zero_a = jnp.zeros_like(action)

        # target value successor measure N^- at s' — NO a'~pi. This single substitution
        # (gamma M^-(s', a'~pi, .) -> gamma N^-(s', .)) is the whole structural change vs FB.
        nle = self._le(target["left_encoder"], next_obs)
        tFv = self._Fv(target["value_forward"], nle, z, zero_a)          # [P,B,z]
        tB = self._B(target["backward"], goal)                           # [B,z]
        tN = jnp.einsum("pbz,cz->pbc", tFv, tB)                          # [P,B,B]
        tmean, tunc = targets_uncertainty(tN, self.num_parallel)
        tN_M = jax.lax.stop_gradient(tmean - self.fb_pessimism_penalty * tunc)   # [B,B]

        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)                   # [P,B,z]
        Fv = self._Fv(params["value_forward"], le, z, zero_a)            # [P,B,z]
        Bm = self._B(params["backward"], goal)                           # [B,z]
        M = jnp.einsum("pbz,cz->pbc", Fs, Bm)                            # [P,B,B]
        N = jnp.einsum("pbz,cz->pbc", Fv, Bm)                            # [P,B,B]
        # scalar readouts, matching the flow actor's Q exactly (NO /sqrt(d)).
        q = (Fs * z).sum(-1)                                             # [P,B]
        v = (Fv * z).sum(-1)                                             # [P,B]
        return dict(le=le, Fs=Fs, Fv=Fv, B=Bm, M=M, N=N, tN=tN, tN_M=tN_M, q=q, v=v)

    def debug_terms(self, params, target, batch, cond):
        return self._terms(params, target, batch, cond)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "isfb"
        # `boot` is intentionally UNUSED: ISFB never samples a policy action. (torch asserts
        # next_action_override is None; reweighting / goal_cond / onestep are unsupported.)
        del boot
        t = self._terms(params, target, batch, cond)
        M, N, q, v = t["M"], t["N"], t["q"], t["v"]
        P = self.num_parallel
        tau = self.expectile
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        # (1) scalar gate: the ONLY action-selection operator. q is detached, so gradient
        # flows into F_V (and the shared left encoder) only. The `* P` puts L_V on
        # fb_successor_terms' head-summed scale so alpha_n/alpha_m stay balanced vs L_M.
        L_V = P * _expectile_loss(jax.lax.stop_gradient(q) - v, tau)
        A = jax.lax.stop_gradient(q - v).mean(0)                          # head-mean adv [B]

        # (2) improve the vector value measure N under the scalar gate. M is detached, so
        # gradient flows to F_V, B and the left encoder — not to F.
        Msg = jax.lax.stop_gradient(M)
        if self.measure_insample == "coord":
            L_N = P * _expectile_loss(Msg - N, tau)
        elif self.measure_insample in ("scalar_gate_hard", "scalar_gate_awr"):
            if self.measure_insample == "scalar_gate_awr":
                # torch Tensor.std() is UNBIASED (ddof=1); jnp.std defaults to ddof=0.
                An = A / (jnp.std(A, ddof=1) + 1e-6) if self.awr_normalize_adv else A
                wgt = jnp.clip(jnp.exp(self.awr_beta * An), 0.0, self.awr_wmax)
            else:
                wgt = jnp.where(A > 0, tau, 1.0 - tau)
            wbar = wgt / (wgt.mean() + 1e-6)
            L_N = P * (wbar.reshape(1, -1, 1) * (Msg - N) ** 2).mean()
        else:
            raise ValueError(f"unknown measure_insample {self.measure_insample!r}")

        # (3) measure TD: M bootstraps through the TARGET value measure at s'. The mask is
        # the one place we follow OGBench rather than torch: torch passes the env-termination
        # discount (never set on cube offline data, so effectively unmasked), while GCIQL's
        # mask = 1 - success cuts the bootstrap at the relabeled goal. Without it the goal
        # state collects reward forever and V/Q inflate until the actor collapses
        # (agents/fb/agent.py::iql_goal_target documents the failure at length). Our GCDataset
        # already emits `masks` in OGBench semantics.
        disc = self.discount * batch["masks"][:, None] if self.goal_mask else self.discount
        diff = M - disc * t["tN_M"]                                       # [P,B,B]
        # head-SUMMED off-diag / diag x P — byte-identical to FBCritic.stage_loss.
        fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        L_M = fb_off + fb_diag

        # (4) FB orthonormality on the single (non-ensembled) backward map — unchanged from FB.
        cov = t["B"] @ t["B"].T
        orth = 0.5 * jnp.sum((cov * off) ** 2) / off_sum - jnp.mean(jnp.diagonal(cov))

        loss = self.alpha_n * L_N + self.alpha_m * L_M + self.ortho_coef * orth
        if self.measure_insample != "coord":
            loss = loss + self.alpha_v * L_V
        else:
            L_V = jnp.zeros((), jnp.float64)   # reported as 0.0, as torch does

        return loss, {"isfb_loss": loss, "isfb_L_V": L_V, "isfb_L_N": L_N, "isfb_L_M": L_M,
                      "isfb_offdiag": fb_off, "isfb_diag": fb_diag, "ortho_loss": orth,
                      "adv_mean": A.mean(), "adv_std": jnp.std(A, ddof=1), "M_mean": M.mean()}

    def v_fn(self, params, obs, cond):
        """Scalar state value <F_V(s,z), z>, head-reduced exactly as q_fn — so an AWR actor's
        A = q - v is literally the ISFB scalar gate's advantage. ISFB is the first FB-family
        critic with PROVIDES_V=True."""
        z = cond["z"]
        le = self._le(params["left_encoder"], obs)
        zero_a = jnp.zeros(obs.shape[:-1] + (self.action_dim,), le.dtype)
        Fv = self._Fv(params["value_forward"], le, z, zero_a)
        Vs = (Fv * z).sum(-1)
        vmean, _ = targets_uncertainty(Vs, self.num_parallel)
        return vmean

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        z = cond["z"]
        le = self._le(params["left_encoder"], obs)
        if stop_encoder_grad:
            le = jax.lax.stop_gradient(le)
        Fs = self._F(params["forward"], le, z, actions)
        Qs = (Fs * z).sum(-1)
        qmean, _ = targets_uncertainty(Qs, self.num_parallel)
        return qmean, Qs

    def cond_vec(self, cond):
        """The vector actors concatenate onto obs to condition. ISFB (like FB): the latent z."""
        return cond["z"]

    def sample_cond(self, net, batch, rng):
        # Verbatim FBCritic.sample_cond — ISFB inherits FB's z-drawing unchanged.
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        r_z, r_mask, r_perm = jax.random.split(rng, 3)
        z_gauss = project_z(jax.random.normal(r_z, (B, self.z_dim)), self.norm_z)
        goals = project_z(
            self._B(net["backward"].params, next_obs)[jax.random.permutation(r_perm, B)],
            self.norm_z,
        )
        mix_mask = (jax.random.uniform(r_mask, (B,)) < self.train_goal_ratio)[:, None]
        z = jnp.where(mix_mask, goals, z_gauss)
        return {"z": z}

    def infer_cond(self, net, obs, rewards):
        Bm = self._B(net["backward"].params, obs)
        z = (rewards.reshape(1, -1) @ Bm).reshape(-1) / Bm.shape[0]
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        # Validate the enum at BUILD time (named error), not at stage_loss trace time.
        insample = str(config["measure_insample"])
        if insample not in MEASURE_INSAMPLE:
            raise IsfbConfigError(
                f"measure_insample must be one of {MEASURE_INSAMPLE}, got {insample!r}")
        z_dim, L_dim, P = config["z_dim"], config["L_dim"], config["num_parallel"]
        fwd_kw = dict(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"], num_parallel=P,
                      hidden_layers=config["forward"]["hidden_layers"],
                      embedding_layers=config["forward"]["embedding_layers"])
        forward = ForwardMap(**fwd_kw)
        value_forward = ForwardMap(**fwd_kw)      # IDENTICAL architecture, fed a zeroed action
        backward = PhiMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                          hidden_layers=config["backward"]["hidden_layers"],
                          norm=config["backward"]["norm"])
        left_encoder = PhiMap(z_dim=L_dim, hidden_dim=config["left_encoder"]["hidden_dim"],
                              hidden_layers=config["left_encoder"]["hidden_layers"],
                              norm=config["left_encoder"]["norm"])
        rf, rfv, rb, rle = jr.split(jr.PRNGKey(config["seed"]), 4)
        ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        params = {
            "left_encoder": left_encoder.init(rle, ex_obs)["params"],
            "backward": backward.init(rb, ex_obs)["params"],
            "forward": forward.init(rf, ex_feat, ex_z, ex_actions)["params"],
            "value_forward": value_forward.init(rfv, ex_feat, ex_z,
                                                jnp.zeros_like(ex_actions))["params"],
        }
        targets = {g: copy.deepcopy(params[g]) for g in cls.GROUPS}
        mode = str(config["measure_insample"])
        critic = cls(
            forward=forward, value_forward=value_forward, backward=backward,
            left_encoder=left_encoder, z_dim=z_dim, action_dim=int(ex_actions.shape[-1]),
            num_parallel=P,
            discount=float(config["discount"]), ortho_coef=float(config["ortho_coef"]),
            fb_pessimism_penalty=float(config["fb_pessimism_penalty"]),
            train_goal_ratio=float(config["train_goal_ratio"]), norm_z=bool(config["norm_z"]),
            f_tau=float(config["f_target_tau"]), b_tau=float(config["b_target_tau"]),
            expectile=float(config["iql_expectile"]), measure_insample=mode,
            alpha_n=float(config["alpha_n"]), alpha_m=float(config["alpha_m"]),
            # coord's objective (isfb.tex) contains no L_V; force alpha_v to 0 there so the
            # coord baseline is the paper's, whatever the config says.
            alpha_v=0.0 if mode == "coord" else float(config["alpha_v"]),
            awr_beta=float(config["awr_beta"]), awr_wmax=float(config["awr_wmax"]),
            awr_normalize_adv=bool(config["awr_normalize_adv"]),
            goal_mask=bool(config["goal_mask"]),
        )
        return critic, params, targets
