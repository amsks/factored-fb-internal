"""HILP — Hilbert-representation successor features (Park et al., ICML 2024), transcribed
from the OFFICIAL repo: github.com/seohongpark/HILP, `hilp_zsrl/url_benchmark/agent/sf.py`
(+ `ddpg.py` for the extractor), cloned to ~/git/baselines/HILP.

WHY THE OFFICIAL REPO AND NOT A LOCAL COPY. Two other HILP implementations sit in
~/git/baselines (`RLU/controllable_agent/.../hilp.py`, Meta's reimplementation, and
`td_jepa/metamotivo/agents/hilp/`). Porting from a reimplementation is exactly what
produced the RLDP run-1 failure: the seam faithfully transcribed td_jepa's RLDP, which is
a structurally different algorithm from the official one, and the difference only surfaced
after 12 GPU-hours. See docs/notes/2026-08-28-official-parity-campaign-1.md.

THE FIVE TRANSCRIPTION TRAPS (docs/notes/2026-08-28-hilp-paper-eval.md records the audit):

  1. THREE reductions in one phi loss. `adv` is built from min(next_v1, next_v2) but
     subtracts the MEAN of the two TARGET heads; the two regression targets q1/q2 use
     next_v1/next_v2 SEPARATELY. Collapsing any of these to a single convention changes
     the objective.
  2. `hilp_expectile` ships at 0.5 (SFAgentConfig), NOT the 0.7 default in
     `expectile_loss`'s own signature. At 0.5 the weight is where(adv>=0, .5, .5) -- a
     CONSTANT -- so the IQL asymmetry is numerically INERT at the published setting and
     the loss is plain 0.5*diff^2. `adv` is still computed and logged. A port that
     "corrects" this to 0.7 is not running the published configuration.
  3. `feature_net` subtracts an EMA running mean from phi1 (0.995/0.005). `running_std` is
     maintained upstream in the same place and never read; carried here for fidelity and
     for the diagnostic, not used in any expression.
  4. The SF target uses the ONLINE feature net, detached -- NOT a target phi. The target
     phi nets exist only inside the phi value loss. (Compare the RLDP bug, where reading
     the target rather than the online backward tower diverged the run: same class of
     mistake, opposite direction.)
  5. `q_loss: True` by default, so the SF loss regresses the SCALAR Q = <F, z>, not the
     vector F; and `next_F` is selected PER SAMPLE by argmin over Q
     (`where(next_Q1 < next_Q2, next_F1, next_F2)`) -- a whole-vector choice, not an
     elementwise min.

STATUS: the phi learner below is transcribed and pinned by an independent numpy
reconstruction (tests/impls/test_hilp_seam.py). The SF layer and the extractor are
transcribed but NOT yet pinned by a parity test, and this critic is deliberately NOT
registered in impls/agents/__init__.py::CRITICS until they are -- an unregistered critic
cannot be selected by --critic and so cannot silently produce a number.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


def hilp_value(phi_a, phi_b):
    """v = -||phi(a) - phi(b)||, with the reference's clamp INSIDE the sqrt.

    `-torch.sqrt(torch.clamp(squared_dist, min=1e-6))` (sf.py:139-142). Clamping the
    SQUARE, not the root, matters: it floors v at -1e-3, and it keeps the gradient finite
    at phi(a) == phi(b), which is exactly the case the hindsight-goal relabelling makes
    common (obs and goal coincide whenever the sampled future is the current step)."""
    sq = jnp.sum((phi_a - phi_b) ** 2, axis=-1)
    return -jnp.sqrt(jnp.clip(sq, 1e-6, None))


def expectile_loss(adv, diff, expectile):
    """weight * diff^2 with weight = expectile if adv >= 0 else 1 - expectile (sf.py:150).

    NOTE the shipped expectile is 0.5, which makes the weight a constant 0.5 and this an
    ordinary squared error -- see trap 2 in the module docstring."""
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * diff ** 2


def hilp_phi_loss(phi1_p, phi2_p, tphi1_p, tphi2_p, apply_phi,
                  obs, goals, next_obs, discount, expectile):
    """The official HILP feature loss, sf.py:154-198, as a pure function.

    Split out of the critic so an independent numpy reconstruction can call exactly this
    with exactly these arrays (tests/impls/test_hilp_seam.py). `apply_phi(params, x)`
    applies one phi tower.

    The upstream `forward()` also soft-updates the target phis and the running mean as
    SIDE EFFECTS after computing the loss. Those are not side effects here -- the seam
    polyaks targets in Agent.apply_update and the running mean in `update_stats`, both
    strictly after the gradient step, which preserves the upstream ordering.
    """
    same = (jnp.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(obs.dtype)
    masks = 1.0 - same
    rewards = same - 1.0

    # target heads, stop-gradded (upstream: is_target=True detaches)
    tn1 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi1_p, next_obs), apply_phi(tphi1_p, goals)))
    tn2 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi2_p, next_obs), apply_phi(tphi2_p, goals)))
    tv1 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi1_p, obs), apply_phi(tphi1_p, goals)))
    tv2 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi2_p, obs), apply_phi(tphi2_p, goals)))

    # trap 1: min for adv's q, MEAN of the target heads for its baseline, per-head for
    # the regression targets. Three different reductions, deliberately.
    next_v = jnp.minimum(tn1, tn2)
    q = rewards + discount * masks * next_v
    v_t = (tv1 + tv2) / 2.0
    adv = q - v_t

    q1 = rewards + discount * masks * tn1
    q2 = rewards + discount * masks * tn2
    v1 = hilp_value(apply_phi(phi1_p, obs), apply_phi(phi1_p, goals))
    v2 = hilp_value(apply_phi(phi2_p, obs), apply_phi(phi2_p, goals))

    loss = (jnp.mean(expectile_loss(adv, q1 - v1, expectile))
            + jnp.mean(expectile_loss(adv, q2 - v2, expectile)))
    info = {"hilp_value_loss": loss, "hilp_v_mean": jnp.mean((v1 + v2) / 2.0),
            "hilp_adv_mean": jnp.mean(adv), "hilp_abs_adv_mean": jnp.mean(jnp.abs(adv)),
            "hilp_accept_prob": jnp.mean((adv >= 0).astype(loss.dtype))}
    return loss, info


@dataclasses.dataclass(frozen=True)
class HILPCritic:
    """HILP as a seam critic: a Hilbert phi learner + a successor-feature layer.

    ARCHITECTURE NOTE. The seam's `PhiMap` is already exactly HILP's phi tower --
    `mlp(obs_dim, hidden, "ntanh", hidden, "relu", feature_dim)` is Dense -> LayerNorm ->
    tanh -> Dense -> relu -> Dense, which is PhiMap's body verbatim (its docstring even
    calls the first block "ntanh"). The one setting that matters is `norm=False`: HILP's
    phi layer list ends at `feature_dim` with no trailing "L2", unlike its `Identity`
    feature learner which does.
    """
    phi1: Any
    phi2: Any
    sf: Any
    z_dim: int
    num_parallel: int
    discount: float
    hilp_discount: float
    hilp_expectile: float
    phi_tau: float
    sf_tau: float
    norm_z: bool
    stats_momentum: float
    q_loss: bool

    GROUPS = ("phi1", "phi2", "sf_psi", "hilp_stats")
    STAGES = ("phi", "sf")
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"z"})
    # z is solved by least squares against phi(s'), so the readout must be posed on the
    # same rows the SF target's phi is read at (`feature_net(next_obs)`, sf.py:614).
    INFER_COND_MARGINAL = "next_observations"
    BOOT_COND = "actor_goal"

    @property
    def TARGETS(self):
        # `hilp_stats` is a params-only slot (the running mean), never polyaked -- it is
        # written by update_stats. Giving it tau 0 keeps it present without drifting.
        return {"phi1": self.phi_tau, "phi2": self.phi_tau,
                "sf_psi": self.sf_tau, "hilp_stats": 0.0}

    @property
    def cvec_dim(self):
        return self.z_dim

    def stage_groups(self, stage):
        # Upstream runs two separate .backward() calls: the phi learner's value loss steps
        # only the phi towers, and the SF loss detaches phi entirely (it reads
        # `feature_net(...).detach()`), so no phi gradient flows from the SF stage.
        if stage == "phi":
            return ("phi1", "phi2")
        assert stage == "sf", stage
        return ("sf_psi",)

    # --- towers -------------------------------------------------------------
    def _phi(self, p, x):
        return self.phi1.apply({"params": p}, x)

    def _phi2(self, p, x):
        return self.phi2.apply({"params": p}, x)

    def _sf(self, p, obs, z, action):
        return self.sf.apply({"params": p}, obs, z, action)

    def _feature_net(self, net_or_params, stats_params, x):
        """phi1(x) - running_mean (sf.py:120-124).

        TRAP 3: the running mean is subtracted at READ time, so every consumer of the
        feature -- the SF target and the z readout both -- sees the centered feature,
        while the phi VALUE loss uses the raw towers. `running_std` is maintained upstream
        beside it and never read; it is kept in the stats slot for fidelity and as a
        diagnostic, not used in any expression."""
        return self._phi(net_or_params, x) - stats_params["mean"]

    # --- losses -------------------------------------------------------------
    def stage_loss(self, stage, params, target, batch, cond, boot):
        if stage == "phi":
            # Goals are the hindsight future observations the dataset supplies.
            return hilp_phi_loss(
                params["phi1"], params["phi2"], target["phi1"], target["phi2"],
                lambda p, x: self.phi1.apply({"params": p}, x),
                batch["observations"], batch["actor_goals"], batch["next_observations"],
                self.hilp_discount, self.hilp_expectile)

        assert stage == "sf", stage
        obs, action = batch["observations"], batch["actions"]
        nobs, z = batch["next_observations"], cond["z"]
        P = self.num_parallel

        tF = self._sf(target["sf_psi"], nobs, z, boot["next_action"])        # [P, B, d]
        # TRAP 4: the SF target reads the ONLINE feature net, detached -- NOT a target phi.
        # The target phi towers exist only inside the phi value loss. (The RLDP run-1
        # divergence was the mirror image of this: reading the target tower where official
        # reads the online one.)
        target_phi = jax.lax.stop_gradient(
            self._feature_net(params["phi1"], params["hilp_stats"], nobs))   # [B, d]
        # TRAP 5b: next_F is chosen PER SAMPLE by argmin over Q -- the whole vector of the
        # head with the smaller Q, not an elementwise min over the two vectors.
        tQ = jnp.einsum("pbd,bd->pb", tF, z)                                 # [P, B]
        pick = (tQ[0] < tQ[1])[:, None]                                      # [B, 1]
        next_F = jnp.where(pick, tF[0], tF[1])                               # [B, d]
        target_F = jax.lax.stop_gradient(target_phi + self.discount * next_F)

        F = self._sf(params["sf_psi"], obs, z, action)                       # [P, B, d]
        if self.q_loss:
            # TRAP 5a: with q_loss (the shipped default) the regression is on the SCALAR
            # Q = <F, z>, not on the vector F.
            Q = jnp.einsum("pbd,bd->pb", F, z)
            tQs = jnp.einsum("bd,bd->b", target_F, z)
            sf_loss = jnp.sum(jnp.mean((Q - tQs[None]) ** 2, axis=-1))
        else:
            sf_loss = jnp.sum(jnp.mean((F - target_F[None]) ** 2, axis=(-1, -2)))
        return sf_loss, {"sf_loss": sf_loss, "target_F": jnp.mean(target_F),
                         "phi_norm": jnp.mean(jnp.linalg.norm(target_phi, axis=-1)),
                         "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1))}

    # --- control ------------------------------------------------------------
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        z = cond["z"]
        Qs = jnp.einsum("pbd,bd->pb", self._sf(params["sf_psi"], obs, z, actions), z)
        # Upstream update_actor: Q = min(Q1, Q2), then actor_loss = -Q.mean() -- a pure
        # TD3 objective with no BC term, exactly as official RLDP's extractor.
        qmin = jnp.min(Qs, axis=0)
        return qmin, Qs

    def cond_vec(self, cond):
        return cond["z"]

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,))}

    def sample_cond(self, net, batch, rng):
        """z ~ sqrt(d) * normalize(gaussian) (sf.py::sample_z)."""
        B = batch["observations"].shape[0]
        g = jax.random.normal(rng, (B, self.z_dim))
        return {"z": project_z(g, self.norm_z)}

    def infer_cond(self, net, obs, rewards):
        """z = lstsq(phi(s'), r), then sqrt(d)-normalized (sf.py:535-552).

        This is HILP's readout and it is NOT FB's reward moment: a least-squares SOLVE
        against the feature matrix, not `E[r phi]`. Spec sec.2 records 10,000 transitions
        for this domain, against RLDP's 50,000."""
        phi = self._feature_net(net["phi1"].params, net["hilp_stats"].params,
                                jnp.asarray(obs))
        r = jnp.asarray(rewards).reshape(-1, 1)
        z = jnp.linalg.lstsq(phi, r, rcond=None)[0].reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    def update_stats(self, net, pre_net, batch):
        """EMA the phi running mean (sf.py:184-187), from the PRE-STEP online tower.

        Upstream does this inside `forward()` as a side effect, ordered AFTER the loss and
        AFTER the target polyak. Agent.apply_update calls update_stats after both, so the
        ordering is preserved rather than merely approximated."""
        m = self.stats_momentum
        p = net["hilp_stats"].params
        phi1 = self._phi(pre_net["phi1"].params, batch["observations"])
        net = dict(net)
        net["hilp_stats"] = net["hilp_stats"].replace(params={
            "mean": p["mean"] * (1.0 - m) + jnp.mean(phi1, axis=0) * m,
            "std": p["std"] * (1.0 - m) + jnp.std(phi1, axis=0) * m})
        return net


class _HILPStats(nn.Module):
    """The phi running mean/std (sf.py's `register_buffer('running_mean'|'running_std')`).

    Ordinary param leaves, following density_fb's `_ObsNormState`, for the same reason:
    they SERIALIZE, so an eval harness centers phi with the identical numbers training
    used. Init mean 0 / std 1 matches upstream's buffer init, so step 0 reads the raw
    tower. Written only by `update_stats`; the gradient is identically zero and the group
    appears in no stage_groups.

    `std` is carried because upstream maintains it in the same EMA and it serializes with
    the checkpoint; nothing reads it. It is kept rather than dropped so that a later
    change upstream (or a diagnostic here) has the number, and so the checkpoint shape
    matches what the reference stores."""
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self):
        mean = self.param("mean", lambda _k: jnp.zeros((self.dim,), self.dtype))
        std = self.param("std", lambda _k: jnp.ones((self.dim,), self.dtype))
        return mean, std


def _hilp_build(cls, ex_obs, ex_actions, config):
    import jax.random as jr
    z_dim = int(config["z_dim"])
    P = int(config["num_parallel"])
    feature_type = str(config.get("feature_type", "state"))
    if feature_type != "state":
        # 'diff' and 'concat' are real upstream options (sf.py:614-620) that change what
        # the SF target regresses and, for 'concat', halve the phi width. Refused rather
        # than silently approximated: the published ExoRL configuration is 'state'.
        raise NotImplementedError(
            f"feature_type={feature_type!r} is not ported; upstream's 'diff' and 'concat' "
            f"change the SF target (and 'concat' halves the phi width to z_dim//2). The "
            f"published ExoRL setting is 'state'.")
    feature_dim = z_dim

    # PhiMap IS HILP's phi tower (see the class docstring); norm=False because HILP's
    # layer list ends at feature_dim with no trailing "L2".
    mk_phi = lambda: PhiMap(z_dim=feature_dim, hidden_dim=config["phi"]["hidden_dim"],
                            hidden_layers=config["phi"]["hidden_layers"], norm=False)
    phi1, phi2 = mk_phi(), mk_phi()
    sf = ForwardMap(z_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"], num_parallel=P,
                    hidden_layers=config["sf"]["hidden_layers"],
                    embedding_layers=config["sf"]["embedding_layers"])
    stats_mod = _HILPStats(dim=feature_dim, dtype=jnp.zeros((), ex_obs.dtype).dtype)

    r1, r2, rs, rst = jr.split(jr.PRNGKey(config["seed"]), 4)
    ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
    params = {
        "phi1": phi1.init(r1, ex_obs)["params"],
        "phi2": phi2.init(r2, ex_obs)["params"],
        "sf_psi": sf.init(rs, ex_obs, ex_z, ex_actions)["params"],
        "hilp_stats": stats_mod.init(rst)["params"],
    }
    critic = cls(
        phi1=phi1, phi2=phi2, sf=sf, z_dim=z_dim, num_parallel=P,
        discount=float(config["discount"]),
        hilp_discount=float(config.get("hilp_discount", 0.98)),
        hilp_expectile=float(config.get("hilp_expectile", 0.5)),
        phi_tau=float(config.get("phi_target_tau", 0.005)),
        sf_tau=float(config.get("sf_target_tau", 0.01)),
        norm_z=bool(config.get("norm_z", True)),
        stats_momentum=float(config.get("hilp_stats_momentum", 0.005)),
        q_loss=bool(config.get("hilp_q_loss", True)),
    )
    targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
    return critic, params, targets


HILPCritic.build = classmethod(_hilp_build)
