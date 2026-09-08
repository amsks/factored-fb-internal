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

STATUS (2026-08-30). Both stages are pinned by independent numpy reconstructions
(tests/impls/test_hilp_seam.py) and the critic IS registered in
impls/agents/__init__.py::CRITICS. An earlier version of this docstring claimed it was
deliberately unregistered; that stopped being true when it was registered, and the
"cannot silently produce a number" guarantee no longer holds -- it can be selected by
--critic today.

KNOWN DIVERGENCES FROM OFFICIAL, from the 2026-08-30 audit
(docs/notes/2026-08-30-hilp-audit.md). This critic is NOT yet a one-to-one port:
  * `mix_ratio` 0.5 z-relabelling is ABSENT. Official replaces half the batch's z with
    whitened feature directions, sqrt(d)*normalize(phi[perm] @ pinv(cov)), and uses that
    for BOTH the SF and actor updates (sf.py:735-759).
  * phi's learning rate is 5x too small: official is `lr_coef * lr` = 5e-4 (sf.py:37,490),
    ours rides lr_b = 1e-4.
  * the successor net ensemblizes the WHOLE tower, so each head gets its own embedding
    branches; official SHARES the two embeddings and separates only the 2-layer heads
    (fb_modules.py:148-193). Ours also has one extra trunk layer and orthogonal init
    gain sqrt(2) where official uses 1.0.
  * the published ExoRL launch passes p_randomgoal=0.375 (37.5% of goals drawn from a
    DIFFERENT episode), per-domain hilp_discount (0.96 walker), and q_loss=False for
    walker/cheetah. None of these is expressible here yet.
Verified as matching: the phi tower architecture, the phi value loss arithmetic, the SF
loss arithmetic, and z inference/sampling.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from impls.critics._measure_common import parse_hit_spec, scaled_hit_cols
from impls.utils.networks import ForwardMap, HILPForwardMap, PhiMap
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
    # Field names ARE the group names: Agent.create does getattr(critic, group) to
    # find each module. They are namespaced `hilp_*` because _LR_KEY is keyed by
    # GROUP NAME across all critics -- an unprefixed `sf_psi` silently re-keyed PSM's
    # own sf_psi group from lr_sf to lr_f and broke every PSM pair.
    hilp_phi1: Any
    hilp_phi2: Any
    hilp_sf: Any
    hilp_stats: Any
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
    mix_ratio: float
    # td_jepa parity knobs. Defaults reproduce OFFICIAL HILP, which is what critic/hilp.yaml
    # is transcribed from; critic/hilp_tdj.yaml flips all three. They are separate flags
    # rather than one "tdjepa mode" because they are independent choices in independent
    # places, and bundling them would hide which one moved a number.
    head_reduce: str        # SF TARGET across parallel heads: "argmin_q" (official) | "mean"
    actor_head_reduce: str  # ACTOR's Q across heads: "min" (official) | "mean"
    sf_loss_scale: float    # multiplies the SF loss: 1.0 (official) | 0.5 (td_jepa)
    # `backward_input` ("state" | "hit_spec"): what the phi towers consume as their RAW
    # input, the same knob rldp.py and psm.py carry for B / phi. "state" (default) is the
    # identity and reproduces every published HILP cell byte-for-byte; "hit_spec" feeds
    # the eps-scaled object-coordinate block slice instead (density_fb's
    # `_density_queries` semantics). On cube this is the ingredient that separates the
    # 1.0 rldp_density_oc cell from the 0.2 state-input one.
    #
    # NOTE what it changes for HILP specifically, which is MORE than it changes for RLDP:
    # phi is not only the successor outcome space here, it is also the Hilbert METRIC the
    # IQL value loss fits. Under "hit_spec" that metric measures temporal distance between
    # OBJECT configurations with the arm marginalized out, and r(u) = phi(u)^T z becomes a
    # purely object-defined reward. That is the intended transfer on manipulation, but it
    # is a change to the representation's meaning, not just to a tower's input width.
    backward_input: str = "state"
    blocks: tuple = ()
    # td_jepa's LEFT ENCODER (OGBench BASE_CFG: BackwardArchi 512x4, norm, -> L_dim 50).
    # Their SF net is built on `self._left_encoder.output_space`
    # (metamotivo/agents/sf/model.py:82), so it ALWAYS consumes the encoded observation --
    # `actor_encode_obs` gates only the ACTOR (:84, :117), and is False on OGBench. The
    # encoder is trained by the SF optimizer at lr_sf (sf/agent.py:91-92) and its target is
    # polyaked at sf_target_tau (:266-267), which is why it rides `sf_tau` below.
    # Official HILP has no such tower, so this is OFF by default and every pre-existing arm
    # (hilp, hilp_cube, hilp_oc, hilp_tdj, hilp_density_*) is bit-identical without it.
    hilp_le: Any = None
    le_on: bool = False

    # A DATACLASS FIELD, not a class constant (the density_fb.py:452 pattern): `hilp_le`
    # exists only when the td_jepa left encoder is on, so every HILP checkpoint written
    # before this knob keeps exactly the param keys it was saved with and stays restorable
    # under the strict `restore_agent` path.
    GROUPS: tuple = ("hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats")
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
        t = {"hilp_phi1": self.phi_tau, "hilp_phi2": self.phi_tau,
             "hilp_sf": self.sf_tau, "hilp_stats": 0.0}
        if self.le_on:
            # sf_target_tau, not phi_tau: _soft_update_params(left_encoder, target, tau)
            # is called with cfg.train.sf_target_tau (sf/agent.py:266-267).
            t["hilp_le"] = self.sf_tau
        return t

    @property
    def cvec_dim(self):
        return self.z_dim

    def stage_groups(self, stage):
        # Upstream runs two separate .backward() calls: the phi learner's value loss steps
        # only the phi towers, and the SF loss detaches phi entirely (it reads
        # `feature_net(...).detach()`), so no phi gradient flows from the SF stage.
        if stage == "phi":
            return ("hilp_phi1", "hilp_phi2")
        assert stage == "sf", stage
        # sf_optimizer = Adam(sf.parameters() + left_encoder.parameters() + ...) at lr_sf
        # (sf/agent.py:91-92), so the encoder is stepped by THIS stage, not the phi stage.
        return ("hilp_sf", "hilp_le") if self.le_on else ("hilp_sf",)

    # --- towers -------------------------------------------------------------
    def _phi_in(self, x):
        """phi's raw input under `backward_input`. THE choke point: every phi application
        routes through here -- `_phi`, `_phi2`, and the `apply_phi` closure the value loss
        is handed -- so no caller can feed the value loss a full state while the SF branch
        reads a slice. In "state" mode this is the identity; in "hit_spec" mode it is the
        eps-scaled hit-spec block slice. No caller pre-slices."""
        if self.backward_input == "state":
            return x
        return scaled_hit_cols(x, self.blocks)

    def _phi(self, p, x):
        return self.hilp_phi1.apply({"params": p}, self._phi_in(x))

    def _phi2(self, p, x):
        return self.hilp_phi2.apply({"params": p}, self._phi_in(x))

    def _le(self, container, x):
        """The SF net's view of an observation. Identity unless the td_jepa left encoder is
        on; `container` is the params/target dict, indexed only when the tower exists so an
        arm without it never looks up a missing key."""
        if not self.le_on:
            return x
        return self.hilp_le.apply({"params": container["hilp_le"]}, x)

    def _sf(self, p, obs, z, action):
        return self.hilp_sf.apply({"params": p}, obs, z, action)

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
            # Goals are the GEOMETRICALLY-sampled hindsight futures from
            # HILPFutureDataset (impls/utils/datasets.py). NOT `actor_goals`: the
            # ExoRL path uses FBDataset, which emits no goal keys at all, so
            # reading actor_goals raised KeyError on the first real batch --
            # after 15 unit tests passed, because they feed arrays directly.
            return hilp_phi_loss(
                params["hilp_phi1"], params["hilp_phi2"], target["hilp_phi1"], target["hilp_phi2"],
                # THROUGH the choke point: phi1's module applied to whichever tower's
                # params, on the `backward_input` view. Bypassing `_phi_in` here would fit
                # the Hilbert metric on full states while the SF branch reads object
                # coordinates -- two different representations under one name.
                lambda p, x: self._phi(p, x),
                batch["observations"], batch["hilp_future_observations"], batch["next_observations"],
                self.hilp_discount, self.hilp_expectile)

        assert stage == "sf", stage
        obs, action = batch["observations"], batch["actions"]
        nobs, z = batch["next_observations"], cond["z"]
        P = self.num_parallel

        # TARGET encoder on next_obs (sf/agent.py:282), online encoder on obs (:278).
        tF = self._sf(target["hilp_sf"], self._le(target, nobs), z,
                      boot["next_action"])                                    # [P, B, d]
        # TRAP 4: the SF target reads the ONLINE feature net, detached -- NOT a target phi.
        # The target phi towers exist only inside the phi value loss. (The RLDP run-1
        # divergence was the mirror image of this: reading the target tower where official
        # reads the online one.)
        target_phi = jax.lax.stop_gradient(
            self._feature_net(params["hilp_phi1"], params["hilp_stats"], nobs))   # [B, d]
        # TRAP 5b: official HILP chooses next_F PER SAMPLE by argmin over Q -- the whole
        # vector of the head with the smaller Q, not an elementwise min over the vectors.
        #
        # td_jepa does NOT do this. Its sf_loss calls
        # get_targets_uncertainty(next_SFs, sf_pessimism_penalty) and the penalty defaults
        # to 0.0 (agents/sf/agent.py:29), which returns preds.mean(dim=0) -- a plain MEAN
        # over heads, with no pessimism anywhere. Replicating td_jepa therefore requires
        # "mean"; leaving it at "argmin_q" makes the SF target pessimistic in a way theirs
        # never is.
        if self.head_reduce == "mean":
            next_F = jnp.mean(tF, axis=0)                                    # [B, d]
        elif self.head_reduce == "argmin_q":
            tQ = jnp.einsum("pbd,bd->pb", tF, z)                             # [P, B]
            pick = (tQ[0] < tQ[1])[:, None]                                  # [B, 1]
            next_F = jnp.where(pick, tF[0], tF[1])                           # [B, d]
        else:
            raise ValueError(f"hilp_head_reduce must be 'mean' or 'argmin_q', "
                             f"got {self.head_reduce!r}")
        target_F = jax.lax.stop_gradient(target_phi + self.discount * next_F)

        F = self._sf(params["hilp_sf"], self._le(params, obs), z, action)     # [P, B, d]
        if self.q_loss:
            # TRAP 5a: with q_loss (the shipped default) the regression is on the SCALAR
            # Q = <F, z>, not on the vector F.
            Q = jnp.einsum("pbd,bd->pb", F, z)
            tQs = jnp.einsum("bd,bd->b", target_F, z)
            sf_loss = jnp.sum(jnp.mean((Q - tQs[None]) ** 2, axis=-1))
        else:
            sf_loss = jnp.sum(jnp.mean((F - target_F[None]) ** 2, axis=(-1, -2)))
        # td_jepa's vector branch is 0.5 * P * F.mse_loss(SFs, target_SF), and torch's
        # mse_loss averages over ALL P*B*d elements, giving 0.5/(B*d) * sum. The line above
        # gives 1/(B*d) * sum, i.e. exactly 2x theirs -- a 2x learning rate on hilp_sf by
        # another name. sf_loss_scale=0.5 recovers their magnitude.
        sf_loss = self.sf_loss_scale * sf_loss
        return sf_loss, {"sf_loss": sf_loss, "target_F": jnp.mean(target_F),
                         "phi_norm": jnp.mean(jnp.linalg.norm(target_phi, axis=-1)),
                         "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1))}

    # --- control ------------------------------------------------------------
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        z = cond["z"]
        # The actor's Q still goes through the encoder even though the POLICY net reads raw
        # obs under actor_encode_obs=False -- sf/agent.py:321 encodes before calling _sf.
        Qs = jnp.einsum("pbd,bd->pb",
                        self._sf(params["hilp_sf"], self._le(params, obs), z, actions), z)
        # Official HILP's update_actor: Q = min(Q1, Q2), then actor_loss = -Q.mean() -- a
        # pure TD3 objective with no BC term.
        #
        # td_jepa again does not: its update_actor reduces with
        # get_targets_uncertainty(Qs, actor_pessimism_penalty), penalty 0.0 by default
        # (agents/sf/agent.py:30), i.e. the MEAN over heads. Its bc_coeff is 0.0 too, so the
        # Qs.abs() normalisation on the next line never fires -- which our
        # actor.alpha=0.0 / actor.q_normalize=false already match.
        q = jnp.mean(Qs, axis=0) if self.actor_head_reduce == "mean" else jnp.min(Qs, axis=0)
        return q, Qs

    def cond_vec(self, cond):
        return cond["z"]

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,))}

    def map_eval_cond(self, net, cond, observations=None):
        """Eval-time hook: the env's goal frame -> the z the actor rolls on. Official HILP's
        `get_goal_meta` (hilp_zsrl/url_benchmark/agent/sf.py:514-532):

            z = sqrt(d) * normalize(feature_net(g) - feature_net(s))

        feature_net = phi1 - running_mean (sf.py:120-122; the mean cancels in the
        difference but is applied to both terms for fidelity), s = the CURRENT observation,
        and the reference recomputes z EVERY STEP (train_offline.py:325-327), which is
        exactly what a per-call hook does. td_jepa has no goal-conditioned HILP eval at
        all (only `reward_inference`, sf/model.py:144), so official HILP is the reference.
        `hilp_density` inherits this unchanged: its whitened-goal encoding lives in
        sample_cond (training, goal_kernel only) and never touches the eval route.

        Before this hook existed (2026-09-08) `Agent.sample_actions` found no
        map_eval_cond on the HILP family and rolled eval_cond0's ZERO z for the whole
        episode -- a goal-blind policy behind every goal-conditioned HILP cell."""
        if "goal" not in cond:
            return cond
        if observations is None:
            raise ValueError("HILP's goal->z map is joint in goal AND current observation "
                             "(z = normalize(phi(g) - phi(s))); pass observations=.")
        p, st = net["hilp_phi1"].params, net["hilp_stats"].params
        z = self._feature_net(p, st, cond["goal"]) - self._feature_net(p, st, observations)
        return {**cond, "z": project_z(z, self.norm_z)}

    def sample_cond(self, net, batch, rng):
        """z ~ sqrt(d)*normalize(gaussian) (sf.py::sample_z), then MIX_RATIO relabelling.

        Official replaces a `mix_ratio` fraction of the batch's z with WHITENED feature
        directions (sf.py:735-759):

            phi     = feature_net(next_obs[perm])            # no grad, feature_type='state'
            cov     = phi.T @ phi / N ;  inv_cov = pinv(cov)
            new_z   = sqrt(d) * normalize(phi[mix] @ inv_cov)

        and the mixed z drives BOTH update_sf and update_actor. This is what grounds the
        policy family in directions the features can actually express; training on
        isotropic Gaussian z alone is a different algorithm. The whitening is not optional
        decoration -- PSM's superficially similar mix omits `inv_cov`, so it is NOT a
        drop-in for this.

        mix_ratio=0 (the default here) reproduces the pure-Gaussian behavior every earlier
        cell ran, so this is additive.
        """
        B = batch["observations"].shape[0]
        r_g, r_perm, r_mask = jax.random.split(rng, 3)
        z = project_z(jax.random.normal(r_g, (B, self.z_dim)), self.norm_z)
        if self.mix_ratio <= 0.0:
            return {"z": z}
        nobs = batch["next_observations"]
        phi = jax.lax.stop_gradient(self._feature_net(
            net["hilp_phi1"].params, net["hilp_stats"].params,
            nobs[jax.random.permutation(r_perm, B)]))              # [B, d]
        cov = (phi.T @ phi) / phi.shape[0]
        inv_cov = jnp.linalg.pinv(cov)
        new_z = project_z(phi @ inv_cov, self.norm_z)
        # A boolean mask, not an index gather: the count must be static under jit, and
        # upstream's `np.where(uniform < mix_ratio)` is exactly a per-row Bernoulli draw.
        mix = (jax.random.uniform(r_mask, (B,)) < self.mix_ratio)[:, None]
        return {"z": jnp.where(mix, new_z, z)}

    def infer_cond(self, net, obs, rewards):
        """z = lstsq(phi(s'), r), then sqrt(d)-normalized (sf.py:535-552).

        This is HILP's readout and it is NOT FB's reward moment: a least-squares SOLVE
        against the feature matrix, not `E[r phi]`. Spec sec.2 records 10,000 transitions
        for this domain, against RLDP's 50,000."""
        phi = self._feature_net(net["hilp_phi1"].params, net["hilp_stats"].params,
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
        phi1 = self._phi(pre_net["hilp_phi1"].params, batch["observations"])
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

    backward_input = str(config.get("backward_input", "state"))
    assert backward_input in ("state", "hit_spec"), (
        f"backward_input must be 'state' or 'hit_spec', got {backward_input!r}")
    blocks = ()
    ex_phi_in = ex_obs
    if backward_input == "hit_spec":
        # phi's input becomes a literal column slice of the observation, so the spec must
        # fit inside it -- jax would otherwise clip short rows into a silently different
        # (narrower) input at trace time. Same check as rldp.py / psm.py.
        blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        need = max(b for (_a, b, _e) in blocks)
        obs_dim = int(ex_obs.shape[-1])
        assert obs_dim >= need, (
            f"backward_input='hit_spec' needs observations wide enough for every "
            f"measure_hit_spec block: the spec reaches column {need} but the "
            f"observations are only {obs_dim} wide.")
        ex_phi_in = scaled_hit_cols(ex_obs, blocks)

    # PhiMap IS HILP's phi tower (see the class docstring); norm=False because HILP's
    # layer list ends at feature_dim with no trailing "L2".
    mk_phi = lambda: PhiMap(z_dim=feature_dim, hidden_dim=config["phi"]["hidden_dim"],
                            hidden_layers=config["phi"]["hidden_layers"], norm=False)
    phi1, phi2 = mk_phi(), mk_phi()
    # HILPForwardMap, not the seam's ForwardMap: upstream SHARES the two embedding
    # branches between F1/F2 and separates only the heads, uses ONE trunk layer per head,
    # and carries `feature_dim` as its own knob rather than hidden_dim//2. See the class
    # docstring. num_parallel is not a parameter -- the two heads are structural.
    if int(config.get("num_parallel", 2)) != 2:
        raise ValueError("HILP's successor net is F1/F2 by construction; num_parallel "
                         "must be 2.")
    # SF TOWER FAMILY. "hilp" (default) is official HILP's ForwardMap -- shared embedding
    # branches, one trunk layer per head, `feature_dim` its own knob. "tdjepa" is the
    # generic ForwardArchi td_jepa's OGBench BASE_CFG names for `successor_features`:
    # per-head parallel embeddings and a `hidden_layers`-deep trunk, i.e. the seam's own
    # ForwardMap, which FB already uses and whose equivalence to ForwardArchi the
    # 2026-09-03 delta audit verified for rldp_tdj (its predictor block, item U4).
    sf_archi = str(config.get("hilp_sf_archi", "hilp"))
    if sf_archi == "hilp":
        sf = HILPForwardMap(z_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                            feature_dim=int(config["sf"].get("feature_dim",
                                                             config["sf"]["hidden_dim"] // 2)))
    elif sf_archi == "tdjepa":
        sf = ForwardMap(z_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                        num_parallel=P, hidden_layers=config["sf"]["hidden_layers"],
                        embedding_layers=int(config["sf"].get("embedding_layers", 2)))
    else:
        raise ValueError(f"hilp_sf_archi must be 'hilp' or 'tdjepa', got {sf_archi!r}")

    # The left encoder, and the observation the SF tower is INITIALISED on -- it must be the
    # encoder's output, not the raw row, or the tower is built at the wrong input width.
    le_cfg = config.get("hilp_left_encoder") or {}
    le_on = bool(le_cfg) and not bool(le_cfg.get("identity", False))
    le = None
    ex_sf_in = ex_obs
    if le_on:
        L_dim = int(config.get("L_dim", z_dim))
        le = PhiMap(z_dim=L_dim, hidden_dim=int(le_cfg["hidden_dim"]),
                    hidden_layers=int(le_cfg["hidden_layers"]),
                    norm=bool(le_cfg.get("norm", True)))
        ex_sf_in = jnp.zeros((ex_obs.shape[0], L_dim), ex_obs.dtype)
    stats_mod = _HILPStats(dim=feature_dim, dtype=jnp.zeros((), ex_obs.dtype).dtype)

    r1, r2, rs, rst = jr.split(jr.PRNGKey(config["seed"]), 4)
    ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
    params = {
        "hilp_phi1": phi1.init(r1, ex_phi_in)["params"],
        "hilp_phi2": phi2.init(r2, ex_phi_in)["params"],
        "hilp_sf": sf.init(rs, ex_sf_in, ex_z, ex_actions)["params"],
        "hilp_stats": stats_mod.init(rst)["params"],
    }
    groups = ("hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats")
    if le_on:
        rle = jr.PRNGKey(config["seed"] + 913)
        params["hilp_le"] = le.init(rle, ex_obs)["params"]
        groups = groups + ("hilp_le",)
    critic = cls(
        hilp_phi1=phi1, hilp_phi2=phi2, hilp_sf=sf, hilp_stats=stats_mod, z_dim=z_dim, num_parallel=P,
        discount=float(config["discount"]),
        hilp_discount=float(config.get("hilp_discount", 0.98)),
        hilp_expectile=float(config.get("hilp_expectile", 0.5)),
        phi_tau=float(config.get("phi_target_tau", 0.005)),
        sf_tau=float(config.get("sf_target_tau", 0.01)),
        norm_z=bool(config.get("norm_z", True)),
        stats_momentum=float(config.get("hilp_stats_momentum", 0.005)),
        q_loss=bool(config.get("hilp_q_loss", True)),
        mix_ratio=float(config.get("hilp_mix_ratio", 0.0)),
        head_reduce=str(config.get("hilp_head_reduce", "argmin_q")),
        actor_head_reduce=str(config.get("hilp_actor_head_reduce", "min")),
        sf_loss_scale=float(config.get("hilp_sf_loss_scale", 1.0)),
        backward_input=backward_input, blocks=blocks,
        hilp_le=le, le_on=le_on, GROUPS=groups,
    )
    targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
    if bool(config.get("hilp_independent_target_init", False)):
        # UPSTREAM BEHAVIOR, and very likely an upstream BUG -- reproduced because it is
        # the published configuration, not because it looks right. sf.py:111-114 does:
        #     target_phi1.load_state_dict(phi1.state_dict())
        #     target_phi2.load_state_dict(phi2.state_dict())
        #     self.apply(utils.weight_init)          <-- AFTER the copy
        # `self.apply` re-initializes EVERY Linear in the module, targets included, each
        # with a fresh orthogonal draw. So after __init__ the target phis are INDEPENDENT
        # random networks, not copies, and the first few thousand steps bootstrap against
        # an unrelated network until tau=0.005 washes it out (~10^3 steps).
        # Default off: "fixing" it is a deviation from the reference either way, so the
        # choice is explicit rather than silent.
        rt1, rt2 = jr.split(jr.PRNGKey(config["seed"] + 7717), 2)
        targets["hilp_phi1"] = mk_phi().init(rt1, ex_phi_in)["params"]
        targets["hilp_phi2"] = mk_phi().init(rt2, ex_phi_in)["params"]
    return critic, params, targets


HILPCritic.build = classmethod(_hilp_build)
