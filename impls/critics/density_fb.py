"""DensityFBCritic — the continuous normalized successor density (bridge spec 18).

One-stage, from-scratch, two-tower softmax density f(s,a,g,u) = F(s,a,g)^T B(u) / tau over
a query support U resampled EVERY update from the batch's own next-state object
coordinates (so U == the batch's next_observations block columns; no prototype file, no
fixed support). Cube-single 0.929 on the bridge (origin/gciql-fb-bridge @ 542c058,
tools/wandb_mode_shim/gciql_flow.py — `density_loss` ~1975, the density branch of
`actor_loss` ~1089, the tower registration in `create` ~3147, the target soft-update in
`update` ~2587). Ported near-verbatim per docs/specs/2026-08-10-measure-objectives-on-mw-
play.md P1; see docs/successor-measure-objectives.md §1-5 for the code-reading guide.

Contract details preserved (each is pinned by a test in tests/impls/test_density_fb.py):
  * rewards/kernel counted at the NEXT state (x_{t+1});
  * `_delta_phi` eps-scales INTERNALLY — the kernel term gets RAW block cols
    (`_raw_block_cols`), the B tower gets PRE-SCALED queries (`_density_queries`); feeding
    the wrong one to the wrong callsite silently trains a different object;
  * softmax is over the QUERY axis (last axis), not the batch axis;
  * the bootstrap action a+ is drawn at the VALUE goal — `boot["next_action_vgoal"]`, i.e.
    a+ ~ pi(.|s', cond["value_goal"]) — NOT the seam's default `boot["next_action"]`, which
    `_cvec(cond)` conditions on the ACTOR goal. The bridge draws it as
    `actor(nobs, batch['value_goals'], noise)` (gciql_flow.py ~2030), and every other
    bootstrap in that file is at `vgoals` too. Reading the actor-goal action makes w_bar the
    occupancy of a policy heading to a different goal, i.e. a goal-averaged (behaviour-like)
    density: measured density_ess_median ~310 vs the bridge's healthy ~89 at M=B=1024;
  * queries are resampled every update from the CURRENT batch's own next-state object
    coords (stage_loss recomputes them from `batch['next_observations']` every call —
    there is no stored/cached support);
  * NO orthonormalization, NO mass anchor, NO scale normalization on this path — each was
    separately measured harmful on the bridge (ortho took cube-single 0.967 -> 0.225).

`measure_density_z` (default FALSE -- the whole z path is opt-in): swaps the policy-family
index from the raw goal to a REWARD-SAMPLE index z, so a task can be specified by
{(u_j, r_j)} instead of a goal. Only the SHIPPED spec-30 variant is ported: the
translation-gauge-invariant CENTERED DUAL encoder,

    m_c  = (1/M) sum_j (B(u_j) - mu_hat)(r_j - rbar),      rbar = mean_j r_j
    c    = (C_hat + lam I)^-1 m_c,  lam = lambda_mult * tr(C_hat)/d
    zhat = sqrt(d) * c / ||c||          (all-ones fallback when ||c|| < const_eps)

which is exactly invariant to r -> r + beta, to r -> alpha r (alpha > 0) and to
B -> B + b. The earlier spec-24 sphere index, the spec-29 uncentered dual, the analytic
point/Dirac prompt and every orthogonality term are deliberately NOT ported (the point
prompt is what made spec 29's .812 uninterpretable; ortho was separately measured
harmful). See PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md and the shim
tools/wandb_mode_shim/gciql_flow.py ~1176-1341 (`_density_z_moments`,
`_density_z_centered_coef`, `_density_z_centered_index`, `_density_z_centered_moment`,
`density_z_from_reward_samples`) plus `_sphere_or_neutral` ~34-50.

TWO DELIBERATE DEVIATIONS FROM THE SHIM, both forced by this seam and both documented at
their call sites: (a) the shim holds mu_hat/C_hat as EMA-tracked PARAM LEAVES written
inside its `update` override; this seam has no critic-owned mutable state and no
post-update hook, so the moments are the PLUG-IN estimate over the query pool the caller
supplies (the batch's own next-state query marginal at training, the reward-sample pool at
eval) -- the same population objects the pinned base measure rho_phi defines, but
unsmoothed. (b) the shim encodes with a slow-EMA copy of the backward tower
(`density_b_zenc`); here there is only the online `density_b`, read under stop_gradient.
Both paths (train and eval) use the same construction, which is the property the shim's
train/eval contract actually protects.

`actor_q_center` (bridge's "known gap", per the spec): the bridge's actor centers the
Q-LOSS NORMALIZER (sg[mean|q - mean(q)|] instead of sg[mean|q|]); our FlowBCActor has no
such knob. Per spec §2, implemented instead INSIDE `q_fn` — subtract the batch mean of Q
before it reaches the actor's own sg[|Q|.mean()] normalization. Whether this composition
reproduces the bridge's numbers is exactly what the cube anchor (P5) gates.

Self-contained by design (spec §2): the tower modules (MLP, the ensemble helper) are
copied here rather than imported from impls/utils/networks.py, so this file's isolation
from other critics is structural, not just a registry line. The only symbol shared with
td_fb.py is `impls.critics._measure_common.parse_hit_spec` (the hit-spec grammar).
"""
import copy
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.critics._measure_common import parse_hit_spec

# ── self-contained tower primitives (copied from the vendored OGBench networks.py the
# bridge itself imports MLP/ensemblize from — third_party/ogbench/impls/utils/networks.py
# in the bridge worktree — so the density towers' init/activation match the bridge's). ──
_DEFAULT_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


def _ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(cls, variable_axes={'params': 0}, split_rngs={'params': True},
                   in_axes=None, out_axes=out_axes, axis_size=num_qs, **kwargs)


class _MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=_DEFAULT_INIT)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


# ── sqrt(d)-sphere normalizations (bridge gciql_flow.py ~28-50), transcribed. ──
def _sphere(x, d):
    """Project x onto the sphere of radius sqrt(d) along the last axis — FB pins
    ||B||=||z||=sqrt(d). The +1e-6 in the DENOMINATOR is the shipped guard; it makes the
    map only approximately scale-invariant, which is why the centered index below uses
    `_sphere_or_neutral` instead. Kept for the task-prior sphere draws, where the argument
    is a fresh gaussian and the guard is irrelevant."""
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _sphere_or_neutral(x, d, eps):
    """EXACT sqrt(d)-normalization (no epsilon in the denominator) with the
    degenerate/constant-reward fallback z_neutral = 1_d (which itself has norm sqrt(d)).

    `_sphere`'s denominator guard makes it only approximately scale-invariant: under
    x -> alpha x the returned norm is sqrt(d)(1 - 1e-6/(alpha||x||)), so a small alpha
    visibly moves the result (bridge measurement: a 100x downscale moved z by 5e-3). The
    centered encoder's whole claim is EXACT invariance to r -> alpha r and to the additive
    gauge, so it normalizes exactly and handles ||x|| ~ 0 explicitly — which it must do
    anyway, since a (near-)constant reward is a real input, not a numerical accident."""
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    degenerate = nrm < eps
    safe = jnp.where(degenerate, jnp.ones_like(nrm), nrm)
    return jnp.where(degenerate, jnp.ones_like(x), x * jnp.sqrt(d * 1.0) / safe)


# ── the density kernel — _delta_phi / _density_queries / _raw_block_cols (bridge
# gciql_flow.py ~1403-1449), NOT shared with td_fb (td_fb uses _hit_matrix, not this). ──
def _density_queries(nobs, blocks):
    """[M, d_c] eps-scaled block coords of next_observations — the resampled query
    support U AND the B-tower input rows."""
    parts = [nobs[:, a:b] / e for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _raw_block_cols(obs_rows, blocks):
    """[N, d_c] RAW (unscaled) concatenated block columns — the twin of
    `_density_queries` used as the `protos` argument to `_delta_phi`, which applies its
    own per-block eps scaling internally. Do NOT pre-scale this input."""
    parts = [obs_rows[:, a:b] for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _delta_phi(obs_rows, protos, blocks):
    """softmax_j( -0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2 ), over the QUERY
    axis (last). `obs_rows` [N, obs_dim] RAW; `protos` [M, d_c] RAW block-concatenated
    columns (the `_raw_block_cols` convention) — eps scaling happens INTERNALLY here, so
    callers must feed both sides RAW (the `_density_queries` pre-scaled rows are the
    B-tower's own input, never this function's)."""
    protos = jnp.asarray(protos, jnp.float32)
    parts_x, parts_p, off = [], [], 0
    for (a, b, e) in blocks:
        w = b - a
        parts_x.append(obs_rows[:, a:b] / e)
        parts_p.append(protos[:, off:off + w] / e)
        off += w
    xc = jnp.concatenate(parts_x, axis=-1)            # [N, d_c]
    pc = jnp.concatenate(parts_p, axis=-1)             # [M, d_c]
    diffs = xc[:, None, :] - pc[None, :, :]            # [N, M, d_c]
    logits = -0.5 * jnp.sum(diffs ** 2, axis=-1)       # [N, M]
    return jax.nn.softmax(logits, axis=-1)             # query axis = last axis


@dataclasses.dataclass(frozen=True)
class DensityFBCritic:
    density_f: Any
    density_b: Any
    obs_dim: int
    blocks: tuple           # parsed measure_hit_spec, nonpytree (plain python tuple)
    discount: float
    temp: float
    tau: float
    actor_q_center: bool
    hard: bool
    # W1 (edge-attribution spec): 'actor' (default, bit-identical) bootstraps the TD target at
    # the actor's action, so the actor sits inside the critic's fixed point — mechanism M1,
    # the confirmed collapse cause. 'sarsa' bootstraps at the DATASET's next action instead,
    # giving GCIQL's in-sample-backup property while keeping the kernel readout, which is what
    # separates M1 from M2.
    bootstrap: str = "actor"
    # W2: temperature for the ACTOR READOUT ONLY (q_fn's softmax). None -> `temp`, i.e.
    # bit-identical. The CE training target in stage_loss always keeps `temp`; a knob that
    # moved both would confound readout smoothness (M2) with the measure being trained.
    actor_temp: Any = None

    # --- spec 30 P1 reward-sample index (measure_density_z, DEFAULT OFF) ------------------
    # Every one of these is read ONLY when z_on is True; with z_on False not a single extra
    # op is traced, so the goal-conditioned arm stays byte-identical (tests/impls/
    # test_boot_contract.py is the 60-pair gate on exactly that).
    z_on: bool = False              # measure_density_z
    z_dim: int = 0                  # measure_density_dim, echoed here so cvec_dim can read it
    z_rand_frac: float = 0.5        # measure_density_z_rand_frac: fraction of rows replaced by
                                    # a fresh sqrt(d)-sphere draw (FB's task-prior mix)
    z_ridge_lambda_mult: float = 0.01   # measure_density_z_ridge_lambda_mult: lam = mult*tr(C)/d
    z_cov_eig_floor: float = 1e-6   # measure_density_z_cov_eig_floor: eigenvalue floor on C_hat,
                                    # also a multiple of the mean eigenvalue tr(C)/d
    z_const_eps: float = 1e-6       # measure_density_z_const_eps: ||c|| below which the reward
                                    # counts as constant and the index falls back to 1_d

    GROUPS = ("density_f", "density_b")
    STAGES = ("density",)
    PROVIDES_V = False
    # An INSTANCE field (annotated), not a class attribute: `build` widens it to include "z"
    # when the index is on, and check_combo/_cvec read it off the instance. The class-level
    # value is still the goal-conditioned default, which is what every class-level consumer
    # (tests/impls/test_boot_contract.py) sees.
    PROVIDES_COND: frozenset = frozenset({"value_goal", "goal"})

    # F(s, a, g): the row is g-indexed, so the TD target must bootstrap at that same g.
    # Reading the actor-goal action instead makes w_bar a goal-averaged (behaviour-like)
    # density -- measured density_ess_median ~310 against a healthy ~89.
    BOOT_COND = "value_goal"

    @property
    def TARGETS(self):
        return {"density_f": self.tau, "density_b": self.tau}

    @property
    def cvec_dim(self):
        """Width of cond_vec(cond) — what the actor's conditioning leg is sized to. With the
        reward-sample index on that is the index width d, not the observation width: the
        actor rides z exactly as it otherwise rides the goal."""
        return self.z_dim if self.z_on else self.obs_dim

    def stage_groups(self, stage):
        assert stage == "density"
        return self.GROUPS

    # --- module apply helpers ---
    def _F(self, p, x):
        return self.density_f.apply({"params": p}, x)     # [2, B, d]

    def _B(self, p, u):
        return self.density_b.apply({"params": p}, u)      # [M, d]

    # ------------------------------------------------------------------------------------
    # spec 30 P1 -- the TRANSLATION-GAUGE-INVARIANT CENTERED DUAL reward-sample index.
    #
    # ONE pinned base measure rho_phi = the next-state query marginal, used for mu_hat,
    # C_hat, rbar AND m_c consistently; reward samples are expected from rho_phi (no
    # arbitrary-proposal importance correction).
    #
    # Invariances (population exact; the plug-in moment estimate approximates them at
    # finite M -- say both, in that order):
    #   * B -> B + b     m_c is literally independent of mu (sum_j (r_j - rbar) = 0) and
    #                    C_hat is a covariance, so both are unchanged.
    #   * r -> r + beta  exact: r - rbar is unchanged.
    #   * r -> alpha r   (alpha > 0) exact after the EXACT sqrt(d) normalization.
    #
    # ⚠ CONSEQUENCE OF THE PLUG-IN MOMENTS (deviation (a) in the module docstring): because
    # mu_hat is the mean of B over the SAME pool the reward samples live on, sum_j
    # (B(u_j) - mu_hat) is EXACTLY zero (measured 9.6e-14). The reward-centering (r - rbar)
    # and the B-centering (B - mu) therefore each kill BOTH gauges on their own, and MASK
    # each other: deleting either one leaves every end-to-end invariance test green
    # (verified by mutation). Both are kept because both are in the ported formula and
    # because the redundancy DISAPPEARS the moment the moments stop being the plug-in ones
    # -- which is exactly what a future EMA port would do. The discriminating tests are the
    # formula-level ones in tests/impls/test_density_fb_reward_index.py section 3b; do not
    # read the end-to-end invariance tests as evidence about which centering is present.
    #
    # The FULL linear gauge B -> A B, F -> A^-T F is NOT fixed: an isotropic ridge and a
    # sphere normalization are not invariant under a general A. The claim is scoped to the
    # ADDITIVE gauge (spec 30 P1, Terminology, binding).
    #
    # There is NO one-sample / Dirac limit: m_c is IDENTICALLY ZERO at M=1, so M=1 raises
    # rather than silently returning a zero index. Never describe this encoder as having a
    # point limit -- that phrasing belongs to the spec-29 uncentered encoder, which is not
    # ported here, and it is what let spec 29's .812 read as general-encoder evidence when
    # it was a goal shortcut (spec 30 sections 4.8 and Rev 1).
    # ------------------------------------------------------------------------------------
    def _z_moments(self, b_params, u_sc):
        """(mu_hat [d], C_hat [d, d]) of the index tower's embeddings over the query pool.

        DEVIATION (see the module docstring): the shim holds these as EMA-tracked param
        leaves written inside its `update` override, so training and eval read the identical
        serialized object. This seam has no critic-owned mutable state and no post-update
        hook, so they are the PLUG-IN estimate over whatever pool the caller passes -- the
        batch's own next-state query set during training, the reward-sample pool at eval.
        Same population objects, same 1/M centered convention as the shim's EMA source
        (gciql_flow.py ~3903-3912), but unsmoothed: a finite-M covariance of a d-wide tower
        is a noisier, more step-to-step-variable object than the EMA it replaces.

        Read at stop_gradient throughout: the index must never become a path by which the
        actor's loss reshapes B."""
        b = jax.lax.stop_gradient(self._B(b_params, u_sc))              # [M, d]
        mu = jnp.mean(b, axis=0)                                        # [d]
        db = b - mu[None]
        cov = db.T @ db / b.shape[0]                                    # centered, 1/M
        return mu, 0.5 * (cov + cov.T)                                  # SYMMETRIZE

    def _z_centered_moment(self, b_params, u_sc, mu, r):
        """m_c = (1/M) sum_j (B(u_j) - mu_hat)(r_j - rbar), rbar = (1/M) sum_j r_j.

        `u_sc` [M, d_c] eps-scaled queries, `r` [B, M] (or [M]) rewards on them -> [B, d].
        M = 1 RAISES: the centered moment is identically zero there, so a one-sample call is
        a caller error, not a degraded estimate. The raise happens at trace time (shapes are
        static in jax), so it surfaces at the call site rather than as a silent zero index."""
        r2 = jnp.atleast_2d(r)                                          # [B, M]
        m_samples = int(r2.shape[-1])
        if m_samples < 2:
            raise ValueError(
                "the centered reward-sample encoder requires M > 1 reward samples, got "
                f"M={m_samples}. The centered moment m_c = (1/M) sum_j (B(u_j)-mu)(r_j-rbar) "
                "is IDENTICALLY ZERO at M=1 (r_1 - rbar = 0), so this encoder has no "
                "one-sample/Dirac limit (spec 30 Rev 1, P1). Realistic sample pools are "
                "M >= 256; the analytic point prompt is a SEPARATE api and is not ported.")
        b_u = jax.lax.stop_gradient(self._B(b_params, u_sc))            # [M, d]
        db = b_u - mu[None]                                             # [M, d]
        dr = r2 - jnp.mean(r2, axis=-1, keepdims=True)                  # [B, M]
        return jnp.einsum('bm,md->bd', dr, db) / m_samples              # [B, d]

    def _z_centered_coef(self, m_c, cov):
        """c = (C_hat + lam I)^-1 m_c, the UNNORMALIZED centered ridge coefficient.

        C_hat is symmetrized again here (belt and braces against float drift) and its
        spectrum is FLOORED before the solve: eigenvalues below z_cov_eig_floor * tr(C)/d
        are lifted to that floor. The centered covariance of a d=50 tower estimated from
        finite batches genuinely has near-zero directions (bridge measurement: effective
        rank 6-21 of 50), and an unfloored inverse would amplify whatever numerical noise
        sits in them.

        lam = z_ridge_lambda_mult * tr(C_hat)/d -- the MEAN-EIGENVALUE convention, computed
        from the UNFLOORED trace so the number is the one the bridge's preflight probes
        scored. Dimensionless, so it is comparable across checkpoints and towers."""
        d = float(self.z_dim)
        cov = 0.5 * (cov + cov.T)
        tr = jnp.trace(cov)
        ev, vec = jnp.linalg.eigh(cov)                                  # ascending
        ev = jnp.maximum(ev, self.z_cov_eig_floor * tr / d)
        lam = self.z_ridge_lambda_mult * tr / d
        return ((m_c @ vec) / (ev + lam)[None, :]) @ vec.T              # [B, d]

    def _z_centered_index(self, m_c, cov):
        """zhat = sqrt(d) * normalize(c), with the CONSTANT-REWARD FALLBACK: a reward that
        is (near-)constant on the pool has m_c ~ 0 and therefore ||c|| ~ 0, and the
        normalization would then amplify pure numerical noise into a confident-looking task
        index. When ||c|| < z_const_eps the encoder returns z_neutral = 1_d instead -- a
        fixed, reward-free direction that still carries the standing ||z|| = sqrt(d)
        invariant (||1_d|| = sqrt(d)), so it is IN the conditioning distribution rather than
        a zero vector that is not."""
        c = self._z_centered_coef(m_c, cov)
        return jax.lax.stop_gradient(
            _sphere_or_neutral(c, float(self.z_dim), self.z_const_eps))

    def z_from_reward_samples(self, b_params, u_sc, r):
        """THE GENERAL REWARD INTERFACE: reward samples {(u_j, r_j)}_{j=1..M} -> index
        [B, d]. No goal anywhere. `u_sc` is an [M, d_c] eps-scaled query pool
        (`_density_queries` output) and `r` is [B, M] (or [M]) reward values on it.

        This is the single solve BOTH the training rows (`sample_cond`, reward samples =
        the batch query set with r = the hindsight kernel kg) and the harness-facing eval
        (`infer_cond`, arbitrary reward families) go through, so the two cannot drift apart
        in it -- the property the shim's serialized-moments contract exists to protect."""
        mu, cov = self._z_moments(b_params, u_sc)
        return self._z_centered_index(self._z_centered_moment(b_params, u_sc, mu, r), cov)

    def _z_mix(self, z_g, rng):
        """FB's standard task-prior mix: keep the encoded index with probability
        1 - z_rand_frac, else a FRESH uniform sqrt(d)-sphere draw (bridge
        `_density_z_masked`, gciql_flow.py ~1412-1436; the hindsight indicator that method
        also returns feeds a BC mask that is not ported). Both branches carry
        ||z|| = sqrt(d). The whole result is stop_gradded."""
        d, n = int(self.z_dim), z_g.shape[0]
        rng_sphere, rng_mix = jax.random.split(rng)
        z_rand = _sphere(jax.random.normal(rng_sphere, (n, d)), d)
        use_rand = jax.random.bernoulli(rng_mix, self.z_rand_frac, (n, 1))
        return jax.lax.stop_gradient(jnp.where(use_rand, z_rand, z_g))

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "density"
        obs, nobs = batch["observations"], batch["next_observations"]
        acts = batch["actions"]
        # The F tower's conditioning leg AT THE VALUE GOAL. With the index off that is the
        # raw value goal (byte-identical); with it on it is z(value_goals) -- which is the
        # SAME vector cond["_cvec_vgoal"] publishes for the TD bootstrap draw, so F and the
        # policy whose occupancy it bootstraps are conditioned on one and the same thing.
        # Reading cond["value_goal"] here under the index would hand a goal-shaped vector to
        # a z-shaped net; that exact class of bug has been shipped twice in this tree.
        g = cond["_cvec_vgoal"] if self.z_on else cond["value_goal"]
        gamma, tau = self.discount, self.temp
        blocks = self.blocks

        u_raw = _raw_block_cols(nobs, blocks)                             # [M, d_c] RAW
        u_sc = _density_queries(nobs, blocks)                             # [M, d_c] eps-scaled

        b_emb = self._B(params["density_b"], u_sc)                        # [M, d]
        f_emb = self._F(params["density_f"], jnp.concatenate([obs, acts, g], -1))  # [2,B,d]
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau            # [2, B, M]
        w = jax.nn.softmax(logits, -1)
        logp = jax.nn.log_softmax(logits, -1)

        # Bootstrap action from the actor (already computed outside the grad tape by
        # Agent.update -> actor.sample_bootstrap; stop_gradient is defense-in-depth).
        # It MUST be the action for the VALUE goal g (= cond["value_goal"], the same g the
        # F towers are conditioned on here) — the bridge draws it as
        # `a_plus = sg(actor(nobs, batch['value_goals'], noise))` (gciql_flow.py ~2030).
        # `boot["next_action"]` is conditioned on the ACTOR goal (`_cvec(cond)`), which is a
        # different goal distribution (p_trajgoal=1.0 vs the 0.2/0.5/0.3 value mix): using it
        # makes w_bar the occupancy of a policy heading elsewhere, i.e. a goal-averaged,
        # much more DIFFUSE density (measured: density_ess_median ~310 vs the bridge's ~89).
        # Indexed, not .get()-defaulted: a silent fallback is exactly how this was missed.
        if self.bootstrap == "sarsa":
            # W1 / KT-B: bootstrap at the action the BEHAVIOUR policy took at s'. The actor is
            # then outside the critic's fixed point entirely — it can no longer walk the
            # target around, which is GCIQL's M1 property, while the softmax-kernel readout
            # (M2) stays exactly as it is. That is what makes this arm a clean M1-vs-M2
            # discriminator rather than another mitigation.
            if "next_actions" not in batch:
                raise KeyError(
                    "measure_density_bootstrap='sarsa' needs batch['next_actions'], which "
                    "GCDataset emits. A dataset_class that does not serve it cannot run this "
                    "mode — do not fall back to the actor action, that silently restores M1.")
            a_plus = jax.lax.stop_gradient(batch["next_actions"])
        elif self.bootstrap == "actor":
            if "next_action_vgoal" not in boot:
                raise KeyError(
                    "density_fb needs boot['next_action_vgoal'] — the bootstrap action drawn "
                    "at cond['value_goal'] (FlowBCActor.sample_bootstrap emits it whenever "
                    "the critic's cond carries 'value_goal'). boot['next_action'] is the "
                    "ACTOR-goal action and is NOT a valid bootstrap for this target.")
            a_plus = jax.lax.stop_gradient(boot["next_action_vgoal"])
        else:
            raise ValueError(f"measure_density_bootstrap must be 'actor' or 'sarsa', "
                             f"got {self.bootstrap!r}")
        tb = self._B(target["density_b"], u_sc)                           # frozen target
        tf = self._F(target["density_f"], jnp.concatenate([nobs, a_plus, g], -1))
        w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)  # [2, B, M]

        if self.hard:
            # spec 21: one-hot immediate target at the row's OWN successor (M == B because
            # the query support is this batch's next states). All (1-gamma) mass on the
            # observed next state; the soft eps-kernel was diluting the readout.
            k = jnp.eye(nobs.shape[0])
        else:
            k = _delta_phi(nobs, u_raw, blocks)                            # [B, M]
        t = jax.lax.stop_gradient((1.0 - gamma) * k[None] + gamma * w_bar)
        ce = -jnp.mean(jnp.sum(t * logp, axis=-1))

        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [2, B]
        ess = jnp.mean(ess_rows)
        ess_median = jnp.median(ess_rows)                                  # MEDIAN, not mean — the gate stat
        ess_h0 = jnp.mean(ess_rows[0])
        ess_h1 = jnp.mean(ess_rows[1])
        k_self = jnp.mean(jnp.diagonal(k))
        info = {
            "density_ce": ce,
            "density_ess": ess,
            "density_ess_median": ess_median,
            "density_ess_h0": ess_h0,
            "density_ess_h1": ess_h1,
            "density_k_self": k_self,
        }
        # How far apart the two bootstrap choices are. Under `sarsa` this is the size of the
        # substitution being made; under `actor` it is the same R1 signal from the critic's
        # side — how far the policy has walked off the behaviour support at s'. Logged in both
        # modes so the two arms are directly comparable, and only when the actor's draw is
        # available (a critic-only harness may not supply it).
        if "next_action_vgoal" in boot and "next_actions" in batch:
            info["density_sarsa_subst"] = jnp.linalg.norm(
                jax.lax.stop_gradient(boot["next_action_vgoal"]) - batch["next_actions"],
                axis=-1).mean()
        return ce, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """The actor's greedy readout: q_h = sum_j w_h(s,a,g,u_j) * kernel(goal, u_j), the
        density towers evaluated at FROZEN (stop-gradient) params — the measure trains
        only through stage_loss, never through the actor's -Q.mean() term. Requires
        `cond["_density_u_raw"]`/`cond["_density_u_sc"]` (stashed by `sample_cond` from the
        SAME batch's next_observations `q_fn` itself has no access to — see sample_cond)."""
        del stop_encoder_grad
        u_raw = cond["_density_u_raw"]
        u_sc = cond["_density_u_sc"]
        f_p = jax.lax.stop_gradient(params["density_f"])
        b_p = jax.lax.stop_gradient(params["density_b"])
        b_emb = self._B(b_p, u_sc)
        if self.z_on:
            # Index mode (bridge `_density_actor_q_z`, gciql_flow.py ~1483-1529, at its
            # SHIPPED default measure_density_z_actor_reward='latent'; the 'kernel' cell is
            # deliberately not ported). F conditions on z, and the reward the actor ascends
            # is FB's own linear model implied by the index, r_z(u) = B(u)^T z / sqrt(d),
            # read on the SAME query support the measure is trained against. The shim reads
            # r_z off the TARGET backward tower; this seam's q_fn has no target params, so
            # it reads the ONLINE tower under the stop_gradient it already applies -- the
            # frozen-params discipline is preserved, the tower identity is not.
            g = cond["z"]
            kg = jax.lax.stop_gradient(
                jnp.einsum('md,bd->bm', b_emb, g) / jnp.sqrt(float(self.z_dim)))
        else:
            g = cond["goal"]
            kg = jax.lax.stop_gradient(_delta_phi(g, u_raw, self.blocks))  # [B, M]
        f_emb = self._F(f_p, jnp.concatenate([obs, actions, g], -1))
        # W2 / KT-C: the ACTOR readout may use its own temperature. Raising it flattens the
        # softmax the actor ascends (M2: a spiky readout is the residual suspect for the
        # precision gap, ESS ~13 under qc03) WITHOUT touching what the measure is trained to
        # be — stage_loss keeps self.temp. Default None => self.temp => bit-identical.
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / a_tau, -1)
        q1, q2 = (w * kg[None]).sum(-1)                                    # [2, B] -> [B],[B]
        q = jnp.minimum(q1, q2)
        if self.actor_q_center:
            # Bridge semantics (gciql_flow.py ~1168): actor_q_center touches ONLY the
            # normalizer — q_scale = |q - q.mean()|.mean(), with -q.mean()'s gradient
            # unchanged. FlowBC divides its q-term by sg[|Qs|.mean()], so the centering
            # goes into the SECOND return. Centering the first (the numerator) makes
            # -Q.mean() the batch-mean of a batch-centered quantity — identically ~0,
            # no greedy signal, pure-BC actor (the failed first anchor generation).
            return q, (q - q.mean())[None]
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        """The vector actors concatenate onto obs. The raw actor goal, or -- with the
        reward-sample index on -- the index built at that goal."""
        return cond["z"] if self.z_on else cond["goal"]

    def sample_cond(self, net, batch, rng):
        # queries resampled every update from THIS batch's own next-state object coords.
        nobs = batch["next_observations"]
        u_raw = _raw_block_cols(nobs, self.blocks)
        u_sc = _density_queries(nobs, self.blocks)
        # `_cvec_vgoal` is THIS critic's conditioning vector at the value goal, published for
        # the actors' TD bootstrap draw (BOOT_COND == "value_goal"). cond_vec here is
        # cond["goal"] -- the raw goal -- so the value-goal counterpart is value_goals.
        # Publishing it is what makes the draw correct rather than inferred; see the block
        # comment in impls/actors/ddpgbc.py::sample_bootstrap.
        if self.z_on:
            # The index replaces the goal on BOTH legs. The row's reward samples are (this
            # batch's query set, kg) with kg = _delta_phi(goal, u_raw): the QUERY-SET-
            # NORMALIZED kernel, the same object the goal path integrates -- the standing
            # spec-28 parity rule, without which the two arms are not comparable (bridge
            # actor_loss ~1691-1699 and density_loss ~2800-2806).
            #
            # TWO indices, because the two legs are taken at two DIFFERENT goal
            # distributions: the actor leg at actor_goals (p_trajgoal=1.0 -- this is what
            # cond_vec/_cvec hands the policy) and the value leg at value_goals (the
            # 0.2/0.5/0.3 mix -- this is what F is conditioned on in stage_loss AND what the
            # TD bootstrap action must be drawn at). Publishing z(value_goals) as
            # `_cvec_vgoal` is the whole point: it is the actor-conditioning vector at the
            # value goal, so the bootstrap draw is z-shaped against a z-shaped actor net.
            b_p = jax.lax.stop_gradient(net["density_b"].params)
            mu, cov = self._z_moments(b_p, u_sc)
            r_a, r_v = jax.random.split(rng)

            def _z_at(goals, r_mix):
                r = jax.lax.stop_gradient(_delta_phi(goals, u_raw, self.blocks))  # [B, M]
                m_c = self._z_centered_moment(b_p, u_sc, mu, r)
                return self._z_mix(self._z_centered_index(m_c, cov), r_mix)

            return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                    "z": _z_at(batch["actor_goals"], r_a),
                    "_cvec_vgoal": _z_at(batch["value_goals"], r_v),
                    "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                "_cvec_vgoal": batch["value_goals"],
                "_density_u_raw": u_raw, "_density_u_sc": u_sc}

    def eval_cond0(self, ex_obs, config):
        if self.z_on:
            # Placeholder until infer_cond is called, mirroring FBCritic.eval_cond0. A zero
            # z is OFF the sqrt(d) sphere and therefore out of the conditioning
            # distribution; the point of --infer_cond_episodes is that it gets replaced.
            return {"z": jnp.zeros((self.z_dim,), jnp.float64)}
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        """Zero-shot task specification FROM REWARD SAMPLES: `obs` [M, obs_dim] raw
        observations drawn from the base measure and `rewards` [M] (or [B, M]) their reward
        values -> the conditioning the actor rolls on. No goal is involved.

        With the index off this is the pre-existing no-op placeholder (impls/main.py's
        `infer_cond_noop_reason` detects and reports that case), so nothing changes for the
        goal-conditioned arm.

        `obs` is eps-scaled here exactly as the training query support is: the B tower's
        input is always `_density_queries` output, never raw block columns.

        TWO THINGS THE CALLER OWNS, neither of which this method can check for you:
          * WHICH ROWS. The pinned base measure rho_phi is the NEXT-state query marginal --
            that is what the measure is trained against and what mu_hat/C_hat are estimated
            from here. impls/main.py's --infer_cond_episodes path passes rb["observations"],
            i.e. the CURRENT states. On a play dataset the two marginals are close, but they
            are not the same object, and nothing downstream will complain. Pass
            next_observations if you want the index solved against the measure's own support.
          * WHICH REWARDS. `rewards` must be the task reward on those rows. GCDataset's
            reward column is the GOAL-REACHING reward, which is near-constant on most
            samples; a near-constant reward legitimately carries no task information and the
            encoder returns the neutral 1_d index for it. That is the designed behaviour, not
            a failure -- but it is also not a zero-shot result, and
            impls/main.py::infer_cond_noop_reason will NOT flag it (cond did change)."""
        if not self.z_on:
            return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        z = self.z_from_reward_samples(net["density_b"].params,
                                       _density_queries(obs, self.blocks), rewards)
        # A flat [M] reward vector describes ONE task -> return a 1-D z, matching
        # eval_cond0's shape and FBCritic.infer_cond's convention.
        return {"z": z[0] if jnp.ndim(rewards) == 1 else z}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["measure_density_dim"])
        blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        obs_dim = int(ex_obs.shape[-1])
        z_on = bool(config.get("measure_density_z", False))
        z_rand_frac = float(config.get("measure_density_z_rand_frac", 0.5))
        z_lam = float(config.get("measure_density_z_ridge_lambda_mult", 0.01))
        z_floor = float(config.get("measure_density_z_cov_eig_floor", 1e-6))
        z_eps = float(config.get("measure_density_z_const_eps", 1e-6))
        if z_on:
            assert 0.0 <= z_rand_frac <= 1.0, \
                f"measure_density_z_rand_frac must be in [0,1], got {z_rand_frac}"
            assert z_lam > 0.0, \
                "measure_density_z_ridge_lambda_mult must be > 0 (lam = mult * tr(C)/d; " \
                "at 0 the floored solve is an unregularized inverse)"
            assert z_floor > 0.0, \
                "measure_density_z_cov_eig_floor must be > 0 (it is a multiple of the mean " \
                "eigenvalue tr(C)/d and is what keeps C_hat's near-null directions from " \
                "amplifying numerical noise into a confident-looking index)"
            assert z_eps > 0.0, \
                "measure_density_z_const_eps must be > 0 -- it is the constant-reward " \
                "fallback threshold, and at 0 a constant reward encodes pure noise"

        density_f = _ensemblize(_MLP, 2)((*hd, d), activate_final=False, layer_norm=ln)
        density_b = _MLP((*hd, d), activate_final=False, layer_norm=ln)

        rf, rb = jr.split(jr.PRNGKey(config["seed"] + 6000), 2)
        # F's conditioning leg is the index (width d) under measure_density_z, the raw goal
        # (width obs_dim) otherwise -- the tower has to be INITIALIZED at that width or the
        # first stage_loss dies with a ScopeParamShapeError.
        ex_f = jnp.concatenate(
            [ex_obs, ex_actions,
             jnp.zeros((ex_obs.shape[0], d), ex_obs.dtype) if z_on else ex_obs], -1)
        ex_u = _density_queries(ex_obs, blocks)
        params = {
            "density_f": density_f.init(rf, ex_f)["params"],
            "density_b": density_b.init(rb, ex_u)["params"],
        }
        targets = {"density_f": copy.deepcopy(params["density_f"]),
                   "density_b": copy.deepcopy(params["density_b"])}
        obj = cls(density_f=density_f, density_b=density_b, obs_dim=obs_dim, blocks=blocks,
                  discount=float(config["discount"]),
                  temp=float(config["measure_density_temp"]), tau=float(config["tau"]),
                  actor_q_center=bool(config.get("actor_q_center", True)),
                  # DEFAULT TRUE. The yaml has said true since the bridge-port spec's item
                  # A2, but this fallback said False -- so any path building the critic from
                  # a partial config silently got the losing variant. Now they agree.
                  hard=bool(config.get("measure_density_hard", True)),
                  bootstrap=str(config.get("measure_density_bootstrap", "actor")),
                  actor_temp=config.get("measure_density_actor_temp", None),
                  z_on=z_on, z_dim=d, z_rand_frac=z_rand_frac,
                  z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                  # Declaring "z" is what tells check_combo (and any actor that inspects the
                  # critic) that this critic's conditioning is a latent, not a goal.
                  PROVIDES_COND=(frozenset({"value_goal", "goal", "z"}) if z_on
                                 else frozenset({"value_goal", "goal"})))
        return obj, params, targets
