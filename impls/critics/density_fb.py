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

STATE, and why it lives where it does. The index is NOT a function of the current batch:
the shim states outright that batch statistics are forbidden (they reintroduce the
train/eval mismatch that broke the first z build), so three objects persist across updates
and serialize with the checkpoint, exactly as they do there:

  * `density_b_zenc` -- the INDEX TOWER. A separate slow-EMA copy of density_b, init = a
    copy of density_b (bridge ~5729-5733), mixed at measure_density_z_enc_tau inside the
    update (~3849-3859). It is NOT the online tower and NOT `target_density_b`: the shipped
    centered arm asserts measure_density_z_enc_tau > 0 (~4837-4841), which selects
    `density_b_zenc` over `target_density_b` in `_density_z_enc_module` (~1125-1137),
    because running the solve through the fast-drifting target tower is the known
    non-stationarity risk.
  * `density_zstats.mu` / `.cov` -- mu_hat and C_hat (bridge `ZCenteredMomentsState` ~189),
    EMA'd at measure_density_z_sigma_tau over each batch's query set at the PRE-STEP index
    tower (~3903-3921). Init mu = 0, C = I, so the first steps solve (I + lam I)^-1 m_c ~
    m_c and the EMA walks onto the real moments with no cold-start blow-up.
  * `density_zstats.pool` / `.filled` -- the FROZEN M-row eval pool (bridge `ZEvalPoolState`
    ~218), a fixed draw of RAW hit-spec block coords from rho_phi written ONCE on the first
    update (~3921-3939). It is what lets a native-goal eval go through the REWARD-SAMPLE API
    (M = measure_density_z_eval_pool_m = 256 reward samples over the pool) instead of
    silently substituting a point prompt -- the substitution that made spec 29's .812
    uninterpretable.

All three live in `net` (as ordinary groups that never receive a gradient) rather than in
`target`, because `net` is the only container every consumer can reach -- `sample_cond` and
`infer_cond`/`map_eval_cond` are handed `net`, and `q_fn` sees them through its `params`
argument -- and because `net` is what `save_agent` serializes. They are written by the
additive `update_stats` hook in impls/agents/__init__.py, which is a no-op for every critic
that does not define it.

ONE REMAINING STRUCTURAL DIFFERENCE, semantically neutral: the shim keeps mu/C
(`ZCenteredMomentsState`) and the pool (`ZEvalPoolState`) as two separate registered
modules; here they are four leaves of one `density_zstats` group. Nothing reads the param
tree layout, and the checkpoints were never interchangeable (different repos).

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


# ── PIXEL towers (docs/specs/2026-08-21-density-fb-pixels-drq.md sections 4-5). ─────────
# TWO encoders, one per tower, mirroring FB's `_fw_encoder` / `_bw_encoder`
# (agents/fb/model.py) and TD-JEPA upstream. They are not a shared trunk: FB's optimizer
# split puts `_bw_encoder` on the BACKWARD optimizer (its own comment: "grad -> trains B
# end-to-end") and `_fw_encoder` on the FORWARD one. Here the two towers are the two seam
# GROUPS -- `density_b` and `density_f` -- so declaring one encoder inside each reproduces
# that split exactly, with no new group type and no new lr key.
#
# The ACTOR owns neither. It reads features through `cond` (published stop-gradient'd by
# `sample_cond`) and takes its Q-term through `q_fn`, which already stop-gradients the
# critic params -- the seam's analogue of FB's `with torch.no_grad():` around the actor's
# encode.
class _PixelDensityF(nn.Module):
    """F(s, a, g) on images: ONE encoder shared by the observation and the goal image, then
    the ensembled MLP on [enc(s), a, enc(g)].

    ONE encoder for both legs, not two: this is `_fw_encoder`, the F side's single view of
    the pixel world, and the goal is a state of that same world. `encode_cond=False` is the
    `measure_density_z` case, where the conditioning leg is already a latent."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    encoder_cls: Any
    num_parallel: int = 2
    encode_cond: bool = True

    def setup(self):
        self.encoder = self.encoder_cls()
        self.mlp = _ensemblize(_MLP, self.num_parallel)(
            self.hidden_dims, activate_final=False, layer_norm=self.layer_norm)

    def encode(self, x):
        """The encoder alone — what the actor's conditioning and observation legs read
        (always through a stop_gradient at the call site)."""
        return self.encoder(x)

    def __call__(self, obs, actions, cond):
        h = self.encoder(obs)
        c = self.encoder(cond) if self.encode_cond else cond
        return self.mlp(jnp.concatenate([h, actions, c], -1))


class _PixelDensityB(nn.Module):
    """B(u) on images: `_bw_encoder` then the backward MLP. Trained END-TO-END by the
    density loss, exactly as FB trains its `_bw_encoder` through the backward optimizer."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    encoder_cls: Any

    def setup(self):
        self.encoder = self.encoder_cls()
        self.mlp = _MLP(self.hidden_dims, activate_final=False, layer_norm=self.layer_norm)

    def encode(self, x):
        return self.encoder(x)

    def __call__(self, u):
        return self.mlp(self.encoder(u))


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


class _ZStatsState(nn.Module):
    """The persistent index state, held as ORDINARY PARAM LEAVES: mu_hat [d], C_hat [d, d],
    the frozen eval pool [M, d_c] of RAW block coords, and its `filled` flag.

    Combines the bridge's `ZCenteredMomentsState` (~189) and `ZEvalPoolState` (~218). They
    are stored the way the towers are stored -- as params, mixed by hand in the update --
    rather than recomputed per batch, for the two reasons the bridge gives:
      * they SERIALIZE, so every eval harness reads the IDENTICAL object the training-time
        conditioning used. Batch statistics are forbidden by the contract precisely because
        they reintroduce the train/eval mismatch that broke the first z build (8/8 unit
        tests green while the eval path was wrong, because no test drove the eval entry);
      * the ridge solve (C_hat + lam I)^-1 m_c is only a stable, comparable object if C_hat
        is quasi-stationary. A per-batch d x d covariance is a noisy, rank-deficient
        estimate that would make the index jitter step to step.

    Init mu = 0, C = I: at init the solve is (I + lam I)^-1 m_c ~ m_c and the EMA walks onto
    the real moments within ~1/tau steps (a zero-C init would blow up). The pool inits to
    zeros -- an eval before the first update then sees a constant kernel row, which the
    constant-reward fallback catches and reports rather than silently encoding noise; in
    practice `filled` is 1 after step 1.

    The gradient is identically zero on every leaf (they are only ever read at frozen params
    inside a stop_gradient), and the group is in no stage_groups, so `update_stats` is their
    sole writer."""
    dim: int
    pool_m: int
    d_c: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self):
        mu = self.param('mu', lambda _k: jnp.zeros((self.dim,), self.dtype))
        cov = self.param('cov', lambda _k: jnp.eye(self.dim, dtype=self.dtype))
        pool = self.param('pool',
                          lambda _k: jnp.zeros((self.pool_m, self.d_c), self.dtype))
        filled = self.param('filled', lambda _k: jnp.zeros((), self.dtype))
        return mu, cov, pool, filled


# ── THE SILENT-FAILURE GUARD (pixels spec section 5.2). ─────────────────────────────────
# This measure is DEFINED BY COLUMN SLICES. `_raw_block_cols` takes obs_rows[:, 19:22];
# `_delta_phi` slices both of its arguments the same way. Hand either one a (B, 64, 64, 3)
# image and jax slices the LAST axis without complaint: [:, 19:22] of a rank-4 array slices
# the HEIGHT axis and returns a (B, 3, 64, 3) tensor that broadcasts its way through the
# softmax; a flattened (B, 12288) image returns three arbitrary pixel intensities. Either way
# the kernel returns NUMBERS, training proceeds against a measure computed from pixel values,
# and nothing downstream -- not a shape error, not a nan, not a loss spike -- can tell you.
#
# So every array entering the kernel is checked at the boundary, on rank AND on exact width.
# Exact width is what catches the flattened-image case, which a rank check alone lets through.
#
# Shapes are STATIC under jax tracing, so these run once at trace time and add nothing to the
# compiled graph: the state path stays byte-identical (tests/impls/test_boot_contract.py).
class MeasureChannelError(TypeError):
    """An array reached the density kernel that is not an object-coordinate row block."""


def _blocks_width(blocks):
    """The concatenated width `_raw_block_cols`/`_density_queries` produce from `blocks`."""
    return int(sum(b - a for (a, b, _e) in blocks))


def _check_rows(arr, blocks, where, expect_dim=None):
    """`arr` must be [N, D] object-coordinate ROWS wide enough for every block in `blocks`
    (and exactly `expect_dim` wide when the critic knows what that is)."""
    need = max(b for (_a, b, _e) in blocks)
    shape = tuple(jnp.shape(arr))
    if len(shape) != 2 or shape[-1] < need:
        raise MeasureChannelError(
            f"{where}: the density measure needs [N, D] object-coordinate rows with "
            f"D >= {need} (the widest column in measure_hit_spec), got shape {shape}. "
            f"On pixels this is the image tensor arriving where the PRIVILEGED QPOS "
            f"CHANNEL belongs -- slicing it would return numbers computed from pixel "
            f"values and train a silently different measure. Pass the `*_qpos` channel "
            f"GCDataset emits under `measure_qpos_slices`.")
    if expect_dim is not None and shape[-1] != int(expect_dim):
        raise MeasureChannelError(
            f"{where}: expected the measure channel to be {int(expect_dim)} columns wide "
            f"(what this critic was built against), got shape {shape}. A width mismatch "
            f"here means the hit spec is indexing a different quantity than it was tuned "
            f"on -- e.g. a FLATTENED image, which passes a rank check but not this one.")


def _check_cols(arr, blocks, where):
    """`arr` must be [M, d_c] ALREADY-SLICED block columns (the `_raw_block_cols` output
    convention), i.e. exactly as wide as `blocks` concatenates to."""
    d_c = _blocks_width(blocks)
    shape = tuple(jnp.shape(arr))
    if len(shape) != 2 or shape[-1] != d_c:
        raise MeasureChannelError(
            f"{where}: the density kernel's query support must be [M, {d_c}] concatenated "
            f"block columns (`_raw_block_cols` convention), got shape {shape}.")


# ── the density kernel — _delta_phi / _density_queries / _raw_block_cols (bridge
# gciql_flow.py ~1403-1449), NOT shared with td_fb (td_fb uses _hit_matrix, not this). ──
def _density_queries(nobs, blocks):
    """[M, d_c] eps-scaled block coords of next_observations — the resampled query
    support U AND (on state) the B-tower input rows."""
    _check_rows(nobs, blocks, "_density_queries(nobs)")
    parts = [nobs[:, a:b] / e for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _raw_block_cols(obs_rows, blocks):
    """[N, d_c] RAW (unscaled) concatenated block columns — the twin of
    `_density_queries` used as the `protos` argument to `_delta_phi`, which applies its
    own per-block eps scaling internally. Do NOT pre-scale this input."""
    _check_rows(obs_rows, blocks, "_raw_block_cols(obs_rows)")
    parts = [obs_rows[:, a:b] for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _scale_block_cols(raw_cols, blocks):
    """The eps scaling of `_density_queries` applied to ALREADY-SLICED raw block columns
    ([*, d_c] in hit-spec order, i.e. `_raw_block_cols` output) rather than to full
    observation rows (bridge gciql_flow.py ~2173-2185).

    Needed by the frozen eval pool, which stores RAW coords because `_delta_phi` needs raw
    rows while the B tower needs the eps-scaled twin. The two conventions must not be
    crossed -- that is the double-scale trap this file's header warns about."""
    _check_cols(raw_cols, blocks, "_scale_block_cols(raw_cols)")
    parts, off = [], 0
    for (a, b, e) in blocks:
        w = b - a
        parts.append(raw_cols[..., off:off + w] / e)
        off += w
    return jnp.concatenate(parts, axis=-1)


def _delta_phi(obs_rows, protos, blocks):
    """softmax_j( -0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2 ), over the QUERY
    axis (last). `obs_rows` [N, obs_dim] RAW; `protos` [M, d_c] RAW block-concatenated
    columns (the `_raw_block_cols` convention) — eps scaling happens INTERNALLY here, so
    callers must feed both sides RAW (the `_density_queries` pre-scaled rows are the
    B-tower's own input, never this function's)."""
    _check_rows(obs_rows, blocks, "_delta_phi(obs_rows)")
    _check_cols(protos, blocks, "_delta_phi(protos)")
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

    # --- PIXEL path (docs/specs/2026-08-21-density-fb-pixels-drq.md), DEFAULT OFF ---------
    # `pixel` is set by `build` from the example observation's rank, so it cannot be
    # configured into disagreement with the data. With it False not one branch below is
    # taken and the state arm is byte-identical.
    #
    # THE TWO CHANNELS. On pixels every one of u (the query support) and g (the goal) exists
    # twice: as an IMAGE, which the towers encode, and as a PRIVILEGED OBJECT-COORDINATE ROW,
    # which the hit kernel slices. The image channel keeps the existing key names
    # (`_density_u_sc`, cond["goal"], cond["value_goal"]); the qpos channel rides alongside
    # under `_density_u_raw`, cond["goal_qpos"] and cond["value_goal_qpos"]. `qpos_dim` is the
    # exact width of that channel and is what `_check_rows` enforces at every kernel boundary.
    pixel: bool = False
    qpos_dim: int = 0
    enc_dim: int = 0                # encoder feature width; the actor's conditioning width

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
    z_enc_tau: float = 0.0005       # measure_density_z_enc_tau (tau_z): EMA rate of the index
                                    # tower density_b_zenc. The centered encoder REQUIRES > 0
                                    # (bridge assertion ~4837) -- that is what makes the index
                                    # tower the slow copy rather than the target tower.
    z_sigma_tau: float = 0.005      # measure_density_z_sigma_tau: EMA rate of mu_hat / C_hat
    z_eval_pool_m: int = 256        # measure_density_z_eval_pool_m: M for the frozen eval pool
    # The two index-state groups' modules. None when the index is off, in which case neither
    # group is registered and the param tree is byte-identical to the goal-conditioned arm.
    density_b_zenc: Any = None
    density_zstats: Any = None

    # An INSTANCE field: `build` widens it to carry the index-state groups when the index is
    # on. Agent.create reads it off the instance to size `net`; stage_groups deliberately
    # does NOT return it, so the state groups never enter a gradient tape.
    GROUPS: tuple = ("density_f", "density_b")
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
        actor rides z exactly as it otherwise rides the goal.

        On PIXELS with the index off it is the ENCODER width: the actor cannot be handed a
        (64, 64, 3) goal image to concatenate onto its input, so what it rides is the F-side
        encoder's embedding of that goal, detached (spec section 5.3)."""
        if self.z_on:
            return self.z_dim
        return self.enc_dim if self.pixel else self.obs_dim

    @property
    def actor_obs_dim(self):
        """Width of the OBSERVATION leg the actor's policy nets are sized to. On pixels the
        actor never sees the image -- it sees this critic's F-side encoder features, detached
        (spec section 5.3) -- so it is the encoder width, not `obs_dim` (which on an image is
        the channel count, 3). None off pixels, so Agent.create publishes nothing and every
        other actor sizes itself from ex_obs exactly as it always did."""
        return self.enc_dim if self.pixel else None

    # The critic's conditioning needs the target towers (see the Agent's `update` for why
    # a static declaration rather than signature sniffing).
    SAMPLE_COND_WANTS_TARGET = True

    def stage_groups(self, stage):
        assert stage == "density"
        # NOT self.GROUPS: under the index that also carries density_b_zenc/density_zstats,
        # which are running statistics written by update_stats and must never appear in a
        # gradient tape (their gradient is identically zero, but an optimizer step on them
        # would still be a second writer).
        return ("density_f", "density_b")

    # --- module apply helpers ---
    def _F(self, p, obs, acts, g):
        """F(s, a, g) -> [2, B, d]. On state the three legs are concatenated by the CALLER's
        equivalent of this line and handed to one MLP -- kept here so the pixel tower, which
        must encode `obs` and `g` separately, has the same call signature."""
        if self.pixel:
            return self.density_f.apply({"params": p}, obs, acts, g)
        return self.density_f.apply({"params": p}, jnp.concatenate([obs, acts, g], -1))

    def _B(self, p, u):
        """B(u) -> [M, d]. `u` is the eps-scaled query row on state, the query IMAGE on
        pixels (the tower encodes it)."""
        return self.density_b.apply({"params": p}, u)      # [M, d]

    def _enc_f(self, p, x):
        """F-side encoder alone, STOP-GRADIENT'd — the actor's view of an observation or a
        goal. Detaching here is the seam's `with torch.no_grad():` (spec section 4): the
        encoder is trained by the density loss and by nothing else.

        Rank-tolerant, because the eval loop hands `sample_actions` a SINGLE unbatched
        (H, W, C) frame while training always passes (B, H, W, C)."""
        x = jnp.asarray(x)
        unbatched = x.ndim == 3
        xb = x[None] if unbatched else x
        h = jax.lax.stop_gradient(
            self.density_f.apply({"params": p}, xb, method=_PixelDensityF.encode))
        return h[0] if unbatched else h

    def _kernel_rows(self, batch_or_cond, key, arr):
        """The OBJECT-COORDINATE rows for `key` — the ONLY thing the hit kernel may slice.

        On STATE the observation IS the coordinate row, so this returns `arr` unchanged (the
        same object, so the trace is identical). On PIXELS it returns the privileged channel
        `<key>_qpos`, gathered by GCDataset at exactly the row index `key` came from.

        INDEXED, never `.get()`-defaulted: a fallback to `arr` here is precisely the silent
        failure this whole path exists to prevent -- the image would slice, and the measure
        would quietly become a function of pixel values."""
        if not self.pixel:
            return arr
        qk = f"{key}_qpos"
        if qk not in batch_or_cond:
            raise MeasureChannelError(
                f"pixel density_fb needs {qk!r} -- the privileged object-coordinate twin of "
                f"{key!r}. GCDataset emits it when `measure_qpos_slices` is set, and "
                f"evaluate() builds it from the env. There is deliberately no fallback to "
                f"{key!r}: slicing the image would return numbers and train a different "
                f"measure with nothing failing.")
        rows = batch_or_cond[qk]
        _check_rows(rows, self.blocks, f"density_fb kernel channel {qk!r}",
                    expect_dim=self.qpos_dim)
        return rows

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
    def _zstats(self, net):
        """The persistent (mu_hat, C_hat, pool, filled) leaves. `net` is the agent's group
        dict; the same leaves reach `q_fn` through its `params` argument, so both spellings
        exist and neither recomputes anything from a batch."""
        return net["density_zstats"].params

    def _zenc(self, net):
        """The INDEX TOWER's params -- density_b_zenc, the slow-EMA copy of density_b. Never
        the online tower (whose values move every step) and never target_density_b: the
        shipped centered arm asserts measure_density_z_enc_tau > 0, which is exactly the
        condition that selects this tower (bridge `_density_z_enc_module` ~1125-1137)."""
        return net["density_b_zenc"].params

    def _z_centered_moment(self, zenc_params, u_sc, mu, r):
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
        b_u = jax.lax.stop_gradient(self._B(zenc_params, u_sc))         # [M, d]
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

    def _z_index(self, zenc_params, zstats, u_sc, r):
        """The one solve every caller goes through: reward samples -> index, at the STORED
        moments and the STORED index tower. Training rows, the native-goal eval and the
        harness-facing reward interface all funnel through here, so they cannot drift apart
        -- which is the entire point of persisting the moments rather than estimating them
        per call."""
        m_c = self._z_centered_moment(zenc_params, u_sc, zstats["mu"], r)
        return self._z_centered_index(m_c, zstats["cov"])

    def z_from_reward_samples(self, net, u_sc, r):
        """THE GENERAL REWARD INTERFACE: reward samples {(u_j, r_j)}_{j=1..M} -> index
        [B, d]. No goal anywhere. `u_sc` is an [M, d_c] eps-scaled query pool
        (`_density_queries` output) and `r` is [B, M] (or [M]) reward values on it."""
        return self._z_index(self._zenc(net), self._zstats(net), u_sc, r)

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

    def update_stats(self, net, pre_net, batch):
        """Write the three persistent index objects. Called once per update by
        Agent.apply_update AFTER the gradient step and the polyak targets; `pre_net` is the
        PRE-STEP online net, which is the source the bridge uses for all three
        (gciql_flow.py `update` reads self.network.params, not the post-step new_network).

        No-op with the index off -- a Python-level return, so nothing extra is traced and
        the goal-conditioned arm's update is byte-identical.

        Order matches the bridge: the index tower is mixed from the PRE-STEP online tower
        first (~3849-3859), then the moments are EMA'd over this batch's query set at the
        PRE-STEP index tower (~3903-3921, i.e. the OLD zenc, not the one just written), then
        the pool is frozen on the first update (~3921-3939)."""
        if not self.z_on:
            return net
        blocks = self.blocks
        nobs = batch["next_observations"]
        u_sc = _density_queries(nobs, blocks)                            # [M, d_c] eps-scaled

        # (1) the index tower: a slow-EMA copy of the PRE-STEP online density_b at tau_z.
        zenc_old = net["density_b_zenc"].params
        zenc_new = jax.tree_util.tree_map(
            lambda p, ep: p * self.z_enc_tau + ep * (1.0 - self.z_enc_tau),
            pre_net["density_b"].params, zenc_old)

        # (2) mu_hat / C_hat over the batch query set at the PRE-STEP index tower. The base
        #     measure is the batch next-state query marginal -- the ONE pinned rho_phi used
        #     for mu, C, rbar and m_c alike.
        b = jax.lax.stop_gradient(self._B(zenc_old, u_sc))                # [M, d]
        mu_b = jnp.mean(b, axis=0)
        db = b - mu_b[None]
        cov_b = db.T @ db / b.shape[0]                                    # centered, 1/M
        st = net["density_zstats"].params
        tau = self.z_sigma_tau
        mu_new = st["mu"] * (1.0 - tau) + mu_b * tau
        cov_new = st["cov"] * (1.0 - tau) + cov_b * tau
        cov_new = 0.5 * (cov_new + cov_new.T)                             # SYMMETRIZE

        # (3) the frozen eval pool: written ONCE, on the first update, from this batch's RAW
        #     query coords -- a fixed draw from rho_phi, so the native-goal eval index is a
        #     stable function of the params rather than a statistic of whichever batch
        #     happened to be last.
        raw = _raw_block_cols(nobs, blocks)
        if raw.shape[0] < self.z_eval_pool_m:
            raise ValueError(
                f"measure_density_z_eval_pool_m={self.z_eval_pool_m} needs a batch of at "
                f"least that many next states to draw the frozen eval pool from, got "
                f"{raw.shape[0]}. Lower the pool or raise batch_size -- do not let the pool "
                f"silently shrink, it is the M of the reward-sample eval API.")
        pool_new = jnp.where(st["filled"] > 0.5, st["pool"], raw[:self.z_eval_pool_m])

        # Cast back to the stored dtypes so the Agent pytree never changes dtype between
        # steps (which would retrace `update` every call).
        cast = lambda new, old: jnp.asarray(new, old.dtype)
        net = dict(net)
        net["density_b_zenc"] = net["density_b_zenc"].replace(params=zenc_new)
        net["density_zstats"] = net["density_zstats"].replace(params={
            "mu": cast(mu_new, st["mu"]), "cov": cast(cov_new, st["cov"]),
            "pool": cast(pool_new, st["pool"]),
            "filled": jnp.ones_like(st["filled"])})
        return net

    def map_eval_cond(self, net, cond, observations=None):
        """Eval-time hook: turn the env's raw goal into an index THROUGH THE REWARD-SAMPLE
        API, exactly as the bridge's `sample_actions` does under
        measure_density_z_prompt='sample' (`_density_z_eval_samples` ~1296-1307).

        The goal becomes M = measure_density_z_eval_pool_m reward samples over the FROZEN
        pool: r = kg(goal, pool) = _delta_phi(goal, pool_raw), the same query-set-normalized
        kernel the training rows integrate (the standing parity rule), and that reward vector
        goes through the identical centered solve. It NEVER shortcuts to an analytic point
        prompt -- that silent substitution is what made spec 29's number uninterpretable, and
        the point prompt is deliberately not implemented in this port at all.

        PRECEDENCE, which the caller owns: Agent.sample_actions merges `goals` into cond
        whenever it is given, so a goal ALWAYS wins over an index previously inferred by
        `infer_cond`. That is the bridge's split too (`sample_actions(obs, goal)` maps the
        goal; `sample_actions_with_z` rolls a precomputed z), and it is FBCritic's behaviour
        here as well. Consequence worth knowing: impls/main.py's eval loop always passes a
        goal, so --infer_cond_episodes sets a cond_eval that the very next sample_actions
        overwrites. Roll a reward-inferred index by calling sample_actions with goals=None.

        PIXEL BRANCH (spec section 7). The eval loop hands us `info['goal']`, which on a
        `visual-*` OGBench env is the goal IMAGE -- verified on this box. (The spec's section
        2 claim that `info['goal']` is "the goal in STATE form" is FALSE for visual envs:
        both `goal` and `goal_rendered` come back (64, 64, 3) uint8, because the env's
        `compute_observation` returns pixels.) So the image channel is cond["goal"] and the
        object-coordinate channel must be supplied separately as cond["goal_qpos"], which
        impls/utils/evaluation.py builds from the env's task info. Here we encode the goal
        image into the actor's conditioning vector and the current observation into the
        actor's observation leg -- exactly the two things `sample_cond` publishes at training
        time, so the policy sees the same inputs in both regimes."""
        if self.pixel:
            out = dict(cond)
            fp = net["density_f"].params
            if "goal" in cond:
                out["_cvec"] = self._enc_f(fp, cond["goal"])
            if observations is not None:
                out["_obs_feat"] = self._enc_f(fp, observations)
            return out
        if not self.z_on or "goal" not in cond:
            return cond
        g = cond["goal"]
        unbatched = g.ndim == 1
        g2 = g[None] if unbatched else g
        st = self._zstats(net)
        pool_raw = st["pool"]                                       # [M, d_c] RAW
        u_sc = _scale_block_cols(pool_raw, self.blocks)             # [M, d_c] eps-scaled
        kg = jax.lax.stop_gradient(_delta_phi(g2, pool_raw, self.blocks))   # [B, M]
        z = self._z_index(self._zenc(net), st, u_sc, kg)
        return {**cond, "z": z[0] if unbatched else z}

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

        # THE TWO CHANNELS. `nobs_q` is the object-coordinate row the kernel slices; `nobs`
        # is what the B tower embeds. On state they are one and the same array (byte-
        # identical); on pixels `nobs` is the image and `nobs_q` is the privileged qpos
        # channel GCDataset gathered at the SAME row index.
        nobs_q = self._kernel_rows(batch, "next_observations", nobs)
        u_raw = _raw_block_cols(nobs_q, blocks)                           # [M, d_c] RAW
        # The B tower's input: eps-scaled query rows on state, the query IMAGE on pixels.
        u_sc = nobs if self.pixel else _density_queries(nobs_q, blocks)

        b_emb = self._B(params["density_b"], u_sc)                        # [M, d]
        f_emb = self._F(params["density_f"], obs, acts, g)                # [2,B,d]
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
        tf = self._F(target["density_f"], nobs, a_plus, g)
        w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)  # [2, B, M]

        if self.hard:
            # spec 21: one-hot immediate target at the row's OWN successor (M == B because
            # the query support is this batch's next states). All (1-gamma) mass on the
            # observed next state; the soft eps-kernel was diluting the readout.
            k = jnp.eye(nobs.shape[0])
        else:
            # The kernel is counted at the NEXT state, in OBJECT COORDINATES -- `nobs_q`,
            # never the image.
            k = _delta_phi(nobs_q, u_raw, blocks)                          # [B, M]
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
            # is FB's own linear model implied by the index, r_z(u) = B~(u)^T z / sqrt(d),
            # read on the SAME query support the measure is trained against.
            #
            # B~ is the TARGET backward tower, matching the frozen-params discipline of the
            # goal-indexed branch's kg. q_fn is handed online params only, so sample_cond
            # publishes that embedding as cond["_density_tb_u"] -- indexed, not .get()-
            # defaulted, because falling back to the online tower here would be a silent
            # tower substitution rather than a loud failure.
            g = cond["z"]
            kg = jax.lax.stop_gradient(
                jnp.einsum('md,bd->bm', cond["_density_tb_u"], g)
                / jnp.sqrt(float(self.z_dim)))
        else:
            # TWO CHANNELS AGAIN: F is conditioned on the goal the tower understands (the
            # goal IMAGE on pixels), while the reward vector kg is the hit kernel, which only
            # means anything in object coordinates.
            g = cond["goal"]
            g_rows = self._kernel_rows(cond, "goal", g)
            kg = jax.lax.stop_gradient(_delta_phi(g_rows, u_raw, self.blocks))  # [B, M]
        f_emb = self._F(f_p, obs, actions, g)
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
        """The vector actors concatenate onto obs. The raw actor goal; with the reward-sample
        index on, the index built at that goal; on PIXELS, the F-side encoder's DETACHED
        embedding of the goal image (a policy net cannot concatenate a (64, 64, 3) tensor
        onto its input, and the actor owns no encoder of its own -- spec section 5.3)."""
        if self.z_on:
            return cond["z"]
        return cond["_cvec"] if self.pixel else cond["goal"]

    def sample_cond(self, net, batch, rng, target=None):
        # queries resampled every update from THIS batch's own next-state object coords.
        nobs = batch["next_observations"]
        nobs_q = self._kernel_rows(batch, "next_observations", nobs)
        u_raw = _raw_block_cols(nobs_q, self.blocks)
        # The B tower's query rows: eps-scaled object coords on state, the query IMAGE on
        # pixels. `_density_u_raw` keeps its meaning on both (raw block columns).
        u_sc = nobs if self.pixel else _density_queries(nobs_q, self.blocks)
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
            #
            # The index is built at the STORED tower and the STORED moments -- never a
            # statistic of this batch. update_stats is their only writer.
            if target is None:
                raise ValueError(
                    "density_fb under measure_density_z needs the TARGET params in "
                    "sample_cond (its actor-q reward vector is read off target_density_b). "
                    "Agent.update passes them because the critic declares "
                    "SAMPLE_COND_WANTS_TARGET; a direct caller must pass target=... too.")
            zenc_p, st = self._zenc(net), self._zstats(net)
            r_a, r_v = jax.random.split(rng)

            def _z_at(goals, r_mix):
                r = jax.lax.stop_gradient(_delta_phi(goals, u_raw, self.blocks))  # [B, M]
                return self._z_mix(self._z_index(zenc_p, st, u_sc, r), r_mix)

            return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                    "z": _z_at(batch["actor_goals"], r_a),
                    "_cvec_vgoal": _z_at(batch["value_goals"], r_v),
                    # The TARGET backward embedding of the query support, for q_fn's reward
                    # vector r_z = B~(u)^T z / sqrt(d) (q_fn gets online params only).
                    "_density_tb_u": jax.lax.stop_gradient(
                        self._B(target["density_b"], u_sc)),
                    "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        cond = {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                "_cvec_vgoal": batch["value_goals"],
                "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        if self.pixel:
            # Everything the ACTOR needs, published here because the actor cannot reach the
            # encoder any other way: `sample_bootstrap`/`sample_actions` are handed `net` and
            # `cond` but never the critic. Each is DETACHED (`_enc_f` stop-gradients), which
            # is what makes the actor's stage move no encoder parameter -- FB's
            # `with torch.no_grad():` expressed through the seam.
            fp = net["density_f"].params
            cond.update({
                # The qpos twin of the ACTOR goal: q_fn's reward vector kg is
                # `_delta_phi(goal, u_raw)`, and cond["goal"] is an image.
                "goal_qpos": self._kernel_rows(batch, "actor_goals", None),
                # The value goal's twin. NOTHING READS THIS TODAY -- stage_loss conditions F
                # on the value-goal IMAGE and takes its kernel at the next state, not at the
                # goal. It is published because the two goal legs must stay paired: any
                # future kernel evaluated at the value goal (a sarsa-style readout, the
                # reward index) would otherwise reach for cond["value_goal"] and get pixels.
                "value_goal_qpos": self._kernel_rows(batch, "value_goals", None),
                # the actor's conditioning leg at the actor goal and at the value goal
                "_cvec": self._enc_f(fp, batch["actor_goals"]),
                "_cvec_vgoal": self._enc_f(fp, batch["value_goals"]),
                # the actor's OBSERVATION leg -- its policy/flow nets take features, not
                # images (they concatenate obs with the conditioning vector).
                "_obs_feat": self._enc_f(fp, batch["observations"]),
                "_obs_feat_next": self._enc_f(fp, nobs),
            })
        return cond

    def eval_cond0(self, ex_obs, config):
        if self.z_on:
            # Placeholder until infer_cond is called, mirroring FBCritic.eval_cond0. A zero
            # z is OFF the sqrt(d) sphere and therefore out of the conditioning
            # distribution; the point of --infer_cond_episodes is that it gets replaced.
            return {"z": jnp.zeros((self.z_dim,), jnp.float64)}
        if self.pixel:
            # A single blank FRAME, so the cond is well-formed before evaluate() supplies the
            # real goal image. `evaluate` always passes `goals`, so this is never rolled on.
            return {"goal": jnp.zeros(tuple(ex_obs.shape[1:]), jnp.uint8)}
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
        if self.pixel:
            # Not a placeholder: on pixels `obs_dim` is the CHANNEL COUNT (3), so the
            # state branch's zero-goal would hand the eval path a 3-vector where an image
            # belongs. And the z branch cannot run here at all (see `build`).
            raise NotImplementedError(
                "density_fb has no reward-sample task interface on pixels: the index "
                "(measure_density_z) is not implemented there, so --infer_cond_episodes has "
                "nothing to infer. Evaluate against the env goal instead.")
        if not self.z_on:
            return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        z = self.z_from_reward_samples(net, _density_queries(obs, self.blocks), rewards)
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
        # PIXEL MODE is inferred from the DATA, not configured: an observation of rank > 2
        # (i.e. [B, H, W, C]) is an image and there is no honest way to run the state path on
        # it. Configuring this instead would let a config say "state" about image data, which
        # is exactly the disagreement the kernel guard exists to make impossible.
        pixel = int(jnp.ndim(ex_obs)) > 2
        z_on = bool(config.get("measure_density_z", False))
        if pixel and z_on:
            # BEFORE the z_on assertion block, because this is the more fundamental refusal:
            # the index's FROZEN EVAL POOL stores RAW object-coordinate columns and re-scales
            # them for the B tower (`_scale_block_cols` -> `_z_index`); on pixels the B tower
            # consumes IMAGES, so that pool cannot be fed to it at all. Making the pool store
            # images instead is a design change (~256 frames carried in the checkpoint), not a
            # wiring fix -- docs/specs/2026-08-21-density-fb-pixels-drq.md section 10.
            raise NotImplementedError(
                "measure_density_z on pixels is not implemented and must not be faked: the "
                "index's FROZEN EVAL POOL holds object-coordinate columns, which the pixel B "
                "tower (which consumes images) cannot be fed.")
        z_rand_frac = float(config.get("measure_density_z_rand_frac", 0.5))
        z_lam = float(config.get("measure_density_z_ridge_lambda_mult", 0.01))
        z_floor = float(config.get("measure_density_z_cov_eig_floor", 1e-6))
        z_eps = float(config.get("measure_density_z_const_eps", 1e-6))
        z_enc_tau = float(config.get("measure_density_z_enc_tau", 0.0005))
        z_sigma_tau = float(config.get("measure_density_z_sigma_tau", 0.005))
        z_pool_m = int(config.get("measure_density_z_eval_pool_m", 256))
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
            # The centered encoder REQUIRES a slow index tower (bridge assertion ~4837):
            # tau_z = 0 would select target_density_b, and running the solve through the
            # fast-drifting target tower is the known non-stationarity risk.
            assert 0.0 < z_enc_tau <= 0.1, \
                "measure_density_z_enc_tau (tau_z) must be in (0, 0.1]: the centered " \
                "encoder REQUIRES a slow-EMA index tower, and tau_z=0 would encode with " \
                f"the fast target tower instead. Round-1 dose 0.0005; got {z_enc_tau}"
            assert 0.0 < z_sigma_tau <= 1.0, \
                "measure_density_z_sigma_tau (the mu_hat/C_hat EMA rate) must be in (0, 1]" \
                f"; got {z_sigma_tau}. 0 would freeze C_hat at its identity init"
            assert z_pool_m > 1, \
                "measure_density_z_eval_pool_m must be > 1: it is the M of the reward-" \
                "sample eval API, and the centered moment is identically zero at M=1"
            _bs = int(config.get("batch_size", z_pool_m))
            assert z_pool_m <= _bs, \
                f"measure_density_z_eval_pool_m={z_pool_m} exceeds batch_size={_bs}; the " \
                f"frozen pool is drawn from one batch's next states"

        rf, rb, rz = jr.split(jr.PRNGKey(config["seed"] + 6000), 3)
        qpos_dim, enc_dim = 0, 0
        if pixel:
            from impls.utils.datasets import parse_qpos_slices
            from impls.utils.encoders import encoder_modules
            enc_name = config.get("encoder")
            if enc_name is None:
                raise KeyError(
                    "image observations need config['encoder'] (a key of "
                    "impls.utils.encoders.encoder_modules, e.g. 'drq'); there is no default "
                    "visual front-end and running an MLP on raw pixels is not one.")
            if enc_name not in encoder_modules:
                raise KeyError(f"unknown encoder {enc_name!r}; known: {sorted(encoder_modules)}")
            encoder_cls = encoder_modules[enc_name]
            enc_dim = int(encoder_cls().feature_dim)
            qslices = parse_qpos_slices(config.get("measure_qpos_slices"))
            if qslices is None:
                raise KeyError(
                    "image observations need config['measure_qpos_slices'] (e.g. '14:17' on "
                    "visual-cube-single-play-v0). density_fb's hit kernel and query support "
                    "are literal column slices of object coordinates; on an image there is "
                    "nothing to slice, so the measure is undefined without the privileged "
                    "channel (spec section 1). This is the asymmetry the spec says to report, "
                    "not a limitation to paper over.")
            # `qslices` are (a, b) pairs (no eps -- they say where the channel comes FROM,
            # not how the kernel scales it), unlike hit-spec blocks, which are (a, b, eps).
            qpos_dim = int(sum(b - a for (a, b) in qslices))
            need = max(b for (_a, b, _e) in blocks)
            if qpos_dim < need:
                raise ValueError(
                    f"measure_hit_spec reaches column {need} but measure_qpos_slices builds "
                    f"only a {qpos_dim}-column channel; the kernel would slice past its end")
            density_f = _PixelDensityF(hidden_dims=(*hd, d), layer_norm=ln,
                                       encoder_cls=encoder_cls, num_parallel=2,
                                       encode_cond=True)
            density_b = _PixelDensityB(hidden_dims=(*hd, d), layer_norm=ln,
                                       encoder_cls=encoder_cls)
            params = {
                # the goal leg is initialized at an IMAGE, because that is what it is fed
                "density_f": density_f.init(rf, ex_obs, ex_actions, ex_obs)["params"],
                "density_b": density_b.init(rb, ex_obs)["params"],
            }
        else:
            density_f = _ensemblize(_MLP, 2)((*hd, d), activate_final=False, layer_norm=ln)
            density_b = _MLP((*hd, d), activate_final=False, layer_norm=ln)
            # F's conditioning leg is the index (width d) under measure_density_z, the raw
            # goal (width obs_dim) otherwise -- the tower has to be INITIALIZED at that width
            # or the first stage_loss dies with a ScopeParamShapeError.
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
        groups = ("density_f", "density_b")
        zstats_mod = None
        if z_on:
            # The index tower starts as a COPY of density_b (bridge ~5729-5733), not at an
            # independent init: at step 0 the index is the one density_b itself implies, and
            # the EMA then holds it quasi-stationary while density_b trains away.
            params["density_b_zenc"] = copy.deepcopy(params["density_b"])
            d_c = int(sum(b - a for (a, b, _e) in blocks))
            zstats_mod = _ZStatsState(dim=d, pool_m=z_pool_m, d_c=d_c,
                                      dtype=jnp.zeros((), ex_obs.dtype).dtype)
            params["density_zstats"] = zstats_mod.init(rz)["params"]
            groups = groups + ("density_b_zenc", "density_zstats")
        obj = cls(density_f=density_f, density_b=density_b, obs_dim=obs_dim, blocks=blocks,
                  discount=float(config["discount"]),
                  temp=float(config["measure_density_temp"]), tau=float(config["tau"]),
                  actor_q_center=bool(config.get("actor_q_center", True)),
                  # DEFAULT TRUE. The yaml has said true since the bridge-port spec's item
                  # A2, but this fallback said False -- so any path building the critic from
                  # a partial config silently got the losing variant. Now they agree.
                  hard=bool(config.get("measure_density_hard", True)),
                  pixel=pixel, qpos_dim=qpos_dim, enc_dim=enc_dim,
                  bootstrap=str(config.get("measure_density_bootstrap", "actor")),
                  actor_temp=config.get("measure_density_actor_temp", None),
                  z_on=z_on, z_dim=d, z_rand_frac=z_rand_frac,
                  z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                  z_enc_tau=z_enc_tau, z_sigma_tau=z_sigma_tau, z_eval_pool_m=z_pool_m,
                  GROUPS=groups,
                  # `density_b_zenc` shares density_b's module DEF (same architecture); only
                  # the params differ. Agent.create reads these off the critic to build the
                  # TrainStates for the two state groups.
                  density_b_zenc=(density_b if z_on else None),
                  density_zstats=zstats_mod,
                  # Declaring "z" is what tells check_combo (and any actor that inspects the
                  # critic) that this critic's conditioning is a latent, not a goal.
                  PROVIDES_COND=(frozenset({"value_goal", "goal", "z"}) if z_on
                                 else frozenset({"value_goal", "goal"})))
        return obj, params, targets
