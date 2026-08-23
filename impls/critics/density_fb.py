"""DensityFBCritic — continuous normalized successor density (bridge spec 18).

Two-tower softmax density f(s,a,g,u) = F(s,a,g)^T B(u) / tau over a query support U
resampled every update from the batch's own next-state object coordinates. Ported from
the bridge's tools/wandb_mode_shim/gciql_flow.py; provenance, rationale and the
supporting measurements are in docs/density-fb-design.md.

Contract, each pinned by a test in tests/impls/test_density_fb.py:
  * rewards/kernel are counted at the NEXT state (x_{t+1});
  * `_delta_phi` eps-scales INTERNALLY — the kernel term gets RAW block columns
    (`_raw_block_cols`), the B tower gets PRE-SCALED queries (`_density_queries`); feeding
    the wrong one to the wrong callsite silently trains a different object;
  * the softmax is over the QUERY axis (last), not the batch axis;
  * the bootstrap action is drawn at the VALUE goal, `boot["next_action_vgoal"]`, not the
    seam's default `boot["next_action"]`, which conditions on the ACTOR goal. Reading the
    actor-goal action makes w_bar a goal-averaged density: measured density_ess_median
    ~310 against a healthy ~89 at M=B=1024;
  * queries are resampled every update; there is no stored or cached support;
  * NO orthonormalization, mass anchor or scale normalization — each measured separately
    harmful on the bridge (ortho took cube-single 0.967 -> 0.225).

`measure_density_z` (default FALSE) swaps the policy-family index from the goal to a
REWARD-SAMPLE index z, so a task can be given as {(u_j, r_j)} instead of a goal. Only the
spec-30 translation-gauge-invariant centered dual encoder is ported:

    m_c  = (1/M) sum_j (B(u_j) - mu_hat)(r_j - rbar),      rbar = mean_j r_j
    c    = (C_hat + lam I)^-1 m_c,  lam = lambda_mult * tr(C_hat)/d
    zhat = sqrt(d) * c / ||c||          (all-ones fallback when ||c|| < const_eps)

exactly invariant to r -> r + beta, to r -> alpha r (alpha > 0) and to B -> B + b. It has
NO one-sample/Dirac limit: m_c is identically zero at M=1, so M=1 raises. The spec-29
uncentered encoder, its analytic point prompt and every orthogonality term are
deliberately not ported.

Three index objects persist across updates and serialize with the checkpoint, because
batch statistics reintroduce the train/eval mismatch that broke the first z build:
`density_b_zenc` (the slow-EMA index tower — NOT the online tower and NOT
`target_density_b`), `density_zstats.mu`/`.cov` (mu_hat, C_hat), and
`density_zstats.pool`/`.filled` (the frozen eval pool that lets a native-goal eval go
through the reward-sample API rather than a point prompt). All three live in `net`, the
only container every consumer can reach and the one `save_agent` serializes; the additive
`update_stats` hook in impls/agents/__init__.py is their sole writer.

The tower modules are copied here rather than imported from impls/utils/networks.py, so
this file's isolation from other critics is structural. The only symbol shared with
td_fb.py is `impls.critics._measure_common.parse_hit_spec`.
"""
import copy
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.critics._measure_common import parse_hit_spec

# ── tower primitives, copied from the OGBench networks.py the bridge imports them from,
# so the density towers' init and activation match it. ──
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
# TWO encoders, one per tower, not a shared trunk: the towers are the two seam GROUPS, so
# one encoder inside each reproduces FB's forward/backward optimizer split. The ACTOR owns
# neither -- it reads features through `cond` and its Q-term through `q_fn`, both detached.
class _PixelDensityF(nn.Module):
    """F(s, a, g) on images: ONE encoder (`_fw_encoder`, the F side's single view of the
    pixel world) shared by the observation and the goal image, then the ensembled MLP on
    [enc(s), a, enc(g)]. `encode_cond=False` is the `measure_density_z` case, where the
    conditioning leg is already a latent."""

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
        (always behind a stop_gradient at the call site)."""
        return self.encoder(x)

    def __call__(self, obs, actions, cond):
        h = self.encoder(obs)
        c = self.encoder(cond) if self.encode_cond else cond
        return self.mlp(jnp.concatenate([h, actions, c], -1))


class _PixelDensityB(nn.Module):
    """B(u) on images: `_bw_encoder` then the backward MLP, trained END-TO-END by the
    density loss as FB trains its `_bw_encoder` through the backward optimizer."""

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


# ── sqrt(d)-sphere normalizations, transcribed from the bridge. ──
def _sphere(x, d):
    """Project x onto the sphere of radius sqrt(d) along the last axis — FB pins
    ||B||=||z||=sqrt(d). The +1e-6 denominator guard makes the map only approximately
    scale-invariant, so the centered index uses `_sphere_or_neutral` instead; this form is
    kept for the task-prior draws, where the argument is a fresh gaussian."""
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _sphere_or_neutral(x, d, eps):
    """EXACT sqrt(d)-normalization (no epsilon in the denominator) with the
    degenerate/constant-reward fallback z_neutral = 1_d, which itself has norm sqrt(d).

    The centered encoder's claim is EXACT invariance to r -> alpha r and to the additive
    gauge, which `_sphere`'s guard does not give (docs/density-fb-design.md). ||x|| ~ 0 is
    handled explicitly because a near-constant reward is a real input, not a numerical
    accident."""
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    degenerate = nrm < eps
    safe = jnp.where(degenerate, jnp.ones_like(nrm), nrm)
    return jnp.where(degenerate, jnp.ones_like(x), x * jnp.sqrt(d * 1.0) / safe)


class _ZStatsState(nn.Module):
    """The persistent index state as ORDINARY PARAM LEAVES: mu_hat [d], C_hat [d, d], the
    frozen eval pool [M, d_c] of RAW block coords, and its `filled` flag.

    Stored as params and mixed by hand in the update rather than recomputed per batch, so
    that they SERIALIZE: every eval harness then reads the IDENTICAL object the
    training-time conditioning used. Init mu = 0, C = I keeps the first solve well-posed
    (a zero-C init would blow up); the pool inits to zeros and `filled` is 1 after step 1.

    The gradient is identically zero on every leaf and the group is in no stage_groups, so
    `update_stats` is their sole writer. Rationale: docs/density-fb-design.md."""
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


# ── THE SILENT-FAILURE GUARD. ───────────────────────────────────────────────────────────
# The measure is DEFINED BY COLUMN SLICES, and jax slices an image without complaint: the
# kernel returns NUMBERS computed from pixel values and nothing downstream -- no shape
# error, no nan, no loss spike -- can tell you. So every array entering the kernel is
# checked on rank AND exact width; exact width is what catches a flattened image, which a
# rank check alone lets through. Shapes are static, so the checks cost nothing at runtime.
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


# ── the density kernel; NOT shared with td_fb, which uses _hit_matrix instead. ──
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
    observation rows.

    Needed by the frozen eval pool, which stores RAW coords because `_delta_phi` needs raw
    rows while the B tower needs the eps-scaled twin. Crossing the two conventions is the
    double-scale trap."""
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
    # W1: 'actor' (default, bit-identical) bootstraps the TD target at the actor's action,
    # putting the actor inside the critic's fixed point; 'sarsa' bootstraps at the DATASET's
    # next action instead. See docs/density-fb-design.md.
    bootstrap: str = "actor"
    # W2: temperature for the ACTOR READOUT ONLY (q_fn's softmax). None -> `temp`, i.e.
    # bit-identical; the CE training target in stage_loss always keeps `temp`.
    actor_temp: Any = None

    # --- PIXEL path (docs/specs/2026-08-21-density-fb-pixels-drq.md), DEFAULT OFF ---------
    # `pixel` is set by `build` from the observation's rank, so it cannot be configured into
    # disagreement with the data; with it False the state arm is byte-identical. On pixels
    # each of u and g exists twice, as an IMAGE the towers encode and as a PRIVILEGED
    # OBJECT-COORDINATE ROW the kernel slices; `qpos_dim` is the width `_check_rows` enforces.
    pixel: bool = False
    qpos_dim: int = 0
    enc_dim: int = 0                # encoder feature width; the actor's conditioning width

    # --- spec 30 P1 reward-sample index (measure_density_z, DEFAULT OFF) ------------------
    # Read ONLY when z_on is True, so the goal-conditioned arm traces no extra op and stays
    # byte-identical (tests/impls/test_boot_contract.py).
    z_on: bool = False              # measure_density_z
    z_dim: int = 0                  # measure_density_dim, echoed here so cvec_dim can read it
    z_rand_frac: float = 0.5        # fraction of rows replaced by a fresh sqrt(d)-sphere draw
    z_ridge_lambda_mult: float = 0.01   # lam = mult * tr(C)/d
    z_cov_eig_floor: float = 1e-6   # eigenvalue floor on C_hat, a multiple of tr(C)/d
    z_const_eps: float = 1e-6       # ||c|| below which the reward counts as constant
    z_enc_tau: float = 0.0005       # tau_z, the index tower's EMA rate; must be > 0
    z_sigma_tau: float = 0.005      # EMA rate of mu_hat / C_hat
    z_eval_pool_m: int = 256        # M for the frozen eval pool
    # The index-state modules; None with the index off, so no group is registered.
    density_b_zenc: Any = None
    density_zstats: Any = None

    # An INSTANCE field: `build` widens it to carry the index-state groups when the index is
    # on. stage_groups deliberately does NOT return them, so they never enter a gradient tape.
    GROUPS: tuple = ("density_f", "density_b")
    STAGES = ("density",)
    PROVIDES_V = False
    # An INSTANCE field (annotated): `build` widens it to include "z" when the index is on.
    # The class-level value is the goal-conditioned default that class-level consumers see.
    PROVIDES_COND: frozenset = frozenset({"value_goal", "goal"})

    # infer_cond's base measure: the pinned rho_phi the centered solve is estimated against.
    # Unannotated, so it stays a CLASS attribute; with the index OFF `obs` is ignored.
    INFER_COND_MARGINAL = "next_observations"

    # F(s, a, g): the row is g-indexed, so the TD target must bootstrap at that same g.
    # Reading the actor-goal action instead makes w_bar a goal-averaged (behaviour-like)
    # density -- measured density_ess_median ~310 against a healthy ~89.
    BOOT_COND = "value_goal"

    @property
    def TARGETS(self):
        return {"density_f": self.tau, "density_b": self.tau}

    @property
    def cvec_dim(self):
        """Width of cond_vec(cond) — what the actor's conditioning leg is sized to. Under the
        reward-sample index that is the index width d; on PIXELS it is the ENCODER width,
        since the actor rides the F-side encoder's detached embedding of the goal image
        rather than the (64, 64, 3) image itself."""
        if self.z_on:
            return self.z_dim
        return self.enc_dim if self.pixel else self.obs_dim

    @property
    def actor_obs_dim(self):
        """Width of the OBSERVATION leg the actor's policy nets are sized to. On pixels the
        actor sees this critic's detached F-side encoder features, so it is the encoder
        width, not `obs_dim` (on an image, the channel count 3). None off pixels, so every
        other actor sizes itself from ex_obs unchanged."""
        return self.enc_dim if self.pixel else None

    # The critic's conditioning needs the target towers.
    SAMPLE_COND_WANTS_TARGET = True

    def stage_groups(self, stage):
        assert stage == "density"
        # NOT self.GROUPS: under the index that also carries density_b_zenc/density_zstats,
        # running statistics that update_stats owns and no optimizer may step.
        return ("density_f", "density_b")

    def _F(self, p, obs, acts, g):
        """F(s, a, g) -> [2, B, d]. On state the three legs are concatenated and handed to
        one MLP; wrapped so the pixel tower, which encodes `obs` and `g` separately, keeps
        the same call signature."""
        if self.pixel:
            return self.density_f.apply({"params": p}, obs, acts, g)
        return self.density_f.apply({"params": p}, jnp.concatenate([obs, acts, g], -1))

    def _B(self, p, u):
        """B(u) -> [M, d]. `u` is the eps-scaled query row on state, the query IMAGE on
        pixels (the tower encodes it)."""
        return self.density_b.apply({"params": p}, u)      # [M, d]

    def _enc_f(self, p, x):
        """F-side encoder alone, STOP-GRADIENT'd — the actor's view of an observation or a
        goal. The encoder is trained by the density loss and by nothing else.

        Rank-tolerant: the eval loop hands `sample_actions` a single unbatched (H, W, C)
        frame while training always passes (B, H, W, C)."""
        x = jnp.asarray(x)
        unbatched = x.ndim == 3
        xb = x[None] if unbatched else x
        h = jax.lax.stop_gradient(
            self.density_f.apply({"params": p}, xb, method=_PixelDensityF.encode))
        return h[0] if unbatched else h

    def _kernel_rows(self, batch_or_cond, key, arr):
        """The OBJECT-COORDINATE rows for `key` — the ONLY thing the hit kernel may slice.

        On STATE the observation IS the coordinate row, so this returns `arr` unchanged. On
        PIXELS it returns the privileged channel `<key>_qpos`, gathered by GCDataset at
        exactly the row index `key` came from.

        INDEXED, never `.get()`-defaulted: a fallback to `arr` is the silent failure this
        path exists to prevent -- the image would slice and the measure would quietly become
        a function of pixel values."""
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

    # ── spec 30 P1: the centered dual reward-sample index. ──────────────────────────────
    # ONE pinned base measure rho_phi = the next-state query marginal, used for mu_hat,
    # C_hat, rbar and m_c alike. Invariant to B -> B + b, r -> r + beta and r -> alpha r
    # (alpha > 0); the FULL linear gauge B -> A B, F -> A^-T F is NOT fixed.
    #
    # ⚠ Under the plug-in moments the reward-centering (r - rbar) and the B-centering
    # (B - mu) each kill both gauges alone and MASK each other, so deleting either leaves
    # every end-to-end invariance test green. Both are kept. Judge a change here by the
    # formula-level tests in tests/impls/test_density_fb_reward_index.py section 3b, never
    # by the end-to-end invariance tests. Details: docs/density-fb-design.md.
    def _zstats(self, net):
        """The persistent (mu_hat, C_hat, pool, filled) leaves. The same leaves reach `q_fn`
        through its `params` argument; neither spelling recomputes anything from a batch."""
        return net["density_zstats"].params

    def _zenc(self, net):
        """The INDEX TOWER's params -- density_b_zenc, the slow-EMA copy of density_b. Never
        the online tower and never target_density_b: the shipped centered arm asserts
        measure_density_z_enc_tau > 0, which is what selects this tower."""
        return net["density_b_zenc"].params

    def _z_centered_moment(self, zenc_params, u_sc, mu, r):
        """m_c = (1/M) sum_j (B(u_j) - mu_hat)(r_j - rbar), rbar = (1/M) sum_j r_j.

        `u_sc` [M, d_c] eps-scaled queries, `r` [B, M] (or [M]) rewards on them -> [B, d].
        M = 1 RAISES: the centered moment is identically zero there, so a one-sample call is
        a caller error, not a degraded estimate. Shapes are static in jax, so the raise
        surfaces at the call site rather than as a silent zero index."""
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

        C_hat is symmetrized against float drift and its spectrum FLOORED before the solve:
        eigenvalues below z_cov_eig_floor * tr(C)/d are lifted to that floor, because the
        centered covariance genuinely has near-zero directions (measured effective rank
        6-21 of 50) and an unfloored inverse would amplify the noise in them.

        lam = z_ridge_lambda_mult * tr(C_hat)/d is the mean-eigenvalue convention, computed
        from the UNFLOORED trace so the number matches the bridge's preflight probes."""
        d = float(self.z_dim)
        cov = 0.5 * (cov + cov.T)
        tr = jnp.trace(cov)
        ev, vec = jnp.linalg.eigh(cov)                                  # ascending
        ev = jnp.maximum(ev, self.z_cov_eig_floor * tr / d)
        lam = self.z_ridge_lambda_mult * tr / d
        return ((m_c @ vec) / (ev + lam)[None, :]) @ vec.T              # [B, d]

    def _z_centered_index(self, m_c, cov):
        """zhat = sqrt(d) * normalize(c), with the CONSTANT-REWARD FALLBACK: a near-constant
        reward has m_c ~ 0, so ||c|| ~ 0 and the normalization would amplify numerical noise
        into a confident-looking task index. When ||c|| < z_const_eps the encoder returns
        z_neutral = 1_d, a reward-free direction that still carries ||z|| = sqrt(d) and so is
        IN the conditioning distribution, unlike a zero vector."""
        c = self._z_centered_coef(m_c, cov)
        return jax.lax.stop_gradient(
            _sphere_or_neutral(c, float(self.z_dim), self.z_const_eps))

    def _z_index(self, zenc_params, zstats, u_sc, r):
        """The one solve every caller goes through: reward samples -> index, at the STORED
        moments and the STORED index tower. Training rows, the native-goal eval and the
        harness-facing reward interface all funnel through here so they cannot drift
        apart."""
        m_c = self._z_centered_moment(zenc_params, u_sc, zstats["mu"], r)
        return self._z_centered_index(m_c, zstats["cov"])

    def z_from_reward_samples(self, net, u_sc, r):
        """THE GENERAL REWARD INTERFACE: reward samples {(u_j, r_j)}_{j=1..M} -> index
        [B, d]. No goal anywhere. `u_sc` is an [M, d_c] eps-scaled query pool
        (`_density_queries` output) and `r` is [B, M] (or [M]) reward values on it."""
        return self._z_index(self._zenc(net), self._zstats(net), u_sc, r)

    def _z_mix(self, z_g, rng):
        """FB's standard task-prior mix: keep the encoded index with probability
        1 - z_rand_frac, else a FRESH uniform sqrt(d)-sphere draw. Both branches carry
        ||z|| = sqrt(d); the whole result is stop_gradded. (The bridge's hindsight indicator
        feeds a BC mask that is not ported.)"""
        d, n = int(self.z_dim), z_g.shape[0]
        rng_sphere, rng_mix = jax.random.split(rng)
        z_rand = _sphere(jax.random.normal(rng_sphere, (n, d)), d)
        use_rand = jax.random.bernoulli(rng_mix, self.z_rand_frac, (n, 1))
        return jax.lax.stop_gradient(jnp.where(use_rand, z_rand, z_g))

    def update_stats(self, net, pre_net, batch):
        """Write the three persistent index objects. Called once per update by
        Agent.apply_update AFTER the gradient step and the polyak targets. `pre_net` is the
        PRE-STEP online net, the source for all three.

        No-op with the index off -- a Python-level return, so nothing extra is traced and
        the goal-conditioned arm's update is byte-identical.

        Order matters and matches the bridge: index tower first, then the moments EMA'd at
        the OLD zenc (not the one just written), then the pool frozen on the first update."""
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

        # (2) mu_hat / C_hat over the batch query set at the PRE-STEP index tower.
        b = jax.lax.stop_gradient(self._B(zenc_old, u_sc))                # [M, d]
        mu_b = jnp.mean(b, axis=0)
        db = b - mu_b[None]
        cov_b = db.T @ db / b.shape[0]                                    # centered, 1/M
        st = net["density_zstats"].params
        tau = self.z_sigma_tau
        mu_new = st["mu"] * (1.0 - tau) + mu_b * tau
        cov_new = st["cov"] * (1.0 - tau) + cov_b * tau
        cov_new = 0.5 * (cov_new + cov_new.T)                             # SYMMETRIZE

        # (3) the frozen eval pool: written ONCE, on the first update, so the native-goal
        #     eval index is a function of the params, not of whichever batch was last.
        raw = _raw_block_cols(nobs, blocks)
        if raw.shape[0] < self.z_eval_pool_m:
            raise ValueError(
                f"measure_density_z_eval_pool_m={self.z_eval_pool_m} needs a batch of at "
                f"least that many next states to draw the frozen eval pool from, got "
                f"{raw.shape[0]}. Lower the pool or raise batch_size -- do not let the pool "
                f"silently shrink, it is the M of the reward-sample eval API.")
        pool_new = jnp.where(st["filled"] > 0.5, st["pool"], raw[:self.z_eval_pool_m])

        # Cast back to the stored dtypes: a dtype change would retrace `update` every step.
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
        API, as the bridge does under measure_density_z_prompt='sample'.

        The goal becomes M = measure_density_z_eval_pool_m reward samples over the FROZEN
        pool, r = _delta_phi(goal, pool_raw) -- the same query-set-normalized kernel the
        training rows integrate -- and that reward vector goes through the identical centered
        solve. It NEVER shortcuts to an analytic point prompt; that substitution is not
        implemented in this port at all.

        PRECEDENCE, which the caller owns: Agent.sample_actions merges `goals` into cond
        whenever it is given, so a goal ALWAYS wins over an index previously inferred by
        `infer_cond`. impls/main.py's eval loop always passes a goal, so
        --infer_cond_episodes sets a cond_eval the very next sample_actions overwrites; roll
        a reward-inferred index by calling sample_actions with goals=None.

        PIXEL BRANCH: on a `visual-*` OGBench env `info['goal']` is the goal IMAGE, not the
        goal in state form, so the object-coordinate channel must arrive separately as
        cond["goal_qpos"] (built by impls/utils/evaluation.py). This encodes the goal image
        into the actor's conditioning vector and the observation into its observation leg --
        the same two things `sample_cond` publishes at training time."""
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
        # The F tower's conditioning leg AT THE VALUE GOAL -- the same vector
        # cond["_cvec_vgoal"] publishes for the TD bootstrap draw, so F and the policy whose
        # occupancy it bootstraps are conditioned on one and the same thing. Reading
        # cond["value_goal"] under the index hands a goal-shaped vector to a z-shaped net.
        g = cond["_cvec_vgoal"] if self.z_on else cond["value_goal"]
        gamma, tau = self.discount, self.temp
        blocks = self.blocks

        # `nobs_q` is the object-coordinate row the kernel slices, `nobs` what the B tower
        # embeds: the same array on state, image vs qpos channel on pixels.
        nobs_q = self._kernel_rows(batch, "next_observations", nobs)
        u_raw = _raw_block_cols(nobs_q, blocks)                           # [M, d_c] RAW
        # The B tower's input: eps-scaled query rows on state, the query IMAGE on pixels.
        u_sc = nobs if self.pixel else _density_queries(nobs_q, blocks)

        b_emb = self._B(params["density_b"], u_sc)                        # [M, d]
        f_emb = self._F(params["density_f"], obs, acts, g)                # [2,B,d]
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau            # [2, B, M]
        w = jax.nn.softmax(logits, -1)
        logp = jax.nn.log_softmax(logits, -1)

        # Bootstrap action from the actor, drawn outside the grad tape by
        # actor.sample_bootstrap. It MUST be the action for the VALUE goal g the F towers are
        # conditioned on here; `boot["next_action"]` is at the ACTOR goal, a different goal
        # distribution (p_trajgoal=1.0 vs the 0.2/0.5/0.3 value mix), which makes w_bar the
        # occupancy of a policy heading elsewhere -- density_ess_median ~310 against ~89.
        # Indexed, not .get()-defaulted: a silent fallback is how this was missed.
        if self.bootstrap == "sarsa":
            # W1: bootstrap at the BEHAVIOUR policy's action at s', putting the actor outside
            # the critic's fixed point while the softmax-kernel readout stays as it is.
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
            # spec 21: one-hot immediate target at the row's OWN successor (M == B, the query
            # support being this batch's next states); the soft eps-kernel diluted the readout.
            k = jnp.eye(nobs.shape[0])
        else:
            # Counted at the NEXT state, in OBJECT COORDINATES -- `nobs_q`, never the image.
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
        # How far apart the two bootstrap choices are, logged in both modes so the arms
        # compare; only available when the actor's draw is.
        if "next_action_vgoal" in boot and "next_actions" in batch:
            info["density_sarsa_subst"] = jnp.linalg.norm(
                jax.lax.stop_gradient(boot["next_action_vgoal"]) - batch["next_actions"],
                axis=-1).mean()
        return ce, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """The actor's greedy readout: q_h = sum_j w_h(s,a,g,u_j) * kernel(goal, u_j), with
        the density towers at FROZEN (stop-gradient) params — the measure trains only through
        stage_loss, never through the actor's -Q.mean() term. Requires
        `cond["_density_u_raw"]`/`cond["_density_u_sc"]`, which `sample_cond` stashes from
        the same batch's next_observations that `q_fn` cannot reach."""
        del stop_encoder_grad
        u_raw = cond["_density_u_raw"]
        u_sc = cond["_density_u_sc"]
        f_p = jax.lax.stop_gradient(params["density_f"])
        b_p = jax.lax.stop_gradient(params["density_b"])
        b_emb = self._B(b_p, u_sc)
        if self.z_on:
            # Index mode at measure_density_z_actor_reward='latent' (the 'kernel' cell is
            # deliberately not ported): F conditions on z and the actor ascends
            # r_z(u) = B~(u)^T z / sqrt(d) on the query support the measure is trained
            # against. B~ is the TARGET backward tower, matching the goal branch's frozen-
            # params discipline; q_fn gets online params only, so sample_cond publishes it as
            # cond["_density_tb_u"] -- indexed, since falling back to the online tower would
            # substitute towers silently.
            g = cond["z"]
            kg = jax.lax.stop_gradient(
                jnp.einsum('md,bd->bm', cond["_density_tb_u"], g)
                / jnp.sqrt(float(self.z_dim)))
        else:
            # F takes the goal the tower understands (an IMAGE on pixels); kg is the hit
            # kernel, which only means anything in object coordinates.
            g = cond["goal"]
            g_rows = self._kernel_rows(cond, "goal", g)
            kg = jax.lax.stop_gradient(_delta_phi(g_rows, u_raw, self.blocks))  # [B, M]
        f_emb = self._F(f_p, obs, actions, g)
        # W2: the ACTOR readout may use its own temperature, flattening the softmax the actor
        # ascends without touching what the measure is trained to be (stage_loss keeps
        # self.temp). Default None => self.temp => bit-identical.
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / a_tau, -1)
        q1, q2 = (w * kg[None]).sum(-1)                                    # [2, B] -> [B],[B]
        q = jnp.minimum(q1, q2)
        if self.actor_q_center:
            # actor_q_center touches ONLY the normalizer, q_scale = |q - q.mean()|.mean(),
            # so the centering goes into the SECOND return (FlowBC divides its q-term by
            # sg[|Qs|.mean()]). Centering the first makes -Q.mean() the batch mean of a
            # batch-centered quantity -- identically ~0, no greedy signal, a pure-BC actor.
            return q, (q - q.mean())[None]
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        """The vector actors concatenate onto obs: the raw actor goal; with the reward-sample
        index on, the index built at that goal; on PIXELS, the F-side encoder's DETACHED
        embedding of the goal image (a policy net cannot concatenate a (64, 64, 3) tensor onto
        its input, and the actor owns no encoder)."""
        if self.z_on:
            return cond["z"]
        return cond["_cvec"] if self.pixel else cond["goal"]

    def sample_cond(self, net, batch, rng, target=None):
        # queries resampled every update from THIS batch's own next-state object coords.
        nobs = batch["next_observations"]
        nobs_q = self._kernel_rows(batch, "next_observations", nobs)
        u_raw = _raw_block_cols(nobs_q, self.blocks)
        # The B tower's query rows: eps-scaled object coords on state, the query IMAGE on
        # pixels. `_density_u_raw` keeps its meaning on both.
        u_sc = nobs if self.pixel else _density_queries(nobs_q, self.blocks)
        # `_cvec_vgoal` is THIS critic's conditioning vector at the value goal, published so
        # the actors' TD bootstrap draw is correct rather than inferred (impls/actors/ddpgbc).
        if self.z_on:
            # The index replaces the goal on BOTH legs, from reward samples
            # kg = _delta_phi(goal, u_raw) -- the QUERY-SET-NORMALIZED kernel the goal path
            # also integrates (the spec-28 parity rule; without it the arms are not
            # comparable). TWO indices, because the legs sit at DIFFERENT goal distributions:
            # the actor leg at actor_goals, the value leg at value_goals (what F is
            # conditioned on and what the bootstrap action must be drawn at). Built at the
            # STORED tower and moments, never a statistic of this batch.
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
                    # TARGET backward embedding of the query support (q_fn sees only online
                    # params), for its reward vector r_z = B~(u)^T z / sqrt(d).
                    "_density_tb_u": jax.lax.stop_gradient(
                        self._B(target["density_b"], u_sc)),
                    "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        cond = {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                "_cvec_vgoal": batch["value_goals"],
                "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        if self.pixel:
            # Everything the ACTOR needs: `sample_bootstrap`/`sample_actions` get `net` and
            # `cond` but never the critic, so the encoder is unreachable any other way. Each
            # is DETACHED, so the actor's stage moves no encoder parameter.
            fp = net["density_f"].params
            cond.update({
                # The qpos twin of the ACTOR goal (cond["goal"] itself is an image).
                "goal_qpos": self._kernel_rows(batch, "actor_goals", None),
                # The value goal's twin. NOTHING READS IT TODAY, but the goal legs must stay
                # paired: a kernel at the value goal would else reach for pixels.
                "value_goal_qpos": self._kernel_rows(batch, "value_goals", None),
                # the actor's conditioning leg at the actor goal and at the value goal
                "_cvec": self._enc_f(fp, batch["actor_goals"]),
                "_cvec_vgoal": self._enc_f(fp, batch["value_goals"]),
                # the actor's OBSERVATION leg -- policy/flow nets take features, not images.
                "_obs_feat": self._enc_f(fp, batch["observations"]),
                "_obs_feat_next": self._enc_f(fp, nobs),
            })
        return cond

    def eval_cond0(self, ex_obs, config):
        if self.z_on:
            # Placeholder until infer_cond is called, mirroring FBCritic.eval_cond0. A zero z
            # is off the sqrt(d) sphere; --infer_cond_episodes is what replaces it.
            return {"z": jnp.zeros((self.z_dim,), jnp.float64)}
        if self.pixel:
            # A blank FRAME so the cond is well-formed before evaluate() supplies the real
            # goal image; `evaluate` always passes `goals`, so this is never rolled on.
            return {"goal": jnp.zeros(tuple(ex_obs.shape[1:]), jnp.uint8)}
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        """Zero-shot task specification FROM REWARD SAMPLES: `obs` [M, obs_dim] raw
        observations drawn from the base measure and `rewards` [M] (or [B, M]) their reward
        values -> the conditioning the actor rolls on. No goal is involved.

        With the index off this is the pre-existing no-op placeholder, which
        impls/main.py's `infer_cond_noop_reason` detects and reports. `obs` is eps-scaled
        here exactly as the training query support is.

        TWO THINGS THE CALLER OWNS, neither checkable here:
          * WHICH ROWS -- the base measure is the NEXT-state query marginal, declared as
            `INFER_COND_MARGINAL` so impls/main.py slices the batch by the declaration
            rather than guessing.
          * WHICH REWARDS -- `rewards` must be the task reward on those rows. GCDataset's
            reward column is the goal-reaching reward, near-constant on most samples, and a
            near-constant reward makes the encoder return the neutral 1_d index. That is
            designed behaviour, but it is not a zero-shot result, and
            impls/main.py::infer_cond_noop_reason will NOT flag it (cond did change)."""
        if self.pixel:
            # Not a placeholder: on pixels `obs_dim` is the CHANNEL COUNT (3), so the state
            # branch's zero-goal would hand the eval path a 3-vector where an image belongs.
            raise NotImplementedError(
                "density_fb has no reward-sample task interface on pixels: the index "
                "(measure_density_z) is not implemented there, so --infer_cond_episodes has "
                "nothing to infer. Evaluate against the env goal instead.")
        if not self.z_on:
            return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        z = self.z_from_reward_samples(net, _density_queries(obs, self.blocks), rewards)
        # A flat [M] reward vector is ONE task -> a 1-D z, matching eval_cond0's shape.
        return {"z": z[0] if jnp.ndim(rewards) == 1 else z}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["measure_density_dim"])
        blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        obs_dim = int(ex_obs.shape[-1])
        # PIXEL MODE is inferred from the DATA, not configured: configuring it would let a
        # config say "state" about image data, the disagreement the kernel guard forbids.
        pixel = int(jnp.ndim(ex_obs)) > 2
        z_on = bool(config.get("measure_density_z", False))
        if pixel and z_on:
            # Before the z_on assertions, being the more fundamental refusal: the FROZEN EVAL
            # POOL stores RAW object-coordinate columns, which the image-consuming pixel B
            # tower cannot be fed. Storing images is a design change, not a wiring fix.
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
            # The centered encoder REQUIRES a slow index tower: tau_z = 0 selects
            # target_density_b, whose drift is the known non-stationarity risk.
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
            # `qslices` are (a, b) pairs, not hit-spec (a, b, eps) blocks: they say where the
            # channel comes FROM, not how the kernel scales it.
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
            # F's conditioning leg is the index (width d) under measure_density_z and the raw
            # goal otherwise; initializing at the wrong width dies in the first stage_loss.
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
            # The index tower starts as a COPY of density_b, not an independent init, so at
            # step 0 it is the index density_b implies; the EMA then holds it quasi-stationary.
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
                  # DEFAULT TRUE, matching the yaml: a False fallback would silently give any
                  # path building from a partial config the losing variant.
                  hard=bool(config.get("measure_density_hard", True)),
                  pixel=pixel, qpos_dim=qpos_dim, enc_dim=enc_dim,
                  bootstrap=str(config.get("measure_density_bootstrap", "actor")),
                  actor_temp=config.get("measure_density_actor_temp", None),
                  z_on=z_on, z_dim=d, z_rand_frac=z_rand_frac,
                  z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                  z_enc_tau=z_enc_tau, z_sigma_tau=z_sigma_tau, z_eval_pool_m=z_pool_m,
                  GROUPS=groups,
                  # `density_b_zenc` shares density_b's module DEF; only the params differ.
                  density_b_zenc=(density_b if z_on else None),
                  density_zstats=zstats_mod,
                  # "z" tells check_combo the conditioning is a latent, not a goal.
                  PROVIDES_COND=(frozenset({"value_goal", "goal", "z"}) if z_on
                                 else frozenset({"value_goal", "goal"})))
        return obj, params, targets
