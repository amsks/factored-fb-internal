"""DensityFBCritic: a forward-backward critic whose successor measure is a categorical
distribution over the batch's next states.

Critic. For a state s, action a and task (a goal g, or a latent task vector z) the critic
predicts a softmax distribution over the M = B next states u_j of the current batch,

    w(s, a, g, u_j) = softmax_j( F(s, a, g)^T B(u_j) / temp ),

two F heads, one B tower. It is trained by cross-entropy to the TD target

    t = (1 - gamma) * onehot(own next state) + gamma * w_target(s', a+, g, .),

where a+ is the actor's action at s' for the same task and w_target uses the polyak target
towers. There is no orthonormality loss, mass anchor or other regulariser on F or B.

Actor. The actor maximises the reward averaged under the predicted distribution,
Q(s, a, g) = sum_j w(s, a, g, u_j) r(u_j), with the critic frozen and the smaller of the
two heads taken. The reward depends on the task interface:
  * goal-conditioned (`infer_z` false): r(u) is the hit kernel
    softmax_j(-0.5 ||(u_j - g) / eps||^2) over the observation blocks and radii named by
    `goal_spec`;
  * reward inference (`infer_z` true): the task is a latent z and
    r(u) = (B_slow(u) - mu)^T z / sqrt(d), the linear reward z encodes.
`actor_value: linear_fz` replaces the softmax average by FB's linear
readout F(s, a, z)^T z (with z = B(g) on the goal arm), leaving the critic unchanged.

Inferring z from reward samples {(u_j, r_j)}: ridge regression of the reward on the
centered slow backward features,

    m   = (1/M) sum_j (B_slow(u_j) - mu)(r_j - rbar)
    c   = (C + lam I)^-1 m,      lam = lambda_mult * tr(C) / d
    z   = sqrt(d) * c / ||c||    (the all-ones vector when ||c|| < const_eps)

B_slow is a slow polyak copy of B and (mu, C) are running estimates of its mean and
covariance over next states. z is unchanged by adding a constant to, or positively
rescaling, the reward. During training each row's z is inferred from the hit-kernel
reward of its hindsight goal, or with probability `z_random_frac` replaced
by a uniform draw on the sqrt(d) sphere. A goal given at evaluation goes through the same
regression, on a pool of next states frozen at the first update.

B_slow, (mu, C) and the frozen pool are parameter groups that serialise with the
checkpoint; `update_stats` is their only writer and no optimiser touches them.
"""
import copy
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.critics._measure_common import parse_hit_spec

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


# ── Meta Motivo's network shapes (metamotivo/nn_models.py), `archi: motivo`.
# F is two embedding branches, over (obs, cond) and (obs, action), each narrowing to
# hidden_dim // 2 and concatenated back to hidden_dim, then a trunk. The hidden-dims lists
# are read as (hidden_dim, n_layers): [1024, 1024] is width 1024 with 2 trunk layers.
class _MotivoEmbed(nn.Module):
    """`simple_embedding`: Dense(h) + LayerNorm + Tanh, layers-2 ReLU layers of width h,
    then Dense(h // 2) + ReLU."""
    hidden_dim: int
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x)))
        for _ in range(self.layers - 2):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2, kernel_init=_DEFAULT_INIT)(x))


class _MotivoF(nn.Module):
    """`ForwardMap`. Takes the concatenated [obs, action, cond] row and splits it by width,
    so it has the same call shape as the MLP F."""
    obs_dim: int
    act_dim: int
    hidden_dim: int
    out_dim: int
    trunk_layers: int = 2
    embed_layers: int = 2

    @nn.compact
    def __call__(self, x):
        o = x[..., :self.obs_dim]
        a = x[..., self.obs_dim:self.obs_dim + self.act_dim]
        c = x[..., self.obs_dim + self.act_dim:]
        e_z = _MotivoEmbed(self.hidden_dim, self.embed_layers)(jnp.concatenate([o, c], -1))
        e_sa = _MotivoEmbed(self.hidden_dim, self.embed_layers)(jnp.concatenate([o, a], -1))
        h = jnp.concatenate([e_sa, e_z], -1)
        for _ in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(h))
        return nn.Dense(self.out_dim, kernel_init=_DEFAULT_INIT)(h)


class _MotivoB(nn.Module):
    """`BackwardMap`: Dense(h) + LayerNorm + Tanh, layers-1 ReLU layers, Dense(out). The
    sqrt(d) output normalisation is applied in `DensityFBCritic._B`."""
    hidden_dim: int
    out_dim: int
    layers: int = 1

    @nn.compact
    def __call__(self, x):
        h = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x)))
        for _ in range(self.layers - 1):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(h))
        return nn.Dense(self.out_dim, kernel_init=_DEFAULT_INIT)(h)


def _sphere(x, d):
    """Project x onto the sphere of radius sqrt(d) along the last axis."""
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _sphere_or_neutral(x, d, eps):
    """Exact sqrt(d) normalisation, falling back to the all-ones vector (norm sqrt(d)) when
    ||x|| < eps, i.e. when the reward was constant on the samples and carries no task."""
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    degenerate = nrm < eps
    safe = jnp.where(degenerate, jnp.ones_like(nrm), nrm)
    return jnp.where(degenerate, jnp.ones_like(x), x * jnp.sqrt(d * 1.0) / safe)


class _ZStatsState(nn.Module):
    """Running feature mean mu [d], covariance C [d, d], the frozen evaluation pool
    [M, d_c] of raw hit-spec columns, and its `filled` flag. Stored as parameters so they
    serialise; written only by `update_stats`. C starts at the identity so the first
    regression is well posed."""
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


# ── hit-spec columns and the hit kernel ─────────────────────────────────────────────────
# `_delta_phi` scales by eps itself, so it takes RAW columns (`_raw_block_cols`); the B tower
# takes the eps-scaled columns (`_density_queries`). Mixing the two scales twice.
def _density_queries(nobs, blocks):
    """[M, d_c] eps-scaled hit-spec columns: the B tower's input rows."""
    return jnp.concatenate([nobs[:, a:b] / e for (a, b, e) in blocks], axis=-1)


def _raw_block_cols(obs_rows, blocks):
    """[N, d_c] raw (unscaled) hit-spec columns: `_delta_phi`'s `protos` argument."""
    return jnp.concatenate([obs_rows[:, a:b] for (a, b, e) in blocks], axis=-1)


def _scale_block_cols(raw_cols, blocks):
    """`_density_queries` applied to already-sliced raw columns (the frozen pool)."""
    parts, off = [], 0
    for (a, b, e) in blocks:
        w = b - a
        parts.append(raw_cols[..., off:off + w] / e)
        off += w
    return jnp.concatenate(parts, axis=-1)


def _delta_phi(obs_rows, protos, blocks):
    """Hit-kernel reward softmax_j(-0.5 sum_blocks ||(x_blk - u_j,blk) / eps_blk||^2),
    normalised over the candidate axis (last). `obs_rows` [N, obs_dim] raw rows, `protos`
    [M, d_c] raw hit-spec columns."""
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
    return jax.nn.softmax(logits, axis=-1)


@dataclasses.dataclass(frozen=True)
class DensityFBCritic:
    density_f: Any
    density_b: Any
    obs_dim: int
    blocks: tuple           # parsed goal_spec: ((start, stop, eps), ...)
    discount: float
    temp: float
    tau: float
    # 'softmax': the actor maximises the reward averaged under the predicted distribution.
    # 'linear_fz': it maximises F(s,a,z)^T z instead (z = B(g) on the goal arm).
    actor_value: str = "softmax"
    archi: str = "mlp"      # 'mlp' | 'motivo' network shapes
    b_norm: bool = False    # project B's output onto the sqrt(d) sphere
    # 'hit_spec': B reads the eps-scaled hit-spec columns. 'state': B reads the full
    # observation (goal arm only); the hit kernel still uses the hit-spec columns.
    backward_input: str = "hit_spec"
    # reward-inference arm (infer_z); the z_* fields are read only when z_on
    z_on: bool = False
    z_dim: int = 0
    z_rand_frac: float = 0.5        # fraction of rows whose z is a sqrt(d)-sphere draw
    z_ridge_lambda_mult: float = 0.01   # lam = mult * tr(C)/d
    z_cov_eig_floor: float = 1e-6   # eigenvalue floor on C, a multiple of tr(C)/d
    z_const_eps: float = 1e-6       # ||c|| below which the reward counts as constant
    z_enc_tau: float = 0.0005       # polyak rate of B_slow
    z_sigma_tau: float = 0.005      # running-average rate of mu and C
    z_eval_pool_m: int = 256        # size of the frozen evaluation pool
    density_b_zenc: Any = None
    density_zstats: Any = None

    # Instance fields: `build` adds the reward-inference groups and the "z" cond key.
    GROUPS: tuple = ("density_f", "density_b")
    PROVIDES_COND: frozenset = frozenset({"value_goal", "goal"})
    STAGES = ("density",)
    # infer_cond's reward samples are drawn on next states, the candidates the critic
    # predicts over.
    INFER_COND_MARGINAL = "next_observations"
    # F is task-indexed, so the TD target's next action must be drawn for the value goal.
    BOOT_COND = "value_goal"

    @property
    def TARGETS(self):
        return {"density_f": self.tau, "density_b": self.tau}

    @property
    def cvec_dim(self):
        """Width of the actor's conditioning vector: z, or the raw goal."""
        return self.z_dim if self.z_on else self.obs_dim

    def stage_groups(self, stage):
        assert stage == "density"
        return ("density_f", "density_b")

    def _F(self, p, obs, acts, g):
        """F(s, a, g) -> [2, B, d]."""
        return self.density_f.apply({"params": p}, jnp.concatenate([obs, acts, g], -1))

    def _B(self, p, u):
        """B(u) -> [M, d], projected onto the sqrt(d) sphere when `b_norm`."""
        out = self.density_b.apply({"params": p}, u)
        if self.b_norm:
            nrm = jnp.linalg.norm(out, axis=-1, keepdims=True)
            out = jnp.sqrt(float(out.shape[-1])) * out / jnp.maximum(nrm, 1e-8)
        return out

    def _b_query(self, rows):
        """The B tower's input for full observation rows."""
        if self.backward_input == "state":
            return rows
        return _density_queries(rows, self.blocks)

    # ── reward inference: reward samples -> z ─────────────────────────────────────────
    def _zstats(self, net):
        return net["density_zstats"].params

    def _zenc(self, net):
        """Parameters of B_slow, the slow polyak copy of B used to encode z."""
        return net["density_b_zenc"].params

    def _z_centered_moment(self, zenc_params, u_sc, mu, r):
        """m = (1/M) sum_j (B_slow(u_j) - mu)(r_j - rbar). `u_sc` [M, d_c], `r` [B, M] or
        [M] -> [B, d]. Raises at M = 1, where m is identically zero."""
        r2 = jnp.atleast_2d(r)
        m_samples = int(r2.shape[-1])
        if m_samples < 2:
            raise ValueError(
                f"reward inference needs M > 1 reward samples, got M={m_samples}: the "
                "centered moment is identically zero at M=1.")
        b_u = jax.lax.stop_gradient(self._B(zenc_params, u_sc))         # [M, d]
        db = b_u - mu[None]
        dr = r2 - jnp.mean(r2, axis=-1, keepdims=True)                  # [B, M]
        return jnp.einsum('bm,md->bd', dr, db) / m_samples

    def _z_centered_coef(self, m_c, cov):
        """c = (C + lam I)^-1 m. C's eigenvalues are floored at z_cov_eig_floor * tr(C)/d
        because the feature covariance has near-null directions whose inverse would amplify
        noise; lam uses the unfloored trace."""
        d = float(self.z_dim)
        cov = 0.5 * (cov + cov.T)
        tr = jnp.trace(cov)
        ev, vec = jnp.linalg.eigh(cov)
        # the absolute floor makes a zero covariance give c = 0 rather than a singular solve
        ev = jnp.maximum(ev, jnp.maximum(self.z_cov_eig_floor * tr / d, 1e-12))
        lam = self.z_ridge_lambda_mult * tr / d
        return ((m_c @ vec) / (ev + lam)[None, :]) @ vec.T

    def _z_index(self, zenc_params, zstats, u_sc, r):
        """Reward samples -> z. Training rows, goal evaluation and `infer_cond` all use it."""
        m_c = self._z_centered_moment(zenc_params, u_sc, zstats["mu"], r)
        c = self._z_centered_coef(m_c, zstats["cov"])
        return jax.lax.stop_gradient(_sphere_or_neutral(c, float(self.z_dim), self.z_const_eps))

    def z_from_reward_samples(self, net, u_sc, r):
        """`u_sc` [M, d_c] B-tower inputs, `r` [B, M] or [M] rewards on them -> z [B, d]."""
        return self._z_index(self._zenc(net), self._zstats(net), u_sc, r)

    def _z_mix(self, z_g, rng):
        """Keep each row's inferred z with probability 1 - z_rand_frac, else a uniform draw
        on the sqrt(d) sphere."""
        d, n = int(self.z_dim), z_g.shape[0]
        rng_sphere, rng_mix = jax.random.split(rng)
        z_rand = _sphere(jax.random.normal(rng_sphere, (n, d)), d)
        use_rand = jax.random.bernoulli(rng_mix, self.z_rand_frac, (n, 1))
        return jax.lax.stop_gradient(jnp.where(use_rand, z_rand, z_g))

    def update_stats(self, net, pre_net, batch):
        """After each gradient step: polyak B_slow toward the pre-step B, update (mu, C) at
        the previous B_slow, and freeze the evaluation pool on the first update."""
        if not self.z_on:
            return net
        nobs = batch["next_observations"]
        u_sc = self._b_query(nobs)

        zenc_old = net["density_b_zenc"].params
        zenc_new = jax.tree_util.tree_map(
            lambda p, ep: p * self.z_enc_tau + ep * (1.0 - self.z_enc_tau),
            pre_net["density_b"].params, zenc_old)

        b = jax.lax.stop_gradient(self._B(zenc_old, u_sc))                # [M, d]
        mu_b = jnp.mean(b, axis=0)
        db = b - mu_b[None]
        cov_b = db.T @ db / b.shape[0]
        st = net["density_zstats"].params
        tau = self.z_sigma_tau
        mu_new = st["mu"] * (1.0 - tau) + mu_b * tau
        cov_new = st["cov"] * (1.0 - tau) + cov_b * tau
        cov_new = 0.5 * (cov_new + cov_new.T)

        raw = _raw_block_cols(nobs, self.blocks)
        if raw.shape[0] < self.z_eval_pool_m:
            raise ValueError(
                f"eval_pool_size={self.z_eval_pool_m} exceeds the batch "
                f"({raw.shape[0]} next states) the evaluation pool is drawn from.")
        pool_new = jnp.where(st["filled"] > 0.5, st["pool"], raw[:self.z_eval_pool_m])

        # keep the stored dtypes, or `update` retraces every step
        cast = lambda new, old: jnp.asarray(new, old.dtype)
        net = dict(net)
        net["density_b_zenc"] = net["density_b_zenc"].replace(params=zenc_new)
        net["density_zstats"] = net["density_zstats"].replace(params={
            "mu": cast(mu_new, st["mu"]), "cov": cast(cov_new, st["cov"]),
            "pool": cast(pool_new, st["pool"]),
            "filled": jnp.ones_like(st["filled"])})
        return net

    def map_eval_cond(self, net, cond, observations=None):
        """Evaluation with a goal. On the reward-inference arm the goal's hit-kernel reward
        on the frozen pool is regressed into z, exactly as for training rows. A given goal
        overrides a z set earlier by `infer_cond`."""
        del observations
        if not self.z_on or "goal" not in cond:
            return cond
        g = cond["goal"]
        unbatched = g.ndim == 1
        g2 = g[None] if unbatched else g
        st = self._zstats(net)
        pool_raw = st["pool"]                                        # [M, d_c] raw
        u_sc = _scale_block_cols(pool_raw, self.blocks)
        kg = jax.lax.stop_gradient(_delta_phi(g2, pool_raw, self.blocks))   # [B, M]
        z = self._z_index(self._zenc(net), st, u_sc, kg)
        return {**cond, "z": z[0] if unbatched else z}

    # ── critic loss ─────────────────────────────────────────────────────────────────────
    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "density"
        obs, nobs, acts = batch["observations"], batch["next_observations"], batch["actions"]
        # the task at the value goal: the raw goal, or its z on the reward-inference arm
        g = cond["_cvec_vgoal"] if self.z_on else cond["value_goal"]
        tau = self.temp
        u_sc = self._b_query(nobs)                                        # candidates u_j

        b_emb = self._B(params["density_b"], u_sc)                        # [M, d]
        f_emb = self._F(params["density_f"], obs, acts, g)                # [2, B, d]
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau            # [2, B, M]
        w = jax.nn.softmax(logits, -1)
        logp = jax.nn.log_softmax(logits, -1)

        # Next action for the TD target: the actor's action at s' for the VALUE goal.
        # boot["next_action"] is drawn for the actor goal, a different task.
        if "next_action_vgoal" not in boot:
            raise KeyError("density_fb needs boot['next_action_vgoal'], the actor's next "
                           "action drawn at cond['_cvec_vgoal'].")
        a_plus = jax.lax.stop_gradient(boot["next_action_vgoal"])
        tb = self._B(target["density_b"], u_sc)
        tf = self._F(target["density_f"], nobs, a_plus, g)
        w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)  # [2, B, M]

        # TD target: (1-gamma) on the row's own next state (candidate j = b) + gamma * w_bar
        k = jnp.eye(nobs.shape[0])
        t = jax.lax.stop_gradient((1.0 - self.discount) * k[None] + self.discount * w_bar)
        ce = -jnp.mean(jnp.sum(t * logp, axis=-1))

        # effective number of candidates the prediction spreads over
        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [2, B]
        info = {
            "density_ce": ce,
            "density_ess": jnp.mean(ess_rows),
            "density_ess_median": jnp.median(ess_rows),
            "density_ess_h0": jnp.mean(ess_rows[0]),
            "density_ess_h1": jnp.mean(ess_rows[1]),
        }
        return ce, info

    # ── actor readout ───────────────────────────────────────────────────────────────────
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Q(s, a) for the actor: the reward averaged under the predicted distribution over
        this batch's next states, at frozen critic parameters, smaller head.

        Returns (q, q - mean(q)). The actor maximises the first and divides by the mean
        absolute value of the second; centering the first would remove the gradient."""
        del stop_encoder_grad
        f_p = jax.lax.stop_gradient(params["density_f"])
        b_p = jax.lax.stop_gradient(params["density_b"])
        g = cond["z"] if self.z_on else cond["goal"]
        f_emb = self._F(f_p, obs, actions, g)
        if self.actor_value == "linear_fz":
            # FB's linear readout F(s,a,z)^T z / sqrt(d), with z = B(g) on the goal arm
            zvec = g if self.z_on else self._B(b_p, self._b_query(g))
            q1, q2 = jnp.einsum('hbd,bd->hb', f_emb, zvec) / jnp.sqrt(float(f_emb.shape[-1]))
        else:
            if self.z_on:
                # r(u) = (B_slow(u) - mu)^T z / sqrt(d), the linear reward z was regressed from
                r = jnp.einsum('md,bd->bm', cond["_density_zenc_u"], g) / jnp.sqrt(float(self.z_dim))
            else:
                r = _delta_phi(g, cond["_density_u_raw"], self.blocks)      # hit-kernel reward
            r = jax.lax.stop_gradient(r)                                    # [B, M]
            b_emb = self._B(b_p, cond["_density_u_sc"])
            w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / self.temp, -1)
            q1, q2 = (w * r[None]).sum(-1)
        q = jnp.minimum(q1, q2)
        return q, (q - q.mean())[None]

    def cond_vec(self, cond):
        """The actor's conditioning vector: z, or the raw actor goal."""
        return cond["z"] if self.z_on else cond["goal"]

    def sample_cond(self, net, batch, rng):
        """Training conditioning. The candidates are this batch's next states. On the
        reward-inference arm every row gets a z for its actor goal and one for its value
        goal, each regressed from that goal's hit-kernel reward and mixed with sphere draws."""
        nobs = batch["next_observations"]
        u_raw = _raw_block_cols(nobs, self.blocks)
        u_sc = self._b_query(nobs)
        if not self.z_on:
            return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                    "_cvec_vgoal": batch["value_goals"],
                    "_density_u_raw": u_raw, "_density_u_sc": u_sc}
        zenc_p, st = self._zenc(net), self._zstats(net)
        r_a, r_v = jax.random.split(rng)

        def _z_at(goals, r_mix):
            r = jax.lax.stop_gradient(_delta_phi(goals, u_raw, self.blocks))  # [B, M]
            return self._z_mix(self._z_index(zenc_p, st, u_sc, r), r_mix)

        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                "z": _z_at(batch["actor_goals"], r_a),
                "_cvec_vgoal": _z_at(batch["value_goals"], r_v),
                # centered B_slow features of the candidates: the actor's reward basis
                "_density_zenc_u": jax.lax.stop_gradient(
                    self._B(zenc_p, u_sc) - st["mu"][None]),
                "_density_u_raw": u_raw, "_density_u_sc": u_sc}

    def eval_cond0(self, ex_obs, config):
        if self.z_on:
            # placeholder until infer_cond or a goal supplies z
            return {"z": jnp.zeros((self.z_dim,), jnp.float64)}
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        """Reward inference: `obs` [M, obs_dim] next states and `rewards` [M] (or [B, M])
        their task rewards -> z. The goal-conditioned arm has no reward interface and
        returns the zero-goal placeholder."""
        if not self.z_on:
            return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
        z = self.z_from_reward_samples(net, self._b_query(obs), rewards)
        return {"z": z[0] if jnp.ndim(rewards) == 1 else z}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        if jnp.ndim(ex_obs) != 2:
            raise ValueError(f"density_fb takes [B, obs_dim] state observations, got shape "
                             f"{tuple(jnp.shape(ex_obs))}")
        hd = tuple(config["value_hidden_dims"])
        _hb = config.get("backward_hidden_dims") or None
        hd_b = tuple(_hb) if _hb else hd                     # null: same as F
        ln = bool(config["layer_norm"])
        d = int(config["z_dim"])
        blocks = parse_hit_spec(config.get("goal_spec", "19:22:0.4"))
        obs_dim = int(ex_obs.shape[-1])
        if max(b for (_a, b, _e) in blocks) > obs_dim:
            raise ValueError(f"goal_spec {blocks} reaches past obs_dim={obs_dim}")
        z_on = bool(config.get("infer_z", False))
        actor_value = str(config.get("actor_value", "softmax"))
        if actor_value not in ("softmax", "linear_fz"):
            raise ValueError(f"actor_value must be 'softmax' or 'linear_fz', "
                             f"got {actor_value!r}")
        backward_input = str(config.get("backward_input", "hit_spec"))
        if backward_input not in ("hit_spec", "state"):
            raise ValueError(f"backward_input must be 'hit_spec' or 'state', "
                             f"got {backward_input!r}")
        if z_on and backward_input != "hit_spec":
            raise ValueError("infer_z requires backward_input='hit_spec'")
        archi = str(config.get("archi", "mlp"))
        if archi not in ("mlp", "motivo"):
            raise ValueError(f"archi must be 'mlp' or 'motivo', got {archi!r}")
        b_norm = bool(config.get("norm_backward", False))
        z_rand_frac = float(config.get("z_random_frac", 0.5))
        z_lam = float(config.get("ridge_lambda", 0.01))
        z_floor = float(config.get("cov_eig_floor", 1e-6))
        z_eps = float(config.get("const_reward_eps", 1e-6))
        z_enc_tau = float(config.get("slow_backward_tau", 0.0005))
        z_sigma_tau = float(config.get("feature_stats_tau", 0.005))
        z_pool_m = int(config.get("eval_pool_size", 256))
        if z_on:
            assert 0.0 <= z_rand_frac <= 1.0, "z_random_frac must be in [0, 1]"
            assert z_lam > 0.0, "ridge_lambda must be > 0"
            assert z_floor > 0.0, "cov_eig_floor must be > 0"
            assert z_eps > 0.0, "const_reward_eps must be > 0"
            assert 0.0 < z_enc_tau <= 0.1, "slow_backward_tau must be in (0, 0.1]"
            assert 0.0 < z_sigma_tau <= 1.0, "feature_stats_tau must be in (0, 1]"
            assert z_pool_m > 1, "eval_pool_size must be > 1"
            _bs = int(config.get("batch_size", z_pool_m))
            assert z_pool_m <= _bs, \
                f"eval_pool_size={z_pool_m} exceeds batch_size={_bs}"

        rf, rb, rz = jr.split(jr.PRNGKey(config["seed"] + 6000), 3)
        if archi == "motivo":
            # hidden dims read as (width, n_layers), so every entry must be equal
            if len(set(hd)) != 1 or len(set(hd_b)) != 1:
                raise ValueError(
                    f"archi='motivo' needs equal hidden-dim entries; got "
                    f"value_hidden_dims={list(hd)} and b={list(hd_b)}")
            density_f = _ensemblize(_MotivoF, 2)(
                obs_dim=obs_dim, act_dim=ex_actions.shape[-1], hidden_dim=hd[0],
                out_dim=d, trunk_layers=len(hd), embed_layers=2)
            density_b = _MotivoB(hidden_dim=hd_b[0], out_dim=d, layers=len(hd_b))
        else:
            density_f = _ensemblize(_MLP, 2)((*hd, d), activate_final=False, layer_norm=ln)
            density_b = _MLP((*hd_b, d), activate_final=False, layer_norm=ln)
        # F's task input is z (width d) or the raw goal
        ex_f = jnp.concatenate(
            [ex_obs, ex_actions,
             jnp.zeros((ex_obs.shape[0], d), ex_obs.dtype) if z_on else ex_obs], -1)
        ex_u = ex_obs if backward_input == "state" else _density_queries(ex_obs, blocks)
        params = {
            "density_f": density_f.init(rf, ex_f)["params"],
            "density_b": density_b.init(rb, ex_u)["params"],
        }
        targets = {"density_f": copy.deepcopy(params["density_f"]),
                   "density_b": copy.deepcopy(params["density_b"])}
        zstats_mod = None
        if z_on:
            # B_slow starts as a copy of B
            params["density_b_zenc"] = copy.deepcopy(params["density_b"])
            d_c = int(sum(b - a for (a, b, _e) in blocks))
            zstats_mod = _ZStatsState(dim=d, pool_m=z_pool_m, d_c=d_c,
                                      dtype=jnp.zeros((), ex_obs.dtype).dtype)
            params["density_zstats"] = zstats_mod.init(rz)["params"]
        obj = cls(density_f=density_f, density_b=density_b, obs_dim=obs_dim, blocks=blocks,
                  discount=float(config["discount"]),
                  temp=float(config["temperature"]), tau=float(config["tau"]),
                  actor_value=actor_value, archi=archi, b_norm=b_norm,
                  backward_input=backward_input,
                  z_on=z_on, z_dim=d, z_rand_frac=z_rand_frac,
                  z_ridge_lambda_mult=z_lam, z_cov_eig_floor=z_floor, z_const_eps=z_eps,
                  z_enc_tau=z_enc_tau, z_sigma_tau=z_sigma_tau, z_eval_pool_m=z_pool_m,
                  # B_slow shares B's module definition; only the parameters differ
                  density_b_zenc=(density_b if z_on else None),
                  density_zstats=zstats_mod,
                  GROUPS=(("density_f", "density_b", "density_b_zenc", "density_zstats")
                          if z_on else ("density_f", "density_b")),
                  PROVIDES_COND=(frozenset({"value_goal", "goal", "z"}) if z_on
                                 else frozenset({"value_goal", "goal"})))
        return obj, params, targets
