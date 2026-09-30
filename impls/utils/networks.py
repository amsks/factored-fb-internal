"""Networks for the FB, HILP and RLDP baselines and the flow actor: the backward/phi tower,
the forward tower F(s,a,z), RLDP's multi-step predictor, and the noise-conditioned policy.

Reference weight_init: nn.Linear -> orthogonal(gain=1); DenseParallel -> orthogonal(gain=relu≈√2).
flax LayerNorm uses epsilon=1e-5 to match torch's default. GELU is exact (erf), not the tanh approx.
"""
import math

import flax.linen as nn
import jax.numpy as jnp


# ── ensemble primitive (from utils/networks.py) ──────────────────────────────
def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module via nn.vmap."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


# ── inits (from utils/psm_networks.py) ───────────────────────────────────────
_ORTH1 = nn.initializers.orthogonal()                          # gain 1 (plain Linear)
_ORTH_RELU = nn.initializers.orthogonal(scale=math.sqrt(2.0))  # ReLU gain (ensembled towers)


def psm_norm(x):
    """Reference `Norm`/`_L2`: sqrt(dim) * x / ||x||_2 (torch F.normalize eps=1e-12)."""
    d = x.shape[-1]
    denom = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return jnp.sqrt(d) * x / denom


class PhiMap(nn.Module):
    """phi(goal) -> R^z_dim. Sequence: Dense, ntanh, [Dense, relu]*(L-1), Dense, [norm]."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = jnp.tanh(x)
        for _ in range(self.hidden_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
            x = nn.relu(x)
        x = nn.Dense(self.z_dim, kernel_init=_ORTH1)(x)
        if self.norm:
            x = psm_norm(x)
        return x


def _simple_embedding(x, hidden_dim, embedding_layers):
    """Reference nn_models.simple_embedding (num_parallel=1): Linear->LayerNorm->Tanh,
    (embedding_layers-2)x[Linear->ReLU], Linear(hidden//2)->ReLU. Returns hidden//2."""
    assert embedding_layers >= 2, "must have at least 2 embedding layers"
    x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x)))
    for _ in range(embedding_layers - 2):
        x = nn.relu(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x))
    return nn.relu(nn.Dense(hidden_dim // 2, kernel_init=_ORTH1)(x))


class NoiseConditionedActor(nn.Module):
    """Port of nn_models.NoiseConditionedActor (the FlowBC one-step actor head).

    a = tanh(policy(concat[ embed_s([obs,noise]), embed_z([obs,z,noise]) ])).
    """

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z, noise):
        z_embedding = _simple_embedding(jnp.concatenate([obs, z, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        s_embedding = _simple_embedding(jnp.concatenate([obs, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([s_embedding, z_embedding], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


class FlowVectorField(nn.Module):
    """Port of nn_models.VectorField (SimpleVectorField): unconditional flow velocity
    v(obs, x_t, t). Linear->GELU, (hidden_layers-1)x[Linear->GELU], Linear(adim)."""

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        x = jnp.concatenate([obs, action, t], -1)
        # torch nn.GELU() is the exact (erf) gelu; flax nn.gelu defaults to the tanh approx.
        x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        for _ in range(self.hidden_layers - 1):
            x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        return nn.Dense(self.action_dim, kernel_init=_ORTH1)(x)


# ── FB forward/backward maps (from utils/fb_networks.py) ──────────────────────
# The backward map and left encoder are PhiMaps.


class _ForwardTower(nn.Module):
    """One (non-ensembled) forward-map tower. embed_z over [obs_feat, z], embed_sa over
    [obs_feat, action]; concat -> `hidden_layers` x [Dense->relu] -> Dense(z_dim)."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, "only embedding_layers=2 supported (reference default)"
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        # `hidden_layers` trunk linears (each relu'd) + a final output linear (no relu).
        self.fs = [nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)] \
            + [nn.Dense(self.z_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, obs_feat, z, action):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(
            self.embed_z_0(jnp.concatenate([obs_feat, z], -1))))))
        sa = nn.relu(self.embed_sa_3(jnp.tanh(self.embed_sa_ln(
            self.embed_sa_0(jnp.concatenate([obs_feat, action], -1))))))
        x = jnp.concatenate([sa, ze], -1)
        for layer in self.fs[:-1]:
            x = nn.relu(layer(x))
        return self.fs[-1](x)


class ForwardMap(nn.Module):
    """Ensembled forward map F(left_enc(obs), z, action) -> [num_parallel, B, z_dim]."""

    z_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs_feat, z, action):
        tower = ensemblize(_ForwardTower, self.num_parallel, in_axes=None)(
            z_dim=self.z_dim, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(obs_feat, z, action)


class _PredictorTower(nn.Module):
    """One (non-ensembled) self-predictive tower. Transcribed from nn_models.VForwardMap
    (the module `agents/rldp/model.py` builds as `_predictor`), which differs from
    `_ForwardTower`/nn_models.ForwardMap in exactly one place: its SECOND embedding sees
    the state alone (`embed_s(obs)`) rather than the state-action pair
    (`embed_sa([obs, action])`). Concat order is `[s_embedding, z_embedding]`, matching
    `VForwardMap.forward`.

    Here `x` plays VForwardMap's `obs` (a z_dim-wide backward embedding) and `action`
    plays its `z`."""

    out_dim: int
    hidden_dim: int
    hidden_layers: int = 1
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, "only embedding_layers=2 supported (reference default)"
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_s_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_s_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_s_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.fs = [nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)] \
            + [nn.Dense(self.out_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, x, action):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(
            self.embed_z_0(jnp.concatenate([x, action], -1))))))
        se = nn.relu(self.embed_s_3(jnp.tanh(self.embed_s_ln(self.embed_s_0(x)))))
        y = jnp.concatenate([se, ze], -1)
        for layer in self.fs[:-1]:
            y = nn.relu(layer(y))
        return self.fs[-1](y)


class PredictorMap(nn.Module):
    """Ensembled self-predictive head P(x, a) -> [num_parallel, B, out_dim].

    Head p reads ITS OWN slice `x[p]`, which is what makes the rollout
    `curr = P(curr, a_t)` well defined: the torch reference gets this from
    `obs.expand(num_parallel, -1, -1)` + DenseParallel's per-head bmm (a no-op expand once
    `curr` is already [P, B, z]). So `x` is [num_parallel, B, in_dim] on EVERY call --
    the caller broadcasts the [B, z_dim] starting point itself -- while `action` is
    [B, action_dim] and is shared across heads."""

    out_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 1
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, x, action):
        tower = ensemblize(_PredictorTower, self.num_parallel, in_axes=(0, None))(
            out_dim=self.out_dim, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(x, action)
