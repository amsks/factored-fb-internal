"""Networks shared by the critics and actors."""

import math
from typing import Any, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


# Ensembles
def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module via nn.vmap."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


# Initializers
_ORTH1 = nn.initializers.orthogonal()
_ORTH_RELU = nn.initializers.orthogonal(scale=math.sqrt(2.0))  # ensembled towers


def psm_norm(x):
    """Scale x to norm sqrt(dim)."""
    d = x.shape[-1]
    denom = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return jnp.sqrt(d) * x / denom


class PhiMap(nn.Module):
    """phi(goal) -> R^z_dim, optionally normalized to norm sqrt(z_dim)."""

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
    """Dense + LayerNorm + tanh, ReLU layers, then a ReLU layer of width hidden_dim // 2."""
    assert embedding_layers >= 2, "must have at least 2 embedding layers"
    x = jnp.tanh(
        nn.LayerNorm(epsilon=1e-5)(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x))
    )
    for _ in range(embedding_layers - 2):
        x = nn.relu(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x))
    return nn.relu(nn.Dense(hidden_dim // 2, kernel_init=_ORTH1)(x))


class NoiseConditionedActor(nn.Module):
    """One-step policy a = pi(obs, z, noise) in [-1, 1]."""

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z, noise):
        z_embedding = _simple_embedding(
            jnp.concatenate([obs, z, noise], -1), self.hidden_dim, self.embedding_layers
        )
        s_embedding = _simple_embedding(
            jnp.concatenate([obs, noise], -1), self.hidden_dim, self.embedding_layers
        )
        h = jnp.concatenate([s_embedding, z_embedding], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


class FlowVectorField(nn.Module):
    """Flow velocity v(obs, x_t, t)."""

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        x = jnp.concatenate([obs, action, t], -1)
        x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        for _ in range(self.hidden_layers - 1):
            x = nn.gelu(
                nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False
            )
        return nn.Dense(self.action_dim, kernel_init=_ORTH1)(x)


# Forward maps (the backward map and left encoder are PhiMaps)


class _ForwardTower(nn.Module):
    """One forward-map tower over embeddings of [obs, z] and [obs, action]."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, (
            "only embedding_layers=2 supported (reference default)"
        )
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.fs = [
            nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)
        ] + [nn.Dense(self.z_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, obs_feat, z, action):
        ze = nn.relu(
            self.embed_z_3(
                jnp.tanh(
                    self.embed_z_ln(self.embed_z_0(jnp.concatenate([obs_feat, z], -1)))
                )
            )
        )
        sa = nn.relu(
            self.embed_sa_3(
                jnp.tanh(
                    self.embed_sa_ln(
                        self.embed_sa_0(jnp.concatenate([obs_feat, action], -1))
                    )
                )
            )
        )
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
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(obs_feat, z, action)


class _PredictorTower(nn.Module):
    """One predictor tower over embeddings of [x, action] and x alone."""

    out_dim: int
    hidden_dim: int
    hidden_layers: int = 1
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, (
            "only embedding_layers=2 supported (reference default)"
        )
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_s_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_s_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_s_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.fs = [
            nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)
        ] + [nn.Dense(self.out_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, x, action):
        ze = nn.relu(
            self.embed_z_3(
                jnp.tanh(
                    self.embed_z_ln(self.embed_z_0(jnp.concatenate([x, action], -1)))
                )
            )
        )
        se = nn.relu(self.embed_s_3(jnp.tanh(self.embed_s_ln(self.embed_s_0(x)))))
        y = jnp.concatenate([se, ze], -1)
        for layer in self.fs[:-1]:
            y = nn.relu(layer(y))
        return self.fs[-1](y)


class PredictorMap(nn.Module):
    """Ensembled predictor P(x, a) -> [num_parallel, B, out_dim].

    x is [num_parallel, B, in_dim] (head p reads x[p]); action is [B, action_dim], shared.
    """

    out_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 1
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, x, action):
        tower = ensemblize(_PredictorTower, self.num_parallel, in_axes=(0, None))(
            out_dim=self.out_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(x, action)


_GLOROT_UNIFORM = nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")


class MLP(nn.Module):
    """MLP; layer norm after each hidden activation if layer_norm."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=_GLOROT_UNIFORM)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


# Meta Motivo network shapes
class MotivoEmbed(nn.Module):
    """Embedding branch with output width hidden_dim // 2."""

    hidden_dim: int
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(
            nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(x))
        )
        for _ in range(self.layers - 2):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2, kernel_init=_GLOROT_UNIFORM)(x))


class MotivoForward(nn.Module):
    """Forward map on a concatenated [obs, action, cond] row."""

    obs_dim: int
    act_dim: int
    hidden_dim: int
    out_dim: int
    trunk_layers: int = 2
    embed_layers: int = 2

    @nn.compact
    def __call__(self, x):
        o = x[..., : self.obs_dim]
        a = x[..., self.obs_dim : self.obs_dim + self.act_dim]
        c = x[..., self.obs_dim + self.act_dim :]
        # Explicit names keep the parameter names of saved checkpoints
        e_z = MotivoEmbed(self.hidden_dim, self.embed_layers, name="_MotivoEmbed_0")(
            jnp.concatenate([o, c], -1)
        )
        e_sa = MotivoEmbed(self.hidden_dim, self.embed_layers, name="_MotivoEmbed_1")(
            jnp.concatenate([o, a], -1)
        )
        h = jnp.concatenate([e_sa, e_z], -1)
        for _ in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(h))
        return nn.Dense(self.out_dim, kernel_init=_GLOROT_UNIFORM)(h)


class MotivoBackward(nn.Module):
    """Backward map; output normalisation is applied in `DensityFBCritic._B`."""

    hidden_dim: int
    out_dim: int
    layers: int = 1

    @nn.compact
    def __call__(self, x):
        h = jnp.tanh(
            nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(x))
        )
        for _ in range(self.layers - 1):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(h))
        return nn.Dense(self.out_dim, kernel_init=_GLOROT_UNIFORM)(h)


class MotivoPolicy(nn.Module):
    """Policy over the concatenated [obs, cond] input, split at `obs_dim`."""

    obs_dim: int
    action_dim: int
    hidden_dim: int
    trunk_layers: int = 2
    embed_layers: int = 2

    @nn.compact
    def __call__(self, x, temperature=1.0):
        o = x[..., : self.obs_dim]
        c = x[..., self.obs_dim :]
        e_z = MotivoEmbed(self.hidden_dim, self.embed_layers)(
            jnp.concatenate([o, c], -1)
        )
        e_s = MotivoEmbed(self.hidden_dim, self.embed_layers)(o)
        h = jnp.concatenate([e_s, e_z], -1)
        for _ in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_GLOROT_UNIFORM)(h))
        means = jnp.tanh(nn.Dense(self.action_dim, kernel_init=_GLOROT_UNIFORM)(h))
        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.ones_like(means) * temperature
        )
