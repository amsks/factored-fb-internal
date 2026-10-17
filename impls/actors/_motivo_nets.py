"""Meta Motivo policy network.

Embeddings of (obs, cond) and of obs, a ReLU trunk and a tanh mean. Returns a unit-scale
Gaussian; callers clip noise and actions.
"""
import distrax
import flax.linen as nn
import jax.numpy as jnp

_DEFAULT_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


class MotivoEmbed(nn.Module):
    """Dense + LayerNorm + tanh, ReLU layers, then a ReLU layer of width h // 2."""
    hidden_dim: int
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x)))
        for _ in range(self.layers - 2):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2, kernel_init=_DEFAULT_INIT)(x))


class MotivoPolicy(nn.Module):
    """Policy over the concatenated [obs, cond] input, split at `obs_dim`."""
    obs_dim: int
    action_dim: int
    hidden_dim: int
    trunk_layers: int = 2
    embed_layers: int = 2

    @nn.compact
    def __call__(self, x, temperature=1.0):
        o = x[..., :self.obs_dim]
        c = x[..., self.obs_dim:]
        e_z = MotivoEmbed(self.hidden_dim, self.embed_layers)(jnp.concatenate([o, c], -1))
        e_s = MotivoEmbed(self.hidden_dim, self.embed_layers)(o)
        h = jnp.concatenate([e_s, e_z], -1)
        for _ in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(h))
        means = jnp.tanh(nn.Dense(self.action_dim, kernel_init=_DEFAULT_INIT)(h))
        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.ones_like(means) * temperature)
