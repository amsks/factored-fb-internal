"""Meta Motivo's policy network (metamotivo/nn_models.py: `simple_embedding`, `Actor`).

Two embedding branches, one over (obs, cond) and one over obs alone, each narrowing to
hidden_dim // 2 and concatenated back to hidden_dim, then a ReLU trunk and a tanh mean.
The reference samples from a TruncatedNormal around the mean; here the policy returns a
unit-scale Gaussian and truncation happens at the call sites (ddpgbc.py clips the action
noise and every action is clipped to [-1, 1]).
"""
import distrax
import flax.linen as nn
import jax.numpy as jnp

_DEFAULT_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


class MotivoEmbed(nn.Module):
    """Dense(h) + LayerNorm + tanh, then layers-2 ReLU layers of width h, then
    Dense(h // 2) + ReLU."""
    hidden_dim: int
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x)))
        for _ in range(self.layers - 2):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2, kernel_init=_DEFAULT_INIT)(x))


class MotivoPolicy(nn.Module):
    """Takes the concatenated [obs, cond] row and splits it at `obs_dim`."""
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
