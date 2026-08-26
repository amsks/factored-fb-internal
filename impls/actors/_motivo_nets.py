"""Meta Motivo's own policy shape, for the parity arms.

Transcribed from `metamotivo/nn_models.py` @ facebookresearch/metamotivo main:
`simple_embedding` (~167) and `Actor` (~214). Selected by `actor.motivo_archi: true`;
without it nothing here is imported and every existing cell keeps the plain MLP policy.

WHY THIS IS NOT A WIDTH KNOB. Their policy is two embedding branches -- one over
(obs, cond), one over obs alone -- each narrowing to `hidden_dim // 2` and concatenated
back to `hidden_dim`, then a trunk. `hidden_dims=[1024, 1024]` therefore does NOT describe
it: that is one MLP of two 1024 layers over the concatenation, roughly half the depth and a
different parameter count.

DELIBERATE DIFFERENCE, stated rather than buried: the reference draws from a
`TruncatedNormal(mu, std)`. This returns distrax's `MultivariateNormalDiag`, the
distribution every actor in this tree returns, so `mode()`/`sample()`/`log_prob()` keep
their meanings for the callers. Truncation is supplied at the call sites instead --
`ddpgbc.py::_smooth` clips the bootstrap noise to `boot_noise_clip` and every action is
clipped to [-1, 1].
"""
from typing import Any

import distrax
import flax.linen as nn
import jax.numpy as jnp

_DEFAULT_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


class MotivoEmbed(nn.Module):
    """`simple_embedding`: Dense(h) + LayerNorm + Tanh, then layers-2 ReLU layers of width
    h, then Dense(h // 2) + ReLU. The output is h // 2 -- that is what lets two branches
    concatenate back to h."""
    hidden_dim: int
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x)))
        for _ in range(self.layers - 2):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2, kernel_init=_DEFAULT_INIT)(x))


class MotivoPolicy(nn.Module):
    """`Actor`. Takes the CONCATENATED [obs, cond] row the seam builds and splits it by
    `obs_dim`, so `DDPGBCActor._dist` needs no change.

    `const_std` follows the tree's convention: True gives a unit-scale diagonal (the q-term
    is then taken at `mode()`), False learns a per-dimension log-std."""
    obs_dim: int
    action_dim: int
    hidden_dim: int
    trunk_layers: int = 2
    embed_layers: int = 2
    const_std: bool = True
    log_std_min: Any = -5
    log_std_max: Any = 2

    @nn.compact
    def __call__(self, x, temperature=1.0):
        o = x[..., :self.obs_dim]
        c = x[..., self.obs_dim:]
        e_z = MotivoEmbed(self.hidden_dim, self.embed_layers)(jnp.concatenate([o, c], -1))
        e_s = MotivoEmbed(self.hidden_dim, self.embed_layers)(o)
        h = jnp.concatenate([e_s, e_z], -1)
        for _ in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_DEFAULT_INIT)(h))
        # `torch.tanh(self.policy(embedding))` in the reference: the MEAN is squashed, and
        # the distribution is not a tanh-transformed one.
        means = jnp.tanh(nn.Dense(self.action_dim, kernel_init=_DEFAULT_INIT)(h))
        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))
            log_stds = jnp.broadcast_to(log_stds, means.shape)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature)
