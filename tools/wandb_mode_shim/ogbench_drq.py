"""JAX/Flax DrQ-v2 visual front-end for vendored OGBench (GCIQL/GCIVL).

Imported by the sitecustomize shim inside the OGBench child process (jax env).
Mirrors FB's PyTorch DrQEncoder (nn_models.py:480) and Augmentator
(nn_models.py:530) so OGBench's pixel baselines share FB's encoder, augmentation,
frame-stacking and normalization. Never edits vendored OGBench.
"""
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class DrQEncoder(nn.Module):
    """DrQ-v2 RGB encoder (Flax), numerically matching FB's torch DrQEncoder.

    Normalizes internally (x/255 - 0.5, == FB's RGBNorm) because OGBench encoders
    own normalization (cf. ImpalaEncoder's x/255). __call__ takes (train, cond_var)
    for signature parity with ImpalaEncoder; both are ignored.
    """

    feature_dim: int = 256

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        init = nn.initializers.xavier_uniform()
        x = x.astype(jnp.float32) / 255.0 - 0.5
        x = nn.relu(nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=init, name="conv0")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv1")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv2")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv3")(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.feature_dim, kernel_init=init, name="proj")(x)
        x = nn.LayerNorm(epsilon=1e-5, name="proj_ln")(x)  # 1e-5 == torch default
        return jnp.tanh(x)
