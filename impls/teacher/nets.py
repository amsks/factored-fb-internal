"""impls/teacher/nets.py — the two pretrained-teacher policy heads, shared by
campaign/pretrain_teacher.py (training) and impls/teacher/loader.py (loading). Kept
byte-identical to what pretrained the current checkpoints so their flax params reload.
"""
import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """l2bc head: obs -> action."""
    hidden: tuple
    out: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out)(x)


class FlowVF(nn.Module):
    """dp head: velocity field v(obs, a, t) -> da/dt for flow-matching BC."""
    hidden: tuple
    adim: int

    @nn.compact
    def __call__(self, obs, a, t):
        x = jnp.concatenate([obs, a, t], -1)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.adim)(x)
