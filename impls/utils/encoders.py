"""Visual encoders for the impls/ tree, plus the host-side image augmentation the seam's
`GCDataset` dropped when it was ported from vendored OGBench.

`DrQEncoder` and `random_shifts_batch` are transcribed from the bridge's JAX front-end
(`tools/wandb_mode_shim/ogbench_drq.py` @ origin/gciql-fb-bridge), which itself mirrors FB's
PyTorch `DrQEncoder` (nn_models.py:480) and `Augmentator` (nn_models.py:530). Keeping the two
byte-comparable is the point: the pixel arm of the critic comparison must use the same
encoder, the same augmentation and the same normalization as the FB/TD-JEPA reference, or a
pixel-vs-state gap is confounded with a front-end change.

NORMALIZATION LIVES IN THE ENCODER (`x/255 - 0.5`, FB's RGBNorm), matching OGBench's own
convention (`ImpalaEncoder` divides by 255 internally). Datasets therefore hand the encoder
RAW uint8 images and never pre-scale them -- doing both would halve the input range silently.
"""
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class DrQEncoder(nn.Module):
    """DrQ-v2 RGB encoder (Flax), numerically matching FB's torch DrQEncoder.

    `__call__` takes `(train, cond_var)` for signature parity with OGBench's ImpalaEncoder;
    both are ignored (there is no BatchNorm and no conditioning in this front-end).
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


def _shift_batch_np(arr, shifts, pad):
    """Replicate-pad by `pad`, then integer-crop each image at its own shift.

    `arr` (N, H, W, C); `shifts` (N, 2) ints in [0, 2*pad]. Pure NumPy (CPU): a single
    vectorized gather, bit-identical to FB's `random_shifts` (edge-pad + crop).
    """
    n, h, w, _ = arr.shape
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    rows = shifts[:, 0:1] + np.arange(h)                    # (N, H)
    cols = shifts[:, 1:2] + np.arange(w)                    # (N, W)
    ni = np.arange(n)[:, None, None]
    return padded[ni, rows[:, :, None], cols[:, None, :], :]


def random_shifts_batch(batch, keys, pad=2, rng=None):
    """In-place FB-faithful `random_shifts`: replicate-pad by `pad`, integer-crop.

    Independent per-key shifts (matching FB's separate `_augmentator` calls -- the obs, the
    next obs and each goal are augmented with DIFFERENT shifts, which is what makes the
    augmentation a nuisance the encoder has to be invariant to rather than a shared
    translation the bilinear form could absorb). Only 4-D arrays (N, H, W, C) are touched;
    every other key passes through untouched, so a mixed state/pixel batch is safe.

    `rng` is an `np.random.Generator`; None means THE GLOBAL NUMPY STREAM (`np.random`),
    not a fresh entropy-seeded generator. That matters: `impls/main.py` calls
    `np.random.seed(FLAGS.seed)` precisely so a seed replays the batch stream, and every
    other draw in `GCDataset` goes through the same global RNG. A `default_rng()` here would
    make augmentation the one part of the pipeline a seed does not reproduce.

    STAYS ON THE HOST BY DESIGN. This is numpy, called from the data-loading side, so the
    shifts for step t+1 are computed while the jitted device step for t is still running.
    Moving it on-device (the vendored `batched_random_crop`) forces a host<->device
    round-trip per key per step and serializes against the update. Do not "optimise" it.
    """
    draw = ((lambda n: np.random.randint(0, 2 * pad + 1, size=(n, 2))) if rng is None
            else (lambda n: rng.integers(0, 2 * pad + 1, size=(n, 2))))
    for key in keys:
        arr = batch.get(key)
        if getattr(arr, "ndim", 0) != 4:
            continue
        batch[key] = _shift_batch_np(np.asarray(arr), draw(arr.shape[0]), pad)
    return batch


# Name -> nn.Module factory, the OGBench-idiomatic registry `main.py`/configs reference via
# `config['encoder']`. `drq` is the FB/TD-JEPA front-end at its reference width.
encoder_modules = {
    "drq": DrQEncoder,
    "drq_128": partial(DrQEncoder, feature_dim=128),
    "drq_256": partial(DrQEncoder, feature_dim=256),
    "drq_512": partial(DrQEncoder, feature_dim=512),
}
