"""Shared FB/PSM measure math — transcribed verbatim from PSMFlows agents/psm.py."""
import jax.numpy as jnp


def project_z(z, norm_z):
    if not norm_z:
        return z
    d = z.shape[-1]
    return jnp.sqrt(d) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def targets_uncertainty(preds, num_parallel):
    mean = preds.mean(axis=0)
    d1 = preds[None]
    d2 = preds[:, None]
    unc = jnp.sum(jnp.abs(d1 - d2), axis=(0, 1)) / (num_parallel ** 2 - num_parallel)
    return mean, unc


def contrastive_loss(M, target_M, discount, off_diag, off_diag_sum):
    diff = M - discount * target_M
    offdiag = 0.5 * jnp.sum((diff * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * M.shape[0]
    return offdiag + diag, diag, offdiag


def ortho_loss(phi, off_diag, off_diag_sum):
    cov = phi @ phi.T
    offdiag = 0.5 * jnp.sum((cov * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(cov))
    return offdiag + diag, diag, offdiag
