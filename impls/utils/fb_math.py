"""Task-vector normalisation shared by the FB-family critics."""
import jax.numpy as jnp


def project_z(z, norm_z):
    """Scale z onto the sphere of radius sqrt(d) (when norm_z)."""
    if not norm_z:
        return z
    d = z.shape[-1]
    return jnp.sqrt(d) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
