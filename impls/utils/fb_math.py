"""Task-vector normalization."""
import jax.numpy as jnp


def project_z(z, norm_z):
    """Project z onto the sphere of radius sqrt(d) if norm_z."""
    if not norm_z:
        return z
    d = z.shape[-1]
    return jnp.sqrt(d) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
