import jax.numpy as jnp
import numpy as np

from impls.utils.fb_math import project_z


def test_project_z_norm():
    z = jnp.array([[3.0, 4.0]])  # ||z||=5, d=2
    out = project_z(z, True)
    np.testing.assert_allclose(
        np.asarray(out), np.sqrt(2) * np.array([[0.6, 0.8]]), atol=1e-12
    )
    # norm_z=False is identity
    np.testing.assert_allclose(np.asarray(project_z(z, False)), np.asarray(z))
