import jax.numpy as jnp, numpy as np
from tests.impls.test_fb_critic_static import _critic


def test_fb_cond_vec_returns_z():
    c = _critic()
    z = jnp.ones((4, 8))
    np.testing.assert_array_equal(np.asarray(c.cond_vec({"z": z})), np.asarray(z))
