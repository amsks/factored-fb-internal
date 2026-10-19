import jax.numpy as jnp
import numpy as np

from impls.critics.fb import FBCritic
from impls.utils.networks import ForwardMap, PhiMap

H, Z, L, P = 32, 8, 8, 2


def _critic():
    return FBCritic(
        forward=ForwardMap(
            z_dim=Z, hidden_dim=H, num_parallel=P, hidden_layers=2, embedding_layers=2
        ),
        backward=PhiMap(z_dim=Z, hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder=PhiMap(z_dim=L, hidden_dim=H, hidden_layers=4, norm=True),
        z_dim=Z,
        num_parallel=P,
        discount=0.99,
        ortho_coef=1.0,
        train_goal_ratio=0.5,
        norm_z=True,
        f_tau=0.005,
        b_tau=0.005,
    )


def test_fb_cond_vec_returns_z():
    c = _critic()
    z = jnp.ones((4, 8))
    np.testing.assert_array_equal(np.asarray(c.cond_vec({"z": z})), np.asarray(z))
