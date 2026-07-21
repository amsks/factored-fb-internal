import jax.numpy as jnp, numpy as np
from impls.critics.fb import FBCritic
from impls.utils import torch_to_flax as t2f

ATOL = 1e-10
H, Z, L, A, P = 32, 8, 8, 5, 2


def _critic():
    from impls.utils.networks import ForwardMap, PhiMap
    return FBCritic(
        forward=ForwardMap(z_dim=Z, hidden_dim=H, num_parallel=P, hidden_layers=2, embedding_layers=2),
        backward=PhiMap(z_dim=Z, hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder=PhiMap(z_dim=L, hidden_dim=H, hidden_layers=4, norm=True),
        z_dim=Z, num_parallel=P, discount=0.99, ortho_coef=1.0, fb_pessimism_penalty=0.0,
        train_goal_ratio=0.5, norm_z=True, f_tau=0.005, b_tau=0.005,
    )


def _params_targets(fx):
    online = {
        "forward": t2f.load_forward_params(fx),
        "backward": t2f.load_backward_params(fx),
        "left_encoder": t2f.load_left_encoder_params(fx),
    }
    targets = {
        "forward": t2f.load_forward_params(fx, prefix="_target_forward_map"),
        "backward": t2f.load_backward_params(fx, prefix="_target_backward_map"),
        "left_encoder": t2f.load_backward_params(fx, prefix="_target_left_encoder"),
    }
    return online, targets


def test_fb_static_loss_matches_fixture(fb_fixture):
    fx = fb_fixture
    critic = _critic()
    online, targets = _params_targets(fx)
    batch = {k: jnp.asarray(fx["in__" + k], jnp.float64)
             for k in ("observations", "actions", "next_observations")}
    cond = {"z": jnp.asarray(fx["in__z"], jnp.float64)}
    boot = {"next_action": jnp.asarray(fx["in__next_action"], jnp.float64)}
    loss, info = critic.stage_loss("fb", online, targets, batch, cond, boot)
    np.testing.assert_allclose(float(info["fb_offdiag"]), float(fx["out__fb_offdiag"]), atol=ATOL)
    np.testing.assert_allclose(float(info["fb_diag"]), float(fx["out__fb_diag"]), atol=ATOL)
    np.testing.assert_allclose(float(info["ortho_loss"]), float(fx["out__ortho_loss"]), atol=ATOL)
    np.testing.assert_allclose(float(info["fb_loss"]), float(fx["out__fb_loss"]), atol=ATOL)
