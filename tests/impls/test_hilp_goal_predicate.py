"""The phi value loss recognises a reached goal by equality of the full observation."""

import jax
import jax.numpy as jnp
import numpy as np

from env import make_config
from impls.critics.hilp import HILPCritic, hilp_phi_loss


def _critic():
    cfg = make_config(
        "hilp_tdjc",
        "flowbc",
        "cube_single",
        [
            "z_dim=2",
            "L_dim=2",
            "phi.hidden_dim=8",
            "phi.hidden_layers=2",
            "sf.hidden_dim=8",
            "sf.hidden_layers=1",
            "sf.embedding_layers=2",
            "left_encoder.hidden_dim=8",
            "left_encoder.hidden_layers=2",
        ],
    ) | {"seed": 0}
    obs = jnp.zeros((1, 37)).at[:, 19:22].set(jnp.array([0.1, 0.2, 0.3]))
    critic, params, target = HILPCritic.build(obs, jnp.zeros((1, 5)), cfg)
    return critic, params, target, obs


def _loss(critic, params, target, obs, goal, nobs):
    batch = {
        "observations": obs,
        "next_observations": nobs,
        "hilp_future_observations": goal,
    }
    return critic.stage_loss("phi", params, target, batch, {}, {})[0]


def test_stage_loss_is_the_phi_value_loss_on_full_states():
    critic, params, target, obs = _critic()
    goal, nobs = obs.at[:, 0].set(1.0), obs.at[:, 0].set(2.0)
    actual = _loss(critic, params, target, obs, goal, nobs)
    expected, _ = hilp_phi_loss(
        params["hilp_phi1"],
        params["hilp_phi2"],
        target["hilp_phi1"],
        target["hilp_phi2"],
        lambda p, x: critic._phi(p, x),
        obs,
        goal,
        nobs,
        critic.phi_discount,
        critic.expectile,
    )
    np.testing.assert_array_equal(actual, expected)


def test_collapsed_features_cannot_declare_a_different_state_reached():
    critic, params, target, obs = _critic()
    params, target = (
        jax.tree.map(jnp.zeros_like, params),
        jax.tree.map(jnp.zeros_like, target),
    )
    goal, nobs = obs.at[:, 0].set(1.0), obs.at[:, 0].set(2.0)
    collapsed = _loss(critic, params, target, obs, goal, nobs)
    # Reward -1 and bootstrap at the distance floor -1e-3, against V = -1e-3.
    np.testing.assert_allclose(
        collapsed, (-1.0 - critic.phi_discount * 0.001 + 0.001) ** 2, rtol=0, atol=1e-12
    )
