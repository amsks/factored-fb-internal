"""Hilbert goal recognition must use the coordinates visible to the feature map."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.critics.hilp import HILPCritic, hilp_phi_loss
from impls.critics.hilp_density import HILPDensityCritic


@pytest.fixture(scope="module", params=["hilp_oc", "hilp_density_oc"])
def object_critic(request):
    cfg = make_config(request.param, "flowbc", "cube_single", [
        "z_dim=2", "phi.hidden_dim=8", "phi.hidden_layers=2",
        "sf.hidden_dim=8", "sf.feature_dim=4", "sf.hidden_layers=1",
        "sf.embedding_layers=2",
    ]) | {"seed": 0}
    cls = HILPDensityCritic if "density" in request.param else HILPCritic
    obs = jnp.zeros((1, 37)).at[:, 19:22].set(jnp.array([.1, .2, .3]))
    critic, params, target = cls.build(obs, jnp.zeros((1, 5)), cfg)
    return critic, params, target, obs, cfg, cls


def _loss(critic, params, target, obs, goal, nobs):
    batch = {"observations": obs, "next_observations": nobs,
             "hilp_future_observations": goal}
    return critic.stage_loss("phi", params, target, batch, {}, {})[0]


def test_equal_objects_are_a_reached_goal_despite_other_state_changes(object_critic):
    critic, params, target, obs, _, _ = object_critic
    goal, nobs = obs.at[:, 0].set(1.), obs.at[:, 0].set(2.)
    # Real neural towers must give identical outputs for these three inputs.
    np.testing.assert_array_equal(critic._phi(params["hilp_phi1"], obs),
                                  critic._phi(params["hilp_phi1"], goal))
    loss = _loss(critic, params, target, obs, goal, nobs)
    # Reached-goal target is zero. HILP's distance floor is -1e-3, leaving
    # two half-weighted squared residuals of size 1e-6.
    np.testing.assert_allclose(loss, 1e-6, rtol=0, atol=1e-12)


def test_different_objects_stay_unreached_even_if_the_learned_features_collapse(object_critic):
    critic, params, target, obs, _, _ = object_critic
    params, target = jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, target)
    goal = obs.at[:, 19].add(.1)
    loss = _loss(critic, params, target, obs, goal, obs)
    # Recognition uses observed object coordinates, not equality of learned embeddings.
    expected = (-1. - critic.hilp_discount * .001 + .001) ** 2
    np.testing.assert_allclose(loss, expected, rtol=0, atol=1e-12)


def test_full_state_mode_retains_the_original_goal_test(object_critic):
    _, _, _, obs, cfg, cls = object_critic
    critic, params, target = cls.build(obs, jnp.zeros((1, 5)), {**cfg, "backward_input": "state"})
    goal, nobs = obs.at[:, 0].set(1.), obs.at[:, 0].set(2.)
    actual = _loss(critic, params, target, obs, goal, nobs)
    expected, _ = hilp_phi_loss(
        params["hilp_phi1"], params["hilp_phi2"], target["hilp_phi1"], target["hilp_phi2"],
        lambda p, x: critic._phi(p, x), obs, goal, nobs,
        critic.hilp_discount, critic.hilp_expectile)
    np.testing.assert_array_equal(actual, expected)
    # A collapsed full-state feature still cannot declare a different arm state reached.
    params, target = jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, target)
    collapsed = _loss(critic, params, target, obs, goal, nobs)
    np.testing.assert_allclose(collapsed, (-1. - critic.hilp_discount * .001 + .001) ** 2,
                               rtol=0, atol=1e-12)
