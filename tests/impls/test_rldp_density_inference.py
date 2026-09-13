"""Reward inference must recover coefficients of Density-RLDP's B(u)^T z reward.

Controlled feature tables isolate the inference/readout contract. Production
infer_cond, backward-input routing, normalization, and categorical q_fn all run.
"""
import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.rldp import RLDPCritic
from tests.impls.test_rldp_density import _build


class _Coordinates:
    def apply(self, variables, obs):
        return obs


class _ActionLogits:
    def apply(self, variables, obs, z, actions):
        logits = 10.0 * jnp.concatenate([1.0 - actions, actions], axis=-1)
        return jnp.stack([logits, logits])


@pytest.fixture(scope="module")
def inference_case():
    critic, _, _ = _build(z_dim=2, temp=1.0)
    critic = dataclasses.replace(critic, backward=_Coordinates(),
                                 left_encoder=_Coordinates(), forward=_ActionLogits())
    net = {"backward": SimpleNamespace(params={})}
    return critic, net


def _expected_z(coefficient, normalize):
    c = np.asarray(coefficient, dtype=float)
    return np.sqrt(len(c)) * c / (np.linalg.norm(c) + 1e-12) if normalize else c


def _action_gap(critic, B, z):
    cond = {"_pool_B": B, "z": jnp.broadcast_to(z, (2, 2))}
    q, _ = critic.q_fn({"forward": {}, "left_encoder": {}}, jnp.zeros((2, 2)),
                       cond, jnp.array([[0.], [1.]]))
    return float(q[1] - q[0])


@pytest.mark.parametrize("normalize", [False, True])
def test_nonwhite_features_recover_reward_and_correct_action_ranking(inference_case, normalize):
    critic, net = inference_case
    critic = dataclasses.replace(critic, norm_z=normalize)
    # Uniform reference rows, spherical B, full column rank; C=diag(1.5,.5).
    B = jnp.sqrt(2.) * jnp.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    rewards = B @ jnp.array([1., 2.])
    old = RLDPCritic.infer_cond(critic, net, B, rewards)["z"]
    assert _action_gap(critic, B, old) < 0.0
    actual = critic.infer_cond(net, B, rewards)["z"]
    np.testing.assert_allclose(actual, _expected_z([1., 2.], normalize), rtol=1e-12, atol=1e-12)
    assert _action_gap(critic, B, actual) > 0.0
    if not normalize:
        np.testing.assert_allclose(B @ actual, rewards, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("normalize", [False, True])
def test_white_features_match_parent_reward_moment(inference_case, normalize):
    critic, net = inference_case
    critic = dataclasses.replace(critic, norm_z=normalize)
    B = jnp.sqrt(2.) * jnp.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    rewards = B @ jnp.array([1., 2.])
    actual = critic.infer_cond(net, B, rewards)["z"]
    expected = RLDPCritic.infer_cond(critic, net, B, rewards)["z"]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("features,rewards,coefficient", [
    ([[1., 1.], [-1., -1.]], [4., -4.], [2., 2.]),
    ([[1., 1.]], [1.], [.5, .5]),
    ([[1., 0.], [0., 1.]], [0., 0.], [0., 0.]),
    ([[0., 0.], [0., 0.]], [1., -1.], [0., 0.]),
])
def test_singular_and_zero_reward_problems_return_finite_minimum_norm_coefficients(
        inference_case, normalize, features, rewards, coefficient):
    critic, net = inference_case
    critic = dataclasses.replace(critic, norm_z=normalize)
    actual = critic.infer_cond(net, jnp.array(features), jnp.array(rewards))["z"]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, _expected_z(coefficient, normalize), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("normalize", [False, True])
def test_goal_kernel_keeps_parent_moment_encoding_and_goal_embedding(inference_case, normalize):
    critic, net = inference_case
    critic = dataclasses.replace(critic, goal_kernel=True, norm_z=normalize)
    B = jnp.sqrt(2.) * jnp.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    rewards = jnp.array([1., 0., 0., 1.])
    actual = critic.infer_cond(net, B, rewards)["z"]
    expected = RLDPCritic.infer_cond(critic, net, B, rewards)["z"]
    np.testing.assert_array_equal(actual, expected)
    goal = B[-1]
    mapped = critic.map_eval_cond(net, {"goal": goal})["z"]
    np.testing.assert_allclose(mapped, _expected_z(goal, normalize), rtol=1e-12, atol=1e-12)


def test_real_backward_network_recovers_a_representable_reward_coefficient():
    critic, params, _ = _build(z_dim=2, norm_z=False)
    obs = jnp.arange(72., dtype=jnp.float64).reshape(6, 12) / 17.
    net = {"backward": SimpleNamespace(params=params["backward"])}
    B = critic._B(params["backward"], obs)
    assert np.linalg.matrix_rank(np.asarray(B)) == 2
    rewards = B @ jnp.array([1., -2.])
    actual = critic.infer_cond(net, obs, rewards)["z"]
    np.testing.assert_allclose(actual, [1., -2.], rtol=1e-10, atol=1e-10)
