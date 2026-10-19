"""FlowBC's detached value scale must handle a constant critic without NaNs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from impls.actors.flowbc import FlowBCActor


class _ScalarPolicy(FlowBCActor):
    """Linear policy with zero flow, to isolate the actor objective."""

    def _actor(self, parameter, obs, z, noise):
        return jnp.full((len(obs), 1), parameter)

    def _vf(self, parameter, obs, action, time):
        return jnp.zeros_like(action)

    def _rollout(self, parameter, obs, noise):
        return jnp.zeros_like(noise)


class _CenteredValue:
    def __init__(self, offsets, scale=1.0):
        self.offsets, self.scale = jnp.asarray(offsets), scale

    def cond_vec(self, cond):
        return cond["z"]

    def q_fn(self, params, obs, cond, actions, stop_encoder_grad=False):
        q = self.scale * (actions[:, 0] + self.offsets)
        return q, (q - q.mean())[None]


class _ActionDependentSpreadValue:
    def cond_vec(self, cond):
        return cond["z"]

    def q_fn(self, params, obs, cond, actions, stop_encoder_grad=False):
        q = actions[:, 0] * jnp.array([1.0, 2.0])
        return q, (q - q.mean())[None]


def _loss(parameter, offsets, scale=1.0, value_model=None):
    actor = _ScalarPolicy(None, None, 1, 3.0)
    batch = {"observations": jnp.zeros((2, 2)), "actions": jnp.zeros((2, 1))}
    boot = {key: jnp.zeros((2, 1)) for key in ("flow_x0", "flow_t", "actor_noise")}
    if value_model is None:
        value_model = _CenteredValue(offsets, scale)
    return actor.loss(
        {"actor": parameter, "actor_vf": 0.0},
        {},
        batch,
        value_model,
        {"z": jnp.ones((2, 2))},
        boot,
    )[0]


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_constant_centered_values_have_finite_loss_and_policy_gradient(offset):
    value, gradient = jax.value_and_grad(lambda p: _loss(p, [offset, offset]))(
        jnp.array(0.1)
    )
    assert np.isfinite(value)
    assert np.isfinite(gradient)


def test_nonzero_value_scale_keeps_the_existing_objective_and_gradient():
    value, gradient = jax.value_and_grad(lambda p: _loss(p, [10.0, 12.0]))(
        jnp.array(0.1)
    )
    # Q=(10.1,12.1), scale=1, distillation=.01. Scale is detached.
    np.testing.assert_allclose(value, -11.07, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, -0.4, atol=1e-12, rtol=0)


def test_action_dependent_value_scale_is_detached_from_the_policy_gradient():
    value_model = _ActionDependentSpreadValue()
    value, gradient = jax.value_and_grad(
        lambda p: _loss(p, [0.0, 0.0], value_model=value_model)
    )(jnp.array(0.1))
    # Q=(a,2a): scale=.05 depends on a, but the normalizer is detached.
    np.testing.assert_allclose(value, -2.97, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, -29.4, atol=1e-12, rtol=0)


def test_positive_value_rescaling_still_cancels_above_the_floor():
    value, gradient = jax.value_and_grad(lambda p: _loss(p, [10.0, 12.0], scale=0.01))(
        jnp.array(0.1)
    )
    np.testing.assert_allclose(value, -11.07, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, -0.4, atol=1e-12, rtol=0)
