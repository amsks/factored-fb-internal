"""Regression checks for Density-HILP configuration and actor head selection."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.critics.hilp_density import HILPDensityCritic, HilpDensityConfigError


def _config(name="hilp_density_tdjc", overrides=()):
    small = ["z_dim=2", "L_dim=2", "phi.hidden_dim=8", "phi.hidden_layers=2",
             "sf.hidden_dim=8", "sf.hidden_layers=1", "sf.embedding_layers=2",
             "hilp_left_encoder.identity=true"]
    return make_config(name, "flowbc", "cube_single", small + list(overrides)) | {"seed": 0}


def _build(config):
    return HILPDensityCritic.build(jnp.zeros((2, 37)), jnp.zeros((2, 5)), config)[0]


def test_tdjc_temperature_overrides_reach_the_critic_and_actor():
    critic = _build(_config(overrides=["measure_density_temp=0.25",
                                       "measure_density_actor_temp=0.5"]))
    assert critic.temp == .25
    assert critic.actor_temp == .5


def test_legacy_temperature_fields_remain_supported():
    critic = _build(_config("hilp_density_tdj", ["temp=0.25", "actor_temp=0.5"]))
    assert critic.temp == .25
    assert critic.actor_temp == .5


@pytest.mark.parametrize("canonical,legacy", [
    ("measure_density_temp", "temp"),
    ("measure_density_actor_temp", "actor_temp"),
])
def test_conflicting_temperature_names_fail_clearly(canonical, legacy):
    cfg = _config(overrides=[f"{canonical}=0.25", f"{legacy}=0.5"])
    with pytest.raises(HilpDensityConfigError, match="[Cc]onflict"):
        _build(cfg)


def test_equal_temperature_aliases_are_accepted():
    critic = _build(_config(overrides=["measure_density_temp=0.25", "temp=0.25",
                                       "measure_density_actor_temp=0.5", "actor_temp=0.5"]))
    assert critic.temp == .25
    assert critic.actor_temp == .5


class _FixedForwardHeads:
    """Controlled logits; production q_fn still computes probabilities and values."""

    def apply(self, variables, obs, z, actions):
        return jnp.broadcast_to(jnp.array([[[2., 0.]], [[-2., 0.]]]), (2, len(obs), 2))


@pytest.fixture(scope="module")
def built():
    return _build(_config())


@pytest.mark.parametrize("reduction,expected", [("mean", 0.), ("min", -.9640275800758169)])
@pytest.mark.parametrize("centered", [False, True])
def test_actor_head_reduction_is_applied_after_reward_integration(built, reduction, expected, centered):
    critic = dataclasses.replace(built, hilp_sf=_FixedForwardHeads(),
                                 actor_head_reduce=reduction, actor_q_center=centered)
    cond = {"_pool_phi": jnp.array([[-1., 0.], [1., 0.]]),
            "z": jnp.array([[1., 0.], [1., 0.]])}
    q, scale_values = critic.q_fn({"hilp_sf": {}}, jnp.zeros((2, 37)), cond, jnp.zeros((2, 5)))
    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)
    if centered:
        np.testing.assert_array_equal(scale_values, jnp.zeros((1, 2)))
    else:
        np.testing.assert_allclose(scale_values, [[.9640275800758169] * 2, [-.9640275800758169] * 2], atol=1e-12, rtol=0)


class _DifferentRowsForwardHeads:
    def apply(self, variables, obs, z, actions):
        return jnp.array([[[2., 0.], [0., 0.]], [[-2., 0.], [2., 0.]]])


def test_minimum_actor_value_is_centered_for_the_normalizer(built):
    critic = dataclasses.replace(built, hilp_sf=_DifferentRowsForwardHeads(),
                                 actor_head_reduce="min", actor_q_center=True)
    cond = {"_pool_phi": jnp.array([[-1., 0.], [1., 0.]]),
            "z": jnp.array([[1., 0.], [1., 0.]])}
    q, scale_values = critic.q_fn({"hilp_sf": {}}, jnp.zeros((2, 37)), cond, jnp.zeros((2, 5)))
    np.testing.assert_allclose(q, [-.9640275800758169, 0.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(scale_values, [[-.48201379003790846, .48201379003790846]], atol=1e-12, rtol=0)
