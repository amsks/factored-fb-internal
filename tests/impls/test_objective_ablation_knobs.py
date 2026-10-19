"""Density-FB edge cases: cross-entropy loss and the z regression on a degenerate covariance."""

import jax.numpy as jnp
import numpy as np

from impls.critics.density_fb import (
    DensityFBCritic,
    _density_queries,
    _raw_block_cols,
    _sphere_or_neutral,
)
from tests.impls.test_density_fb import _batch, _boot, _cfg, _ex


def _run(cfg):
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, cfg)
    b = _batch()
    cond = {
        "value_goal": b["value_goals"],
        "goal": b["actor_goals"],
        "_density_u_raw": _raw_block_cols(b["next_observations"], c.blocks),
        "_density_u_sc": _density_queries(b["next_observations"], c.blocks),
    }
    return c.stage_loss("density", params, targets, b, cond, boot=_boot())


def test_loss_is_the_cross_entropy():
    loss, info = _run(_cfg())
    assert np.isclose(float(loss), float(info["density_ce"]))


def test_degenerate_covariance_gives_neutral_z_not_nan():
    """A zero-trace feature covariance gives the neutral z, not NaN."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(
        ex_obs, ex_act, _cfg(infer_z=True, eval_pool_size=8)
    )
    d = int(c.z_dim)
    m_c = jnp.zeros((2, d))
    cov = jnp.zeros((d, d))
    coef = c._z_centered_coef(m_c, cov)
    np.testing.assert_array_equal(np.asarray(coef), 0.0)
    z = _sphere_or_neutral(coef, float(d), c.const_reward_eps)
    assert bool(jnp.all(jnp.isfinite(z)))
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(z, axis=-1)), np.sqrt(d), rtol=1e-5
    )
