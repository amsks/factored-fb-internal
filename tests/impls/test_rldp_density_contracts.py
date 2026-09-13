"""Density-RLDP must honor the inherited official-recipe options.

These tests exercise real networks. Regressions caught: reading the stale target B
when online B was requested, adding whitening when only the constant sphere norm
penalty was requested, and silently dropping scalar actor pessimism.
"""
import dataclasses

import jax
import numpy as np
import pytest

from tests.impls.test_rldp_density import _batch, _build, _cond_boot


@pytest.fixture(scope="module")
def density_case():
    critic, params, target = _build(temp=1.0, ortho_coef=1.0)
    batch = _batch()
    cond, boot = _cond_boot()
    return critic, params, target, batch, cond, boot


def test_online_backward_target_ignores_stale_backward_snapshot(density_case):
    critic, params, target, batch, cond, boot = density_case
    stale = {**target, "backward": jax.tree.map(lambda x: -x, target["backward"])}

    def ce(c, t):
        return c.stage_loss("fb", params, t, batch, cond, boot)[1]["density_ce"]

    # The fixture distinguishes the snapshots under the default target-B recipe.
    assert abs(float(ce(critic, target) - ce(critic, stale))) > 1e-4
    online = dataclasses.replace(critic, phase="fb", fb_target_online_b=True)
    np.testing.assert_allclose(ce(online, target), ce(online, stale), rtol=0, atol=1e-12)


def test_inert_orthogonality_preserves_spherical_sp_backward_gradient(density_case):
    critic, params, target, batch, cond, boot = density_case
    critic = dataclasses.replace(critic, phase="mr", ortho_inert=True)

    def total(backward):
        p = {**params, "backward": backward}
        return critic.stage_loss("fb", p, target, batch, cond, boot)[0]

    def sp(backward):
        return critic._sp_loss({**params, "backward": backward}, target, batch)[0]

    # B has fixed norm sqrt(d), so the requested diagonal-only penalty is constant.
    # CE also detaches B: the only remaining backward gradient must come from SP.
    actual = jax.grad(total)(params["backward"])
    expected = jax.grad(sp)(params["backward"])
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("center", [False, True])
def test_actor_pessimism_uses_integrated_head_values(density_case, center):
    critic, params, target, batch, cond, boot = density_case
    pool = critic._B(params["backward"], batch["next_observations"])
    cond = {**cond, "_pool_B": pool}
    base = dataclasses.replace(critic, actor_q_center=False, actor_pessimism=0.0)
    mean, heads = base.q_fn(params, batch["observations"], cond, batch["actions"])
    expected = np.asarray(heads).min(axis=0)
    assert np.max(np.abs(np.asarray(mean) - expected)) > 1e-4

    # Two heads with coefficient 1/2 give exactly min(Q1,Q2), after integration.
    pessimistic = dataclasses.replace(critic, actor_q_center=center, actor_pessimism=0.5)
    actual, scale_values = pessimistic.q_fn(params, batch["observations"], cond, batch["actions"])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    if center:
        np.testing.assert_allclose(scale_values, (expected - expected.mean())[None], rtol=1e-10, atol=1e-10)
    else:
        np.testing.assert_allclose(scale_values, heads, rtol=0, atol=0)
