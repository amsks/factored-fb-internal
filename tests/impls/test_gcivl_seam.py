"""GCIVLCritic — goal-conditioned implicit V-learning on the factored seam.

Transcribed from vendored third_party/ogbench/impls/agents/gcivl.py. Nothing vendored
is imported (the seam's standing rule); parity is asserted against an independent
reconstruction of vendored's value_loss math, the same numpy/direct-module fallback
tests/impls/test_density_fb.py uses.

The load-bearing test here is test_awr_advantage_is_not_degenerate. Vendored GCIVL's
AWR uses adv = V(s',g) - V(s,g); the seam's AWRActor computes adv = q_fn - v_fn, and
the seam convention for V-only critics (qrl, vip) is q_fn := V(s,g), which makes that
adv identically zero and silently turns AWR into plain BC. GCIVLCritic supplies the
optional adv_fn hook so its AWR cell is the real method.
"""
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tests.impls.gen_boot_contract_fixture import make_batch, B, OBS, ACT


def _build(**over):
    from impls.agents import Agent
    from impls.utils.config_utils import load_config
    cfg = load_config("gcivl", over.pop("actor", "awr"), "cube_single")
    cfg["batch_size"] = B
    cfg.update(over)
    return Agent.create("gcivl", cfg.get("_actor", "awr"), 0,
                        jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg), cfg


def _expectile_loss(adv, diff, expectile):
    weight = np.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * diff ** 2


def test_gcivl_is_registered():
    from impls.agents import CRITICS
    from impls.critics.gcivl import GCIVLCritic
    assert CRITICS["gcivl"] is GCIVLCritic


def test_value_loss_matches_an_independent_reconstruction():
    """Reconstructs vendored gcivl.py::value_loss (lines 29-56) directly from the flax
    module, and compares against GCIVLCritic.stage_loss."""
    agent, cfg = _build()
    critic = agent.critic
    params = {g: agent.net[g].params for g in agent.net}
    batch = make_batch()
    cond = critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))

    apply_v = lambda p, o, g: np.asarray(critic.value.apply({"params": p}, o, g))
    tp, op = agent.target["value"], params["value"]
    vg = np.asarray(cond["value_goal"])
    obs, nobs = np.asarray(batch["observations"]), np.asarray(batch["next_observations"])
    r, m = np.asarray(batch["rewards"]), np.asarray(batch["masks"])
    disc, exp_ = float(cfg["discount"]), float(cfg["expectile"])

    nv1_t, nv2_t = apply_v(tp, nobs, vg)
    q = r + disc * m * np.minimum(nv1_t, nv2_t)
    v1_t, v2_t = apply_v(tp, obs, vg)
    adv = q - (v1_t + v2_t) / 2
    q1 = r + disc * m * nv1_t
    q2 = r + disc * m * nv2_t
    v1, v2 = apply_v(op, obs, vg)
    expected = (_expectile_loss(adv, q1 - v1, exp_).mean()
                + _expectile_loss(adv, q2 - v2, exp_).mean())

    loss, info = critic.stage_loss("ivl", params, agent.target, batch, cond, {})
    assert np.allclose(float(loss), expected, rtol=1e-10, atol=1e-12), (
        f"stage_loss {float(loss)} != reconstruction {expected}")


def test_adv_fn_is_the_next_state_advantage():
    """vendored gcivl.py::actor_loss lines 65-69: adv = mean(V(s',g)) - mean(V(s,g))."""
    agent, _ = _build()
    critic = agent.critic
    params = {g: agent.net[g].params for g in agent.net}
    batch = make_batch()
    cond = critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))

    apply_v = lambda o: np.asarray(critic.value.apply({"params": params["value"]}, o, cond["goal"]))
    v1, v2 = apply_v(np.asarray(batch["observations"]))
    nv1, nv2 = apply_v(np.asarray(batch["next_observations"]))
    expected = (nv1 + nv2) / 2 - (v1 + v2) / 2

    got = np.asarray(critic.adv_fn(params, batch, cond))
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_awr_advantage_is_not_degenerate():
    """The whole reason the adv_fn hook exists. Without it the seam's q_fn - v_fn is
    identically zero for a V-only critic and AWR silently becomes plain BC -- which is
    measurably true today for qrl x awr and vip x awr (both report adv = 0)."""
    from tests.impls.gen_boot_contract_fixture import pair_statics
    _, info = pair_statics("gcivl", "awr")
    adv = np.asarray(info["adv"])
    assert np.isfinite(adv).all()
    assert float(np.abs(adv)) > 0.0, "gcivl x awr collapsed to BC (adv == 0)"


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc", "awr", "fql"])
def test_gcivl_builds_and_updates_with_every_actor(actor):
    from impls.agents import Agent
    from impls.utils.config_utils import load_config
    cfg = load_config("gcivl", actor, "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create("gcivl", actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    updated, info = agent.update(make_batch())
    assert np.isfinite(np.asarray(info["value_loss"])).all()
    assert np.isfinite(np.asarray(info["actor_loss"])).all()
    before = jax.tree_util.tree_leaves(agent.net["value"].params)
    after = jax.tree_util.tree_leaves(updated.net["value"].params)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))


def test_adv_fn_is_optional_so_existing_critics_are_untouched():
    """The hook must be opt-in: a critic that does not define adv_fn keeps the old
    q_fn - v_fn path exactly. gciql is the reference case."""
    from impls.critics.gciql import GCIQLCritic
    from impls.critics.gcivl import GCIVLCritic
    assert not hasattr(GCIQLCritic, "adv_fn")
    assert hasattr(GCIVLCritic, "adv_fn")
