"""Goal-conditioned DensityFBCritic: loss, TD target, eps-scaling, target towers, and bootstrap actions."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import CRITICS, Agent
from impls.critics.density_fb import (
    DensityFBCritic,
    _delta_phi,
    _density_queries,
    _raw_block_cols,
)
from impls.critics.goal_spec import parse_goal_spec

B, OBS, ACT, D = 12, 40, 5, 8


def _cfg(**over):
    cfg = dict(
        lr=3e-4,
        lr_actor=3e-4,
        batch_size=B,
        discount=0.99,
        tau=0.005,
        seed=0,
        value_hidden_dims=[32, 32],
        layer_norm=True,
        goal_spec="19:22:0.4",
        z_dim=D,
        temperature=1.0,
    )
    cfg.update(over)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(rng.standard_normal((B, OBS))),
        jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
    )


def _batch(seed=1, b=B):
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.standard_normal((b, OBS)))
    return dict(
        observations=jnp.asarray(rng.standard_normal((b, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((b, OBS))),
        value_goals=g,
        actor_goals=g,
        rewards=jnp.asarray(rng.integers(0, 2, size=b) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=b).astype(float)),
    )


def _boot(b=B, seed=2):
    """Two distinct bootstrap actions, at the actor goal and at the value goal."""
    rng = np.random.default_rng(seed)
    return {
        "next_action": jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        "next_action_vgoal": jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
    }


# Kernel takes raw columns, B takes eps-scaled queries
def test_query_scaling_matches_delta_phi_internal_convention():
    blocks = parse_goal_spec("19:22:0.4,28:31:0.8")
    rng = np.random.default_rng(3)
    nobs = rng.standard_normal((9, OBS)).astype(np.float32)
    nobs_j = jnp.asarray(nobs)

    def phi_np_xc(obs, blocks):
        parts = [obs[:, a:b] / e for (a, b, e) in blocks]
        return np.concatenate(parts, axis=1)

    u_sc = np.asarray(_density_queries(nobs_j, blocks))
    xc_ref = phi_np_xc(nobs, blocks)
    assert np.allclose(u_sc, xc_ref, atol=1e-6)

    u_raw = np.asarray(_raw_block_cols(nobs_j, blocks))
    raw_ref = np.concatenate([nobs[:, a:b] for (a, b, e) in blocks], axis=1)
    assert np.allclose(u_raw, raw_ref, atol=1e-6)
    assert not np.allclose(u_raw, u_sc, atol=1e-3), (
        "u_raw must be RAW obs units, not eps-scaled (the double-scale trap)"
    )

    k = np.asarray(_delta_phi(nobs_j, u_raw, blocks))
    assert k.shape == (9, 9)
    assert np.allclose(k.sum(-1), 1.0, atol=1e-5)
    # Row i's nearest query is query i.
    assert (np.argmax(k, axis=-1) == np.arange(9)).all()


def test_softmax_is_over_the_query_axis():
    """The softmax rows of w and k sum to 1 over the query axis."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {
        "value_goal": b["value_goals"],
        "goal": b["actor_goals"],
        "_density_u_raw": _raw_block_cols(b["next_observations"], c.blocks),
        "_density_u_sc": _density_queries(b["next_observations"], c.blocks),
    }
    loss, info = c.stage_loss("density", params, targets, b, cond, boot=_boot())
    assert jnp.isfinite(loss)
    k = _delta_phi(
        b["next_observations"],
        _raw_block_cols(b["next_observations"], c.blocks),
        c.blocks,
    )
    np.testing.assert_allclose(np.asarray(k.sum(-1)), 1.0, atol=1e-5)
    assert not np.allclose(np.asarray(k.sum(0)), 1.0, atol=1e-2), (
        "k summed over the WRONG (batch) axis should not also land on 1 -- otherwise this "
        "test cannot discriminate the axis convention on this batch"
    )


def test_target_towers_init_mirror_and_soft_update():
    from impls.utils.config_utils import load_config

    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["z_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)

    for tower in ("density_f", "density_b"):
        onl0 = jax.tree_util.tree_leaves(agent.net[tower].params)
        tgt0 = jax.tree_util.tree_leaves(agent.target[tower])
        assert len(onl0) == len(tgt0) and len(onl0) > 0
        assert all(
            np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(onl0, tgt0)
        )

    b1 = _batch(seed=0, b=16)
    a1, info1 = agent.update(b1)
    tau = cfg["tau"]
    for tower in ("density_f", "density_b"):
        onl0 = jax.tree_util.tree_leaves(agent.net[tower].params)
        onl1 = jax.tree_util.tree_leaves(a1.net[tower].params)
        assert any(
            not np.allclose(np.asarray(o0), np.asarray(o1))
            for o0, o1 in zip(onl0, onl1)
        ), f"{tower} did not move after one update()"

        b2 = _batch(seed=1, b=16)
        a2, info2 = a1.update(b2)
        # The EMA uses the post-step params of the same update.
        onl2 = jax.tree_util.tree_leaves(a2.net[tower].params)
        tgt1 = jax.tree_util.tree_leaves(a1.target[tower])
        tgt2 = jax.tree_util.tree_leaves(a2.target[tower])
        for onl_post, old_pre, new in zip(onl2, tgt1, tgt2):
            expected = tau * np.asarray(onl_post) + (1 - tau) * np.asarray(old_pre)
            np.testing.assert_allclose(np.asarray(new), expected, atol=1e-6)


def test_stage_loss_reads_the_TARGET_towers_and_not_the_online_ones():
    """The TD target reads the target towers, not the online ones."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b, boot = _batch(), _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    base = float(c.stage_loss("density", params, targets, b, cond, boot)[0])
    drifted = jax.tree_util.tree_map(lambda x: x + 0.25, targets)
    moved = float(c.stage_loss("density", params, drifted, b, cond, boot)[0])
    assert not np.isclose(base, moved, rtol=1e-9, atol=1e-9), (
        "the density loss is INVARIANT to its target towers -- it is bootstrapping off the "
        "online towers"
    )

    # The loss also depends on the online towers.
    p_drift = jax.tree_util.tree_map(lambda x: x + 0.25, params)
    assert not np.isclose(
        base,
        float(c.stage_loss("density", p_drift, targets, b, cond, boot)[0]),
        rtol=1e-9,
        atol=1e-9,
    )


def test_stage_loss_grad_touches_only_the_two_towers():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    boot = _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    def loss_fn(p):
        full = {"density_f": p["density_f"], "density_b": p["density_b"]}
        return c.stage_loss("density", full, targets, b, cond, boot)[0]

    grads = jax.grad(loss_fn)(params)
    for group in ("density_f", "density_b"):
        assert any(
            float(jnp.sum(jnp.abs(l))) > 1e-8
            for l in jax.tree_util.tree_leaves(grads[group])
        )


def test_actor_q_term_reads_frozen_towers():
    """q_fn passes no gradient into density_f or density_b."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    nobs = b["next_observations"]
    cond = {
        "value_goal": b["value_goals"],
        "goal": b["actor_goals"],
        "_density_u_raw": _raw_block_cols(nobs, c.blocks),
        "_density_u_sc": _density_queries(nobs, c.blocks),
    }

    def q_of(p):
        q, _ = c.q_fn(p, b["observations"], cond, b["actions"])
        return q.mean()

    grads = jax.grad(q_of)(params)
    for group in ("density_f", "density_b"):
        for leaf in jax.tree_util.tree_leaves(grads[group]):
            np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))


def test_registered():
    assert CRITICS["density_fb"] is DensityFBCritic


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc"])
def test_combo_builds_and_updates(actor):
    from impls.utils.config_utils import load_config

    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", actor, "cube_single")
    cfg["z_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    cfg["batch_size"] = 16
    agent = Agent.create("density_fb", actor, 0, ex_obs, ex_act, cfg)
    b = _batch(b=16)
    agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"


def test_q_fn_returns_q_and_its_batch_centered_copy():
    """q_fn returns Q and its batch-centered copy."""
    ex_obs, ex_act = _ex()
    c, params, _ = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    nobs = b["next_observations"]
    cond = {
        "value_goal": b["value_goals"],
        "goal": b["actor_goals"],
        "_density_u_raw": _raw_block_cols(nobs, c.blocks),
        "_density_u_sc": _density_queries(nobs, c.blocks),
    }
    q, qs = c.q_fn(params, b["observations"], cond, b["actions"])
    assert qs.shape == (1, B)
    np.testing.assert_allclose(np.asarray(qs[0]), np.asarray(q - q.mean()), atol=1e-12)
    assert not np.allclose(float(q.mean()), 0.0, atol=1e-6)


def test_ce_and_ess_match_a_direct_reconstruction():
    """Cross-entropy and ESS match a direct reconstruction with a one-hot immediate term."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch(seed=11, b=6)
    boot = _boot(b=6, seed=12)
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    loss, info = c.stage_loss("density", params, targets, b, cond, boot)

    blocks = c.blocks
    nobs, obs, acts, g = (
        b["next_observations"],
        b["observations"],
        b["actions"],
        b["value_goals"],
    )
    u_sc = _density_queries(nobs, blocks)
    b_emb = c.density_b.apply({"params": params["density_b"]}, u_sc)
    f_emb = c.density_f.apply(
        {"params": params["density_f"]}, jnp.concatenate([obs, acts, g], -1)
    )
    logits = jnp.einsum("hbd,md->hbm", f_emb, b_emb) / c.temperature
    logp = np.asarray(jax.nn.log_softmax(logits, -1))
    a_plus = boot["next_action_vgoal"]  # value-goal bootstrap
    tb = c.density_b.apply({"params": targets["density_b"]}, u_sc)
    tf = c.density_f.apply(
        {"params": targets["density_f"]}, jnp.concatenate([nobs, a_plus, g], -1)
    )
    w_bar = np.asarray(
        jax.nn.softmax(jnp.einsum("hbd,md->hbm", tf, tb) / c.temperature, -1)
    )
    k = np.eye(nobs.shape[0])
    t = (1.0 - c.discount) * k[None] + c.discount * w_bar
    ce_ref = -np.mean(np.sum(t * logp, axis=-1))
    assert np.allclose(ce_ref, float(loss), atol=1e-5)
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)

    w = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    ess_rows = 1.0 / np.sum(w**2, axis=-1)
    assert np.allclose(np.mean(ess_rows), float(info["density_ess"]), atol=1e-5)
    assert np.allclose(
        np.median(ess_rows), float(info["density_ess_median"]), atol=1e-5
    )
    assert np.allclose(np.mean(ess_rows[0]), float(info["density_ess_h0"]), atol=1e-5)
    assert np.allclose(np.mean(ess_rows[1]), float(info["density_ess_h1"]), atol=1e-5)


def test_stage_loss_bootstraps_at_the_value_goal_action():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    boot = _boot()
    l0, _ = c.stage_loss("density", params, targets, b, cond, boot)
    # The actor-goal bootstrap does not affect the loss
    boot_a = {**boot, "next_action": jnp.zeros_like(boot["next_action"])}
    l1, _ = c.stage_loss("density", params, targets, b, cond, boot_a)
    np.testing.assert_allclose(float(l1), float(l0), atol=0)
    # The value-goal bootstrap does
    boot_v = {**boot, "next_action_vgoal": jnp.zeros_like(boot["next_action_vgoal"])}
    l2, _ = c.stage_loss("density", params, targets, b, cond, boot_v)
    assert not np.allclose(float(l2), float(l0), atol=1e-9)


def test_stage_loss_refuses_a_boot_without_the_value_goal_action():
    """stage_loss raises when the value-goal bootstrap action is missing."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    with pytest.raises(KeyError, match="next_action_vgoal"):
        c.stage_loss(
            "density", params, targets, b, cond, {"next_action": _boot()["next_action"]}
        )


def test_flowbc_bootstrap_draws_the_value_goal_action():
    """FlowBCActor draws both bootstrap actions from the same noise."""
    from impls.utils.config_utils import load_config

    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["z_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)
    b = _batch(seed=5)
    b["value_goals"] = jnp.asarray(np.random.default_rng(7).standard_normal((B, OBS)))
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(3)
    boot = agent.actor.sample_bootstrap(agent.net, b, cond, rng)

    noise = jax.random.normal(jax.random.split(rng, 4)[0], (B, ACT))
    p = agent.net["actor"].params
    a_actor = agent.actor._actor(p, b["next_observations"], b["actor_goals"], noise)
    a_value = agent.actor._actor(p, b["next_observations"], b["value_goals"], noise)
    np.testing.assert_allclose(
        np.asarray(boot["next_action"]), np.asarray(a_actor), atol=0
    )
    np.testing.assert_allclose(
        np.asarray(boot["next_action_vgoal"]), np.asarray(a_value), atol=0
    )
    assert not np.allclose(np.asarray(a_actor), np.asarray(a_value), atol=1e-6), (
        "test setup: the two goals must give different actions for this to discriminate"
    )


def test_flowbc_drift_instrumentation_does_not_change_the_update():
    """FlowBC drift logging does not change the update."""
    from impls.actors.flowbc import FlowBCActor

    ex_obs, ex_act = _ex()
    from impls.utils.config_utils import load_config

    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["z_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    b = _batch()

    a_real, _ = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg).update(b)
    orig = FlowBCActor._policy_drift
    try:
        FlowBCActor._policy_drift = staticmethod(lambda *a, **k: {})
        a_stub, _ = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg).update(
            b
        )
    finally:
        # Restore as a staticmethod, or loss becomes an instance method.
        FlowBCActor._policy_drift = staticmethod(orig)

    for g in a_real.net:
        for x, y in zip(
            jax.tree_util.tree_leaves(a_real.net[g].params),
            jax.tree_util.tree_leaves(a_stub.net[g].params),
        ):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_removed_config_keys_are_ignored():
    """Density-FB ignores config keys removed in the cleanup."""
    ex_obs, ex_act = _ex()
    old = dict(
        measure_density_hard=False,
        measure_density_loss="sq",
        measure_density_bootstrap="sarsa",
        measure_density_actor_temp=5.0,
        measure_density_actor_head_reduce="mean",
        actor_q_center=False,
        measure_density_z_actor_reward="kernel",
        norm_obs=True,
    )
    c0, p0, t0 = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    c1, p1, t1 = DensityFBCritic.build(ex_obs, ex_act, _cfg(**old))
    b, boot = _batch(), _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    l0, _ = c0.stage_loss("density", p0, t0, b, cond, boot)
    l1, _ = c1.stage_loss("density", p1, t1, b, cond, boot)
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
