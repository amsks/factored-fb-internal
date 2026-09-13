"""The three Density-RLDP "levers" (2026-09-12): the ways the retrofit differs from
Density-FB, each config-gated and default-OFF so every existing rldp_density config is
bit-identical (tests/impls/test_rldp_density*.py and test_rldp_seam.py pin that).

  LEVER 1  `rldp_density_train_b`      the density CE also trains the backward tower.
  LEVER 2  `backward_input: hit_spec`   object-coordinate B input (pre-existing; composed).
  LEVER 3  `rldp_density_task_index`    'feature' (shipped) | 'regression' (density_fb's
                                        centered ridge index of the goal-kernel reward on a
                                        slow EMA copy of B, with EMA'd mu_hat / C_hat).
"""
import dataclasses
from types import SimpleNamespace as NS

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, CRITIC_ALIASES, _LR_KEY
from impls.critics.density_fb import DensityFBCritic, _delta_phi, _raw_block_cols
from impls.critics.rldp_density import (RLDPDensityCritic, RldpDensityConfigError,
                                        _REG_STATE_GROUPS)
from impls.utils.config_utils import load_config
from impls.utils.datasets import Dataset, make_dataset
from tests.impls.test_rldp_density import (B, OBS, ACT, Z, _batch, _build, _cfg,
                                           _cond_boot, _ex)

SPEC = "3:6:0.5"                 # fits inside the OBS=12-wide test observations
BLOCKS = ((3, 6, 0.5),)


def _gc_batch(seed=1):
    """A windowed-dataset batch that ALSO carries the GC goal keys (what
    `windowed_goal_sampling: true` makes WindowedDataset emit)."""
    rng = np.random.default_rng(seed + 100)
    b = _batch(seed)
    return {**b,
            "actor_goals": jnp.asarray(rng.standard_normal((B, OBS))),
            "value_goals": jnp.asarray(rng.standard_normal((B, OBS)))}


def _reg(**over):
    over.setdefault("rldp_density_task_index", "regression")
    over.setdefault("measure_hit_spec", SPEC)
    over.setdefault("rldp_density_z_eval_pool_m", B)
    return _build(**over)


def _net(critic, params):
    return {g: NS(params=params[g]) for g in critic.GROUPS}


def _leaf_abs_sum(tree):
    return sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(tree))


# ==========================================================================================
# LEVER 1 -- the density CE trains B
# ==========================================================================================
def test_train_b_default_off_is_bit_identical_and_ce_has_no_backward_gradient():
    base, params, target = _build()
    off, p2, t2 = _build(rldp_density_train_b=False)
    assert base.train_b is False and off.train_b is False
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    batch, (cond, boot) = _batch(), _cond_boot()
    l1, i1 = base.stage_loss("fb", params, target, batch, cond, boot)
    l2, i2 = off.stage_loss("fb", p2, t2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    np.testing.assert_array_equal(np.asarray(i1["density_ce"]), np.asarray(i2["density_ce"]))
    g = jax.grad(lambda bp: off.stage_loss(
        "fb", {**p2, "backward": bp}, t2, batch, cond, boot)[1]["density_ce"])(p2["backward"])
    assert _leaf_abs_sum(g) == 0.0, "with the lever off the CE must not touch B"


def test_train_b_on_sends_the_ce_gradient_into_backward_and_leaves_the_value_unchanged():
    off, params, target = _build()
    on, p2, t2 = _build(rldp_density_train_b=True)
    assert on.train_b is True
    batch, (cond, boot) = _batch(), _cond_boot()
    # Same VALUE: the lever removes a stop_gradient, it does not change the objective.
    _, i_off = off.stage_loss("fb", params, target, batch, cond, boot)
    l_on, i_on = on.stage_loss("fb", p2, t2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(i_off["density_ce"]), np.asarray(i_on["density_ce"]))
    np.testing.assert_array_equal(np.asarray(i_off["fb_loss"]), np.asarray(l_on))
    g = jax.grad(lambda bp: on.stage_loss(
        "fb", {**p2, "backward": bp}, t2, batch, cond, boot)[1]["density_ce"])(p2["backward"])
    assert _leaf_abs_sum(g) > 0.0, "with the lever on the CE must train B"
    # ...and the SP + ortho gradients into B are untouched: the total backward gradient
    # differs from the lever-off one by exactly the CE gradient.
    g_on = jax.grad(lambda bp: on.stage_loss(
        "fb", {**p2, "backward": bp}, t2, batch, cond, boot)[0])(p2["backward"])
    g_off = jax.grad(lambda bp: off.stage_loss(
        "fb", {**params, "backward": bp}, target, batch, cond, boot)[0])(params["backward"])
    # (float32 params -> float32 gradients; 1e-5 is round-off, not a semantic difference)
    jax.tree_util.tree_map(
        lambda a, b, c: np.testing.assert_allclose(a - b, c, rtol=1e-4, atol=1e-6), g_on, g_off, g)


def test_train_b_gradient_reaches_the_backward_optimizer_group():
    """stage_groups must include "backward" (joint phase), so Agent.apply_update steps it
    with the CE gradient; the two-phase schedule freezes B in phase 2 and is refused."""
    on, _p, _t = _build(rldp_density_train_b=True)
    assert "backward" in on.stage_groups("fb")
    with pytest.raises(RldpDensityConfigError, match="train_b"):
        _build(rldp_density_train_b=True, rldp_two_phase=True)


def test_train_b_agent_update_moves_backward_differently():
    """End to end through Agent.update: one step from identical params with the lever on
    and off must leave the backward group at DIFFERENT values (the CE gradient landed),
    while the lever-off step reproduces the shipped critic exactly."""
    small = {k: v for k, v in _cfg().items() if k != "horizon"}
    cfg = load_config("rldp_density", "flowbc", "cube_single")
    cfg.update(small); cfg["horizon"] = 1
    ex_obs, ex_act = _ex()
    a_off = Agent.create("rldp_density", "flowbc", 0, ex_obs, ex_act, cfg)
    a_on = Agent.create("rldp_density_trainb", "flowbc", 0, ex_obs, ex_act,
                        {**cfg, "rldp_density_train_b": True})
    batch = _batch()
    n_off = a_off.update(batch)[0].net["backward"].params
    n_on = a_on.update(batch)[0].net["backward"].params
    moved = any(not np.allclose(np.asarray(a), np.asarray(b), atol=0)
                for a, b in zip(jax.tree_util.tree_leaves(n_off), jax.tree_util.tree_leaves(n_on)))
    assert moved


# ==========================================================================================
# LEVER 3 -- the regression task index
# ==========================================================================================
def test_task_index_feature_is_bit_identical_to_the_shipped_critic():
    base, params, target = _build()
    feat, p2, t2 = _build(rldp_density_task_index="feature", measure_hit_spec=SPEC)
    assert feat.task_index == "feature"
    assert set(feat.GROUPS) == set(base.GROUPS) == {"forward", "backward", "left_encoder", "predictor"}
    assert set(p2) == set(params)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    batch, (cond, boot) = _gc_batch(), _cond_boot()
    l1, _ = base.stage_loss("fb", params, target, batch, cond, boot)
    l2, _ = feat.stage_loss("fb", p2, t2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    c1 = base.sample_cond(_net(base, params), batch, jax.random.PRNGKey(0))
    c2 = feat.sample_cond(_net(feat, p2), batch, jax.random.PRNGKey(0))
    assert set(c1) == set(c2) == {"z", "_pool_B"}
    np.testing.assert_array_equal(np.asarray(c1["z"]), np.asarray(c2["z"]))
    q1 = base.q_fn(params, batch["observations"], c1, batch["actions"])
    q2 = feat.q_fn(p2, batch["observations"], c2, batch["actions"])
    for a, b in zip(q1, q2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    r = jnp.asarray(np.random.default_rng(3).standard_normal((B,)))
    z1 = base.infer_cond(_net(base, params), batch["next_observations"], r)["z"]
    z2 = feat.infer_cond(_net(feat, p2), batch["next_observations"], r)["z"]
    np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))
    net = _net(feat, p2)
    assert feat.update_stats(net, net, batch) is net


def test_regression_build_adds_the_state_groups_and_refuses_bad_configs():
    critic, params, target = _reg()
    assert critic.task_index == "regression" and critic.blocks == BLOCKS
    assert tuple(critic.GROUPS) == ("forward", "backward", "left_encoder", "predictor") + _REG_STATE_GROUPS
    assert set(params) == set(critic.GROUPS)
    # the slow tower starts as a COPY of B; the stats at mu=0, C=I, empty pool
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b),
                           params["backward_zenc"], params["backward"])
    st = params["rldp_zstats"]
    assert np.allclose(st["mu"], 0) and np.allclose(st["cov"], np.eye(Z)) and float(st["filled"]) == 0
    assert st["pool"].shape == (B, OBS)
    # the state groups take no gradient step and no polyak
    assert set(critic.stage_groups("fb")) == {"forward", "backward", "left_encoder", "predictor"}
    assert not (set(_REG_STATE_GROUPS) & set(critic.TARGETS))
    assert set(target) == {"forward", "left_encoder", "backward"}
    for g in _REG_STATE_GROUPS:
        assert g in _LR_KEY
    with pytest.raises(RldpDensityConfigError, match="task_index"):
        _build(rldp_density_task_index="bogus")
    with pytest.raises(RldpDensityConfigError, match="goal_kernel"):
        _reg(goal_kernel=True)
    with pytest.raises(RldpDensityConfigError, match="hit_spec"):
        _build(rldp_density_task_index="regression", measure_hit_spec="19:22:0.4")
    with pytest.raises(RldpDensityConfigError, match="eval_pool_m"):
        _reg(rldp_density_z_eval_pool_m=B + 1)


def test_regression_index_helpers_are_density_fbs_own():
    """Reuse, not a copy: the centered moment / ridge coefficient / sqrt(d) index are the
    very functions density_fb runs, and the knob names they read carry the same values."""
    assert RLDPDensityCritic._z_centered_moment is DensityFBCritic._z_centered_moment
    assert RLDPDensityCritic._z_centered_coef is DensityFBCritic._z_centered_coef
    assert RLDPDensityCritic._z_centered_index is DensityFBCritic._z_centered_index
    critic, _p, _t = _reg()
    assert (critic.z_ridge_lambda_mult, critic.z_cov_eig_floor, critic.z_const_eps,
            critic.z_enc_tau, critic.z_sigma_tau) == (0.01, 1e-6, 1e-6, 0.0005, 0.005)
    # numerically: the same (m_c, C) through a REAL density_fb instance at the same knobs
    from tests.impls.test_density_fb_reward_index import _built as _dfb_built
    dfb, _dp = _dfb_built(measure_density_dim=Z)
    assert (dfb.z_dim, dfb.z_ridge_lambda_mult, dfb.z_cov_eig_floor, dfb.z_const_eps) == \
        (Z, critic.z_ridge_lambda_mult, critic.z_cov_eig_floor, critic.z_const_eps)
    rng = np.random.default_rng(4)
    m_c = jnp.asarray(rng.standard_normal((3, Z)))
    a = rng.standard_normal((Z, Z)); cov = jnp.asarray(a @ a.T / Z)
    np.testing.assert_array_equal(np.asarray(critic._z_centered_index(m_c, cov)),
                                  np.asarray(dfb._z_centered_index(m_c, cov)))
    np.testing.assert_array_equal(np.asarray(critic._z_centered_coef(m_c, cov)),
                                  np.asarray(dfb._z_centered_coef(m_c, cov)))


def test_regression_index_is_invariant_to_a_constant_shift_of_B():
    """B -> B + b (every feature shifted by the same vector) leaves the goal index
    unchanged: the centered moment multiplies (B - mu) by (r - rbar), and sum_j (r_j -
    rbar) = 0 kills the shift. Needs an unnormalized B so a bias on the last Dense IS a
    feature shift."""
    critic, params, _t = _reg(backward=dict(hidden_dim=16, hidden_layers=4, norm=False))
    batch = _gc_batch()
    nobs = batch["next_observations"]
    r = _delta_phi(batch["actor_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)     # [B, B]
    st = {**params["rldp_zstats"],
          "mu": jnp.asarray(np.random.default_rng(5).standard_normal((Z,)))}
    zenc = params["backward_zenc"]
    b = jnp.asarray(np.random.default_rng(6).standard_normal((Z,)) * 3.0)
    last = sorted(k for k in zenc if k.startswith("Dense_"))[-1]
    shifted = {**zenc, last: {**zenc[last], "bias": zenc[last]["bias"] + b}}
    # sanity: the shift is a feature shift
    np.testing.assert_allclose(np.asarray(critic._B(shifted, nobs)),
                               np.asarray(critic._B(zenc, nobs)) + np.asarray(b)[None], atol=1e-10)
    z0 = critic._z_index(zenc, st, nobs, r)
    z1 = critic._z_index(shifted, st, nobs, r)
    assert np.allclose(np.linalg.norm(np.asarray(z0), axis=-1), np.sqrt(Z), atol=1e-10)
    np.testing.assert_allclose(np.asarray(z0), np.asarray(z1), atol=1e-8)


def test_regression_training_index_equals_infer_cond_on_the_same_reward_samples():
    """One solve for both: a training row's goal index (no sphere rows: train_goal_ratio=1)
    equals infer_cond on that row's kernel reward over the same candidate set."""
    critic, params, _t = _reg(train_goal_ratio=1.0)
    net = _net(critic, params)
    batch = _gc_batch()
    nobs = batch["next_observations"]
    cond = critic.sample_cond(net, batch, jax.random.PRNGKey(0))
    assert set(cond) >= {"z", "_cvec_vgoal", "_pool_B", "_pool_zenc_c"}
    r = _delta_phi(batch["actor_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)     # [B, B]
    for i in (0, 3):
        zi = critic.infer_cond(net, nobs, r[i])["z"]
        assert zi.shape == (Z,)
        np.testing.assert_allclose(np.asarray(zi), np.asarray(cond["z"][i]), atol=1e-12)
    # batched rewards -> batched z, identical to the training rows
    zb = critic.infer_cond(net, nobs, r)["z"]
    np.testing.assert_allclose(np.asarray(zb), np.asarray(cond["z"]), atol=1e-12)
    # the VALUE leg is the same map at value_goals
    rv = _delta_phi(batch["value_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)
    np.testing.assert_allclose(np.asarray(critic.infer_cond(net, nobs, rv)["z"]),
                               np.asarray(cond["_cvec_vgoal"]), atol=1e-12)
    assert np.allclose(np.linalg.norm(np.asarray(cond["z"]), axis=-1), np.sqrt(Z), atol=1e-10)


def test_regression_mixes_half_the_rows_with_sphere_draws():
    critic, params, _t = _reg(train_goal_ratio=0.5)
    net = _net(critic, params)
    batch = _gc_batch()
    nobs = batch["next_observations"]
    r = _delta_phi(batch["actor_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)
    z_g = np.asarray(critic.infer_cond(net, nobs, r)["z"])
    kept = []
    for s in range(8):
        cond = critic.sample_cond(net, batch, jax.random.PRNGKey(s))
        z = np.asarray(cond["z"])
        kept += [np.allclose(z[i], z_g[i], atol=1e-10) for i in range(B)]
        assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(Z), atol=1e-10)
    frac = np.mean(kept)
    assert 0.25 < frac < 0.75, f"kept-goal fraction {frac} is not a 0.5 mix"
    # missing goal keys is a named error, not a silent fallback
    with pytest.raises(KeyError, match="windowed_goal_sampling"):
        critic.sample_cond(net, _batch(), jax.random.PRNGKey(0))


def test_regression_stage_loss_conditions_on_the_value_leg_and_bootstraps_there():
    critic, params, target = _reg()
    batch = _gc_batch()
    cond = critic.sample_cond(_net(critic, params), batch, jax.random.PRNGKey(0))
    rng = np.random.default_rng(2)
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
            "next_action_vgoal": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))}
    l0, _ = critic.stage_loss("fb", params, target, batch, cond, boot)
    # the ACTOR-leg z is not what F conditions on here
    l1, _ = critic.stage_loss("fb", params, target, batch,
                              {**cond, "z": -cond["z"]}, boot)
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    l2, _ = critic.stage_loss("fb", params, target, batch,
                              {**cond, "_cvec_vgoal": -cond["_cvec_vgoal"]}, boot)
    assert not np.allclose(np.asarray(l0), np.asarray(l2))
    l3, _ = critic.stage_loss("fb", params, target, batch, cond,
                              {**boot, "next_action": -boot["next_action"]})
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l3))
    with pytest.raises(KeyError, match="next_action_vgoal"):
        critic.stage_loss("fb", params, target, batch, cond, {"next_action": boot["next_action"]})


def test_regression_q_fn_reads_the_centered_slow_tower_reward():
    critic, params, _t = _reg()
    batch = _gc_batch()
    cond = critic.sample_cond(_net(critic, params), batch, jax.random.PRNGKey(0))
    obs, actions = batch["observations"], batch["actions"]
    q, qs2 = critic.q_fn(params, obs, cond, actions)
    le = critic._le(params["left_encoder"], obs)
    Fs = np.asarray(critic._F(params["forward"], le, cond["z"], actions))
    logits = np.einsum("pbz,cz->pbc", Fs, np.asarray(cond["_pool_B"])) / critic.temp
    w = np.exp(logits - logits.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    mu = np.asarray(params["rldp_zstats"]["mu"])
    bz = np.asarray(critic._B(params["backward_zenc"], batch["next_observations"])) - mu[None]
    np.testing.assert_allclose(np.asarray(cond["_pool_zenc_c"]), bz, atol=1e-12)
    kg = np.einsum("cz,bz->bc", bz, np.asarray(cond["z"])) / np.sqrt(Z)
    q_np = (w * kg[None]).sum(-1).mean(0)
    np.testing.assert_allclose(np.asarray(q), q_np, atol=1e-10)
    np.testing.assert_allclose(np.asarray(qs2), (q_np - q_np.mean())[None], atol=1e-10)
    gp = jax.grad(lambda p: critic.q_fn(p, obs, cond, actions)[0].sum())(params)
    assert _leaf_abs_sum(gp) == 0.0, "q_fn leaked gradient into critic params"


def _agent(critic_name, **over):
    cfg = load_config(critic_name, "flowbc", "cube_single")
    cfg.update({k: v for k, v in _cfg().items() if k != "horizon"})
    cfg["horizon"] = 1
    cfg["measure_hit_spec"] = SPEC
    cfg["rldp_density_z_eval_pool_m"] = B
    cfg.update(over)
    ex_obs, ex_act = _ex()
    return Agent.create(critic_name, "flowbc", 0, ex_obs, ex_act, cfg)


def test_regression_agent_update_stats_writes_the_slow_tower_moments_and_pool():
    agent = _agent("rldp_density_reg")
    assert agent.critic.task_index == "regression"
    b1 = _gc_batch(1)
    pre = agent.net
    a1, info = agent.update(b1)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), k
    # state groups: no optimizer step, written by update_stats only
    for g in _REG_STATE_GROUPS:
        assert int(a1.net[g].step) == int(pre[g].step)
    assert int(a1.net["backward"].step) == int(pre["backward"].step) + 1
    tau = agent.critic.z_enc_tau
    exp_zenc = jax.tree_util.tree_map(lambda p, ep: p * tau + ep * (1 - tau),
                                      pre["backward"].params, pre["backward_zenc"].params)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12),
                           a1.net["backward_zenc"].params, exp_zenc)
    st = a1.net["rldp_zstats"].params
    assert float(st["filled"]) == 1.0
    np.testing.assert_array_equal(np.asarray(st["pool"]), np.asarray(b1["next_observations"])[:B])
    b = np.asarray(agent.critic._B(pre["backward_zenc"].params, b1["next_observations"]))
    mu_b = b.mean(0); db = b - mu_b
    s = agent.critic.z_sigma_tau
    np.testing.assert_allclose(np.asarray(st["mu"]), mu_b * s, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st["cov"]), np.eye(Z) * (1 - s) + (db.T @ db / B) * s, atol=1e-12)
    # the pool is frozen on the first update
    a2 = a1.update(_gc_batch(2))[0]
    np.testing.assert_array_equal(np.asarray(a2.net["rldp_zstats"].params["pool"]), np.asarray(st["pool"]))
    # map_eval_cond: the env goal goes through the same solve on the frozen pool
    g = jnp.asarray(np.random.default_rng(9).standard_normal((OBS,)))
    z = a2.critic.map_eval_cond(a2.net, {"z": jnp.zeros((Z,)), "goal": g})["z"]
    kg = _delta_phi(g[None], _raw_block_cols(a2.net["rldp_zstats"].params["pool"], BLOCKS), BLOCKS)
    z_ref = a2.critic.infer_cond(a2.net, a2.net["rldp_zstats"].params["pool"], kg[0])["z"]
    np.testing.assert_allclose(np.asarray(z), np.asarray(z_ref), atol=1e-12)
    # the eval harness rolls ONE observation at a time against the env's goal frame
    act = a2.sample_actions(b1["observations"][0], goals=g)
    assert act.shape == (ACT,) and np.isfinite(np.asarray(act)).all()
    # Route B through the Agent
    r = jnp.asarray(np.random.default_rng(8).standard_normal((B,)))
    a3 = a2.infer_eval_cond(b1["next_observations"], r)
    assert np.allclose(np.linalg.norm(np.asarray(a3.cond_eval["z"])), np.sqrt(Z), atol=1e-10)


# ==========================================================================================
# LEVER 2 + composition -- the object-coordinate B input composes with 1 and 3
# ==========================================================================================
def test_oc_yaml_sets_hit_spec_input_and_full_yaml_turns_every_lever_on():
    oc = load_config("rldp_density_oc", "flowbc", "cube_single")
    assert oc["backward_input"] == "hit_spec"
    full = load_config("rldp_density_full", "flowbc", "cube_single")
    assert full["backward_input"] == "hit_spec"
    assert full["rldp_density_train_b"] is True
    assert full["rldp_density_task_index"] == "regression"
    assert full["windowed_goal_sampling"] is True and full["dataset_class"] == "WindowedDataset"
    assert full["actor_p_trajgoal"] == 1.0 and full["actor_p_curgoal"] == 0.0
    reg = load_config("rldp_density_reg", "flowbc", "cube_single")
    assert reg["rldp_density_task_index"] == "regression" and reg["windowed_goal_sampling"] is True
    assert reg.get("backward_input", "state") == "state" and not reg.get("rldp_density_train_b", False)
    tb = load_config("rldp_density_trainb", "flowbc", "cube_single")
    assert tb["rldp_density_train_b"] is True and tb.get("rldp_density_task_index", "feature") == "feature"
    base = load_config("rldp_density", "flowbc", "cube_single")
    for name, c in (("trainb", tb), ("reg", reg), ("full", full)):
        for k in ("z_dim", "ortho_coef", "horizon", "temp", "batch_size", "lr_b"):
            assert c[k] == base[k], (name, k)
    for alias in ("rldp_density_trainb", "rldp_density_reg", "rldp_density_full"):
        assert CRITICS[CRITIC_ALIASES[alias]] is RLDPDensityCritic


def test_all_three_levers_compose():
    critic, params, target = _build(rldp_density_train_b=True,
                                    rldp_density_task_index="regression",
                                    backward_input="hit_spec", measure_hit_spec=SPEC,
                                    rldp_density_z_eval_pool_m=B)
    assert critic.train_b and critic.task_index == "regression"
    assert critic.backward_input == "hit_spec" and critic.blocks == BLOCKS
    # B's first Dense is 3 wide: it consumes the sliced block, not the 12-wide state
    assert params["backward"]["Dense_0"]["kernel"].shape[0] == 3
    batch = _gc_batch()
    net = _net(critic, params)
    cond = critic.sample_cond(net, batch, jax.random.PRNGKey(0))
    rng = np.random.default_rng(2)
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
            "next_action_vgoal": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))}
    loss, info = critic.stage_loss("fb", params, target, batch, cond, boot)
    assert np.isfinite(float(loss)) and all(np.isfinite(np.asarray(v)).all() for v in info.values())
    g = jax.grad(lambda bp: critic.stage_loss(
        "fb", {**params, "backward": bp}, target, batch, cond, boot)[1]["density_ce"])(params["backward"])
    assert _leaf_abs_sum(g) > 0.0
    q, qs2 = critic.q_fn(params, batch["observations"], cond, batch["actions"])
    assert np.isfinite(np.asarray(q)).all() and np.isfinite(np.asarray(qs2)).all()
    r = jnp.asarray(rng.standard_normal((B,)))
    z = critic.infer_cond(net, batch["next_observations"], r)["z"]
    assert z.shape == (Z,) and np.isfinite(np.asarray(z)).all()
    # end to end through the Agent with the full yaml
    agent = _agent("rldp_density_full")
    agent, info = agent.update(_gc_batch(1))
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())


# ==========================================================================================
# the dataset: WindowedDataset + hindsight goals
# ==========================================================================================
def _toy_dataset(n=60, T=10, obs=OBS, act=ACT, seed=0):
    rng = np.random.default_rng(seed)
    terminals = np.zeros(n, np.float32)
    terminals[np.arange(1, n // T + 1) * T - 1] = 1.0
    return Dataset.create(observations=rng.standard_normal((n, obs)).astype(np.float32),
                          actions=rng.standard_normal((n, act)).astype(np.float32),
                          terminals=terminals)


def test_windowed_dataset_serves_hindsight_goals_only_when_asked():
    base = _toy_dataset()
    cfg = load_config("rldp_density", "flowbc", "cube_single")
    cfg["horizon"] = 3
    np.random.seed(0)
    b0 = make_dataset(base, cfg).sample(16)
    assert "actor_goals" not in b0 and "value_goals" not in b0
    # bit-identical draw with the key explicitly false
    np.random.seed(0)
    b0b = make_dataset(base, {**cfg, "windowed_goal_sampling": False}).sample(16)
    for k in b0:
        np.testing.assert_array_equal(b0[k], b0b[k])

    reg = load_config("rldp_density_reg", "flowbc", "cube_single")
    reg["horizon"] = 3
    ds = make_dataset(base, reg)
    np.random.seed(0)
    idxs = base.get_random_idxs(16)
    np.random.seed(0)
    b1 = ds.sample(16)
    assert b1["future_observations"].shape == (16, 2, OBS)      # the window survives
    assert b1["actor_goals"].shape == b1["value_goals"].shape == (16, OBS)
    obs = np.asarray(base["observations"])
    (term,) = np.nonzero(np.asarray(base["terminals"]) > 0)
    final = term[np.searchsorted(term, idxs)]
    for i, idx in enumerate(idxs):
        rows = np.flatnonzero((obs == b1["actor_goals"][i]).all(-1))
        assert len(rows) == 1
        # actor_p_trajgoal = 1, actor_p_curgoal = 0: a strictly-future state of the SAME
        # trajectory (uniform on [idx+1, final])
        assert idx < rows[0] <= final[i] or (idx == final[i] and rows[0] == final[i])
    np.testing.assert_array_equal(b1["observations"], obs[idxs])


def test_update_stats_definers_now_include_rldp_density():
    defining = {n for n, c in CRITICS.items() if hasattr(c, "update_stats")}
    assert "rldp_density" in defining
