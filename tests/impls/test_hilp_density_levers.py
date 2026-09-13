"""Density-HILP's regression task index (2026-09-13): the LEVER 3 port from Density-RLDP
(tests/impls/test_rldp_density_levers.py is the model), config-gated and default-OFF so
every existing hilp_density config is bit-identical (tests/impls/test_hilp_density*.py,
test_hilp_seam.py, test_hilp_oc.py, test_hilp_goal_predicate.py pin that).

  `hilp_density_task_index`   'feature' (shipped: whitened phi(g) mixed with sphere draws,
                              actor reward phi(u)^T z, infer_cond = lstsq on phi)
                              | 'regression' (density_fb's centered ridge index of the
                              goal-kernel reward on a slow EMA copy of phi, with EMA'd
                              mu_hat / C_hat; one solve for training rows, the value leg,
                              infer_cond and map_eval_cond).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent, CRITICS, CRITIC_ALIASES, _LR_KEY
from impls.critics.density_fb import DensityFBCritic, _delta_phi, _raw_block_cols
from impls.critics.hilp import HILPCritic
from impls.critics.hilp_density import (HILPDensityCritic, HilpDensityConfigError,
                                        _REG_STATE_GROUPS)
from impls.utils.datasets import Dataset, make_dataset

# 37 = cube-double's real width, so the cube-single hit spec "19:22:0.4" fits.
OBS_DIM, ACT_DIM, B, Z = 37, 5, 6, 5
SPEC = "19:22:0.4"
BLOCKS = ((19, 22, 0.4),)
SMALL = ["z_dim=5", "L_dim=5", "phi.hidden_dim=8", "phi.hidden_layers=2",
         "sf.hidden_dim=8", "sf.hidden_layers=1", "sf.embedding_layers=2",
         "hilp_left_encoder.identity=true"]
REG_SMALL = SMALL + [f"hilp_density_z_eval_pool_m={B}"]


def _cfg(name="hilp_density_reg", overrides=()):
    small = REG_SMALL if "reg" in name else SMALL
    return make_config(name, "flowbc", "cube_single", small + list(overrides)) | {"seed": 0}


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS_DIM))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT_DIM)))))


def _build(name="hilp_density_reg", overrides=()):
    ex_obs, ex_act = _ex()
    return HILPDensityCritic.build(ex_obs, ex_act, _cfg(name, overrides))


def _batch(seed=1, goals=True):
    rng = np.random.default_rng(seed)
    b = dict(
        observations=jnp.asarray(rng.standard_normal((B, OBS_DIM))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((B, ACT_DIM)))),
        next_observations=jnp.asarray(rng.standard_normal((B, OBS_DIM))),
        hilp_future_observations=jnp.asarray(rng.standard_normal((B, OBS_DIM))),
        terminals=jnp.zeros((B,)),
        rewards=jnp.zeros((B, 1)),
    )
    if goals:
        b["actor_goals"] = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
        b["value_goals"] = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    return b


def _cond_boot(seed=2):
    rng = np.random.default_rng(seed)
    return ({"z": jnp.asarray(rng.standard_normal((B, Z)))},
            {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT_DIM))))})


class _TS:
    """A TrainState stand-in: `.params` plus the `.replace` update_stats writes through."""
    def __init__(self, params):
        self.params = params

    def replace(self, params):
        return _TS(params)


def _net(critic, params):
    return {g: _TS(params[g]) for g in critic.GROUPS}


def _leaf_abs_sum(tree):
    return sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(tree))


def _tree_equal(a, b):
    jax.tree_util.tree_map(lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)), a, b)


# ==========================================================================================
# 'feature' -- the shipped critic, bit for bit
# ==========================================================================================
def test_task_index_feature_is_bit_identical_to_the_shipped_critic():
    base, params, target = _build("hilp_density_tdjc")
    feat, p2, t2 = _build("hilp_density_tdjc", ["hilp_density_task_index=feature"])
    assert base.task_index == feat.task_index == "feature"
    assert set(feat.GROUPS) == set(base.GROUPS) == {"hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats"}
    assert set(p2) == set(params) and set(t2) == set(target)
    _tree_equal(params, p2)
    batch, (cond, boot) = _batch(), _cond_boot()
    for stage in ("phi", "sf"):
        l1, _ = base.stage_loss(stage, params, target, batch, cond, boot)
        l2, _ = feat.stage_loss(stage, p2, t2, batch, cond, boot)
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    c1 = base.sample_cond(_net(base, params), batch, jax.random.PRNGKey(0))
    c2 = feat.sample_cond(_net(feat, p2), batch, jax.random.PRNGKey(0))
    assert set(c1) == set(c2) == {"z", "_pool_phi"}
    np.testing.assert_array_equal(np.asarray(c1["z"]), np.asarray(c2["z"]))
    q1 = base.q_fn(params, batch["observations"], c1, batch["actions"])
    q2 = feat.q_fn(p2, batch["observations"], c2, batch["actions"])
    for a, b in zip(q1, q2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    r = jnp.asarray(np.random.default_rng(3).standard_normal((B,)))
    z1 = base.infer_cond(_net(base, params), batch["next_observations"], r)["z"]
    z2 = feat.infer_cond(_net(feat, p2), batch["next_observations"], r)["z"]
    np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))
    # update_stats: the regression branch is a Python-level no-op, so the result is exactly
    # the PARENT's (HILP's own phi running-mean EMA) and no extra group appears.
    net = _net(feat, p2)
    out = feat.update_stats(net, net, batch)
    ref = HILPCritic.update_stats(feat, net, net, batch)
    assert set(out) == set(ref) == set(net)
    for g in ("hilp_phi1", "hilp_phi2", "hilp_sf"):
        assert out[g] is net[g]
    _tree_equal(out["hilp_stats"].params, ref["hilp_stats"].params)


def test_regression_build_adds_the_state_groups_and_refuses_bad_configs():
    critic, params, target = _build()
    assert critic.task_index == "regression" and critic.regression
    assert critic.kernel_blocks == BLOCKS
    assert tuple(critic.GROUPS) == ("hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats") + _REG_STATE_GROUPS
    assert set(params) == set(critic.GROUPS)
    # the slow tower starts as a COPY of phi1; the stats at mu=0, C=I, empty pool of FULL rows
    _tree_equal(params["hilp_phi_zenc"], params["hilp_phi1"])
    st = params["hilp_zstats"]
    assert np.allclose(st["mu"], 0) and np.allclose(st["cov"], np.eye(Z)) and float(st["filled"]) == 0
    assert st["pool"].shape == (B, OBS_DIM)
    # the state groups take no gradient step and no polyak
    for stage in ("phi", "sf"):
        assert not (set(critic.stage_groups(stage)) & set(_REG_STATE_GROUPS))
    assert not (set(_REG_STATE_GROUPS) & set(critic.TARGETS))
    assert set(target) == set(critic.TARGETS)
    for g in _REG_STATE_GROUPS:
        assert g in _LR_KEY
    # HILP's own knobs are untouched by the lever
    assert (critic.hilp_expectile, critic.q_loss, critic.head_reduce, critic.sf_loss_scale) == \
        (0.5, False, "mean", 0.5)
    with pytest.raises(HilpDensityConfigError, match="task_index"):
        _build("hilp_density_tdjc", ["hilp_density_task_index=bogus"])
    with pytest.raises(HilpDensityConfigError, match="goal_kernel"):
        _build(overrides=["goal_kernel=true"])
    with pytest.raises(HilpDensityConfigError, match="hit_spec"):
        _build(overrides=['measure_hit_spec="40:43:0.4"'])
    with pytest.raises(HilpDensityConfigError, match="eval_pool_m"):
        # the pool is drawn from ONE batch, so it is bounded by batch_size, not by ex_obs
        _build(overrides=[f"batch_size={B}", f"hilp_density_z_eval_pool_m={B + 1}"])
    with pytest.raises(HilpDensityConfigError, match="enc_tau"):
        _build(overrides=["hilp_density_z_enc_tau=0.5"])


def test_regression_index_helpers_are_density_fbs_own():
    """Reuse, not a copy: the centered moment / ridge coefficient / sqrt(d) index are the
    very function objects density_fb runs, and the knobs they read carry the same values."""
    assert HILPDensityCritic._z_centered_moment is DensityFBCritic._z_centered_moment
    assert HILPDensityCritic._z_centered_coef is DensityFBCritic._z_centered_coef
    assert HILPDensityCritic._z_centered_index is DensityFBCritic._z_centered_index
    critic, _p, _t = _build()
    assert (critic.z_ridge_lambda_mult, critic.z_cov_eig_floor, critic.z_const_eps,
            critic.z_enc_tau, critic.z_sigma_tau) == (0.01, 1e-6, 1e-6, 0.0005, 0.005)
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


def test_regression_index_is_invariant_to_a_constant_shift_of_phi():
    """phi -> phi + b leaves the goal index unchanged: the centered moment multiplies
    (phi - mu) by (r - rbar) and sum_j (r_j - rbar) = 0 kills the shift. HILP's PhiMap has
    no trailing normalization, so a bias on its last Dense IS a feature shift."""
    critic, params, _t = _build()
    batch = _batch()
    nobs = batch["next_observations"]
    r = _delta_phi(batch["actor_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)     # [B, B]
    st = {**params["hilp_zstats"],
          "mu": jnp.asarray(np.random.default_rng(5).standard_normal((Z,)))}
    zenc = params["hilp_phi_zenc"]
    b = jnp.asarray(np.random.default_rng(6).standard_normal((Z,)) * 3.0)
    last = sorted(k for k in zenc if k.startswith("Dense_"))[-1]
    shifted = {**zenc, last: {**zenc[last], "bias": zenc[last]["bias"] + b}}
    np.testing.assert_allclose(np.asarray(critic._B(shifted, nobs)),
                               np.asarray(critic._B(zenc, nobs)) + np.asarray(b)[None], atol=1e-10)
    z0 = critic._z_index(zenc, st, nobs, r)
    z1 = critic._z_index(shifted, st, nobs, r)
    assert np.allclose(np.linalg.norm(np.asarray(z0), axis=-1), np.sqrt(Z), atol=1e-10)
    np.testing.assert_allclose(np.asarray(z0), np.asarray(z1), atol=1e-8)


def test_regression_training_index_equals_infer_cond_on_the_same_reward_samples():
    """One solve for both: a training row's goal index (no sphere rows: hilp_mix_ratio=1)
    equals infer_cond on that row's kernel reward over the same candidate set."""
    critic, params, _t = _build(overrides=["hilp_mix_ratio=1.0"])
    net = _net(critic, params)
    batch = _batch()
    nobs = batch["next_observations"]
    cond = critic.sample_cond(net, batch, jax.random.PRNGKey(0))
    assert set(cond) >= {"z", "_cvec_vgoal", "_pool_phi", "_pool_zenc_c"}
    r = _delta_phi(batch["actor_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)     # [B, B]
    for i in (0, 3):
        zi = critic.infer_cond(net, nobs, r[i])["z"]
        assert zi.shape == (Z,)
        np.testing.assert_allclose(np.asarray(zi), np.asarray(cond["z"][i]), atol=1e-12)
    zb = critic.infer_cond(net, nobs, r)["z"]
    np.testing.assert_allclose(np.asarray(zb), np.asarray(cond["z"]), atol=1e-12)
    # the VALUE leg is the same map at value_goals
    rv = _delta_phi(batch["value_goals"], _raw_block_cols(nobs, BLOCKS), BLOCKS)
    np.testing.assert_allclose(np.asarray(critic.infer_cond(net, nobs, rv)["z"]),
                               np.asarray(cond["_cvec_vgoal"]), atol=1e-12)
    assert np.allclose(np.linalg.norm(np.asarray(cond["z"]), axis=-1), np.sqrt(Z), atol=1e-10)
    # the pool q_fn integrates over is still the ONLINE centered feature (unchanged)
    np.testing.assert_allclose(
        np.asarray(cond["_pool_phi"]),
        np.asarray(critic._feature_net(params["hilp_phi1"], params["hilp_stats"], nobs)), atol=1e-12)


def test_regression_mixes_half_the_rows_with_sphere_draws():
    critic, params, _t = _build()
    assert critic.mix_ratio == 0.5           # the tdjc substrate's train_goal_ratio
    net = _net(critic, params)
    batch = _batch()
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
    # missing goal keys is a named error, not a permuted-next-state fallback
    with pytest.raises(KeyError, match="hilp_goal_sampling"):
        critic.sample_cond(net, _batch(goals=False), jax.random.PRNGKey(0))


def test_regression_stage_loss_conditions_on_the_value_leg_and_bootstraps_there():
    critic, params, target = _build()
    batch = _batch()
    cond = critic.sample_cond(_net(critic, params), batch, jax.random.PRNGKey(0))
    rng = np.random.default_rng(2)
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT_DIM)))),
            "next_action_vgoal": jnp.asarray(np.tanh(rng.standard_normal((B, ACT_DIM))))}
    l0, _ = critic.stage_loss("sf", params, target, batch, cond, boot)
    l1, _ = critic.stage_loss("sf", params, target, batch, {**cond, "z": -cond["z"]}, boot)
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    l2, _ = critic.stage_loss("sf", params, target, batch,
                              {**cond, "_cvec_vgoal": -cond["_cvec_vgoal"]}, boot)
    assert not np.allclose(np.asarray(l0), np.asarray(l2))
    l3, _ = critic.stage_loss("sf", params, target, batch, cond,
                              {**boot, "next_action": -boot["next_action"]})
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l3))
    with pytest.raises(KeyError, match="next_action_vgoal"):
        critic.stage_loss("sf", params, target, batch, cond, {"next_action": boot["next_action"]})
    # the phi stage is the parent's, verbatim, and the CE still sends no gradient into phi
    lp, _ = critic.stage_loss("phi", params, target, batch, cond, boot)
    lp_ref, _ = HILPCritic.stage_loss(critic, "phi", params, target, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(lp), np.asarray(lp_ref))
    g = jax.grad(lambda pp: critic.stage_loss(
        "sf", {**params, "hilp_phi1": pp}, target, batch, cond, boot)[0])(params["hilp_phi1"])
    assert _leaf_abs_sum(g) == 0.0


def test_regression_q_fn_reads_the_centered_slow_tower_reward():
    critic, params, _t = _build()
    batch = _batch()
    cond = critic.sample_cond(_net(critic, params), batch, jax.random.PRNGKey(0))
    obs, actions = batch["observations"], batch["actions"]
    q, qs2 = critic.q_fn(params, obs, cond, actions)
    psi = np.asarray(critic._sf(params["hilp_sf"], critic._le(params, obs), cond["z"], actions))
    logits = np.einsum("pbd,cd->pbc", psi, np.asarray(cond["_pool_phi"])) / critic.temp
    w = np.exp(logits - logits.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    mu = np.asarray(params["hilp_zstats"]["mu"])
    bz = np.asarray(critic._phi(params["hilp_phi_zenc"], batch["next_observations"])) - mu[None]
    np.testing.assert_allclose(np.asarray(cond["_pool_zenc_c"]), bz, atol=1e-12)
    kg = np.einsum("cz,bz->bc", bz, np.asarray(cond["z"])) / np.sqrt(Z)
    q_np = (w * kg[None]).sum(-1).mean(0)
    np.testing.assert_allclose(np.asarray(q), q_np, atol=1e-10)
    np.testing.assert_allclose(np.asarray(qs2), (q_np - q_np.mean())[None], atol=1e-10)
    gp = jax.grad(lambda p: critic.q_fn(p, obs, cond, actions)[0].sum())(params)
    assert _leaf_abs_sum(gp) == 0.0, "q_fn leaked gradient into critic params"
    ga = jax.grad(lambda a: critic.q_fn(params, obs, cond, a)[0].sum())(actions)
    assert _leaf_abs_sum(ga) > 0.0


def _agent(name, overrides=()):
    ex_obs, ex_act = _ex()
    return Agent.create(name, "flowbc", 0, ex_obs, ex_act, _cfg(name, overrides))


def test_regression_agent_update_stats_writes_the_slow_tower_moments_and_pool():
    agent = _agent("hilp_density_reg")
    assert agent.critic.task_index == "regression"
    b1 = _batch(1)
    pre = agent.net
    a1, info = agent.update(b1)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), k
    for g in _REG_STATE_GROUPS:
        assert int(a1.net[g].step) == int(pre[g].step)
    assert int(a1.net["hilp_phi1"].step) == int(pre["hilp_phi1"].step) + 1
    assert int(a1.net["hilp_sf"].step) == int(pre["hilp_sf"].step) + 1
    tau = agent.critic.z_enc_tau
    exp_zenc = jax.tree_util.tree_map(lambda p, ep: p * tau + ep * (1 - tau),
                                      pre["hilp_phi1"].params, pre["hilp_phi_zenc"].params)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12),
                           a1.net["hilp_phi_zenc"].params, exp_zenc)
    st = a1.net["hilp_zstats"].params
    assert float(st["filled"]) == 1.0
    np.testing.assert_array_equal(np.asarray(st["pool"]), np.asarray(b1["next_observations"])[:B])
    b = np.asarray(agent.critic._phi(pre["hilp_phi_zenc"].params, b1["next_observations"]))
    mu_b = b.mean(0); db = b - mu_b
    s = agent.critic.z_sigma_tau
    np.testing.assert_allclose(np.asarray(st["mu"]), mu_b * s, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st["cov"]), np.eye(Z) * (1 - s) + (db.T @ db / B) * s, atol=1e-12)
    # HILP's own running mean still moves, exactly as the parent writes it
    m = agent.critic.stats_momentum
    phi1 = np.asarray(agent.critic._phi(pre["hilp_phi1"].params, b1["observations"]))
    np.testing.assert_allclose(np.asarray(a1.net["hilp_stats"].params["mean"]),
                               phi1.mean(0) * m, atol=1e-12)
    # the pool is frozen on the first update
    a2 = a1.update(_batch(2))[0]
    np.testing.assert_array_equal(np.asarray(a2.net["hilp_zstats"].params["pool"]), np.asarray(st["pool"]))
    # map_eval_cond: the env goal goes through the same solve on the frozen pool -- and
    # NOT through HILP's phi(g) - phi(s) direction, so it no longer needs observations
    g = jnp.asarray(np.random.default_rng(9).standard_normal((OBS_DIM,)))
    z = a2.critic.map_eval_cond(a2.net, {"z": jnp.zeros((Z,)), "goal": g})["z"]
    pool = a2.net["hilp_zstats"].params["pool"]
    kg = _delta_phi(g[None], _raw_block_cols(pool, BLOCKS), BLOCKS)
    z_ref = a2.critic.infer_cond(a2.net, pool, kg[0])["z"]
    np.testing.assert_allclose(np.asarray(z), np.asarray(z_ref), atol=1e-12)
    z_obs = a2.critic.map_eval_cond(a2.net, {"z": jnp.zeros((Z,)), "goal": g},
                                    observations=b1["observations"][0])["z"]
    np.testing.assert_array_equal(np.asarray(z), np.asarray(z_obs))
    z_hilp = HILPCritic.map_eval_cond(a2.critic, a2.net, {"z": jnp.zeros((Z,)), "goal": g},
                                      observations=b1["observations"][0])["z"]
    assert not np.allclose(np.asarray(z), np.asarray(z_hilp))
    # a cond without a goal passes through untouched
    assert a2.critic.map_eval_cond(a2.net, {"z": z})["z"] is z
    # the eval harness rolls ONE observation at a time against the env's goal frame
    act = a2.sample_actions(b1["observations"][0], goals=g)
    assert act.shape == (ACT_DIM,) and np.isfinite(np.asarray(act)).all()
    # Route B through the Agent
    r = jnp.asarray(np.random.default_rng(8).standard_normal((B,)))
    a3 = a2.infer_eval_cond(b1["next_observations"], r)
    assert np.allclose(np.linalg.norm(np.asarray(a3.cond_eval["z"])), np.sqrt(Z), atol=1e-10)


def test_regression_agent_runs_with_the_tdjc_left_encoder_and_the_oc_input():
    """The campaign substrate has td_jepa's left encoder ON; the _oc yaml adds the
    object-coordinate phi input. Both must update end to end with finite losses."""
    le = ["~hilp_left_encoder.identity", "hilp_left_encoder.hidden_dim=8",
          "hilp_left_encoder.hidden_layers=2"]
    ag = _agent("hilp_density_reg", le)
    assert ag.critic.le_on and "hilp_le" in ag.net
    ag, info = ag.update(_batch(1))
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    oc = _agent("hilp_density_reg_oc", le)
    assert oc.critic.backward_input == "hit_spec" and oc.critic.blocks == BLOCKS
    assert oc.critic.kernel_blocks == BLOCKS
    # phi's first Dense is 3 wide: it consumes the sliced block, not the 37-wide state
    assert oc.net["hilp_phi1"].params["Dense_0"]["kernel"].shape[0] == 3
    assert oc.net["hilp_phi_zenc"].params["Dense_0"]["kernel"].shape[0] == 3
    oc, info = oc.update(_batch(1))
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    g = jnp.asarray(np.random.default_rng(9).standard_normal((OBS_DIM,)))
    act = oc.sample_actions(_batch(1)["observations"][0], goals=g)
    assert np.isfinite(np.asarray(act)).all()


# ==========================================================================================
# the yamls, the aliases, the definers set
# ==========================================================================================
def test_reg_yamls_are_the_tdjc_substrate_plus_the_regression_block():
    base = make_config("hilp_density_tdjc", "flowbc", "cube_single", [])
    reg = make_config("hilp_density_reg", "flowbc", "cube_single", [])
    oc = make_config("hilp_density_reg_oc", "flowbc", "cube_single", [])
    for c in (reg, oc):
        assert c["hilp_density_task_index"] == "regression"
        assert c["hilp_goal_sampling"] is True and c["dataset_class"] == "HILPFutureDataset"
        assert c["actor_p_trajgoal"] == 1.0 and c["actor_p_curgoal"] == 0.0
        assert c["hilp_density_z_eval_pool_m"] == 256 and c["batch_size"] == 256
        for k in ("z_dim", "L_dim", "phi", "sf", "hilp_sf_archi", "hilp_left_encoder",
                  "hilp_q_loss", "hilp_head_reduce", "hilp_actor_head_reduce",
                  "hilp_sf_loss_scale", "hilp_mix_ratio", "hilp_expectile", "group_lr",
                  "measure_density_temp", "measure_density_actor_temp", "actor_q_center",
                  "stages_from_pre_step_params", "hilp_p_randomgoal", "measure_hit_spec"):
            assert c[k] == base[k], k
    assert reg["backward_input"] == "state" and oc["backward_input"] == "hit_spec"
    assert "hilp_density_task_index" not in base and "hilp_goal_sampling" not in base
    for alias in ("hilp_density_reg", "hilp_density_reg_oc"):
        assert CRITICS[CRITIC_ALIASES[alias]] is HILPDensityCritic
    assert hasattr(HILPDensityCritic, "update_stats")


# ==========================================================================================
# the dataset: HILPFutureDataset + hindsight goals
# ==========================================================================================
def _toy_dataset(n=60, T=10, obs=OBS_DIM, act=ACT_DIM, seed=0):
    rng = np.random.default_rng(seed)
    terminals = np.zeros(n, np.float32)
    terminals[np.arange(1, n // T + 1) * T - 1] = 1.0
    return Dataset.create(observations=rng.standard_normal((n, obs)).astype(np.float32),
                          actions=rng.standard_normal((n, act)).astype(np.float32),
                          terminals=terminals)


def test_hilp_future_dataset_serves_hindsight_goals_only_when_asked():
    base = _toy_dataset()
    cfg = make_config("hilp_density_tdjc", "flowbc", "cube_single", [])
    np.random.seed(0)
    b0 = make_dataset(base, cfg).sample(16)
    assert "actor_goals" not in b0 and "value_goals" not in b0
    assert b0["hilp_future_observations"].shape == (16, OBS_DIM)
    np.random.seed(0)
    b0b = make_dataset(base, {**cfg, "hilp_goal_sampling": False}).sample(16)
    for k in b0:
        np.testing.assert_array_equal(b0[k], b0b[k])

    reg = make_config("hilp_density_reg", "flowbc", "cube_single", [])
    ds = make_dataset(base, reg)
    np.random.seed(0)
    idxs = base.get_random_idxs(16)
    np.random.seed(0)
    b1 = ds.sample(16)
    # the geometric future draw is untouched: same np.random prefix as the plain dataset
    np.random.seed(0)
    b0c = make_dataset(base, cfg).sample(16)
    np.testing.assert_array_equal(b1["hilp_future_observations"], b0c["hilp_future_observations"])
    assert b1["actor_goals"].shape == b1["value_goals"].shape == (16, OBS_DIM)
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
