"""Density-FB z encoder: the centered regression from reward samples to z."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent
from impls.critics.density_fb import (
    DensityFBCritic,
    _delta_phi,
    _density_queries,
    _raw_block_cols,
    _scale_block_cols,
    _sphere_or_neutral,
)
from impls.utils.config_utils import load_config

B, OBS, ACT, D = 16, 40, 5, 8
POOL_M = 256  # eval pool size


def _cfg(z=True, **over):
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
        infer_z=z,
    )
    cfg.update(over)
    return cfg


def _ex(n=B):
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(rng.standard_normal((n, OBS))),
        jnp.asarray(np.tanh(rng.standard_normal((n, ACT)))),
    )


def _batch(seed=1, b=B):
    rng = np.random.default_rng(seed)
    return dict(
        observations=jnp.asarray(rng.standard_normal((b, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((b, OBS))),
        value_goals=jnp.asarray(rng.standard_normal((b, OBS))),
        actor_goals=jnp.asarray(rng.standard_normal((b, OBS))),
        rewards=jnp.asarray(rng.integers(0, 2, size=b) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=b).astype(float)),
    )


def _built(z=True, **over):
    """Return (critic, params) from build()."""
    ex_obs, ex_act = _ex()
    over.setdefault("eval_pool_size", 8)
    critic, params, _targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(z, **over))
    return critic, params


class _FakeTS:
    """TrainState stand-in exposing only `.params`."""

    def __init__(self, params):
        self.params = params


def _net_from(params):
    """Wrap raw param trees as a `net` dict."""
    return {k: _FakeTS(v) for k, v in params.items()}


def _stats(params, **over):
    """Stored z-encoder stats with optional overrides."""
    return {**params["density_zstats"], **over}


def _pool(m=POOL_M, seed=7):
    """Raw [m, OBS] observations and their eps-scaled block coords."""
    obs = jnp.asarray(np.random.default_rng(seed).standard_normal((m, OBS)))
    return obs, _density_queries(obs, _built()[0].blocks)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_z_encoder_is_off_by_default_in_the_shipped_config():
    """The z encoder is off in the default config."""
    cfg = load_config("density_fb", "flowbc", "cube_single")
    assert cfg["infer_z"] is False
    critic, _params, _t = DensityFBCritic.build(
        *_ex(), {**cfg, "seed": 0, "batch_size": B}
    )
    assert critic.infer_z is False
    assert critic.cvec_dim == OBS, "conditioning width must stay the observation width"
    assert critic.PROVIDES_COND == frozenset({"value_goal", "goal"})


def test_infer_cond_off_is_still_the_placeholder():
    """With the z encoder off, infer_cond returns a zero goal."""
    ex_obs, ex_act = _ex()
    critic, params, _t = DensityFBCritic.build(ex_obs, ex_act, _cfg(z=False))
    out = critic.infer_cond(None, ex_obs, jnp.ones((B,)))
    assert set(out) == {"goal"}
    assert np.array_equal(np.asarray(out["goal"]), np.zeros(OBS))


def test_sample_cond_off_publishes_the_raw_value_goal():
    critic, _params = _built(z=False)
    batch = _batch()
    cond = critic.sample_cond(None, batch, jax.random.PRNGKey(0))
    assert "z" not in cond
    assert np.array_equal(
        np.asarray(cond["_cvec_vgoal"]), np.asarray(batch["value_goals"])
    )


def test_z_encoder_is_reward_sensitive():
    """Different rewards give different indices."""
    critic, params = _built()
    _obs, u_sc = _pool()
    rng = np.random.default_rng(11)
    rewards = jnp.asarray(rng.standard_normal((5, POOL_M)))
    z = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, rewards))
    assert z.shape == (5, D)
    assert np.isfinite(z).all()
    pair_cos = [abs(_cos(z[i], z[j])) for i in range(5) for j in range(i + 1, 5)]
    assert max(pair_cos) < 0.99, (
        f"five independent reward draws produced near-identical indices (max |cos| = "
        f"{max(pair_cos):.4f}) -- the encoder is not reading the reward"
    )


def test_z_encoder_norm_is_sqrt_d():
    """The inferred z has norm sqrt(d)."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(12).standard_normal((4, POOL_M)))
    z = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D), atol=1e-10)


@pytest.mark.parametrize("beta", [0.5, -3.0, 100.0])
def test_z_encoder_is_invariant_to_an_additive_reward_shift(beta):
    """The inferred z is invariant to r -> r + beta."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(13).standard_normal((3, POOL_M)))
    z0 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r + beta))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"an additive reward gauge moved the inferred z by "
        f"{np.abs(z0 - z1).max():.3e} at beta={beta}"
    )


@pytest.mark.parametrize("alpha", [1e-3, 7.0, 1e3])
def test_z_encoder_is_invariant_to_a_positive_reward_rescale(alpha):
    """The inferred z is invariant to r -> alpha r for alpha > 0."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(14).standard_normal((3, POOL_M)))
    z0 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, alpha * r))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"a positive reward rescale moved the inferred z by "
        f"{np.abs(z0 - z1).max():.3e} at alpha={alpha}"
    )


def test_z_encoder_is_invariant_to_the_additive_gauge_on_the_backward_tower():
    """The inferred z is invariant to B -> B + b when mu shifts with it."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(15).standard_normal((3, POOL_M)))

    b_params = params["density_b_zenc"]
    last = max(b_params, key=lambda k: int(k.split("_")[-1]))
    shift = jnp.asarray(np.random.default_rng(16).standard_normal(D) * 25.0)
    shifted = {
        **b_params,
        last: {**b_params[last], "bias": b_params[last]["bias"] + shift},
    }
    # The shift must move B
    b0 = np.asarray(critic._B(b_params, u_sc))
    b1 = np.asarray(critic._B(shifted, u_sc))
    assert np.allclose(b1 - b0, np.asarray(shift)[None], atol=1e-8)
    assert np.abs(b1 - b0).max() > 1.0

    # mu shifts with B
    mu = jnp.mean(critic._B(b_params, u_sc), axis=0)
    st0 = _stats(params, mu=mu)
    st1 = _stats(params, mu=mu + shift)
    z0 = np.asarray(critic._infer_z_from_rewards(b_params, st0, u_sc, r))
    z1 = np.asarray(critic._infer_z_from_rewards(shifted, st1, u_sc, r))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"the additive gauge on B moved the inferred z by {np.abs(z0 - z1).max():.3e}"
    )


def test_a_constant_reward_falls_back_to_the_neutral_z():
    """A constant reward gives the neutral z 1_d."""
    critic, params = _built()
    _obs, u_sc = _pool()
    z = np.asarray(
        critic.z_from_reward_samples(
            _net_from(params), u_sc, jnp.full((2, POOL_M), 3.5)
        )
    )
    assert np.allclose(z, np.ones((2, D)), atol=1e-12)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D))


def test_sphere_or_neutral_normalizes_exactly():
    """_sphere_or_neutral gives norm sqrt(d), or 1_d on a degenerate input."""
    x = jnp.asarray([[3.0, 4.0, 0.0, 0.0]])
    out = np.asarray(_sphere_or_neutral(x, 4.0, 1e-6))
    assert np.allclose(np.linalg.norm(out), 2.0, atol=0)
    assert np.allclose(
        np.asarray(_sphere_or_neutral(1e-9 * x, 4.0, 1e-6)), np.ones((1, 4))
    )


def test_centered_moment_matches_a_numpy_reference():
    """The centered moment m_c matches a numpy reference."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    rng = np.random.default_rng(31)
    r = jnp.asarray(rng.standard_normal((3, 64)))
    mu = jnp.asarray(rng.standard_normal(D))  # not the pool mean

    got = np.asarray(critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r))
    b = np.asarray(critic._B(params["density_b_zenc"], u_sc))  # [M, d]
    db = b - np.asarray(mu)[None]
    dr = np.asarray(r) - np.asarray(r).mean(axis=-1, keepdims=True)
    assert np.allclose(got, dr @ db / 64, atol=1e-12)
    # Not the uncentered moment
    assert not np.allclose(got, np.asarray(r) @ db / 64, atol=1e-6)


def test_moments_are_the_centered_covariance_not_the_second_moment():
    """The stored C_hat is an EMA of the centered covariance of B."""
    agent, _cfg_ = _agent()
    batch = _batch()
    st_before = agent.net["density_zstats"].params
    updated = agent.update(batch)[0]
    st = updated.net["density_zstats"].params

    # EMA target for this step
    u_sc = _density_queries(batch["next_observations"], agent.critic.blocks)
    b = np.asarray(agent.critic._B(agent.net["density_b_zenc"].params, u_sc))
    mu_b, tau = b.mean(0), agent.critic.feature_stats_tau
    db = b - mu_b[None]
    cov_b = db.T @ db / b.shape[0]
    assert np.allclose(
        np.asarray(st["mu"]),
        np.asarray(st_before["mu"]) * (1 - tau) + mu_b * tau,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray(st["cov"]),
        np.asarray(st_before["cov"]) * (1 - tau) + cov_b * tau,
        atol=1e-10,
    )
    # Centered, not the second moment
    cov_unc = np.asarray(st_before["cov"]) * (1 - tau) + (b.T @ b / b.shape[0]) * tau
    assert not np.allclose(np.asarray(st["cov"]), cov_unc, atol=1e-6), (
        "C_hat is the UNCENTERED second moment -- the exact object spec 30 replaced"
    )
    assert np.allclose(np.asarray(st["cov"]), np.asarray(st["cov"]).T, atol=0), (
        "C_hat must be symmetric"
    )


def test_reward_centering_holds_at_a_base_mean_that_is_not_the_pool_mean():
    """m_c is invariant to a reward shift for a mu away from the pool mean."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    rng = np.random.default_rng(32)
    r = jnp.asarray(rng.standard_normal((3, 64)))
    mu = jnp.asarray(rng.standard_normal(D) * 10.0)  # not the pool mean
    b = np.asarray(critic._B(params["density_b_zenc"], u_sc))
    assert np.abs((b - np.asarray(mu)[None]).sum(0)).max() > 1.0, (
        "this mu is too close to the pool mean for the test to discriminate"
    )

    m0 = np.asarray(critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r))
    m1 = np.asarray(
        critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r + 7.0)
    )
    assert np.allclose(m0, m1, atol=1e-10), (
        f"an additive reward gauge moved m_c by {np.abs(m0 - m1).max():.3e} at an off-pool "
        f"base mean -- the reward-centering (r - rbar) is missing or wrong"
    )


def test_the_ridge_uses_the_mean_eigenvalue_convention_and_floors_the_spectrum():
    """The ridge solve uses mean-eigenvalue units and floors the spectrum."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    b = critic._B(params["density_b_zenc"], u_sc)
    dbv = b - jnp.mean(b, axis=0)[None]
    cov = dbv.T @ dbv / b.shape[0]
    m_c = jnp.asarray(np.random.default_rng(33).standard_normal((2, D)))
    got = np.asarray(critic._z_centered_coef(m_c, cov))

    c = np.asarray(cov)
    tr = np.trace(c)
    ev, vec = np.linalg.eigh(0.5 * (c + c.T))
    ev_f = np.maximum(ev, critic.cov_eig_floor * tr / D)
    lam = critic.ridge_lambda * tr / D
    assert np.allclose(
        got, ((np.asarray(m_c) @ vec) / (ev_f + lam)[None, :]) @ vec.T, atol=1e-10
    )
    # A larger ridge shrinks the coefficient
    critic_big = _built(ridge_lambda=10.0)[0]
    assert np.linalg.norm(
        np.asarray(critic_big._z_centered_coef(m_c, cov))
    ) < np.linalg.norm(got)


def test_one_reward_sample_raises_instead_of_returning_a_zero_z():
    """One reward sample raises instead of returning a zero z."""
    critic, params = _built()
    obs, _u = _pool(m=1)
    u_sc = _density_queries(obs, critic.blocks)
    with pytest.raises(ValueError, match="M > 1"):
        critic.z_from_reward_samples(_net_from(params), u_sc, jnp.ones((1,)))


def _agent(actor="flowbc", z=True, **over):
    cfg = load_config("density_fb", actor, "cube_single")
    cfg = {
        **cfg,
        "batch_size": B,
        "z_dim": D,
        "value_hidden_dims": [32, 32],
        "infer_z": z,
        "eval_pool_size": 8,
        **over,
    }
    ex_obs, ex_act = _ex()
    return Agent.create("density_fb", actor, 0, ex_obs, ex_act, cfg), cfg


def test_infer_eval_cond_runs_the_m256_reward_sample_path():
    """Agent.infer_eval_cond builds z from 256 reward samples and the actor runs on it."""
    agent, cfg = _agent()
    assert agent.critic.infer_z and agent.critic.cvec_dim == D
    assert "z" in agent.critic.PROVIDES_COND

    obs_pool = jnp.asarray(np.random.default_rng(21).standard_normal((POOL_M, OBS)))
    rewards = jnp.asarray(np.random.default_rng(22).standard_normal((POOL_M,)))
    assert obs_pool.shape[0] == POOL_M == 256

    updated = agent.infer_eval_cond(obs_pool, rewards)
    z = np.asarray(updated.cond_eval["z"])
    assert set(updated.cond_eval) == {"z"}
    assert z.shape == (D,), "a flat [M] reward vector describes ONE task -> a 1-D z"
    assert np.isfinite(z).all()
    assert np.allclose(np.linalg.norm(z), np.sqrt(D), atol=1e-10)
    # Not the placeholder
    assert not np.allclose(z, np.asarray(agent.cond_eval["z"]))

    # goals=None rolls the inferred z
    acts = updated.sample_actions(jnp.zeros((OBS,)), seed=jax.random.PRNGKey(3))
    assert np.asarray(acts).shape == (ACT,) and np.isfinite(np.asarray(acts)).all()


def test_the_whole_256_sample_pool_is_read():
    """The last pool row affects the inferred z."""
    agent, _cfg = _agent()
    obs_pool = jnp.asarray(np.random.default_rng(21).standard_normal((POOL_M, OBS)))
    rewards = jnp.asarray(np.random.default_rng(22).standard_normal((POOL_M,)))
    z0 = np.asarray(agent.infer_eval_cond(obs_pool, rewards).cond_eval["z"])
    bumped = rewards.at[POOL_M - 1].add(50.0)
    z1 = np.asarray(agent.infer_eval_cond(obs_pool, bumped).cond_eval["z"])
    assert not np.allclose(z0, z1, atol=1e-6), (
        "the last pool row had no effect on the inferred z"
    )


def test_infer_cond_uses_the_eps_scaled_queries_not_the_raw_block_columns():
    """infer_cond feeds eps-scaled queries, not raw block columns, to B."""
    critic, params = _built()
    obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(23).standard_normal((POOL_M,)))
    z_scaled = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z_raw = np.asarray(
        critic.z_from_reward_samples(
            _net_from(params), _raw_block_cols(obs, critic.blocks), r
        )
    )
    assert not np.allclose(z_scaled, z_raw, atol=1e-6)

    z_infer = np.asarray(critic.infer_cond(_net_from(params), obs, r)["z"])
    assert np.allclose(z_infer, z_scaled[0], atol=1e-12)


def test_cvec_vgoal_is_z_shaped_and_matches_the_actor_width():
    """_cvec_vgoal is the inferred z at the value goal, with the actor's width."""
    agent, _cfg = _agent()
    batch = _batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))

    # Width matches the actor
    assert agent.critic.cvec_dim == D
    assert cond["z"].shape == (B, D)
    assert cond["_cvec_vgoal"].shape == (B, D), (
        f"_cvec_vgoal is {cond['_cvec_vgoal'].shape}, not the actor's (B, cvec_dim)"
    )
    assert cond["_cvec_vgoal"].shape[-1] == agent.critic.cvec_dim
    assert cond["_cvec_vgoal"].shape != batch["value_goals"].shape, (
        "_cvec_vgoal is still the RAW value goal -- the exact bug this test exists for"
    )
    assert np.allclose(
        np.linalg.norm(np.asarray(cond["_cvec_vgoal"]), axis=-1), np.sqrt(D)
    )

    # Inferred z at the value goal, not the actor goal
    assert not np.allclose(np.asarray(cond["z"]), np.asarray(cond["_cvec_vgoal"]))

    # Value-goal bootstrap action differs from the actor-goal one
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    assert "next_action_vgoal" in boot
    assert not np.allclose(
        np.asarray(boot["next_action"]), np.asarray(boot["next_action_vgoal"])
    )

    # The raw value goal has the wrong width
    with pytest.raises(Exception):
        agent.actor.sample_bootstrap(
            agent.net,
            batch,
            {**cond, "_cvec_vgoal": batch["value_goals"]},
            jax.random.PRNGKey(1),
        )


def test_stage_loss_conditions_f_on_the_value_goal_z():
    """F in the stage loss is conditioned on _cvec_vgoal, not the raw value goal."""
    agent, _cfg = _agent()
    batch = _batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))

    def ce(c):
        return float(np.asarray(agent.compute_static(batch, c, boot)["density_ce"]))

    base = ce(cond)
    assert ce({**cond, "_cvec_vgoal": jnp.zeros_like(cond["_cvec_vgoal"])}) != base
    assert ce({**cond, "value_goal": jnp.zeros_like(cond["value_goal"])}) == base


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc"])
def test_a_real_update_runs_with_the_z_encoder_on(actor):
    """A full update runs with the z encoder on."""
    agent, _cfg = _agent(actor)
    updated, info = agent.update(_batch())
    assert np.isfinite(float(np.asarray(info["density_ce"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    moved = [
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(agent.net["density_f"].params),
            jax.tree_util.tree_leaves(updated.net["density_f"].params),
        )
    ]
    assert any(moved), "density_f params did not move"


def test_the_z_encoder_never_carries_a_gradient_into_the_backward_tower():
    """The z encoder passes no gradient to the backward tower."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(24).standard_normal((2, 64)))

    def scalar(bp):
        return jnp.sum(critic._infer_z_from_rewards(bp, _stats(params), u_sc, r))

    g = jax.grad(scalar)(params["density_b_zenc"])
    assert all(np.all(np.asarray(l) == 0.0) for l in jax.tree_util.tree_leaves(g))


def test_the_training_z_mixes_in_sphere_draws():
    """z_random_frac mixes sphere draws into the training z."""
    batch = _batch()
    a0, _ = _agent(z=True, z_random_frac=0.0)
    a1, _ = _agent(z=True, z_random_frac=1.0)
    c0 = a0.critic.sample_cond(a0.net, batch, jax.random.PRNGKey(5))
    c1 = a1.critic.sample_cond(a1.net, batch, jax.random.PRNGKey(5))
    assert not np.allclose(np.asarray(c0["z"]), np.asarray(c1["z"]))
    # Both branches have norm sqrt(d)
    for c in (c0, c1):
        assert np.allclose(
            np.linalg.norm(np.asarray(c["z"]), axis=-1), np.sqrt(D), atol=1e-6
        )
    # At frac=0 every row is the encoder's output
    u_raw = _raw_block_cols(batch["next_observations"], a0.critic.blocks)
    u_sc = _density_queries(batch["next_observations"], a0.critic.blocks)
    r = _delta_phi(batch["actor_goals"], u_raw, a0.critic.blocks)
    z_enc = np.asarray(a0.critic.z_from_reward_samples(a0.net, u_sc, r))
    assert np.allclose(np.asarray(c0["z"]), z_enc, atol=1e-10)


def test_the_z_encoder_state_groups_exist_only_when_the_z_encoder_is_on():
    """z-encoder state groups exist only with the z encoder on."""
    off, _ = _agent(z=False)
    on, _ = _agent(z=True)
    assert set(off.net) - {"actor", "actor_vf"} == {"density_f", "density_b"}
    assert {"density_b_zenc", "density_zstats"} <= set(on.net)
    assert off.critic.GROUPS == ("density_f", "density_b")
    assert on.critic.GROUPS == (
        "density_f",
        "density_b",
        "density_b_zenc",
        "density_zstats",
    )


def test_the_z_encoder_tower_starts_as_a_copy_of_density_b_and_then_lags_it():
    """The slow copy of B starts as a copy of density_b and tracks it as a slow EMA."""
    agent, _c = _agent()
    b0 = agent.net["density_b"].params
    z0 = agent.net["density_b_zenc"].params
    for a, b in zip(jax.tree_util.tree_leaves(b0), jax.tree_util.tree_leaves(z0)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "zenc must init as a copy"

    updated = agent.update(_batch())[0]
    tau = agent.critic.slow_backward_tau
    expect = jax.tree_util.tree_map(lambda p, ep: p * tau + ep * (1 - tau), b0, z0)
    for got, exp in zip(
        jax.tree_util.tree_leaves(updated.net["density_b_zenc"].params),
        jax.tree_util.tree_leaves(expect),
    ):
        assert np.allclose(np.asarray(got), np.asarray(exp), atol=1e-12)

    # It differs from the online tower
    on_new = updated.net["density_b"].params
    zn_new = updated.net["density_b_zenc"].params
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
        for a, b in zip(
            jax.tree_util.tree_leaves(on_new), jax.tree_util.tree_leaves(zn_new)
        )
    ), "the z-encoder tower equals the online tower -- it is not lagging anything"

    # and from the target tower
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
        for a, b in zip(
            jax.tree_util.tree_leaves(updated.target["density_b"]),
            jax.tree_util.tree_leaves(zn_new),
        )
    )


def test_the_z_encoder_reads_the_slow_tower_not_the_online_or_target_one():
    """The z encoder reads the slow tower, not the online or target tower."""
    agent, _c = _agent()
    for seed in (1, 2, 3):
        agent = agent.update(_batch(seed=seed))[0]
    critic, net = agent.critic, agent.net
    st = net["density_zstats"].params
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(61).standard_normal((3, 64)))

    z_slow = np.asarray(
        critic._infer_z_from_rewards(net["density_b_zenc"].params, st, u_sc, r)
    )
    z_online = np.asarray(
        critic._infer_z_from_rewards(net["density_b"].params, st, u_sc, r)
    )
    z_target = np.asarray(
        critic._infer_z_from_rewards(agent.target["density_b"], st, u_sc, r)
    )
    # The three towers must have diverged
    assert not np.allclose(z_slow, z_online, atol=1e-6), (
        "online and slow towers still agree -- run more updates or the test is vacuous"
    )
    assert not np.allclose(z_slow, z_target, atol=1e-6), (
        "target and slow towers still agree -- run more updates or the test is vacuous"
    )

    got = np.asarray(critic.z_from_reward_samples(net, u_sc, r))
    assert np.allclose(got, z_slow, atol=0), (
        "the inferred z does not read density_b_zenc"
    )
    assert not np.allclose(got, z_online, atol=1e-6), (
        "the inferred z reads the ONLINE tower"
    )
    assert not np.allclose(got, z_target, atol=1e-6), (
        "the inferred z reads the TARGET tower"
    )

    # map_eval_cond uses the same tower
    mapped = critic.map_eval_cond(net, {"goal": jnp.zeros((2, OBS))})
    pool_sc = _scale_block_cols(st["pool"], critic.blocks)
    kg = _delta_phi(jnp.zeros((2, OBS)), st["pool"], critic.blocks)
    assert np.allclose(
        np.asarray(mapped["z"]),
        np.asarray(
            critic._infer_z_from_rewards(net["density_b_zenc"].params, st, pool_sc, kg)
        ),
        atol=0,
    )


def test_the_moments_are_stored_and_not_recomputed_from_the_batch():
    """The moments are a stored EMA, not batch statistics."""
    agent, _c = _agent()
    a1 = agent.update(_batch(seed=1))[0]
    a2 = a1.update(_batch(seed=2))[0]
    st = a2.net["density_zstats"].params

    # Moments of the last batch alone
    b2 = _batch(seed=2)
    u_sc = _density_queries(b2["next_observations"], agent.critic.blocks)
    b = np.asarray(agent.critic._B(a1.net["density_b_zenc"].params, u_sc))
    mu_batch, cov_batch = b.mean(0), (b - b.mean(0)).T @ (b - b.mean(0)) / b.shape[0]

    assert not np.allclose(np.asarray(st["mu"]), mu_batch, atol=1e-6), (
        "mu_hat equals the last batch's mean -- these are batch statistics, not an EMA"
    )
    assert not np.allclose(np.asarray(st["cov"]), cov_batch, atol=1e-6), (
        "C_hat equals the last batch's covariance -- these are batch statistics, not an EMA"
    )
    # The EMA has moved off its init
    assert np.abs(np.asarray(st["mu"])).max() > 0
    assert not np.allclose(np.asarray(st["cov"]), np.eye(D), atol=1e-9)


def test_the_stored_moments_change_the_z():
    """Different stored moments give a different inferred z."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(41).standard_normal((2, 64)))
    z_init = np.asarray(
        critic._infer_z_from_rewards(params["density_b_zenc"], _stats(params), u_sc, r)
    )
    b = critic._B(params["density_b_zenc"], u_sc)
    dbv = b - jnp.mean(b, axis=0)[None]
    z_real = np.asarray(
        critic._infer_z_from_rewards(
            params["density_b_zenc"],
            _stats(params, mu=jnp.mean(b, axis=0), cov=dbv.T @ dbv / b.shape[0]),
            u_sc,
            r,
        )
    )
    assert not np.allclose(z_init, z_real, atol=1e-6), (
        "the stored moments do not reach the inferred z -- the EMA is not load-bearing"
    )


def test_the_z_encoder_state_survives_a_serialization_round_trip():
    """z-encoder state survives a serialization round trip."""
    import flax

    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    blob = flax.serialization.to_state_dict(trained)
    fresh, _c2 = _agent()
    restored = flax.serialization.from_state_dict(fresh, blob)

    for k in ("mu", "cov", "pool", "filled"):
        assert np.allclose(
            np.asarray(restored.net["density_zstats"].params[k]),
            np.asarray(trained.net["density_zstats"].params[k]),
            atol=0,
        ), k
    for a, b in zip(
        jax.tree_util.tree_leaves(restored.net["density_b_zenc"].params),
        jax.tree_util.tree_leaves(trained.net["density_b_zenc"].params),
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_z_encoder_state_takes_no_gradient_and_no_optimizer_step():
    """z-encoder state groups take no optimizer step."""
    agent, _c = _agent()
    assert agent.critic.stage_groups("density") == ("density_f", "density_b")
    assert "density_b_zenc" not in agent.critic.TARGETS
    assert "density_zstats" not in agent.critic.TARGETS
    # Only the two towers step
    updated = agent.update(_batch())[0]
    assert int(updated.net["density_f"].step) > int(agent.net["density_f"].step)
    assert int(updated.net["density_zstats"].step) == int(
        agent.net["density_zstats"].step
    )
    assert int(updated.net["density_b_zenc"].step) == int(
        agent.net["density_b_zenc"].step
    )


def test_update_stats_is_a_noop_for_every_other_critic():
    """Only density_fb and hilp define update_stats; it is a no-op with the z encoder off."""
    from impls.agents import CRITICS

    defining = {n for n, c in CRITICS.items() if hasattr(c, "update_stats")}
    assert defining == {"density_fb", "hilp"}
    off, _c = _agent(z=False)
    same = off.critic.update_stats(off.net, off.net, _batch())
    assert same is off.net


def test_the_pool_is_frozen_on_the_first_update_and_never_rewritten():
    """The eval pool is filled on the first update and never rewritten."""
    agent, _c = _agent()
    st0 = agent.net["density_zstats"].params
    assert float(st0["filled"]) == 0.0
    assert np.allclose(np.asarray(st0["pool"]), 0.0), "pool inits degenerate"

    b1 = _batch(seed=1)
    a1 = agent.update(b1)[0]
    st1 = a1.net["density_zstats"].params
    m = agent.critic.eval_pool_size
    assert float(st1["filled"]) == 1.0
    expect = np.asarray(_raw_block_cols(b1["next_observations"], agent.critic.blocks))[
        :m
    ]
    assert np.allclose(np.asarray(st1["pool"]), expect, atol=0)

    a2 = a1.update(_batch(seed=2))[0]
    assert np.allclose(
        np.asarray(a2.net["density_zstats"].params["pool"]),
        np.asarray(st1["pool"]),
        atol=0,
    ), "the pool was rewritten"


def test_map_eval_cond_routes_a_raw_goal_through_the_reward_sample_api():
    """map_eval_cond maps a goal to z via its reward over the pool."""
    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    goal = jnp.asarray(np.random.default_rng(51).standard_normal((3, OBS)))
    mapped = trained.critic.map_eval_cond(trained.net, {"goal": goal})
    z = np.asarray(mapped["z"])
    assert z.shape == (3, D)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D), atol=1e-10)

    # Same as the reward-sample solve on kg over the pool
    st = trained.net["density_zstats"].params
    kg = _delta_phi(goal, st["pool"], trained.critic.blocks)
    u_sc = _scale_block_cols(st["pool"], trained.critic.blocks)
    assert np.allclose(
        z,
        np.asarray(trained.critic.z_from_reward_samples(trained.net, u_sc, kg)),
        atol=1e-12,
    )

    assert max(abs(_cos(z[i], z[j])) for i in range(3) for j in range(i + 1, 3)) < 0.99


def test_map_eval_cond_handles_the_unbatched_eval_goal():
    """map_eval_cond accepts an unbatched goal."""
    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    g = jnp.asarray(np.random.default_rng(52).standard_normal(OBS))
    z = np.asarray(trained.critic.map_eval_cond(trained.net, {"goal": g})["z"])
    assert z.shape == (D,)
    acts = trained.sample_actions(
        jnp.zeros((OBS,)), goals=g, seed=jax.random.PRNGKey(1)
    )
    assert np.asarray(acts).shape == (ACT,) and np.isfinite(np.asarray(acts)).all()


def test_the_pool_uses_the_raw_scaled_coordinate_conventions_correctly():
    """The pool stores raw block coords; B reads the eps-scaled ones."""
    agent, _c = _agent()
    b1 = _batch(seed=1)
    trained = agent.update(b1)[0]
    st = trained.net["density_zstats"].params
    blocks = trained.critic.blocks
    raw = np.asarray(_raw_block_cols(b1["next_observations"], blocks))[
        : agent.critic.eval_pool_size
    ]
    assert np.allclose(np.asarray(st["pool"]), raw, atol=0), (
        "pool must store RAW coords"
    )
    scaled = np.asarray(_scale_block_cols(st["pool"], blocks))
    assert not np.allclose(scaled, np.asarray(st["pool"]), atol=1e-3)
    assert np.allclose(
        scaled,
        np.asarray(_density_queries(b1["next_observations"], blocks))[
            : agent.critic.eval_pool_size
        ],
        atol=1e-10,
    )


def test_the_real_m256_pool_runs_end_to_end():
    """An M=256 eval pool runs end to end."""
    agent, cfg = _agent(eval_pool_size=256, batch_size=POOL_M)
    assert agent.critic.eval_pool_size == 256
    trained = agent.update(_batch(b=POOL_M))[0]
    st = trained.net["density_zstats"].params
    assert np.asarray(st["pool"]).shape[0] == 256
    assert float(st["filled"]) == 1.0
    z = np.asarray(
        trained.critic.map_eval_cond(trained.net, {"goal": jnp.zeros((1, OBS))})["z"]
    )
    assert z.shape == (1, D) and np.isfinite(z).all()


def test_a_pool_larger_than_the_batch_is_refused():
    """An eval pool larger than the batch is refused."""
    with pytest.raises(AssertionError, match="exceeds batch_size"):
        DensityFBCritic.build(*_ex(), _cfg(z=True, eval_pool_size=1024))


def test_the_centered_encoder_refuses_a_fast_z_encoder_tower():
    """The centered z encoder requires slow_backward_tau > 0."""
    with pytest.raises(AssertionError, match="slow_backward_tau"):
        DensityFBCritic.build(
            *_ex(), _cfg(z=True, slow_backward_tau=0.0, eval_pool_size=8)
        )


def _leafmax(a, b):
    return max(
        float(np.abs(np.asarray(x) - np.asarray(y)).max())
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_the_z_encoder_tower_ema_rate_is_pinned_across_several_updates():
    """The slow copy of B is the tau_z EMA of the pre-step online tower."""
    agent, _c = _agent()
    tau = agent.critic.slow_backward_tau
    assert 0.0 < tau < 1.0, "the centered encoder requires a slow z-encoder tower"

    a = agent
    for seed in (1, 2, 3):
        pre_b = a.net["density_b"].params
        pre_z = a.net["density_b_zenc"].params
        a = a.update(_batch(seed=seed))[0]
        expect = jax.tree_util.tree_map(
            lambda p, ep: p * tau + ep * (1 - tau), pre_b, pre_z
        )
        # rtol=0 so small rate errors are visible
        assert _leafmax(a.net["density_b_zenc"].params, expect) < 1e-12, (
            "the z-encoder tower is not the tau_z EMA of the PRE-step online tower"
        )

    # The towers must have separated
    assert _leafmax(a.net["density_b"].params, a.net["density_b_zenc"].params) > 1e-6, (
        "online and z-encoder towers never separated -- the rate above is untested"
    )


def test_a_different_z_encoder_tower_rate_produces_a_different_tower():
    """slow_backward_tau changes the slow copy of B."""
    slow, _c = _agent(slow_backward_tau=1e-4)
    fast, _c = _agent(slow_backward_tau=0.05)
    for seed in (1, 2, 3):
        b = _batch(seed=seed)
        slow, fast = slow.update(b)[0], fast.update(b)[0]
    zgap = _leafmax(
        slow.net["density_b_zenc"].params, fast.net["density_b_zenc"].params
    )
    bgap = _leafmax(slow.net["density_b"].params, fast.net["density_b"].params)
    assert zgap > 1e-6, "slow_backward_tau does not reach the z-encoder tower"
    # The rate also feeds back into the online tower through the actor
    assert zgap > 100 * bgap, (
        f"z-encoder-tower gap {zgap:.3e} is not dominated by the knob (online gap {bgap:.3e})"
    )


def test_the_moments_ema_reads_the_pre_step_z_encoder_tower():
    """The moment EMA uses the pre-step slow copy of B."""
    a, _c = _agent()
    for seed in (1, 2, 3):
        b = _batch(seed=seed)
        pre_zenc = a.net["density_b_zenc"].params
        st_before = a.net["density_zstats"].params
        a = a.update(b)[0]
        st = a.net["density_zstats"].params

        u_sc = _density_queries(b["next_observations"], a.critic.blocks)
        bb = np.asarray(a.critic._B(pre_zenc, u_sc))
        mu_b = bb.mean(0)
        db = bb - mu_b[None]
        cov_b = db.T @ db / bb.shape[0]
        t = a.critic.feature_stats_tau
        assert np.allclose(
            np.asarray(st["mu"]),
            np.asarray(st_before["mu"]) * (1 - t) + mu_b * t,
            rtol=0,
            atol=1e-9,
        ), "mu_hat is not EMA'd at the PRE-step tower"
        assert np.allclose(
            np.asarray(st["cov"]),
            np.asarray(st_before["cov"]) * (1 - t) + cov_b * t,
            rtol=0,
            atol=1e-9,
        ), "C_hat is not EMA'd at the PRE-step tower"

    # Pre- and post-step towers must differ
    pre = a.net["density_b_zenc"].params
    post = jax.tree_util.tree_map(
        lambda p, ep: (
            p * a.critic.slow_backward_tau + ep * (1 - a.critic.slow_backward_tau)
        ),
        a.net["density_b"].params,
        pre,
    )
    assert _leafmax(pre, post) > 1e-9, (
        "pre- and post-step z-encoder towers coincide -- the ordering above is untested"
    )


def test_the_covariance_eigenvalue_floor_is_exercised_and_load_bearing():
    """The ridge solve floors small covariance eigenvalues."""
    critic, _params = _built()
    rng = np.random.default_rng(77)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    cov_np = Q @ np.diag(np.array([1.0, 0.5] + [0.0] * (D - 2))) @ Q.T
    cov = jnp.asarray(cov_np)
    m_c = jnp.asarray(rng.standard_normal((2, D)))

    tr = float(np.trace(cov_np))
    ev, vec = np.linalg.eigh(0.5 * (cov_np + cov_np.T))
    floor = critic.cov_eig_floor * tr / D
    assert (ev < floor).sum() >= D - 2, (
        "this covariance does not exercise the floor -- the test would be vacuous"
    )

    lam = critic.ridge_lambda * tr / D
    got = np.asarray(critic._z_centered_coef(m_c, cov))
    floored = ((np.asarray(m_c) @ vec) / (np.maximum(ev, floor) + lam)[None, :]) @ vec.T
    unfloored = ((np.asarray(m_c) @ vec) / (ev + lam)[None, :]) @ vec.T
    assert np.allclose(got, floored, rtol=0, atol=1e-8), "the spectrum is not floored"
    assert not np.allclose(floored, unfloored, atol=1e-6), (
        "floored and unfloored solves agree -- the assertion above cannot discriminate"
    )


def _built_full(**over):
    ex_obs, ex_act = _ex()
    over.setdefault("eval_pool_size", 8)
    return DensityFBCritic.build(ex_obs, ex_act, _cfg(True, **over))


def _perturbed(params, seed=3):
    """Move the slow tower and mu away from their init values."""
    rng = np.random.default_rng(seed)
    p = dict(params)
    p["density_b_zenc"] = jax.tree_util.tree_map(
        lambda x: x * 1.5 + 0.1 * jnp.asarray(rng.standard_normal(x.shape)),
        params["density_b_zenc"],
    )
    st = dict(params["density_zstats"])
    st["mu"] = jnp.asarray(rng.standard_normal(np.asarray(st["mu"]).shape))
    p["density_zstats"] = st
    return p


def test_sample_cond_publishes_the_centered_slow_basis():
    critic, params, _t = _built_full()
    params = _perturbed(params)
    cond = critic.sample_cond(_net_from(params), _batch(), jax.random.PRNGKey(0))
    expect = (
        np.asarray(critic._B(params["density_b_zenc"], cond["_density_u_sc"]))
        - np.asarray(params["density_zstats"]["mu"])[None]
    )
    np.testing.assert_allclose(
        np.asarray(cond["_density_zenc_u"]), expect, rtol=1e-5, atol=1e-6
    )


def test_q_fn_averages_the_linear_z_reward_under_the_predicted_distribution():
    """q = min over heads of sum_j w(s,a,z,u_j) (B_slow(u_j) - mu)^T z / sqrt(d), B_slow the slow copy of B."""
    obs, act = _ex()
    critic, params, _t = _built_full()
    params = _perturbed(params)
    cond = critic.sample_cond(_net_from(params), _batch(), jax.random.PRNGKey(0))
    q, qs = critic.q_fn(params, obs, cond, act)
    z = cond["z"]
    f = critic._F(params["density_f"], obs, act, z)
    b = critic._B(params["density_b"], cond["_density_u_sc"])
    w = np.asarray(
        jax.nn.softmax(jnp.einsum("hbd,md->hbm", f, b) / critic.temperature, -1)
    )
    r = np.asarray(cond["_density_zenc_u"]) @ np.asarray(z).T / np.sqrt(D)  # [M, B]
    heads = (w * r.T[None]).sum(-1)
    np.testing.assert_allclose(np.asarray(q), heads.min(0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(qs[0]), heads.min(0) - heads.min(0).mean(), rtol=1e-6, atol=1e-8
    )


def test_the_z_encoder_requires_goal_spec_backward_input():
    with pytest.raises(ValueError, match="backward_input"):
        _built_full(backward_input="state")


def test_goal_spec_pool_width_is_the_hit_spec_width():
    _c, params, _t = _built_full()
    assert params["density_zstats"]["pool"].shape[-1] == 3
