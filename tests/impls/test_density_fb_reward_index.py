"""density_fb's REWARD-SAMPLE INDEX (spec 30 P1, the centered dual encoder).

`measure_density_z` (default OFF) replaces the critic's goal conditioning with an index z
inferred from reward samples {(u_j, r_j)}, which is what makes `infer_cond(net, obs,
rewards)` a real zero-shot task interface instead of the placeholder it was.

What this file pins, in the order the properties matter:

  1. OFF is OFF -- `infer_cond` is the old placeholder, PROVIDES_COND is unchanged, and
     cond carries no "z". (The 60-pair byte-identity gate lives in test_boot_contract.py;
     this only names the contract.)
  2. The index RESPONDS to the reward -- different reward families give different z.
  3. The index is INVARIANT to the additive reward gauge r -> r + beta. That is the whole
     point of the centered variant: spec 28's "rank-1 collapse" was largely a softmax-gauge
     mean, and centering alone took the point index's reward reconstruction from AUC .43-.86
     to .99+. It is also invariant to a positive rescale r -> alpha r (the EXACT sqrt(d)
     normalization, no epsilon in the denominator) and to the additive gauge on the backward
     tower B -> B + b (the density softmax is exactly invariant to that, and ||mu_B|| is
     33-103x the centered signal, so it is not a small correction).

     ⚠ READ BEFORE TRUSTING THOSE TWO TESTS. This port computes mu_hat/C_hat as the PLUG-IN
     moments of the very pool the reward samples live on (the shim keeps EMA-tracked ones --
     see the deviation note in impls/critics/density_fb.py). That makes
     sum_j (B(u_j) - mu_hat) EXACTLY zero, so the reward-centering and the B-centering each
     kill BOTH gauges on their own and MASK each other: deleting either one leaves the
     end-to-end invariance tests green (verified by mutation). The formula-level tests in
     section 3b are the ones that actually discriminate -- a numpy reference for m_c, a
     check that C_hat is the CENTERED covariance and not E[BB^T], and the reward-shift
     invariance driven at a base mean that is NOT the pool mean (which is the situation the
     shim's EMA moments are always in).
  4. M = 1 RAISES. The centered moment is identically zero at M=1, so there is no
     one-sample/Dirac limit; a silent zero index there is exactly the failure mode that made
     spec 29's number uninterpretable.
  5. The M=256 reward-sample path actually runs end to end through Agent.infer_eval_cond.
  6. cond["_cvec_vgoal"] is the Z AT THE VALUE GOAL, not the raw goal -- so the TD bootstrap
     draw is z-shaped against a z-shaped actor net. Handing the raw goal there has been
     shipped twice in this tree; it is asserted structurally (width), behaviourally (the
     update runs) and negatively (substituting the raw goal fails).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent
from impls.critics.density_fb import (
    DensityFBCritic, _density_queries, _raw_block_cols, _delta_phi, _sphere_or_neutral,
)
from impls.utils.config_utils import load_config

B, OBS, ACT, D = 16, 40, 5, 8
POOL_M = 256          # the M the spec-30 eval pool is pinned at


def _cfg(z=True, **over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, tau=0.005, seed=0,
        value_hidden_dims=[32, 32], layer_norm=True,
        measure_hit_spec="19:22:0.4", measure_density_dim=D, measure_density_temp=1.0,
        actor_q_center=True, measure_density_z=z,
    )
    cfg.update(over)
    return cfg


def _ex(n=B):
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((n, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((n, ACT)))))


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
    """(critic, params) from build() -- enough for every pure-index assertion."""
    ex_obs, ex_act = _ex()
    critic, params, _targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(z, **over))
    return critic, params


def _pool(m=POOL_M, seed=7):
    """[m, OBS] raw observations drawn from the base measure, and their eps-scaled
    hit-spec block coords (the B tower's own input convention)."""
    obs = jnp.asarray(np.random.default_rng(seed).standard_normal((m, OBS)))
    return obs, _density_queries(obs, _built()[0].blocks)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==========================================================================================
# 1. default OFF -- the goal-conditioned arm is untouched
# ==========================================================================================
def test_index_is_off_by_default_in_the_shipped_config():
    """The knob has to be asked for. Every published density_fb cell is goal-conditioned;
    a default-on index would silently change all of them (and their checkpoint widths)."""
    cfg = load_config("density_fb", "flowbc", "cube_single")
    assert cfg["measure_density_z"] is False
    critic, _params, _t = DensityFBCritic.build(*_ex(), {**cfg, "seed": 0, "batch_size": B})
    assert critic.z_on is False
    assert critic.cvec_dim == OBS, "conditioning width must stay the observation width"
    assert critic.PROVIDES_COND == frozenset({"value_goal", "goal"})


def test_infer_cond_off_is_still_the_placeholder():
    """Unchanged behaviour, and impls/main.py's infer_cond_noop_reason still detects it."""
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
    assert np.array_equal(np.asarray(cond["_cvec_vgoal"]), np.asarray(batch["value_goals"]))


# ==========================================================================================
# 2. the index responds to the reward
# ==========================================================================================
def test_index_is_reward_sensitive():
    """Different reward families -> different index. Without this the "interface" is a
    constant and every downstream number is a goal shortcut."""
    critic, params = _built()
    _obs, u_sc = _pool()
    rng = np.random.default_rng(11)
    rewards = jnp.asarray(rng.standard_normal((5, POOL_M)))
    z = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, rewards))
    assert z.shape == (5, D)
    assert np.isfinite(z).all()
    pair_cos = [abs(_cos(z[i], z[j])) for i in range(5) for j in range(i + 1, 5)]
    assert max(pair_cos) < 0.99, (
        f"five independent reward draws produced near-identical indices (max |cos| = "
        f"{max(pair_cos):.4f}) -- the encoder is not reading the reward")


def test_index_norm_is_sqrt_d():
    """The standing invariant ||z|| = sqrt(d): F's conditioning leg, the actor's leg and the
    sphere rows it is mixed with all assume it, and a z of another length is simply out of
    the conditioning distribution."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(12).standard_normal((4, POOL_M)))
    z = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, r))
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D), atol=1e-10)


# ==========================================================================================
# 3. the invariances -- the reason the CENTERED variant is the one that ships
# ==========================================================================================
@pytest.mark.parametrize("beta", [0.5, -3.0, 100.0])
def test_index_is_invariant_to_an_additive_reward_shift(beta):
    """r -> r + beta leaves the index UNCHANGED. Population-exact: m_c is built from
    r - rbar, and sum_j (r_j - rbar) = 0. Only float round-off separates the two."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(13).standard_normal((3, POOL_M)))
    z0 = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, r + beta))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"an additive reward gauge moved the index by "
        f"{np.abs(z0 - z1).max():.3e} at beta={beta}")


@pytest.mark.parametrize("alpha", [1e-3, 7.0, 1e3])
def test_index_is_invariant_to_a_positive_reward_rescale(alpha):
    """r -> alpha r (alpha > 0) leaves the index UNCHANGED. This is EXACT only because the
    centered index normalizes with `_sphere_or_neutral` (no epsilon in the denominator);
    the shipped `_sphere` guard would move it visibly at small alpha."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(14).standard_normal((3, POOL_M)))
    z0 = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, alpha * r))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"a positive reward rescale moved the index by "
        f"{np.abs(z0 - z1).max():.3e} at alpha={alpha}")


def test_index_is_invariant_to_the_additive_gauge_on_the_backward_tower():
    """B -> B + b leaves the index unchanged: mu_hat absorbs b and C_hat is a covariance.
    The density softmax is exactly invariant to this shift, so the mean of B is pure gauge
    -- and it is large (bridge measurement: ||mu_B|| is 33-103x the centered signal), which
    is what made the UNCENTERED second moment look rank-1.

    The shift is applied to the LAST Dense bias of the backward MLP, which adds a constant
    vector to B(x) for every x -- exactly the gauge transform."""
    critic, params = _built()
    _obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(15).standard_normal((3, POOL_M)))

    b_params = params["density_b"]
    last = max(b_params, key=lambda k: int(k.split("_")[-1]))
    shift = jnp.asarray(np.random.default_rng(16).standard_normal(D) * 25.0)
    shifted = {**b_params, last: {**b_params[last],
                                 "bias": b_params[last]["bias"] + shift}}
    # guard: the shift really does move B (otherwise the invariance is vacuous)
    b0 = np.asarray(critic._B(b_params, u_sc))
    b1 = np.asarray(critic._B(shifted, u_sc))
    assert np.allclose(b1 - b0, np.asarray(shift)[None], atol=1e-8)
    assert np.abs(b1 - b0).max() > 1.0

    z0 = np.asarray(critic.z_from_reward_samples(b_params, u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(shifted, u_sc, r))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"the additive gauge on B moved the index by {np.abs(z0 - z1).max():.3e}")


def test_a_constant_reward_falls_back_to_the_neutral_index():
    """A (near-)constant reward has m_c ~ 0, so ||c|| ~ 0 and normalizing would amplify pure
    numerical noise into a confident-looking task index. The encoder returns 1_d instead --
    reward-free, and still of norm sqrt(d), so it is IN the conditioning distribution."""
    critic, params = _built()
    _obs, u_sc = _pool()
    z = np.asarray(critic.z_from_reward_samples(
        params["density_b"], u_sc, jnp.full((2, POOL_M), 3.5)))
    assert np.allclose(z, np.ones((2, D)), atol=1e-12)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D))


def test_sphere_or_neutral_normalizes_exactly():
    """The helper itself: exact sqrt(d) on a live vector, 1_d on a degenerate one. The
    shipped `_sphere` (epsilon in the denominator) is only approximately scale-invariant,
    which is why the centered index does not use it."""
    x = jnp.asarray([[3.0, 4.0, 0.0, 0.0]])
    out = np.asarray(_sphere_or_neutral(x, 4.0, 1e-6))
    assert np.allclose(np.linalg.norm(out), 2.0, atol=0)
    assert np.allclose(np.asarray(_sphere_or_neutral(1e-9 * x, 4.0, 1e-6)), np.ones((1, 4)))


# ==========================================================================================
# 3b. formula-level pins -- these are the ones a mutation cannot slip past
# ==========================================================================================
def test_centered_moment_matches_a_numpy_reference():
    """m_c = (1/M) sum_j (B(u_j) - mu)(r_j - rbar), transcribed. Pinned against an
    independent numpy reconstruction because the two centerings mask each other in every
    end-to-end invariance check (see this module's header)."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    rng = np.random.default_rng(31)
    r = jnp.asarray(rng.standard_normal((3, 64)))
    mu = jnp.asarray(rng.standard_normal(D))          # deliberately NOT the pool mean

    got = np.asarray(critic._z_centered_moment(params["density_b"], u_sc, mu, r))
    b = np.asarray(critic._B(params["density_b"], u_sc))            # [M, d]
    db = b - np.asarray(mu)[None]
    dr = np.asarray(r) - np.asarray(r).mean(axis=-1, keepdims=True)
    assert np.allclose(got, dr @ db / 64, atol=1e-12)
    # and it is NOT the uncentered moment (so the reference is discriminating)
    assert not np.allclose(got, np.asarray(r) @ db / 64, atol=1e-6)


def test_moments_are_the_centered_covariance_not_the_second_moment():
    """C_hat must be Cov_rho(B(u)), not E[B B^T]. This is the whole spec-30 finding: the
    mean of B is pure gauge and it is huge, which made the UNCENTERED second moment look
    rank-1 (effective rank 1.0002/50) while the centered covariance has effective rank
    6-21/50. A port that kept the second moment would reproduce spec 28's "collapse"."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    mu, cov = critic._z_moments(params["density_b"], u_sc)
    b = np.asarray(critic._B(params["density_b"], u_sc))
    assert np.allclose(np.asarray(mu), b.mean(0), atol=1e-12)
    db = b - b.mean(0)[None]
    assert np.allclose(np.asarray(cov), db.T @ db / 64, atol=1e-12)
    assert not np.allclose(np.asarray(cov), b.T @ b / 64, atol=1e-6), \
        "C_hat is the UNCENTERED second moment -- the exact object spec 30 replaced"
    assert np.allclose(np.asarray(cov), np.asarray(cov).T, atol=0), "C_hat must be symmetric"


def test_reward_centering_holds_at_a_base_mean_that_is_not_the_pool_mean():
    """The load-bearing form of translation invariance: m_c is unchanged by r -> r + beta
    for ANY mu, because sum_j (r_j - rbar) = 0. With the plug-in mu this file otherwise
    uses, sum_j (B(u_j) - mu) is also zero and the beta term dies for the other reason; an
    off-pool mu removes that cover. This is the configuration the shim's EMA-tracked mu_hat
    is permanently in, so it is also the check that a future EMA port must keep passing."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    rng = np.random.default_rng(32)
    r = jnp.asarray(rng.standard_normal((3, 64)))
    mu = jnp.asarray(rng.standard_normal(D) * 10.0)     # NOT the pool mean
    b = np.asarray(critic._B(params["density_b"], u_sc))
    assert np.abs((b - np.asarray(mu)[None]).sum(0)).max() > 1.0, \
        "this mu is too close to the pool mean for the test to discriminate"

    m0 = np.asarray(critic._z_centered_moment(params["density_b"], u_sc, mu, r))
    m1 = np.asarray(critic._z_centered_moment(params["density_b"], u_sc, mu, r + 7.0))
    assert np.allclose(m0, m1, atol=1e-10), (
        f"an additive reward gauge moved m_c by {np.abs(m0 - m1).max():.3e} at an off-pool "
        f"base mean -- the reward-centering (r - rbar) is missing or wrong")


def test_the_ridge_uses_the_mean_eigenvalue_convention_and_floors_the_spectrum():
    """lam = mult * tr(C)/d computed from the UNFLOORED trace, and eigenvalues below
    floor * tr(C)/d lifted to it. Pinned against a numpy reconstruction: both quantities are
    dimensionless multiples of the MEAN eigenvalue, which is what makes them comparable
    across checkpoints and towers (and what the bridge's preflight probes scored)."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    _mu, cov = critic._z_moments(params["density_b"], u_sc)
    m_c = jnp.asarray(np.random.default_rng(33).standard_normal((2, D)))
    got = np.asarray(critic._z_centered_coef(m_c, cov))

    c = np.asarray(cov)
    tr = np.trace(c)
    ev, vec = np.linalg.eigh(0.5 * (c + c.T))
    ev_f = np.maximum(ev, critic.z_cov_eig_floor * tr / D)
    lam = critic.z_ridge_lambda_mult * tr / D
    assert np.allclose(got, ((np.asarray(m_c) @ vec) / (ev_f + lam)[None, :]) @ vec.T,
                       atol=1e-10)
    # a bigger ridge must shrink the coefficient -- the knob is live
    critic_big = _built(measure_density_z_ridge_lambda_mult=10.0)[0]
    assert np.linalg.norm(np.asarray(critic_big._z_centered_coef(m_c, cov))) \
        < np.linalg.norm(got)


# ==========================================================================================
# 4. M = 1 has no meaning for this encoder
# ==========================================================================================
def test_one_reward_sample_raises_instead_of_returning_a_zero_index():
    """m_c = (1/M) sum_j (B(u_j)-mu)(r_j-rbar) is IDENTICALLY ZERO at M=1, so the centered
    encoder has no one-sample/Dirac limit. Returning the neutral fallback silently would be
    the spec-29 ambiguity all over again: a "reward-sample" number that is really a goal
    shortcut. It must be a caller error."""
    critic, params = _built()
    obs, _u = _pool(m=1)
    u_sc = _density_queries(obs, critic.blocks)
    with pytest.raises(ValueError, match="M > 1"):
        critic.z_from_reward_samples(params["density_b"], u_sc, jnp.ones((1,)))


# ==========================================================================================
# 5. the M=256 reward-sample path, end to end through the agent
# ==========================================================================================
def _agent(actor="flowbc", z=True, **over):
    cfg = load_config("density_fb", actor, "cube_single")
    cfg = {**cfg, "batch_size": B, "measure_density_dim": D,
           "value_hidden_dims": [32, 32], "measure_density_z": z, **over}
    ex_obs, ex_act = _ex()
    return Agent.create("density_fb", actor, 0, ex_obs, ex_act, cfg), cfg


def test_infer_eval_cond_runs_the_m256_reward_sample_path():
    """The binding requirement: a task specified by REWARD SAMPLES, no goal anywhere.
    Agent.infer_eval_cond -> critic.infer_cond with a pool of M=256 observations and their
    rewards, and the resulting cond is what sample_actions then rolls on."""
    agent, cfg = _agent()
    assert agent.critic.z_on and agent.critic.cvec_dim == D
    assert "z" in agent.critic.PROVIDES_COND

    obs_pool = jnp.asarray(np.random.default_rng(21).standard_normal((POOL_M, OBS)))
    rewards = jnp.asarray(np.random.default_rng(22).standard_normal((POOL_M,)))
    assert obs_pool.shape[0] == POOL_M == 256

    updated = agent.infer_eval_cond(obs_pool, rewards)
    z = np.asarray(updated.cond_eval["z"])
    assert set(updated.cond_eval) == {"z"}
    assert z.shape == (D,), "a flat [M] reward vector describes ONE task -> a 1-D index"
    assert np.isfinite(z).all()
    assert np.allclose(np.linalg.norm(z), np.sqrt(D), atol=1e-10)
    # it is NOT the eval_cond0 placeholder
    assert not np.allclose(z, np.asarray(agent.cond_eval["z"]))

    # The actor actually rolls on it -- the width check on the EVAL path. The eval loop
    # (impls/utils/evaluation.py) steps one unbatched observation at a time, and it hands
    # the env goal alongside; _cvec must prefer the inferred index over that goal, or the
    # "zero-shot from rewards" claim is a goal rollout wearing a z's name.
    acts = updated.sample_actions(jnp.zeros((OBS,)), goals=jnp.zeros((OBS,)),
                                  seed=jax.random.PRNGKey(3))
    assert np.asarray(acts).shape == (ACT,) and np.isfinite(np.asarray(acts)).all()


def test_the_whole_256_sample_pool_is_read():
    """Perturbing a single reward far into the pool must move the index -- otherwise the
    encoder is quietly using a prefix (or a shortcut) rather than all M samples."""
    agent, _cfg = _agent()
    obs_pool = jnp.asarray(np.random.default_rng(21).standard_normal((POOL_M, OBS)))
    rewards = jnp.asarray(np.random.default_rng(22).standard_normal((POOL_M,)))
    z0 = np.asarray(agent.infer_eval_cond(obs_pool, rewards).cond_eval["z"])
    bumped = rewards.at[POOL_M - 1].add(50.0)
    z1 = np.asarray(agent.infer_eval_cond(obs_pool, bumped).cond_eval["z"])
    assert not np.allclose(z0, z1, atol=1e-6), "the last pool row had no effect on the index"


def test_infer_cond_uses_the_eps_scaled_queries_not_the_raw_block_columns():
    """The standing eps-scaling contract: the B tower's input is always `_density_queries`
    output. Feeding it raw block columns silently encodes a different object -- so the index
    from raw columns must differ from the shipped one."""
    critic, params = _built()
    obs, u_sc = _pool()
    r = jnp.asarray(np.random.default_rng(23).standard_normal((POOL_M,)))
    z_scaled = np.asarray(critic.z_from_reward_samples(params["density_b"], u_sc, r))
    z_raw = np.asarray(critic.z_from_reward_samples(
        params["density_b"], _raw_block_cols(obs, critic.blocks), r))
    assert not np.allclose(z_scaled, z_raw, atol=1e-6)

    net = {"density_b": type("TS", (), {"params": params["density_b"]})()}
    z_infer = np.asarray(critic.infer_cond(net, obs, r)["z"])
    assert np.allclose(z_infer, z_scaled[0], atol=1e-12)


# ==========================================================================================
# 6. _cvec_vgoal is the Z AT THE VALUE GOAL
# ==========================================================================================
def test_cvec_vgoal_is_z_shaped_and_matches_the_actor_width():
    """The bug class this guards: density_fb declares BOOT_COND="value_goal", so every actor
    draws its TD bootstrap action conditioned on cond["_cvec_vgoal"]. With the index on the
    actor net is z-wide (cvec_dim = measure_density_dim), so publishing the raw goal there
    hands a goal-shaped vector to a z-shaped net. Asserted three ways."""
    agent, _cfg = _agent()
    batch = _batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))

    # (a) structural: the published vector has the ACTOR's conditioning width
    assert agent.critic.cvec_dim == D
    assert cond["z"].shape == (B, D)
    assert cond["_cvec_vgoal"].shape == (B, D), (
        f"_cvec_vgoal is {cond['_cvec_vgoal'].shape}, not the actor's (B, cvec_dim)")
    assert cond["_cvec_vgoal"].shape[-1] == agent.critic.cvec_dim
    assert cond["_cvec_vgoal"].shape != batch["value_goals"].shape, \
        "_cvec_vgoal is still the RAW value goal -- the exact bug this test exists for"
    assert np.allclose(np.linalg.norm(np.asarray(cond["_cvec_vgoal"]), axis=-1), np.sqrt(D))

    # (b) it is the index AT THE VALUE GOAL, not at the actor goal (different goal
    #     distributions -> different indices; value_goals != actor_goals in this batch)
    assert not np.allclose(np.asarray(cond["z"]), np.asarray(cond["_cvec_vgoal"]))

    # (c) behavioural: the actor's value-goal draw actually computes, and differs from the
    #     actor-goal draw
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    assert "next_action_vgoal" in boot
    assert not np.allclose(np.asarray(boot["next_action"]),
                           np.asarray(boot["next_action_vgoal"]))

    # (d) negative: the raw value goal is the WRONG width for that net
    with pytest.raises(Exception):
        agent.actor.sample_bootstrap(
            agent.net, batch, {**cond, "_cvec_vgoal": batch["value_goals"]},
            jax.random.PRNGKey(1))


def test_stage_loss_conditions_f_on_the_value_goal_index():
    """F's conditioning leg must be the SAME vector the bootstrap action was drawn at.
    Perturbing _cvec_vgoal moves density_ce; perturbing the raw cond["value_goal"] (which
    the index path no longer reads) does not."""
    agent, _cfg = _agent()
    batch = _batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))

    def ce(c):
        return float(np.asarray(agent.compute_static(batch, c, boot)["density_ce"]))

    base = ce(cond)
    assert ce({**cond, "_cvec_vgoal": jnp.zeros_like(cond["_cvec_vgoal"])}) != base
    assert ce({**cond, "value_goal": jnp.zeros_like(cond["value_goal"])}) == base


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc", "fql"])
def test_a_real_update_runs_with_the_index_on(actor):
    """Everything above is static; this takes a gradient step through the whole composed
    agent, which is the only thing that exercises every width at once."""
    agent, _cfg = _agent(actor)
    updated, info = agent.update(_batch())
    assert np.isfinite(float(np.asarray(info["density_ce"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    moved = [not np.array_equal(np.asarray(a), np.asarray(b))
             for a, b in zip(jax.tree_util.tree_leaves(agent.net["density_f"].params),
                             jax.tree_util.tree_leaves(updated.net["density_f"].params))]
    assert any(moved), "density_f params did not move"


def test_the_index_never_carries_a_gradient_into_the_backward_tower():
    """The index is stop-gradded everywhere (it is built from the backward tower it would
    otherwise reshape). A gradient of the index w.r.t. density_b must be exactly zero."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(24).standard_normal((2, 64)))

    def scalar(bp):
        return jnp.sum(critic.z_from_reward_samples(bp, u_sc, r))

    g = jax.grad(scalar)(params["density_b"])
    assert all(np.all(np.asarray(l) == 0.0) for l in jax.tree_util.tree_leaves(g))


def test_the_training_index_mixes_in_sphere_draws():
    """FB's standard task-prior mix (measure_density_z_rand_frac): a fraction of the rows
    carry a fresh sqrt(d)-sphere draw rather than the encoded index, so the actor sees the
    whole conditioning sphere. At frac=0 every row is the encoder's; at frac=1 none is."""
    batch = _batch()
    a0, _ = _agent(z=True, measure_density_z_rand_frac=0.0)
    a1, _ = _agent(z=True, measure_density_z_rand_frac=1.0)
    c0 = a0.critic.sample_cond(a0.net, batch, jax.random.PRNGKey(5))
    c1 = a1.critic.sample_cond(a1.net, batch, jax.random.PRNGKey(5))
    assert not np.allclose(np.asarray(c0["z"]), np.asarray(c1["z"]))
    # both branches carry ||z|| = sqrt(d)
    for c in (c0, c1):
        assert np.allclose(np.linalg.norm(np.asarray(c["z"]), axis=-1), np.sqrt(D),
                           atol=1e-6)
    # frac=0 rows are exactly the encoder's own output
    u_raw = _raw_block_cols(batch["next_observations"], a0.critic.blocks)
    u_sc = _density_queries(batch["next_observations"], a0.critic.blocks)
    r = _delta_phi(batch["actor_goals"], u_raw, a0.critic.blocks)
    z_enc = np.asarray(a0.critic.z_from_reward_samples(
        a0.net["density_b"].params, u_sc, r))
    assert np.allclose(np.asarray(c0["z"]), z_enc, atol=1e-10)
