"""density_fb's REWARD-SAMPLE INDEX (spec 30 P1, the centered dual encoder).

`infer_z` (default OFF) replaces the critic's goal conditioning with an index z
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

     ⚠ HISTORICAL NOTE ON THESE TWO TESTS. An earlier revision of this port computed
     mu_hat/C_hat as PLUG-IN moments of the very pool the reward samples live on. That makes
     sum_j (B(u_j) - mu_hat) exactly zero, so the reward-centering and the B-centering each
     killed BOTH gauges alone and MASKED each other -- deleting either one left the
     end-to-end invariance tests green (found by mutation). The moments are now genuinely
     PERSISTENT (EMA leaves in `net`, section 7), so mu_hat is no longer the pool's own mean
     and the masking is gone. The formula-level tests in section 3b are kept anyway: they
     are what caught it, and they discriminate regardless of where the moments come from.
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
    DensityFBCritic, _density_queries, _raw_block_cols, _delta_phi, _scale_block_cols,
    _sphere_or_neutral,
)
from impls.utils.config_utils import load_config

B, OBS, ACT, D = 16, 40, 5, 8
POOL_M = 256          # the M the spec-30 eval pool is pinned at


def _cfg(z=True, **over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, tau=0.005, seed=0,
        value_hidden_dims=[32, 32], layer_norm=True,
        goal_spec="19:22:0.4", z_dim=D, temperature=1.0,
        infer_z=z,
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
    """(critic, params) from build() -- enough for every pure-index assertion. `params`
    carries `density_b_zenc` and `density_zstats` when the index is on."""
    ex_obs, ex_act = _ex()
    over.setdefault("eval_pool_size", 8)   # B=16 batches in this file
    critic, params, _targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(z, **over))
    return critic, params


class _FakeTS:
    """Minimal stand-in for a TrainState: the index reads only `.params`."""

    def __init__(self, params):
        self.params = params


def _net_from(params):
    """A `net`-shaped dict over raw param trees, for tests that do not need a full Agent."""
    return {k: _FakeTS(v) for k, v in params.items()}


def _stats(params, **over):
    """The stored stats leaves, optionally overridden -- used to drive the index at moments
    that are NOT any pool's own statistics (which is the whole point of persisting them)."""
    return {**params["density_zstats"], **over}


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
    assert cfg["infer_z"] is False
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
    z = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, rewards))
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
    z = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
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
    z0 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r + beta))
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
    z0 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z1 = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, alpha * r))
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

    b_params = params["density_b_zenc"]
    last = max(b_params, key=lambda k: int(k.split("_")[-1]))
    shift = jnp.asarray(np.random.default_rng(16).standard_normal(D) * 25.0)
    shifted = {**b_params, last: {**b_params[last],
                                 "bias": b_params[last]["bias"] + shift}}
    # guard: the shift really does move B (otherwise the invariance is vacuous)
    b0 = np.asarray(critic._B(b_params, u_sc))
    b1 = np.asarray(critic._B(shifted, u_sc))
    assert np.allclose(b1 - b0, np.asarray(shift)[None], atol=1e-8)
    assert np.abs(b1 - b0).max() > 1.0

    # B -> B + b shifts mu_hat by the same b (that is what the EMA would converge to), so
    # the honest test moves BOTH together. Driving the stored mu from a real pool rather
    # than leaving it at its zero init is what makes this non-vacuous.
    mu = jnp.mean(critic._B(b_params, u_sc), axis=0)
    st0 = _stats(params, mu=mu)
    st1 = _stats(params, mu=mu + shift)
    z0 = np.asarray(critic._z_index(b_params, st0, u_sc, r))
    z1 = np.asarray(critic._z_index(shifted, st1, u_sc, r))
    assert np.allclose(z0, z1, atol=1e-8), (
        f"the additive gauge on B moved the index by {np.abs(z0 - z1).max():.3e}")


def test_a_constant_reward_falls_back_to_the_neutral_index():
    """A (near-)constant reward has m_c ~ 0, so ||c|| ~ 0 and normalizing would amplify pure
    numerical noise into a confident-looking task index. The encoder returns 1_d instead --
    reward-free, and still of norm sqrt(d), so it is IN the conditioning distribution."""
    critic, params = _built()
    _obs, u_sc = _pool()
    z = np.asarray(critic.z_from_reward_samples(
        _net_from(params), u_sc, jnp.full((2, POOL_M), 3.5)))
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

    got = np.asarray(critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r))
    b = np.asarray(critic._B(params["density_b_zenc"], u_sc))       # [M, d]
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
    agent, _cfg_ = _agent()
    batch = _batch()
    st_before = agent.net["density_zstats"].params
    updated = agent.update(batch)[0]
    st = updated.net["density_zstats"].params

    # the EMA target for this step, computed independently
    u_sc = _density_queries(batch["next_observations"], agent.critic.blocks)
    b = np.asarray(agent.critic._B(agent.net["density_b_zenc"].params, u_sc))
    mu_b, tau = b.mean(0), agent.critic.z_sigma_tau
    db = b - mu_b[None]
    cov_b = db.T @ db / b.shape[0]
    assert np.allclose(np.asarray(st["mu"]),
                       np.asarray(st_before["mu"]) * (1 - tau) + mu_b * tau, atol=1e-10)
    assert np.allclose(np.asarray(st["cov"]),
                       np.asarray(st_before["cov"]) * (1 - tau) + cov_b * tau, atol=1e-10)
    # CENTERED, not the uncentered second moment spec 30 replaced
    cov_unc = np.asarray(st_before["cov"]) * (1 - tau) + (b.T @ b / b.shape[0]) * tau
    assert not np.allclose(np.asarray(st["cov"]), cov_unc, atol=1e-6), \
        "C_hat is the UNCENTERED second moment -- the exact object spec 30 replaced"
    assert np.allclose(np.asarray(st["cov"]), np.asarray(st["cov"]).T, atol=0), \
        "C_hat must be symmetric"


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
    b = np.asarray(critic._B(params["density_b_zenc"], u_sc))
    assert np.abs((b - np.asarray(mu)[None]).sum(0)).max() > 1.0, \
        "this mu is too close to the pool mean for the test to discriminate"

    m0 = np.asarray(critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r))
    m1 = np.asarray(critic._z_centered_moment(params["density_b_zenc"], u_sc, mu, r + 7.0))
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
    b = critic._B(params["density_b_zenc"], u_sc)
    dbv = b - jnp.mean(b, axis=0)[None]
    cov = dbv.T @ dbv / b.shape[0]
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
    critic_big = _built(ridge_lambda=10.0)[0]
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
        critic.z_from_reward_samples(_net_from(params), u_sc, jnp.ones((1,)))


# ==========================================================================================
# 5. the M=256 reward-sample path, end to end through the agent
# ==========================================================================================
def _agent(actor="flowbc", z=True, **over):
    cfg = load_config("density_fb", actor, "cube_single")
    cfg = {**cfg, "batch_size": B, "z_dim": D,
           "value_hidden_dims": [32, 32], "infer_z": z,
           "eval_pool_size": 8, **over}
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
    # (impls/utils/evaluation.py) steps one unbatched observation at a time. goals=None is
    # the "roll the inferred index" entrypoint (the bridge's sample_actions_with_z); passing
    # a goal instead routes through map_eval_cond, which is section 8's business.
    acts = updated.sample_actions(jnp.zeros((OBS,)), seed=jax.random.PRNGKey(3))
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
    z_scaled = np.asarray(critic.z_from_reward_samples(_net_from(params), u_sc, r))
    z_raw = np.asarray(critic.z_from_reward_samples(
        _net_from(params), _raw_block_cols(obs, critic.blocks), r))
    assert not np.allclose(z_scaled, z_raw, atol=1e-6)

    z_infer = np.asarray(critic.infer_cond(_net_from(params), obs, r)["z"])
    assert np.allclose(z_infer, z_scaled[0], atol=1e-12)


# ==========================================================================================
# 6. _cvec_vgoal is the Z AT THE VALUE GOAL
# ==========================================================================================
def test_cvec_vgoal_is_z_shaped_and_matches_the_actor_width():
    """The bug class this guards: density_fb declares BOOT_COND="value_goal", so every actor
    draws its TD bootstrap action conditioned on cond["_cvec_vgoal"]. With the index on the
    actor net is z-wide (cvec_dim = z_dim), so publishing the raw goal there
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


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc"])
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
        return jnp.sum(critic._z_index(bp, _stats(params), u_sc, r))

    g = jax.grad(scalar)(params["density_b_zenc"])
    assert all(np.all(np.asarray(l) == 0.0) for l in jax.tree_util.tree_leaves(g))


def test_the_training_index_mixes_in_sphere_draws():
    """FB's standard task-prior mix (z_random_frac): a fraction of the rows
    carry a fresh sqrt(d)-sphere draw rather than the encoded index, so the actor sees the
    whole conditioning sphere. At frac=0 every row is the encoder's; at frac=1 none is."""
    batch = _batch()
    a0, _ = _agent(z=True, z_random_frac=0.0)
    a1, _ = _agent(z=True, z_random_frac=1.0)
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
    z_enc = np.asarray(a0.critic.z_from_reward_samples(a0.net, u_sc, r))
    assert np.allclose(np.asarray(c0["z"]), z_enc, atol=1e-10)

# ==========================================================================================
# 7. the index state is PERSISTENT -- EMA leaves, not batch statistics
# ==========================================================================================
def test_the_index_state_groups_exist_only_when_the_index_is_on():
    """Off, the param tree is exactly the two towers -- so the goal-conditioned arm's
    checkpoints and byte-identity are untouched by any of this."""
    off, _ = _agent(z=False)
    on, _ = _agent(z=True)
    assert set(off.net) - {"actor", "actor_vf"} == {"density_f", "density_b"}
    assert {"density_b_zenc", "density_zstats"} <= set(on.net)
    assert off.critic.GROUPS == ("density_f", "density_b")
    assert on.critic.GROUPS == ("density_f", "density_b", "density_b_zenc", "density_zstats")


def test_the_index_tower_starts_as_a_copy_of_density_b_and_then_lags_it():
    """density_b_zenc inits EQUAL to density_b (so step 0's index is the one density_b
    itself implies) and is then mixed at tau_z, which is what makes it quasi-stationary
    while density_b trains away. It is NOT the online tower and NOT the polyak target."""
    agent, _c = _agent()
    b0 = agent.net["density_b"].params
    z0 = agent.net["density_b_zenc"].params
    for a, b in zip(jax.tree_util.tree_leaves(b0), jax.tree_util.tree_leaves(z0)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "zenc must init as a copy"

    updated = agent.update(_batch())[0]
    tau = agent.critic.z_enc_tau
    expect = jax.tree_util.tree_map(lambda p, ep: p * tau + ep * (1 - tau), b0, z0)
    for got, exp in zip(jax.tree_util.tree_leaves(updated.net["density_b_zenc"].params),
                        jax.tree_util.tree_leaves(expect)):
        assert np.allclose(np.asarray(got), np.asarray(exp), atol=1e-12)

    # and it now LAGS the online tower, which moved under its gradient
    on_new = updated.net["density_b"].params
    zn_new = updated.net["density_b_zenc"].params
    assert any(not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
               for a, b in zip(jax.tree_util.tree_leaves(on_new),
                               jax.tree_util.tree_leaves(zn_new))), \
        "the index tower equals the online tower -- it is not lagging anything"

    # ... and it is not the polyak TARGET tower either (different rate, different object)
    assert any(not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
               for a, b in zip(jax.tree_util.tree_leaves(updated.target["density_b"]),
                               jax.tree_util.tree_leaves(zn_new)))


def test_the_index_reads_the_slow_tower_not_the_online_or_target_one():
    """WHICH tower encodes the index. Found by mutation: swapping `_zenc` to return the
    ONLINE density_b left the whole suite green, because every other test either drives the
    math with an explicitly-named tower or compares two paths that mutate together -- and at
    build time all three towers are identical copies, so nothing separated them.

    The three towers must have DIVERGED for this to mean anything, so run a few updates
    first, then assert the index tracks density_b_zenc specifically. The shipped centered
    arm asserts slow_backward_tau > 0, which is exactly the condition selecting the
    slow copy over target_density_b (bridge `_density_z_enc_module`)."""
    agent, _c = _agent()
    for seed in (1, 2, 3):
        agent = agent.update(_batch(seed=seed))[0]
    critic, net = agent.critic, agent.net
    st = net["density_zstats"].params
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(61).standard_normal((3, 64)))

    z_slow = np.asarray(critic._z_index(net["density_b_zenc"].params, st, u_sc, r))
    z_online = np.asarray(critic._z_index(net["density_b"].params, st, u_sc, r))
    z_target = np.asarray(critic._z_index(agent.target["density_b"], st, u_sc, r))
    # guard: the three towers really are distinct objects by now
    assert not np.allclose(z_slow, z_online, atol=1e-6), \
        "online and slow towers still agree -- run more updates or the test is vacuous"
    assert not np.allclose(z_slow, z_target, atol=1e-6), \
        "target and slow towers still agree -- run more updates or the test is vacuous"

    got = np.asarray(critic.z_from_reward_samples(net, u_sc, r))
    assert np.allclose(got, z_slow, atol=0), "the index does not read density_b_zenc"
    assert not np.allclose(got, z_online, atol=1e-6), "the index reads the ONLINE tower"
    assert not np.allclose(got, z_target, atol=1e-6), "the index reads the TARGET tower"

    # the same tower must serve the native-goal eval path
    mapped = critic.map_eval_cond(net, {"goal": jnp.zeros((2, OBS))})
    pool_sc = _scale_block_cols(st["pool"], critic.blocks)
    kg = _delta_phi(jnp.zeros((2, OBS)), st["pool"], critic.blocks)
    assert np.allclose(np.asarray(mapped["z"]),
                       np.asarray(critic._z_index(net["density_b_zenc"].params, st,
                                                  pool_sc, kg)), atol=0)


def test_the_moments_are_stored_and_not_recomputed_from_the_batch():
    """THE load-bearing property, and the one the bridge states outright: batch statistics
    are forbidden. Driving two updates with DIFFERENT batches must leave the index a
    function of the accumulated EMA, not of whichever batch was last -- so the stored
    moments must differ from the last batch's own statistics."""
    agent, _c = _agent()
    a1 = agent.update(_batch(seed=1))[0]
    a2 = a1.update(_batch(seed=2))[0]
    st = a2.net["density_zstats"].params

    # the second batch's own plug-in moments, at the tower the index actually used
    b2 = _batch(seed=2)
    u_sc = _density_queries(b2["next_observations"], agent.critic.blocks)
    b = np.asarray(agent.critic._B(a1.net["density_b_zenc"].params, u_sc))
    mu_batch, cov_batch = b.mean(0), (b - b.mean(0)).T @ (b - b.mean(0)) / b.shape[0]

    assert not np.allclose(np.asarray(st["mu"]), mu_batch, atol=1e-6), \
        "mu_hat equals the last batch's mean -- these are batch statistics, not an EMA"
    assert not np.allclose(np.asarray(st["cov"]), cov_batch, atol=1e-6), \
        "C_hat equals the last batch's covariance -- these are batch statistics, not an EMA"
    # they HAVE moved off their init (mu=0, C=I), i.e. the EMA is running
    assert np.abs(np.asarray(st["mu"])).max() > 0
    assert not np.allclose(np.asarray(st["cov"]), np.eye(D), atol=1e-9)


def test_the_stored_moments_change_the_index():
    """If the index came out the same whatever the stored moments are, persisting them
    would be theatre. Same tower, same reward samples, different stored C_hat -> different
    index."""
    critic, params = _built()
    _obs, u_sc = _pool(m=64)
    r = jnp.asarray(np.random.default_rng(41).standard_normal((2, 64)))
    z_init = np.asarray(critic._z_index(params["density_b_zenc"], _stats(params), u_sc, r))
    b = critic._B(params["density_b_zenc"], u_sc)
    dbv = b - jnp.mean(b, axis=0)[None]
    z_real = np.asarray(critic._z_index(
        params["density_b_zenc"],
        _stats(params, mu=jnp.mean(b, axis=0), cov=dbv.T @ dbv / b.shape[0]), u_sc, r))
    assert not np.allclose(z_init, z_real, atol=1e-6), \
        "the stored moments do not reach the index -- the EMA is not load-bearing"


def test_the_index_state_survives_a_serialization_round_trip():
    """The bridge's first requirement for these leaves: they SERIALIZE, so every eval
    harness reads the identical object training conditioned on. save_agent pickles the whole
    agent, so this is the property that makes train and eval agree across a checkpoint."""
    import flax
    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    blob = flax.serialization.to_state_dict(trained)
    fresh, _c2 = _agent()
    restored = flax.serialization.from_state_dict(fresh, blob)

    for k in ("mu", "cov", "pool", "filled"):
        assert np.allclose(np.asarray(restored.net["density_zstats"].params[k]),
                           np.asarray(trained.net["density_zstats"].params[k]), atol=0), k
    for a, b in zip(jax.tree_util.tree_leaves(restored.net["density_b_zenc"].params),
                    jax.tree_util.tree_leaves(trained.net["density_b_zenc"].params)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_index_state_takes_no_gradient_and_no_optimizer_step():
    """Neither state group is in stage_groups or the actor's GROUPS, so update_stats is
    their sole writer. If an optimizer were stepping them the EMA would be fighting adam."""
    agent, _c = _agent()
    assert agent.critic.stage_groups("density") == ("density_f", "density_b")
    assert "density_b_zenc" not in agent.critic.TARGETS
    assert "density_zstats" not in agent.critic.TARGETS
    # step counters: the two towers advance, the state groups do not
    updated = agent.update(_batch())[0]
    assert int(updated.net["density_f"].step) > int(agent.net["density_f"].step)
    assert int(updated.net["density_zstats"].step) == int(agent.net["density_zstats"].step)
    assert int(updated.net["density_b_zenc"].step) == int(agent.net["density_b_zenc"].step)


def test_update_stats_is_a_noop_for_every_other_critic():
    """The Agent hook is additive: only density_fb (slow tower, feature moments, pool) and
    hilp (running mean of phi) define update_stats, and density_fb's returns the net
    unchanged on the goal-conditioned arm."""
    from impls.agents import CRITICS
    defining = {n for n, c in CRITICS.items() if hasattr(c, "update_stats")}
    assert defining == {"density_fb", "hilp"}
    off, _c = _agent(z=False)
    same = off.critic.update_stats(off.net, off.net, _batch())
    assert same is off.net


# ==========================================================================================
# 8. the FROZEN M=256 eval pool and the native-goal reward-sample API
# ==========================================================================================
def test_the_pool_is_frozen_on_the_first_update_and_never_rewritten():
    """A fixed draw from rho_phi, so the eval index is a stable function of the params rather
    than a statistic of whichever batch happened to be last."""
    agent, _c = _agent()
    st0 = agent.net["density_zstats"].params
    assert float(st0["filled"]) == 0.0
    assert np.allclose(np.asarray(st0["pool"]), 0.0), "pool inits degenerate"

    b1 = _batch(seed=1)
    a1 = agent.update(b1)[0]
    st1 = a1.net["density_zstats"].params
    m = agent.critic.z_eval_pool_m
    assert float(st1["filled"]) == 1.0
    expect = np.asarray(_raw_block_cols(b1["next_observations"], agent.critic.blocks))[:m]
    assert np.allclose(np.asarray(st1["pool"]), expect, atol=0)

    a2 = a1.update(_batch(seed=2))[0]
    assert np.allclose(np.asarray(a2.net["density_zstats"].params["pool"]),
                       np.asarray(st1["pool"]), atol=0), "the pool was rewritten"


def test_map_eval_cond_routes_a_raw_goal_through_the_reward_sample_api():
    """Native-goal eval goes through the REWARD-SAMPLE path: goal -> r = kg(goal, pool) over
    the frozen M-row pool -> the identical centered solve. No point/Dirac shortcut exists in
    this port, which is the whole reason the pool is here."""
    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    goal = jnp.asarray(np.random.default_rng(51).standard_normal((3, OBS)))
    mapped = trained.critic.map_eval_cond(trained.net, {"goal": goal})
    z = np.asarray(mapped["z"])
    assert z.shape == (3, D)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(D), atol=1e-10)

    # it is EXACTLY the reward-sample solve on kg over the pool
    st = trained.net["density_zstats"].params
    kg = _delta_phi(goal, st["pool"], trained.critic.blocks)
    u_sc = _scale_block_cols(st["pool"], trained.critic.blocks)
    assert np.allclose(z, np.asarray(trained.critic.z_from_reward_samples(
        trained.net, u_sc, kg)), atol=1e-12)

    # different goals -> different indices
    assert max(abs(_cos(z[i], z[j])) for i in range(3) for j in range(i + 1, 3)) < 0.99


def test_map_eval_cond_handles_the_unbatched_eval_goal():
    """impls/utils/evaluation.py steps one unbatched observation and goal at a time."""
    agent, _c = _agent()
    trained = agent.update(_batch())[0]
    g = jnp.asarray(np.random.default_rng(52).standard_normal(OBS))
    z = np.asarray(trained.critic.map_eval_cond(trained.net, {"goal": g})["z"])
    assert z.shape == (D,)
    acts = trained.sample_actions(jnp.zeros((OBS,)), goals=g, seed=jax.random.PRNGKey(1))
    assert np.asarray(acts).shape == (ACT,) and np.isfinite(np.asarray(acts)).all()


def test_the_pool_uses_the_raw_scaled_coordinate_conventions_correctly():
    """The double-scale trap: the pool stores RAW block coords because _delta_phi needs raw
    rows, while the B tower needs the eps-scaled twin. Crossing them silently encodes a
    different object."""
    agent, _c = _agent()
    b1 = _batch(seed=1)
    trained = agent.update(b1)[0]
    st = trained.net["density_zstats"].params
    blocks = trained.critic.blocks
    raw = np.asarray(_raw_block_cols(b1["next_observations"], blocks))[:agent.critic.z_eval_pool_m]
    assert np.allclose(np.asarray(st["pool"]), raw, atol=0), "pool must store RAW coords"
    scaled = np.asarray(_scale_block_cols(st["pool"], blocks))
    assert not np.allclose(scaled, np.asarray(st["pool"]), atol=1e-3)
    assert np.allclose(
        scaled,
        np.asarray(_density_queries(b1["next_observations"], blocks))[:agent.critic.z_eval_pool_m],
        atol=1e-10)


def test_the_real_m256_pool_runs_end_to_end():
    """The pinned M. Everything else in this file runs a small pool so the batches stay
    cheap; this one uses the shipped eval_pool_size = 256 with a batch big
    enough to draw it, and drives the native-goal reward-sample API through it."""
    agent, cfg = _agent(eval_pool_size=256, batch_size=POOL_M)
    assert agent.critic.z_eval_pool_m == 256
    trained = agent.update(_batch(b=POOL_M))[0]
    st = trained.net["density_zstats"].params
    assert np.asarray(st["pool"]).shape[0] == 256
    assert float(st["filled"]) == 1.0
    z = np.asarray(trained.critic.map_eval_cond(
        trained.net, {"goal": jnp.zeros((1, OBS))})["z"])
    assert z.shape == (1, D) and np.isfinite(z).all()


def test_a_pool_larger_than_the_batch_is_refused():
    """The pool is drawn from one batch's next states, so it cannot exceed batch_size. The
    bridge asserts this; a silently-shrunk pool would change the eval M without saying so."""
    with pytest.raises(AssertionError, match="exceeds batch_size"):
        DensityFBCritic.build(*_ex(), _cfg(z=True, eval_pool_size=1024))


def test_the_centered_encoder_refuses_a_fast_index_tower():
    """slow_backward_tau must be > 0: at 0 the bridge's _density_z_enc_module selects
    target_density_b, and running the solve through the fast-drifting target tower is the
    known non-stationarity risk the contract exists to avoid."""
    with pytest.raises(AssertionError, match="slow_backward_tau"):
        DensityFBCritic.build(*_ex(), _cfg(z=True, slow_backward_tau=0.0,
                                           eval_pool_size=8))


# ==========================================================================================
# 9. the RATES and the GUARD -- found by mutation (audit 2026-08-23)
#
# Section 7 pins WHICH tower the index reads and THAT the moments are stored rather than
# recomputed. It does not pin the RATES at which either is mixed, nor the covariance
# eigenvalue floor. Three mutations survived the whole suite:
#
#   * slow_backward_tau -> 1.0 (the index tower stops being a slow copy);
#   * the moments EMA'd at the JUST-WRITTEN index tower instead of the pre-step one;
#   * the eigenvalue floor deleted from the ridge solve.
#
# The first two survived for the SAME structural reason as the earlier `_zenc` mutation:
# density_b_zenc inits as an exact copy of density_b, the existing tests drive exactly ONE
# update, and after one update from coincident towers the two rules agree to 3e-8 -- which
# `np.allclose(..., atol=1e-12)` accepts, because its DEFAULT rtol=1e-5 dominates the
# comparison. Every check below therefore (a) runs several updates, (b) compares with
# rtol=0, and (c) carries an explicit vacuity guard asserting the towers have separated.
# ==========================================================================================
def _leafmax(a, b):
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max())
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_the_index_tower_ema_rate_is_pinned_across_several_updates():
    """tau_z is what makes the index tower QUASI-STATIONARY -- the property the whole
    persisted-moments design rests on. A one-update check cannot see it: zenc inits equal to
    density_b, so at step 1 `p*tau + ep*(1-tau)` and the tau=1 rule return the same array.
    Run three updates, compare with rtol=0, and guard that the towers actually separated."""
    agent, _c = _agent()
    tau = agent.critic.z_enc_tau
    assert 0.0 < tau < 1.0, "the centered arm requires a slow index tower"

    a = agent
    for seed in (1, 2, 3):
        pre_b = a.net["density_b"].params            # PRE-step online tower: the EMA source
        pre_z = a.net["density_b_zenc"].params
        a = a.update(_batch(seed=seed))[0]
        expect = jax.tree_util.tree_map(lambda p, ep: p * tau + ep * (1 - tau), pre_b, pre_z)
        # atol only -- the module runs with x64 on, so a float32 leaf recombined through a
        # python-float tau carries ~1e-14 of promotion noise. rtol is pinned to 0 on purpose:
        # numpy's DEFAULT rtol=1e-5 is what let the tau_z=1.0 mutant through the one-update
        # check in section 7 (it differs by only 3e-8 after a single step).
        assert _leafmax(a.net["density_b_zenc"].params, expect) < 1e-12, (
            "the index tower is not the tau_z EMA of the PRE-step online tower")

    # vacuity guard: without this the assertions above hold for a tower that never moved
    assert _leafmax(a.net["density_b"].params, a.net["density_b_zenc"].params) > 1e-6, \
        "online and index towers never separated -- the rate above is untested"


def test_a_different_index_tower_rate_produces_a_different_tower():
    """The rate is read from the CONFIG, not hard-coded: two agents identical but for
    slow_backward_tau must disagree after the same updates. A build that ignored the
    knob (or pinned it to 1.0) passes every other test in this file."""
    # both inside the (0, 0.1] band the build asserts for the centered arm
    slow, _c = _agent(slow_backward_tau=1e-4)
    fast, _c = _agent(slow_backward_tau=0.05)
    for seed in (1, 2, 3):
        b = _batch(seed=seed)
        slow, fast = slow.update(b)[0], fast.update(b)[0]
    zgap = _leafmax(slow.net["density_b_zenc"].params, fast.net["density_b_zenc"].params)
    bgap = _leafmax(slow.net["density_b"].params, fast.net["density_b"].params)
    assert zgap > 1e-6, "slow_backward_tau does not reach the index tower"
    # The online towers drift apart a little too -- the index conditions the actor, so the
    # rate feeds back into training. The effect must still be OVERWHELMINGLY on the tower
    # the knob names, or the assertion above could be satisfied by that feedback alone.
    assert zgap > 100 * bgap, \
        f"index-tower gap {zgap:.3e} is not dominated by the knob (online gap {bgap:.3e})"


def test_the_moments_ema_reads_the_pre_step_index_tower():
    """Order inside update_stats: the index tower is mixed FIRST, then the moments are EMA'd
    over this batch's query set at the tower as it was BEFORE that mix (bridge ~3903-3921).
    Using the just-written tower instead is a one-step lookahead that survives every
    existing check, because at step 1 the two towers coincide."""
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
        t = a.critic.z_sigma_tau
        assert np.allclose(np.asarray(st["mu"]),
                           np.asarray(st_before["mu"]) * (1 - t) + mu_b * t,
                           rtol=0, atol=1e-9), "mu_hat is not EMA'd at the PRE-step tower"
        assert np.allclose(np.asarray(st["cov"]),
                           np.asarray(st_before["cov"]) * (1 - t) + cov_b * t,
                           rtol=0, atol=1e-9), "C_hat is not EMA'd at the PRE-step tower"

    # vacuity guard: the pre- and post-step index towers must differ by now, or the
    # distinction the assertions draw does not exist on this trajectory
    pre = a.net["density_b_zenc"].params
    post = jax.tree_util.tree_map(
        lambda p, ep: p * a.critic.z_enc_tau + ep * (1 - a.critic.z_enc_tau),
        a.net["density_b"].params, pre)
    assert _leafmax(pre, post) > 1e-9, \
        "pre- and post-step index towers coincide -- the ordering above is untested"


def test_the_covariance_eigenvalue_floor_is_exercised_and_load_bearing():
    """The floor exists because the centered covariance of the d-dim tower genuinely has
    near-zero directions (bridge: effective rank 6-21 of 50). Every other test drives the
    solve with a FULL-RANK sample covariance whose smallest eigenvalue sits ~4.5 orders of
    magnitude ABOVE floor*tr(C)/d, so deleting the floor changes nothing they can see.
    Drive it with a deliberately rank-deficient C instead."""
    critic, _params = _built()
    rng = np.random.default_rng(77)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    cov_np = Q @ np.diag(np.array([1.0, 0.5] + [0.0] * (D - 2))) @ Q.T
    cov = jnp.asarray(cov_np)
    m_c = jnp.asarray(rng.standard_normal((2, D)))

    tr = float(np.trace(cov_np))
    ev, vec = np.linalg.eigh(0.5 * (cov_np + cov_np.T))
    floor = critic.z_cov_eig_floor * tr / D
    assert (ev < floor).sum() >= D - 2, \
        "this covariance does not exercise the floor -- the test would be vacuous"

    lam = critic.z_ridge_lambda_mult * tr / D
    got = np.asarray(critic._z_centered_coef(m_c, cov))
    floored = ((np.asarray(m_c) @ vec) / (np.maximum(ev, floor) + lam)[None, :]) @ vec.T
    unfloored = ((np.asarray(m_c) @ vec) / (ev + lam)[None, :]) @ vec.T
    assert np.allclose(got, floored, rtol=0, atol=1e-8), "the spectrum is not floored"
    assert not np.allclose(floored, unfloored, atol=1e-6), \
        "floored and unfloored solves agree -- the assertion above cannot discriminate"


# ==========================================================================================
# 10. the actor's reward on the reward-inference arm: (B_slow(u) - mu)^T z / sqrt(d)
# ==========================================================================================
def _built_full(**over):
    ex_obs, ex_act = _ex()
    over.setdefault("eval_pool_size", 8)
    return DensityFBCritic.build(ex_obs, ex_act, _cfg(True, **over))


def _perturbed(params, seed=3):
    """At build time B_slow equals B and mu is 0. Move both the way training would, so the
    tests below see a centered slow basis distinct from the online tower."""
    rng = np.random.default_rng(seed)
    p = dict(params)
    p["density_b_zenc"] = jax.tree_util.tree_map(
        lambda x: x * 1.5 + 0.1 * jnp.asarray(rng.standard_normal(x.shape)), params["density_b_zenc"])
    st = dict(params["density_zstats"])
    st["mu"] = jnp.asarray(rng.standard_normal(np.asarray(st["mu"]).shape))
    p["density_zstats"] = st
    return p


def test_sample_cond_publishes_the_centered_slow_basis():
    critic, params, _t = _built_full()
    params = _perturbed(params)
    cond = critic.sample_cond(_net_from(params), _batch(), jax.random.PRNGKey(0))
    expect = np.asarray(critic._B(params["density_b_zenc"], cond["_density_u_sc"])) \
        - np.asarray(params["density_zstats"]["mu"])[None]
    np.testing.assert_allclose(np.asarray(cond["_density_zenc_u"]), expect, rtol=1e-5, atol=1e-6)


def test_q_fn_averages_the_linear_z_reward_under_the_predicted_distribution():
    """q = min over heads of sum_j w(s,a,z,u_j) (B_slow(u_j) - mu)^T z / sqrt(d)."""
    obs, act = _ex()
    critic, params, _t = _built_full()
    params = _perturbed(params)
    cond = critic.sample_cond(_net_from(params), _batch(), jax.random.PRNGKey(0))
    q, qs = critic.q_fn(params, obs, cond, act)
    z = cond["z"]
    f = critic._F(params["density_f"], obs, act, z)
    b = critic._B(params["density_b"], cond["_density_u_sc"])
    w = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', f, b) / critic.temp, -1))
    r = np.asarray(cond["_density_zenc_u"]) @ np.asarray(z).T / np.sqrt(D)    # [M, B]
    heads = (w * r.T[None]).sum(-1)
    np.testing.assert_allclose(np.asarray(q), heads.min(0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(qs[0]), heads.min(0) - heads.min(0).mean(),
                               rtol=1e-6, atol=1e-8)


def test_the_index_requires_hit_spec_backward_input():
    with pytest.raises(ValueError, match="backward_input"):
        _built_full(backward_input="state")


def test_hit_spec_pool_width_is_the_hit_spec_width():
    _c, params, _t = _built_full()
    assert params["density_zstats"]["pool"].shape[-1] == 3
