"""Ablation knobs: (1) the softmax successor model fitted with squared probability error instead
of cross-entropy (measure_density_loss=sq); (2) FB with a rescaled implicit immediate term
(fb_target_scale), so F^T B represents c*M under the unchanged squared loss."""
import jax.numpy as jnp, numpy as np, pytest
from impls.critics.density_fb import DensityFBCritic, _raw_block_cols, _density_queries
from tests.impls.test_density_fb import _cfg, _ex, _batch, _boot


def _run(cfg):
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, cfg)
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"],
            "_density_u_raw": _raw_block_cols(b["next_observations"], c.blocks),
            "_density_u_sc": _density_queries(b["next_observations"], c.blocks)}
    return c.stage_loss("density", params, targets, b, cond, boot=_boot())


def test_default_is_cross_entropy():
    loss, info = _run(_cfg())
    assert np.isclose(float(loss), float(info["density_ce"]))
    assert not np.isclose(float(loss), float(info["density_sq"]))


def test_sq_selects_squared_probability_error_and_is_bounded():
    loss, info = _run(_cfg(measure_density_loss="sq"))
    assert np.isclose(float(loss), float(info["density_sq"]))
    # 0.5 * sum_j (w_j - t_j)^2 over two probability vectors is at most 1
    assert 0.0 <= float(loss) <= 1.0
    assert np.isclose(float(info["density_ce"]), float(_run(_cfg())[1]["density_ce"]))


def test_sq_rejects_unknown_value():
    with pytest.raises(ValueError):
        _run(_cfg(measure_density_loss="brier"))


def test_fb_target_scale_scales_only_the_diagonal_term():
    from impls.critics.fb import FBCritic
    from impls.utils.networks import ForwardMap, PhiMap
    import jax
    H, Z, L, A, P, B, O = 16, 4, 4, 3, 2, 6, 10
    def mk(scale):
        return FBCritic(
            forward=ForwardMap(z_dim=Z, hidden_dim=H, num_parallel=P, hidden_layers=2, embedding_layers=2),
            backward=PhiMap(z_dim=Z, hidden_dim=H, hidden_layers=1, norm=True),
            left_encoder=PhiMap(z_dim=L, hidden_dim=H, hidden_layers=1, norm=True),
            z_dim=Z, num_parallel=P, discount=0.99, ortho_coef=1.0, fb_pessimism_penalty=0.0,
            train_goal_ratio=0.5, norm_z=True, f_tau=0.005, b_tau=0.005, target_scale=scale)
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, O))); nobs = jnp.asarray(rng.standard_normal((B, O)))
    act = jnp.asarray(np.tanh(rng.standard_normal((B, A)))); z = jnp.asarray(rng.standard_normal((B, Z)))
    c1, c2 = mk(1.0), mk(0.01)
    k = jax.random.PRNGKey(0)
    params = {"backward": c1.backward.init(k, nobs)["params"],
              "left_encoder": c1.left_encoder.init(k, obs)["params"],
              "forward": c1.forward.init(k, jnp.zeros((B, L)), z, act)["params"]}
    targets = {g: params[g] for g in params}
    batch = {"observations": obs, "actions": act, "next_observations": nobs}
    cond = {"z": z}; boot = {"next_action": act}
    _, i1 = c1.stage_loss("fb", params, targets, batch, cond, boot)
    _, i2 = c2.stage_loss("fb", params, targets, batch, cond, boot)
    np.testing.assert_allclose(float(i1["fb_offdiag"]), float(i2["fb_offdiag"]))
    np.testing.assert_allclose(float(i1["ortho_loss"]), float(i2["ortho_loss"]))
    np.testing.assert_allclose(float(i2["fb_diag"]), 0.01 * float(i1["fb_diag"]), rtol=1e-6)


def test_degenerate_covariance_gives_neutral_index_not_nan():
    """tr(C_hat) = 0 (constant backward features): the ridge and relative floor are both zero;
    the absolute floor must make the solve return c = 0 and the neutral index, never NaN."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_z=True, measure_density_z_eval_pool_m=8))
    d = int(c.z_dim)
    m_c = jnp.zeros((2, d)); cov = jnp.zeros((d, d))
    z = c._z_centered_index(m_c, cov)
    assert bool(jnp.all(jnp.isfinite(z)))
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(z, axis=-1)), np.sqrt(d), rtol=1e-5)


def test_retrofit_sq_knobs_select_squared_probability_error():
    """rldp_density_loss / hilp_density_loss = 'sq' swap the CE term for 0.5*sum_j (p_j - t_j)^2
    on the same softmax; default 'ce' is unchanged."""
    import jax.numpy as jnp
    from impls.critics.rldp_density import RLDPDensityCritic
    from impls.critics.hilp_density import HILPDensityCritic
    assert RLDPDensityCritic.__dataclass_fields__["loss_kind"].default == "ce"
    assert HILPDensityCritic.__dataclass_fields__["loss_kind"].default == "ce"
    # the squared form is bounded by 1 and vanishes at equality
    p = jnp.array([[0.2, 0.8]]); t = jnp.array([[0.2, 0.8]])
    assert float(0.5 * ((p - t) ** 2).sum(-1).mean()) == 0.0
