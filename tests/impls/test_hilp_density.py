"""Density-HILP: the estimator swap must change ONLY the estimator.

The design claim is that hilp_density_tdj vs hilp_tdj isolates the successor estimator,
and it rests on one structural fact: HILP's phi is trained entirely by the Hilbert value
loss, and the successor objective reads it detached. If a phi gradient ever flowed from
the density cross-entropy, the two arms would differ in how phi is learned as well, and
the comparison would stop meaning what it claims.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent
from impls.critics.hilp_density import HilpDensityConfigError


@pytest.fixture(scope="module")
def built():
    cfg = make_config("hilp_density_tdj", "ddpgbc", "dmc-walker", [])
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((6, 24)))
    act = jnp.asarray(rng.standard_normal((6, 6)))
    ag = Agent.create("hilp_density_tdj", "ddpgbc", 0, obs, act, cfg)
    return ag, obs, act, cfg


def test_config_is_the_substrate_plus_density_knobs(built):
    _, _, _, cfg = built
    base = make_config("hilp_tdj", "ddpgbc", "dmc-walker", [])
    assert float(cfg["temp"]) == 1.0 and bool(cfg["actor_q_center"]) is True
    # the four td_jepa parity fixes must be INHERITED, not restated or dropped
    for k in ("hilp_q_loss", "hilp_head_reduce", "hilp_actor_head_reduce", "hilp_sf_loss_scale"):
        assert cfg.get(k) == base.get(k), f"{k} drifted from the hilp_tdj substrate"


def test_q_loss_true_is_refused():
    """hilp_q_loss selects vector-vs-scalar REGRESSION; this critic does neither."""
    cfg = make_config("hilp_density_tdj", "ddpgbc", "dmc-walker", ["hilp_q_loss=true"])
    rng = np.random.default_rng(0)
    with pytest.raises(HilpDensityConfigError, match="hilp_q_loss"):
        Agent.create("hilp_density_tdj", "ddpgbc", 0,
                     jnp.asarray(rng.standard_normal((6, 24))),
                     jnp.asarray(rng.standard_normal((6, 6))), cfg)


def test_density_ce_sends_no_gradient_into_phi(built):
    """THE test. The sf stage must leave phi's parameters untouched."""
    ag, obs, act, _ = built
    cr = ag.critic
    batch = {"observations": obs, "next_observations": obs[::-1], "actions": act,
             "terminals": jnp.zeros(obs.shape[0])}
    z = jnp.asarray(np.random.default_rng(1).standard_normal((obs.shape[0], cr.z_dim)))
    cond, boot = {"z": z}, {"next_action": act}
    params = {g: ag.net[g].params for g in cr.GROUPS}
    target = {g: ag.target[g] for g in cr.TARGETS}

    def loss_of(p):
        merged = {**params, **p}
        return cr.stage_loss("sf", merged, target, batch, cond, boot)[0]

    g = jax.grad(loss_of)({"hilp_phi1": params["hilp_phi1"],
                           "hilp_sf": params["hilp_sf"]})
    phi_g = jnp.concatenate([x.reshape(-1) for x in jax.tree_util.tree_leaves(g["hilp_phi1"])])
    sf_g = jnp.concatenate([x.reshape(-1) for x in jax.tree_util.tree_leaves(g["hilp_sf"])])
    assert float(jnp.abs(phi_g).max()) == 0.0, "density CE leaked a gradient into phi"
    assert float(jnp.abs(sf_g).max()) > 0.0, "density CE trains nothing -- test is vacuous"


def test_target_rows_are_distributions(built):
    """(1-d)onehot + d*softmax must stay on the simplex, including at terminals."""
    ag, obs, act, _ = built
    cr = ag.critic
    B = obs.shape[0]
    for term in (jnp.zeros(B), jnp.ones(B)):
        batch = {"observations": obs, "next_observations": obs[::-1], "actions": act,
                 "terminals": term}
        z = jnp.asarray(np.random.default_rng(2).standard_normal((B, cr.z_dim)))
        _, info = cr.stage_loss("sf", {g: ag.net[g].params for g in cr.GROUPS},
                                {g: ag.target[g] for g in cr.TARGETS},
                                batch, {"z": z}, {"next_action": act})
        assert np.isfinite(float(info["density_ce"]))
        # A terminal row is a pure one-hot, so its target entropy must be ~0. Not exactly
        # 0: the metric is -sum t*log(t + 1e-8), and log(1 + 1e-8) makes a one-hot come out
        # at -1e-8. Assert magnitude, not sign.
        ent = float(info["density_tgt_entropy"])
        if float(term[0]) == 1.0:
            assert abs(ent) < 1e-5, f"terminal target is not a one-hot (entropy {ent})"
        else:
            assert ent > 1e-3, f"non-terminal target collapsed to a one-hot (entropy {ent})"
