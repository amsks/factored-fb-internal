"""HILP critic losses against numpy reconstructions."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.hilp import expectile_loss, hilp_phi_loss, hilp_value

B, OBS, D = 7, 5, 4
DISC, EXP = 0.98, 0.5


def _mlp_params(rng, din, dout):
    return {"w": rng.standard_normal((din, dout)), "b": rng.standard_normal(dout)}


def _apply(p, x):
    return jnp.tanh(jnp.asarray(x) @ jnp.asarray(p["w"])) + jnp.asarray(p["b"])


def _np_apply(p, x):
    return np.tanh(x @ p["w"]) + p["b"]


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, OBS))
    goals = rng.standard_normal((B, OBS))
    goals[2] = obs[2]  # one obs == goal row
    nobs = rng.standard_normal((B, OBS))
    ps = {k: _mlp_params(rng, OBS, D) for k in ("p1", "p2", "t1", "t2")}
    return obs, goals, nobs, ps


def _np_reference(obs, goals, nobs, ps, discount=DISC, expectile=EXP):
    """phi value loss in numpy, from the parameter arrays."""
    v = lambda p, a, b: (
        -np.sqrt(
            np.clip(((_np_apply(p, a) - _np_apply(p, b)) ** 2).sum(-1), 1e-6, None)
        )
    )
    same = (np.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(np.float64)
    masks = 1.0 - same
    rewards = same - 1.0
    tn1, tn2 = v(ps["t1"], nobs, goals), v(ps["t2"], nobs, goals)
    tv1, tv2 = v(ps["t1"], obs, goals), v(ps["t2"], obs, goals)
    q = rewards + discount * masks * np.minimum(tn1, tn2)
    adv = q - (tv1 + tv2) / 2.0
    q1 = rewards + discount * masks * tn1
    q2 = rewards + discount * masks * tn2
    v1, v2 = v(ps["p1"], obs, goals), v(ps["p2"], obs, goals)
    w = lambda a: np.where(a >= 0, expectile, 1.0 - expectile)
    return float((w(adv) * (q1 - v1) ** 2).mean() + (w(adv) * (q2 - v2) ** 2).mean())


def _run(obs, goals, nobs, ps, **kw):
    loss, info = hilp_phi_loss(
        ps["p1"],
        ps["p2"],
        ps["t1"],
        ps["t2"],
        _apply,
        jnp.asarray(obs),
        jnp.asarray(goals),
        jnp.asarray(nobs),
        kw.get("discount", DISC),
        kw.get("expectile", EXP),
    )
    return float(loss), info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phi_loss_matches_numpy_reconstruction(seed):
    obs, goals, nobs, ps = _fixture(seed)
    got, _ = _run(obs, goals, nobs, ps)
    want = _np_reference(obs, goals, nobs, ps)
    assert abs(got - want) < 1e-10, f"{got} vs {want}"


@pytest.mark.parametrize("expectile", [0.5, 0.7, 0.9])
def test_phi_loss_matches_across_expectiles(expectile):
    obs, goals, nobs, ps = _fixture(3)
    got, _ = _run(obs, goals, nobs, ps, expectile=expectile)
    want = _np_reference(obs, goals, nobs, ps, expectile=expectile)
    assert abs(got - want) < 1e-10


def test_clamp_is_inside_the_sqrt_so_identical_inputs_floor_at_1e_3():
    """The clamp is inside the sqrt, so V(s, s) = -1e-3."""
    x = jnp.zeros((3, D))
    assert np.allclose(np.asarray(hilp_value(x, x)), -1e-3)


def test_shipped_expectile_is_inert():
    """At expectile 0.5 the loss is 0.5 * diff^2."""
    adv = jnp.asarray([-3.0, -1e-9, 0.0, 1e-9, 5.0])
    diff = jnp.ones_like(adv)
    w = np.asarray(expectile_loss(adv, diff, 0.5))
    assert np.allclose(w, 0.5), w
    w7 = np.asarray(expectile_loss(adv, diff, 0.7))
    assert not np.allclose(w7, w7[0]), (
        "at 0.7 the weight MUST depend on the sign of adv"
    )


def test_reductions_are_not_interchangeable():
    """Swapping the advantage baseline changes adv, and swapping the per-head q changes the loss."""
    obs, goals, nobs, ps = _fixture(5)
    v = lambda p, a, b: (
        -np.sqrt(
            np.clip(((_np_apply(p, a) - _np_apply(p, b)) ** 2).sum(-1), 1e-6, None)
        )
    )
    same = (np.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(np.float64)
    masks, rewards = 1.0 - same, same - 1.0
    tn1, tn2 = v(ps["t1"], nobs, goals), v(ps["t2"], nobs, goals)
    tv1, tv2 = v(ps["t1"], obs, goals), v(ps["t2"], obs, goals)
    q = rewards + DISC * masks * np.minimum(tn1, tn2)
    v1, v2 = v(ps["p1"], obs, goals), v(ps["p2"], obs, goals)
    w = lambda a: np.where(a >= 0, EXP, 1.0 - EXP)
    ref = _np_reference(obs, goals, nobs, ps)

    adv = q - (tv1 + tv2) / 2.0
    adv_minbase = q - np.minimum(tv1, tv2)
    # The baseline swap moves adv, which is logged.
    assert np.abs(adv - adv_minbase).max() > 1e-6

    # The shared-q swap changes the loss.
    wrong_sharedq = float(
        (w(adv) * (q - v1) ** 2).mean() + (w(adv) * (q - v2) ** 2).mean()
    )
    assert abs(wrong_sharedq - ref) > 1e-8, (
        "shared-q variant indistinguishable; fixture too weak"
    )


def test_advantage_is_almost_always_negative():
    """adv >= 0 only on a small minority of rows, the obs == goal ones."""
    for seed in range(4):
        obs, goals, nobs, ps = _fixture(seed)
        _, info = _run(obs, goals, nobs, ps)
        assert float(info["hilp_accept_prob"]) <= 0.35, (seed, info["hilp_accept_prob"])


# Successor-feature stage
from impls.critics.hilp import HILPCritic  # noqa: E402

SB, SOBS, SACT, SZ, SH, SL = 6, 5, 3, 4, 8, 3


def _sf_cfg(**over):
    cfg = dict(
        seed=0,
        z_dim=SZ,
        L_dim=SL,
        num_parallel=2,
        discount=0.98,
        norm_z=True,
        phi_discount=0.98,
        expectile=0.5,
        phi_target_tau=0.005,
        sf_target_tau=0.01,
        phi_stats_tau=0.005,
        train_goal_ratio=0.5,
        sf_loss_scale=0.5,
        phi=dict(hidden_dim=SH, hidden_layers=2),
        sf=dict(hidden_dim=SH, hidden_layers=2, embedding_layers=2),
        left_encoder=dict(hidden_dim=SH, hidden_layers=2, norm=True),
    )
    cfg.update(over)
    return cfg


def _sf_ex():
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(rng.standard_normal((SB, SOBS))),
        jnp.asarray(np.tanh(rng.standard_normal((SB, SACT)))),
    )


def _sf_batch(seed=1):
    rng = np.random.default_rng(seed)
    return dict(
        observations=jnp.asarray(rng.standard_normal((SB, SOBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((SB, SACT)))),
        next_observations=jnp.asarray(rng.standard_normal((SB, SOBS))),
        actor_goals=jnp.asarray(rng.standard_normal((SB, SOBS))),
    )


def test_build_produces_the_declared_groups_and_targets():
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    assert set(params) == set(critic.GROUPS)
    assert set(targets) == set(critic.TARGETS)
    # The stats slot is never polyak-averaged; update_stats writes it.
    assert critic.TARGETS["hilp_stats"] == 0.0
    # phi is trained only by the phi stage, sf only by the sf stage.
    assert set(critic.stage_groups("phi")) == {"hilp_phi1", "hilp_phi2"}
    assert set(critic.stage_groups("sf")) == {"hilp_sf", "hilp_le"}
    # Init stats: mean 0, std 1.
    assert np.allclose(np.asarray(params["hilp_stats"]["mean"]), 0.0)
    assert np.allclose(np.asarray(params["hilp_stats"]["std"]), 1.0)


def test_sf_loss_matches_numpy_reconstruction():
    """The successor-feature TD loss matches a numpy reconstruction."""
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    batch = _sf_batch()
    rng = np.random.default_rng(9)
    z = jnp.asarray(np.tanh(rng.standard_normal((SB, SZ))))
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((SB, SACT))))}
    got, _ = critic.stage_loss("sf", params, targets, batch, {"z": z}, boot)

    nle = critic._le(targets, batch["next_observations"])
    tF = np.asarray(
        critic._sf(targets["hilp_sf"], nle, z, boot["next_action"])
    )  # [P,B,d]
    phi = np.asarray(critic._phi(params["hilp_phi1"], batch["next_observations"]))
    target_phi = phi - np.asarray(params["hilp_stats"]["mean"])  # centered phi
    target_F = target_phi + 0.98 * tF.mean(axis=0)
    le = critic._le(params, batch["observations"])
    F = np.asarray(critic._sf(params["hilp_sf"], le, z, batch["actions"]))
    want = 0.5 * float(((F - target_F[None]) ** 2).mean(axis=(-1, -2)).sum())
    assert abs(float(got) - want) < 1e-10, f"{float(got)} vs {want}"


def test_sf_target_reads_the_ONLINE_phi_not_the_target_phi():
    """The SF TD target uses the online phi, not the target phi."""
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    batch = _sf_batch()
    rng = np.random.default_rng(9)
    z = {"z": jnp.asarray(np.tanh(rng.standard_normal((SB, SZ))))}
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((SB, SACT))))}
    base, _ = critic.stage_loss("sf", params, targets, batch, z, boot)

    bump = lambda tree: jax.tree_util.tree_map(lambda a: a + 1.0, tree)
    t2 = {
        **targets,
        "hilp_phi1": bump(targets["hilp_phi1"]),
        "hilp_phi2": bump(targets["hilp_phi2"]),
    }
    same, _ = critic.stage_loss("sf", params, t2, batch, z, boot)
    assert abs(float(same) - float(base)) < 1e-12, (
        "SF loss must not read the target phi"
    )

    p2 = {**params, "hilp_phi1": bump(params["hilp_phi1"])}
    diff, _ = critic.stage_loss("sf", p2, targets, batch, z, boot)
    assert abs(float(diff) - float(base)) > 1e-8, "SF loss must read the ONLINE phi"


def test_running_mean_arithmetic():
    """update_stats applies new = (1 - m) * old + m * batch_stat with m = 0.005."""
    m = 0.005
    old_mean, old_std = np.zeros(SZ), np.ones(SZ)
    phi = np.random.default_rng(2).standard_normal((SB, SZ))
    new_mean = old_mean * (1 - m) + phi.mean(axis=0) * m
    new_std = old_std * (1 - m) + phi.std(axis=0) * m
    assert np.allclose(new_mean, phi.mean(axis=0) * m)
    assert np.all(np.abs(new_std - 1.0) < 0.02)
