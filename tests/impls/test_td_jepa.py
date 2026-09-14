"""TD-JEPA port contracts and reference parity.

The port is a transcription of ~/git/baselines/td_jepa/metamotivo/agents/td_jepa
(`update_tdjepa_asym`, `update_actor`, `TDJEPAModel.reward_inference`). These tests pin
the pieces that a transcription can get silently wrong, which on the two previous ports
from this same codebase were all CONFIG values rather than math: HILP ran four of
td_jepa's values differently and psi collapsed to 2.5 effective dimensions of 50, and
RLDP's orthonormality term turned out to have no gradient at all as shipped.

The TD target is the one formula worth restating, because each predictor's target is the
OTHER encoder one step ahead:

    target_phi = psi_tgt(s') + gamma * mean_P Pred_phi_tgt(phi_tgt(s'), z, a')
    target_psi = phi_tgt(s') + gamma * mean_P Pred_psi_tgt(psi_tgt(s'), z, a')
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.critics.td_jepa import TDJEPACritic

B, OBS, ACT, ZD, PHID, P = 6, 7, 3, 5, 4, 2


def _config(**over):
    c = make_config("td_jepa", "flowbc", "cube_single", [])
    c["z_dim"], c["tdjepa_phi_dim"], c["num_parallel"], c["seed"] = ZD, PHID, P, 0
    c["backward"]["hidden_dim"], c["backward"]["hidden_layers"] = 16, 2
    c["forward"]["hidden_dim"], c["forward"]["hidden_layers"] = 16, 1
    for k, v in over.items():
        c[k] = v
    return c


def _build(**over):
    rng = np.random.default_rng(0)
    ex_obs = jnp.asarray(rng.normal(size=(B, OBS)), jnp.float32)
    ex_act = jnp.asarray(rng.normal(size=(B, ACT)), jnp.float32)
    critic, params, target = TDJEPACritic.build(ex_obs, ex_act, _config(**over))
    return critic, params, target


def _batch():
    rng = np.random.default_rng(1)
    return {"observations": jnp.asarray(rng.normal(size=(B, OBS)), jnp.float32),
            "actions": jnp.asarray(rng.normal(size=(B, ACT)), jnp.float32),
            "next_observations": jnp.asarray(rng.normal(size=(B, OBS)), jnp.float32)}


def _cond_boot():
    rng = np.random.default_rng(2)
    return ({"z": jnp.asarray(rng.normal(size=(B, ZD)), jnp.float32)},
            {"next_action": jnp.asarray(rng.normal(size=(B, ACT)), jnp.float32)})


@pytest.fixture(scope="module")
def case():
    critic, params, target = _build()
    return critic, params, target, _batch(), *_cond_boot()


# --- shapes: the asymmetric variant's crossed widths are its whole point -------------
def test_predictors_cross_the_encoder_widths(case):
    critic, params, _t, batch, cond, _b = case
    phi = critic._phi(params["phi_enc"], batch["observations"])
    psi = critic._psi(params["psi_enc"], batch["observations"])
    assert phi.shape == (B, PHID) and psi.shape == (B, ZD)
    # Pred_phi lives in phi-space and outputs PSI's width; Pred_psi the reverse.
    assert critic._pred_phi(params["phi_pred"], phi, cond["z"], batch["actions"]).shape == (P, B, ZD)
    assert critic._pred_psi(params["psi_pred"], psi, cond["z"], batch["actions"]).shape == (P, B, PHID)


# --- the TD target ------------------------------------------------------------------
def test_td_targets_are_the_other_encoder_plus_discounted_prediction(case):
    """Reproduce both targets independently and check the loss they induce."""
    critic, params, target, batch, cond, boot = case
    z, na, nobs = cond["z"], boot["next_action"], batch["next_observations"]
    nphi = critic._phi(target["phi_enc"], nobs)
    npsi = critic._psi(target["psi_enc"], nobs)
    g = critic.discount
    exp_phi = npsi + g * critic._pred_phi(target["phi_pred"], nphi, z, na).mean(0)
    exp_psi = nphi + g * critic._pred_psi(target["psi_pred"], npsi, z, na).mean(0)

    phi_enc = critic._phi(params["phi_enc"], batch["observations"])
    psi_enc = critic._psi(params["psi_enc"], batch["observations"])
    pp = critic._pred_phi(params["phi_pred"], phi_enc, z, batch["actions"])
    sp = critic._pred_psi(params["psi_pred"], psi_enc, z, batch["actions"])
    exp_phi_loss = jnp.mean(jnp.sum((pp - exp_phi[None]) ** 2, -1))
    exp_psi_loss = jnp.mean(jnp.sum((sp - exp_psi[None]) ** 2, -1))

    _loss, info = critic.stage_loss("tdjepa", params, target, batch, cond, boot)
    np.testing.assert_allclose(info["phi_tdjepa_loss"], exp_phi_loss, rtol=1e-6)
    np.testing.assert_allclose(info["psi_tdjepa_loss"], exp_psi_loss, rtol=1e-6)
    np.testing.assert_allclose(info["td_target_phi"], exp_phi.mean(), rtol=1e-6)


def test_target_towers_only_the_online_ones_receive_gradient(case):
    """The target block is under stop_gradient: perturbing the ONLINE params must move
    the loss, perturbing them through the target dict must not."""
    critic, params, target, batch, cond, boot = case
    g = jax.grad(lambda t: critic.stage_loss("tdjepa", params, t, batch, cond, boot)[0])(target)
    flat = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    np.testing.assert_allclose(flat, jnp.zeros_like(flat), atol=0)
    go = jax.grad(lambda p: critic.stage_loss("tdjepa", p, target, batch, cond, boot)[0])(params)
    for grp in critic.GROUPS:
        assert max(abs(float(x)) for x in jnp.concatenate(
            [v.ravel() for v in jax.tree.leaves(go[grp])])) > 0, f"{grp} got no gradient"


# --- the orthonormality term --------------------------------------------------------
def test_ortho_is_the_reference_batch_gram_form(case):
    critic, params, target, batch, cond, boot = case
    _loss, info = critic.stage_loss("tdjepa", params, target, batch, cond, boot)
    enc = critic._phi(params["phi_enc"], batch["observations"])
    cov = enc @ enc.T
    off = 1.0 - jnp.eye(B)
    exp_off = 0.5 * jnp.sum((cov * off) ** 2) / off.sum()
    exp_diag = -jnp.mean(jnp.diagonal(cov))
    np.testing.assert_allclose(info["phi_orth_offdiag"], exp_off, rtol=1e-6)
    np.testing.assert_allclose(info["phi_orth_diag"], exp_diag, rtol=1e-6)
    np.testing.assert_allclose(info["phi_orth_loss"], exp_off + exp_diag, rtol=1e-6)


def test_total_loss_sums_both_predictors_and_both_ortho_terms(case):
    critic, params, target, batch, cond, boot = case
    loss, info = critic.stage_loss("tdjepa", params, target, batch, cond, boot)
    exp = (info["phi_tdjepa_loss"] + info["psi_tdjepa_loss"]
           + critic.phi_ortho_coef * info["phi_orth_loss"]
           + critic.psi_ortho_coef * info["psi_orth_loss"])
    np.testing.assert_allclose(loss, exp, rtol=1e-6)


# --- the readouts -------------------------------------------------------------------
def test_q_is_the_phi_predictor_against_z_and_never_trains_the_encoder(case):
    critic, params, _t, batch, cond, _b = case
    phi = critic._phi(params["phi_enc"], batch["observations"])
    preds = critic._pred_phi(params["phi_pred"], phi, cond["z"], batch["actions"])
    q, _qs = critic.q_fn(params, batch["observations"], cond, batch["actions"])
    np.testing.assert_allclose(q, (preds * cond["z"]).sum(-1).mean(0), rtol=1e-6)
    # update_actor encodes phi under no_grad, so the actor's objective cannot move it.
    g = jax.grad(lambda p: critic.q_fn(p, batch["observations"], cond, batch["actions"])[0].sum())(params)
    enc_g = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g["phi_enc"])])
    np.testing.assert_allclose(enc_g, jnp.zeros_like(enc_g), atol=0)


def test_infer_cond_is_a_least_squares_solve_on_psi_then_normalized(case):
    """reward_inference: z = lstsq(psi(s'), r), sqrt(d)-normalized. On a reward that IS
    linear in psi the solve must recover the generating direction up to that norm."""
    critic, params, _t, batch, _c, _b = case
    net = {"psi_enc": type("S", (), {"params": params["psi_enc"]})}
    obs = batch["next_observations"]
    psi = critic._psi(params["psi_enc"], obs)
    w = jnp.asarray(np.random.default_rng(3).normal(size=(ZD,)), jnp.float32)
    z = critic.infer_cond(net, obs, psi @ w)["z"]
    np.testing.assert_allclose(float(jnp.linalg.norm(z)), np.sqrt(ZD), rtol=1e-5)
    cos = float(z @ w / (jnp.linalg.norm(z) * jnp.linalg.norm(w)))
    assert cos > 0.999, f"solve did not recover the generating direction (cos {cos})"


def test_sample_cond_mixes_goal_embeddings_and_stays_on_the_sphere(case):
    critic, params, _t, batch, _c, _b = case
    net = {"psi_enc": type("S", (), {"params": params["psi_enc"]})}
    z = critic.sample_cond(net, batch, jax.random.PRNGKey(0))["z"]
    assert z.shape == (B, ZD)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(z), axis=-1),
                               np.full(B, np.sqrt(ZD)), rtol=1e-5)
    # ratio 0 keeps the Gaussian draw only; ratio 1 takes every row from psi(goal).
    allg = dataclasses.replace(critic, train_goal_ratio=1.0)
    zg = allg.sample_cond(net, batch, jax.random.PRNGKey(0))["z"]
    assert not np.allclose(np.asarray(z), np.asarray(zg))


def test_map_eval_cond_is_not_goal_blind(case):
    """Without this hook eval runs on eval_cond0's zero z and Q == 0 for every action --
    the failure that made eight HILP cells meaningless on 2026-09-08."""
    critic, params, _t, batch, _c, _b = case
    net = {"psi_enc": type("S", (), {"params": params["psi_enc"]})}
    out = critic.map_eval_cond(net, {"goal": batch["next_observations"]})
    np.testing.assert_allclose(np.linalg.norm(np.asarray(out["z"]), axis=-1),
                               np.full(B, np.sqrt(ZD)), rtol=1e-5)
    assert critic.map_eval_cond(net, {"z": jnp.ones((B, ZD))}).keys() == {"z"}


# --- config parity ------------------------------------------------------------------
def test_dmc_arm_carries_the_reference_train_config():
    c = make_config("td_jepa_tdj", "ddpgbc", "dmc-walker", [])
    assert int(c["batch_size"]) == 1024
    assert float(c["discount"]) == 0.98
    assert float(c["ortho_coef"]) == 1.0            # phi_ortho_coef = psi_ortho_coef = 1.0
    assert float(c["train_goal_ratio"]) == 0.5
    assert float(c["b_target_tau"]) == 0.005        # encoder_target_tau
    assert float(c["f_target_tau"]) == 0.005        # predictor_target_tau
    assert float(c["lr_f"]) == float(c["lr_b"]) == 1.0e-4
    assert bool(c["norm_z"]) is True
    assert int(c["z_dim"]) == int(c["tdjepa_phi_dim"]) == 50


def test_actor_learning_rate_must_be_overridden_at_launch():
    """The reference sets lr_actor 1e-4 (TDJEPAAgentTrainConfig.lr_actor). Config merge
    order is (domain, critic, actor) with the ACTOR winning, and actor/ddpgbc.yaml
    declares 3e-4, so the critic yaml cannot own this value -- every ddpgbc pair in this
    tree resolves to 3e-4, which is the F1 defect recorded for the existing walker arms
    in docs/notes/2026-09-14-walker-rescore.md ("3x hot on the actor").

    This test exists so the discrepancy is loud rather than silent: any TD-JEPA launch
    must pass `--override lr_actor=1e-4`, and scripts/launch_td_jepa.sh does. Editing
    actor/ddpgbc.yaml instead would move every other arm in the tree."""
    c = make_config("td_jepa_tdj", "ddpgbc", "dmc-walker", [])
    assert float(c["lr_actor"]) == 3.0e-4, "actor/ddpgbc.yaml no longer owns lr_actor"
    over = make_config("td_jepa_tdj", "ddpgbc", "dmc-walker", ["lr_actor=1.0e-4"])
    assert float(over["lr_actor"]) == 1.0e-4, "the launch override no longer takes effect"


def test_ogbench_arm_sits_in_this_trees_protocol():
    c = make_config("td_jepa", "flowbc", "cube_single", [])
    r = make_config("rldp", "flowbc", "cube_single", [])
    assert int(c["batch_size"]) == int(r["batch_size"]) == 256
    assert float(c["discount"]) == float(r["discount"]) == 0.99


def test_unported_branches_refuse_rather_than_silently_differ():
    rng = np.random.default_rng(0)
    ex_obs = jnp.asarray(rng.normal(size=(B, OBS)), jnp.float32)
    ex_act = jnp.asarray(rng.normal(size=(B, ACT)), jnp.float32)
    for key in ("tdjepa_symmetric", "tdjepa_actor_full_encoder"):
        with pytest.raises(NotImplementedError):
            TDJEPACritic.build(ex_obs, ex_act, _config(**{key: True}))


def test_update_runs_in_f32(tmp_path):
    """The path that actually ships: a real Agent.update in f32, which is what every
    training run uses. This exists because test_phase0_gate.py has to SKIP this critic
    under jax_enable_x64 -- four test modules turn x64 on process-globally at import and
    XLA 0.10 then builds a cyclic HLO on the critic's two identical prediction branches
    (full analysis in impls/critics/td_jepa.py). Without this test that skip would leave
    the critic's end-to-end update untested inside the suite.

    x64 is toggled off for the duration rather than assumed off, since import order
    decides it."""
    import jax as _jax
    from impls.agents import Agent
    from impls.utils.config_utils import load_config
    from impls.utils.datasets import make_dataset
    from impls.utils.env_utils import make_env_and_datasets

    was = _jax.config.jax_enable_x64
    _jax.config.update("jax_enable_x64", False)
    try:
        cfg = load_config("td_jepa", "flowbc", "cube_single")
        cfg["batch_size"] = 8
        _env, train_ds, _val = make_env_and_datasets(cfg["env_name"])
        ds = make_dataset(train_ds, cfg)
        ex = ds.sample(1)
        agent = Agent.create("td_jepa", "flowbc", 0, jnp.asarray(ex["observations"]),
                             jnp.asarray(ex["actions"]), cfg)
        for _ in range(3):
            agent, info = agent.update(ds.sample(8))
        for k in ("total_loss", "phi_tdjepa_loss", "psi_tdjepa_loss", "actor_loss"):
            assert np.isfinite(float(info[k])), f"{k} is not finite: {info[k]}"
    finally:
        _jax.config.update("jax_enable_x64", was)


def test_all_four_groups_are_polyaked_at_the_reference_rates(case):
    critic, _p, _t, _b, _c, _bo = case
    assert set(critic.TARGETS) == set(critic.GROUPS)
    assert critic.TARGETS["phi_enc"] == critic.TARGETS["psi_enc"] == critic.enc_tau
    assert critic.TARGETS["phi_pred"] == critic.TARGETS["psi_pred"] == critic.pred_tau
