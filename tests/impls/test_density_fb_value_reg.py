"""Value-regression auxiliary (measure_density_value_reg_coef).

Adds a scalar-value TD term to the density stage loss: regress the reward-weighted
successor value <w, kg> toward its Bellman target <t, kg>, where t is the SAME
stop-gradiented distributional target the CE already uses and kg is the actor's
per-candidate reward. Off by default (coef 0.0 -> bit-identical to the shipped loss).
Motivation: docs value-inference probe showed the failure on walker/cheetah-run is the
accuracy of the critic's value across task vectors, which the distribution-only CE does
not directly supervise.
"""
import dataclasses
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.utils.config_utils import load_config
from impls.critics.density_fb import (
    DensityFBCritic, _delta_phi, _raw_block_cols, _density_queries,
)
from tests.impls.test_density_fb import _cfg, _ex, _batch, _boot


def _goal_cond(b):
    return {"value_goal": b["value_goals"], "goal": b["actor_goals"]}


def test_value_reg_coef_zero_is_bit_identical():
    """The default (coef 0.0) must not touch the loss the shipped cells were trained with."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch(seed=11)
    base = float(c.stage_loss("density", params, targets, b, _goal_cond(b), _boot())[0])
    c0 = dataclasses.replace(c, value_reg_coef=0.0)
    got = float(c0.stage_loss("density", params, targets, b, _goal_cond(b), _boot())[0])
    assert got == base  # exact: the coef>0 branch must be gated out entirely


def test_value_reg_adds_scalar_td_term_goal_arm():
    """coef>0 adds coef * 0.5 * mean((<w,kg> - <t,kg>)^2), kg the goal reward at the actor goal."""
    ex_obs, ex_act = _ex()
    coef = 0.7
    c, params, targets = DensityFBCritic.build(
        ex_obs, ex_act, _cfg(measure_density_value_reg_coef=coef))
    b = _batch(seed=11)
    cond, boot = _goal_cond(b), _boot()

    base = float(dataclasses.replace(c, value_reg_coef=0.0)
                 .stage_loss("density", params, targets, b, cond, boot)[0])
    total, info = c.stage_loss("density", params, targets, b, cond, boot)

    # numpy reference for the vr term (hard target default -> k = eye)
    blocks = c.blocks
    nobs, obs, acts, g = b["next_observations"], b["observations"], b["actions"], b["value_goals"]
    u_raw = _raw_block_cols(nobs, blocks)
    u_sc = _density_queries(nobs, blocks)
    b_emb = c.density_b.apply({"params": params["density_b"]}, u_sc)
    f_emb = c.density_f.apply({"params": params["density_f"]}, jnp.concatenate([obs, acts, g], -1))
    w = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / c.temp, -1))
    a_plus = boot["next_action_vgoal"]
    tb = c.density_b.apply({"params": targets["density_b"]}, u_sc)
    tf = c.density_f.apply({"params": targets["density_f"]}, jnp.concatenate([nobs, a_plus, g], -1))
    w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / c.temp, -1))
    k = np.eye(nobs.shape[0])                                    # hard default
    t = (1.0 - c.discount) * k[None] + c.discount * w_bar
    kg = np.asarray(_delta_phi(b["actor_goals"], u_raw, blocks))  # goal-arm reward
    q_pred = np.einsum('hbm,bm->hb', w, kg)
    q_tgt = np.einsum('hbm,bm->hb', t, kg)
    vr_ref = 0.5 * np.mean((q_pred - q_tgt) ** 2)

    assert np.allclose(float(info["density_value_reg"]), vr_ref, atol=1e-6)
    assert np.allclose(float(total), base + coef * vr_ref, atol=1e-6)
    assert not np.isclose(vr_ref, 0.0), "test batch gives a zero vr term; cannot discriminate"


def test_value_reg_z_arm_updates_end_to_end():
    """The ExoRL target is the z arm (measure_density_z=True), whose reward is the actor's
    latent index reward on the batch candidate support. Run a real agent update to catch any
    candidate-axis misalignment between z_actor_reward_rows and stage_loss's w."""
    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg = {**cfg, "batch_size": 16, "measure_density_dim": 8, "value_hidden_dims": [32, 32],
           "measure_density_z": True, "measure_density_z_eval_pool_m": 8,
           "measure_density_value_reg_coef": 0.5}
    agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)
    assert agent.critic.z_on and agent.critic.value_reg_coef == 0.5
    _a, info = agent.update(_batch(seed=0, b=16))
    assert np.isfinite(float(info["density_value_reg"])) and float(info["density_value_reg"]) >= 0.0
