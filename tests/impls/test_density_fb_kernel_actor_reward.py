"""density_fb's THIRD z-branch actor reward, `measure_density_z_actor_reward: kernel`
(2026-09-08; the bridge stack's 'kernel' cell, never ported until now).

Under the reward-sample index the critic conditions F and the actor on z exactly as under
'latent' / 'latent_centered'. What changes is ONLY the per-candidate reward kg the actor's
greedy readout Q = sum_m w_m kg_m integrates: under 'kernel' it is the EXACT training-task
reward on the candidate set, _delta_phi(goal rows, u_raw, blocks) -- the same estimator the
goal-conditioned branch uses (93 / 69 on cube-single / cube-double) -- stop-gradiented.
Rows whose z is a sphere draw (measure_density_z_rand_frac) have no goal, so they fall back
to the 'latent_centered' proxy; stage_loss reports the fraction that used the exact reward.
Nothing at eval / infer_cond time changes: the actor just rolls on z.

  1. kernel + z_rand_frac=0: q_fn's kg IS the goal branch's _delta_phi on the same batch,
     and Q differs from 'latent_centered'.
  2. the actor-loss gradient w.r.t. the action is finite and nonzero; nothing flows into kg.
  3. 'latent' and 'latent_centered' are bit-identical to their pre-change formulas; the
     kernel fallback on all-sphere rows IS 'latent_centered', exactly.
  4. the option round-trips through load_config / env.config.make_config overrides.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.density_fb import DensityFBCritic, _delta_phi
from impls.utils.config_utils import load_config
from tests.impls.test_density_fb_reward_index import (
    B, D, OBS, _agent, _batch, _built_full, _cfg, _ex, _net_from, _perturbed,
)


def _kernel_setup(z_rand_frac=0.0, seed=0):
    obs, act = _ex()
    critic, params, targets = _built_full(measure_density_z_actor_reward="kernel",
                                          measure_density_z_rand_frac=z_rand_frac)
    params = _perturbed(params)
    batch = _batch()
    cond = critic.sample_cond(_net_from(params), batch, jax.random.PRNGKey(seed), target=targets)
    return critic, params, targets, batch, cond, obs, act


# ==========================================================================================
# 1. kernel == the goal branch's exact reward on the candidate set
# ==========================================================================================
def test_kernel_option_is_accepted_and_declared():
    critic, _p, _t = _built_full(measure_density_z_actor_reward="kernel")
    assert critic.z_on and critic.z_actor_reward == "kernel"


def test_kernel_kg_equals_the_goal_branch_delta_phi_at_zero_rand_frac():
    critic, params, targets, batch, cond, obs, act = _kernel_setup(z_rand_frac=0.0)
    assert np.all(np.asarray(cond["_z_hindsight"]) == 1.0), "frac=0 => every row keeps its goal"
    kg = np.asarray(critic.z_actor_reward_rows(cond))
    # the goal branch: a z=OFF critic on the SAME batch reads _delta_phi(actor goal, u_raw)
    goal_critic, goal_params, _ = DensityFBCritic.build(obs, act, _cfg(False))
    goal_cond = goal_critic.sample_cond(_net_from(goal_params), batch, jax.random.PRNGKey(0))
    g_rows = goal_critic._kernel_rows(goal_cond, "goal", goal_cond["goal"])
    ref = np.asarray(_delta_phi(g_rows, goal_cond["_density_u_raw"], goal_critic.blocks))
    assert kg.shape == (B, B)
    np.testing.assert_allclose(kg, ref, rtol=1e-6, atol=1e-8)
    # and the goal the z was built from is the row's actor goal, which cond still carries
    np.testing.assert_array_equal(np.asarray(cond["goal"]), np.asarray(batch["actor_goals"]))


def test_kernel_q_differs_from_latent_centered_on_identical_rows():
    critic_k, params, targets, batch, cond, obs, act = _kernel_setup(z_rand_frac=0.0)
    critic_c, _p, _t = _built_full(measure_density_z_actor_reward="latent_centered")
    q_k, _ = critic_k.q_fn(params, obs, cond, act)
    q_c, _ = critic_c.q_fn(params, obs, cond, act)
    assert np.isfinite(np.asarray(q_k)).all() and np.isfinite(np.asarray(q_c)).all()
    assert not np.allclose(np.asarray(q_k), np.asarray(q_c)), "kernel must not be the proxy"


# ==========================================================================================
# 2. gradients: through the action, never into kg
# ==========================================================================================
def test_kernel_actor_gradient_flows_through_the_action_and_not_into_kg():
    critic, params, targets, batch, cond, obs, act = _kernel_setup(z_rand_frac=0.0)

    def q_of(a, goal):
        q, _ = critic.q_fn(params, obs, {**cond, "goal": goal}, a)
        return q.sum()

    g_act, g_goal = jax.grad(q_of, argnums=(0, 1))(act, cond["goal"])
    assert np.isfinite(np.asarray(g_act)).all()
    assert np.abs(np.asarray(g_act)).max() > 0.0, "no greedy signal reaches the action"
    # kg is stop-gradiented: the goal rows enter q_fn ONLY through kg under the index, so
    # a nonzero gradient there would be the gradient leaking into the reward.
    assert np.all(np.asarray(g_goal) == 0.0), "gradient leaked into kg through the goal rows"


def test_kernel_runs_a_real_update_and_reports_the_exact_reward_fraction():
    agent, _cfg_ = _agent(measure_density_z_actor_reward="kernel", measure_density_z_rand_frac=0.0)
    _updated, info = agent.update(_batch())
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    assert np.isfinite(float(np.asarray(info["density_ce"])))
    assert float(np.asarray(info["density_z_kernel_frac"])) == 1.0
    agent1, _ = _agent(measure_density_z_actor_reward="kernel", measure_density_z_rand_frac=1.0)
    _u, info1 = agent1.update(_batch())
    assert float(np.asarray(info1["density_z_kernel_frac"])) == 0.0
    # the proxy arms do not grow the key: their info dicts are unchanged
    agent_c, _ = _agent(measure_density_z_actor_reward="latent_centered")
    _u, info_c = agent_c.update(_batch())
    assert "density_z_kernel_frac" not in info_c


# ==========================================================================================
# 3. the two existing options are bit-identical to their pre-change formulas
# ==========================================================================================
def _reference_q(critic, params, obs, cond, act, basis):
    """The pre-2026-09-08 z-branch readout, written out: F(s,a,z) . B(u) softmax, kg =
    basis(u) . z / sqrt(d), min over heads, actor_q_center normalizer."""
    f_p = jax.lax.stop_gradient(params["density_f"])
    b_p = jax.lax.stop_gradient(params["density_b"])
    b_emb = critic._B(b_p, cond["_density_u_sc"])
    z = cond["z"]
    kg = jax.lax.stop_gradient(jnp.einsum('md,bd->bm', basis, z) / jnp.sqrt(float(critic.z_dim)))
    f_emb = critic._F(f_p, critic._sobs(params, obs), act, z)
    a_tau = critic.temp if critic.actor_temp is None else float(critic.actor_temp)
    w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / a_tau, -1)
    q1, q2 = (w * kg[None]).sum(-1)
    q = jnp.minimum(q1, q2)
    return (q, (q - q.mean())[None]) if critic.actor_q_center else (q, jnp.stack([q1, q2], 0))


@pytest.mark.parametrize("option,basis_key", [("latent", "_density_tb_u"),
                                              ("latent_centered", "_density_zenc_u")])
def test_existing_options_are_bit_identical_to_their_formulas(option, basis_key):
    obs, act = _ex()
    critic, params, targets = _built_full(measure_density_z_actor_reward=option)
    params = _perturbed(params)
    cond = critic.sample_cond(_net_from(params), _batch(), jax.random.PRNGKey(0), target=targets)
    q, qs = critic.q_fn(params, obs, cond, act)
    q_ref, qs_ref = _reference_q(critic, params, obs, cond, act, cond[basis_key])
    np.testing.assert_array_equal(np.asarray(q), np.asarray(q_ref))
    np.testing.assert_array_equal(np.asarray(qs), np.asarray(qs_ref))
    # the helper exposes the same kg the readout integrates
    kg_ref = np.asarray(jnp.einsum('md,bd->bm', cond[basis_key], cond["z"]) / np.sqrt(float(D)))
    np.testing.assert_array_equal(np.asarray(critic.z_actor_reward_rows(cond)), kg_ref)


def test_kernel_fallback_on_sphere_rows_is_latent_centered_exactly():
    """z_rand_frac=1: no row has a goal, so every row uses the proxy -- and that proxy must
    be the 'latent_centered' value bit for bit (same cond, same params)."""
    critic_k, params, targets, batch, cond, obs, act = _kernel_setup(z_rand_frac=1.0)
    assert np.all(np.asarray(cond["_z_hindsight"]) == 0.0)
    critic_c, _p, _t = _built_full(measure_density_z_actor_reward="latent_centered")
    q_k, qs_k = critic_k.q_fn(params, obs, cond, act)
    q_c, qs_c = critic_c.q_fn(params, obs, cond, act)
    np.testing.assert_array_equal(np.asarray(q_k), np.asarray(q_c))
    np.testing.assert_array_equal(np.asarray(qs_k), np.asarray(qs_c))


def test_kernel_mixes_rowwise_by_the_hindsight_mask():
    """A hand-built mask: goal rows read the exact kernel, sphere rows the proxy, row by row."""
    critic_k, params, targets, batch, cond, obs, act = _kernel_setup(z_rand_frac=0.0)
    critic_c, _p, _t = _built_full(measure_density_z_actor_reward="latent_centered")
    hs = jnp.asarray((np.arange(B) % 2 == 0).astype(float))[:, None]
    mixed = {**cond, "_z_hindsight": hs}
    kg = np.asarray(critic_k.z_actor_reward_rows(mixed))
    kg_exact = np.asarray(_delta_phi(cond["goal"], cond["_density_u_raw"], critic_k.blocks))
    kg_proxy = np.asarray(critic_c.z_actor_reward_rows(cond))
    on = np.asarray(hs)[:, 0] > 0.5
    np.testing.assert_array_equal(kg[on], kg_exact[on])
    np.testing.assert_array_equal(kg[~on], kg_proxy[~on])


# ==========================================================================================
# 4. config round trip
# ==========================================================================================
def test_kernel_round_trips_through_the_config_loaders():
    import yaml
    from env.config import make_config
    ov = ["measure_density_z_actor_reward=kernel"]
    cfg = load_config("density_fb_z", "flowbc", "cube_single", overrides=ov)
    assert cfg["measure_density_z_actor_reward"] == "kernel" and cfg["measure_density_z"] is True
    cfg_main = make_config("density_fb_z", "flowbc", "cube_single", overrides=ov)
    assert cfg_main["measure_density_z_actor_reward"] == "kernel"
    critic, _p, _t = DensityFBCritic.build(
        *_ex(), {**cfg_main, "seed": 0, "batch_size": B, "measure_density_dim": D,
                 "value_hidden_dims": [32, 32], "measure_density_z_eval_pool_m": 8})
    assert critic.z_actor_reward == "kernel"
    # the shipped z config keeps its default; the goal-conditioned config stays untouched
    assert yaml.safe_load(open("impls/configs/critic/density_fb_z.yaml"))[
        "measure_density_z_actor_reward"] == "latent_centered"
    assert "measure_density_z_actor_reward" not in yaml.safe_load(
        open("impls/configs/critic/density_fb.yaml"))


def test_unknown_option_is_still_refused():
    with pytest.raises(ValueError):
        _built_full(measure_density_z_actor_reward="bogus")
