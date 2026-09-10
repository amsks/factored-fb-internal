"""density_fb's THIRD z training-task family, `measure_density_z_khot_frac` (2026-09-10).

Today every z-indexed training row is either a HINDSIGHT GOAL task (z from the reward-sample
index of _delta_phi(actor_goal, u) on the batch's candidate set) or a SPHERE DRAW
(measure_density_z_rand_frac). Evaluation rewards on scene / puzzle / cube-double are
CONJUNCTIONS over the hit-spec blocks, and their inferred z lands far from every hindsight
goal's z. The k-hot family covers that region with an EXACT reward: pick a non-empty subset
S of the blocks (uniform over the size 1..n_blocks, then a uniform subset of that size), and
reward the candidate u by the product over b in S of the per-block Gaussian bump around the
row's own hindsight goal (blocks outside S ignored). z is encoded from that reward on the
same candidate set, through the same solve, as the goal rows.

Row precedence: the k-hot fraction is taken from the SPHERE fraction first, then from the
goal rows once khot_frac exceeds rand_frac:
    P(khot) = khot_frac,  P(sphere) = max(rand_frac - khot_frac, 0),  P(goal) = the rest.

  1. khot_frac=0 (the default) leaves cond["z"], the actor loss and stage_loss bit-identical.
  2. on a 2-block env every k-hot row's mask has 1 or 2 blocks set, and the exact reward at
     the goal itself is 1 whatever the mask.
  3. a candidate matching only block 0 is rewarded 1 under mask {0} and ~0 under {0, 1}.
  4. the k-hot z is sqrt(d)-scaled and differs from the goal z on a strict subset.
  5. the option round-trips through load_config / make_config and build validates it.
  6. with ONE block the k-hot path is the goal path, bit for bit.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.density_fb import (
    DensityFBCritic, _delta_phi, _delta_phi_masked, _khot_bump, _raw_block_cols,
)
from impls.utils.config_utils import load_config
from tests.impls.test_density_fb_reward_index import (
    B, D, OBS, _agent, _batch, _built_full, _ex, _net_from, _perturbed,
)

TWO = "19:22:0.4,28:31:0.4"       # cube-double-like: two 3-wide blocks inside OBS=40
ONE = "19:22:0.4"


def _setup(seed=0, **over):
    obs, act = _ex()
    over.setdefault("measure_density_z_actor_reward", "kernel")
    critic, params, targets = _built_full(**over)
    params = _perturbed(params)
    batch = _batch()
    cond = critic.sample_cond(_net_from(params), batch, jax.random.PRNGKey(seed), target=targets)
    return critic, params, targets, batch, cond, obs, act


def _reference_z_today(critic, params, cond, batch, rng):
    """The pre-change actor-leg z, written out: index at _delta_phi(actor_goal, u) then the
    sphere mix `_z_mix_masked`."""
    net = _net_from(params)
    r = _delta_phi(batch["actor_goals"], cond["_density_u_raw"], critic.blocks)
    z_g = critic._z_index(critic._zenc(net), critic._zstats(net), cond["_density_u_sc"], r)
    r_a, _r_v = jax.random.split(rng)
    return critic._z_mix_masked(z_g, r_a)


# ==========================================================================================
# 1. default OFF is bit-identical
# ==========================================================================================
def test_khot_default_is_zero_and_bit_identical_in_sample_cond():
    critic0, params, targets, batch, cond0, obs, act = _setup(measure_density_z_rand_frac=0.5)
    assert critic0.z_khot_frac == 0.0
    critic1, _p, _t = _built_full(measure_density_z_actor_reward="kernel",
                                  measure_density_z_rand_frac=0.5,
                                  measure_density_z_khot_frac=0.0)
    cond1 = critic1.sample_cond(_net_from(params), batch, jax.random.PRNGKey(0), target=targets)
    assert set(cond0) == set(cond1)
    for k in cond0:
        np.testing.assert_array_equal(np.asarray(cond0[k]), np.asarray(cond1[k]), err_msg=k)
    # and both are today's formula
    z_ref, hs_ref = _reference_z_today(critic0, params, cond0, batch, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(cond1["z"]), np.asarray(z_ref))
    np.testing.assert_array_equal(np.asarray(cond1["_z_hindsight"]), np.asarray(hs_ref))
    assert "_z_khot_mask" not in cond1 and "_z_khot" not in cond1
    q0, qs0 = critic0.q_fn(params, obs, cond0, act)
    q1, qs1 = critic1.q_fn(params, obs, cond1, act)
    np.testing.assert_array_equal(np.asarray(q0), np.asarray(q1))
    np.testing.assert_array_equal(np.asarray(qs0), np.asarray(qs1))


def test_khot_zero_leaves_actor_loss_and_stage_loss_bit_identical():
    agent0, _ = _agent(measure_density_z_actor_reward="kernel", measure_density_z_rand_frac=0.5)
    agent1, _ = _agent(measure_density_z_actor_reward="kernel", measure_density_z_rand_frac=0.5,
                       measure_density_z_khot_frac=0.0)
    _u0, info0 = agent0.update(_batch())
    _u1, info1 = agent1.update(_batch())
    assert set(info0) == set(info1)
    assert "density_z_khot_frac_effective" not in info0
    for k in info0:
        np.testing.assert_array_equal(np.asarray(info0[k]), np.asarray(info1[k]), err_msg=k)
    assert np.isfinite(float(info0["actor_loss"])) and np.isfinite(float(info0["density_ce"]))


# ==========================================================================================
# 2./3. the mask and the exact reward
# ==========================================================================================
def test_khot_two_block_mask_and_reward_at_the_goal():
    critic, params, targets, batch, cond, obs, act = _setup(
        measure_hit_spec=TWO, measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    assert len(critic.blocks) == 2
    mask = np.asarray(cond["_z_khot_mask"])
    assert mask.shape == (B, 2)
    assert set(np.unique(mask)) <= {0.0, 1.0}
    sizes = mask.sum(-1)
    assert set(np.unique(sizes)) == {1.0, 2.0}, sizes
    assert np.all(np.asarray(cond["_z_khot"]) == 1.0)
    assert np.all(np.asarray(cond["_z_hindsight"]) == 0.0), "k-hot rows are not goal rows"
    # the exact reward on a candidate EQUAL to the goal is 1 whatever the mask
    goals = cond["goal"]
    cand = _raw_block_cols(goals, critic.blocks)                          # [B, d_c]
    bump = np.asarray(_khot_bump(goals, cand, critic.blocks, cond["_z_khot_mask"]))
    np.testing.assert_allclose(np.diagonal(bump), 1.0, rtol=0, atol=1e-12)
    # and what the actor reads on a k-hot row under 'kernel' is that reward, normalized
    # over the candidate set exactly as _delta_phi normalizes the goal rows
    kg = np.asarray(critic.z_actor_reward_rows({**cond, "_density_u_raw": jnp.asarray(cand)}))
    np.testing.assert_allclose(kg, bump / bump.sum(-1, keepdims=True), rtol=1e-6, atol=1e-9)
    assert np.all(np.argmax(kg, -1) == np.arange(B))


def test_khot_reward_on_a_candidate_matching_only_block_zero():
    critic, _p, _t = _built_full(measure_hit_spec=TWO, measure_density_z_actor_reward="kernel",
                                 measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    (a0, b0, e0), (a1, b1, e1) = critic.blocks
    goal = jnp.asarray(np.random.default_rng(5).standard_normal((1, OBS)))
    cand = np.asarray(_raw_block_cols(goal, critic.blocks)).copy()      # [1, 6]
    cand[:, b0 - a0:] += 10.0 * e1                                        # block 1 far away
    cand = jnp.asarray(cand)
    r_only0 = float(_khot_bump(goal, cand, critic.blocks, jnp.asarray([[1.0, 0.0]]))[0, 0])
    r_both = float(_khot_bump(goal, cand, critic.blocks, jnp.asarray([[1.0, 1.0]]))[0, 0])
    r_only1 = float(_khot_bump(goal, cand, critic.blocks, jnp.asarray([[0.0, 1.0]]))[0, 0])
    assert r_only0 == pytest.approx(1.0, abs=1e-12)
    assert r_both < 1e-30 and r_only1 < 1e-30
    # the full mask IS _delta_phi's kernel (normalized over the candidate set)
    cands = jnp.asarray(np.random.default_rng(6).standard_normal((7, 6)))
    full = jnp.ones((1, 2))
    np.testing.assert_allclose(np.asarray(_delta_phi_masked(goal, cands, critic.blocks, full)),
                               np.asarray(_delta_phi(goal, cands, critic.blocks)),
                               rtol=1e-12, atol=1e-15)


def test_khot_mixes_rowwise_by_the_khot_indicator_under_kernel():
    """Hand-built indicators: goal rows read _delta_phi, k-hot rows the masked kernel,
    sphere rows the latent_centered proxy, row by row."""
    critic, params, targets, batch, cond, obs, act = _setup(
        measure_hit_spec=TWO, measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    critic_c, _p, _t = _built_full(measure_hit_spec=TWO,
                                   measure_density_z_actor_reward="latent_centered")
    kind = np.arange(B) % 3                                              # 0 goal, 1 khot, 2 sphere
    hs = jnp.asarray((kind == 0).astype(float))[:, None]
    kh = jnp.asarray((kind == 1).astype(float))[:, None]
    mixed = {**cond, "_z_hindsight": hs, "_z_khot": kh}
    kg = np.asarray(critic.z_actor_reward_rows(mixed))
    u = cond["_density_u_raw"]
    kg_goal = np.asarray(_delta_phi(cond["goal"], u, critic.blocks))
    kg_khot = np.asarray(_delta_phi_masked(cond["goal"], u, critic.blocks, cond["_z_khot_mask"]))
    kg_proxy = np.asarray(critic_c.z_actor_reward_rows(cond))
    np.testing.assert_array_equal(kg[kind == 0], kg_goal[kind == 0])
    np.testing.assert_array_equal(kg[kind == 1], kg_khot[kind == 1])
    np.testing.assert_array_equal(kg[kind == 2], kg_proxy[kind == 2])
    # under the proxy options k-hot rows use the proxy like sphere rows do
    for opt in ("latent", "latent_centered"):
        critic_p, _p2, _t2 = _built_full(measure_hit_spec=TWO, measure_density_z_actor_reward=opt,
                                         measure_density_z_khot_frac=1.0,
                                         measure_density_z_rand_frac=0.0)
        basis = cond["_density_zenc_u" if opt == "latent_centered" else "_density_tb_u"]
        ref = np.asarray(jnp.einsum('md,bd->bm', basis, cond["z"]) / np.sqrt(float(D)))
        np.testing.assert_array_equal(np.asarray(critic_p.z_actor_reward_rows(cond)), ref)


# ==========================================================================================
# 4. the k-hot z
# ==========================================================================================
def test_khot_z_is_unit_scaled_and_differs_from_the_goal_z_on_strict_subsets():
    critic, params, targets, batch, cond, obs, act = _setup(
        measure_hit_spec=TWO, measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    z = np.asarray(cond["z"])
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1), np.sqrt(float(D)), rtol=1e-6)
    net = _net_from(params)
    r_goal = _delta_phi(cond["goal"], cond["_density_u_raw"], critic.blocks)
    z_goal = np.asarray(critic.z_from_reward_samples(net, cond["_density_u_sc"], r_goal))
    sizes = np.asarray(cond["_z_khot_mask"]).sum(-1)
    strict, full = sizes < 2, sizes == 2
    assert strict.any() and full.any()
    for i in np.flatnonzero(strict):
        assert not np.allclose(z[i], z_goal[i], atol=1e-3), f"row {i}: strict subset z == goal z"
    np.testing.assert_allclose(z[full], z_goal[full], rtol=1e-6, atol=1e-8)
    # the value leg carries the same mix: sqrt(d)-scaled and finite
    zv = np.asarray(cond["_cvec_vgoal"])
    np.testing.assert_allclose(np.linalg.norm(zv, axis=-1), np.sqrt(float(D)), rtol=1e-6)


def test_khot_row_precedence_takes_the_sphere_fraction_first():
    critic, _p, _t = _built_full(measure_hit_spec=TWO, measure_density_z_actor_reward="kernel",
                                 measure_density_z_khot_frac=0.3, measure_density_z_rand_frac=0.5)
    n = 40000
    is_k, is_s = critic._z_row_kinds(jax.random.PRNGKey(1), n)
    fk, fs = float(jnp.mean(is_k)), float(jnp.mean(is_s))
    assert abs(fk - 0.3) < 0.02 and abs(fs - 0.2) < 0.02 and not bool(jnp.any(is_k & is_s))
    # khot beyond rand_frac eats into the goal rows: no sphere rows are left
    critic2, _p, _t = _built_full(measure_hit_spec=TWO, measure_density_z_actor_reward="kernel",
                                  measure_density_z_khot_frac=0.8, measure_density_z_rand_frac=0.5)
    is_k, is_s = critic2._z_row_kinds(jax.random.PRNGKey(1), n)
    assert abs(float(jnp.mean(is_k)) - 0.8) < 0.02 and float(jnp.mean(is_s)) == 0.0
    # subset sizes are uniform on 1..n_blocks
    mask = critic2._z_khot_masks(jax.random.PRNGKey(2), n)
    sizes = np.asarray(mask).sum(-1)
    assert abs(np.mean(sizes == 1) - 0.5) < 0.02 and set(np.unique(sizes)) == {1.0, 2.0}
    # and within a size, uniform over blocks
    one = np.asarray(mask)[sizes == 1]
    assert abs(one[:, 0].mean() - 0.5) < 0.03


def test_khot_info_entries_from_a_real_update():
    agent, _ = _agent(measure_hit_spec=TWO, measure_density_z_actor_reward="kernel",
                      measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    _u, info = agent.update(_batch())
    assert float(info["density_z_khot_frac_effective"]) == 1.0
    assert 1.0 <= float(info["density_z_khot_mean_size"]) <= 2.0
    # every row used an exact reward: the goal rows' fraction now counts k-hot rows too
    assert float(info["density_z_kernel_frac"]) == 1.0
    assert np.isfinite(float(info["actor_loss"])) and np.isfinite(float(info["density_ce"]))


# ==========================================================================================
# 5. config round trip and validation
# ==========================================================================================
def test_khot_round_trips_through_the_config_loaders():
    import yaml
    from env.config import make_config
    ov = ["measure_density_z_actor_reward=kernel", "measure_density_z_khot_frac=0.5",
          "measure_density_z_rand_frac=0.0"]
    cfg = load_config("density_fb_z", "flowbc", "cube_double", overrides=ov)
    assert cfg["measure_density_z_khot_frac"] == 0.5 and cfg["measure_density_z"] is True
    cfg_main = make_config("density_fb_z", "flowbc", "cube_double", overrides=ov)
    assert cfg_main["measure_density_z_khot_frac"] == 0.5
    critic, _p, _t = DensityFBCritic.build(
        *_ex(), {**cfg_main, "seed": 0, "batch_size": B, "measure_density_dim": D,
                 "value_hidden_dims": [32, 32], "measure_density_z_eval_pool_m": 8})
    assert critic.z_khot_frac == 0.5 and critic.z_rand_frac == 0.0
    assert yaml.safe_load(open("impls/configs/critic/density_fb_z.yaml"))[
        "measure_density_z_khot_frac"] == 0.0
    assert "measure_density_z_khot_frac" not in yaml.safe_load(
        open("impls/configs/critic/density_fb.yaml"))


def test_khot_frac_is_validated_at_build():
    with pytest.raises(AssertionError):
        _built_full(measure_density_z_khot_frac=1.5)
    with pytest.raises(AssertionError):
        _built_full(measure_density_z_khot_frac=-0.1)
    # off the index the knob is not read at all
    obs, act = _ex()
    from tests.impls.test_density_fb_reward_index import _cfg
    critic, _p, _t = DensityFBCritic.build(obs, act, _cfg(False, measure_density_z_khot_frac=1.5))
    assert critic.z_on is False


# ==========================================================================================
# 6. one block: the k-hot path IS the goal path
# ==========================================================================================
def test_one_block_khot_equals_the_goal_path_bit_identically():
    critic_k, params, targets, batch, cond_k, obs, act = _setup(
        measure_hit_spec=ONE, measure_density_z_khot_frac=1.0, measure_density_z_rand_frac=0.0)
    critic_g, _p, _t = _built_full(measure_hit_spec=ONE, measure_density_z_actor_reward="kernel",
                                   measure_density_z_rand_frac=0.0)
    cond_g = critic_g.sample_cond(_net_from(params), batch, jax.random.PRNGKey(0), target=targets)
    assert np.all(np.asarray(cond_k["_z_khot_mask"]) == 1.0)
    np.testing.assert_array_equal(np.asarray(cond_k["z"]), np.asarray(cond_g["z"]))
    np.testing.assert_array_equal(np.asarray(cond_k["_cvec_vgoal"]), np.asarray(cond_g["_cvec_vgoal"]))
    kg_k = critic_k.z_actor_reward_rows(cond_k)
    kg_g = critic_g.z_actor_reward_rows(cond_g)
    np.testing.assert_array_equal(np.asarray(kg_k), np.asarray(kg_g))
    q_k, _ = critic_k.q_fn(params, obs, cond_k, act)
    q_g, _ = critic_g.q_fn(params, obs, cond_g, act)
    np.testing.assert_array_equal(np.asarray(q_k), np.asarray(q_g))
