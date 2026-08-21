"""density_fb composes with every actor family (workstream 0, tasks 3-5).

Before this workstream only FlowBCActor emitted the value-goal bootstrap draw
density_fb requires, so density_fb x {ddpgbc, awr, fql} raised KeyError from
stage_loss. These are the tests for each actor gaining it.
"""
import jax
import numpy as np
import pytest

from tests.impls.gen_boot_contract_fixture import pair_statics


def _check(actor):
    boot, info = pair_statics("density_fb", actor)
    assert "next_action_vgoal" in boot, f"{actor} did not emit the value-goal draw"
    assert not np.allclose(boot["next_action"], boot["next_action_vgoal"]), (
        f"{actor}'s value-goal draw equals its actor-goal draw -- it was almost "
        f"certainly conditioned on the wrong goal"
    )
    assert np.isfinite(float(np.asarray(info["density_ce"]))), "density_ce is not finite"
    assert np.isfinite(float(np.asarray(info["actor_loss"]))), "actor_loss is not finite"


def test_density_fb_composes_with_ddpgbc():
    _check("ddpgbc")


def test_density_fb_composes_with_awr():
    _check("awr")


def test_density_fb_composes_with_fql():
    _check("fql")


def test_fql_reuses_its_bootstrap_noise():
    """fql is the only one of the three that draws randomness. If the value-goal draw
    split the key again instead of reusing na_noise, `next_action` -- and every existing
    x fql cell with it -- would move. The bit-identity gate catches that too; this names
    the cause directly."""
    from tests.impls.gen_boot_contract_fixture import FIXTURE
    baseline = dict(np.load(FIXTURE, allow_pickle=True))
    boot, _ = pair_statics("gciql", "fql")
    for key in ("next_action", "flow_x0", "flow_t", "actor_noise"):
        assert np.array_equal(boot[key], baseline[f"boot|gciql|fql|{key}"]), (
            f"fql's {key} moved -- the value-goal draw spent a split it should have reused"
        )


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc", "awr", "fql", "ddpgbc_density"])
def test_density_fb_runs_a_real_update_with_every_actor(actor):
    """The point of workstream 0: density_fb had one usable actor and now has five.
    compute_static only evaluates losses -- this runs a real gradient step."""
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.utils.config_utils import load_config
    from tests.impls.gen_boot_contract_fixture import make_batch, B, OBS, ACT

    cfg = load_config("density_fb", actor, "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create("density_fb", actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    updated, info = agent.update(make_batch())
    assert np.isfinite(float(np.asarray(info["density_ce"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    # the update actually moved the density towers
    before = agent.net["density_f"].params
    after = updated.net["density_f"].params
    leaves_changed = [not np.array_equal(np.asarray(a), np.asarray(b))
                      for a, b in zip(jax.tree_util.tree_leaves(before),
                                      jax.tree_util.tree_leaves(after))]
    assert any(leaves_changed), "density_f params did not move"


def test_density_fb_reads_the_key_its_declaration_names():
    """Ties Task 2's declaration to Tasks 3-5's emission: perturbing the key BOOT_COND
    names must change density_ce, and perturbing the other one must not."""
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.critics.base import boot_key
    from impls.critics.density_fb import DensityFBCritic
    from impls.utils.config_utils import load_config
    from tests.impls.gen_boot_contract_fixture import make_batch, B, OBS, ACT

    assert boot_key(DensityFBCritic) == "next_action_vgoal"

    cfg = load_config("density_fb", "ddpgbc", "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create("density_fb", "ddpgbc", 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    batch = make_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))

    def density_ce(b):
        return float(np.asarray(agent.compute_static(batch, cond, b)["density_ce"]))

    base = density_ce(boot)
    junk = jnp.zeros_like(boot["next_action"])
    assert density_ce({**boot, "next_action_vgoal": junk}) != base, \
        "density_fb must READ next_action_vgoal -- the key its BOOT_COND names"
    assert density_ce({**boot, "next_action": junk}) == base, \
        "density_fb must IGNORE next_action, the actor-goal draw it explicitly rejects"
