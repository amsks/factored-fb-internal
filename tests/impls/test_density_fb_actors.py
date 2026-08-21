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
