"""Bit-identity gate for the bootstrap contract between critics and actors.

Each actor's `sample_bootstrap` emits `next_action` (drawn at the actor goal) and, when the
critic publishes cond["_cvec_vgoal"], `next_action_vgoal` (drawn at the value goal);
`Critic.BOOT_COND` declares which one a critic's TD target reads.

This test recomputes `sample_bootstrap` and `compute_static` for every critic x actor pair
at fixed seeds and asserts every array recorded in
tests/impls/fixtures/boot_contract_reference.npz is byte-identical. The fixture was
generated from the code before the paper-only cleanup; it is CPU- and x64-specific, so run
with JAX_PLATFORMS=cpu.
"""
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from impls.agents import Agent, CRITICS
from impls.utils.config_utils import load_config

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "boot_contract_reference.npz")
# One critic config per critic class (hilp's only yaml is hilp_tdjc).
CRITIC_CONFIGS = ("density_fb", "fb", "hilp_tdjc", "rldp")
ACTOR_FAMILIES = ("flowbc", "ddpgbc")
# Recorded arrays: every bootstrap draw, plus the info scalars named here. Diagnostics are
# left out on purpose so they can be added or dropped without regenerating the fixture.
INFO_KEYS = ("actor_loss", "q_loss", "fb_loss", "density_ce", "sf_loss", "hilp_value_loss",
             "bc_flow_loss", "bc_loss")
B, OBS, ACT = 8, 40, 5


def make_batch(seed=1):
    r = np.random.default_rng(seed)
    # value_goals and actor_goals MUST differ: if they were the same array, an actor that
    # drew its value-goal bootstrap at the actor goal would pass every test.
    g_value = jnp.asarray(r.standard_normal((B, OBS)))
    g_actor = jnp.asarray(r.standard_normal((B, OBS)))
    # Separate rngs for the critic-specific keys, so adding one never shifts the arrays
    # drawn from `r` (and every recorded loss with them).
    rh = np.random.default_rng(seed + 9000)
    return dict(
        observations=jnp.asarray(r.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(r.standard_normal((B, OBS))),
        next_actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        initial_observations=jnp.asarray(r.standard_normal((B, OBS))),
        value_goals=g_value, actor_goals=g_actor,
        # hilp: one geometric hindsight future, [B, obs].
        hilp_future_observations=jnp.asarray(
            np.random.default_rng(seed + 4100).standard_normal((B, OBS))),
        # rldp: the h-step window, [B, h-1, ...].
        future_observations=jnp.asarray(rh.standard_normal((B, 4, OBS))),
        future_actions=jnp.asarray(np.tanh(rh.standard_normal((B, 4, ACT)))),
        index=jnp.asarray([5, 11, 23, 4, 9, 1, 6, 2][:B] * (B // 8 + 1))[:B],
        rewards=jnp.asarray(r.integers(0, 2, size=B) - 1.0),
        masks=jnp.asarray(r.integers(0, 2, size=B).astype(float)),
    )


def pair_statics(critic, actor):
    """(boot dict, info dict) for one critic x actor pair, at fixed seeds."""
    cfg = load_config(critic, actor, "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create(critic, actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    batch = make_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    info = agent.compute_static(batch, cond, boot)
    return boot, info


def collect():
    out = {}
    for critic in CRITIC_CONFIGS:
        for actor in ACTOR_FAMILIES:
            boot, info = pair_statics(critic, actor)
            for k, v in boot.items():
                out[f"boot|{critic}|{actor}|{k}"] = np.asarray(v)
            for k in INFO_KEYS:
                if k in info:
                    out[f"info|{critic}|{actor}|{k}"] = np.asarray(jnp.asarray(info[k]).mean())
    return out


if __name__ == "__main__":
    # Regenerate the fixture (only when a behaviour change is intended):
    #   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD python tests/impls/test_boot_contract.py
    jax.config.update("jax_enable_x64", True)
    data = collect()
    np.savez(FIXTURE, **data)
    print(f"wrote {FIXTURE}: {len(data)} arrays")


@pytest.fixture(scope="module", autouse=True)
def _require_cpu():
    backend = jax.default_backend()
    if backend != "cpu":
        pytest.fail(f"boot-contract gate ran on {backend!r}, but the baseline fixture is "
                    f"CPU. Re-run with JAX_PLATFORMS=cpu.")


@pytest.fixture(scope="module")
def baseline():
    return dict(np.load(FIXTURE, allow_pickle=True))


@pytest.fixture(scope="module")
def current():
    return collect()


def test_every_baseline_pair_still_computes(baseline, current):
    missing = sorted(set(baseline) - set(current))
    assert not missing, (
        f"{len(missing)} baseline array(s) can no longer be computed; first few: {missing[:8]}")


def test_every_baseline_array_is_byte_identical(baseline, current):
    moved = sorted(k for k in baseline
                   if k in current and not np.array_equal(baseline[k], current[k]))
    assert not moved, f"{len(moved)} array(s) moved; first few: {moved[:8]}"


def test_value_goal_draw_goes_only_to_the_critic_that_publishes_it(current):
    """The value-goal draw is published by the critic as cond["_cvec_vgoal"]. density_fb is
    the only critic that publishes it (BOOT_COND "value_goal"). value_goals != actor_goals in
    the synthetic batch, so the two bootstrap actions must differ."""
    a = current["boot|density_fb|flowbc|next_action"]
    v = current["boot|density_fb|flowbc|next_action_vgoal"]
    assert not np.allclose(a, v)
    leaked = sorted(k for k in current
                    if k.endswith("next_action_vgoal") and "|density_fb|" not in k)
    assert not leaked, f"critics that do not publish _cvec_vgoal received the draw: {leaked}"


def test_every_critic_declares_a_known_boot_cond():
    from impls.critics.base import BOOT_KEY
    for name, critic in sorted(CRITICS.items()):
        cond = getattr(critic, "BOOT_COND", "actor_goal")
        assert cond in BOOT_KEY, f"{name} declares BOOT_COND={cond!r}"


def test_density_fb_is_the_only_value_goal_critic():
    """density_fb's forward net is F(s,a,g), so its target must bootstrap at the same g."""
    value_goal = {n for n, c in CRITICS.items()
                  if getattr(c, "BOOT_COND", "actor_goal") == "value_goal"}
    assert value_goal == {"density_fb"}


def test_boot_key_maps_the_declaration_to_the_dict_key():
    from impls.critics.base import boot_key
    from impls.critics.density_fb import DensityFBCritic
    from impls.critics.fb import FBCritic
    assert boot_key(DensityFBCritic) == "next_action_vgoal"
    assert boot_key(FBCritic) == "next_action"


def test_check_combo_rejects_an_unknown_boot_cond():
    from impls.critics.base import check_combo, ComboError
    from impls.actors.ddpgbc import DDPGBCActor

    class _Bogus:
        BOOT_COND = "somewhere_else"
        PROVIDES_COND = frozenset({"goal"})

    with pytest.raises(ComboError, match="BOOT_COND"):
        check_combo(_Bogus, DDPGBCActor)


def test_check_combo_rejects_value_goal_the_critic_never_produces():
    """A critic that declares BOOT_COND="value_goal" but whose sample_cond omits it would
    KeyError deep inside stage_loss; catch it at build time."""
    from impls.critics.base import check_combo, ComboError
    from impls.actors.ddpgbc import DDPGBCActor

    class _Inconsistent:
        BOOT_COND = "value_goal"
        PROVIDES_COND = frozenset({"goal"})     # no "value_goal"

    with pytest.raises(ComboError, match="value_goal"):
        check_combo(_Inconsistent, DDPGBCActor)
