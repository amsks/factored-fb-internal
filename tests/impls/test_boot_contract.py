"""Reference check of the bootstrap actions and TD statistics for every critic x actor pair.

Arrays must match tests/impls/fixtures/boot_contract_reference.npz exactly. The fixture is
CPU- and x64-specific; run with JAX_PLATFORMS=cpu.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import CRITICS, Agent
from impls.utils.config_utils import load_config

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "boot_contract_reference.npz"
)
# One config per critic class.
CRITIC_CONFIGS = ("density_fb", "fb", "hilp", "rldp")
# The fixture was recorded when the hilp config was named hilp_tdjc; its keys keep that name.
FIXTURE_KEY = {"hilp": "hilp_tdjc"}
ACTOR_FAMILIES = ("flowbc", "ddpgbc")
# Recorded: every bootstrap draw plus these info scalars.
INFO_KEYS = (
    "actor_loss",
    "q_loss",
    "fb_loss",
    "density_ce",
    "sf_loss",
    "hilp_value_loss",
    "bc_flow_loss",
    "bc_loss",
)
B, OBS, ACT = 8, 40, 5


def make_batch(seed=1):
    r = np.random.default_rng(seed)
    # value_goals and actor_goals differ so the two bootstrap draws are distinguishable.
    g_value = jnp.asarray(r.standard_normal((B, OBS)))
    g_actor = jnp.asarray(r.standard_normal((B, OBS)))
    # Separate rngs so critic-specific keys do not shift the shared arrays.
    rh = np.random.default_rng(seed + 9000)
    return dict(
        observations=jnp.asarray(r.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(r.standard_normal((B, OBS))),
        next_actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        initial_observations=jnp.asarray(r.standard_normal((B, OBS))),
        value_goals=g_value,
        actor_goals=g_actor,
        # hilp: one geometric hindsight future, [B, obs].
        hilp_future_observations=jnp.asarray(
            np.random.default_rng(seed + 4100).standard_normal((B, OBS))
        ),
        # rldp: the h-step window, [B, h-1, ...].
        future_observations=jnp.asarray(rh.standard_normal((B, 4, OBS))),
        future_actions=jnp.asarray(np.tanh(rh.standard_normal((B, 4, ACT)))),
        index=jnp.asarray([5, 11, 23, 4, 9, 1, 6, 2][:B] * (B // 8 + 1))[:B],
        rewards=jnp.asarray(r.integers(0, 2, size=B) - 1.0),
        masks=jnp.asarray(r.integers(0, 2, size=B).astype(float)),
    )


def pair_statics(critic, actor, overrides=None):
    """Return (boot, info) for one critic x actor pair at fixed seeds."""
    cfg = load_config(critic, actor, "cube_single")
    cfg["batch_size"] = B
    # The fixture predates HILP's running-covariance whitening.
    cfg["whiten_running_cov"] = False
    cfg.update(overrides or {})
    agent = Agent.create(
        critic, actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg
    )
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
                out[f"boot|{FIXTURE_KEY.get(critic, critic)}|{actor}|{k}"] = np.asarray(
                    v
                )
            for k in INFO_KEYS:
                if k in info:
                    out[f"info|{FIXTURE_KEY.get(critic, critic)}|{actor}|{k}"] = (
                        np.asarray(jnp.asarray(info[k]).mean())
                    )
    return out


if __name__ == "__main__":
    # Regenerate the fixture:
    #   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD python tests/impls/test_boot_contract.py
    jax.config.update("jax_enable_x64", True)
    data = collect()
    np.savez(FIXTURE, **data)
    print(f"wrote {FIXTURE}: {len(data)} arrays")


@pytest.fixture(scope="module", autouse=True)
def _require_cpu():
    backend = jax.default_backend()
    if backend != "cpu":
        pytest.fail(
            f"boot-contract gate ran on {backend!r}, but the baseline fixture is "
            f"CPU. Re-run with JAX_PLATFORMS=cpu."
        )


@pytest.fixture(scope="module")
def baseline():
    return dict(np.load(FIXTURE, allow_pickle=True))


@pytest.fixture(scope="module")
def current():
    return collect()


def test_every_baseline_pair_still_computes(baseline, current):
    missing = sorted(set(baseline) - set(current))
    assert not missing, (
        f"{len(missing)} baseline array(s) can no longer be computed; first few: {missing[:8]}"
    )


def test_every_baseline_array_is_byte_identical(baseline, current):
    moved = sorted(
        k
        for k in baseline
        if k in current and not np.array_equal(baseline[k], current[k])
    )
    assert not moved, f"{len(moved)} array(s) moved; first few: {moved[:8]}"


def test_value_goal_draw_goes_only_to_the_critic_that_publishes_it(current):
    """Only density_fb publishes _cvec_vgoal, and its two bootstrap actions differ."""
    a = current["boot|density_fb|flowbc|next_action"]
    v = current["boot|density_fb|flowbc|next_action_vgoal"]
    assert not np.allclose(a, v)
    leaked = sorted(
        k
        for k in current
        if k.endswith("next_action_vgoal") and "|density_fb|" not in k
    )
    assert not leaked, (
        f"critics that do not publish _cvec_vgoal received the draw: {leaked}"
    )


def test_every_critic_declares_a_known_boot_cond():
    from impls.critics.base import BOOT_KEY

    for name, critic in sorted(CRITICS.items()):
        cond = getattr(critic, "BOOT_COND", "actor_goal")
        assert cond in BOOT_KEY, f"{name} declares BOOT_COND={cond!r}"


def test_density_fb_is_the_only_value_goal_critic():
    """density_fb is the only critic with BOOT_COND value_goal."""
    value_goal = {
        n
        for n, c in CRITICS.items()
        if getattr(c, "BOOT_COND", "actor_goal") == "value_goal"
    }
    assert value_goal == {"density_fb"}


def test_boot_key_maps_the_declaration_to_the_dict_key():
    from impls.critics.base import boot_key
    from impls.critics.density_fb import DensityFBCritic
    from impls.critics.fb import FBCritic

    assert boot_key(DensityFBCritic) == "next_action_vgoal"
    assert boot_key(FBCritic) == "next_action"


def test_check_combo_rejects_an_unknown_boot_cond():
    from impls.actors.ddpgbc import DDPGBCActor
    from impls.critics.base import ComboError, check_combo

    class _Bogus:
        BOOT_COND = "somewhere_else"
        PROVIDES_COND = frozenset({"goal"})

    with pytest.raises(ComboError, match="BOOT_COND"):
        check_combo(_Bogus, DDPGBCActor)


def test_check_combo_rejects_value_goal_the_critic_never_produces():
    """check_combo rejects a value_goal BOOT_COND the critic does not provide."""
    from impls.actors.ddpgbc import DDPGBCActor
    from impls.critics.base import ComboError, check_combo

    class _Inconsistent:
        BOOT_COND = "value_goal"
        PROVIDES_COND = frozenset({"goal"})  # no "value_goal"

    with pytest.raises(ComboError, match="value_goal"):
        check_combo(_Inconsistent, DDPGBCActor)


def test_hilp_running_cov_whitening_computes():
    """HILP with running-covariance whitening gives finite bootstrap and losses."""
    boot, info = pair_statics("hilp", "flowbc", {"whiten_running_cov": True})
    for v in list(boot.values()) + list(info.values()):
        assert np.all(np.isfinite(np.asarray(v)))
