import os

import jax.numpy as jnp
import pytest

from impls.utils.config_utils import CONFIG_ROOT, load_config
import impls.main as m

CRITIC_CONFIGS = sorted(f[:-5] for f in os.listdir(os.path.join(CONFIG_ROOT, "critic"))
                        if f.endswith(".yaml"))
ACTOR_CONFIGS = sorted(f[:-5] for f in os.listdir(os.path.join(CONFIG_ROOT, "actor"))
                       if f.endswith(".yaml"))


def test_main_has_no_get_config():
    """impls/main.py defines no get_config()."""
    assert not hasattr(m, "get_config")


def test_every_config_name_is_registered():
    from impls.agents import ACTORS, CRITIC_ALIASES, CRITICS
    assert set(CRITIC_CONFIGS) <= set(CRITICS) | set(CRITIC_ALIASES)
    assert set(ACTOR_CONFIGS) == set(ACTORS)


@pytest.mark.parametrize("critic", CRITIC_CONFIGS)
@pytest.mark.parametrize("actor", ACTOR_CONFIGS)
def test_every_config_pair_builds_from_the_real_yamls(critic, actor):
    """Every (critic yaml, actor yaml) pair builds an Agent."""
    from impls.agents import Agent

    cfg = load_config(critic, actor, "cube_single")
    agent = Agent.create(critic, actor, 0, jnp.zeros((2, 28)), jnp.zeros((2, 5)), cfg)
    assert "actor" in agent.net
