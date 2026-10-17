"""Every (critic config, actor) pair trains a few steps on cube-single with a finite actor loss."""

import jax
import jax.numpy as jnp
import pytest

from impls.agents import ACTORS, Agent
from impls.utils.config_utils import load_config
from impls.utils.datasets import make_dataset
from impls.utils.env_utils import make_env_and_datasets

# One config per critic class.
CRITIC_CONFIGS = ("fb", "density_fb", "hilp", "rldp")
PAIRS = [(c, a) for c in CRITIC_CONFIGS for a in ACTORS]


@pytest.fixture(autouse=True)
def _drop_compilation_caches():
    """Clear the jax compilation cache between pairs."""
    yield
    jax.clear_caches()


@pytest.mark.parametrize("critic,actor", PAIRS)
def test_pair_trains(critic, actor):
    cfg = load_config(critic, actor, "cube_single")
    cfg["batch_size"] = 8
    _, train_ds, _ = make_env_and_datasets(cfg["env_name"])
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(
        critic,
        actor,
        0,
        jnp.asarray(ex["observations"]),
        jnp.asarray(ex["actions"]),
        cfg,
    )
    for _ in range(3):
        agent, info = agent.update(train_ds.sample(8))
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()
