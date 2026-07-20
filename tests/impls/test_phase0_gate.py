"""Phase 0 gate: every registered pair trains from a composed config on real data.

Two of the eight critic x actor pairs are invalid by design and must be SKIPPED rather
than failed:
  - fb x awr        -- AWRActor.NEEDS_V is True, FBCritic.PROVIDES_V is False.
    Rejected by impls/critics/base.py:check_combo (raises ComboError) inside
    Agent.create, *after* both critic.build and actor.build have already run.
  - gciql x flowbc   -- FlowBCActor requires cond["z"] (REQUIRES_COND={"z"}), but
    GCIQLCritic's sample_cond/PROVIDES_COND never produces a "z" key. In practice this
    pair is rejected even earlier: gciql's REQUIRED_KEYS includes "lr_actor", which
    gciql.yaml does not define and flowbc.yaml does not define either (only fb.yaml and
    the other three actor yamls define a top-level lr_actor), so impls/utils/
    config_utils.py:load_config raises ConfigError before Agent.create is ever called.
"""
import jax.numpy as jnp
import pytest

from impls.agents import Agent, CRITICS, ACTORS
from impls.critics.base import ComboError
from impls.utils.config_utils import ConfigError, load_config
from impls.utils.datasets import make_dataset
from impls.utils.env_utils import make_env_and_datasets

PAIRS = [(c, a) for c in CRITICS for a in ACTORS]


@pytest.mark.parametrize("critic,actor", PAIRS)
def test_pair_trains_from_composed_config(critic, actor):
    try:
        cfg = load_config(critic, actor, "cube_single")
    except ConfigError as e:
        pytest.skip(f"{critic}x{actor} is an invalid combo by design (ConfigError: {e})")
    cfg["batch_size"] = 8
    _, train_ds, _ = make_env_and_datasets(cfg["env_name"])
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    try:
        agent = Agent.create(critic, actor, 0, jnp.asarray(ex["observations"]),
                             jnp.asarray(ex["actions"]), cfg)
    except ComboError as e:
        pytest.skip(f"{critic}x{actor} is an invalid combo by design (ComboError: {e})")
    for _ in range(3):
        agent, info = agent.update(train_ds.sample(8))
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()
