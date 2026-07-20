"""Phase 0 gate: every registered pair either trains from a composed config on real data,
or is rejected as a structurally invalid combo — with a fixed, exact expectation for both
counts, so a broken YAML turns this gate red rather than quietly being skipped away.

Two of the eight critic x actor pairs are invalid by design and MUST be rejected via
ComboError (never a skip, never a ConfigError):
  - fb x awr        -- AWRActor.NEEDS_V is True, FBCritic.PROVIDES_V is False.
    Rejected by impls/critics/base.py:check_combo (raises ComboError) inside
    Agent.create, which runs check_combo right after critic.build and *before*
    actor.build (impls/agents/__init__.py) -- so this never reaches actor construction.
  - gciql x flowbc  -- FlowBCActor requires cond["z"] (REQUIRES_COND={"z"}), but
    GCIQLCritic's sample_cond/PROVIDES_COND never produces a "z" key. The composed
    config loads fine (actor/flowbc.yaml defines its own top-level lr_actor, so gciql's
    REQUIRED_KEYS is satisfied); check_combo is what rejects this pair, again before
    actor.build ever runs.

The other six pairs (fb x {flowbc,ddpgbc,fql}, gciql x {ddpgbc,awr,fql}) must construct
and train for a few steps with a finite actor_loss.

A ConfigError from load_config() for ANY pair is a hard failure here, not a skip: it means
a config YAML is broken (e.g. a required key like critic/fb.yaml's `ortho_coef` went
missing), and this gate exists to catch exactly that.
"""
import jax.numpy as jnp
import pytest

from impls.agents import Agent, CRITICS, ACTORS
from impls.critics.base import ComboError
from impls.utils.config_utils import load_config
from impls.utils.datasets import make_dataset
from impls.utils.env_utils import make_env_and_datasets

PAIRS = [(c, a) for c in CRITICS for a in ACTORS]

# The only two pairs that are structurally invalid by design. Every other pair in PAIRS
# must construct and train; if this set and PAIRS ever drift (e.g. a new critic/actor is
# registered), the assertions below will fail loudly rather than silently skip.
EXPECTED_COMBO_ERRORS = {("fb", "awr"), ("gciql", "flowbc")}


@pytest.mark.parametrize("critic,actor", PAIRS)
def test_pair_trains_or_is_rejected(critic, actor):
    # No try/except ConfigError here: a broken YAML must fail this test, not skip it.
    cfg = load_config(critic, actor, "cube_single")
    cfg["batch_size"] = 8
    _, train_ds, _ = make_env_and_datasets(cfg["env_name"])
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)

    if (critic, actor) in EXPECTED_COMBO_ERRORS:
        with pytest.raises(ComboError):
            Agent.create(critic, actor, 0, jnp.asarray(ex["observations"]),
                         jnp.asarray(ex["actions"]), cfg)
        return

    agent = Agent.create(critic, actor, 0, jnp.asarray(ex["observations"]),
                         jnp.asarray(ex["actions"]), cfg)
    for _ in range(3):
        agent, info = agent.update(train_ds.sample(8))
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()


def test_expected_combo_errors_cover_exactly_two_of_eight_pairs():
    """Pins the gate's exact floor: 6 pairs train, 2 are rejected. If PAIRS grows without
    updating EXPECTED_COMBO_ERRORS, this is the assertion that catches it."""
    assert len(PAIRS) == 8
    assert EXPECTED_COMBO_ERRORS <= set(PAIRS)
    assert len(EXPECTED_COMBO_ERRORS) == 2
    assert len(set(PAIRS) - EXPECTED_COMBO_ERRORS) == 6
