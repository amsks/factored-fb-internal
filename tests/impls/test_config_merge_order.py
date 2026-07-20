"""Pin the config merge-order semantics.

load_config merges (domain, critic, actor) with the actor winning ties. That makes
lr_actor actor-owned for every pair. Before flowbc.yaml declared its own lr_actor,
fb x flowbc was the lone pair inheriting it from critic/fb.yaml, so which file owned
the key depended on whether the actor yaml happened to declare it. These tests lock
the resolved values so that stays deliberate rather than accidental.
"""
import pytest

from impls.utils.config_utils import load_config

# (critic, actor) -> effective top-level lr_actor
EXPECTED_LR_ACTOR = {
    ("fb", "flowbc"): 1.0e-4,
    ("fb", "ddpgbc"): 3.0e-4,
    ("fb", "fql"): 3.0e-4,
    ("gciql", "ddpgbc"): 3.0e-4,
    ("gciql", "awr"): 3.0e-4,
    ("gciql", "fql"): 3.0e-4,
}


@pytest.mark.parametrize("pair,expected", sorted(EXPECTED_LR_ACTOR.items()))
def test_effective_lr_actor(pair, expected):
    critic, actor = pair
    cfg = load_config(critic, actor, "cube_single")
    assert cfg["lr_actor"] == pytest.approx(expected)


def test_every_actor_declares_its_own_lr_actor():
    """Ownership must not depend on the critic. Each actor yaml supplies lr_actor, so
    swapping the critic never silently changes the actor's learning rate."""
    for actor in ("flowbc", "ddpgbc", "awr", "fql"):
        critic = "fb" if actor == "flowbc" else "gciql"
        cfg = load_config(critic, actor, "cube_single")
        assert cfg["lr_actor"] is not None
    # fb x flowbc must be unaffected by flowbc.yaml gaining an explicit lr_actor:
    # PR1's launch value was 1e-4 and the FB path is numerically locked.
    assert load_config("fb", "flowbc", "cube_single")["lr_actor"] == pytest.approx(1.0e-4)


def test_critic_lr_does_not_leak_into_actor_lr():
    """fb.yaml sets lr_f/lr_b/lr_actor; gciql.yaml sets lr. Neither may be mistaken
    for the other by the loader."""
    fb = load_config("fb", "flowbc", "cube_single")
    gc = load_config("gciql", "fql", "cube_single")
    assert fb["lr_f"] == pytest.approx(1.0e-4)
    assert fb["lr_b"] == pytest.approx(1.0e-4)
    assert gc["lr"] == pytest.approx(3.0e-4)
    assert "lr" not in fb or fb.get("lr") != fb["lr_f"]
