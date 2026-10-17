import pytest

from impls.actors.base import Actor
from impls.critics.base import ComboError, Critic, check_combo


class _ZCritic:
    PROVIDES_COND = frozenset({"z"})


class _NeedsGoal:
    REQUIRES_COND = frozenset({"goal"})


class _NeedsNothing:
    REQUIRES_COND = frozenset()


def test_check_combo_rejects_an_actor_requiring_a_cond_key_the_critic_lacks():
    with pytest.raises(ComboError, match="requires cond key"):
        check_combo(_ZCritic(), _NeedsGoal())


def test_check_combo_accepts_valid_pairs():
    check_combo(_ZCritic(), _NeedsNothing())


def test_protocols_are_runtime_checkable_shape():
    for attr in ("GROUPS", "STAGES", "TARGETS", "PROVIDES_COND", "INFER_COND_MARGINAL"):
        assert attr in Critic.__annotations__ or hasattr(Critic, attr)
    for attr in ("GROUPS", "REQUIRES_COND"):
        assert attr in Actor.__annotations__ or hasattr(Actor, attr)
