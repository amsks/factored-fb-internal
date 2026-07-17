import pytest
from impls.critics.base import Critic, ComboError, check_combo
from impls.actors.base import Actor


class _V:      # provides V
    PROVIDES_V = True
class _NoV:    # no V
    PROVIDES_V = False
class _NeedsV:
    NEEDS_V = True
class _NoNeed:
    NEEDS_V = False


def test_check_combo_rejects_awr_without_v():
    with pytest.raises(ComboError, match="requires a V function"):
        check_combo(_NoV(), _NeedsV())


def test_check_combo_accepts_valid_pairs():
    check_combo(_V(), _NeedsV())      # ok
    check_combo(_NoV(), _NoNeed())    # ok (flowbc needs no V)


def test_protocols_are_runtime_checkable_shape():
    for attr in ("GROUPS", "STAGES", "TARGETS", "PROVIDES_V"):
        assert attr in Critic.__annotations__ or hasattr(Critic, attr)
    for attr in ("GROUPS", "NEEDS_V"):
        assert attr in Actor.__annotations__ or hasattr(Actor, attr)
