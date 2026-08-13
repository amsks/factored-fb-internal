"""Region-measure objective composed with a NON-flow actor (2026-08-05).

The region-measure critic and the flow-BC actor have been welded together by
inheritance since the port (`FBMeasureFlowBCAgent(FBFlowBCAgent)`), so every
region-FB number the project has is a flow-actor number. To ask whether the
flow actor helps or hurts the two-phase method we need the same critic behind
the base FB actor (TD3+BC), which means the measure logic has to become a mixin.

These tests pin the two things that could silently go wrong:
  1. the FLOW class must resolve every method exactly as it did before the
     refactor (the paper's whole torch stack rides on it), and
  2. the NO-FLOW class must take the measure critic from the mixin and the
     actor from FBAgent, not from FBFlowBCAgent.
"""
import inspect

import torch

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.agent import FBFlowBCAgent
from agents.fb.measure.agent import (
    FBMeasureAgent,
    FBMeasureFlowBCAgent,
    RegionMeasureMixin,
)

CRITIC_SIDE = ("update_fb", "hit_matrix")
ACTOR_SIDE = ("update_actor", "act", "sample_action_from_norm_obs")


def _owner(cls, name):
    """The class in the MRO that actually provides `name`."""
    for c in cls.__mro__:
        if name in c.__dict__:
            return c
    raise AttributeError(name)


def test_flow_class_mro_is_behaviourally_unchanged():
    """The mixin must sit directly before FBFlowBCAgent, so resolution order for
    every method is identical to the pre-refactor single-class layout."""
    mro = FBMeasureFlowBCAgent.__mro__
    assert mro[1] is RegionMeasureMixin
    assert mro[2] is FBFlowBCAgent
    assert mro[3] is FBAgent
    for name in CRITIC_SIDE:
        assert _owner(FBMeasureFlowBCAgent, name) is RegionMeasureMixin, name
    for name in ACTOR_SIDE:
        assert _owner(FBMeasureFlowBCAgent, name) is FBFlowBCAgent, name


def test_noflow_class_takes_critic_from_mixin_and_actor_from_fbagent():
    mro = FBMeasureAgent.__mro__
    assert mro[1] is RegionMeasureMixin
    assert mro[2] is FBAgent
    assert FBFlowBCAgent not in mro, "the no-flow arm must not inherit the flow actor"
    for name in CRITIC_SIDE:
        assert _owner(FBMeasureAgent, name) is RegionMeasureMixin, name
    for name in ACTOR_SIDE:
        assert _owner(FBMeasureAgent, name) is FBAgent, name


def test_both_arms_share_one_critic_implementation():
    """Same function object, or the two arms are not a controlled comparison."""
    for name in CRITIC_SIDE:
        assert (getattr(FBMeasureFlowBCAgent, name)
                is getattr(FBMeasureAgent, name)), name


def test_measure_knobs_survive_on_the_noflow_arm():
    """__init__ is on the mixin, so every region knob must reach the no-flow arm."""
    sig = inspect.signature(FBMeasureAgent.__init__).parameters
    # bound through the mixin's **kwargs chain, so check the mixin's own signature
    msig = inspect.signature(RegionMeasureMixin.__init__).parameters
    for k in ("measure_mode", "hit_eps", "hit_slice", "hit_spec", "freeze_b",
              "nhat_head", "warm_start_ckpt", "nonneg_coef", "mass_coef"):
        assert k in msig, k


def test_hit_matrix_is_identical_across_the_two_arms():
    torch.manual_seed(0)
    obs = torch.randn(12, 40) * 0.3

    class _S:
        hit_identity = False
        hit_kernel = "ball"
        hit_slice = (19, 22)
        hit_eps = 0.4
        hit_blocks = None

    a = FBMeasureFlowBCAgent.hit_matrix(_S(), obs)
    b = FBMeasureAgent.hit_matrix(_S(), obs)
    assert torch.equal(a, b)
