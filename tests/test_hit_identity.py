"""hit_identity: the bounded objective on FB's OWN Dirac support.

Completes the 2x2 that separates loss form from reward support:

                     FB loss (linear diag + ortho) | bounded loss (squared, no ortho)
    Dirac support I           0.42 (stock FB)      |   <- this knob
    cube ball                 0.367 (hitdirac)     |   0.967 (phase 1)

Without this cell we cannot say whether the bounded objective is a general
recipe or one that only works when handed a privileged region structure.

Note hit_eps -> 0 does NOT produce a Dirac on this data: the cube is stationary
for long stretches of play, so exact cube-position duplicates are common
(measured: 6.6% of off-diagonal pairs at eps=0, and float32 cdist leaves 7% of
the DIAGONAL below threshold). The identity must be constructed, not approached.
"""
import gymnasium as gym
import pytest
import torch

from agents.fb.measure.agent import FBMeasureFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

B, OBS = 16, 28


def _agent(**kw):
    return FBMeasureFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (OBS,)), action_dim=5, batch_size=B,
        z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2,
                                                   embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def _obs_with_duplicate_cubes():
    """Rows 0 and 1 share an exact cube position, as stationary play data does."""
    o = torch.randn(B, OBS)
    o[1, 19:22] = o[0, 19:22]
    return o


def test_default_is_the_ball_unchanged():
    a = _agent(hit_eps=0.4)
    o = _obs_with_duplicate_cubes()
    h = a.hit_matrix(o)
    assert h.shape == (B, B)
    assert (h.diagonal() == 1).all()
    assert h[0, 1] == 1 and h[1, 0] == 1      # duplicates hit each other


def test_identity_is_exactly_the_identity():
    a = _agent(hit_identity=True)
    h = a.hit_matrix(_obs_with_duplicate_cubes())
    assert torch.equal(h, torch.eye(B)), "hit must be exactly I"
    # the duplicate-cube pair must NOT hit: that is the whole point of a Dirac
    assert h[0, 1] == 0


def test_identity_ignores_eps_and_slice():
    """A Dirac has no radius and reads no abstraction; both knobs must be inert."""
    o = _obs_with_duplicate_cubes()
    for kw in ({"hit_eps": 1.6}, {"hit_slice": (12, 15)}, {"hit_eps": 0.0}):
        h = _agent(hit_identity=True, **kw).hit_matrix(o)
        assert torch.equal(h, torch.eye(B)), kw


def test_eps_zero_is_not_a_dirac():
    """Guard the trap this knob exists to avoid."""
    a = _agent(hit_eps=0.0)
    h = a.hit_matrix(_obs_with_duplicate_cubes())
    assert h[0, 1] == 1, "eps=0 still matches exact cube duplicates, so it is not a Dirac"


def test_dtype_and_device_match_the_ball_path():
    o = _obs_with_duplicate_cubes()
    hb = _agent(hit_eps=0.4).hit_matrix(o)
    hi = _agent(hit_identity=True).hit_matrix(o)
    assert hi.dtype == hb.dtype and hi.device == hb.device


def test_train_py_propagates_hit_identity():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    import train as train_mod

    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train",
                      overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                 "device=cpu", "hit_identity=true"])
    agent = train_mod.make_agent(cfg, gym.spaces.Box(-1, 1, (OBS,)), 5)
    assert agent.hit_identity is True
