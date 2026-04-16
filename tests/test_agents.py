"""
tests/test_agents.py — Smoke tests for FBAgent and FBFlowBCAgent.

All tests use tiny networks (hidden_dim=32) and synthetic data so they
run in seconds without a GPU or OGBench installation.
"""

import numpy as np
import gymnasium
import torch
import pytest


def _obs_space(dim=10):
    return gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
    )


def _make_batch(B=16, obs_dim=10, act_dim=4, device="cpu"):
    """Return a batch dict in the DictBuffer layout."""
    return {
        "observation": torch.randn(B, obs_dim, device=device),
        "action":      torch.randn(B, act_dim, device=device),
        "next": {
            "observation": torch.randn(B, obs_dim, device=device),
            "terminated":  torch.zeros(B, device=device),
        },
    }


def _small_forward_cfg():
    from nn_models import ForwardArchiConfig
    return ForwardArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2, num_parallel=2)


def _small_backward_cfg():
    from nn_models import BackwardArchiConfig
    return BackwardArchiConfig(hidden_dim=32, hidden_layers=1)


def _small_actor_cfg():
    from nn_models import SimpleActorArchiConfig
    return SimpleActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)


def _small_flowbc_cfg():
    from nn_models import NoiseConditionedActorArchiConfig
    return NoiseConditionedActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)


# ── FBModel ──────────────────────────────────────────────────────────────── #

def test_fb_model_construction():
    from agents.fb.model import FBModel
    obs_space = _obs_space(10)
    model = FBModel(
        obs_space=obs_space,
        action_dim=4,
        z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        device="cpu",
    )
    assert model.z_dim == 8


def test_fb_model_backward_shape():
    from agents.fb.model import FBModel
    model = FBModel(_obs_space(10), 4, z_dim=8,
                    forward_cfg=_small_forward_cfg(),
                    backward_cfg=_small_backward_cfg())
    obs = torch.randn(4, 10)
    z = model.backward_map(obs)
    assert z.shape == (4, 8)


def test_fb_model_forward_shape():
    from agents.fb.model import FBModel
    model = FBModel(_obs_space(10), 4, z_dim=8,
                    forward_cfg=_small_forward_cfg(),
                    backward_cfg=_small_backward_cfg())
    obs = torch.randn(4, 10)
    z   = torch.randn(4, 8)
    act = torch.randn(4, 4)
    F_out = model.forward_map(obs, z, act)
    assert F_out.shape == (2, 4, 8)   # (num_parallel, batch, z_dim)


def test_fb_model_infer_z():
    from agents.fb.model import FBModel
    model = FBModel(_obs_space(10), 4, z_dim=8,
                    forward_cfg=_small_forward_cfg(),
                    backward_cfg=_small_backward_cfg())
    obs    = torch.randn(32, 10)
    reward = torch.randn(32)
    z = model.infer_z(obs, reward)
    assert z.shape == (8,)
    # Should be normalised to sqrt(z_dim)
    import math
    assert abs(z.norm().item() - math.sqrt(8)) < 0.1


def test_fb_model_mixed_z():
    from agents.fb.model import FBModel
    model = FBModel(_obs_space(10), 4, z_dim=8,
                    forward_cfg=_small_forward_cfg(),
                    backward_cfg=_small_backward_cfg())
    next_obs = torch.randn(16, 10)
    z = model.mixed_z(next_obs, train_goal_ratio=0.5)
    assert z.shape == (16, 8)


# ── FBAgent ──────────────────────────────────────────────────────────────── #

def test_fb_agent_update_returns_metrics():
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
        device="cpu",
    )
    batch = _make_batch()
    metrics = agent.update(batch, step=1)
    assert "loss/fb" in metrics
    assert "loss/ortho" in metrics
    assert "loss/actor" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_fb_agent_update_losses_are_finite():
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    for step in range(3):
        metrics = agent.update(_make_batch(), step=step)
    assert all(np.isfinite(v) for v in metrics.values()), \
        f"Non-finite loss: {metrics}"


def test_fb_agent_act_shape():
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    obs = torch.randn(1, 10)
    z   = agent.model.sample_z(1)
    action = agent.act(obs=obs, z=z)
    assert action.shape == (1, 4)
    assert action.abs().max() <= 1.0 + 1e-5


def test_fb_agent_act_deterministic_at_eval():
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    obs = torch.randn(1, 10)
    z   = agent.model.sample_z(1)
    a1 = agent.act(obs=obs, z=z, eval_mode=True)
    a2 = agent.act(obs=obs, z=z, eval_mode=True)
    assert torch.allclose(a1, a2), "eval act should be deterministic"


def test_fb_agent_infer_z_shape():
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    obs    = torch.randn(64, 10)
    reward = torch.randn(64)
    z = agent.infer_z(obs, reward)
    assert z.shape == (8,)


def test_fb_agent_checkpoint_roundtrip(tmp_path):
    from agents.fb.agent import FBAgent
    agent = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    agent.update(_make_batch(), step=1)

    ckpt = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), ckpt)

    agent2 = FBAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_actor_cfg(),
    )
    agent2.load_state_dict(torch.load(ckpt, map_location="cpu"))

    obs = torch.randn(1, 10)
    z   = agent.model.sample_z(1)
    assert torch.allclose(agent.act(obs=obs, z=z), agent2.act(obs=obs, z=z))


# ── FBFlowBCAgent ────────────────────────────────────────────────────────── #

def test_fb_flowbc_agent_update():
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    agent = FBFlowBCAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_flowbc_cfg(),
        bc_coeff=0.3,
        n_action_samples=3,
    )
    metrics = agent.update(_make_batch(), step=1)
    assert "loss/actor_bc" in metrics
    assert "loss/actor_rl" in metrics
    assert all(np.isfinite(v) for v in metrics.values())


def test_fb_flowbc_agent_act_shape():
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    agent = FBFlowBCAgent(
        obs_space=_obs_space(10), action_dim=4, z_dim=8,
        forward_cfg=_small_forward_cfg(),
        backward_cfg=_small_backward_cfg(),
        actor_cfg=_small_flowbc_cfg(),
        n_action_samples=4,
    )
    obs = torch.randn(2, 10)
    z   = agent.model.sample_z(2)
    action = agent.act(obs=obs, z=z, eval_mode=True)
    assert action.shape == (2, 4)