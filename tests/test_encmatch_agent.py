"""cube_single_encmatch: state FB+FlowBC with two param-matched MLP encoders.

Verifies (a) the agent builds with MLPEncoder in the fw/bw slots, (b) one update()
step produces finite losses, and (c) total trainable online params match the pixel
DrQ run (~15.54M) within 1% — the point of the ablation.
"""
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from envs.ogbench import create_ogbench_env
from train import make_agent
from nn_models import MLPEncoder

PIXEL_ONLINE_PARAMS = 15_539_936  # FB+FlowBC visual_cube_single, targets excluded


def _build():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=["domain=cube_single_encmatch"])
    with open_dict(cfg):
        cfg.device = "cpu"
    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=0,
                                frame_stack=cfg.get("frame_stack", 1))
    agent = make_agent(cfg, env.observation_space, env.action_space.shape[0])
    return cfg, agent, env.observation_space.shape[0], env.action_space.shape[0]


def _online_params(model):
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith("_target_"))


def test_encoders_are_mlp_in_both_slots():
    _cfg, agent, _od, _ad = _build()
    assert isinstance(agent.model._fw_encoder, MLPEncoder)
    assert isinstance(agent.model._bw_encoder, MLPEncoder)
    assert agent.model._fw_encoder.output_space.shape == (256,)


def test_online_param_count_matches_pixel():
    _cfg, agent, _od, _ad = _build()
    n = _online_params(agent.model)
    assert abs(n - PIXEL_ONLINE_PARAMS) / PIXEL_ONLINE_PARAMS < 0.01, n


def test_one_update_is_finite():
    cfg, agent, obs_dim, act_dim = _build()
    B = cfg.batch_size  # agent permutes z over self.batch_size; batch must match
    g = torch.Generator().manual_seed(0)
    batch = {
        "observation": torch.randn(B, obs_dim, generator=g),
        "action": torch.randn(B, act_dim, generator=g).clamp(-1, 1),
        "next": {
            "observation": torch.randn(B, obs_dim, generator=g),
            "terminated": torch.zeros(B, dtype=torch.bool),
        },
    }
    metrics = agent.update(batch, step=0)
    for k in ("fb_loss", "actor_loss"):
        assert torch.isfinite(torch.as_tensor(metrics[k])).all(), (k, metrics[k])
