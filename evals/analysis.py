"""evals/analysis.py — importable helpers for checkpoint re-run analysis.

Reuses train.make_agent + envs/evals; never modifies training paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf


def load_cfg(config_path: str | Path, device: str = "cpu"):
    """Load a run's .hydra/config.yaml and force the device."""
    cfg = OmegaConf.load(str(config_path))
    cfg.device = device
    return cfg


def build_env_and_agent(cfg):
    """Build (env, agent, obs_space, action_dim) from a loaded cfg."""
    from envs.ogbench import create_ogbench_env
    from train import make_agent

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    agent = make_agent(cfg, obs_space, action_dim)
    return env, agent, obs_space, action_dim


def load_checkpoint(agent, ckpt_path: str | Path, map_location: str = "cpu") -> None:
    """Load a torch state_dict checkpoint into agent with a clear error
    message when the checkpoint does not match the agent variant."""
    state = torch.load(str(ckpt_path), map_location=map_location)
    try:
        agent.load_state_dict(state)
    except (KeyError, RuntimeError) as e:
        raise RuntimeError(
            f"checkpoint does not match agent (cfg.agent build). "
            f"Loading {ckpt_path} failed: {e}"
        ) from e
