"""Construct the OmegaConf object TD-MPC2's internals expect, from flat kwargs.

Replaces upstream Hydra config.yaml + common/parser.py. Defaults mirror
../baselines/tdmpc2/tdmpc2/config.yaml exactly; only single-task / state /
non-episodic fields are specialized for OGBench cube-single.
"""
from __future__ import annotations

from omegaconf import OmegaConf

# Verbatim numeric defaults from upstream config.yaml (the values, not the Hydra wiring).
_DEFAULTS = dict(
    # training
    batch_size=256, reward_coef=0.1, value_coef=0.1, termination_coef=1.0,
    consistency_coef=20.0, rho=0.5, lr=3e-4, enc_lr_scale=0.3,
    grad_clip_norm=20.0, tau=0.01,
    discount_denom=5, discount_min=0.95, discount_max=0.995,
    # planning
    mpc=True, iterations=6, num_samples=512, num_elites=64, num_pi_trajs=24,
    horizon=3, min_std=0.05, max_std=2.0, temperature=0.5,
    # actor
    log_std_min=-10, log_std_max=2, entropy_coef=1e-4,
    # critic
    num_bins=101, vmin=-10, vmax=10,
    # architecture
    num_enc_layers=2, enc_dim=256, num_channels=32, mlp_dim=512,
    latent_dim=512, num_q=5, dropout=0.01, simnorm_dim=8,
    # misc
    episodic=False, multitask=False, compile=False, obs="state",
    episode_length=200,
)


def build_tdmpc2_cfg(*, obs_dim: int, action_dim: int, device: str = "cpu", **overrides):
    """obs_dim is the RAW obs dim; the agent folds a 3-dim goal on top, so the
    model's input dim is obs_dim+3 (see agent.GOAL_DIM)."""
    from agents.tdmpc2.agent import GOAL_DIM  # local import to avoid cycle

    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["device"] = device
    cfg["action_dim"] = int(action_dim)
    cfg["task_dim"] = 0            # single-task: no task embedding concatenated
    cfg["tasks"] = None
    cfg["obs_shape"] = {"state": (int(obs_dim) + GOAL_DIM,)}
    # derived (upstream computes these in common/parser.py)
    cfg["bin_size"] = (cfg["vmax"] - cfg["vmin"]) / (cfg["num_bins"] - 1)
    return OmegaConf.create(cfg)
