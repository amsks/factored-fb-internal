"""Environment adapter for rlmini_envs."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def create_rlmini_env(
    domain: str,
    *,
    cfg: Any = None,
    seed: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
):
    """Create a flat-observation Gymnasium env from ``rlmini_envs``.

    ``domain`` is used as the rlmini env name unless ``cfg.rlmini_env`` is set.
    """
    if obs_type != "state":
        raise ValueError("rlmini backend currently supports obs_type='state' only")
    if frame_stack != 1:
        raise ValueError("rlmini backend currently supports frame_stack=1 only")

    try:
        import rlmini_envs
    except ImportError as exc:
        raise ImportError(
            "rlmini_envs is not installed. Install it from the sibling repo, e.g. "
            "`python -m pip install -e ../rlmini_envs`."
        ) from exc

    env_name = _cfg_get(cfg, "rlmini_env", domain)
    variant = _cfg_get(cfg, "rlmini_variant", "default")
    horizon = int(_cfg_get(cfg, "rlmini_horizon", 1000))
    render = bool(_cfg_get(cfg, "rlmini_render", False))
    obs_keys = _plain(_cfg_get(cfg, "rlmini_obs_keys", None))
    obs_slices = _plain(_cfg_get(cfg, "rlmini_obs_slices", None))
    extra = _plain(_cfg_get(cfg, "rlmini_extra", {})) or {}

    env = rlmini_envs.make(
        env_name,
        variant=variant,
        horizon=horizon,
        seed=0 if seed is None else seed,
        render=render,
        gym_to_gymnasium=True,
        select_obs=True,
        obs_keys=obs_keys,
        obs_slices=obs_slices,
        obs_flat=True,
        obs_include_metadata=False,
        **extra,
    )
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(f"rlmini observation_space must be Box, got {env.observation_space}")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(f"rlmini action_space must be continuous Box, got {env.action_space}")
    if len(env.action_space.shape) != 1:
        raise ValueError(f"rlmini action_space must be 1D, got {env.action_space.shape}")

    # Most wrappers expose float64 bounds; Factored-FB casts observations to float32
    # at the model boundary, so we only enforce that reset produces a numeric vector.
    obs, info = env.reset(seed=seed)
    obs = np.asarray(obs)
    if obs.ndim != 1:
        raise ValueError(f"rlmini observations must be flat vectors, got shape {obs.shape}")
    return env, info
