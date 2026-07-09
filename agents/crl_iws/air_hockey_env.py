"""Air-hockey training env for CRL+IWS (Factored-FB only, UM-aligned).

Uses ``air-hockey-rl`` directly with local config overrides — no rlmini-envs
wrapper changes required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/air_hockey/puck_goal_real_transfer_s4.yaml"


def _ensure_airhockey_importable() -> None:
    try:
        import airhockey  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    try:
        import rlmini_envs

        rlmini_root = Path(rlmini_envs.__file__).resolve().parent
        candidates.append(rlmini_root / "Environments/AirHockey/air-hockey-rl")
    except ImportError:
        pass
    candidates.append(REPO_ROOT.parent / "rlmini-envs/src/rlmini_envs/Environments/AirHockey/air-hockey-rl")
    for candidate in candidates:
        if (candidate / "airhockey").is_dir():
            sys.path.insert(0, str(candidate))
            return
    raise ImportError(
        "Could not import airhockey. Install the rlmini-envs extra (air-hockey-rl submodule)."
    )


def resolve_air_hockey_config(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Air hockey config not found: {config_path}")


def load_air_hockey_config(
    config_path: str | Path,
    *,
    seed: int,
    total_training_steps: int,
) -> dict[str, Any]:
    root = yaml.safe_load(resolve_air_hockey_config(config_path).read_text())
    env_config = dict(root["air_hockey"])
    env_config["task"] = "puck_goal_position"
    env_config["seed"] = int(seed)
    env_config["n_training_steps"] = int(total_training_steps)
    env_config.setdefault("return_goal_obs", False)
    env_config.setdefault("goal_radius_type", "fixed")
    env_config.setdefault("base_goal_radius", 0.15)
    env_config.setdefault("puck_goal_success_bonus", 0.0)
    env_config.setdefault("paddle_puck_success_bonus", 0.0)
    return env_config


def _strip_goal_tail(raw_observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
    if observation.shape[0] < 3:
        return observation.copy()
    return observation[:-2].copy()


def _paddle_position(env: Any) -> np.ndarray:
    position = env.current_state["paddles"]["paddle_ego"]["position"]
    return np.asarray(position[:2], dtype=np.float32)


def _puck_position(env: Any) -> np.ndarray:
    return _puck_position_from_state(env.current_state)


def _puck_position_from_state(state: Any) -> np.ndarray:
    position = state["pucks"][0]["position"]
    return np.asarray(position[:2], dtype=np.float32)


class _LocalAirHockeyPuckGoalPositionEnv:
    """UM-compatible override for randomized paddle spawn (real_transfer_s4)."""

    @staticmethod
    def build_class():
        _ensure_airhockey_importable()
        from airhockey.airhockey_tasks.puck_goal_position import AirHockeyPuckGoalPositionEnv

        class LocalAirHockeyPuckGoalPositionEnv(AirHockeyPuckGoalPositionEnv):
            def __init__(self, **kwargs):
                kwargs = dict(kwargs)
                kwargs.setdefault("goal_radius_type", "fixed")
                kwargs.setdefault("puck_goal_success_bonus", 0.0)
                kwargs.setdefault("paddle_puck_success_bonus", 0.0)
                kwargs.setdefault("base_goal_radius", 0.15)
                self.randomize_paddle_spawn = bool(kwargs.get("randomize_paddle_spawn", False))
                super().__init__(**kwargs)

            @staticmethod
            def from_dict(state_dict):
                return LocalAirHockeyPuckGoalPositionEnv(**state_dict)

            def get_paddle_configuration(self, name):
                if name == "paddle_ego" and self.randomize_paddle_spawn:
                    x_low = 0 + 2 * self.paddle_radius
                    x_high = self.table_x_bot - 2 * self.paddle_radius
                    y_low = self.table_y_left + 2 * self.paddle_radius
                    y_high = self.table_y_right - 2 * self.paddle_radius
                    x_pos = self.rng.uniform(low=x_low, high=x_high)
                    y_pos = self.rng.uniform(low=y_low, high=y_high)
                    return (x_pos, y_pos), (0, 0)
                return super().get_paddle_configuration(name)

        return LocalAirHockeyPuckGoalPositionEnv


def build_air_hockey_env(env_config: dict[str, Any]) -> Any:
    _ensure_airhockey_importable()
    from airhockey import AirHockeyEnv

    task = env_config.get("task")
    if task == "puck_goal_position" and bool(env_config.get("randomize_paddle_spawn", False)):
        local_cls = _LocalAirHockeyPuckGoalPositionEnv.build_class()
        return local_cls.from_dict(dict(env_config))
    return AirHockeyEnv(dict(env_config))


class AirHockeyCRLIWSEnv(gym.Env):
    """Single air-hockey env exposing UM-style state/goals for CRL+IWS training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG,
        seed: int = 0,
        horizon: int = 250,
        total_training_steps: int = 20_000_000,
        terminate_on_success: bool = True,
        auto_reset: bool = True,
        capture_video: bool = False,
    ) -> None:
        super().__init__()
        self._terminate_on_success = bool(terminate_on_success)
        self._auto_reset = bool(auto_reset)
        self._renderer = None
        env_config = load_air_hockey_config(
            config_path,
            seed=int(seed),
            total_training_steps=int(total_training_steps),
        )
        env_config["max_timesteps"] = int(horizon)
        self._env = build_air_hockey_env(env_config)
        self._seed = int(seed)
        if capture_video:
            _ensure_airhockey_importable()
            from airhockey.renderers.render import AirHockeyRenderer

            self._renderer = AirHockeyRenderer(self._env)

        obs, _info = self._env.reset(seed=self._seed)
        state = _strip_goal_tail(obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

    def _adapt(self, raw_obs: np.ndarray, info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        state = _strip_goal_tail(raw_obs)
        desired = np.asarray(self._env.get_desired_goal(), dtype=np.float32).reshape(-1)
        achieved = np.asarray(
            self._env.get_achieved_goal(self._env.current_state),
            dtype=np.float32,
        ).reshape(-1)
        success = bool(info.get("success", False))
        hand = _paddle_position(self._env)
        obj = _puck_position(self._env)
        return state, {
            "desired_goal": desired,
            "achieved_goal": achieved,
            "success": np.bool_(success),
            "goal_distance": np.float32(np.linalg.norm(achieved - desired)),
            "hand_position": hand.copy(),
            "object_position": obj.copy(),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        raw_obs, info = self._env.reset(seed=self._seed)
        state, out_info = self._adapt(raw_obs, dict(info) if info is not None else {})
        out_info = {
            **out_info,
            "transition_state": state.copy(),
            "transition_desired_goal": out_info["desired_goal"].copy(),
            "transition_achieved_goal": out_info["achieved_goal"].copy(),
            "transition_success": out_info["success"],
            "transition_goal_distance": out_info["goal_distance"],
            "transition_hand_position": out_info["hand_position"].copy(),
            "transition_object_position": out_info["object_position"].copy(),
        }
        return state, out_info

    def step(self, action: np.ndarray):
        raw_obs, _reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        info_dict = dict(info) if info is not None else {}
        success = bool(info_dict.get("success", False))
        if self._terminate_on_success and success and not terminated:
            info_dict["terminated_on_success"] = True
            terminated = True

        done = bool(terminated or truncated)
        transition_state, transition_info = self._adapt(raw_obs, info_dict)
        out_info = {
            **transition_info,
            "transition_state": transition_state.copy(),
            "transition_desired_goal": transition_info["desired_goal"].copy(),
            "transition_achieved_goal": transition_info["achieved_goal"].copy(),
            "transition_success": transition_info["success"],
            "transition_goal_distance": transition_info["goal_distance"],
            "transition_hand_position": transition_info["hand_position"].copy(),
            "transition_object_position": transition_info["object_position"].copy(),
        }

        if done:
            if self._auto_reset:
                reset_obs, reset_info = self._env.reset()
                next_state, next_info = self._adapt(
                    reset_obs,
                    dict(reset_info) if reset_info is not None else {},
                )
                out_info["desired_goal"] = next_info["desired_goal"]
                out_info["achieved_goal"] = next_info["achieved_goal"]
                out_info["success"] = next_info["success"]
                out_info["goal_distance"] = next_info["goal_distance"]
                out_info["hand_position"] = next_info["hand_position"]
                out_info["object_position"] = next_info["object_position"]
                state = next_state
            else:
                state = transition_state
        else:
            state = transition_state

        return state, 0.0, terminated, truncated, out_info

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            return None
        frame_bgr = self._renderer.get_frame()
        if frame_bgr is None:
            return None
        frame_rgb = np.asarray(frame_bgr, dtype=np.uint8)
        if frame_rgb.ndim != 3 or frame_rgb.shape[-1] < 3:
            return None
        return frame_rgb[..., :3][:, :, ::-1].copy()

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def make_air_hockey_vec_env(
    *,
    num_envs: int,
    seed: int,
    horizon: int,
    config_path: str | Path = DEFAULT_CONFIG,
    total_training_steps: int = 20_000_000,
    auto_reset: bool = True,
    capture_video: bool = False,
) -> gym.vector.SyncVectorEnv:
    def _factory(env_index: int):
        def _init():
            return AirHockeyCRLIWSEnv(
                config_path=config_path,
                seed=int(seed + env_index),
                horizon=int(horizon),
                total_training_steps=int(total_training_steps),
                auto_reset=auto_reset,
                capture_video=bool(capture_video and env_index == 0),
            )

        return _init

    return gym.vector.SyncVectorEnv([_factory(i) for i in range(int(num_envs))])
