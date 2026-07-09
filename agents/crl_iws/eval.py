"""Periodic air-hockey policy evaluation (UM-aligned metrics + GIF rollouts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.crl_iws.air_hockey_env import AirHockeyCRLIWSEnv, make_air_hockey_vec_env


def _annotate_frame(
    frame_rgb: np.ndarray,
    *,
    timestep: int,
    goal_distance: float,
    success: bool,
    success_once: bool,
    success_timestep: int | None,
) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8).copy()
    lines = [
        f"t={timestep:04d}",
        f"goal_dist={goal_distance:.4f}",
        f"success={int(success)}",
        f"success_once={int(success_once)}",
    ]
    if success_timestep is not None:
        lines.append(f"success@t={success_timestep:04d}")
    try:
        import cv2
    except ImportError:
        return frame

    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if success else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return frame


def _resize_frame(frame_rgb: np.ndarray, *, width: int = 320) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    height, frame_width = frame.shape[:2]
    if frame_width <= width:
        return frame
    scale = width / float(frame_width)
    new_height = max(1, int(round(height * scale)))
    try:
        import cv2

        return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)
    except ImportError:
        return frame


@dataclass
class EvalResult:
    metrics: dict[str, float]
    successful_seeds: list[int]
    video_frames: list[np.ndarray]
    success_video_frames: list[np.ndarray]


def evaluate_air_hockey_policy(
    agent: Any,
    *,
    config_path: str,
    seed: int,
    horizon: int,
    total_training_steps: int,
    episodes: int,
    num_envs: int,
    eval_index: int,
    deterministic: bool = True,
) -> tuple[dict[str, float], list[int]]:
    """Run batched eval episodes and return metrics + successful seeds."""
    episodes = int(episodes)
    batch_size = max(1, min(int(num_envs), episodes))
    success_values: list[float] = []
    successful_seeds: list[int] = []

    episodes_done = 0
    batch_id = 0
    while episodes_done < episodes:
        current_batch = min(batch_size, episodes - episodes_done)
        batch_seed_base = int(seed) + int(eval_index) * 10_000 + batch_id * 1_000
        batch_seeds = [batch_seed_base + i for i in range(current_batch)]

        vec_env = make_air_hockey_vec_env(
            num_envs=current_batch,
            seed=batch_seeds[0],
            horizon=horizon,
            config_path=config_path,
            total_training_steps=total_training_steps,
            auto_reset=False,
        )
        try:
            raw_obs, infos = vec_env.reset(seed=batch_seeds)
            state_dim = int(np.prod(vec_env.single_observation_space.shape))
            obs = raw_obs.reshape(current_batch, state_dim).astype(np.float32)
            desired = _stack_goal(infos, current_batch, obs)
            success_once = np.zeros(current_batch, dtype=bool)

            for _ in range(int(horizon)):
                action = agent.sample_actions(
                    obs,
                    desired,
                    deterministic=deterministic,
                )
                raw_next, _, terminated, truncated, step_infos = vec_env.step(action)
                next_obs = raw_next.reshape(current_batch, state_dim).astype(np.float32)
                success = _stack_success(step_infos, current_batch)
                success_once |= success
                obs = next_obs
                desired = _stack_goal(step_infos, current_batch, desired)
                if bool(np.all(truncated | terminated)):
                    break

            for env_i, episode_seed in enumerate(batch_seeds):
                success_values.append(float(success_once[env_i]))
                if success_once[env_i]:
                    successful_seeds.append(int(episode_seed))
        finally:
            vec_env.close()

        episodes_done += current_batch
        batch_id += 1

    success_arr = np.asarray(success_values, dtype=np.float32)
    metrics = {
        "eval/success_once": float(success_arr.mean()) if success_arr.size else 0.0,
        "eval/episodes": float(success_arr.size),
    }
    return metrics, successful_seeds


def rollout_eval_gif(
    agent: Any,
    *,
    config_path: str,
    seed: int | None,
    horizon: int,
    total_training_steps: int,
    deterministic: bool = True,
    stop_after_success: bool = False,
) -> tuple[list[np.ndarray], bool]:
    """Roll out one episode with rendering; return RGB frames and success flag."""
    env = AirHockeyCRLIWSEnv(
        config_path=config_path,
        seed=0 if seed is None else int(seed),
        horizon=int(horizon),
        total_training_steps=int(total_training_steps),
        auto_reset=False,
        capture_video=True,
    )
    frames: list[np.ndarray] = []
    success_once = False
    success_timestep: int | None = None
    try:
        obs, info = env.reset(seed=seed)
        desired = info["desired_goal"]
        for timestep in range(int(horizon) + 1):
            frame = env.render()
            if frame is not None:
                frames.append(
                    _resize_frame(
                        _annotate_frame(
                            frame,
                            timestep=timestep,
                            goal_distance=float(info.get("goal_distance", 0.0)),
                            success=bool(info.get("success", False)),
                            success_once=success_once,
                            success_timestep=success_timestep,
                        )
                    )
                )
            if timestep == int(horizon):
                break

            action = agent.sample_actions(
                obs[None],
                desired[None],
                deterministic=deterministic,
            )
            next_obs, _, _term, trunc, info = env.step(action[0])
            if bool(info.get("transition_success", info.get("success", False))):
                if not success_once:
                    success_timestep = timestep + 1
                success_once = True
            obs = next_obs
            desired = info["desired_goal"]
            if stop_after_success and success_once:
                frame = env.render()
                if frame is not None:
                    frames.append(
                        _resize_frame(
                            _annotate_frame(
                                frame,
                                timestep=timestep + 1,
                                goal_distance=float(info.get("transition_goal_distance", info.get("goal_distance", 0.0))),
                                success=True,
                                success_once=True,
                                success_timestep=success_timestep,
                            )
                        )
                    )
                break
            if bool(trunc):
                frame = env.render()
                if frame is not None:
                    frames.append(
                        _resize_frame(
                            _annotate_frame(
                                frame,
                                timestep=timestep + 1,
                                goal_distance=float(info.get("transition_goal_distance", info.get("goal_distance", 0.0))),
                                success=bool(info.get("transition_success", False)),
                                success_once=success_once,
                                success_timestep=success_timestep,
                            )
                        )
                    )
                break
    finally:
        env.close()
    return frames, success_once


def run_periodic_eval(
    agent: Any,
    *,
    config_path: str,
    seed: int,
    horizon: int,
    total_training_steps: int,
    eval_index: int,
    episodes: int,
    num_envs: int,
    deterministic: bool,
    capture_video: bool,
    capture_success_video: bool,
) -> EvalResult:
    metrics, successful_seeds = evaluate_air_hockey_policy(
        agent,
        config_path=config_path,
        seed=seed,
        horizon=horizon,
        total_training_steps=total_training_steps,
        episodes=episodes,
        num_envs=num_envs,
        eval_index=eval_index,
        deterministic=deterministic,
    )

    video_frames: list[np.ndarray] = []
    success_video_frames: list[np.ndarray] = []
    if capture_video:
        video_seed = None if seed is None else int(seed) + int(eval_index) * 10_000 + 999
        video_frames, _ = rollout_eval_gif(
            agent,
            config_path=config_path,
            seed=video_seed,
            horizon=horizon,
            total_training_steps=total_training_steps,
            deterministic=deterministic,
        )
    if capture_success_video and successful_seeds:
        for success_seed in successful_seeds:
            frames, ok = rollout_eval_gif(
                agent,
                config_path=config_path,
                seed=int(success_seed),
                horizon=horizon,
                total_training_steps=total_training_steps,
                deterministic=deterministic,
                stop_after_success=True,
            )
            if ok and frames:
                success_video_frames = frames
                break

    return EvalResult(
        metrics=metrics,
        successful_seeds=successful_seeds,
        video_frames=video_frames,
        success_video_frames=success_video_frames,
    )


def _stack_goal(
    infos: Any,
    num_envs: int,
    fallback: np.ndarray,
) -> np.ndarray:
    rows = []
    for i in range(num_envs):
        val = _info_field(infos, "desired_goal", i)
        if val is None:
            rows.append(np.asarray(fallback[i], dtype=np.float32).reshape(-1))
        else:
            rows.append(np.asarray(val, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)


def _stack_success(infos: Any, num_envs: int) -> np.ndarray:
    success = np.zeros(num_envs, dtype=bool)
    for i in range(num_envs):
        val = _info_field(infos, "transition_success", i)
        if val is None:
            val = _info_field(infos, "success", i)
        if val is not None:
            success[i] = bool(val)
    return success


def _info_field(infos: Any, key: str, env_i: int) -> Any:
    if isinstance(infos, dict):
        if key not in infos:
            return None
        val = infos[key]
        if isinstance(val, (list, tuple)):
            return val[env_i]
        arr = np.asarray(val)
        if arr.ndim == 0:
            return arr.item()
        return arr[env_i]
    if isinstance(infos, (list, tuple)) and env_i < len(infos):
        item = infos[env_i]
        return item.get(key) if isinstance(item, dict) else None
    return None
