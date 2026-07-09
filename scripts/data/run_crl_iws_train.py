"""Online CRL training (PyTorch). Saves periodic checkpoints; no episode export.

Supports two sampling modes aligned with Unsupervised_Manipulation:

- ``sgcrl`` — uniform episode replay + geometric future goals (``train_air_hockey_task.sh``)
- ``iws`` — interaction-weighted sampling (``benchmark/testbed/algorithms/iws.yaml``)

Usage
-----
    # UM-matched SGCRL baseline:
    bash scripts/agents/crl_iws/air_hockey_sgcrl.sh

    # Paper-scale air hockey IWS:
    bash scripts/agents/crl_iws/air_hockey.sh

    # Short local smoke:
    python scripts/data/run_crl_iws_train.py \\
        --algorithm sgcrl \\
        --total-steps 500000 \\
        --num-envs 64 \\
        --random-exploration-steps 5000 \\
        --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from agents.crl_iws.agent_torch import CRLIWSAgentTorch
from agents.crl_iws.air_hockey_env import DEFAULT_CONFIG, make_air_hockey_vec_env
from agents.crl_iws.eval import run_periodic_eval
from agents.crl_iws.phase import compute_phase_labels, hand_object_distance
from agents.crl_iws.replay import IWSReplayBuffer
from agents.crl_iws.schedules import (
    advance_schedule_index,
    build_aligned_schedule,
    next_pending_schedule_index,
    normalize_completed_eval_steps,
)
from agents.crl_iws.tracker import CRLIWSTracker


# ---------------------------------------------------------------------------
# UM-aligned defaults (air hockey IWS)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    algorithm: str = "iws"
    env: str = "AirHockeyPuckGoalPosition-v0"
    air_hockey_config: str = str(DEFAULT_CONFIG.relative_to(REPO))
    num_envs: int = 256
    horizon: int = 250
    max_episode_steps: int = 250
    seed: int = 11
    device: str = "cuda"
    total_steps: int = 20_000_000
    random_exploration_steps: int = 200_000
    min_replay_size: int = 1024
    checkpoint_dir: str | None = None
    checkpoint_every: int = 200_000
    log_every: int = 1_000
    eval_every_steps: int = 500_000
    eval_episodes: int = 32
    eval_num_envs: int = 128
    eval_deterministic: bool = True
    eval_capture_video: bool = True
    eval_capture_success_video: bool = True
    eval_video_fps: int = 20
    hidden_dims: tuple[int, ...] = (512, 512)
    repr_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coeff: float = 0.0
    logsumexp_coeff: float = 0.01
    random_goals: float = 0.5
    use_action_entropy: bool = True
    critic_score_type: str = "cosine"
    replay_capacity: int = 200_000
    batch_size: int = 64
    updates_per_iter: int = 16
    future_discount: float = 0.99
    context_repetition_factor: int = 8
    contact_distance: float = 0.12
    contact_sigma: float = 30.0
    near_contact_distance: float = 0.18
    near_goal_distance: float = 0.18
    post_contact_steps: int = 20
    iws_random_fraction: float = 0.0
    iws_full_episode_future: bool = True
    iws_event_selection: str = "uniform"
    output_root: str = "runs"
    wandb_project: str = "air_hockey"
    run_name: str = "crl_iws_air_hockey"
    use_wandb: bool = True
    wandb_mode: str = "online"
    wandb_entity: str | None = None
    wandb_group: str | None = "air_hockey"
    wandb_tags: tuple[str, ...] = ("crl_iws", "interaction_weighted_sampling", "iws")
    progress_bar: bool = True


def _resolve_algorithm_config(args: argparse.Namespace) -> None:
    """Apply algorithm-specific defaults (UM ``sgcrl`` vs ``iws``)."""
    algorithm = str(args.algorithm).lower()
    if algorithm not in {"sgcrl", "iws"}:
        raise ValueError(f"Unsupported algorithm {args.algorithm!r}; expected 'sgcrl' or 'iws'.")

    if algorithm == "sgcrl":
        args.context_repetition_factor = 1
        args.iws_full_episode_future = False
        args.use_iws = False
        base_tags = {"sgcrl", "crl"}
    else:
        # UM benchmark/testbed/algorithms/iws.yaml backend_defaults.air_hockey
        args.use_iws = True
        args.context_repetition_factor = 8
        args.iws_full_episode_future = True
        args.contact_distance = 0.12
        args.contact_sigma = 30.0
        args.near_contact_distance = 0.24
        args.near_goal_distance = 0.18
        args.post_contact_steps = 20
        args.iws_event_selection = "uniform"
        base_tags = {"crl_iws", "interaction_weighted_sampling", "iws"}
    args.wandb_tags = sorted(base_tags | set(args.wandb_tags or []))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _max_episode_steps(vec_env, default: int) -> int:
    spec = getattr(vec_env, "spec", None)
    if spec is None:
        return default
    first = spec[0] if isinstance(spec, (list, tuple)) else spec
    return getattr(first, "max_episode_steps", None) or default


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


def _stack_from_info(
    infos: Any,
    key: str,
    num_envs: int,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    rows = []
    for i in range(num_envs):
        val = _info_field(infos, key, i)
        if val is None:
            if fallback is None:
                raise KeyError(f"Missing info[{key!r}] for env {i}")
            val = fallback[i]
        rows.append(np.asarray(val, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)


def _make_vec_env(
    env_name: str,
    num_envs: int,
    seed: int,
    *,
    horizon: int,
    config_path: str,
    total_training_steps: int,
):
    del env_name  # kept for CLI compatibility; Factored-FB uses air-hockey-rl directly.
    return make_air_hockey_vec_env(
        num_envs=num_envs,
        seed=seed,
        horizon=horizon,
        config_path=config_path,
        total_training_steps=total_training_steps,
    )


def _goal_distance_mean(desired: np.ndarray, achieved: np.ndarray) -> float:
    if desired.shape != achieved.shape:
        width = min(desired.shape[-1], achieved.shape[-1])
        desired = desired[..., :width]
        achieved = achieved[..., :width]
    return float(np.linalg.norm(desired - achieved, axis=-1).mean())


def _stack_transition_states(
    infos: Any,
    num_envs: int,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    return _stack_from_info(infos, "transition_state", num_envs, fallback=fallback)


def _stack_transition_goals(
    infos: Any,
    num_envs: int,
    *,
    key: str,
    fallback: np.ndarray,
) -> np.ndarray:
    return _stack_from_info(infos, f"transition_{key}", num_envs, fallback=fallback)


def _transition_success(infos: Any, num_envs: int) -> np.ndarray:
    success = np.zeros(num_envs, dtype=bool)
    for i in range(num_envs):
        val = _info_field(infos, "transition_success", i)
        if val is None:
            val = _info_field(infos, "success", i)
        if val is not None:
            success[i] = bool(val)
    return success


def _transition_goal_distance_mean(infos: Any, num_envs: int) -> float:
    distances = []
    for i in range(num_envs):
        val = _info_field(infos, "transition_goal_distance", i)
        if val is not None:
            distances.append(float(val))
            continue
        try:
            desired = _stack_from_info(infos, "transition_desired_goal", num_envs)[i]
            achieved = _stack_from_info(infos, "transition_achieved_goal", num_envs)[i]
            distances.append(float(np.linalg.norm(desired - achieved)))
        except KeyError:
            continue
    if not distances:
        return 0.0
    return float(np.mean(distances))


class _MetricAccumulator:
    """Running mean of per-update metrics between log intervals."""

    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count = 0

    def add(self, metrics: dict[str, float]) -> None:
        self._count += 1
        for key, value in metrics.items():
            self._sum[key] = self._sum.get(key, 0.0) + float(value)

    def mean(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        inv = 1.0 / self._count
        return {key: value * inv for key, value in self._sum.items()}

    def reset(self) -> None:
        self._sum.clear()
        self._count = 0


def _progress_postfix(metrics: dict[str, float], *, live_sps: float | None = None) -> dict[str, float | int]:
    """Compact metrics shown beside the tqdm bar."""
    out: dict[str, float | int] = {}
    if live_sps is not None:
        out["sps"] = int(round(live_sps))
    for src, key in (
        ("train/env_steps_per_second", "sps"),
        ("train/updates", "upd"),
        ("train/categorical_accuracy", "acc"),
        ("train/critic_loss", "critic"),
        ("train/actor_loss", "actor"),
        ("train/goal_distance_mean", "goal"),
        ("train/success_once", "succ"),
        ("eval/success_once", "eval"),
        ("replay/transitions", "trans"),
    ):
        if src not in metrics:
            continue
        if key == "sps" and live_sps is not None:
            continue
        val = float(metrics[src])
        if key == "sps":
            out[key] = int(round(val))
        elif key == "upd":
            out[key] = int(val)
        elif key == "trans":
            out[key] = int(val)
        elif key in {"acc", "succ", "eval"}:
            out[key] = round(val, 3)
        elif key in {"critic", "actor"}:
            out[key] = round(val, 4)
        else:
            out[key] = round(val, 3)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint file or directory to a single ``.pt`` path."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        ckpts = sorted(
            candidate.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint_*.pt files found in {candidate}")
        return ckpts[-1].resolve()
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _load_training_checkpoint(
    ckpt_path: Path,
    agent: CRLIWSAgentTorch,
    *,
    run_label: str,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload["agent"])
    saved_cfg = payload.get("config") or {}
    for key in ("algorithm", "seed", "num_envs", "repr_dim"):
        if key in saved_cfg:
            print(f"[{run_label}] checkpoint {key}={saved_cfg[key]!r}")
    print(
        f"[{run_label}] resumed from {ckpt_path} "
        f"(steps={payload.get('total_steps', 0)}, updates={payload.get('total_updates', 0)})"
    )
    return payload


def _flush_success_metrics(
    interval_success_once: list[float],
    interval_success_ticks: list[float],
) -> dict[str, float]:
    if not interval_success_once:
        return {}
    return {
        "train/success_once": float(np.mean(interval_success_once)),
        "train/success_ticks": float(np.mean(interval_success_ticks)),
        "train/success_episodes": float(len(interval_success_once)),
    }


def _completed_eval_steps(metrics_jsonl: Path) -> set[int]:
    if not metrics_jsonl.exists():
        return set()
    completed: set[int] = set()
    for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "eval/success_once" in row:
            completed.add(int(row["step"]))
    return completed


def _schedule_eval_index(eval_schedule: list[int], completed_eval_steps: set[int]) -> int:
    return sum(1 for step in eval_schedule if step in completed_eval_steps)


def _run_scheduled_eval(
    *,
    agent: CRLIWSAgentTorch,
    tracker: CRLIWSTracker,
    args: argparse.Namespace,
    total_steps: int,
    eval_index: int,
    run_label: str,
    progress: tqdm,
) -> None:
    progress.write(
        f"[{run_label}] eval @ {total_steps} "
        f"({args.eval_episodes} episodes, {args.eval_num_envs} envs)"
    )
    result = run_periodic_eval(
        agent,
        config_path=args.air_hockey_config,
        seed=args.seed,
        horizon=args.horizon,
        total_training_steps=args.total_steps,
        eval_index=eval_index,
        episodes=args.eval_episodes,
        num_envs=args.eval_num_envs,
        deterministic=args.eval_deterministic,
        capture_video=args.eval_capture_video,
        capture_success_video=args.eval_capture_success_video,
    )
    metrics = {
        **result.metrics,
        "eval/checkpoint_env_steps": float(total_steps),
        "eval/index": float(eval_index),
    }
    tracker.log_metrics(metrics, step=total_steps)
    if args.eval_capture_video and result.video_frames:
        gif_path = tracker.log_video(
            "eval/rollout",
            result.video_frames,
            step=total_steps,
            fps=args.eval_video_fps,
        )
        if gif_path is not None:
            progress.write(f"[{run_label}] eval GIF → {gif_path}")
    if args.eval_capture_success_video and result.success_video_frames:
        gif_path = tracker.log_video(
            "eval/success_rollout",
            result.success_video_frames,
            step=total_steps,
            fps=args.eval_video_fps,
        )
        if gif_path is not None:
            progress.write(f"[{run_label}] success GIF → {gif_path}")
    progress.write(
        f"[{run_label}] eval/success_once={metrics['eval/success_once']:.4f} "
        f"({int(metrics['eval/episodes'])} episodes)"
    )


def run_training(args: argparse.Namespace) -> None:
    _resolve_algorithm_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    resume_run_dir = Path(args.resume_run_dir).resolve() if args.resume_run_dir else None
    tracker = CRLIWSTracker(
        output_root=args.output_root,
        project=args.wandb_project,
        run_name=args.run_name,
        config=config_dict,
        use_wandb=args.use_wandb,
        wandb_mode=args.wandb_mode,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=list(args.wandb_tags) if args.wandb_tags else None,
        resume_run_dir=resume_run_dir,
    )
    ckpt_dir = tracker.paths.checkpoint_dir
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_label = f"crl-{args.algorithm}"
    print(f"[{run_label}] run_dir={tracker.paths.run_dir}")
    print(f"[{run_label}] algorithm={args.algorithm}  use_iws={args.use_iws}")
    print(f"[{run_label}] Creating {args.num_envs} envs: {args.env}")
    vec_env = _make_vec_env(
        args.env,
        args.num_envs,
        args.seed,
        horizon=args.horizon,
        config_path=args.air_hockey_config,
        total_training_steps=args.total_steps,
    )

    obs_space = vec_env.single_observation_space
    act_space = vec_env.single_action_space
    state_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(act_space.shape))

    raw_obs, infos = vec_env.reset(seed=list(range(args.seed, args.seed + args.num_envs)))
    obs = raw_obs.reshape(args.num_envs, state_dim).astype(np.float32)
    try:
        desired_goals = _stack_from_info(infos, "desired_goal", args.num_envs)
    except KeyError:
        desired_goals = obs.copy()
    goal_dim = int(desired_goals.shape[1])
    try:
        achieved_goals = _stack_from_info(infos, "achieved_goal", args.num_envs, fallback=obs)
    except KeyError:
        achieved_goals = obs[:, 4:6].copy() if state_dim >= 6 else obs.copy()
    try:
        hand_positions = _stack_from_info(infos, "hand_position", args.num_envs)
        object_positions = _stack_from_info(infos, "object_position", args.num_envs)
    except KeyError:
        hand_positions = obs[:, 12:14].copy() if state_dim >= 14 else obs[:, 0:2].copy()
        object_positions = obs[:, 27:29].copy() if state_dim >= 29 else obs[:, 4:6].copy()

    action_low = np.asarray(act_space.low, dtype=np.float32)
    action_high = np.asarray(act_space.high, dtype=np.float32)
    print(
        f"[{run_label}] state_dim={state_dim}  goal_dim={goal_dim}  "
        f"action_dim={action_dim}  device={device}"
    )

    agent = CRLIWSAgentTorch(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        hidden_dims=tuple(args.hidden_dims),
        repr_dim=args.repr_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coeff=args.entropy_coeff,
        logsumexp_coeff=args.logsumexp_coeff,
        random_goals=args.random_goals,
        use_action_entropy=args.use_action_entropy,
        critic_score_type=args.critic_score_type,
    )

    max_ep_steps = _max_episode_steps(vec_env, args.max_episode_steps)
    replay = IWSReplayBuffer(
        capacity_transitions=args.replay_capacity,
        max_episode_steps=max_ep_steps,
        num_envs=args.num_envs,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        contact_distance=args.contact_distance,
    )
    replay.reset(obs, achieved_goals)

    total_steps = 0
    total_updates = 0
    if args.resume_from:
        ckpt_path = _resolve_checkpoint_path(args.resume_from)
        payload = _load_training_checkpoint(ckpt_path, agent, run_label=run_label)
        total_steps = int(payload.get("total_steps", 0))
        total_updates = int(payload.get("total_updates", 0))
        saved_cfg = payload.get("config") or {}
        for key in ("algorithm", "seed", "num_envs", "repr_dim"):
            if key in saved_cfg and getattr(args, key, None) != saved_cfg[key]:
                print(
                    f"[{run_label}] warning: resume checkpoint {key}="
                    f"{saved_cfg[key]!r} differs from CLI {getattr(args, key)!r}"
                )
    last_log = total_steps
    last_ckpt = total_steps
    last_log_time = time.perf_counter()
    last_metrics: dict[str, float] = {}
    update_metrics = _MetricAccumulator()
    saved_checkpoints: list[dict] = []

    train_success_once = np.zeros(args.num_envs, dtype=bool)
    train_success_ticks = np.zeros(args.num_envs, dtype=np.float32)
    interval_success_once: list[float] = []
    interval_success_ticks: list[float] = []
    interval_goal_distances: list[float] = []
    cumulative_success_once: list[float] = []
    active_post_contact = np.zeros(args.num_envs, dtype=np.int64)
    train_start_time = time.perf_counter()
    bar_postfix: dict[str, float | int] = {}
    completed_eval_steps = normalize_completed_eval_steps(
        _completed_eval_steps(tracker.paths.metrics_jsonl_path),
        total_env_steps=args.total_steps,
        every_n_steps=args.eval_every_steps,
        align_to=args.num_envs,
    )
    eval_schedule = build_aligned_schedule(
        args.total_steps,
        args.eval_every_steps,
        args.num_envs,
    )
    checkpoint_schedule = build_aligned_schedule(
        args.total_steps,
        args.checkpoint_every,
        args.num_envs,
    )
    next_eval_index = next_pending_schedule_index(eval_schedule, completed_eval_steps)
    next_checkpoint_index = advance_schedule_index(checkpoint_schedule, total_steps)
    eval_index = _schedule_eval_index(eval_schedule, completed_eval_steps)
    last_ckpt = (
        max(s for s in checkpoint_schedule if s <= total_steps)
        if any(s <= total_steps for s in checkpoint_schedule)
        else 0
    )

    print(
        f"[{run_label}] Training for {args.total_steps} steps "
        f"({args.random_exploration_steps} random, metrics every {args.log_every}"
        f"{f', eval every {args.eval_every_steps} (align {args.num_envs})' if eval_schedule else ''})"
    )
    if completed_eval_steps:
        next_step = eval_schedule[next_eval_index] if next_eval_index < len(eval_schedule) else None
        print(
            f"[{run_label}] skipping {len(completed_eval_steps)} eval milestone(s) "
            f"already in {tracker.paths.metrics_jsonl_path.name}; next @ {next_step}"
        )

    progress = tqdm(
        total=args.total_steps,
        initial=total_steps,
        desc=run_label,
        unit="step",
        dynamic_ncols=True,
        disable=not args.progress_bar,
        mininterval=1.0,
        smoothing=0.05,
    )

    def _drain_due_evals() -> None:
        nonlocal next_eval_index, eval_index
        while (
            next_eval_index < len(eval_schedule)
            and total_steps >= eval_schedule[next_eval_index]
        ):
            eval_step = eval_schedule[next_eval_index]
            if eval_step not in completed_eval_steps:
                _run_scheduled_eval(
                    agent=agent,
                    tracker=tracker,
                    args=args,
                    total_steps=eval_step,
                    eval_index=eval_index,
                    run_label=run_label,
                    progress=progress,
                )
                completed_eval_steps.add(eval_step)
                eval_index += 1
            next_eval_index += 1

    try:
        _drain_due_evals()
        while total_steps < args.total_steps:
            if total_steps < args.random_exploration_steps:
                action = vec_env.action_space.sample()
            else:
                action = agent.sample_actions(obs, desired_goals)

            raw_next, _, terminated, truncated, infos = vec_env.step(action)
            next_obs = raw_next.reshape(args.num_envs, state_dim).astype(np.float32)
            done = (np.asarray(terminated) | np.asarray(truncated)).astype(bool)

            try:
                next_desired = _stack_from_info(infos, "desired_goal", args.num_envs)
            except KeyError:
                next_desired = desired_goals
            try:
                next_achieved = _stack_from_info(
                    infos, "achieved_goal", args.num_envs, fallback=next_obs
                )
            except KeyError:
                next_achieved = next_obs[:, 4:6].copy() if state_dim >= 6 else next_obs.copy()

            transition_obs = _stack_transition_states(
                infos, args.num_envs, fallback=next_obs
            )
            try:
                transition_achieved = _stack_transition_goals(
                    infos,
                    args.num_envs,
                    key="achieved_goal",
                    fallback=next_achieved,
                )
            except KeyError:
                transition_achieved = next_achieved

            try:
                transition_hand = _stack_from_info(
                    infos, "transition_hand_position", args.num_envs, fallback=hand_positions
                )
                transition_object = _stack_from_info(
                    infos, "transition_object_position", args.num_envs, fallback=object_positions
                )
            except KeyError:
                transition_hand = hand_positions
                transition_object = object_positions

            phase_labels, active_post_contact = compute_phase_labels(
                hand_positions,
                object_positions,
                transition_hand,
                transition_object,
                active_post_contact,
                contact_distance=args.contact_distance,
                near_contact_distance=args.near_contact_distance,
                near_goal_distance=args.near_goal_distance,
                post_contact_steps=args.post_contact_steps,
                transition_goal_distance=np.array(
                    [
                        _info_field(infos, "transition_goal_distance", i)
                        for i in range(args.num_envs)
                    ],
                    dtype=np.float32,
                ),
                dones=done,
            )
            step_hand_object_dist = hand_object_distance(
                hand_positions,
                object_positions,
                transition_hand,
                transition_object,
            )

            success = _transition_success(infos, args.num_envs)
            interval_goal_distances.append(
                _transition_goal_distance_mean(infos, args.num_envs)
            )
            train_success_once |= success
            train_success_ticks += success.astype(np.float32)
            if done.any():
                for i in np.nonzero(done)[0]:
                    once = float(train_success_once[i])
                    ticks = float(train_success_ticks[i])
                    interval_success_once.append(once)
                    interval_success_ticks.append(ticks)
                    cumulative_success_once.append(once)
                    train_success_once[i] = False
                    train_success_ticks[i] = 0.0

            replay.append(
                states=obs,
                achieved_goals=achieved_goals,
                actions=action.astype(np.float32),
                next_states=transition_obs,
                next_achieved_goals=transition_achieved,
                dones=done,
                reset_states=next_obs,
                reset_achieved_goals=next_achieved,
                phase_labels=phase_labels,
                hand_object_distances=step_hand_object_dist,
            )
            obs = next_obs
            desired_goals = next_desired
            achieved_goals = next_achieved
            try:
                hand_positions = _stack_from_info(infos, "hand_position", args.num_envs)
                object_positions = _stack_from_info(infos, "object_position", args.num_envs)
            except KeyError:
                hand_positions = next_obs[:, 12:14].copy() if state_dim >= 14 else next_obs[:, 0:2].copy()
                object_positions = next_obs[:, 27:29].copy() if state_dim >= 29 else next_obs[:, 4:6].copy()
            total_steps += args.num_envs
            progress.update(args.num_envs)

            elapsed_total = max(time.perf_counter() - train_start_time, 1e-6)
            bar_postfix["sps"] = int(total_steps / elapsed_total)
            progress.set_postfix(bar_postfix, refresh=False)

            if (
                total_steps >= args.random_exploration_steps
                and replay.stats().get("replay/transitions", 0) >= args.min_replay_size
                and replay.can_sample(args.batch_size)
            ):
                for _ in range(args.updates_per_iter):
                    batch = replay.sample(
                        args.batch_size,
                        future_discount=args.future_discount,
                        context_repetition_factor=args.context_repetition_factor,
                        contact_sigma=args.contact_sigma,
                        random_fraction=args.iws_random_fraction,
                        full_episode_future=args.iws_full_episode_future,
                        event_selection=args.iws_event_selection,
                        use_iws=args.use_iws,
                    )
                    last_metrics = agent.update({
                        "states": batch.states,
                        "next_states": batch.next_states,
                        "goals": batch.goals,
                        "actions": batch.actions,
                    })
                    update_metrics.add(last_metrics)
                    total_updates += 1

            while (
                next_checkpoint_index < len(checkpoint_schedule)
                and total_steps >= checkpoint_schedule[next_checkpoint_index]
            ):
                ckpt_step = checkpoint_schedule[next_checkpoint_index]
                if ckpt_step > last_ckpt:
                    ckpt_path = ckpt_dir / f"checkpoint_{ckpt_step:08d}.pt"
                    torch.save({
                        "agent": agent.state_dict(),
                        "total_steps": ckpt_step,
                        "total_updates": total_updates,
                        "state_dim": state_dim,
                        "goal_dim": goal_dim,
                        "action_dim": action_dim,
                        "action_low": action_low.tolist(),
                        "action_high": action_high.tolist(),
                        "hidden_dims": list(args.hidden_dims),
                        "repr_dim": args.repr_dim,
                        "critic_score_type": args.critic_score_type,
                        "config": config_dict,
                    }, ckpt_path)
                    saved_checkpoints.append({"path": str(ckpt_path), "steps": ckpt_step})
                    progress.write(f"[{run_label}] checkpoint → {ckpt_path}")
                    last_ckpt = ckpt_step
                next_checkpoint_index += 1

            _drain_due_evals()

            if total_steps - last_log >= args.log_every:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                avg_update_metrics = update_metrics.mean()
                if not avg_update_metrics:
                    avg_update_metrics = last_metrics
                metrics = {
                    "train/env_steps_per_second": float((total_steps - last_log) / elapsed),
                    "train/updates": float(total_updates),
                    "train/goal_distance_mean": float(np.mean(interval_goal_distances))
                    if interval_goal_distances
                    else 0.0,
                    **replay.stats(),
                    **avg_update_metrics,
                    **_flush_success_metrics(interval_success_once, interval_success_ticks),
                }
                interval_goal_distances.clear()
                if cumulative_success_once:
                    metrics["train/success_once_cumulative"] = float(
                        np.mean(cumulative_success_once)
                    )
                tracker.log_metrics(metrics, step=total_steps)
                interval_success_once.clear()
                interval_success_ticks.clear()
                bar_postfix = _progress_postfix(metrics, live_sps=float(bar_postfix["sps"]))
                progress.set_postfix(bar_postfix, refresh=False)
                update_metrics.reset()
                last_log = total_steps
                last_log_time = now
    finally:
        progress.close()
        remaining_success = _flush_success_metrics(interval_success_once, interval_success_ticks)
        if remaining_success:
            tracker.log_metrics({**remaining_success, "train/env_steps": float(total_steps)}, step=total_steps)

    vec_env.close()

    if total_steps > last_ckpt or not saved_checkpoints:
        ckpt_path = ckpt_dir / f"checkpoint_{total_steps:08d}.pt"
        torch.save({
            "agent": agent.state_dict(),
            "total_steps": total_steps,
            "total_updates": total_updates,
            "state_dim": state_dim,
            "goal_dim": goal_dim,
            "action_dim": action_dim,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "hidden_dims": list(args.hidden_dims),
            "repr_dim": args.repr_dim,
            "critic_score_type": args.critic_score_type,
            "config": config_dict,
        }, ckpt_path)
        saved_checkpoints.append({"path": str(ckpt_path), "steps": total_steps})
        print(f"[{run_label}] final checkpoint → {ckpt_path}")

    manifest = {
        "env": args.env,
        "total_steps": total_steps,
        "run_dir": str(tracker.paths.run_dir),
        "checkpoints": saved_checkpoints,
    }
    (ckpt_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    tracker.finish()
    print(f"[{run_label}] Done. {len(saved_checkpoints)} checkpoints; logs in {tracker.paths.run_dir}")


def _parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--algorithm",
        choices=["sgcrl", "iws"],
        default=defaults.algorithm,
        help="sgcrl = UM uniform replay baseline; iws = interaction-weighted sampling.",
    )
    ap.add_argument("--env", default=defaults.env)
    ap.add_argument(
        "--air-hockey-config",
        default=defaults.air_hockey_config,
        help="Path to air-hockey-rl YAML config (UM real_transfer_s4 by default).",
    )
    ap.add_argument("--num-envs", type=int, default=defaults.num_envs)
    ap.add_argument("--horizon", type=int, default=defaults.horizon)
    ap.add_argument("--max-episode-steps", type=int, default=defaults.max_episode_steps)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--device", default=defaults.device)

    ap.add_argument("--total-steps", type=int, default=defaults.total_steps)
    ap.add_argument("--random-exploration-steps", type=int, default=defaults.random_exploration_steps)
    ap.add_argument("--min-replay-size", type=int, default=defaults.min_replay_size)
    ap.add_argument(
        "--checkpoint-dir",
        default=defaults.checkpoint_dir,
        help="Override checkpoint directory (default: <run_dir>/checkpoints).",
    )
    ap.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    ap.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint .pt file or directory (uses latest checkpoint_*.pt) to resume training.",
    )
    ap.add_argument(
        "--resume-run-dir",
        default=None,
        help="Existing run directory to append metrics/checkpoints (use with --resume-from).",
    )
    ap.add_argument("--log-every", type=int, default=defaults.log_every)
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        default=defaults.eval_every_steps,
        help="Run deterministic eval every N env steps (0 disables). UM default: 500000.",
    )
    ap.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    ap.add_argument("--eval-num-envs", type=int, default=defaults.eval_num_envs)
    ap.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=defaults.eval_deterministic,
    )
    ap.add_argument(
        "--eval-capture-video",
        action=argparse.BooleanOptionalAction,
        default=defaults.eval_capture_video,
    )
    ap.add_argument(
        "--eval-capture-success-video",
        action=argparse.BooleanOptionalAction,
        default=defaults.eval_capture_success_video,
    )
    ap.add_argument("--eval-video-fps", type=int, default=defaults.eval_video_fps)

    ap.add_argument("--hidden-dims", type=int, nargs="+", default=list(defaults.hidden_dims))
    ap.add_argument("--repr-dim", type=int, default=defaults.repr_dim)
    ap.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    ap.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    ap.add_argument("--entropy-coeff", type=float, default=defaults.entropy_coeff)
    ap.add_argument("--logsumexp-coeff", type=float, default=defaults.logsumexp_coeff)
    ap.add_argument("--random-goals", type=float, default=defaults.random_goals)
    ap.add_argument("--use-action-entropy", action=argparse.BooleanOptionalAction, default=defaults.use_action_entropy)
    ap.add_argument(
        "--critic-score-type",
        choices=["dot", "cosine"],
        default=defaults.critic_score_type,
    )

    ap.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    ap.add_argument("--batch-size", type=int, default=defaults.batch_size)
    ap.add_argument("--updates-per-iter", type=int, default=defaults.updates_per_iter)
    ap.add_argument("--future-discount", type=float, default=defaults.future_discount)
    ap.add_argument("--context-repetition-factor", type=int, default=defaults.context_repetition_factor)
    ap.add_argument("--contact-distance", type=float, default=defaults.contact_distance)
    ap.add_argument("--contact-sigma", type=float, default=defaults.contact_sigma)
    ap.add_argument("--near-contact-distance", type=float, default=defaults.near_contact_distance)
    ap.add_argument("--near-goal-distance", type=float, default=defaults.near_goal_distance)
    ap.add_argument("--post-contact-steps", type=int, default=defaults.post_contact_steps)
    ap.add_argument("--iws-random-fraction", type=float, default=defaults.iws_random_fraction)
    ap.add_argument(
        "--iws-full-episode-future",
        action=argparse.BooleanOptionalAction,
        default=defaults.iws_full_episode_future,
    )
    ap.add_argument("--iws-event-selection", choices=["uniform", "first"], default=defaults.iws_event_selection)

    ap.add_argument("--output-root", default=defaults.output_root)
    ap.add_argument("--wandb-project", default=defaults.wandb_project)
    ap.add_argument("--run-name", default=defaults.run_name)
    ap.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=defaults.use_wandb)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=defaults.wandb_mode)
    ap.add_argument("--wandb-entity", default=defaults.wandb_entity)
    ap.add_argument("--wandb-group", default=defaults.wandb_group)
    ap.add_argument("--wandb-tag", dest="wandb_tags", action="append")
    ap.add_argument("--no-progress-bar", action="store_false", dest="progress_bar", default=defaults.progress_bar)

    args = ap.parse_args()
    if args.wandb_tags is None:
        args.wandb_tags = []
    return args


if __name__ == "__main__":
    run_training(_parse_args())
