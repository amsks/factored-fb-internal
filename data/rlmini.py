"""Offline dataset loader for rlmini_envs episode files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from buffers.transition import DictBuffer
from data.episodes import load_episode_dataset


def load_rlmini_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
) -> DictBuffer:
    """Load transition-aligned rlmini episode files into a ``DictBuffer``."""
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Generate rlmini data with scripts/data/export_rlmini_random.py."
        )
    return load_episode_dataset(
        files,
        device=device,
        n_transitions=n_transitions,
        obs_type=obs_type,
        frame_stack=frame_stack,
        action_alignment="transition",
    )
