"""
data/ogbench.py — Offline dataset loader for OGBench.

Episode .npz files are expected at:
    <data_path>/<domain>/buffer/ep*.npz

Each file stores one episode with keys:
    observation  [T, obs_dim]    float32
    action       [T, act_dim]    float32
    physics      [T, phys_dim]   float32  (used for reward relabelling at eval)
    discount     [T]             float32  (0 at terminal transitions)

Episodes of T timesteps yield T-1 transitions:
    (obs[t], action[t]) -> (obs[t+1], physics[t+1], terminated)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from buffers.transition import DictBuffer
from data.episodes import load_episode_dataset, load_episode_transitions


def load_transitions(
    npz_files: List[Union[str, Path]], obs_type: str = "state"
) -> Dict:
    """Load episode .npz files into a flat transition dict.

    State (obs_type="state") path is unchanged. Pixel (obs_type="pixels")
    path reads the "pixels" key, stores per-frame CHW uint8, and emits an
    episode-relative "timestep" for frame-stack clamping.
    """
    return load_episode_transitions(
        npz_files,
        obs_type=obs_type,
        action_alignment="exorl",
    )


def load_ogbench_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
) -> DictBuffer:
    """Load offline OGBench data into a DictBuffer.

    Parameters
    ----------
    domain : e.g. "antmaze-medium-navigate-v0"
    data_path : root dir; episodes are at <data_path>/<domain>/buffer/*.npz
    load_n_episodes : max number of episodes to load
    device : device for sampled tensors
    n_transitions : if set, truncate the buffer to this many transitions
    obs_type : "state" (default) or "pixels"
    frame_stack : number of frames to stack on sample (pixels only; 1 = off)
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    return load_episode_dataset(
        files,
        device=device,
        n_transitions=n_transitions,
        obs_type=obs_type,
        frame_stack=frame_stack,
        action_alignment="exorl",
    )
