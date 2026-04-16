"""
data/ogbench.py — Offline dataset loader for OGBench.

Episode .npz files are expected at:
    <data_path>/<domain>/buffer/ep*.npz

Each file stores one episode with keys:
    observation  [T, obs_dim]    float32
    action       [T, act_dim]    float32
    physics      [T, phys_dim]   float32  (used for reward relabelling at eval)
    discount     [T]             float32  (0 at terminal transitions)

``load_transitions`` converts T-step episodes into (T-1) transitions and
returns a nested dict compatible with DictBuffer.  ``load_ogbench_data``
wraps multiple episodes into a DictBuffer and returns ``{"train": buffer}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from buffers.transition import DictBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Episode → transitions
# ──────────────────────────────────────────────────────────────────────────────

def load_transitions(npz_files: List[Union[str, Path]]) -> Dict:
    """Load a list of episode .npz files and return a flat transition dict.

    Given T timesteps per episode, produces T-1 transitions:
        (obs[t], action[t]) → (obs[t+1], terminated[t])

    Parameters
    ----------
    npz_files : list of path-like

    Returns
    -------
    dict with keys:
        "observation"          [N, obs_dim]
        "action"               [N, act_dim]
        "physics"              [N, phys_dim]   (if present in episode)
        "next" / "observation" [N, obs_dim]
        "next" / "terminated"  [N]             bool
    """
    obs_list, act_list, next_obs_list, term_list = [], [], [], []
    phys_list: List[np.ndarray] = []
    has_physics = True

    for f in npz_files:
        ep = np.load(f)
        obs = ep["observation"].astype(np.float32)       # [T, obs_dim]
        act = ep["action"].astype(np.float32)            # [T, act_dim]
        disc = ep["discount"].astype(np.float32) if "discount" in ep else np.ones(len(obs))

        obs_list.append(obs[:-1])
        act_list.append(act[:-1])
        next_obs_list.append(obs[1:])
        # terminal when discount drops to 0
        term_list.append((disc[1:] == 0.0))

        if has_physics and "physics" in ep:
            phys_list.append(ep["physics"].astype(np.float32)[:-1])
        else:
            has_physics = False

    storage: Dict = {
        "observation": np.concatenate(obs_list, axis=0),
        "action": np.concatenate(act_list, axis=0),
        "next": {
            "observation": np.concatenate(next_obs_list, axis=0),
            "terminated": np.concatenate(term_list, axis=0),
        },
    }
    if has_physics and phys_list:
        storage["physics"] = np.concatenate(phys_list, axis=0)

    return storage


# ──────────────────────────────────────────────────────────────────────────────
# Full dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_ogbench_data(
    domain: str,
    data_path: Union[str, Path],
    load_n_episodes: int = 1000,
    device: str = "cpu",
) -> Dict[str, DictBuffer]:
    """Load offline OGBench data from local .npz files into a DictBuffer.

    Parameters
    ----------
    domain : str
        E.g. ``"antmaze-medium-navigate-v0"``.
    data_path : str or Path
        Root directory; episodes are at ``<data_path>/<domain>/buffer/*.npz``.
    load_n_episodes : int
        Maximum number of episodes to load.
    device : str
        Device for sampled tensors.

    Returns
    -------
    ``{"train": DictBuffer}``
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]

    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    storage = load_transitions(files)
    n = len(storage["observation"])

    buffer = DictBuffer(capacity=n, device=device)
    buffer.extend(storage)

    return {"train": buffer}