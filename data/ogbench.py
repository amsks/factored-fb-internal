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

import numpy as np

from buffers.transition import DictBuffer


def load_transitions(npz_files: List[Union[str, Path]]) -> Dict:
    """Load episode .npz files into a flat transition dict.

    Returns a dict with keys:
        observation                [N, obs_dim]
        action                     [N, act_dim]
        next.observation           [N, obs_dim]
        next.terminated            [N, 1]    bool
        next.physics               [N, phys_dim]   if present
    """
    obs_list, act_list, next_obs_list, term_list, next_phys_list = [], [], [], [], []
    has_physics = True

    for f in npz_files:
        ep = np.load(f)
        obs = ep["observation"].astype(np.float32)
        act = ep["action"].astype(np.float32)
        disc = ep["discount"].astype(np.float32) if "discount" in ep else np.ones(len(obs))

        # IMPORTANT: scripts/extract_ogbench.py applies an EXORL-style shift so
        # that episode["action"][t] is the action that LED to obs[t], with
        # episode["action"][0] = 0. We undo that shift here by taking act[1:],
        # matching td_jepa's metamotivo/data_loading/base.py::load_transitions.
        # The pair (obs_list[i], act_list[i], next_obs_list[i]) is then
        # (obs[i], action_taken_from_obs[i], obs[i+1]) — standard RL convention.
        obs_list.append(obs[:-1])
        act_list.append(act[1:])
        next_obs_list.append(obs[1:])
        term_list.append((disc[1:] == 0.0).reshape(-1, 1))

        if has_physics and "physics" in ep:
            next_phys_list.append(ep["physics"].astype(np.float32)[1:])
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
    if has_physics and next_phys_list:
        storage["next"]["physics"] = np.concatenate(next_phys_list, axis=0)

    return storage


def load_ogbench_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
) -> DictBuffer:
    """Load offline OGBench data into a DictBuffer.

    Parameters
    ----------
    domain : e.g. "antmaze-medium-navigate-v0"
    data_path : root dir; episodes are at <data_path>/<domain>/buffer/*.npz
    load_n_episodes : max number of episodes to load
    device : device for sampled tensors
    n_transitions : if set, truncate the buffer to this many transitions
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    storage = load_transitions(files)

    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]

    n = len(storage["observation"])
    buffer = DictBuffer(capacity=n, device=device)
    buffer.extend(storage)
    return buffer
