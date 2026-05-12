"""data/hecrl.py — HECRL ManipObj dataset loader.

The pre-collected HF datasets are a single flat .npz per task with keys:
    observations         pixel observations (uint8)         -- IGNORED
    state_observations   [N, 6+5*num_cubes] float32         -- loaded as 'observation'
    actions              [N, 5] float32
    qpos                 [N, qpos_dim] float32              -- loaded as next.physics
    terminals            [N] bool

We split into episodes via `terminals`, then build T-1 transitions per
episode. The DictBuffer batch layout matches data.ogbench.load_ogbench_dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from buffers.transition import DictBuffer


HF_FILENAME = {
    "ppp-cube": "ppp-cube-noisy-v0-mv.npz",
    # Add new entries here as we adopt more HECRL datasets.
}


def load_hecrl_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets/hecrl",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
) -> DictBuffer:
    """Load a HECRL HF dataset into a DictBuffer.

    Parameters
    ----------
    domain          : short canonical name, e.g. "ppp-cube"
    data_path       : directory containing <HF_FILENAME[domain]>
    load_n_episodes : max number of episodes to materialise
    device          : torch device for sampled tensors
    n_transitions   : if set, truncate the buffer to this many transitions
    """
    if domain not in HF_FILENAME:
        raise ValueError(
            f"Unknown HECRL domain: {domain!r}. Known: {sorted(HF_FILENAME)}"
        )

    npz_path = Path(data_path) / HF_FILENAME[domain]
    if not npz_path.exists():
        raise FileNotFoundError(
            f"HECRL dataset not found at {npz_path}. "
            f"Run: python scripts/download_hecrl.py {domain}"
        )

    raw = np.load(npz_path)
    state_obs = raw["state_observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    qpos = raw["qpos"].astype(np.float32)
    terminals = raw["terminals"].astype(bool)

    # Episode boundaries from terminals.
    ends = np.nonzero(terminals.ravel())[0]
    if ends.size == 0:
        raise ValueError(
            f"No episode terminals found in {npz_path}. "
            "Dataset appears corrupt or has different schema."
        )
    starts = np.concatenate(([0], ends[:-1] + 1))
    starts, ends = starts[:load_n_episodes], ends[:load_n_episodes]

    obs_list, act_list, next_obs_list, term_list, next_qpos_list = [], [], [], [], []
    for s, e in zip(starts, ends):
        # T-step episode -> T-1 transitions (last step has no successor).
        if e <= s:
            continue
        obs_list.append(state_obs[s:e])
        act_list.append(actions[s:e])
        next_obs_list.append(state_obs[s + 1:e + 1])
        next_qpos_list.append(qpos[s + 1:e + 1])
        term_e = np.zeros(e - s, dtype=bool)
        term_e[-1] = True
        term_list.append(term_e.reshape(-1, 1))

    storage: Dict = {
        "observation": np.concatenate(obs_list),
        "action": np.concatenate(act_list),
        "next": {
            "observation": np.concatenate(next_obs_list),
            "terminated": np.concatenate(term_list),
            "physics": np.concatenate(next_qpos_list),   # qpos -> physics alias
        },
    }

    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]

    n = len(storage["observation"])
    buf = DictBuffer(capacity=n, device=device)
    buf.extend(storage)
    return buf
