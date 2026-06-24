"""Generic episode-file loading utilities.

Episode files store full trajectories as arrays with time as the first axis.
This module converts them into the nested transition layout consumed by
``DictBuffer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Union

import numpy as np

from buffers.transition import DictBuffer

ActionAlignment = Literal["exorl", "transition"]


def _episode_observation(ep, f: Union[str, Path], obs_type: str):
    if obs_type == "pixels":
        if "pixels" not in ep:
            raise KeyError(
                f"obs_type='pixels' but episode {f} has no 'pixels' key "
                f"(keys: {list(ep.keys())})"
            )
        # [T, H, W, C] uint8 -> [T, C, H, W] uint8 (CHW per frame)
        pix = np.moveaxis(ep["pixels"], -1, 1)
        return np.ascontiguousarray(pix)
    if obs_type == "state":
        return ep["observation"].astype(np.float32)
    raise ValueError(f"Unsupported obs_type {obs_type!r}")


def load_episode_transitions(
    npz_files: Iterable[Union[str, Path]],
    *,
    obs_type: str = "state",
    action_alignment: ActionAlignment = "exorl",
) -> Dict:
    """Load episode ``.npz`` files into a flat transition dict.

    ``action_alignment="exorl"`` preserves the existing OGBench/td_jepa
    convention where ``action[t]`` led to ``observation[t]`` and ``action[0]``
    is padding. ``"transition"`` is the standard Gym convention where
    ``action[t]`` maps ``observation[t]`` to ``observation[t + 1]``.
    """
    obs_list, act_list, next_obs_list, term_list, next_phys_list = [], [], [], [], []
    next_act_list = []
    reward_list = []
    ts_list = []
    has_physics = True
    has_reward = True

    for f in npz_files:
        ep = np.load(f)
        act = ep["action"].astype(np.float32)
        obs_full = _episode_observation(ep, f, obs_type)
        n_trans = obs_full.shape[0] - 1
        if n_trans <= 0:
            continue

        if action_alignment == "exorl":
            action = act[1:]
            next_action = np.concatenate([act[2:], act[-1:]], axis=0)
        elif action_alignment == "transition":
            action = act[:-1]
            next_action = np.concatenate([act[1:], act[-1:]], axis=0)
        else:
            raise ValueError(f"Unsupported action_alignment {action_alignment!r}")

        obs_list.append(obs_full[:-1])
        act_list.append(action)
        next_obs_list.append(obs_full[1:])
        next_act_list.append(next_action)
        ts_list.append(np.arange(n_trans, dtype=np.int32))

        if "terminated" in ep:
            term = ep["terminated"][1:].reshape(n_trans, 1).astype(bool)
        elif "discount" in ep:
            term = (ep["discount"][1:].reshape(n_trans, 1) <= 0)
        else:
            term = np.zeros((n_trans, 1), dtype=bool)
        term_list.append(term)

        if has_reward and "reward" in ep:
            reward_list.append(ep["reward"][1:].reshape(n_trans, 1).astype(np.float32))
        else:
            has_reward = False

        if has_physics and "physics" in ep:
            phys = ep["physics"].astype(np.float32)
            next_phys_list.append(phys[1:])
        else:
            has_physics = False

    if not obs_list:
        raise ValueError("No transitions found in episode files")

    storage: Dict = {
        "observation": np.concatenate(obs_list, axis=0),
        "action": np.concatenate(act_list, axis=0),
        "next": {
            "observation": np.concatenate(next_obs_list, axis=0),
            "terminated": np.concatenate(term_list, axis=0),
            "action": np.concatenate(next_act_list, axis=0),
        },
        "timestep": np.concatenate(ts_list, axis=0),
    }
    if has_reward and reward_list:
        storage["reward"] = np.concatenate(reward_list, axis=0)
    if has_physics and next_phys_list:
        storage["next"]["physics"] = np.concatenate(next_phys_list, axis=0)
    return storage


def truncate_storage(storage: Dict, n_transitions: int | None) -> Dict:
    if n_transitions is None:
        return storage
    out = {}
    for k, v in storage.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
        else:
            out[k] = v[:n_transitions]
    return out


def load_episode_dataset(
    files: Iterable[Union[str, Path]],
    *,
    device: str = "cpu",
    n_transitions: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
    action_alignment: ActionAlignment = "exorl",
) -> DictBuffer:
    storage = load_episode_transitions(
        files,
        obs_type=obs_type,
        action_alignment=action_alignment,
    )
    storage = truncate_storage(storage, n_transitions)
    n = len(storage["observation"])
    buffer = DictBuffer(capacity=n, device=device, frame_stack=frame_stack, obs_type=obs_type)
    buffer.extend(storage)
    return buffer
