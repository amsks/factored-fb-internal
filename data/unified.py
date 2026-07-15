"""data/unified.py — unified per-episode trajectory format + loader (P1).

One on-disk schema, a superset of the OGBench layout, produced by the
OGBench->unified converter (scripts/data/convert_ogbench_to_unified.py) and by the
collectors, and consumed by FB training via a ``DictBuffer``.

Layout: per-episode ``.npz`` at
``<data_path>/<sanitized-name>/buffer/episode_XXXXXX_<T>.npz`` plus a dataset-level
``meta.json``. Rows are already aligned transitions (row t = the transition out of
step t), so — unlike the OGBench loader — NO EXORL action shift is applied here;
the converter bakes the alignment in.

Per-episode arrays (N transition rows):
    observation       [N, obs_dim]   float32   # s_t
    action            [N, act_dim]   float32   # a_t
    next_observation  [N, obs_dim]   float32   # s_{t+1}
    achieved_goal     [N, g_dim]     float32   # goal-relevant slice at s_{t+1}
                                               #   (generalizes OGBench next/physics)
    reward            [N]            float32
    terminal          [N]            bool
    timeout           [N]            bool
    success           [N]            bool

The loader emits a ``DictBuffer`` with the same nested layout FB agents expect
(``observation``/``action``/``next/{observation,terminated,action,achieved_goal,reward}``
/``timestep``). FB training reads only observation/action/next/{observation,terminated};
achieved_goal + reward are used by the goal evaluator for z-inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from buffers.transition import DictBuffer

META_FILENAME = "meta.json"


def sanitize_name(name: str) -> str:
    """`Metaworld/reach` -> `Metaworld__reach`; `cube-single-play-v0` unchanged."""
    return name.replace("/", "__")


def dataset_dir(data_path: Union[str, Path], name: str) -> Path:
    return Path(data_path) / sanitize_name(name)


# ── writing (converter + collectors) ────────────────────────────────────────

def write_episode(buffer_dir: Union[str, Path], index: int, traj: Dict[str, np.ndarray]) -> Path:
    buffer_dir = Path(buffer_dir)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    n = len(traj["observation"])
    path = buffer_dir / f"episode_{index:06d}_{n:05d}.npz"
    np.savez(
        path,
        observation=np.asarray(traj["observation"], dtype=np.float32),
        action=np.asarray(traj["action"], dtype=np.float32),
        next_observation=np.asarray(traj["next_observation"], dtype=np.float32),
        achieved_goal=np.asarray(traj["achieved_goal"], dtype=np.float32),
        reward=np.asarray(traj["reward"], dtype=np.float32).reshape(-1),
        terminal=np.asarray(traj["terminal"], dtype=bool).reshape(-1),
        timeout=np.asarray(traj["timeout"], dtype=bool).reshape(-1),
        success=np.asarray(traj["success"], dtype=bool).reshape(-1),
    )
    return path


def write_meta(root: Union[str, Path], meta: Dict[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / META_FILENAME
    with path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def read_meta(root: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(root) / META_FILENAME
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# ── loading (training) ──────────────────────────────────────────────────────

def _load_transitions(files: List[Path]) -> Dict[str, Any]:
    obs, act, nobs, ag, rew, term, nact, ts = [], [], [], [], [], [], [], []
    for f in files:
        ep = np.load(f)
        a = ep["action"].astype(np.float32)
        n = a.shape[0]
        obs.append(ep["observation"].astype(np.float32))
        act.append(a)
        nobs.append(ep["next_observation"].astype(np.float32))
        ag.append(ep["achieved_goal"].astype(np.float32))
        rew.append(ep["reward"].astype(np.float32).reshape(-1, 1))
        term.append(ep["terminal"].astype(bool).reshape(-1, 1))
        # SARSA next-action a'_t = a_{t+1} within the episode (last row padded);
        # used only by one-step FB. Mirrors data/ogbench.py's next-action padding.
        nact.append(np.concatenate([a[1:], a[-1:]], axis=0) if n > 1 else a.copy())
        ts.append(np.arange(n, dtype=np.int32))
    return {
        "observation": np.concatenate(obs, axis=0),
        "action": np.concatenate(act, axis=0),
        "next": {
            "observation": np.concatenate(nobs, axis=0),
            "terminated": np.concatenate(term, axis=0),
            "action": np.concatenate(nact, axis=0),
            "achieved_goal": np.concatenate(ag, axis=0),
            "reward": np.concatenate(rew, axis=0),
        },
        "timestep": np.concatenate(ts, axis=0),
    }


def load_dataset(
    name: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: Optional[int] = None,
) -> DictBuffer:
    """Load a unified-format dataset for ``name`` into a DictBuffer."""
    buf_dir = dataset_dir(data_path, name) / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No unified .npz episodes at {buf_dir}. Convert/collect data first "
            f"(scripts/data/convert_ogbench_to_unified.py or collectors/)."
        )
    storage = _load_transitions(files)
    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]
    buffer = DictBuffer(capacity=len(storage["observation"]), device=device)
    buffer.extend(storage)
    return buffer
