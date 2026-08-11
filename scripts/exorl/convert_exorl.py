"""Spec 23 P0 -- convert ExoRL episodic .npz buffers into the repo's
GCDataset format (observations / actions / next_observations / terminals).

ExoRL episode schema (verified against real downloaded data, 2026-08-11 --
see .superpowers/sdd/spec23-p0/task-report.md): each
`episode_<idx>_<N>.npz` stores raw arrays of length N+1 (a "dummy first
transition" convention from `exorl/replay_buffer.py`):

    observation[0 .. N]   -- N+1 states; observation[0] is the true reset obs
    action[0 .. N]        -- action[0] is a dummy [0, 0] padding row;
                              action[1 .. N] are the N real actions taken
    reward, discount, physics -- NOT used here (rewards enter only at eval,
                              per spec 23 section 4.2; physics == observation
                              for point_mass_maze, redundant)

The N real transitions are (observation[t], action[t+1], observation[t+1])
for t = 0 .. N-1. This module turns that into the repo's non-compact
(explicit next_observations) Dataset format so no off-by-one inference is
ever needed at load time -- see third_party/ogbench/ogbench/utils.py
`load_dataset(compact_dataset=...)` for the defect class this sidesteps
(compact-dataset next-observation inference silently reads into the wrong
row at a trajectory's own boundary if 'valids' is mishandled). We always
emit 'next_observations' explicitly instead.

Output: two .npz files (train + '-val') with float32 keys
`observations`, `actions`, `next_observations`, `terminals` -- directly
consumable by `third_party/ogbench/impls/utils/datasets.py`'s
`Dataset.create(**np.load(path))` / `GCDataset`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConvertStats:
    n_episodes_total: int
    n_episodes_train: int
    n_episodes_val: int
    n_transitions_train: int
    n_transitions_val: int
    obs_dim: int
    action_dim: int
    source_bytes: int
    train_bytes: int
    val_bytes: int


def _episode_files(buffer_dir: Path) -> list[Path]:
    files = sorted(buffer_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no episode .npz files under {buffer_dir}")
    return files


def _load_episode_transitions(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (observations, actions, next_observations) for the N real
    transitions in one episode file, per the dummy-first-transition schema
    documented in the module docstring."""
    with np.load(fn) as f:
        obs = f["observation"]
        act = f["action"]
    assert obs.shape[0] == act.shape[0], (fn, obs.shape, act.shape)
    n_declared = None
    stem_parts = fn.stem.split("_")
    if len(stem_parts) >= 2 and stem_parts[-1].isdigit():
        n_declared = int(stem_parts[-1])
    n = obs.shape[0] - 1
    if n_declared is not None:
        assert n == n_declared, (
            f"{fn}: filename declares length {n_declared} but "
            f"observation array implies {n} real transitions (shape {obs.shape})"
        )
    observations = obs[:-1]
    next_observations = obs[1:]
    actions = act[1:]  # drop the dummy action[0]
    return observations, actions, next_observations


def convert_domain(
    buffer_dir: str | Path,
    out_prefix: str | Path,
    max_episodes: int | None = None,
    val_episodes: int = 50,
) -> ConvertStats:
    """Convert an ExoRL `<domain>/<algo>/buffer/` directory into
    `<out_prefix>.npz` (train) + `<out_prefix>-val.npz` (val).

    Split: the LAST `val_episodes` episode files (by sorted filename order,
    i.e. collection order / episode index) are held out as val; everything
    before them is train. Deterministic, no shuffling (matches the ExoRL
    file-index-is-collection-order convention; a shuffle would mix
    early-vs-late-training behavior across the split).
    """
    buffer_dir = Path(buffer_dir)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    files = _episode_files(buffer_dir)
    if max_episodes is not None:
        files = files[:max_episodes]
    if val_episodes >= len(files):
        raise ValueError(
            f"val_episodes={val_episodes} >= n_episodes={len(files)}"
        )
    train_files = files[: len(files) - val_episodes] if val_episodes > 0 else files
    val_files = files[len(files) - val_episodes :] if val_episodes > 0 else []

    def _build(split_files: list[Path]):
        obs_list, act_list, next_obs_list, term_list = [], [], [], []
        for fn in split_files:
            o, a, no = _load_episode_transitions(fn)
            t = np.zeros(o.shape[0], dtype=np.float32)
            t[-1] = 1.0
            obs_list.append(o.astype(np.float32))
            act_list.append(a.astype(np.float32))
            next_obs_list.append(no.astype(np.float32))
            term_list.append(t)
        return (
            np.concatenate(obs_list, axis=0),
            np.concatenate(act_list, axis=0),
            np.concatenate(next_obs_list, axis=0),
            np.concatenate(term_list, axis=0),
        )

    train_obs, train_act, train_next_obs, train_term = _build(train_files)
    train_path = out_prefix.with_suffix(".npz")
    np.savez(
        train_path,
        observations=train_obs,
        actions=train_act,
        next_observations=train_next_obs,
        terminals=train_term,
    )

    val_bytes = 0
    n_val_transitions = 0
    if val_files:
        val_obs, val_act, val_next_obs, val_term = _build(val_files)
        val_path = Path(str(out_prefix) + "-val.npz")
        np.savez(
            val_path,
            observations=val_obs,
            actions=val_act,
            next_observations=val_next_obs,
            terminals=val_term,
        )
        val_bytes = val_path.stat().st_size
        n_val_transitions = val_term.shape[0]

    source_bytes = sum(f.stat().st_size for f in files)
    stats = ConvertStats(
        n_episodes_total=len(files),
        n_episodes_train=len(train_files),
        n_episodes_val=len(val_files),
        n_transitions_train=train_term.shape[0],
        n_transitions_val=n_val_transitions,
        obs_dim=train_obs.shape[1],
        action_dim=train_act.shape[1],
        source_bytes=source_bytes,
        train_bytes=train_path.stat().st_size,
        val_bytes=val_bytes,
    )
    stats_path = Path(str(out_prefix) + ".stats.json")
    stats_path.write_text(json.dumps(stats.__dict__, indent=2))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer-dir", required=True, help="ExoRL .../buffer/ dir")
    p.add_argument("--out-prefix", required=True, help="output path prefix (no .npz)")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--val-episodes", type=int, default=50)
    args = p.parse_args()
    stats = convert_domain(
        args.buffer_dir, args.out_prefix, args.max_episodes, args.val_episodes
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
