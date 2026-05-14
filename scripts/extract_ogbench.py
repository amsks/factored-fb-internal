"""
scripts/extract_ogbench.py — Split OGBench's monolithic .npz datasets into
per-episode .npz files under <output>/<env_name>/buffer/episode_XXXXXX_<len>.npz.

State-based only (no pixel branch). Adapted from td_jepa's
scripts/data_processing/ogbench/extract_all.py.

Usage:
    python scripts/extract_ogbench.py --output_folder datasets
    python scripts/extract_ogbench.py --output_folder datasets --env antmaze-medium-navigate-v0

By default downloads via `ogbench.utils.make_env_and_datasets` into the OGBench
default cache, then splits into per-episode files Factored-FB can load.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from tqdm import tqdm


ALL_ENVS_STATE = [
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-medium-stitch-v0",
    "antmaze-large-stitch-v0",
    "antmaze-medium-explore-v0",
    "antmaze-large-explore-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
]


def extract(env_name: str, output_folder: str, dataset_path: str | None = None) -> None:
    from ogbench.utils import DEFAULT_DATASET_DIR, make_env_and_datasets

    if dataset_path is not None:
        dataset_file = Path(dataset_path) / (env_name + ".npz")
    else:
        # Triggers download into ogbench's default cache if needed
        make_env_and_datasets(env_name)
        dataset_file = Path(DEFAULT_DATASET_DIR).expanduser() / (env_name + ".npz")

    print(f"[{env_name}] dataset: {dataset_file}")
    out_dir = Path(output_folder) / env_name / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{env_name}] writing episodes to: {out_dir}")

    data = np.load(dataset_file)
    data = {k: data[k] for k in data.keys()}

    nz = data["terminals"].ravel().nonzero()[0]
    ends = np.arange(data["terminals"].shape[0])[nz]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts

    for ep_idx, (start, length, end) in enumerate(tqdm(list(zip(starts, lengths, ends)))):
        ep = {
            "observation": data["observations"][start : end + 1],
            "action": data["actions"][start : end + 1].copy(),
            "physics": data["qpos"][start : end + 1],
            "reward": np.zeros((length + 1, 1), dtype=np.float32),
            "discount": np.ones((length + 1, 1), dtype=np.float32),
        }
        # Shift actions so action[t] is the action that led to observation[t].
        # First-step action is set to zero (matches td_jepa convention).
        ep["action"][1:] = ep["action"][:-1]
        ep["action"][0] = 0.0
        # NOTE: td_jepa's extract_all.py leaves discount as all ones and never
        # marks a terminal step. We match that here. The loader treats every
        # transition as non-terminal (terminated = False everywhere), so the
        # TD bootstrap always uses discount = gamma = 0.99.
        # Some envs have additional button states concatenated into physics
        if "button_states" in data:
            ep["physics"] = np.concatenate([ep["physics"], data["button_states"][start : end + 1]], axis=-1)

        filename = f"episode_{ep_idx:06d}_{length}.npz"
        np.savez_compressed(out_dir / filename, **ep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_folder", default="datasets",
                    help="Where to write <env>/buffer/episode_*.npz (default: datasets)")
    ap.add_argument("--dataset_path", default=None,
                    help="Optional: path containing pre-downloaded <env>.npz files. "
                         "If omitted, ogbench downloads to its default cache.")
    ap.add_argument("--env", default=None,
                    help="A single OGBench env name. If omitted, extracts all state envs.")
    args = ap.parse_args()

    envs = [args.env] if args.env else ALL_ENVS_STATE
    for env_name in envs:
        extract(env_name, args.output_folder, args.dataset_path)


if __name__ == "__main__":
    main()
