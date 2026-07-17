"""scripts/render_dataset_traj.py — render stored OGBench dataset trajectories to video.

Replays the per-step physics (qpos) from the offline .npz episodes and renders each
frame, so you can SEE the actual "play" data (the scripted oracle's noisy, task-
agnostic pick-and-place, re-targeting the cube to random spots within one episode).
This renders the DATASET trajectories themselves, not a policy rollout.

    .venv/bin/python scripts/render_dataset_traj.py \
        --domain cube-single-play-v0 --n-episodes 3 --stride 4 --render-size 256
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.ogbench import create_ogbench_env  # noqa: E402


def render_episode(env, physics: np.ndarray, stride: int) -> list:
    import mujoco

    m = env.unwrapped.model
    d = env.unwrapped.data
    nq = min(d.qpos.shape[0], physics.shape[1])
    frames = []
    for t in range(0, physics.shape[0], stride):
        d.qpos[:nq] = physics[t, :nq]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        frames.append(np.asarray(env.render()))
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="cube-single-play-v0")
    p.add_argument("--data", default="/raid/tongle/rl/fb-data/datasets")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--stride", type=int, default=4, help="subsample steps for the video")
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="/raid/tongle/rl/testbed/sanity/videos/ogbench_play")
    a = p.parse_args()

    import imageio

    env, _ = create_ogbench_env(a.domain, render_height=a.render_size, render_width=a.render_size)
    env.reset()
    files = sorted(glob.glob(f"{a.data}/{a.domain}/buffer/*.npz"))[: a.n_episodes]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(files)} dataset trajectories of {a.domain} (stride {a.stride}, {a.render_size}px)")
    for i, f in enumerate(files):
        ep = np.load(f)
        phys = ep["physics"]
        frames = render_episode(env, phys, a.stride)
        # cube travel = how far the cube moved over the episode (illustrates re-targeting)
        cube = phys[:, 14:17]
        travel = float(np.linalg.norm(np.diff(cube, axis=0), axis=1).sum())
        path = out_dir / f"{a.domain}_ep{i}_{phys.shape[0]}steps.mp4"
        imageio.mimsave(path, frames, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
        print(f"  ep{i}: {phys.shape[0]} steps -> {len(frames)} frames, cube_travel={travel:.2f}m -> {path.name}")


if __name__ == "__main__":
    main()
