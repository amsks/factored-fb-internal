"""scripts/rollout_ogbench_cube.py — one random-policy rollout video per OGBench task.

Sanity rollouts (analogous to the ManiSkill video set): for each singletask variant
of an OGBench domain, roll a uniform-random policy and save a GIF of the rendered
episode. Defaults to cube-single-play-v0 (its 5 singletask eval tasks).

    .venv/bin/python scripts/rollout_ogbench_cube.py            # cube-single, 5 tasks
    .venv/bin/python scripts/rollout_ogbench_cube.py --domain scene-play-v0
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.ogbench import ALL_TASKS, create_ogbench_env  # noqa: E402


def rollout(task: str, steps: int, render_size: int, seed: int, out_dir: Path):
    import imageio

    env, _ = create_ogbench_env(task, seed=seed, render_height=render_size, render_width=render_size)
    env.reset(seed=seed)
    frames = [np.asarray(env.render())]
    success_steps = 0
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(np.asarray(env.render()))
        success_steps += int(bool(info.get("success", False)))
        if terminated or truncated:
            break
    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}.gif"
    imageio.mimsave(path, frames, format="GIF", fps=20, loop=0)
    return path, len(frames), success_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="cube-single-play-v0")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/raid/tongle/rl/testbed/sanity/videos/ogbench")
    a = p.parse_args()

    tasks = ALL_TASKS[a.domain]
    out_dir = Path(a.out)
    print(f"rolling out {len(tasks)} tasks of {a.domain} (random policy, {a.render_size}px)")
    for t in tasks:
        try:
            path, n, succ = rollout(t, a.steps, a.render_size, a.seed, out_dir)
            print(f"  OK  {t:<42} frames={n:<4} success_steps={succ} -> {path}")
        except Exception as e:  # per-task guard so one failure doesn't abort the set
            print(f"  FAIL {t:<42} {type(e).__name__}: {str(e).splitlines()[0][:70]}")
            if os.environ.get("DEBUG"):
                traceback.print_exc()


if __name__ == "__main__":
    main()
