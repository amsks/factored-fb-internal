"""scripts/data/convert_ogbench_to_unified.py — OGBench episodes -> unified format.

Sanity-check bridge for the alignment work: re-express an existing OGBench
per-episode dataset (observation/action/physics/reward/discount) in the unified
trajectory format (data/unified.py) WITHOUT changing the transitions, so FB
training on the converted data is identical to training on the native OGBench
loader — any divergence is a dataflow bug, not a data difference.

Transition alignment reproduces data/ogbench.py::load_transitions exactly:
    observation      = obs[:-1]        # s_t
    action           = act[1:]         # a_{t+1}  (OGBench EXORL shift)
    next_observation = obs[1:]         # s_{t+1}
    achieved_goal    = physics[1:, gslice]     # goal-relevant slice at s_{t+1}
    reward/terminal/timeout/success    # carried / zeroed to match the loader

For cube-single, the goal-relevant slice is physics[..., 14:17] == cube xyz
(see envs/ogbench.cube_reward_fn / agents/tdmpc2 CUBE_SLICE), so the achieved_goal
relabel is mathematically identical to cube_reward_fn's ||cube - target|| test.

Usage:
    .venv/bin/python scripts/data/convert_ogbench_to_unified.py \
        --env cube-single-play-v0 \
        --src /raid/tongle/rl/fb-data/datasets \
        --out /raid/tongle/rl/fb-data/unified \
        --goal-slice 14:17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repo importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.unified import dataset_dir, write_episode, write_meta  # noqa: E402


def parse_slice(s: str) -> slice:
    a, b = s.split(":")
    return slice(int(a), int(b))


def convert(env: str, src: Path, out: Path, gslice: slice, max_episodes: int | None) -> None:
    src_buf = Path(src) / env / "buffer"
    files = sorted(src_buf.glob("*.npz"))
    if max_episodes is not None:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No OGBench episodes at {src_buf}")

    out_buf = dataset_dir(out, env) / "buffer"
    out_buf.mkdir(parents=True, exist_ok=True)

    goal_dim = gslice.stop - gslice.start
    n_written = 0
    for i, f in enumerate(files):
        ep = np.load(f)
        obs = ep["observation"].astype(np.float32)
        act = ep["action"].astype(np.float32)
        phys = ep["physics"].astype(np.float32)
        T = obs.shape[0]
        if T < 2:
            continue
        n = T - 1
        reward = ep["reward"].astype(np.float32).reshape(-1)[1:] if "reward" in ep else np.zeros(n, np.float32)
        traj = {
            "observation": obs[:-1],
            "action": act[1:],
            "next_observation": obs[1:],
            "achieved_goal": phys[1:, gslice],
            "reward": reward,
            "terminal": np.zeros(n, dtype=bool),   # matches load_ogbench_dataset (all False)
            "timeout": np.zeros(n, dtype=bool),
            "success": np.zeros(n, dtype=bool),
        }
        write_episode(out_buf, i, traj)
        n_written += 1

    write_meta(
        dataset_dir(out, env),
        {
            "env_family": "ogbench",
            "env_name": env,
            "obs_dim": int(obs.shape[1]),
            "act_dim": int(act.shape[1]),
            "achieved_goal_dim": int(goal_dim),
            "achieved_goal_source": f"physics[{gslice.start}:{gslice.stop}] (next state)",
            "n_episodes": n_written,
            "source": str(src_buf),
            "note": "OGBench->unified, transitions identical to data/ogbench.load_transitions",
        },
    )
    print(f"[convert] wrote {n_written} unified episodes to {out_buf}  (goal_dim={goal_dim})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cube-single-play-v0")
    p.add_argument("--src", default="/raid/tongle/rl/fb-data/datasets")
    p.add_argument("--out", default="/raid/tongle/rl/fb-data/unified")
    p.add_argument("--goal-slice", default="14:17", help="physics slice for achieved_goal (cube xyz)")
    p.add_argument("--max-episodes", type=int, default=None)
    a = p.parse_args()
    convert(a.env, Path(a.src), Path(a.out), parse_slice(a.goal_slice), a.max_episodes)


if __name__ == "__main__":
    main()
