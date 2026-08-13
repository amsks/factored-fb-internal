"""Render film-strip images of policy rollouts for the report.

For each task: roll episodes with the checkpoint's own sample_actions (the same
interface the native eval uses, so goal- and z-conditioned actors both work),
render frames, keep the first successful episode (else the last), and tile
N_FRAMES evenly-spaced frames into one horizontal strip PNG.

Usage:
  MUJOCO_GL=egl python scripts/rescuing_fb/render_rollout_strips.py \
      --run-dir <run> --checkpoint-step 500000 --out-dir reports/figures --tag vis
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/ogbench/impls"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rescuing_fb.t6_gpi_eval import load_agent_flexible  # noqa: E402

ENV = "cube-single-play-v0"
EVAL_TASKS = {t: f"cube-single-play-singletask-task{t}-v0" for t in range(1, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--tag", default="policy")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=200)
    args = ap.parse_args()

    import jax
    import ogbench
    from utils.evaluation import supply_rng

    agent, d, cfg = load_agent_flexible(Path(args.run_dir), args.checkpoint_step,
                                        arch_explicit=False, d=0,
                                        actor_cond="z", actor_vf_cond="none")
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(0))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for t in range(1, 6):
        env = ogbench.make_env_and_datasets(EVAL_TASKS[t], env_only=True,
                                            render_mode="rgb_array")
        best = None
        for ep in range(args.episodes):
            obs, info = env.reset(options=dict(task_id=t, render_goal=False))
            goal = info.get("goal")
            frames, success = [], 0.0
            for step in range(args.horizon):
                a = np.clip(np.array(actor_fn(observations=obs, goals=goal,
                                              temperature=0)), -1, 1)
                obs, _r, term, trunc, info = env.step(a)
                if step % 2 == 0 or term or trunc:
                    frames.append(env.render().copy())
                if term or trunc:
                    break
            success = float(info.get("success", 0.0))
            if best is None or success > best[0]:
                best = (success, frames)
            if success > 0:
                break
        success, frames = best
        idx = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        strip = np.concatenate([frames[i] for i in idx], axis=1)
        from PIL import Image
        p = out / f"strip_{args.tag}_task{t}_{'ok' if success > 0 else 'fail'}.png"
        Image.fromarray(strip).save(p)
        print(f"[strip] task{t} success={success:.0f} frames={len(frames)} -> {p}",
              flush=True)


if __name__ == "__main__":
    main()
