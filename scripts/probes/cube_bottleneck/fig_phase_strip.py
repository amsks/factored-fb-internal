"""Illustration of the four cube-single stages used in the failure analysis
(fig:failure-where): approach, acquisition, maintain, transport, plus the delivered cube.

Frames are real simulator states from one successful Density-FB rollout recorded by
failure_figs.py (analysis/failure_figs/cube-single-play-v0/rollouts/traj_*.npz), restored
with qpos/qvel and rendered by MuJoCo with the goal cube shown. The moment picked for
each stage uses the same thresholds as the stage classifier (phases.PhaseThresholds):

  approach     hand moving toward the cube, still > eps_reach away
  acquisition  first step the gripper is closed on the cube (hand within eps_reach)
  maintain     cube lifted > delta_lift and held, before it starts moving to the goal
  transport    held cube halfway (in distance) from its start to the goal
  success      final state, cube within 4 cm of the goal

Usage:
  MUJOCO_GL=egl python scripts/probes/cube_bottleneck/fig_phase_strip.py \
      [--task 2 --episode 0] [--video]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.failure_figs import TABLE_Z, set_cube_goals
from scripts.probes.cube_bottleneck.phases import PhaseThresholds

ENV = "cube-single-play-v0"
ROLL = Path("analysis/failure_figs") / ENV / "rollouts"
LABEL = "density_fb_z_seed0_1000000"
OUT = Path("PAPER/ICLR/figures/failure/fig_phase_strip")

# colours match the caption of fig:failure (experiments.tex)
STAGES = [("approach", "Approach", "#CC78BC"),
          ("acquisition", "Acquisition", "#DE8F05"),
          ("maintain", "Maintain", "#0173B2"),
          ("transport", "Transport", "#D55E00"),
          ("success", "Success", "#029E73")]


def pick_frames(eff, cube, grip, goal, thr: PhaseThresholds):
    d = np.linalg.norm(eff - cube, axis=1)
    dxy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - TABLE_Z
    held = (grip > thr.tau_grip) & (dxy < thr.eps_reach)
    t_reach = int(np.argmax(d < thr.eps_reach))
    t_grasp = int(np.argmax(held))
    lifted = np.flatnonzero(held & (lift > thr.delta_lift))
    t_lift = int(lifted[min(len(lifted) - 1, 3)])
    start = cube[t_lift, :2]
    total = np.linalg.norm(goal[:2] - start)
    moved = np.linalg.norm(cube[:, :2] - start, axis=1)
    after = np.flatnonzero((np.arange(len(cube)) > t_lift) & (moved > 0.5 * total))
    t_carry = int(after[0]) if len(after) else (t_lift + len(cube) - 1) // 2
    t_approach = max(0, t_reach // 2)
    return dict(approach=t_approach, acquisition=t_grasp, maintain=t_lift,
                transport=t_carry, success=len(cube) - 1)


def render_state(env, task, goal, qpos, qvel):
    env.reset(options=dict(task_id=int(task), render_goal=True))
    set_cube_goals(env, goal)
    R.set_state(env, qpos, qvel)
    return np.asarray(env.render())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="also write a gif of the episode")
    args = ap.parse_args()

    d = np.load(ROLL / f"traj_{LABEL}.npz")
    eps = pd.read_csv(ROLL / f"episodes_{LABEL}.csv")
    if args.task is None:
        # the successful episode whose cube travels furthest to its goal
        ok = eps[eps.success == 1].copy()
        ok["dist"] = [np.linalg.norm(d[f"t{t}_e{e}_goal"][0, :2] - d[f"t{t}_e{e}_init_cube"][0, :2])
                      for t, e in zip(ok.task, ok.episode)]
        row = ok.sort_values("dist").iloc[-1]
        args.task, args.episode = int(row.task), int(row.episode)
    p = f"t{args.task}_e{args.episode}"
    eff, cube, grip = d[f"{p}_eff"], d[f"{p}_cube"][:, 0], d[f"{p}_grip"]
    goal, qpos, qvel = d[f"{p}_goal"], d[f"{p}_qpos"], d[f"{p}_qvel"]
    frames = pick_frames(eff, cube, grip, goal[0], PhaseThresholds())
    print(f"task {args.task} episode {args.episode} length {len(qpos)} frames {frames}")

    import ogbench
    env = ogbench.make_env_and_datasets(ENV, env_only=True, render_mode="rgb_array")
    imgs = {k: render_state(env, args.task, goal, qpos[t], qvel[t]) for k, t in frames.items()}

    h, w = next(iter(imgs.values())).shape[:2]
    top, bot = int(0.10 * h), int(0.92 * h)          # trim sky and floor
    fig, axes = plt.subplots(1, len(STAGES), figsize=(7.0, 1.75))
    for ax, (key, name, col) in zip(axes, STAGES):
        ax.imshow(imgs[key][top:bot])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.0)
        ax.set_title(name, fontsize=11, color=col, pad=3)
        ax.text(0.03, 0.04, f"t={frames[key]}", transform=ax.transAxes, fontsize=7.5,
                color="white", ha="left", va="bottom")
    fig.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.88, bottom=0.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}.{e}", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {OUT}.pdf/.png")

    if args.video:
        import imageio.v2 as imageio
        vid = [render_state(env, args.task, goal, qpos[t], qvel[t]) for t in range(len(qpos))]
        imageio.mimsave(f"{OUT}.gif", vid, duration=100, loop=0)
        print(f"wrote {OUT}.gif")


if __name__ == "__main__":
    main()
