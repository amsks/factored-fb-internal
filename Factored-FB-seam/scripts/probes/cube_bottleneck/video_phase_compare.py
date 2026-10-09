"""Side-by-side video of FB and Density-FB on the same cube-single episode.

Both rollouts were recorded by failure_figs.py from the same start state and goal
(analysis/failure_figs/cube-single-play-v0/rollouts/traj_*.npz). Each frame is restored
from the recorded qpos/qvel and rendered by MuJoCo with the goal cube shown. A banner
above each panel names the current stage, using the thresholds of the stage classifier
(phases.PhaseThresholds):

  Approach     hand still more than eps_reach from the cube
  Acquisition  hand at the cube, grasp not yet lifted
  Maintain     cube held and lifted, not yet moved far toward the goal
  Transport    held cube has covered > 25% of the distance to the goal
  Dropped      the cube was held earlier and is no longer held
  At goal      cube within 4 cm of the goal

Usage:
  MUJOCO_GL=egl python scripts/probes/cube_bottleneck/video_phase_compare.py \
      --seed 0 --task 5 --episode 12 --out website/static/video/cube_s0_t5_e12
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.failure_figs import TABLE_Z, set_cube_goals
from scripts.probes.cube_bottleneck.phases import PhaseThresholds

ENV = "cube-single-play-v0"
ROLL = Path("analysis/failure_figs") / ENV / "rollouts"
METHODS = [("fb_tdj", "FB", "#1f77b4"), ("density_fb_z", "Density-FB", "#e8641b")]
# colours match the caption of fig:failure (experiments.tex)
STAGE_COLOURS = {"Approach": "#CC78BC", "Acquisition": "#DE8F05", "Maintain": "#0173B2",
                 "Transport": "#D55E00", "Dropped": "#BB2525", "At goal": "#029E73"}
SUCCESS_RADIUS = 0.04


def stage_labels(eff, cube, grip, goal, thr: PhaseThresholds):
    d = np.linalg.norm(eff - cube, axis=1)
    dxy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - TABLE_Z
    held = (grip > thr.tau_grip) & (dxy < thr.eps_reach)
    to_goal = np.linalg.norm(cube - goal, axis=1)
    start_xy = cube[0, :2]
    total = max(np.linalg.norm(goal[:2] - start_xy), 1e-6)
    progress = 1.0 - np.linalg.norm(goal[:2] - cube[:, :2], axis=1) / total
    labels, was_lifted = [], False
    for t in range(len(cube)):
        lifted_now = held[t] and lift[t] > thr.delta_lift
        was_lifted |= lifted_now
        if to_goal[t] < SUCCESS_RADIUS:
            labels.append("At goal")
        elif lifted_now:
            labels.append("Transport" if progress[t] > 0.25 else "Maintain")
        elif was_lifted and not held[t]:
            labels.append("Dropped")
        elif d[t] < thr.eps_reach or held[t]:
            labels.append("Acquisition")
        else:
            labels.append("Approach")
    return labels


def load(method, seed, task, episode):
    z = np.load(ROLL / f"traj_{method}_seed{seed}_1000000.npz")
    p = f"t{task}_e{episode}"
    return dict(eff=z[f"{p}_eff"], cube=z[f"{p}_cube"][:, 0], grip=z[f"{p}_grip"],
                goal=z[f"{p}_goal"], qpos=z[f"{p}_qpos"], qvel=z[f"{p}_qvel"],
                success=bool(z[f"{p}_success"]))


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def panel(img, method_name, method_col, stage, t, final_text=None):
    h, w = img.shape[:2]
    head, band = 56, 52
    canvas = Image.new("RGB", (w, h + head + band), "white")
    canvas.paste(Image.fromarray(img), (0, head + band))
    dr = ImageDraw.Draw(canvas)
    dr.text((w // 2, head // 2), method_name, fill=method_col, font=font(32, True), anchor="mm")
    col = STAGE_COLOURS[stage]
    dr.rectangle([0, head, w, head + band], fill=col)
    dr.text((w // 2, head + band // 2), stage, fill="white", font=font(28, True), anchor="mm")
    dr.text((14, h + head + band - 12), f"t = {t}", fill="white", font=font(22), anchor="ld")
    if final_text:
        dr.rectangle([0, head + band + h - 100, w, head + band + h - 56], fill=(0, 0, 0))
        dr.text((w // 2, head + band + h - 78), final_text, fill="white", font=font(24, True), anchor="mm")
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=5)
    ap.add_argument("--episode", type=int, default=12)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tail", type=int, default=45, help="steps kept after the later decisive event")
    ap.add_argument("--out", default="website/static/video/cube_compare")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import ogbench
    env = ogbench.make_env_and_datasets(ENV, env_only=True, render_mode="rgb_array",
                                        width=args.size, height=args.size)
    thr = PhaseThresholds()
    runs = []
    for key, name, col in METHODS:
        ep = load(key, args.seed, args.task, args.episode)
        ep["labels"] = stage_labels(ep["eff"], ep["cube"], ep["grip"], ep["goal"][0], thr)
        runs.append((name, col, ep))
    fb, dfb = runs[0][2], runs[1][2]
    assert np.allclose(fb["qpos"][0], dfb["qpos"][0], atol=1e-6) and np.allclose(fb["goal"], dfb["goal"])

    # stop once both outcomes are clear: Density-FB at the goal and FB's drop shown for a while
    first_drop = next((t for t, l in enumerate(fb["labels"]) if l == "Dropped"), len(fb["labels"]) - 1)
    n = min(max(len(dfb["qpos"]), first_drop) + args.tail, max(len(fb["qpos"]), len(dfb["qpos"])))
    print(f"seed {args.seed} task {args.task} episode {args.episode}: FB len {len(fb['qpos'])} "
          f"first drop {first_drop}, Density-FB len {len(dfb['qpos'])}, video steps {n}")

    def render(ep, t):
        env.reset(options=dict(task_id=int(args.task), render_goal=True))
        set_cube_goals(env, ep["goal"])
        R.set_state(env, ep["qpos"][t], ep["qvel"][t])
        return np.asarray(env.render())

    frames = []
    for t in range(n):
        tiles = []
        for name, col, ep in runs:
            tt = min(t, len(ep["qpos"]) - 1)
            done = t >= len(ep["qpos"]) - 1
            final = None
            if done:
                final = "Success" if ep["success"] else "Episode over: failed"
            tiles.append(panel(render(ep, tt), name, col, ep["labels"][tt], tt, final))
        gap = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
        frames.append(np.concatenate([tiles[0], gap, tiles[1]], axis=1))
    frames += [frames[-1]] * args.fps  # one-second hold on the final frame

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(f"{out}.mp4", frames, fps=args.fps, codec="libx264", quality=8,
                     pixelformat="yuv420p", macro_block_size=16)
    Image.fromarray(frames[min(len(frames) - 1, first_drop)]).save(f"{out}_poster.jpg", quality=88)
    print(f"wrote {out}.mp4 ({len(frames)} frames) and {out}_poster.jpg")


if __name__ == "__main__":
    main()
