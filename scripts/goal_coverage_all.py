#!/usr/bin/env python3
"""scripts/goal_coverage_all.py — goal-coverage maps for EVERY dataset in the testbed.

Supersedes goal_coverage_all1m.py, which hard-coded ten 1M panels. Here the datasets are a
registry, so the small/superseded regimes appear next to the 1M ones and the size confound
(condition 4 of the note) is visible in the same picture as the coverage confound
(condition 2).

Two things are plotted per panel:
  BLUE   training-goal coverage -- every distinct goal the data ever presents.
  ORANGE evaluation goals -- where the benchmark actually tests.

Training goals are per DATASET; evaluation goals are per TASK, because they come from the
env, not from the data. So the stars are read once from results/deconfound/cov-<task>.npz
(written by goal_coverage.py extract, which builds the real env and pulls the true
per-episode targets under eval_goal_mode=synth) and reused across every dataset of that
task. Falling back to the single highest-reward frame -- the old fixed-conditioning star --
would show the eval defect we already repaired rather than the eval protocol now in use.

    python scripts/goal_coverage_all.py --fig PAPER/decoding-bilinear/figs/goal_coverage_all.png \\
        --json-out results/deconfound/goal-coverage-all.json

numpy + matplotlib only: it reads packed npz and the cached extracts, never builds an env,
so it runs on a login node.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
RNG = np.random.default_rng(0)
FB = os.environ.get("FB_DATA", "/scratch/11708/amsks/fb-data")
OG = os.environ.get("OGBENCH_DATA", "/scratch/11708/amsks/ogbench_data/data")

# Goal slice per task family, as verified in env/unified.py::_GOAL_LAYOUT.
#   Metaworld obs 39 = [curr 18 | prev 18 | goal 3]                      -> 36:39
#   ManiSkill PushCube 35 = qpos9|qvel9|tcp_pose7|goal_pos3|obj_pose7    -> 25:28
#   ManiSkill PickCube 42 = qpos9|qvel9|grasp1|tcp7|goal3|obj7|d3|d3     -> 26:29
GOAL_DIMS = {"mw-push": (36, 39), "mw-pick-place": (36, 39),
             "ms-push-cube": (25, 28), "ms-pick-cube": (26, 29)}
# Object (achieved-goal) slice. This is the OTHER goal marginal, and the operative one:
# hindsight relabeling draws goals from states the object actually reached, not from the
# targets the collection policy was commanded to chase. The two coincide only where the
# behaviour policy always succeeds. Plotted coverage below is the commanded marginal (crisp,
# deduped, one point per distinct target); the reported nearest-neighbour numbers use the
# achieved marginal, matching goal_coverage.py's E4 measure.
OBJ_DIMS = {"mw-push": (4, 7), "mw-pick-place": (4, 7),
            "ms-push-cube": (28, 31), "ms-pick-cube": (29, 32)}
# (success radius in the env's own metric, leading components that metric uses). PushCube
# scores planar XY distance to a goal region; everything else is 3-D.
RADIUS = {"mw-push": (0.05, 3), "mw-pick-place": (0.07, 3),
          "ms-push-cube": (0.10, 2), "ms-pick-cube": (0.025, 3),
          "cube-single": (0.04, 3), "cube-double": (0.04, 3), "scene": (0.04, 3),
          "antmaze-medium": (0.5, 2)}


def nn_radii(pool, stars, task, cap=200_000):
    """Median / p95 distance from each eval goal to the nearest training achieved goal, in
    units of that env's success radius. Small => the eval set is inside the training goal
    marginal (condition 2 holds); large => it is not, which is what Arm D induced on cube."""
    if stars is None or len(pool) == 0:
        return None, None
    r, k = RADIUS.get(task, (1.0, pool.shape[1]))
    if len(pool) > cap:
        pool = pool[RNG.choice(len(pool), cap, replace=False)]
    d = np.linalg.norm(pool[None, :, :k] - np.asarray(stars)[:, None, :k], axis=-1).min(1)
    return float(np.median(d) / r), float(np.percentile(d, 95) / r)

# (suite, regime label, root under FB_DATA, family dir, tasks). Order drives the layout.
PACKED = [
    ("Metaworld", "expert",              "expert",             "metaworld", ["mw-push", "mw-pick-place"]),
    ("Metaworld", "play-noise",          "play-noise",         "metaworld", ["mw-push", "mw-pick-place"]),
    ("Metaworld", "play-noise-hard",     "play-noise-hard",    "metaworld", ["mw-push", "mw-pick-place"]),
    ("Metaworld", "play-noise-hard 1M",  "play-noise-hard-1m", "metaworld", ["mw-push", "mw-pick-place"]),
    ("Metaworld", "play-resample v3 1M", "play-resample",      "metaworld", ["mw-push", "mw-pick-place"]),
    ("Metaworld", "play-resample-hard v3 1M", "play-resample-hard", "metaworld", ["mw-push", "mw-pick-place"]),
    ("ManiSkill", "expert",              "expert",             "maniskill", ["ms-push-cube", "ms-pick-cube"]),
    ("ManiSkill", "play-noise",          "play-noise",         "maniskill", ["ms-push-cube", "ms-pick-cube"]),
    ("ManiSkill", "play-resample 48k (upstream)", "upstream-play-resample-packed", "maniskill",
     ["ms-push-cube", "ms-pick-cube"]),
    ("ManiSkill", "play-resample 1M",    "ms-play-resample",   "maniskill", ["ms-push-cube", "ms-pick-cube"]),
]

XYZ_CENTER, XYZ_SCALER = np.array([0.425, 0.0, 0.0]), 10.0
# OGBench: (title, npz, list of block slices, eval goal xy, note). Goals are the task_infos
# goal_xyzs read from the ogbench env source; antmaze's are ij_to_xy(goal_ij) on the medium
# maze_map (maze_unit 4.0, offset 4).
OGBENCH = [
    ("cube-single", "cube-single-play-v0.npz", [(19, 22)],
     [[0.425, -0.1], [0.50, 0.0], [0.35, 0.0], [0.50, 0.2], [0.50, -0.2]], "block xy"),
    ("cube-double", "cube-double-play-v0.npz", [(19, 22), (28, 31)],
     [[0.425, 0.0], [0.425, 0.1], [0.35, 0.1], [0.50, 0.1], [0.425, -0.2],
      [0.425, 0.2], [0.425, 0.1], [0.425, -0.1], [0.425, 0.0], [0.425, 0.0]], "2 blocks"),
    ("scene", "scene-play-v0.npz", [(19, 22)],
     [[0.35, 0.05], [0.35, -0.05], [0.4, 0.15], [0.33, -0.356], [0.33, -0.356]], "cube xy"),
    ("antmaze-medium", "antmaze-medium-navigate-v0.npz", [(0, 2)],
     [[20.0, 20.0], [20.0, 0.0], [4.0, 12.0], [0.0, 20.0], [0.0, 0.0]], "torso xy"),
]


def subsample(a, n=100_000):
    return a if len(a) <= n else a[RNG.choice(len(a), n, replace=False)]


def eval_goals_for(task, covdir):
    """Per-episode eval targets from the cached extract, or None if it was never run."""
    p = os.path.join(covdir, f"cov-{task}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["eval_goals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="PAPER/decoding-bilinear/figs/goal_coverage_all.png")
    ap.add_argument("--json-out", default="results/deconfound/goal-coverage-all.json")
    ap.add_argument("--covdir", default="results/deconfound")
    ap.add_argument("--min-transitions", type=int, default=0,
                    help="drop datasets smaller than this. PAPER/table.tex enforces a "
                         "1M-only policy (sub-1M packs are quarantined and non-citable), so "
                         "the figure cited there is built with --min-transitions 1000000; "
                         "the unfiltered figure is the complete data artifact.")
    args = ap.parse_args()

    panels, stats = [], []
    for suite, regime, root, fam, tasks in PACKED:
        for task in tasks:
            fp = os.path.join(FB, root, "data", fam, f"{task}.npz")
            if not os.path.exists(fp):
                print(f"SKIP {regime}/{task}: no {fp}")
                continue
            z = np.load(fp)
            obs = z["observation"]
            lo, hi = GOAL_DIMS[task]
            goals = obs[..., lo:hi].reshape(-1, hi - lo)
            uniq = np.unique(np.round(goals, 4), axis=0)
            n = int(obs.shape[0] * obs.shape[1])
            if n < args.min_transitions:
                print(f"SKIP {regime}/{task}: {n:,} < {args.min_transitions:,}")
                continue
            stars = eval_goals_for(task, args.covdir)
            alo, ahi = OBJ_DIMS[task]
            ach = z["next_observation"][..., alo:ahi].reshape(-1, ahi - alo)
            nn_med, nn_p95 = nn_radii(ach, stars, task)
            panels.append(dict(suite=suite, regime=regime, task=task,
                               title=f"{task}\n{regime}",
                               sub=f"{n:,} trans · {len(uniq):,} goals",
                               xy=uniq[:, :2], stars=None if stars is None else stars[:, :2]))
            stats.append(dict(suite=suite, regime=regime, task=task, transitions=n,
                              unique_goals=int(len(uniq)),
                              extent_x=float(np.ptp(uniq[:, 0])), extent_y=float(np.ptp(uniq[:, 1])),
                              eval_goals=0 if stars is None else int(len(stars)),
                              nn_median_radii=nn_med, nn_p95_radii=nn_p95))
            nn = "--" if nn_med is None else f"{nn_med:.2f}/{nn_p95:.2f}"
            print(f"{regime:28s} {task:15s} {n:>10,} trans  {len(uniq):>8,} goals  nn med/p95 {nn}")

    for title, fn, slices, stars, note in OGBENCH:
        fp = os.path.join(OG, fn)
        if not os.path.exists(fp):
            print(f"SKIP {title}: no {fp}")
            continue
        obs = np.load(fp)["observations"]
        if title == "antmaze-medium":
            pts = obs[:, 0:2]
        else:
            pts = np.concatenate([obs[:, a:b] / XYZ_SCALER + XYZ_CENTER for a, b in slices])[:, :2]
        n = len(obs)
        panels.append(dict(suite="OGBench", regime="play", task=title,
                           title=f"{title}\nplay",
                           sub=f"{n:,} trans · {note}",
                           xy=subsample(pts), stars=np.asarray(stars)))
        nn_med, nn_p95 = nn_radii(pts, np.asarray(stars), title)
        stats.append(dict(suite="OGBench", regime="play", task=title, transitions=int(n),
                          unique_goals=None, extent_x=float(np.ptp(pts[:, 0])),
                          extent_y=float(np.ptp(pts[:, 1])), eval_goals=len(stars),
                          nn_median_radii=nn_med, nn_p95_radii=nn_p95))
        print(f"{'play':28s} {title:15s} {n:>10,} trans"
              f"                    nn med/p95 {nn_med:.2f}/{nn_p95:.2f}")

    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.15 * nrow))
    fig.patch.set_facecolor("#fcfcfb")
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, p in zip(axes_flat, panels):
        xy = p["xy"]
        ax.set_facecolor("#fcfcfb")
        dense = len(xy) > 20_000
        ax.scatter(xy[:, 0], xy[:, 1], s=2.5 if dense else 8,
                   alpha=0.05 if dense else 0.35, color=BLUE, linewidths=0, rasterized=True)
        if p["stars"] is not None:
            ax.scatter(p["stars"][:, 0], p["stars"][:, 1], marker="*", s=150, color=ORANGE,
                       edgecolors="white", linewidths=0.9, zorder=5)
        ax.set_title(f"{p['title']}\n{p['sub']}", fontsize=8, color=INK)
        ax.tick_params(labelsize=6.5, colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#d8d7d2")
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, alpha=0.7,
                          label="training-goal coverage"),
               plt.Line2D([], [], marker="*", ls="", color=ORANGE, ms=14,
                          markeredgecolor="white", label="evaluation goals (per-episode, synth)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.99])
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    print("saved", args.fig, f"({len(panels)} panels)")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(stats, f, indent=2)
        print("saved", args.json_out)


if __name__ == "__main__":
    main()
