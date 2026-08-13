"""Goal coverage across all 1M datasets: training-goal coverage (blue) vs eval goals (orange stars).

10 panels, 2 rows x 5 cols, grouped by suite:
  Row 1: mw-push (play-resample v3), mw-pick-place (play-resample v3),
         mw-push (play-noise-hard-1m), mw-pick-place (play-noise-hard-1m),
         ms-push-cube (play-resample)
  Row 2: ms-pick-cube (play-resample), cube-single, cube-double, scene, antmaze-medium

Conventions reused from goal_coverage_1m.py:
- BLUE #2a78d6 scatter for training-goal coverage, ORANGE #eb6834 stars for eval goals.
- MW/MS: training goals = unique target coords baked into the obs goal dims (deduped to
  4 decimals); eval star = goal dims of the single highest-reward next_observation frame
  (the fixed conditioning frame used by env/unified.py).
- OGBench: training goals = block/cube positions visited across the play dataset
  (subsampled to 100k frames, seed 0); eval stars = the task_infos goal_xyzs read
  directly from the ogbench env source (cube_env.py / scene_env.py), and for antmaze,
  goal_xy = ij_to_xy(goal_ij) using the medium maze_map + maze_unit=4.0, offset=4
  (verified against ogbench/locomaze/maze.py).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
RNG = np.random.default_rng(0)

panels = []  # each: dict(title, suite_key, subtitle, xy, stars, n, label)


def add_mw_ms_panel(name, path, dims, title):
    d = np.load(path)
    lo, hi = dims
    obs = d["observation"]
    goals = obs[..., lo:hi].reshape(-1, hi - lo)
    uniq = np.unique(np.round(goals, 4), axis=0)
    r = d["reward"].reshape(-1)
    star = d["next_observation"].reshape(-1, obs.shape[-1])[r.argmax(), lo:hi]
    zr = f"z: {uniq[:, 2].min():.2f}..{uniq[:, 2].max():.2f}"
    panels.append(dict(title=title, subtitle=f"{len(uniq):,} unique targets · {zr}",
                        xy=uniq[:, :2], stars=star[None, :2]))
    print(name, "unique targets:", len(uniq), "star:", np.round(star, 3))


def subsample(arr, n=100_000):
    if len(arr) <= n:
        return arr
    idx = RNG.choice(len(arr), n, replace=False)
    return arr[idx]


# ---- Row 1 (cols 0-3): Metaworld, play-resample v3 and play-noise-hard-1m ----
add_mw_ms_panel("mw-push (play-resample v3)",
                 "/scratch/11708/amsks/fb-data/play-resample/data/metaworld/mw-push.npz",
                 (36, 39), "mw-push")
add_mw_ms_panel("mw-pick-place (play-resample v3)",
                 "/scratch/11708/amsks/fb-data/play-resample/data/metaworld/mw-pick-place.npz",
                 (36, 39), "mw-pick-place")
add_mw_ms_panel("mw-push (play-noise-hard-1m)",
                 "/scratch/11708/amsks/fb-data/play-noise-hard-1m/data/metaworld/mw-push.npz",
                 (36, 39), "mw-push")
add_mw_ms_panel("mw-pick-place (play-noise-hard-1m)",
                 "/scratch/11708/amsks/fb-data/play-noise-hard-1m/data/metaworld/mw-pick-place.npz",
                 (36, 39), "mw-pick-place")

# ---- col 4 (row1) + col 0 (row2): ManiSkill play-resample ----
add_mw_ms_panel("ms-push-cube",
                 "/scratch/11708/amsks/fb-data/ms-play-resample/data/maniskill/ms-push-cube.npz",
                 (25, 28), "ms-push-cube")
add_mw_ms_panel("ms-pick-cube",
                 "/scratch/11708/amsks/fb-data/ms-play-resample/data/maniskill/ms-pick-cube.npz",
                 (26, 29), "ms-pick-cube")

# ---- Row 2 (cols 1-4): OGBench ----
XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0

# cube-single: dims 19:22 (verified: obs dim 28 = 19 proprio + 9 single-cube block).
CUBE_SINGLE_GOALS = np.array([
    [0.425, -0.1, 0.02], [0.50, 0.0, 0.02], [0.35, 0.0, 0.02],
    [0.50, 0.2, 0.02], [0.50, -0.2, 0.02]])  # cube_env.py set_tasks(), 'single'
d = np.load("/scratch/11708/amsks/ogbench_data/data/cube-single-play-v0.npz")
obs = d["observations"]
block = obs[:, 19:22] / XYZ_SCALER + XYZ_CENTER
sub = subsample(block)
panels.append(dict(title="cube-single",
                    subtitle=f"{len(block):,} frames · z: {block[:, 2].min():.2f}..{block[:, 2].max():.2f}",
                    xy=sub[:, :2], stars=CUBE_SINGLE_GOALS[:, :2]))
print("cube-single frames:", len(block), "stars:", len(CUBE_SINGLE_GOALS))

# cube-double: obs dim 37 = 19 proprio + 9*2 blocks -> verified block0 pos = 19:22,
# block1 pos = 28:31 (checked against source npz: obs[0,19:22] and obs[0,28:31]).
CUBE_DOUBLE_GOALS = np.array([  # cube_env.py set_tasks(), 'double' -> goal_xyzs, both blocks, all 5 tasks
    [0.425, 0.0, 0.02], [0.425, 0.1, 0.02],
    [0.35, 0.1, 0.02], [0.50, 0.1, 0.02],
    [0.425, -0.2, 0.02], [0.425, 0.2, 0.02],
    [0.425, 0.1, 0.02], [0.425, -0.1, 0.02],
    [0.425, 0.0, 0.02], [0.425, 0.0, 0.06],
])
d = np.load("/scratch/11708/amsks/ogbench_data/data/cube-double-play-v0.npz")
obs = d["observations"]
b0 = obs[:, 19:22] / XYZ_SCALER + XYZ_CENTER
b1 = obs[:, 28:31] / XYZ_SCALER + XYZ_CENTER
both = np.concatenate([b0, b1], axis=0)
sub = subsample(both)
panels.append(dict(title="cube-double",
                    subtitle=f"{len(both):,} block-frames · z: {both[:, 2].min():.2f}..{both[:, 2].max():.2f}",
                    xy=sub[:, :2], stars=CUBE_DOUBLE_GOALS[:, :2]))
print("cube-double block-frames:", len(both), "stars (2 blocks x 5 tasks):", len(CUBE_DOUBLE_GOALS))

# scene: obs dim 40 = 19 proprio + 9 cube block + 8 buttons(2x4) + 4 drawer/window ->
# verified cube block pos = 19:22 (same proprio+block layout as cube_env).
# All 5 scene tasks define a cube goal_xyzs (num_cubes=1); for task1/task2 the "goal"
# cube position is unchanged from init (those tasks are drawer/window-focused), so only
# task3/4/5 involve real cube motion -- but all 5 stars are genuine per-task cube goals.
SCENE_CUBE_GOALS = np.array([
    [0.35, 0.05, 0.02],       # task1_open: cube goal == cube init (task is about drawer/window)
    [0.35, -0.05, 0.02],      # task2_unlock_and_lock: cube goal == cube init
    [0.4, 0.15, 0.02],        # task3_rearrange_medium
    [0.33, -0.356, 0.065986], # task4_put_in_drawer
    [0.33, -0.356, 0.065986], # task5_rearrange_hard (same drawer slot as task4)
])
d = np.load("/scratch/11708/amsks/ogbench_data/data/scene-play-v0.npz")
obs = d["observations"]
cube = obs[:, 19:22] / XYZ_SCALER + XYZ_CENTER
sub = subsample(cube)
panels.append(dict(title="scene",
                    subtitle=f"{len(cube):,} frames · z: {cube[:, 2].min():.2f}..{cube[:, 2].max():.2f}\n5 of 5 tasks set a cube goal (2 coincide w/ init)",
                    xy=sub[:, :2], stars=SCENE_CUBE_GOALS[:, :2]))
print("scene frames:", len(cube), "stars:", len(SCENE_CUBE_GOALS))

# antmaze-medium: obs dim 29 = ant qpos(15) + qvel(14); qpos[:2] = torso x,y (verified
# from ogbench/locomaze/ant.py get_ob = concat(qpos, qvel), standard mujoco ant qpos
# layout [x,y,z,quat(4),8 joint angles]). Goal xy via ij_to_xy(goal_ij), maze_type=
# 'medium', maze_unit=4.0 (default, unregistered override), offset_x=offset_y=4
# (ogbench/locomaze/maze.py set_tasks() + ij_to_xy()).
ANTMAZE_GOALS = np.array([
    [20.0, 20.0], [20.0, 0.0], [4.0, 12.0], [0.0, 20.0], [0.0, 0.0],
])
d = np.load("/scratch/11708/amsks/ogbench_data/data/antmaze-medium-navigate-v0.npz")
obs = d["observations"]
xy = obs[:, :2]
sub = subsample(xy)
panels.append(dict(title="antmaze-medium",
                    subtitle=f"{len(xy):,} frames · torso xy",
                    xy=sub, stars=ANTMAZE_GOALS))
print("antmaze frames:", len(xy), "stars:", len(ANTMAZE_GOALS))

# ---- Plot ----
fig, axes = plt.subplots(2, 5, figsize=(22, 9.2))
fig.patch.set_facecolor("#fcfcfb")
axes_flat = axes.flatten()

for ax, p in zip(axes_flat, panels):
    xy, stars = p["xy"], p["stars"]
    ax.set_facecolor("#fcfcfb")
    dense = len(xy) > 20_000
    ax.scatter(xy[:, 0], xy[:, 1], s=3 if dense else 9,
               alpha=0.05 if dense else 0.35, color=BLUE, linewidths=0, rasterized=True)
    ax.scatter(stars[:, 0], stars[:, 1], marker="*", s=300, color=ORANGE,
               edgecolors="white", linewidths=1.1, zorder=5)
    ax.set_title(f"{p['title']}\n{p['subtitle']}", fontsize=9.5, color=INK)
    ax.set_xlabel("x [m]", fontsize=8.5, color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d8d7d2")
    ax.set_aspect("equal", adjustable="datalim")

axes[0, 0].set_ylabel("y [m]", fontsize=8.5, color=MUTED)
axes[1, 0].set_ylabel("y [m]", fontsize=8.5, color=MUTED)

handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, alpha=0.7,
                      label="training-goal coverage"),
           plt.Line2D([], [], marker="*", ls="", color=ORANGE, ms=15,
                      markeredgecolor="white", label="evaluation goals")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)

# Group titles per row (approximate column spans).
plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.90])
fig.canvas.draw()

row1_groups = [("Metaworld play-resample v3 (1M)", [0, 1]),
               ("Metaworld play-noise-hard-1m", [2, 3]),
               ("ManiSkill play-resample (1M)", [4])]
row2_groups = [("ManiSkill play-resample (1M), cont.", [0]),
               ("OGBench (1M)", [1, 2, 3, 4])]

for gtitle, idxs in row1_groups:
    x0 = min(axes[0, i].get_position().x0 for i in idxs)
    x1 = max(axes[0, i].get_position().x1 for i in idxs)
    y1 = max(axes[0, i].get_position().y1 for i in idxs)
    fig.text((x0 + x1) / 2, y1 + 0.045, gtitle, ha="center", va="bottom",
             fontsize=12, fontweight="bold", color=INK)

for gtitle, idxs in row2_groups:
    x0 = min(axes[1, i].get_position().x0 for i in idxs)
    x1 = max(axes[1, i].get_position().x1 for i in idxs)
    y1 = max(axes[1, i].get_position().y1 for i in idxs)
    fig.text((x0 + x1) / 2, y1 + 0.045, gtitle, ha="center", va="bottom",
             fontsize=12, fontweight="bold", color=INK)

out = "/tmp/claude-911361/-work-11708-amsks-vista-git-Factored-FB/b81edaa3-2aa0-4bca-90a6-5e20a9aceee8/scratchpad/goal_coverage_all1m.png"
fig.savefig(out, dpi=150)
print("saved", out)
