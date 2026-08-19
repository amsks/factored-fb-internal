#!/usr/bin/env python3
"""scripts/workspace_coverage.py — the workspace next to the goals it is covered with.

goal_coverage_all.py plots goals in a vacuum: a scatter with axis ticks in metres tells you
the marginal is wide, but not whether "wide" means the whole table or one corner of it. This
puts a top-down photograph of each environment beside the same scatter, on shared axes, so
coverage is read against the physical workspace.

Alignment is exact, not eyeballed. The camera is ORTHOGRAPHIC (`cam.orthographic = 1`), for
which MuJoCo makes `distance` the vertical world span of the view volume:

    half_height = distance / 2        half_width = half_height * (W / H)
    extent = [cx - half_width, cx + half_width, cy - half_height, cy + half_height]

Under a perspective camera the world span of a pixel depends on that pixel's height above
the camera plane, so goals lying at table height could only be aligned approximately.
Orthographic removes the dependence, and `imshow(..., extent=extent)` then puts the image in
world coordinates, so the scatter is drawn in the same units with no fitting step. `render`
self-checks this by projecting two points whose world xy is known exactly -- the object and
the target -- and storing them for the report to draw on top.

Two passes, because rendering needs a GPU and the envs, while plotting needs neither:

    # compute node, per venv
    python scripts/workspace_coverage.py render --tasks mw-push mw-pick-place cube-single ...
    # login node
    python scripts/workspace_coverage.py report --fig PAPER/.../figs/workspace_coverage.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FB = os.environ.get("FB_DATA", "/scratch/11708/amsks/fb-data")
OG = os.environ.get("OGBENCH_DATA", "/scratch/11708/amsks/ogbench_data/data")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#12a150"
INK, MUTED = "#0b0b0b", "#52514e"

# Per-env camera framing (lookat xy, vertical world span) and a one-line statement of the
# task. The span is chosen to contain the whole reachable workspace, not just the goal box,
# so the scatter can be judged against what the robot could in principle have covered.
ENVS = {
    "mw-push": dict(lookat=(0.0, 0.75), span=0.62,
                    task="Slide the puck along the table onto the target.",
                    detail="goal box x$\\in$[-0.1,0.1], y$\\in$[0.8,0.9]; success $<$5cm"),
    "mw-pick-place": dict(lookat=(0.0, 0.75), span=0.62,
                          task="Lift the puck and place it at the target (target has height).",
                          detail="goal box x$\\in$[-0.1,0.1], y$\\in$[0.8,0.9], z$\\in$[0.05,0.3]; success $<$7cm"),
    "ms-push-cube": dict(lookat=(0.05, 0.0), span=0.6,
                         task="Push the cube into the goal region.",
                         detail="planar XY scoring; success $<$10cm"),
    "ms-pick-cube": dict(lookat=(0.0, 0.0), span=0.6,
                         task="Grasp the cube and bring it to a 3-D goal point.",
                         detail="success $<$2.5cm, and requires a grasp"),
    "cube-single": dict(lookat=(0.425, 0.0), span=0.75,
                        task="Move the block to the target pose.",
                        detail="5 fixed (init, goal) tasks; success $<$4cm"),
    "cube-double": dict(lookat=(0.425, 0.0), span=0.75,
                        task="Move BOTH blocks to their target poses.",
                        detail="5 tasks $\\times$ 2 blocks; success $<$4cm each"),
    "scene": dict(lookat=(0.425, -0.1), span=0.9,
                  task="Manipulate cube, drawer and window (multi-stage).",
                  detail="5 tasks; only 3 move the cube"),
    "antmaze-medium": dict(lookat=(10.0, 10.0), span=30.0,
                           task="Navigate the ant from start to goal cell.",
                           detail="5 fixed goal cells; success $<$0.5m"),
}
GOAL_DIMS = {"mw-push": (36, 39), "mw-pick-place": (36, 39),
             "ms-push-cube": (25, 28), "ms-pick-cube": (26, 29)}
OBJ_DIMS = {"mw-push": (4, 7), "mw-pick-place": (4, 7),
            "ms-push-cube": (28, 31), "ms-pick-cube": (29, 32)}
XYZ_CENTER, XYZ_SCALER = np.array([0.425, 0.0, 0.0]), 10.0
# Body-name fragments identifying the manipulator, for the top-down occlusion fix. Kept
# deliberately narrow: goal/target/mocap markers and the objects themselves must survive.
ROBOT_BODY_KEYS = ("link", "arm", "robot", "ur5e", "sawyer", "gripper", "finger",
                   "hand", "wrist", "shoulder", "elbow", "forearm", "base", "pedestal",
                   "upper", "lower", "pad", "claw", "mount")
# ...but these are TASK geometry and must survive, whatever their name looks like. Without
# this, `scene` lost its drawer: "drawer_base"/"drawer_link" match "base"/"link" above, and
# 3 of scene's 5 tasks are about the drawer, so hiding it would have removed the very thing
# the panel is supposed to show.
KEEP_BODY_KEYS = ("drawer", "window", "cube", "block", "puck", "obj", "target", "goal",
                  "button", "handle", "door", "lid")

# Datasets shown per env, in column order. 1M only (PAPER/table.tex's citable set).
DATASETS = {
    "mw-push": [("play-noise-hard-1m", "play-noise-hard"), ("play-resample", "play-resample v3"),
                ("play-resample-hard", "play-resample-hard v3")],
    "mw-pick-place": [("play-noise-hard-1m", "play-noise-hard"), ("play-resample", "play-resample v3"),
                      ("play-resample-hard", "play-resample-hard v3")],
    "ms-push-cube": [("ms-play-resample", "ms-play-resample")],
    "ms-pick-cube": [("ms-play-resample", "ms-play-resample")],
}
OG_FILES = {"cube-single": ("cube-single-play-v0.npz", [(19, 22)]),
            "cube-double": ("cube-double-play-v0.npz", [(19, 22), (28, 31)]),
            "scene": ("scene-play-v0.npz", [(19, 22)]),
            "antmaze-medium": ("antmaze-medium-navigate-v0.npz", [(0, 2)])}


def _mj(env):
    """(model, data) out of whichever wrapper this env is wearing."""
    for o in (env, getattr(env, "unwrapped", None), getattr(env, "_raw", None)):
        if o is None:
            continue
        for mk, dk in (("model", "data"), ("_model", "_data"), ("mj_model", "mj_data")):
            m, d = getattr(o, mk, None), getattr(o, dk, None)
            if m is not None and d is not None:
                return m, d
    raise RuntimeError(f"no mujoco model/data on {type(env)}")


def build(task):
    """-> (env, model, data, probe_points). probe_points are world xy we know independently,
    used to verify the extent arithmetic."""
    if task.startswith("mw-"):
        from env.metaworld_env import make_metaworld_env
        env, _ = make_metaworld_env(task)
        np.random.seed(0)
        env.reset()
        raw = env.unwrapped
        m, d = _mj(env)
        obs = np.asarray(raw._get_obs())
        return env, m, d, {"object": obs[4:7][:2], "target": np.asarray(raw._target_pos)[:2]}
    if task.startswith("ms-"):
        raise RuntimeError("ManiSkill uses SAPIEN; handled by render_maniskill()")
    import ogbench
    name = {"cube-single": "cube-single-play-v0", "cube-double": "cube-double-play-v0",
            "scene": "scene-play-v0", "antmaze-medium": "antmaze-medium-navigate-v0"}[task]
    env = ogbench.make_env_and_datasets(name, env_only=True)
    env.reset(seed=0)
    m, d = _mj(env)
    return env, m, d, {}


def render_maniskill(task, out_dir, height_above=1.2):
    """ManiSkill runs on SAPIEN, so there is no MuJoCo orthographic camera to borrow. What it
    does expose (probed) is a settable human-render camera pose plus a pinhole intrinsic
    matrix, and for a camera looking straight down at a PLANE those give an exact extent:

        world offset of pixel u  =  (u - cx) * h / fx        (h = height above that plane)

    so half_width = (W/2)*h/fx, half_height = (H/2)*h/fy. This is exact for points ON the
    table and degrades with height above it -- fine for push-cube (planar scoring) and for
    the near-table part of pick-cube, and stated as such rather than implied to be
    orthographic. The probe points below are the check: cube and goal world xy, drawn on the
    image by the report."""
    import sapien
    from mani_skill.utils import sapien_utils
    from env.maniskill_env import make_maniskill_env

    cfg = ENVS[task]
    print(f"[render] {task}: building env...", flush=True)
    env, _ = make_maniskill_env(task)
    obs, _ = env.reset()
    print(f"[render] {task}: env built + reset", flush=True)
    u = env.unwrapped
    cam = u._human_render_cameras["render_camera"]
    cx, cy = cfg["lookat"]
    cam.camera.set_local_pose(sapien_utils.look_at(
        eye=[cx, cy, height_above], target=[cx, cy, 0.0]))
    # SAPIEN caches the render state; after moving a camera the scene must be told to
    # refresh or render() can block on a stale sync point.
    print(f"[render] {task}: camera posed, updating render...", flush=True)
    try:
        u.scene.update_render()
    except Exception as e:                                  # noqa: BLE001
        print(f"[render] {task}: update_render skipped ({type(e).__name__})", flush=True)
    img = np.asarray(env.render())
    print(f"[render] {task}: render returned", flush=True)
    img = img[0] if img.ndim == 4 else img
    K = np.asarray(cam.camera.get_intrinsic_matrix())
    K = K[0] if K.ndim == 3 else K
    H, W = img.shape[0], img.shape[1]
    half_w = (W / 2.0) * height_above / K[0, 0]
    half_h = (H / 2.0) * height_above / K[1, 1]
    extent = np.array([cx - half_w, cx + half_w, cy - half_h, cy + half_h])

    o = np.asarray(obs).reshape(-1)
    glo, ghi = GOAL_DIMS[task]
    olo, ohi = OBJ_DIMS[task]
    probes = {"object": o[olo:ohi][:2], "goal": o[glo:ghi][:2]}
    fp = os.path.join(out_dir, f"ws-{task}.npz")
    np.savez_compressed(fp, img=img, extent=extent,
                        probe_names=np.array(list(probes), dtype=object),
                        probe_xy=np.array(list(probes.values()), dtype=np.float64))
    print(f"[render] {task:16s} img{img.shape} extent {np.round(extent, 3)} "
          f"fx={K[0,0]:.1f} probes {[f'{k}={np.round(v,3)}' for k, v in probes.items()]}")
    return fp


def render_task(task, out_dir, width=760, height=760, hide_robot=True):
    if task.startswith("ms-"):
        return render_maniskill(task, out_dir)
    import mujoco
    cfg = ENVS[task]
    env, model, data, probes = build(task)
    cam = mujoco.MjvCamera()
    cam.orthographic = 1
    cam.elevation, cam.azimuth = -90.0, 90.0
    cam.lookat[:] = [cfg["lookat"][0], cfg["lookat"][1], 0.0]
    cam.distance = cfg["span"]
    # The offscreen framebuffer is sized by the MODEL (<visual><global offwidth offheight>),
    # not by the Renderer: Metaworld ships 224 and the OGBench manipspace models 200, so any
    # larger request raises rather than resizing. Raise the model's own limit first.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    # Straight down, the arm sits between the camera and the table it is working on -- on
    # cube-double it covers most of the workspace. Make robot geoms transparent so the panel
    # shows the surface the goals live on. Purely cosmetic: alpha is touched, no pose, no
    # physics, and the geoms hidden are printed so the choice is auditable rather than a
    # silent edit to what the reader sees.
    hidden = []
    if hide_robot:
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            nm = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if any(k in nm for k in ROBOT_BODY_KEYS) and not any(
                    k in nm for k in KEEP_BODY_KEYS):
                model.geom_rgba[gid, 3] = 0.0
                hidden.append(nm)
    if hidden:
        print(f"[render] {task}: hid {len(hidden)} robot geoms "
              f"({len(set(hidden))} bodies, e.g. {sorted(set(hidden))[:4]})")
    r = mujoco.Renderer(model, height=height, width=width)
    r.update_scene(data, camera=cam, scene_option=mujoco.MjvOption())
    img = r.render()
    half_h = cam.distance / 2.0
    half_w = half_h * (width / height)
    cx, cy = cfg["lookat"]
    extent = np.array([cx - half_w, cx + half_w, cy - half_h, cy + half_h])
    fp = os.path.join(out_dir, f"ws-{task}.npz")
    np.savez_compressed(fp, img=img, extent=extent,
                        probe_names=np.array(list(probes), dtype=object),
                        probe_xy=np.array(list(probes.values()), dtype=np.float64)
                        if probes else np.zeros((0, 2)))
    print(f"[render] {task:16s} img{img.shape} extent {np.round(extent, 3)} "
          f"probes {[f'{k}={np.round(v,3)}' for k, v in probes.items()]}")
    return fp


def training_goals(task):
    """(coverage points, label) for each dataset of this task."""
    out = []
    if task in DATASETS:
        lo, hi = GOAL_DIMS[task]
        fam = "metaworld" if task.startswith("mw-") else "maniskill"
        for root, label in DATASETS[task]:
            fp = os.path.join(FB, root, "data", fam, f"{task}.npz")
            if not os.path.exists(fp):
                continue
            obs = np.load(fp)["observation"]
            g = np.unique(np.round(obs[..., lo:hi].reshape(-1, hi - lo), 4), axis=0)
            out.append((g[:, :2], label, f"{len(g):,} goals"))
    else:
        fn, slices = OG_FILES[task]
        fp = os.path.join(OG, fn)
        if os.path.exists(fp):
            obs = np.load(fp)["observations"]
            if task == "antmaze-medium":
                pts = obs[:, 0:2]
            else:
                pts = np.concatenate([obs[:, a:b] / XYZ_SCALER + XYZ_CENTER
                                      for a, b in slices])[:, :2]
            rng = np.random.default_rng(0)
            if len(pts) > 100_000:
                pts = pts[rng.choice(len(pts), 100_000, replace=False)]
            out.append((pts, "stock play", f"{len(obs):,} trans"))
    return out


def eval_goals(task, covdir):
    p = os.path.join(covdir, f"cov-{task}.npz")
    if os.path.exists(p):
        return np.load(p)["eval_goals"][:, :2]
    OGG = {"cube-single": [[0.425, -0.1], [0.50, 0.0], [0.35, 0.0], [0.50, 0.2], [0.50, -0.2]],
           "cube-double": [[0.425, 0.0], [0.425, 0.1], [0.35, 0.1], [0.50, 0.1], [0.425, -0.2],
                           [0.425, 0.2], [0.425, 0.1], [0.425, -0.1], [0.425, 0.0], [0.425, 0.0]],
           "scene": [[0.35, 0.05], [0.35, -0.05], [0.4, 0.15], [0.33, -0.356], [0.33, -0.356]],
           "antmaze-medium": [[20.0, 20.0], [20.0, 0.0], [4.0, 12.0], [0.0, 20.0], [0.0, 0.0]]}
    return np.asarray(OGG[task]) if task in OGG else None


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in ENVS if os.path.exists(os.path.join(args.indir, f"ws-{t}.npz"))]
    missing = [t for t in ENVS if t not in tasks]
    if missing:
        print("NO RENDER (skipped):", ", ".join(missing))
    rows = []
    for t in tasks:
        rows.append((t, np.load(os.path.join(args.indir, f"ws-{t}.npz"), allow_pickle=True),
                     training_goals(t), eval_goals(t, args.covdir)))
    ncol = 1 + max(len(g) for _, _, g, _ in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.5 * ncol, 3.7 * len(rows)),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for i, (task, ws, goalsets, ev) in enumerate(rows):
        img, extent = ws["img"], ws["extent"]
        cfg = ENVS[task]
        ax = axes[i][0]
        ax.imshow(img, extent=extent, origin="upper")
        # The probe points are the alignment proof: world xy drawn straight onto the image.
        pn, px = list(ws["probe_names"]), ws["probe_xy"]
        for nm, p in zip(pn, px):
            ax.scatter([p[0]], [p[1]], marker="o", s=130, facecolors="none",
                       edgecolors=GREEN, linewidths=2.0)
            ax.annotate(nm, (p[0], p[1]), color=GREEN, fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"{task} — workspace\n{cfg['task']}", fontsize=9, color=INK)
        ax.set_ylabel("y [m]", fontsize=8, color=MUTED)

        for j in range(1, ncol):
            ax = axes[i][j]
            if j - 1 >= len(goalsets):
                ax.axis("off")
                continue
            pts, label, sub = goalsets[j - 1]
            ax.imshow(img, extent=extent, origin="upper", alpha=0.30)
            dense = len(pts) > 20_000
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5 if dense else 7,
                       alpha=0.05 if dense else 0.30, color=BLUE, linewidths=0,
                       rasterized=True)
            if ev is not None:
                ax.scatter(ev[:, 0], ev[:, 1], marker="*", s=150, color=ORANGE,
                           edgecolors="white", linewidths=0.9, zorder=5)
            ax.set_title(f"{label}\n{sub}", fontsize=9, color=INK)

        for ax in axes[i]:
            # Shared limits across the row: the whole point is that the scatter and the
            # photograph are the same coordinate system.
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#d8d7d2")
        axes[i][0].set_xlabel("x [m]", fontsize=8, color=MUTED)

    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, alpha=.7,
                          label="training-goal coverage"),
               plt.Line2D([], [], marker="*", ls="", color=ORANGE, ms=14,
                          markeredgecolor="white", label="evaluation goals"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=GREEN, ms=9,
                          label="alignment probe (world xy on the photo)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0.005, 0.022, 0.995, 0.995])
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    print("saved", args.fig, f"({len(rows)} envs x {ncol} cols)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--tasks", nargs="+", required=True)
    r.add_argument("--out", default="results/deconfound")
    r.add_argument("--show-robot", action="store_true",
                   help="keep the manipulator visible (it occludes the table from above)")
    p = sub.add_parser("report")
    p.add_argument("--indir", default="results/deconfound")
    p.add_argument("--covdir", default="results/deconfound")
    p.add_argument("--fig", default="PAPER/decoding-bilinear/figs/workspace_coverage.png")
    args = ap.parse_args()
    if args.cmd == "render":
        os.makedirs(args.out, exist_ok=True)
        fail = 0
        for t in args.tasks:
            try:
                render_task(t, args.out, hide_robot=not args.show_robot)
            except Exception as e:                       # noqa: BLE001
                fail = 1
                print(f"[render] {t}: FAILED {type(e).__name__}: {e}")
        sys.exit(fail)
    report(args)


if __name__ == "__main__":
    main()
