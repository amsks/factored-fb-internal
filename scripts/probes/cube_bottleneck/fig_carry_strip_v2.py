"""Carry-strip figure: each critic's predicted future cube position, frame by frame, during a recorded carry.

Critic outputs are cached in carry_strip_cache.npz; pass --recompute to rebuild them.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path("analysis/cube_bottleneck/cube-single-play-v0")
CACHE = ROOT / "carry_strip_cache.npz"
CUBE_OBS = slice(19, 22)
ORDER = ["fb_ref", "density_fb_state"]
LABEL = {"fb_ref": "FB", "density_fb_state": "Density-FB"}
COLOR = {"fb_ref": "#1f5fbf", "density_fb_state": "#d9531e"}
CMAPS = {
    "fb_ref": LinearSegmentedColormap.from_list(
        "fbblue", ["#ffffff", "#cfdff5", "#7fa6e0", "#2f6fc4", "#0d2f66"]
    ),
    "density_fb_state": LinearSegmentedColormap.from_list(
        "dfborange", ["#ffffff", "#fbdcc9", "#f4a06e", "#e0602a", "#6e1e05"]
    ),
    "prior": LinearSegmentedColormap.from_list(
        "prior", ["#ffffff", "#e3e3e0", "#b9b9b4", "#807f7a", "#3d3d3a"]
    ),
}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"


def world_to_plot(xy):
    xy = np.asarray(xy)
    return np.stack([xy[..., 1], xy[..., 0]], axis=-1)


# Compute (cached)
def compute(args):
    from scripts.probes.cube_bottleneck import agents as A
    from scripts.probes.cube_bottleneck import banks as BK
    from scripts.probes.cube_bottleneck import restore as R
    from scripts.probes.cube_bottleneck.fig_carry_strip import (
        b_embed,
        fit_probe,
        predicted_cube_mass,
    )
    from scripts.probes.cube_bottleneck.fig_outcome_map import CM, task_goals

    raw = BK.load_raw(args.env)
    rows = np.linspace(args.start, args.end - 1, args.frames).astype(int)
    obs_f = np.asarray(raw["observations"][rows], np.float32)
    act_f = np.asarray(raw["actions"][rows], np.float32)
    true_xy = world_to_plot(obs_f[:, CUBE_OBS][:, :2] * CM)
    lifted = (np.asarray(raw["qpos"])[rows][:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH) & (
        obs_f[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH
    )
    # Full cube path over the segment
    seg = np.arange(args.start, args.end)
    path_xy = world_to_plot(
        np.asarray(raw["observations"][seg], np.float32)[:, CUBE_OBS][:, :2] * CM
    )

    pool_obs = dict(np.load(ROOT / "pool.npz"))["obs"]
    pool_xy = world_to_plot(pool_obs[:, CUBE_OBS][:, :2] * CM)

    import ogbench

    env = ogbench.make_env_and_datasets(
        args.env, env_only=True, render_mode="rgb_array"
    )
    env.reset(seed=0, options=dict(task_id=args.task, render_goal=False))
    imgs = []
    for r in rows:
        R.set_state(env, np.asarray(raw["qpos"][r]), np.asarray(raw["qvel"][r]))
        R.forward(env)
        imgs.append(np.asarray(env.render()))
    imgs = np.stack(imgs)

    probe_idx = np.random.default_rng(0).choice(
        len(raw["observations"]), size=args.probe_rows, replace=False
    )
    obs_p = np.asarray(raw["observations"][probe_idx], np.float32)
    tgt_cube = world_to_plot(obs_p[:, CUBE_OBS][:, :2] * CM)

    out = dict(
        rows=rows,
        true_xy=true_xy,
        lifted=lifted,
        path_xy=path_xy,
        pool_xy=pool_xy,
        imgs=imgs,
    )
    goal_xy = None
    for a in args.critics:
        arm = A.load_arm(Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}")
        goals = task_goals(arm, args.task, args.goal_draws, 20260904)
        if goal_xy is None:
            goal_xy = world_to_plot(
                np.mean([g[CUBE_OBS][:2] for g in goals], axis=0) * CM
            )
        W, r2 = fit_probe(arm, obs_p, tgt_cube)
        E = b_embed(arm, obs_f)
        out[f"dec_{a}"] = np.concatenate([E, np.ones((len(E), 1), np.float32)], 1) @ W
        out[f"r2_{a}"] = r2
        ps, negs = [], []
        for i in range(len(rows)):
            p, neg = predicted_cube_mass(arm, obs_f[i], act_f[i], goals, pool_obs)
            ps.append(p)
            negs.append(neg)
        out[f"mass_{a}"] = np.stack(ps)  # [frames, pool]
        out[f"neg_{a}"] = float(np.mean(negs))
        print(
            f"  {a:20s} probe R2 {r2:+.3f}  decode err (cm) "
            f"{np.round(np.linalg.norm(out[f'dec_{a}'] - true_xy, axis=1), 1).tolist()}"
        )
    out["goal_xy"] = goal_xy
    np.savez(CACHE, **out)
    print(f"cached -> {CACHE}")
    return out


# Render
def kde_grid(pts, w, xs, ys, bw):
    """Weighted Gaussian KDE of 2-D points on a grid; returns [len(xs), len(ys)]."""
    w = np.asarray(w, float)
    w = w / max(w.sum(), 1e-12)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)
    for (px, py), wi in zip(pts, w):
        if wi <= 0:
            continue
        Z += wi * np.exp(-0.5 * (((X - px) ** 2 + (Y - py) ** 2) / bw**2))
    return Z


def render(d, args):
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )
    critics = args.critics
    rows = d["rows"]
    n = len(rows)
    true_xy, lifted, path_xy, pool_xy, goal_xy, imgs = (
        d["true_xy"],
        d["lifted"],
        d["path_xy"],
        d["pool_xy"],
        d["goal_xy"],
        d["imgs"],
    )
    # Extent: pool 0.5-99.5 percentile, cube path and goal, plus margin
    lo = (
        np.minimum(
            np.percentile(pool_xy, 0.5, axis=0), np.minimum(path_xy.min(0), goal_xy)
        )
        - 3.0
    )
    hi = (
        np.maximum(
            np.percentile(pool_xy, 99.5, axis=0), np.maximum(path_xy.max(0), goal_xy)
        )
        + 3.0
    )
    xs = np.linspace(lo[0], hi[0], 140)
    ys = np.linspace(lo[1], hi[1], 70)
    ext = [lo[0], hi[0], lo[1], hi[1]]
    aspect_hw = (hi[1] - lo[1]) / (hi[0] - lo[0])  # panel height / width

    grids = {
        a: [kde_grid(pool_xy, d[f"mass_{a}"][i], xs, ys, args.bw) for i in range(n)]
        for a in critics
    }
    prior = kde_grid(pool_xy, np.ones(len(pool_xy)), xs, ys, args.bw)

    # Crop the sky from the renders
    H = imgs.shape[1]
    crop = slice(int(0.18 * H), int(0.97 * H))
    imgs_c = imgs[:, crop]
    img_hw = imgs_c.shape[1] / imgs_c.shape[2]

    ncol = n + 1  # column 0 = prior / legend
    W = 7.0
    pw = W / (ncol + 0.15)
    heights = [pw * img_hw] + [pw * aspect_hw] * len(critics)
    fig = plt.figure(figsize=(W, sum(heights) + 0.25), facecolor=SURFACE)
    gs = fig.add_gridspec(
        1 + len(critics),
        ncol,
        height_ratios=heights,
        width_ratios=[1.0] * ncol,
        left=0.01,
        right=0.99,
        top=0.95,
        bottom=0.03,
        wspace=0.07,
        hspace=0.12,
    )

    li = np.flatnonzero(lifted)
    first, last = (int(li[0]), int(li[-1])) if len(li) else (n, -1)
    after = getattr(args, "after_label", None) or (
        "placed"
        if ("first_drop" not in d or bool(d.get("success", False)))
        else "dropped"
    )
    phase = [
        "carry" if l else ("reach" if i < first else after)
        for i, l in enumerate(lifted)
    ]
    for c in range(n):
        ax = fig.add_subplot(gs[0, c + 1])
        ax.imshow(imgs_c[c])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cfcfcb")
        ax.set_title(
            f"t{c + 1}  ·  {phase[c]}",
            fontsize=7.5,
            color=INK if lifted[c] else INK2,
            pad=2.5,
        )

    def draw_panel(ax, Z, cmap, c=None, show_dec=None):
        ax.set_facecolor("white")
        ax.imshow(
            (Z / max(Z.max(), 1e-12)).T,
            origin="lower",
            extent=ext,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        ax.plot(
            path_xy[:, 0], path_xy[:, 1], color="#9a9a96", lw=0.7, alpha=0.8, zorder=4
        )
        ax.plot(
            goal_xy[0],
            goal_xy[1],
            marker="*",
            ms=9,
            color="#ffd23f",
            mec="#1a1a1a",
            mew=0.7,
            zorder=7,
            ls="none",
        )
        if c is not None:
            ax.plot(
                true_xy[c, 0],
                true_xy[c, 1],
                marker="o",
                ms=5.2,
                mfc="#0b0b0b",
                mec="white",
                mew=0.9,
                zorder=8,
                ls="none",
            )
        if show_dec is not None:
            ax.plot(
                show_dec[0],
                show_dec[1],
                marker="x",
                ms=6,
                color="#00a86b",
                mew=1.7,
                zorder=8,
                ls="none",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cfcfcb")

    # Left column: data prior and legend
    axp = fig.add_subplot(gs[0, 0])
    axp.set_anchor("S")
    draw_panel(axp, prior, CMAPS["prior"])
    axp.set_aspect(
        aspect_hw / (axp.get_position().height / axp.get_position().width)
        if False
        else "auto"
    )
    axp.set_title("data prior", fontsize=7.5, color=INK, pad=2.5)
    # Match the prior axis to the map aspect
    bb = axp.get_position()
    mh = bb.width * aspect_hw * (fig.get_figwidth() / fig.get_figheight())
    axp.set_position([bb.x0, bb.y0, bb.width, min(bb.height, mh)])

    axl = fig.add_subplot(gs[1:, 0])
    axl.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=5.2,
            mfc="#0b0b0b",
            mec="white",
            mew=0.9,
            label="cube now",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="none",
            ms=6,
            color="#00a86b",
            mew=1.7,
            label="decoded cube",
        ),
        Line2D(
            [],
            [],
            marker="*",
            ls="none",
            ms=9,
            color="#ffd23f",
            mec="#1a1a1a",
            mew=0.7,
            label="goal",
        ),
        Line2D([], [], color="#9a9a96", lw=0.9, label="cube path"),
    ]
    axl.legend(
        handles=handles,
        loc="center",
        fontsize=6.6,
        frameon=False,
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.7,
    )

    for r, a in enumerate(critics, start=1):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c + 1])
            draw_panel(ax, grids[a][c], CMAPS[a], c=c, show_dec=d[f"dec_{a}"][c])

    r2s = "   ".join(
        f"{LABEL[a]} probe R2 = {float(d[f'r2_{a}']):.2f}" for a in critics
    )
    print(f"caption numbers: {r2s}")
    out = Path(args.out) if args.out else ROOT / "figures" / "fig_carry_strip_v2"
    for e in ("pdf", "png"):
        fig.savefig(
            f"{out}.{e}",
            dpi=300,
            facecolor=SURFACE,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    print(f"wrote {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=65090)
    ap.add_argument("--end", type=int, default=65130)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--probe-rows", type=int, default=20000)
    ap.add_argument("--bw", type=float, default=2.5, help="KDE bandwidth, cm")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if CACHE.exists() and not args.recompute:
        d = dict(np.load(CACHE, allow_pickle=True))
        print(f"loaded cache {CACHE}")
    else:
        d = compute(args)
    render(d, args)


if __name__ == "__main__":
    main()
