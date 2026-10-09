"""Schematic: per env, where the actor's TRAINING task vectors z lie (hindsight-goal
region, random-sphere region) versus where the EVALUATION task vectors land.

Reads analysis/z_coverage/z_<env>[_exact].npz (from compute_z_coverage.py). One panel per
env: a 2D PCA fitted on that env's goal z's; the goal region and sphere region are drawn as
kernel-density contours enclosing 50% and 90% of their samples (not scatters); the 5 Route-B
eval z's are dark markers with a thin convex hull; k-hot and dense-bounded reward families
are small distinct markers. SVG keeps editable text (svg.fonttype='none').

  /dev/shm/.venv-jax/bin/python -m analysis.z_coverage.plot_z_coverage            # campaign
  /dev/shm/.venv-jax/bin/python -m analysis.z_coverage.plot_z_coverage --set exact
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN = os.path.join(ROOT, "analysis", "z_coverage")
OUT = os.path.join(ROOT, "PAPER", "ICLR", "figures", "z_coverage")

ENVS = [("cube-single", "cube-single"), ("cube-double", "cube-double"), ("scene", "scene"),
        ("puzzle-3x3", "puzzle-3x3"), ("antmaze-medium", "antmaze-medium")]

# Figure-1 family (scripts/probes/cube_bottleneck/fig_carry_strip_v2.py): Density-FB orange-red,
# data-prior grey, ink. Validated for CVD separation in the README.
GOAL_FILL_90, GOAL_FILL_50, GOAL_EDGE = "#fbdcc9", "#f4a06e", "#e0602a"
SPH_FILL_90, SPH_FILL_50, SPH_EDGE = "#ececea", "#dcdcd8", "#8f8f8a"
INK, INK2, FRAME = "#0b0b0b", "#52514e", "#cfcfcb"
KHOT, DENSE = "#2f6fc4", "#6a4fb5"

plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})


def pca2(zg):
    mu = zg.mean(0)
    u, s, vt = np.linalg.svd(zg - mu, full_matrices=False)
    ev = (s ** 2) / (s ** 2).sum()
    return mu, vt[:2].T, ev[:2].sum()


def kde_levels(xy, grid_n=160, pad=0.15, mass=(0.5, 0.9)):
    """Density on a grid plus the density thresholds enclosing `mass` of the SAMPLE
    (the p-th mass level = the (1-p) quantile of the points' own densities)."""
    kde = gaussian_kde(xy.T, bw_method="scott")
    lo, hi = xy.min(0), xy.max(0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    dens_pts = kde(xy.T)
    levels = [float(np.quantile(dens_pts, 1 - m)) for m in mass]
    return X, Y, Z, levels


def draw_region(ax, xy, fill90, fill50, edge, z0, outline_on_top=False):
    X, Y, Z, (l50, l90) = kde_levels(xy)
    top = Z.max() * 1.01
    ax.contourf(X, Y, Z, levels=[l90, top], colors=[fill90], zorder=z0, antialiased=True)
    ax.contourf(X, Y, Z, levels=[l50, top], colors=[fill50], zorder=z0 + 0.1, antialiased=True)
    if outline_on_top:
        # the sphere blob sits INSIDE the goal blob in every goal-PCA plane (its per-axis std
        # is 1, the goal cloud's ~2), so its fill is covered; a dashed outline keeps it legible.
        ax.contour(X, Y, Z, levels=[l90], colors=[edge], linewidths=0.6, linestyles="dashed", zorder=4)
    else:
        ax.contour(X, Y, Z, levels=[l90], colors=[edge], linewidths=0.5, zorder=z0 + 0.2)


def knn_goal(cloud, q, k=10, loo=False):
    d2 = (q ** 2).sum(-1)[:, None] + (cloud ** 2).sum(-1)[None] - 2 * q @ cloud.T
    if loo:
        d2[np.arange(len(q)), np.arange(len(q))] = np.inf
    return np.sqrt(np.maximum(np.sort(d2, axis=1)[:, :k], 0)).mean(1)


def strip(ax, d, R, first):
    """The 50-d readout the PCA plane cannot show: each z's mean distance to its 10 nearest
    goal z's, in units of the goal cloud's RMS radius R. Bands = 5-95% and 25-75% of the goal
    cloud's own leave-one-out values (orange) and of random-sphere z's (grey)."""
    zg, zs, ze, zk, zd = d["z_goal"], d["z_sphere"], d["z_eval"], d["z_khot"], d["z_dense"]
    g = knn_goal(zg, zg[:2048].copy(), loo=True) / R
    sp = knn_goal(zg, zs[:2048]) / R
    e, kh, de = knn_goal(zg, ze) / R, knn_goal(zg, zk) / R, knn_goal(zg, zd) / R
    for vals, c90, c50, y in ((g, GOAL_FILL_90, GOAL_FILL_50, 0.62), (sp, SPH_FILL_90, SPH_FILL_50, 0.62)):
        q5, q25, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.75, 0.95])
        ax.barh(y, q95 - q5, left=q5, height=0.34, color=c90, zorder=1)
        ax.barh(y, q75 - q25, left=q25, height=0.34, color=c50, zorder=2)
    ax.plot(kh, np.full(len(kh), 0.25), ls="none", marker="^", ms=3.4, mfc=KHOT, mec="white", mew=0.4, zorder=5)
    ax.plot(de, np.full(len(de), 0.25), ls="none", marker="D", ms=2.9, mfc="white", mec=DENSE, mew=0.8, zorder=5)
    ax.plot(e, np.full(len(e), 0.62), ls="none", marker="o", ms=4.2, mfc=INK, mec="white", mew=0.6, zorder=6)
    hi = max(np.quantile(sp, 0.99), e.max(), kh.max(), de.max()) * 1.06
    ax.set_xlim(0, hi); ax.set_ylim(0, 0.95)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0][: 3 if hi > 1.0 else 2])
    ax.tick_params(axis="x", labelsize=5.4, colors=INK2, length=1.5, pad=1, width=0.5)
    for k_, sp_ in ax.spines.items():
        sp_.set_visible(k_ == "bottom"); sp_.set_color(FRAME)
    if first:
        ax.set_xlabel("in 50-d: mean distance to the 10 nearest goal z's, in goal-cloud RMS radii R",
                      fontsize=5.6, color=INK2, labelpad=1, loc="left")
        ax.text(0.01, 0.98, "goal cloud's own", fontsize=5.2, color="#6e1e05", transform=ax.transAxes, va="top")
        ax.text(0.99, 0.98, "random z's", fontsize=5.2, color=INK2, transform=ax.transAxes, va="top", ha="right")


def panel(ax, path, title, first, legend_ax=None):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    zg, zs, ze, zk, zd = d["z_goal"], d["z_sphere"], d["z_eval"], d["z_khot"], d["z_dense"]
    mu, W, evr = pca2(zg)
    P = lambda z: (z - mu) @ W
    pg, ps, pe, pk, pd_ = P(zg), P(zs), P(ze), P(zk), P(zd)

    draw_region(ax, ps, SPH_FILL_90, SPH_FILL_50, SPH_EDGE, z0=1, outline_on_top=True)
    draw_region(ax, pg, GOAL_FILL_90, GOAL_FILL_50, GOAL_EDGE, z0=2)

    ax.plot(pk[:, 0], pk[:, 1], ls="none", marker="^", ms=4.2, mfc=KHOT, mec="white", mew=0.5, zorder=5)
    ax.plot(pd_[:, 0], pd_[:, 1], ls="none", marker="D", ms=3.6, mfc="white", mec=DENSE, mew=0.9, zorder=5)
    if len(pe) >= 3:
        try:
            h = ConvexHull(pe)
            v = np.append(h.vertices, h.vertices[0])
            ax.plot(pe[v, 0], pe[v, 1], color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=6)
        except Exception:
            pass
    ax.plot(pe[:, 0], pe[:, 1], ls="none", marker="o", ms=5.0, mfc=INK, mec="white", mew=0.7, zorder=7)

    # axis extent: everything, with margin
    allp = np.vstack([pg, ps, pe, pk, pd_])
    lo, hi = allp.min(0), allp.max(0)
    span = (hi - lo)
    ax.set_xlim(lo[0] - 0.08 * span[0], hi[0] + 0.08 * span[0])
    ax.set_ylim(lo[1] - 0.08 * span[1], hi[1] + 0.30 * span[1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(FRAME)
    ax.set_facecolor("white")

    med = meta["median_d_goal_rms"]["eval"]
    ax.set_title(title, fontsize=8.5, color=INK, pad=3, loc="left")
    ax.text(0.03, 0.965, f"nearest goal z: median {med:.2f} R",
            transform=ax.transAxes, fontsize=5.9, color=INK2, va="top", ha="left")
    ax.text(0.98 if not first else 0.03, 0.03, f"PCA on goal z's, {100 * evr:.0f}% var.", transform=ax.transAxes,
            fontsize=5.6, color=INK2, va="bottom", ha="right" if not first else "left")

    if first:
        # label the regions once: point at the goal centroid, the sphere centroid, and one eval marker
        gc, sc = pg.mean(0), ps.mean(0)
        # a point ON the goal 90% rim (leftmost goal-region point), the sphere centre, the
        # eval marker farthest from the goal centre
        rim = pg[np.argmin(pg[:, 0] - 0.6 * pg[:, 1])]
        ax.annotate("trained on:\nhindsight goals", xy=rim, xytext=(0.03, 0.86), textcoords="axes fraction",
                    fontsize=5.9, color="#6e1e05", ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color="#6e1e05", lw=0.5, shrinkB=1))
        ax.annotate("trained on:\nrandom sphere", xy=sc, xytext=(0.97, 0.05), textcoords="axes fraction",
                    fontsize=5.9, color=INK2, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkB=1))
        i = int(np.argmax(np.linalg.norm(pe - gc, axis=1)))
        ax.annotate("evaluated on:\ntask rewards", xy=pe[i], xytext=(0.97, 0.84), textcoords="axes fraction",
                    fontsize=5.9, color=INK, ha="right", va="top",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkB=3))
    return meta, d


def make(tag):
    suffix = "" if tag == "campaign" else f"_{tag}"
    files = [(k, t, os.path.join(IN, f"z_{k}{suffix}.npz")) for k, t in ENVS]
    files = [f for f in files if os.path.exists(f[2])]
    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(1.45 * n + 0.2, 2.55), facecolor="white",
                             gridspec_kw={"height_ratios": [1.0, 0.24], "hspace": 0.32, "wspace": 0.08})
    axes = np.atleast_2d(axes).reshape(2, n)
    metas = []
    for i, (k, t, p) in enumerate(files):
        meta, d = panel(axes[0, i], p, t, first=(i == 0))
        strip(axes[1, i], d, meta["goal_rms"], first=(i == 0))
        metas.append(meta)
    handles = [
        Patch(facecolor=GOAL_FILL_50, edgecolor=GOAL_EDGE, lw=0.5, label="goal-task region (50% / 90% mass)"),
        Patch(facecolor=SPH_FILL_50, edgecolor=SPH_EDGE, lw=0.6, ls="--", label="sphere-task region (90% mass)"),
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc=INK, mec="white", mew=0.6, label="OGBench eval rewards (5 tasks)"),
        Line2D([], [], ls="none", marker="^", ms=4.0, mfc=KHOT, mec="white", mew=0.5, label="k-hot rewards"),
        Line2D([], [], ls="none", marker="D", ms=3.4, mfc="white", mec=DENSE, mew=0.9, label="dense bounded rewards"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.4,
               handletextpad=0.5, columnspacing=1.4, bbox_to_anchor=(0.5, -0.015))
    sub = ("actor trained on hindsight goals + random sphere z (1M, z_rand_frac 0.5)" if tag == "campaign"
           else "actor trained on hindsight-goal z only (500k, exact kernel reward, z_rand_frac 0)")
    fig.text(0.01, 0.995, sub, fontsize=6.2, color=INK2, va="top", ha="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.16)
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, "fig_z_coverage" + ("" if tag == "campaign" else f"_{tag}"))
    for e in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{e}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print("wrote", stem + ".{pdf,svg,png}")
    return metas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="campaign", choices=["campaign", "exact"])
    make(ap.parse_args().set)
