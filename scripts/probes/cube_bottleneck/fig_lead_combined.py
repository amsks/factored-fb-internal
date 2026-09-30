"""Lead figure: two policies from the SAME recorded held state, same goal.

    row 1  frames of plain FB's rollout            (it drops the cube)
    row 2  FB's predicted future cube position on those frames
    row 3  Density-FB's predicted future cube position on ITS frames
    row 4  frames of Density-FB's rollout          (it carries the cube to the goal)

Left column: the data prior of cube positions and the legend, vertically centred.
Inputs: drop_strip_cache.npz (FB policy, bank row 21) and drop_strip_cache_density.npz
(Density-FB policy, same row), both produced by fig_drop_strip.py.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.probes.cube_bottleneck import fig_carry_strip_v2 as V2

ROOT = V2.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb-cache", default=str(ROOT / "drop_strip_cache.npz"))
    ap.add_argument("--density-cache", default=str(ROOT / "drop_strip_cache_density.npz"))
    ap.add_argument("--bw", type=float, default=2.5)
    ap.add_argument("--out", default=str(ROOT / "figures" / "fig_lead_combined"))
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

    F = dict(np.load(args.fb_cache, allow_pickle=True))
    D = dict(np.load(args.density_cache, allow_pickle=True))
    assert int(F["bank_row"]) == int(D["bank_row"]), "the two rollouts must start from the same bank row"
    n = len(F["rows"]); assert len(D["rows"]) == n
    pool_xy = F["pool_xy"]; goal_xy = F["goal_xy"]

    # shared extent over both rollouts
    allpath = np.concatenate([F["path_xy"], D["path_xy"]])
    lo = np.minimum(np.percentile(pool_xy, 0.5, axis=0), np.minimum(allpath.min(0), goal_xy)) - 3.0
    hi = np.maximum(np.percentile(pool_xy, 99.5, axis=0), np.maximum(allpath.max(0), goal_xy)) + 3.0
    xs = np.linspace(lo[0], hi[0], 140); ys = np.linspace(lo[1], hi[1], 70)
    ext = [lo[0], hi[0], lo[1], hi[1]]; aspect_hw = (hi[1] - lo[1]) / (hi[0] - lo[0])

    gF = [V2.kde_grid(pool_xy, F["mass_fb_ref"][i], xs, ys, args.bw) for i in range(n)]
    gD = [V2.kde_grid(pool_xy, D["mass_density_fb_state"][i], xs, ys, args.bw) for i in range(n)]
    prior = V2.kde_grid(pool_xy, np.ones(len(pool_xy)), xs, ys, args.bw)

    def crop(imgs):
        H = imgs.shape[1]; return imgs[:, int(0.18 * H):int(0.97 * H)]
    imF, imD = crop(F["imgs"]), crop(D["imgs"])
    img_hw = imF.shape[1] / imF.shape[2]

    def phases(d, at_goal_cm=4.0):
        lifted = d["lifted"]; li = np.flatnonzero(lifted)
        first = int(li[0]) if len(li) else n
        dist = np.linalg.norm(d["true_xy"] - goal_xy, axis=1)
        succ = bool(d.get("success", False))
        out = []
        for i, l in enumerate(lifted):
            if succ and dist[i] < at_goal_cm: out.append("at goal")
            elif l: out.append("carry")
            elif i < first: out.append("reach")
            else: out.append("regrasp" if succ else "dropped")
        return out
    phF, phD = phases(F), phases(D)

    ncol = n + 1
    W = 7.0; pw = W / (ncol + 0.15)
    heights = [pw * img_hw, pw * aspect_hw, pw * aspect_hw, pw * img_hw]
    fig = plt.figure(figsize=(W, sum(heights) + 0.45), facecolor="white")
    gs = fig.add_gridspec(4, ncol, height_ratios=heights, width_ratios=[1.0] * ncol,
                          left=0.01, right=0.99, top=0.955, bottom=0.045, wspace=0.07, hspace=0.14)

    def frame_row(r, imgs, ph, rows, title_top):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#cfcfcb")
            lab = f"t = {int(rows[c])}  ·  {ph[c]}"
            if title_top:
                ax.set_title(lab, fontsize=7.2, color=V2.INK if ph[c] in ("carry", "at goal") else V2.INK2, pad=2.5)
            else:
                ax.set_xlabel(lab, fontsize=7.2, color=V2.INK if ph[c] in ("carry", "at goal") else V2.INK2, labelpad=2.5)

    def map_panel(ax, Z, cmap, path, c=None, true_xy=None, dec=None):
        ax.set_facecolor("white")
        ax.imshow((Z / max(Z.max(), 1e-12)).T, origin="lower", extent=ext, aspect="auto",
                  cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], color="#9a9a96", lw=0.7, alpha=0.85, zorder=4)
        ax.plot(goal_xy[0], goal_xy[1], marker="*", ms=9, color="#ffd23f", mec="#1a1a1a", mew=0.7, zorder=7, ls="none")
        if true_xy is not None:
            ax.plot(true_xy[0], true_xy[1], marker="o", ms=5.2, mfc="#0b0b0b", mec="white", mew=0.9, zorder=8, ls="none")
        if dec is not None:
            ax.plot(dec[0], dec[1], marker="x", ms=6, color="#00a86b", mew=1.7, zorder=8, ls="none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#cfcfcb")

    frame_row(0, imF, phF, F["rows"], title_top=True)
    for c in range(n):
        map_panel(fig.add_subplot(gs[1, c + 1]), gF[c], V2.CMAPS["fb_ref"], F["path_xy"], c,
                  F["true_xy"][c], F["dec_fb_ref"][c])
    for c in range(n):
        map_panel(fig.add_subplot(gs[2, c + 1]), gD[c], V2.CMAPS["density_fb_state"], D["path_xy"], c,
                  D["true_xy"][c], D["dec_density_fb_state"][c])
    frame_row(3, imD, phD, D["rows"], title_top=False)

    # left column: prior + legend, stacked and centred vertically
    sub = gs[:, 0].subgridspec(4, 1, height_ratios=[1.0, 1.15, 1.6, 1.0], hspace=0.05)
    axp = fig.add_subplot(sub[1]); axp.set_anchor("C")
    map_panel(axp, prior, V2.CMAPS["prior"], None)
    bb = axp.get_position(); mh = bb.width * aspect_hw * (fig.get_figwidth() / fig.get_figheight())
    axp.set_position([bb.x0, bb.y0 + (bb.height - min(bb.height, mh)) / 2, bb.width, min(bb.height, mh)])
    axp.set_title("data prior", fontsize=7.5, color=V2.INK, pad=2.5)
    axl = fig.add_subplot(sub[2]); axl.axis("off")
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.2, mfc="#0b0b0b", mec="white", mew=0.9, label="cube now"),
        Line2D([], [], marker="x", ls="none", ms=6, color="#00a86b", mew=1.7, label="decoded cube"),
        Line2D([], [], marker="*", ls="none", ms=9, color="#ffd23f", mec="#1a1a1a", mew=0.7, label="goal"),
        Line2D([], [], color="#9a9a96", lw=0.9, label="cube path"),
    ]
    axl.legend(handles=handles, loc="upper center", fontsize=6.6, frameon=False,
               handletextpad=0.5, borderaxespad=0.0, labelspacing=0.7)

    for e in ("pdf", "png"):
        fig.savefig(f"{args.out}.{e}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print("wrote", args.out, "| FB rows", F["rows"].tolist(), "Density rows", D["rows"].tolist(),
          "| bank row", int(F["bank_row"]))


if __name__ == "__main__":
    main()
