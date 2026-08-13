"""Figures for PAPER/FB-Manipulation-ICLR.

fig_bandwidth : zero-shot control vs reward spatial support, across training z laws.
fig_scale     : predicted vs true return, confirming the predicted 1/N-hat factor.

Palette is the dataviz reference categorical theme, slots 1-3, validated all-pairs
(worst CVD dE 9.2, worst normal-vision dE 24.0). The aqua slot carries a contrast
WARN against the light surface, discharged by direct labels + the paper's tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results/spec10_torchfb")
OUT = Path("PAPER/FB-Manipulation-ICLR/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
SEEDS = (42, 43)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def summ(tag, fam, seeds=SEEDS):
    out = []
    for s in seeds:
        with open(R / f"L2_B_{tag}_s{s}.json") as f:
            out.append(json.load(f)["summary"][fam])
    return out


def recs(tag, seeds=SEEDS):
    out = []
    for s in seeds:
        with open(R / f"L2_A_{tag}_s{s}.json") as f:
            out += json.load(f)["records"]
    return out


# ---------------------------------------------------------------- figure 1 ----
def fig_bandwidth():
    groups = [("indicator", ["ball0.4", "ball0.8", "ball1.6"]),
              ("gaussian", ["smooth0.4", "smooth0.8"]),
              ("composition", ["twozone"])]
    label = {"ball0.4": r"$\epsilon$=0.4", "ball0.8": r"$\epsilon$=0.8",
             "ball1.6": r"$\epsilon$=1.6", "smooth0.4": r"$\sigma$=0.4",
             "smooth0.8": r"$\sigma$=0.8", "twozone": "two-zone"}
    arms = [("p2", "no reward-z (control)", BLUE),
            ("rzK50", "reward-z, matched form", ORANGE),
            ("rzR25", "reward-z, held-out form", AQUA)]

    fams = [f for _, fs in groups for f in fs]
    x = np.arange(len(fams), dtype=float)
    # visual gap between the three families
    for i, f in enumerate(fams):
        gi = next(k for k, (_, fs) in enumerate(groups) if f in fs)
        x[i] += gi * 0.55

    fig, ax = plt.subplots(figsize=(5.5, 2.15))
    w = 0.26
    for k, (tag, name, col) in enumerate(arms):
        v = [np.mean([d["gap_frac"] for d in summ(tag, f)]) for f in fams]
        off = (k - 1) * w
        ax.bar(x + off, v, width=w * 0.92, color=col, label=name,
               linewidth=0, zorder=3)

    ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label[f] for f in fams])
    ax.set_ylabel("control gap vs. random $z$")
    ax.set_ylim(-0.15, 1.55)
    ax.grid(axis="y", color=MUTED, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # group brackets under the axis
    trans = ax.get_xaxis_transform()
    start = 0
    for gname, fs in groups:
        xs = x[start:start + len(fs)]
        ax.plot([xs[0] - 0.42, xs[-1] + 0.42], [-0.155, -0.155], color=MUTED,
                lw=0.6, transform=trans, clip_on=False)
        ax.text((xs[0] + xs[-1]) / 2, -0.20, gname, ha="center", va="top",
                transform=trans, color=INK2, fontsize=7.5)
        start += len(fs)

    ax.annotate("support widens", xy=(x[2] + 0.35, 1.42),
                xytext=(x[0] - 0.1, 1.42), color=INK2, fontsize=7,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.26), handlelength=1.1, columnspacing=1.4)
    fig.savefig(OUT / "bandwidth.pdf"); fig.savefig("/tmp/claude-1001/-home-adityamohan-git-Factored-FB/7631d170-5136-4555-b389-3990c137f9c5/scratchpad/bandwidth.png")
    plt.close(fig)
    print("wrote", OUT / "bandwidth.pdf")


# ---------------------------------------------------------------- figure 2 ----
def fig_scale():
    naive = [(r["mc"], r["pred_raw"]) for r in recs("p2") if r["mc"] > 0]
    wt = []
    for s in SEEDS:
        with open(R / f"L2w_A_p2_s{s}.json") as f:
            wt += [(r["mc"], r["pred_raw"]) for r in json.load(f)["records"]
                   if r["mc"] > 0]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    lo, hi = 1.0, 120.0
    xs = np.array([lo, hi])
    NBAR = 0.0137

    ax.plot(xs, xs, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.plot(xs, NBAR * xs, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(lo * 1.25, lo * 1.9, r"$y=x$", color=INK2, fontsize=7,
            ha="left", va="bottom")
    ax.text(lo * 1.25, NBAR * lo * 0.42, r"$y=\bar{N}x$", color=INK2,
            fontsize=7, ha="left", va="top")

    for (pts, col, name) in ((naive, BLUE, "naive readout"),
                             (wt, ORANGE, r"$1/\hat{N}$-weighted")):
        a = np.array(pts)
        ax.scatter(a[:, 0], a[:, 1], s=5, color=col, alpha=0.45,
                   linewidths=0, zorder=3, label=name)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Monte-Carlo return")
    ax.set_ylabel("predicted return")
    ax.grid(color=MUTED, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(5e-3, 4e2)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0,
              scatterpoints=1, markerscale=2.0)

    med = np.median([p / m for m, p in naive])
    print(f"  [fig_scale] naive median pred/MC = {med:.4f}  (N-bar = {NBAR})")
    fig.savefig(OUT / "scale.pdf"); fig.savefig("/tmp/claude-1001/-home-adityamohan-git-Factored-FB/7631d170-5136-4555-b389-3990c137f9c5/scratchpad/scale.png")
    plt.close(fig)
    print("wrote", OUT / "scale.pdf")


if __name__ == "__main__":
    fig_bandwidth()
    fig_scale()
