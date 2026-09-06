"""Figures for the cube bottleneck probe (spec deliverable 2).

  (a) success H0 vs H1 per arm, per-seed points  -> fig_a_held_start.pdf/png
  (b) held_mass vs p_secure at P2, panel per arm -> fig_b_measure_scatter.pdf/png
  (c) FB row mass by phase                       -> fig_c_row_mass.pdf/png

Colour: the categorical slots of the validated reference palette, assigned to arms in a
FIXED order and never cycled -- an arm keeps its hue in every panel of every figure, so a
reader who learns "blue = plain FB" once is never repainted. Identity is never carried by
colour alone: each arm is also direct-labelled or named on its own panel. Print/paper
figures, so the light surface is the only one built.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical slots (references/palette.md), in slot order 1..4.
ARM_COLOR = {
    "fb_ref": "#2a78d6",            # slot 1 blue   -- primary A, plain FB (Touati parity)
    "density_fb_state": "#eb6834",  # slot 2 orange -- primary B, Density-FB, state input
    "density_fb": "#1baf7a",        # slot 3 aqua   -- secondary, the paper's method
    "fb": "#eda100",                # slot 4 yellow -- reference, repo FB (ortho 1.0)
}
ARM_LABEL = {
    "fb_ref": "FB (plain)",
    "density_fb_state": "Density-FB (state)",
    "density_fb": "Density-FB (obj. coords)",
    "fb": "FB (ortho 1.0)",
}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ORDER = ["fb_ref", "density_fb_state", "density_fb", "fb"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d1")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, axis="y", color="#eceae6", lw=0.8)
    ax.set_axisbelow(True)


def _arms(df):
    return [a for a in ORDER if a in set(df["critic"])]


def fig_a(per_seed: pd.DataFrame, out: Path):
    """Paired slope chart: the job is CHANGE between two conditions, per seed."""
    arms = _arms(per_seed)
    fig, axes = plt.subplots(1, len(arms), figsize=(2.1 * len(arms), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, arm in zip(axes, arms):
        _style(ax)
        sub = per_seed[per_seed.critic == arm].sort_values("seed")
        c = ARM_COLOR[arm]
        # Seeds often land on top of each other at H1; stagger the labels vertically so
        # they stay readable instead of overprinting.
        ends = sorted(float(r["H1"]) for _, r in sub.iterrows() if not pd.isna(r.get("H1")))
        for i, (_, r) in enumerate(sub.iterrows()):
            if pd.isna(r.get("H0")) or pd.isna(r.get("H1")):
                continue
            ax.plot([0, 1], [r["H0"], r["H1"]], color=c, lw=2, marker="o", ms=8,
                    mec=SURFACE, mew=2, zorder=3)
            crowded = sum(abs(e - float(r["H1"])) < 0.04 for e in ends) > 1
            dy = (i - (len(sub) - 1) / 2) * 9 if crowded else 0
            ax.annotate(f"s{int(r['seed'])}", (1, r["H1"]), textcoords="offset points",
                        xytext=(8, dy), fontsize=7, color=INK2, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["standard\nreset (H0)", "held\nstart (H1)"],
                                                  fontsize=8, color=INK2)
        ax.set_xlim(-0.35, 1.45); ax.set_ylim(-0.03, 1.03)
        ax.set_title(ARM_LABEL[arm], fontsize=9, color=INK, pad=8)
    axes[0].set_ylabel("success (mean over 5 tasks)", fontsize=9, color=INK)
    fig.suptitle("Does removing the bottleneck rescue the estimator?", fontsize=10.5,
                 color=INK, y=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)


def fig_b(t2: pd.DataFrame, out: Path, auc_by_arm=None):
    """held_mass vs p_secure at P2, one panel per arm; the P1 null as a faint inset.

    Each panel keeps its OWN y-scale: `held_mass` is a normalized positive-part share for
    FB and a softmax probability for Density-FB, so a shared axis would imply a comparison
    between quantities that are not the same thing. The comparable number is the AUC, which
    is printed on each panel.
    """
    arms = _arms(t2)
    fig, axes = plt.subplots(1, len(arms), figsize=(2.5 * len(arms), 3.0))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, arm in zip(axes, arms):
        _style(ax)
        c = ARM_COLOR[arm]
        p2 = t2[(t2.critic == arm) & (t2.phase == "P2")]
        ax.scatter(p2["p_secure"], p2["held_mass"], s=7, color=c, alpha=0.18,
                   linewidths=0, zorder=3)
        if len(p2) > 2 and p2["p_secure"].std() > 0:
            b, a = np.polyfit(p2["p_secure"], p2["held_mass"], 1)
            xs = np.linspace(0, 1, 20)
            ax.plot(xs, a + b * xs, color=INK, lw=1.5, zorder=5)
        ax.set_xlabel("p_secure (rollout ground truth)", fontsize=8, color=INK2)
        ax.set_title(ARM_LABEL[arm], fontsize=9, color=INK, pad=8)
        ax.set_xlim(-0.05, 1.05)

        # The scatter pools states; the quantity actually read is the WITHIN-STATE AUC, so
        # state it on the panel rather than leaving the reader to infer it from a cloud.
        if auc_by_arm and arm in auc_by_arm:
            ax.annotate(f"within-state AUC {auc_by_arm[arm]:.2f}", xy=(0.03, 0.04),
                        xycoords="axes fraction", fontsize=7.5, color=INK, va="bottom",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="#d6d5d1",
                                  lw=0.6, alpha=0.95))

        p1 = t2[(t2.critic == arm) & (t2.phase == "P1")]
        if len(p1):
            # Top-right: the data cloud fills the lower band, so an inset there sat on top
            # of it.
            ins = ax.inset_axes([0.60, 0.60, 0.36, 0.30])
            ins.scatter(p1["p_secure"], p1["held_mass"], s=3, color=c, alpha=0.14,
                        linewidths=0)
            ins.set_title("P1 null", fontsize=6, color=INK2, pad=2)
            ins.tick_params(labelsize=4.5, colors=INK2, length=2)
            ins.set_facecolor(SURFACE)
            for sp in ins.spines.values():
                sp.set_color("#e3e1dd")
    axes[0].set_ylabel("held mass", fontsize=9, color=INK)
    fig.suptitle("Does the measure put mass on held futures for securing actions?",
                 fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)


def fig_c(t2: pd.DataFrame, out: Path):
    """FB's predicted row mass by phase. The truth is 1/(1-gamma) at EVERY phase, so any
    slope here is the unconstrained-scale route (spec R-MEAS4)."""
    phases = ["P1", "P2", "P3"]
    arms = [a for a in _arms(t2) if a.startswith("fb")]
    if not arms:
        arms = _arms(t2)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    # The true total discounted occupancy is 1/(1-gamma) at EVERY state, so the reference
    # line is the quantity FB's row mass should equal, not merely be flat at.
    truth = 1.0 / (1.0 - 0.99)
    ax.axhline(truth, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(r"truth $1/(1-\gamma) = 100$", xy=(0.02, truth), xycoords=("axes fraction", "data"),
                fontsize=7, color=INK2, va="bottom")
    for arm in arms:
        c = ARM_COLOR[arm]
        sub = t2[t2.critic == arm]
        for seed, g in sub.groupby("seed"):
            vals = [g[g.phase == p]["row_mass"].median() for p in phases]
            ax.plot(range(len(phases)), vals, color=c, lw=2, marker="o", ms=7,
                    mec=SURFACE, mew=1.5, alpha=0.85, zorder=3)
        ax.annotate(ARM_LABEL[arm], (len(phases) - 1, vals[-1]),
                    textcoords="offset points", xytext=(8, 0), fontsize=8, color=c,
                    va="center")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(["P1\npre-contact", "P2\nat contact", "P3\nheld"], fontsize=8,
                       color=INK2)
    ax.set_xlim(-0.3, len(phases) - 0.2)
    ax.set_ylabel("mean row mass  $\\mathbb{E}_u M(s,a,z;u)$", fontsize=9, color=INK)
    ax.set_title("FB's predicted total mass, by phase\n(the truth is constant across phases)",
                 fontsize=9.5, color=INK, pad=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--root", default="analysis/cube_bottleneck")
    args = ap.parse_args()
    root = Path(args.root) / args.env
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    made = []
    f = root / "summary_per_seed.csv"
    if f.exists():
        fig_a(pd.read_csv(f), figs / "fig_a_held_start")
        made.append("a")
    parts = sorted((root / "measure_probe").glob("*.csv"))
    if parts:
        t2 = pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
        auc = {}
        sm = root / "summary_measure.csv"
        if sm.exists():
            m = pd.read_csv(sm)
            m = m[m.phase == "P2"]
            auc = m.groupby("critic")["auc_heldmass"].median().to_dict()
        fig_b(t2, figs / "fig_b_measure_scatter", auc_by_arm=auc)
        fig_c(t2, figs / "fig_c_row_mass")
        made += ["b", "c"]
    print(f"[figures] wrote {made or 'nothing (no inputs yet)'} -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
