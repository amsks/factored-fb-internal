"""Figures for the didactic section (PAPER/ICLR/didactic_new).

fig1_mechanism  the graph, the successor rows the critic must separate, and the
                action gap along the route at rank 2.
fig2_rank       action gap divided by the exact gap against rank, one line per
                room size, one panel per critic. The rank a critic needs to keep
                the decision is the claim; it depends on the loss.
fig3_bridge     the toy route gap beside the cube phase funnel.

Toy numbers come from analysis/didactic_family. Cube numbers are the canonical
terminal-outcome percentages of tab:cross_method_outcomes, copied from
scripts/figures/plot_rlbrew_figures.py in the main worktree; not recomputed.
"""

import argparse, json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import bottleneck_family_td as bf

CB = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
      "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]
FBC, DENS, BAD, GREY = CB[0], CB[4], CB[3], "#949494"
MC = {"FB": CB[0], "GCIQL": CB[1], "CRL": CB[2], "RLDP": CB[3], "PSM": CB[6]}
OUTCOMES = {
    "FB":    {"success": 49.4, "approach": 0.0, "grasp": 8.7,  "maintain": 24.9, "transport": 17.0},
    "GCIQL": {"success": 73.7, "approach": 0.0, "grasp": 11.9, "maintain": 7.8,  "transport": 6.7},
    "CRL":   {"success": 25.0, "approach": 0.9, "grasp": 25.2, "maintain": 26.3, "transport": 22.6},
    "RLDP":  {"success": 21.3, "approach": 0.1, "grasp": 14.7, "maintain": 43.8, "transport": 20.1},
    "PSM":   {"success": 37.3, "approach": 0.8, "grasp": 20.5, "maintain": 36.4, "transport": 5.0},
}
GAMMA, L0, R0 = 0.8, 2, 2                       # the graph drawn in fig1/fig3
SLAB = ["$L_0$", "$L_1$", "$b$", "$R_0$", "$R_1$"]
ROUTE_LAB = ["$L_0$\nfar", "$L_1$\napproach", "$b$\ncross", "$R_0$\ntransport"]
ARMS = [("fb", "main", "FB"), ("fb", "sphere", "FB, spherical $B$"),
        ("normsq", "main", "normalized squared"), ("density", "main", "Density-FB")]


def load(root: Path):
    out = {}
    for tag, sub in [("main", "main2"), ("sphere", "sphere2")]:
        rows = json.load(open(root / sub / "summary.json"))
        g = collections.defaultdict(list)
        for r in rows:
            g[(r["method"], r["d"], r["R"])].append(r)
        out[tag] = g
    return out


def route_ratios(runs, method, tag, d=2):
    g = runs[tag][(method, d, R0)]
    a = np.array([r["state_gap_ratios"] for r in g])[:, :4]   # the goal is not a decision
    return a.mean(0), a.min(0), a.max(0)


# ----------------------------------------------------------------- fig 1
def panel_graph(ax, fb_route):
    keeps = fb_route > 0
    for s in range(5):
        ec = "black" if s == 4 else (FBC if keeps[s] else BAD)
        ax.add_patch(Circle((s, 0), 0.3, fc="white", ec=ec, lw=2.2 if s != 4 else 1.2, zorder=3))
        ax.text(s, 0, SLAB[s], ha="center", va="center", fontsize=11, zorder=4)
    ax.text(4.0, 0.46, "goal", ha="center", fontsize=8.5)
    for a, b, lab in [(0, 1, "move"), (1, 2, "reach"), (2, 3, "cross"), (3, 4, "place")]:
        ax.add_patch(FancyArrowPatch((a + 0.32, 0.1), (b - 0.32, 0.1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.6, color="black", zorder=2))
        ax.text((a + b) / 2, 0.22, lab, ha="center", fontsize=8.5)
    for a, b, lab in [(0, 0, "wait"), (1, 0, "back"), (2, 1, "slip"), (3, 3, "wait")]:
        st = dict(arrowstyle="-|>", mutation_scale=10, lw=1.2, color=GREY, zorder=2)
        if a == b:
            ax.add_patch(FancyArrowPatch((a - .22, -.22), (a + .22, -.22),
                                         connectionstyle="arc3,rad=1.9", **st))
            ax.text(a, -0.92, lab, ha="center", fontsize=8, color=GREY)
        else:
            ax.add_patch(FancyArrowPatch((a - .2, -.22), (b + .2, -.22),
                                         connectionstyle="arc3,rad=0.4", **st))
            ax.text((a + b) / 2, -0.82, lab, ha="center", fontsize=8, color=GREY)
    ax.annotate("FB inverts\nthis decision", xy=(3.0, 0.34), xytext=(3.15, 1.12),
                fontsize=9, color=BAD, ha="center",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.3))
    for x0, x1, lab in [(-.35, 1.35, "approach"), (1.6, 2.4, "acquire control"),
                        (2.6, 4.35, "transport")]:
        ax.plot([x0, x1], [-1.25, -1.25], color="black", lw=0.9)
        ax.text((x0 + x1) / 2, -1.52, lab, ha="center", fontsize=9)
    ax.plot([], [], "o", mfc="white", mec=FBC, mew=2, ms=9, label="gap preserved")
    ax.plot([], [], "o", mfc="white", mec=BAD, mew=2, ms=9, label="gap inverted")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, handletextpad=0.4,
              title="FB, rank 2", title_fontsize=8.5)
    ax.set_xlim(-0.65, 4.65); ax.set_ylim(-1.85, 1.6); ax.axis("off")
    ax.set_title("A  five-state bottleneck", fontsize=10.5, loc="left")


def panel_rows(ax):
    p, n, b, goal, near = bf.build(L0, R0)
    P = bf.exact(p.reshape(-1, n), np.full((n, 2), 0.5), GAMMA)
    idx = [(b, 0, "$b$, cross"), (b, 1, "$b$, slip"),
           (near, 0, "$R_0$, place"), (near, 1, "$R_0$, wait")]
    M = np.stack([P[2 * s + a] for s, a, _ in idx])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=0.65, aspect="auto")
    for i in range(4):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.38 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(SLAB, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([l for _, _, l in idx], fontsize=9)
    ax.set_xlabel("discounted occupancy of each state", fontsize=9)
    ax.set_xlim(-0.5, 7.6); ax.axhline(1.5, color="black", lw=1.2)
    for y0, y1, lab, col in [
            (-.5, 1.5, f"at the crossing the two\nfutures are far apart\n"
                       f"$\\|\\Delta\\|$ = {np.linalg.norm(M[0]-M[1]):.2f}", "black"),
            (1.5, 3.5, f"at transport the two\nfutures are a near-copy\n"
                       f"$\\|\\Delta\\|$ = {np.linalg.norm(M[2]-M[3]):.2f}", BAD)]:
        ax.plot([4.62, 4.78, 4.78, 4.62], [y0 + .06, y0 + .06, y1 - .06, y1 - .06],
                color=col, lw=1.0, clip_on=False)
        ax.text(4.95, (y0 + y1) / 2, lab, fontsize=8.5, color=col, va="center", ha="left")
    ax.set_title("B  what the critic must tell apart", fontsize=10.5, loc="left")


def panel_route(ax, runs, title):
    x = np.arange(4)
    ax.axhspan(-0.6, 0.0, color=BAD, alpha=0.10, lw=0)
    ax.axhline(1.0, color="black", ls="--", lw=1.0); ax.axhline(0.0, color="black", lw=0.8)
    # FB and the normalized-squared control coincide to two decimals; draw the
    # control thin and dotted on top so both stay visible.
    style = {("fb", "main"): dict(ls="-", marker="o", lw=2.6, ms=5, color=FBC),
             ("fb", "sphere"): dict(ls="--", marker="s", lw=2.0, ms=5, color=FBC),
             ("normsq", "main"): dict(ls=":", marker="^", lw=1.3, ms=5, color=CB[2]),
             ("density", "main"): dict(ls="-", marker="o", lw=2.2, ms=5, color=DENS)}
    for method, tag, lab in ARMS:
        m, lo, hi = route_ratios(runs, method, tag)
        st = style[(method, tag)]
        ax.plot(x, m, label=lab, **st)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, lw=0)
    ax.text(0.04, 0.045, "wrong action", transform=ax.transAxes, fontsize=8.5, color=BAD)
    ax.set_xticks(x); ax.set_xticklabels(ROUTE_LAB, fontsize=9)
    ax.set_ylabel("learned gap $\\div$ exact gap", fontsize=9.5)
    ax.set_ylim(-0.6, 1.8); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=10.5, loc="left")


def panel_funnel(ax, title):
    x = np.arange(4)
    for m, o in OUTCOMES.items():
        reached = 100 - o["approach"]; acq = reached - o["grasp"]; mai = acq - o["maintain"]
        ax.plot(x, [reached, acq, mai, o["success"]], "o-", color=MC[m],
                lw=2.2 if m == "FB" else 1.6, ms=4, label=m, alpha=1.0 if m == "FB" else 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["approach", "acquire\ncontrol", "maintain\ncontrol", "transport"],
                       fontsize=9)
    ax.set_ylabel("% of episodes surviving", fontsize=9.5)
    ax.set_ylim(0, 102); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8.5)
    ax.set_title(title, fontsize=10.5, loc="left")


# ----------------------------------------------------------------- fig 2
def fig_rank(runs, ranks, rooms, out):
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.4), sharey=True)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(rooms)))
    for ax, (method, tag, lab) in zip(axes, ARMS):
        ax.axhspan(-0.6, 0.0, color=BAD, alpha=0.10, lw=0)
        ax.axhline(1.0, color="black", ls="--", lw=1.0); ax.axhline(0.0, color="black", lw=0.8)
        for c, R in zip(cols, rooms):
            y = [np.mean([r["gap_ratio_near"] for r in runs[tag][(method, d, R)]]) for d in ranks]
            ax.plot(ranks, y, "o-", color=c, lw=1.8, ms=4, label=f"$n$={R+3}")
        ax.set_xticks(ranks); ax.set_xlabel("rank $d$", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(lab, fontsize=10, loc="left")
    axes[0].set_ylabel("gap $\\div$ exact gap\nat the transport state", fontsize=9.5)
    axes[0].set_ylim(-0.6, 1.35)
    axes[0].text(0.04, 0.05, "wrong action", transform=axes[0].transAxes, fontsize=8.5, color=BAD)
    axes[-1].legend(fontsize=8, title="states", title_fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/didactic_family"))
    ap.add_argument("--out", type=Path, default=Path("PAPER/ICLR/didactic_new/figures"))
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--rooms", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = load(a.root)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.9),
                             gridspec_kw={"width_ratios": [1.12, 1.25, 1.0]})
    panel_graph(axes[0], route_ratios(runs, "fb", "main")[0])
    panel_rows(axes[1])
    panel_route(axes[2], runs, "C  action gap along the route (rank 2)")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(a.out / "fig1_mechanism.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {a.out/'fig1_mechanism.png'}")

    fig_rank(runs, a.ranks, a.rooms, a.out / "fig2_rank.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    panel_route(axes[0], runs, "A  toy: action gap along the route (rank 2)")
    panel_funnel(axes[1], "B  cube-single: episodes surviving each phase")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(a.out / "fig3_bridge.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {a.out/'fig3_bridge.png'}")
