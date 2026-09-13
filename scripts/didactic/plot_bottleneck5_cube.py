"""Two-scale figure: the five-state bottleneck toy next to the cube phase funnel.

A  the toy MDP, each state named with its manipulation analogue, decision states
   coloured by whether unrestricted FB keeps or inverts the action gap.
B  the successor rows the critic has to tell apart at the crossing and at the
   transport step: far apart at the crossing, a near-copy at transport.
C  learned action gap divided by the exact gap under the fixed continuation
   policy, at each decision state. 1 is exact, 0 erases the decision, below 0
   inverts the greedy action.
D  the cube phase funnel already in the paper.

Toy numbers are read from the run directories. Cube numbers are the canonical
terminal-outcome percentages of tab:cross_method_outcomes, copied from
scripts/figures/plot_rlbrew_figures.py in the main worktree; not recomputed here.
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

CB = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
      "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]
FB, DENS, BAD, GREY = CB[0], CB[4], CB[3], "#949494"

OUTCOMES = {
    "FB":    {"success": 49.4, "approach": 0.0, "grasp": 8.7,  "maintain": 24.9, "transport": 17.0},
    "GCIQL": {"success": 73.7, "approach": 0.0, "grasp": 11.9, "maintain": 7.8,  "transport": 6.7},
    "CRL":   {"success": 25.0, "approach": 0.9, "grasp": 25.2, "maintain": 26.3, "transport": 22.6},
    "RLDP":  {"success": 21.3, "approach": 0.1, "grasp": 14.7, "maintain": 43.8, "transport": 20.1},
    "PSM":   {"success": 37.3, "approach": 0.8, "grasp": 20.5, "maintain": 36.4, "transport": 5.0},
}
MC = {"FB": CB[0], "GCIQL": CB[1], "CRL": CB[2], "RLDP": CB[3], "PSM": CB[6]}

GAMMA = 0.8
NEXTS = np.array([[0, 1], [2, 0], [3, 1], [4, 3], [3, 4]])
SLAB = ["$L_0$", "$L_1$", "$b$", "$R_0$", "$R_1$"]
ROUTE = [0, 1, 2, 3]          # the goal state itself is not a control decision
ROUTE_LAB = ["$L_0$\nfar", "$L_1$\napproach", "$b$\ncross", "$R_0$\ntransport"]
POS = {s: (float(s), 0.0) for s in range(5)}


def successors(gamma=GAMMA):
    n = 5
    p = np.eye(n)[NEXTS]
    pi = np.full((n, 2), 0.5)
    T = (pi[:, :, None] * p).sum(1)
    return p.reshape(-1, n) @ ((1 - gamma) * np.linalg.inv(np.eye(n) - gamma * T))


def ratios(run: Path, method: str):
    rows = json.loads((run / f"{method}_latest.json").read_text())
    g = np.array([r["all_state_gaps"] for r in rows])
    t = np.array(rows[0]["true_state_gaps"])
    r = g / t
    return r[:, ROUTE].mean(0), r[:, ROUTE].min(0), r[:, ROUTE].max(0)


def panel_graph(ax, fb_ratio):
    keeps = {s: fb_ratio[i] > 0 for i, s in enumerate(ROUTE)}
    for s, (x, y) in POS.items():
        ec = "black" if s == 4 else (FB if keeps[s] else BAD)
        ax.add_patch(Circle((x, y), 0.3, fc="white", ec=ec, lw=2.2 if s != 4 else 1.2, zorder=3))
        ax.text(x, y, SLAB[s], ha="center", va="center", fontsize=11, zorder=4)
    ax.text(4.0, 0.45, "goal", ha="center", fontsize=8.5)
    for a, b, lab in [(0, 1, "move"), (1, 2, "reach"), (2, 3, "cross"), (3, 4, "place")]:
        ax.add_patch(FancyArrowPatch((a + 0.32, 0.1), (b - 0.32, 0.1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.6, color="black", zorder=2))
        ax.text((a + b) / 2, 0.22, lab, ha="center", fontsize=8.5)
    # competing action at each decision state, drawn below
    for a, b, lab in [(0, 0, "wait"), (1, 0, "back"), (2, 1, "slip"), (3, 3, "wait")]:
        if a == b:
            ax.add_patch(FancyArrowPatch((a - 0.22, -0.22), (a + 0.22, -0.22),
                                         arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                         color=GREY, connectionstyle="arc3,rad=1.9", zorder=2))
            ax.text(a, -0.92, lab, ha="center", fontsize=8, color=GREY)
        else:
            ax.add_patch(FancyArrowPatch((a - 0.2, -0.22), (b + 0.2, -0.22),
                                         arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                         color=GREY, connectionstyle="arc3,rad=0.4", zorder=2))
            ax.text((a + b) / 2, -0.82, lab, ha="center", fontsize=8, color=GREY)
    ax.annotate("FB inverts\nthis decision", xy=(3.0, 0.34), xytext=(3.15, 1.15),
                fontsize=9, color=BAD, ha="center",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.3))
    for (x0, x1, lab) in [(-0.35, 1.35, "approach"), (1.6, 2.4, "acquire control"),
                          (2.6, 4.35, "transport")]:
        ax.plot([x0, x1], [-1.25, -1.25], color="black", lw=0.9)
        ax.text((x0 + x1) / 2, -1.5, lab, ha="center", fontsize=9)
    ax.plot([], [], "o", mfc="white", mec=FB, mew=2, ms=9, label="gap preserved")
    ax.plot([], [], "o", mfc="white", mec=BAD, mew=2, ms=9, label="gap inverted")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, handletextpad=0.4,
              title="unrestricted FB", title_fontsize=8.5)
    ax.set_xlim(-0.65, 4.65); ax.set_ylim(-1.85, 1.6); ax.axis("off")
    ax.set_title("A  five-state bottleneck", fontsize=10.5, loc="left")


def panel_rows(ax):
    P = successors()
    idx = [(2, 0, "$b$, cross"), (2, 1, "$b$, slip"), (3, 0, "$R_0$, place"), (3, 1, "$R_0$, wait")]
    M = np.stack([P[2 * s + a] for s, a, _ in idx])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=0.65, aspect="auto")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.38 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(SLAB, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([l for _, _, l in idx], fontsize=9)
    ax.set_xlabel("discounted occupancy of each state", fontsize=9)
    ax.set_xlim(-0.5, 7.6)
    ax.axhline(1.5, color="black", lw=1.2)
    for y0, y1, lab, col in [
            (-0.5, 1.5, f"at the crossing the two\nfutures are far apart\n$\\|\\Delta\\|$ = {np.linalg.norm(M[0]-M[1]):.2f}", "black"),
            (1.5, 3.5, f"at transport the two\nfutures are a near-copy\n$\\|\\Delta\\|$ = {np.linalg.norm(M[2]-M[3]):.2f}", BAD)]:
        ax.plot([4.62, 4.78, 4.78, 4.62], [y0 + 0.06, y0 + 0.06, y1 - 0.06, y1 - 0.06],
                color=col, lw=1.0, clip_on=False)
        ax.text(4.95, (y0 + y1) / 2, lab, fontsize=8.5, color=col, va="center", ha="left")
    ax.set_title("B  what the critic must tell apart", fontsize=10.5, loc="left")


def panel_gaps(ax, runs):
    x = np.arange(len(ROUTE))
    ax.axhspan(-0.6, 0.0, color=BAD, alpha=0.10, lw=0)
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.axhline(0.0, color="black", lw=0.8)
    for lab, run, method, col, ls, mk in [
            ("Density-FB", runs["nosphere"], "density", DENS, "-", "o"),
            ("FB", runs["nosphere"], "fb", FB, "-", "o"),
            ("FB (spherical $B$)", runs["sphere"], "fb", FB, "--", "s")]:
        m, lo, hi = ratios(run, method)
        ax.plot(x, m, ls, marker=mk, color=col, lw=2, ms=5, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
    ax.text(0.04, 0.045, "wrong action", transform=ax.transAxes, fontsize=8.5, color=BAD)
    ax.set_xticks(x); ax.set_xticklabels(ROUTE_LAB, fontsize=9)
    ax.set_ylabel("learned gap $\\div$ exact gap", fontsize=9.5)
    ax.set_ylim(-0.6, 1.8); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("C  action gap along the route  ($\\gamma$ = 0.8, rank 2, 2 seeds $\\times$ 2 rates)",
                 fontsize=10.5, loc="left")


def panel_funnel(ax):
    x = np.arange(4)
    for m, o in OUTCOMES.items():
        reached = 100 - o["approach"]; acq = reached - o["grasp"]; mai = acq - o["maintain"]
        ax.plot(x, [reached, acq, mai, o["success"]], "o-", color=MC[m], lw=2.2 if m == "FB" else 1.6,
                ms=4, label=m, alpha=1.0 if m == "FB" else 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["approach", "acquire\ncontrol", "maintain\ncontrol", "transport"], fontsize=9)
    ax.set_ylabel("% of episodes surviving", fontsize=9.5)
    ax.set_ylim(0, 102); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8.5)
    ax.set_title("D  cube-single, same phases (1M, flow actor)", fontsize=10.5, loc="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/deterministic_didactic"))
    ap.add_argument("--out", type=Path,
                    default=Path("PAPER/ICLR/figures/didactic/bottleneck5_cube.png"))
    a = ap.parse_args()
    runs = {"sphere": a.root / "bottleneck5_td_test",
            "nosphere": a.root / "bottleneck5_td_test_nosphere"}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0),
                             gridspec_kw={"height_ratios": [1.0, 1.12]})
    panel_graph(axes[0, 0], ratios(runs["nosphere"], "fb")[0])
    panel_rows(axes[0, 1])
    panel_gaps(axes[1, 0], runs)
    panel_funnel(axes[1, 1])
    fig.tight_layout(w_pad=2.5, h_pad=2.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
