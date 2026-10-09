"""Section 6.1, first figure: at the same low rank, FB values the two actions at the bottleneck the
same and Density-FB does not. Eleven-state process, TD-trained critics at rank 3 (10 runs), the
value each critic assigns to a+ and a- at the bottleneck next to the exact values.
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/reward_through_features.py [rank]
"""
import csv, pathlib, sys, statistics as st, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation")); import run as single  # noqa
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"; INK = single.INK
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MDP = "0"                                   # P11
EXACT = (5.0 / 3.0, 0.0)                    # exact action values at the bottleneck (cluster-A reward)
rows = [r for r in csv.DictReader(open(HERE / "sb_trained.csv")) if r["mdp"] == MDP and int(r["d"]) == RANK]
def stats(arm, key):
    v = [float(r[key]) for r in rows if r["arm"] == arm]; m = st.mean(v)
    se = st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, 2.26 * se
groups = [("exact", EXACT, (0.0, 0.0), "#7a7a7a"),
          ("FB", (stats("A", "q0")[0], stats("A", "q1")[0]), (stats("A", "q0")[1], stats("A", "q1")[1]), single.ARM_COLOR["A"]),
          ("Density-FB", (stats("D", "q0")[0], stats("D", "q1")[0]), (stats("D", "q0")[1], stats("D", "q1")[1]), single.ARM_COLOR["D"])]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(3.0, 2.3)); w = 0.34
for i, (name, (qp, qm), (ep, em), col) in enumerate(groups):
    ax.bar(i - w / 2, qp, w, color=col, alpha=1.0, edgecolor=col, zorder=2, label="$a_+$ (to cluster A)" if i == 0 else None)
    ax.bar(i + w / 2, qm, w, color=col, alpha=0.4, edgecolor=col, hatch="//", zorder=2, label="$a_-$ (to cluster B)" if i == 0 else None)
    for x, q, e in ((i - w / 2, qp, ep), (i + w / 2, qm, em)):
        if e > 0:
            ax.errorbar(x, q, yerr=e, ecolor=INK, elinewidth=0.7, capsize=1.5, linestyle="none", zorder=3)
        ax.text(x, q + 0.05 + e, f"{q:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("value at the bottleneck, $Q(s_b, a)$", fontsize=8.5); ax.set_ylim(0, 2.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.6, frameon=False, loc="upper center", handlelength=1.4)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_values_rank{RANK}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / f"fig_sb_values_rank{RANK}.pdf")
for name, q, e, _ in groups: print(f"{name:11s} a+ {q[0]:.2f} +- {e[0]:.2f}   a- {q[1]:.2f} +- {e[1]:.2f}")
