"""Section 6.1, Figure 3(b): the lemma's core claim on the same trained critics as panel (a).
The difference between the two futures at the bottleneck, Delta m, is split into the part inside
FB's trained feature span (Pi_B Delta m) and the residual outside it; each part's contribution to
the action-value difference <., r> is shown next to what FB reads and what it reads once the
residual direction is appended to its features (exact by the lemma). Eleven-state process, rank 3,
10 TD runs (values from sb_lemma.csv: lemma = inside share, discarded = outside share, read).
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/residual_panel.py [rank]
"""
import csv, pathlib, sys, statistics as st, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation")); import run as single  # noqa
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"; INK = single.INK
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
G = 5.0 / 3.0                                             # exact action-value difference, P11
rows = [r for r in csv.DictReader(open(HERE / "sb_lemma.csv")) if r["mdp"] == "0" and int(r["d"]) == RANK and r["kind"] == "trained"]
def ms(key):
    v = [float(r[key]) * G for r in rows]; m = st.mean(v); return m, (2.26 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
inside, outside, read = ms("lemma"), ms("discarded"), ms("read")
bars = [("exact", G, 0.0, "#7a7a7a", None),
        ("part inside\nFB's span", inside[0], inside[1], single.ARM_COLOR["A"], None),
        ("residual\noutside", outside[0], outside[1], single.ARM_COLOR["A"], "//"),
        ("FB reads", read[0], read[1], INK, None),
        ("FB reads,\nresidual added", G, 0.0, "#029E73", None)]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(3.6, 2.3))
for i, (lab, v, e, col, hatch) in enumerate(bars):
    ax.bar(i, v, 0.62, color=col, alpha=0.45 if hatch else 1.0, hatch=hatch, edgecolor=col, zorder=2)
    if e > 0:
        ax.errorbar(i, v, yerr=e, ecolor=INK, elinewidth=0.7, capsize=1.5, linestyle="none", zorder=3)
    ax.text(i, v + 0.05 + e, f"{abs(v):.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=6.8)
ax.set_ylabel("action-value difference $\\langle\\cdot,r\\rangle_\\rho$", fontsize=8); ax.set_ylim(0, 2.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_residual_rank{RANK}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / f"fig_sb_residual_rank{RANK}.pdf")
print(f"rank {RANK}: inside {inside[0]:.3f}+-{inside[1]:.3f}  outside {outside[0]:.3f}+-{outside[1]:.3f}  read {read[0]:.3f}+-{read[1]:.3f}  (sum inside+outside = {inside[0]+outside[0]:.3f})")
