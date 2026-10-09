"""The lemma check as curves against rank (replaces the scatter in the main text).

For every rank shared by the six processes (2..6): the action-value difference the trained
backward features ALLOW (the lemma's projection <Pi_B dm, r>, computed from B alone), what the
trained critic READS with its own forward map, FB's closed-form optimum, and the closed-form fit
with the discarded direction appended (exactly 1). Means and 95% t-intervals over processes x runs.

    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/lemma_curves.py
"""
import csv, pathlib, sys, statistics as st, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
import run as single  # noqa: E402
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"
INK = single.INK

lem = list(csv.DictReader(open(HERE / "sb_lemma.csv")))
con = list(csv.DictReader(open(HERE / "sb_construct.csv")))
ranks = [2, 3, 4, 5, 6]

def tci(vals):
    m = st.mean(vals)
    if len(vals) < 2: return m, m, m
    se = st.stdev(vals) / math.sqrt(len(vals)); t = 2.0 if len(vals) > 30 else 2.26
    return m, m - t * se, m + t * se

def series(key, kind):
    out = []
    for d in ranks:
        v = [float(r[key]) for r in lem if r["kind"] == kind and int(r["d"]) == d]
        out.append(tci(v))
    return np.array(out)

allow = series("lemma", "trained"); read = series("read", "trained")
closed = [st.mean([float(r["gap_rank_d"]) for r in con if int(r["d"]) == d]) for d in ranks]
aug = [st.mean([float(r["gap_aug"]) for r in con if int(r["d"]) == d]) for d in ranks]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axhline(1.0, color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, label="exact", zorder=1)
ax.plot(ranks, aug, color="#029E73", marker="s", markersize=3.8, linewidth=1.3, zorder=3,
        label="features + the discarded direction")
ax.plot(ranks, closed, color="#9a9a9a", marker="o", markersize=4.5, markerfacecolor="none", linewidth=1.0,
        linestyle=":", zorder=2, label="closed-form optimum")
ax.fill_between(ranks, allow[:, 1], allow[:, 2], color=single.ARM_COLOR["A"], alpha=0.15, linewidth=0)
ax.plot(ranks, allow[:, 0], color=single.ARM_COLOR["A"], marker="o", markersize=4.2, linewidth=1.6, zorder=4,
        label="allowed by the trained features (lemma)")
ax.plot(ranks, read[:, 0], color=INK, marker="x", markersize=4.5, linewidth=0, markeredgewidth=1.2, zorder=5,
        label="read by the trained critic")
ax.set_xticks(ranks); ax.set_xlim(1.75, 6.25); ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("representation rank"); ax.set_ylabel("$\\widehat{G}_r\\,/\\,G_r$", fontsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.3, frameon=False, loc="lower right", handlelength=1.8, labelspacing=0.35)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_lemma_curves.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / "fig_sb_lemma_curves.pdf")
for d, a, r, c, g in zip(ranks, allow[:, 0], read[:, 0], closed, aug):
    print(f"rank {d}: allowed {a:.3f}  read {r:.3f}  closed {c:.3f}  augmented {g:.3f}")
