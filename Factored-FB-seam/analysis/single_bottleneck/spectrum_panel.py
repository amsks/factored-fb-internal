"""Section 6.1, Figure 3(b): why the lemma bites. Eleven-state process: the singular directions of
the successor matrix nM ordered by the mass they explain (what FB's squared loss keeps first) against
the share of the bottleneck row difference Delta m each direction carries. FB with 3 features keeps
directions 1-3 (shaded); Delta m sits on direction 4.
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/spectrum_panel.py [rank] [n_dirs]
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation")); import run as single  # noqa
sys.path.insert(0, str(REPO / "analysis" / "lemma_check")); import design_single_bottleneck as ds  # noqa
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"; INK = single.INK
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3; NDIR = int(sys.argv[2]) if len(sys.argv) > 2 else 6
edges, b, A, B, T = ds.build(entry=2, cluster=2, tail=4, merge=True)
X, _ = ds.exact_ratio(edges); n = len(edges)
U, s_, Vt = np.linalg.svd(X, full_matrices=False)
dv = X[2 * b] - X[2 * b + 1]
mass = s_ ** 2 / np.sum(s_ ** 2); share = (Vt @ dv) ** 2 / (dv @ dv)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(3.3, 2.3)); k = np.arange(1, NDIR + 1); w = 0.38
ax.axvspan(0.5, RANK + 0.5, color="#dddddd", alpha=0.5, zorder=0)
ax.bar(k - w / 2, mass[:NDIR], w, color="#8a8a8a", label="of the successor matrix", zorder=2)
ax.bar(k + w / 2, share[:NDIR], w, color=single.ARM_COLOR["A"], label="of the difference $\\Delta m$", zorder=2)
ax.text(RANK / 2 + 0.5, 0.93, f"kept by FB\nwith {RANK} features", ha="center", va="top", fontsize=7, color="#555555")
ax.set_xticks(k); ax.set_xlabel("direction of the successor matrix, by mass", fontsize=8)
ax.set_ylabel("share", fontsize=8.5); ax.set_ylim(0, 1.0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(title="share", title_fontsize=6.8, fontsize=6.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55), handlelength=1.2)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_spectrum_rank{RANK}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / f"fig_sb_spectrum_rank{RANK}.pdf")
print("mass ", np.round(mass[:NDIR], 3)); print("share", np.round(share[:NDIR], 3))
