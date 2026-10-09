"""Section 6.1, Figure 3(b): FB's model of the two futures at the bottleneck. Eleven-state process,
FB's optimum with three features (top-3 singular span of nM; the TD-trained rank-3 features read the
same 0.00). Left: the true futures after a+ and a-, as discounted next-state probabilities. Right: FB's
fitted futures, the projections onto its feature span -- identical, because Delta m has no component
in the span (lemma: m_hat+ - m_hat- = Pi_B Delta m).
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/fitted_rows_panel.py [rank]
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
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
GAMMA = 0.8; DARK, LIGHT = "#3a3a3a", "#b5b5b5"; GREEN, BLUE = "#029E73", "#0173B2"
edges, b, A, B, T = ds.build(entry=2, cluster=2, tail=4, merge=True)
X, t = ds.exact_ratio(edges); n = len(edges)
_, _, Vt = np.linalg.svd(X, full_matrices=False); P = Vt[:RANK].T @ Vt[:RANK]
true = t[[2 * b, 2 * b + 1]]                         # (1-gamma) M rows: discounted next-state probabilities
fit = (X @ P)[[2 * b, 2 * b + 1]] * (1 - GAMMA) / n   # FB's fitted rows on the same scale
rA = np.isin(np.arange(n), A).astype(float)
g_true = (true[0] - true[1]) @ rA / (1 - GAMMA); g_fit = (fit[0] - fit[1]) @ rA / (1 - GAMMA)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.2), sharey=True)
x = np.arange(n); w = 0.38
for ax, rows, title in ((axes[0], true, "true futures"), (axes[1], fit, f"FB's model, {RANK} features")):
    ax.bar(x - w / 2, rows[0], w, color=DARK, label="after $a_+$", zorder=2)
    ax.bar(x + w / 2, rows[1], w, color=LIGHT, label="after $a_-$", zorder=2)
    for s_, col in ((A, GREEN), (B, BLUE)):
        ax.axvspan(s_[0] - 0.5, s_[-1] + 0.5, color=col, alpha=0.10, zorder=0)
    ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in range(n)], fontsize=6)
    ax.set_xlabel("next state $s^+$", fontsize=8)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("discounted probability", fontsize=8)
for ax in axes:
    ax.text(np.mean(A), 0.305, "A", color=GREEN, fontsize=7, ha="center", va="bottom")
    ax.text(np.mean(B), 0.305, "B", color=BLUE, fontsize=7, ha="center", va="bottom")
    ax.set_ylim(0, 0.36)
axes[1].legend(fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88), handlelength=1.2)
fig.tight_layout(pad=0.3, w_pad=0.8)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_fitted_rows_rank{RANK}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / f"fig_sb_fitted_rows_rank{RANK}.pdf")
print(f"true difference {g_true:.3f}, fitted difference {g_fit:.3f}; max |fit+ - fit-| over states {np.abs(fit[0]-fit[1]).max():.2e}")
print("true a+", np.round(true[0], 2)); print("true a-", np.round(true[1], 2)); print("fit  a+", np.round(fit[0], 2)); print("fit  a-", np.round(fit[1], 2))
