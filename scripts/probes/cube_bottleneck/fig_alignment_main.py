"""Per-phase alignment figure: cosine between predicted and rollout action-contrast of future cube positions.

Reads figures/gap_preservation/alignment_by_phase_*.csv from separation_preservation.py.
"""

import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[3]
D = REPO / "figures" / "gap_preservation"
INK = "#2a2a2a"
PHASES = [
    ("approach", "approach", "#CC78BC"),
    ("inhand", "maintain", "#0173B2"),
    ("carrying", "transport", "#D55E00"),
]
STYLE = {
    "fb_tdj": dict(color="#8a8a8a", marker="o", label="FB"),
    "density_fb_z": dict(color="#029E73", marker="^", label="Density-FB"),
}
pooled = pd.read_csv(D / "alignment_by_phase_cell5cm.csv")
seeds = pd.read_csv(D / "alignment_by_phase_per_seed.csv")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8.5,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)
fig, ax = plt.subplots(figsize=(3.2, 2.3))
ax.axhline(0, color="#9a9a9a", linestyle="--", linewidth=0.8, zorder=1)
for i, (bank, label, col) in enumerate(PHASES):
    ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.07, zorder=0)
for j, (critic, st) in enumerate(STYLE.items()):
    dx = -0.14 if j == 0 else 0.14
    for i, (bank, label, col) in enumerate(PHASES):
        row = pooled[(pooled.critic == critic) & (pooled.bank == bank)].iloc[0]
        ps = seeds[(seeds.critic == critic) & (seeds.bank == bank)]
        ax.scatter(
            [i + dx] * len(ps),
            ps.align_mean,
            s=9,
            color=st["color"],
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
        ax.errorbar(
            i + dx,
            row.align_mean,
            yerr=row.align_se,
            fmt=st["marker"],
            color=st["color"],
            markersize=6.5,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
            label=st["label"] if i == 0 else None,
        )
ax.set_xticks(range(len(PHASES)))
ax.set_xticklabels([p[1] for p in PHASES])
for tick, (_, _, col) in zip(ax.get_xticklabels(), PHASES):
    tick.set_color(col)
ax.set_xlim(-0.5, len(PHASES) - 0.5)
ax.set_ylim(-0.1, 0.3)
ax.set_ylabel("alignment (cosine)", fontsize=8.5)
ax.set_xlabel("phase of the restored state", fontsize=8.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.0)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(D / f"fig_alignment_by_phase_main.{ext}", dpi=300)
print("wrote", D / "fig_alignment_by_phase_main.pdf")
for critic in STYLE:
    print(
        critic,
        {
            p[1]: round(
                float(
                    pooled[
                        (pooled.critic == critic) & (pooled.bank == p[0])
                    ].align_mean.iloc[0]
                ),
                3,
            )
            for p in PHASES
        },
    )
