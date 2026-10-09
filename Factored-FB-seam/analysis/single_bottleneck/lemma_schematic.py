"""Section 6.1, Figure 3(b): the lemma as a picture. The feature span is a plane; each true future is
a point; its least-squares fit is the foot of the perpendicular. When the difference between the two
futures is perpendicular to the plane, the two feet coincide and the critic reads no difference.
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/lemma_schematic.py
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation")); import run as single  # noqa
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"; INK = single.INK
ORANGE, GREEN, BLUE, GREY = single.ARM_COLOR["A"], "#029E73", "#0173B2", "#8a8a8a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK})
fig, ax = plt.subplots(figsize=(3.3, 2.3))
# the plane (drawn in oblique projection): x to the right, depth up-right, height up
def P(x, y, z):            # oblique projection of a 3-d point onto the page
    return np.array([x + 0.45 * y, z + 0.28 * y])
plane = [P(0, 0, 0), P(3.2, 0, 0), P(3.2, 2.0, 0), P(0, 2.0, 0)]
ax.add_patch(Polygon(plane, closed=True, facecolor="#f0e6d2", edgecolor=ORANGE, linewidth=1.0, zorder=1))
ax.text(*P(0.12, 0.35, 0), "span of FB's features", color=ORANGE, fontsize=7, ha="left", va="center")
# the two futures: same foot, different heights
foot = (1.55, 1.05); h_plus, h_minus = 1.35, 0.55
f = P(*foot, 0); mp = P(*foot, h_plus); mm = P(*foot, h_minus)
ax.plot([f[0], mp[0]], [f[1], mp[1]], color=GREY, linestyle=(0, (2, 2)), linewidth=0.9, zorder=2)
ax.scatter(*f, s=40, facecolor="white", edgecolor=ORANGE, linewidth=1.4, zorder=4)
ax.scatter(*mp, s=34, color=GREEN, zorder=4); ax.scatter(*mm, s=34, color=BLUE, zorder=4)
ax.text(mp[0] + 0.12, mp[1], "$m_+$  true future after $a_+$", color=GREEN, fontsize=7.5, va="center")
ax.text(mm[0] + 0.12, mm[1], "$m_-$  true future after $a_-$", color=BLUE, fontsize=7.5, va="center")
ax.text(f[0] + 0.12, f[1] - 0.08, "$\\widehat m_+=\\widehat m_-$  what FB fits", color=ORANGE, fontsize=7.5, va="top")
# Delta m as an arrow beside the perpendicular
ax.add_patch(FancyArrowPatch((mm[0] - 0.22, mm[1]), (mp[0] - 0.22, mp[1]), arrowstyle="-|>", mutation_scale=8,
                             linewidth=1.3, color=INK, zorder=3))
ax.text(mp[0] - 0.30, (mp[1] + mm[1]) / 2, "$\\Delta m$", color=INK, fontsize=8, ha="right", va="center")
ax.text(mp[0] - 0.30, (mp[1] + mm[1]) / 2 - 0.22, "outside\nthe span", color="#555555", fontsize=6.5, ha="right", va="center")
# right-angle mark at the foot
ra = 0.09; ax.plot([f[0] - ra, f[0] - ra, f[0]], [f[1], f[1] + ra, f[1] + ra], color=GREY, linewidth=0.7, zorder=3)
ax.set_xlim(-0.15, 4.4); ax.set_ylim(-0.15, 2.05); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_lemma_schematic.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / "fig_sb_lemma_schematic.pdf")
