"""reward_families.png: the three held-out reward families of the density-fb note,
drawn on the cube xy workspace (eps = 0.4 obs units = 4 cm, to scale). Adapted from
PAPER/rescuing-fb/figures/make_reward_figure.py; panels match the families the
zero-shot harness evaluates (dirac / dense at task-style centers, region at
hindsight-drawn dataset centers)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", BLUE])
EPS = 0.4
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.0,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})

gx, gy = np.meshgrid(np.linspace(-1.8, 1.8, 240), np.linspace(-3.0, 2.6, 240))
C = np.array([-0.7, 1.2])                                   # task-style center
H = [np.array(p) for p in ([0.9, -1.4], [-1.1, -0.6], [0.4, 0.9], [1.2, 1.6])]  # hindsight centers


def dist(c):
    return np.sqrt((gx - c[0]) ** 2 + (gy - c[1]) ** 2)


fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
panels = [
    ("sparse:  $r=\\mathbf{1}[\\|c-g\\|\\leq\\epsilon]$, task goals",
     (dist(C) <= EPS).astype(float)),
    ("dense:  $r=\\exp(-\\|c-g\\|/2\\epsilon)$, task goals",
     np.exp(-dist(C) / (2 * EPS))),
    ("held-out regions:  $r=\\mathbf{1}[\\|c-c_k\\|\\leq\\epsilon]$, dataset centers",
     np.max([(dist(h) <= EPS) for h in H], axis=0).astype(float)),
]
for ax, (title, Z) in zip(axes.flat, panels):
    ax.pcolormesh(gx, gy, Z, cmap=SEQ, shading="auto", rasterized=True)
    ax.set_title(title, loc="left", fontsize=9.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("reward_families.png", dpi=220, bbox_inches="tight")
print("wrote reward_families.png")
