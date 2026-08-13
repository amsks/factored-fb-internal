"""Figure 5_rewards.png: what each zero-shot reward family looks like on the cube
workspace (cube xy plane; the ball radius eps=0.4 obs units = 4 cm, gamma-free).

Five families are exact functions of cube position -> heatmaps. LINEAR is a function
of the learned embedding psi(x): we color real dataset states by r = psi(x)^T z~ for
a random unit direction z~, using the trained phase-1 critic psi -- the honest
picture of what a random feature-linear reward looks like in cube space.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, "/home/adityamohan/git/Factored-FB")
sys.path.insert(0, "/home/adityamohan/git/Factored-FB/third_party/ogbench/impls")

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", BLUE])
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#b9b8b4", BLUE])
EPS = 0.4
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 8.5,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})

# workspace grid (bounds from the dataset cube positions, roughly [-1.5,1.5]x[-3,2.5])
gx, gy = np.meshgrid(np.linspace(-1.8, 1.8, 220), np.linspace(-3.0, 2.6, 220))
C1, C2 = np.array([-0.7, 1.2]), np.array([0.9, -1.4])   # illustrative goals
A, B = np.array([0.9, 1.2]), np.array([0.1, -0.1])      # reach / avoid


def dist(c):
    return np.sqrt((gx - c[0]) ** 2 + (gy - c[1]) ** 2)


fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6))

panels = [
    ("dirac:  $r=\\mathbf{1}[\\|c-c^*\\|\\leq\\epsilon]$",
     (dist(C1) <= EPS).astype(float), SEQ, None),
    ("mixture:  $r=\\max_k \\mathbf{1}[\\|c-c_k\\|\\leq\\epsilon]$",
     np.maximum(dist(C1) <= EPS, dist(C2) <= EPS).astype(float), SEQ, None),
    ("region:  $r=\\mathbf{1}[c\\in \\mathrm{box}]$",
     ((np.abs(gx - C2[0]) <= 0.6) & (np.abs(gy - C2[1]) <= 0.6)).astype(float), SEQ, None),
    ("dense:  $r=\\exp(-\\|c-c^*\\|/\\sigma)$",
     np.exp(-dist(C1) / 0.8), SEQ, None),
    ("reach-avoid:  $r=\\mathbf{1}[c\\in A]-\\mathbf{1}[c\\in B]$",
     (dist(A) <= EPS).astype(float) - (dist(B) <= EPS).astype(float), DIV,
     TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=1)),
]
for ax, (title, Z, cmap, norm) in zip(axes.flat, panels):
    ax.pcolormesh(gx, gy, Z, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
for c, lbl, ax_ids in ((C1, "$c^*$", (0, 1, 3)), (C2, "$c_2$", (1,)),
                       (A, "$A$", (4,)), (B, "$B$", (4,))):
    for i in ax_ids:
        axes.flat[i].annotate(lbl, xy=c, xytext=(4, 4),
                              textcoords="offset points", fontsize=8, color=INK)

# --- linear panel: real psi from the phase-1 checkpoint, random direction z~ ---
ax = axes.flat[5]
try:
    from scripts.rescuing_fb.t6_gpi_eval import load_agent_flexible, make_psi
    import ogbench
    run = Path("/home/adityamohan/exp_spec06/factored-fb/"
               "GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s0")
    agent, d, cfg = load_agent_flexible(run, 500000, arch_explicit=False, d=0,
                                        actor_cond="z", actor_vf_cond="none")
    ds = ogbench.make_env_and_datasets("cube-single-play-v0", dataset_only=True)
    if isinstance(ds, tuple):
        ds = ds[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds["observations"]), 4000, replace=False)
    S = np.asarray(ds["observations"], np.float32)[idx]
    act_dim = int(np.asarray(ds["actions"]).shape[-1])
    psi = make_psi(agent, src="critic", act_dim=act_dim, fbz=False)
    P = psi(S)
    z = rng.normal(size=P.shape[-1]); z /= np.linalg.norm(z)
    r = P @ z / np.sqrt(P.shape[-1])
    cube = S[:, 19:21]
    sc = ax.scatter(cube[:, 0], cube[:, 1], c=r, cmap=DIV,
                    norm=TwoSlopeNorm(vcenter=float(np.median(r))), s=3, rasterized=True)
    ax.set_title("linear:  $r=\\psi(x)^{\\top}\\tilde z/\\sqrt{d}$  (real $\\psi$, random $\\tilde z$)",
                 loc="left", fontsize=8.5)
except Exception as e:  # keep the figure buildable without the checkpoint
    ax.text(0.5, 0.5, f"linear panel unavailable\n{e}", ha="center", va="center",
            transform=ax.transAxes, fontsize=7)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Zero-shot reward families on the cube workspace (cube position plane; "
             "$\\epsilon$-balls drawn to scale)", x=0.02, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(Path(__file__).parent / "5_rewards.png", dpi=180)
print("5_rewards.png written")
