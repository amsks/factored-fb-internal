"""Appendix figure fig:fb-features on the nine-state pass-through process (2026-09-25).

(a) FB's learned backward features B(s) at rank 2, one run: the rewarded state A (3) and the unrewarded
    state B (4) get IDENTICAL features (to machine precision, every run), so the average feature vector
    of the future after a+ equals that after a- (Lemma 1's condition).
(the predicted-futures panel was dropped 2026-09-25: it duplicated Fig. 3b of the paper)
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/fb_failure_visual9.py [run]
"""
import importlib.util, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("family9", HERE / "family9.py"); f9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f9)
spec = importlib.util.spec_from_file_location("paper_figs_61", HERE / "paper_figs_61.py"); pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RANK, GAMMA = 2, 0.8
FIG = REPO / "PAPER" / "ICLR" / "figures" / "nine_state"
INK = pf.INK
COL = {"entry": "#9a9a9a", "bottleneck": "#DE8F05", "A": "#029E73", "B": "#0173B2", "tail": "#6f6f6f"}

sb, dsb = f9.load_sb()
m = f9.make(sb, dsb, "P9", np.full((9, 2), 0.5))
n, b, r, T = m["n"], m["decision"], m["r"], m["t_exact"]
group = {0: "entry", 1: "entry", 2: "bottleneck", 3: "A", 4: "B", 5: "tail", 6: "tail", 7: "tail", 8: "tail"}
cfg, pred_fb, _, B_fb = sb.fam.train_group(m, RANK, "fb")
_, pred_d, _, _ = sb.fam.train_group(m, RANK, "density")
P_fb = np.stack([sb.bf.readout(p, "fb", GAMMA, n) for p in pred_fb]); P_d = np.asarray(pred_d, float)
B = np.asarray(B_fb[RUN], float)
assert all(np.allclose(np.asarray(Bi)[3], np.asarray(Bi)[4], atol=1e-8) for Bi in B_fb), "B(A) != B(B) in some run"
mp, mm = T[2 * b] @ B, T[2 * b + 1] @ B
q = lambda P: (P[2 * b] @ r / (1 - GAMMA), P[2 * b + 1] @ r / (1 - GAMMA))
print(f"run {RUN} {cfg[RUN]}: |E+[B]-E-[B]| = {np.linalg.norm(mp - mm):.1e}")

pf.style()
fig, ax = plt.subplots(figsize=(3.3, 2.5))          # single panel (the predicted-futures panel duplicated Fig. 3b)
# (a) feature plane
for s in range(n):
    if s in (3, 4):
        continue
    ax.plot(*B[s], "o", ms=5.5, color=COL[group[s]], mec="white", mew=0.5, zorder=7 if s == b else 4)
ax.plot(*B[3], "o", ms=8, fillstyle="left", color=COL["A"], markerfacecoloralt=COL["B"], mec=INK, mew=0.6, zorder=5)
ax.plot(*mp, "*", ms=11, color="white", mec=INK, mew=0.7, zorder=6)
span = B.max(0) - B.min(0)
off = {s: (5, 3) for s in range(n)}
off.update({2: (-11, 3), 3: (6, -10), 6: (5, -9), 8: (5, -9)})
for s in range(n):
    if s == 4:
        continue
    ax.annotate("3 = 4" if s == 3 else str(s), B[s], xytext=off[s], textcoords="offset points", fontsize=7, color=INK)
ax.annotate("average feature of the future,\nsame after $a_+$ and $a_-$", mp,
            xytext=(B[:, 0].min() + 0.22 * span[0], B[:, 1].min() + 0.62 * span[1]), textcoords="data",
            fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_xlim(B[:, 0].min() - 0.12 * span[0], B[:, 0].max() + 0.2 * span[0])
ax.set_ylim(B[:, 1].min() - 0.12 * span[1], B[:, 1].max() + 0.25 * span[1])
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=c, ms=4.5, label=l) for l, c in
                   [("entry", COL["entry"]), ("bottleneck", COL["bottleneck"]), ("A (reward 1)", COL["A"]),
                    ("B (reward 0)", COL["B"]), ("tail", COL["tail"])]],
          fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.6,
          handletextpad=0.2)
fig.tight_layout(pad=0.3)
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"fig_fb_features.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
print("wrote", FIG / "fig_fb_features.pdf")
