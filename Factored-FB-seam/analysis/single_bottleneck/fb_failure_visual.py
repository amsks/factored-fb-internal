"""Section 6.1 / Lemma 1 illustration: why FB cannot tell the two bottleneck actions apart.

Eleven-state process (P11), TD-trained critics at rank 2 (10 runs each), so the backward
features B(s) live in a plane and can be drawn.
(a) FB's learned features B(s), one point per state, for one run. Rewarded states 3, 4 get the
    same features as unrewarded 5, 6 (every run). The star are the average
    feature vector of each action's TRUE future, sum_s M(b, a, s) B(s) (weights = normalized
    discounted visits from the bottleneck). They coincide: by Lemma 1, the least-squares fit with
    these features then gives both actions the same predicted future, hence the same value.
(b) Predicted distribution over future states from the bottleneck, for a+ and a-, averaged over
    the 10 runs: exact, FB (identical for the two actions), Density-FB (different).
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/fb_failure_visual.py [run]
"""
import importlib.util, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
sys.argv, _argv = [sys.argv[0]], sys.argv
spec.loader.exec_module(sb); sys.argv = _argv
fam, bf, INK = sb.fam, sb.bf, sb.single.INK
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"
RANK, GAMMA = 2, 0.8
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ORANGE, GREEN = sb.single.ARM_COLOR["A"], sb.single.ARM_COLOR["D"]
GROUP_COLOR = {"entry": "#9a9a9a", "bottleneck": INK, "A": "#029E73", "B": "#CC78BC", "tail": "#0173B2"}

mdp = sb.fixed_processes()[1]; assert mdp["n"] == 11
n, b, r = mdp["n"], mdp["decision"], mdp["r"]
T = mdp["t_exact"]                                  # rows: normalized discounted visits, sum to 1
group = {}
for s in range(n):
    group[s] = ("bottleneck" if s == b else "A" if s in (3, 4) else "B" if s in (5, 6)
                else "tail" if s >= 7 else "entry")

cfg, pred_fb, F_fb, B_fb = fam.train_group(mdp, RANK, "fb")
cfg_d, pred_d, _, _ = fam.train_group(mdp, RANK, "density")
P_fb = np.stack([bf.readout(p, "fb", GAMMA, n) for p in pred_fb])      # (runs, 2n, n)
P_d = np.asarray(pred_d, np.float64)

B = np.asarray(B_fb[RUN], np.float64)
mean_plus, mean_minus = T[2 * b] @ B, T[2 * b + 1] @ B
q = lambda P: (P[2 * b] @ r / (1 - GAMMA), P[2 * b + 1] @ r / (1 - GAMMA))
print(f"run {RUN} {cfg[RUN]}: feature averages a+ {mean_plus.round(4)} a- {mean_minus.round(4)} "
      f"|diff| {np.linalg.norm(mean_plus - mean_minus):.1e}")
print("FB values per run", [tuple(np.round(q(P), 2)) for P in P_fb])
print("Density-FB values per run", [tuple(np.round(q(P), 2)) for P in P_d])

spec_s = importlib.util.spec_from_file_location("paper_figs_61", HERE / "paper_figs_61.py")
_pf = importlib.util.module_from_spec(spec_s); spec_s.loader.exec_module(_pf); _pf.style()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw=dict(width_ratios=[1, 1.45]))

# (a) feature plane, zoomed on the states near the bottleneck (0 and 7 sit far out and are marked at
# the edge). FB puts each rewarded state on top of an unrewarded one (B(3)=B(5), B(4)=B(6)); each such
# point is drawn half green, half pink.
pairs = [(3, 5), (4, 6)]
for a_s, b_s in pairs:
    assert np.allclose(B[a_s], B[b_s], atol=1e-8)
near = [s for s in range(n) if s not in (0, 7)]
pts = np.vstack([B[near], mean_plus])
lo, hi = pts.min(0), pts.max(0); pad = 0.18 * (hi - lo)
ax.set_xlim(lo[0] - pad[0], hi[0] + 2.2 * pad[0]); ax.set_ylim(lo[1] - 3.0 * pad[1], hi[1] + pad[1])
for s in near:
    if s in (3, 4, 5, 6):
        continue
    ax.plot(*B[s], "o", ms=6, color=GROUP_COLOR[group[s]], mec="white", mew=0.6, zorder=6)
for a_s, _ in pairs:
    ax.plot(*B[a_s], "o", ms=9, fillstyle="left", color=GROUP_COLOR["A"], markerfacecoloralt=GROUP_COLOR["B"],
            mec=INK, mew=0.6, zorder=4)
ax.plot(*mean_plus, "*", ms=12, color="white", mec=INK, mew=0.7, zorder=5)
lab = {2: ("2", (6, 6)), 3: ("3 = 5", (-30, -4)), 4: ("4 = 6", (8, -9)), 1: ("1", (6, 3)),
       8: ("8 = 10", (8, -2)), 9: ("9", (6, 3))}
for s, (t, off) in lab.items():
    ax.annotate(t, B[s], xytext=off, textcoords="offset points", fontsize=7, color=INK)
ax.annotate("average feature of the future:\nthe same after $a_+$ and after $a_-$", mean_plus, xytext=(10, 0),
            textcoords="offset points", fontsize=6.8, va="center")
ax.text(0.02, 0.02, f"state 7 at ({B[7][0]:.1f}, {B[7][1]:.1f}), outside view", transform=ax.transAxes,
        fontsize=6.3, color=GROUP_COLOR["tail"], ha="left", va="bottom")
ax.text(0.02, 0.09, f"state 0 at ({B[0][0]:.1f}, {B[0][1]:.1f}), outside view", transform=ax.transAxes,
        fontsize=6.3, color=GROUP_COLOR["entry"], ha="left", va="bottom")
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_title("(a) FB's learned features $B(s)$", fontsize=8.5, loc="left")
handles = [Line2D([0], [0], marker="o", ls="", color=c, markersize=5, label=l) for l, c in
           [("entry 0, 1", GROUP_COLOR["entry"]), ("bottleneck 2", GROUP_COLOR["bottleneck"]),
            ("cluster A (reward 1)", GROUP_COLOR["A"]), ("cluster B (reward 0)", GROUP_COLOR["B"]),
            ("tail 7-10", GROUP_COLOR["tail"])]]
ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower center", handletextpad=0.2,
          bbox_to_anchor=(0.5, 1.08), ncol=3, columnspacing=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# (b) predicted futures from the bottleneck
x = np.arange(n); w = 0.13
series = [("exact", T, "#7a7a7a"), ("FB", P_fb.mean(0), ORANGE), ("Density-FB", P_d.mean(0), GREEN)]
for k, (name, P, col) in enumerate(series):
    off = (k - 1) * 2 * w
    bx.bar(x + off - w / 2, P[2 * b], w, color=col, zorder=2)
    bx.bar(x + off + w / 2, P[2 * b + 1], w, color=col, alpha=0.35, hatch="////", edgecolor=col, linewidth=0, zorder=2)
bx.axvspan(2.5, 4.5, color=GREEN, alpha=0.10, zorder=0)
bx.text(3.5, bx.get_ylim()[1] * 0.97, "reward 1", color=GREEN, fontsize=7, ha="center", va="top")
bx.set_xticks(x); bx.set_xticklabels([str(k) for k in range(n)], fontsize=7)
bx.set_xlabel("future state $s^+$"); bx.set_ylabel("predicted share of future visits")
bx.text(0.99, 0.97, "solid: $a_+$\nhatched: $a_-$", transform=bx.transAxes, ha="right", va="top", fontsize=7)
bx.set_title("(b) futures predicted from state 2", fontsize=8.5, loc="left")
leg = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
qs = [q(T)] + [tuple(np.mean([q(P) for P in Ps], 0)) for Ps in (P_fb, P_d)]
bx.legend(leg, [f"{nm}: $Q(a_+)$={a:.2f}, $Q(a_-)$={m:.2f}" for (nm, _, _), (a, m) in zip(series, qs)],
          fontsize=6.3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=1)
for sp in ("top", "right"): bx.spines[sp].set_visible(False)

fig.tight_layout(pad=0.4, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_fb_failure_visual.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / "fig_sb_fb_failure_visual.pdf")
