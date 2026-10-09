"""Figure 3 recomputed in the same statistical manner as Figure 4: over the six eleven-state processes
(P11 + R1-R5) and the ten TD runs each, rank 3. Every process has the same layout (states 0-10, bottleneck
2, A {3,4}, B {5,6}, tail 7-10), so per-state quantities can be averaged across processes; values are
divided by each process's exact action-value difference G so they can be pooled.
(a) fig_sb_values_rank3: Q(a+)/G and Q(a-)/G for exact, FB, Density-FB; mean and 95% interval over 60 runs.
(b) fig_sb_features_lose_rank3: per state, the true difference in discounted visits (mean over processes)
    and the best least-squares fit with each run's frozen FB features (mean over 60 runs), scaled by 1/G
    per process (so the rewarded bars sum to 1).
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/fig3_family.py
"""
import importlib.util, json, os, pathlib, sys, math, statistics as st
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
sys.argv, _a = [sys.argv[0]], sys.argv; spec.loader.exec_module(sb); sys.argv = _a
fam, INK = sb.fam, sb.single.INK
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3; GAMMA = 0.8; ORANGE, GREEN = sb.single.ARM_COLOR["A"], "#029E73"
mdps = [m for m in sb.rebuild_processes(json.load(open(HERE / "sb_processes.json"))) if m["n"] == 11]
assert len(mdps) == 6, len(mdps)
vals = {"A": [], "D": []}; dm_rows = []; fit_rows = []; g_fit = []; dfb_rows = []
for m in mdps:
    n, b, rho, r, G = m["n"], m["decision"], m["rho"], m["r"], m["gap"]
    X = n * m["m_exact"]; dm = X[2 * b] - X[2 * b + 1]; W = np.sqrt(rho)[:, None]
    dm_rows.append(rho * dm / G)
    for meth, key in (("fb", "A"), ("density", "D")):
        cfg, pred, F_all, B_all = fam.train_group(m, RANK, meth, list(fam.SEEDS), list(fam.RATES), steps=fam.TRAIN_STEPS)
        for i in range(len(cfg)):
            P = sb.bf.readout(pred[i], fam._base_method(meth), GAMMA, n)
            q = P @ r / (1 - GAMMA)
            vals[key].append((q[2 * b] / G, q[2 * b + 1] / G))
            if meth == "density":
                dfb_rows.append((pred[i][2 * b] - pred[i][2 * b + 1]) / (1 - GAMMA) / G)
            if meth == "fb":
                B = np.asarray(B_all[i], np.float64)
                proj = B @ np.linalg.lstsq(W * B, np.sqrt(rho) * dm, rcond=None)[0]
                fit_rows.append(rho * proj / G); g_fit.append(float(np.sum(rho * proj * r)) / G)
    print(f"{m['variant']}: G={G:.3f}  FB q+/G {np.mean([v[0] for v in vals['A'][-10:]]):.2f} q-/G {np.mean([v[1] for v in vals['A'][-10:]]):.2f}  "
          f"DFB {np.mean([v[0] for v in vals['D'][-10:]]):.2f}/{np.mean([v[1] for v in vals['D'][-10:]]):.2f}  best-fit diff/G {np.mean(g_fit[-10:]):.2f}", flush=True)
def tci(v): m_ = st.mean(v); return m_, (2.0 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
# (a) values
groups = [("exact", (1.0, 0.0), (0.0, 0.0), "#7a7a7a")]
for key, name, col in (("A", "FB", ORANGE), ("D", "Density-FB", GREEN)):
    qp = tci([v[0] for v in vals[key]]); qm = tci([v[1] for v in vals[key]])
    groups.append((name, (qp[0], qm[0]), (qp[1], qm[1]), col))
fig, ax = plt.subplots(figsize=(3.2, 2.3)); w = 0.34
for i, (name, (qp, qm), (ep, em), col) in enumerate(groups):
    ax.bar(i - w / 2, qp, w, color=col, edgecolor=col, zorder=2, label="$a_+$ (to cluster A)" if i == 0 else None)
    ax.bar(i + w / 2, qm, w, color=col, alpha=0.4, edgecolor=col, hatch="//", zorder=2, label="$a_-$ (to cluster B)" if i == 0 else None)
    for x, q, e in ((i - w / 2, qp, ep), (i + w / 2, qm, em)):
        if e > 0: ax.errorbar(x, q, yerr=e, ecolor=INK, elinewidth=0.7, capsize=1.5, linestyle="none", zorder=3)
        ax.text(x, q + 0.03 + e, f"{q:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("value / exact", fontsize=8.5); ax.set_ylim(0, 1.32)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.6, frameon=False, loc="upper center", handlelength=1.4)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"): fig.savefig(FIG / f"fig_sb_values_rank{RANK}.{ext}", dpi=600)
plt.close(fig)
# (b) features lose
dm_mean = np.mean(dm_rows, 0); fit_mean = np.mean(fit_rows, 0); fit_se = 2.0 * np.std(fit_rows, 0) / math.sqrt(len(fit_rows))
A = [3, 4]; x = np.arange(11)
fig, ax = plt.subplots(figsize=(3.2, 2.3))
ax.axvspan(A[0] - 0.5, A[-1] + 0.5, color=GREEN, alpha=0.12, zorder=0)
ax.text(np.mean(A), 0.86, "Reward = 1", color=GREEN, fontsize=7.5, ha="center", va="center")
ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
ax.bar(x, dm_mean, 0.6, color=INK, zorder=2, label="true difference")
ax.errorbar(x, fit_mean, yerr=fit_se, fmt="none", ecolor=ORANGE, elinewidth=0.7, capsize=1.5, zorder=3)
ax.scatter(x, fit_mean, s=22, facecolors=ORANGE, edgecolors="white", linewidths=0.6, zorder=4, label="FB, best fit")
ax.set_xticks(x); ax.set_xticklabels([str(k) for k in range(11)], fontsize=7)
ax.set_xlabel("future state $s^+$"); ax.set_ylabel("visits / exact", fontsize=8.5)
ax.set_ylim(-0.85, 0.95)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, handlelength=1.0, columnspacing=1.0)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"): fig.savefig(FIG / f"fig_sb_features_lose_rank{RANK}.{ext}", dpi=600)
dfb_mean = np.mean(dfb_rows, 0); dfb_se = 2.0 * np.std(dfb_rows, 0) / math.sqrt(len(dfb_rows))
ax.errorbar(x, dfb_mean, yerr=dfb_se, fmt="none", ecolor=GREEN, elinewidth=0.7, capsize=1.5, zorder=3)
ax.scatter(x, dfb_mean, s=22, facecolors=GREEN, edgecolors="white", linewidths=0.6, zorder=5, label="Density-FB")
ax.legend(fontsize=6.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, handlelength=1.0, columnspacing=1.0)
for ext in ("pdf", "png"): fig.savefig(FIG / f"fig_sb_features_lose_rank{RANK}_dfb.{ext}", dpi=600)
print("Density-FB difference / G on rewarded states:", round(float(dfb_mean[A].sum()), 3))
print("values FB", tci([v[0] for v in vals["A"]]), tci([v[1] for v in vals["A"]]), "DFB", tci([v[0] for v in vals["D"]]), tci([v[1] for v in vals["D"]]))
print("best-fit difference / G, mean over 60 runs:", round(float(np.mean(g_fit)), 3), " rewarded bars true sum:", round(float(dm_mean[A].sum()), 3), " fitted sum:", round(float(fit_mean[A].sum()), 3))
