"""Section 6.1: "What the features lose at the bottleneck" (three panels, user spec 2026-09-20).
Eleven-state process. Backward features B = those of a TD-trained rank-3 FB critic (the critic of the
values figure). With B FROZEN, the two forward coefficient vectors are refit by least squares against
the EXACT successor densities m+/- (density ratios w.r.t. rho = 1/n):
    c_pm = argmin_c sum_s rho(s) [m_pm(s) - c.B(s)]^2
(a) true difference Delta m, the fitted difference B(c+ - c-), and the projection Pi_B Delta m (equal by
    the lemma's first equation), all multiplied by rho (discounted visit differences);
(b) what the fit leaves out, e = Delta m - Pi_B Delta m, times rho, cluster A shaded (rewarded states);
(c) G_r - G_hat_r against sum_s rho(s) e(s) r(s): equal by the lemma's second equation.
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/features_lose_panel.py [rank] [run]
"""
import importlib.util, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
sys.argv, _argv = [sys.argv[0]], sys.argv          # run.py parses flags at import
spec.loader.exec_module(sb); sys.argv = _argv
fam, INK = sb.fam, sb.single.INK
FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GAMMA = 0.8; ORANGE, GREEN, BLUE = sb.single.ARM_COLOR["A"], "#029E73", "#0173B2"

mdp = sb.fixed_processes()[1]; assert mdp["n"] == 11 and abs(mdp["gap"] - 5 / 3) < 1e-9
n = mdp["n"]; b = mdp["decision"]; rho = mdp["rho"]; r = mdp["r"]
A = [s for s in range(n) if r[s] > 0]
X = n * mdp["m_exact"]                                   # density ratios m = M / rho
cfg, pred, F_all, B_all = fam.train_group(mdp, RANK, "fb", list(fam.SEEDS), list(fam.RATES), steps=fam.TRAIN_STEPS)
B = np.asarray(B_all[RUN], np.float64)                  # (n, d), frozen
m_p, m_m = X[2 * b], X[2 * b + 1]
W = np.sqrt(rho)[:, None]
c_p = np.linalg.lstsq(W * B, np.sqrt(rho) * m_p, rcond=None)[0]     # rho-weighted least squares
c_m = np.linalg.lstsq(W * B, np.sqrt(rho) * m_m, rcond=None)[0]
fit_diff = B @ (c_p - c_m)
dm = m_p - m_m
proj = B @ np.linalg.lstsq(W * B, np.sqrt(rho) * dm, rcond=None)[0]  # Pi_B Delta m
assert np.allclose(fit_diff, proj, atol=1e-9), "first equation of the lemma"
e = dm - proj
G = float(np.sum(rho * dm * r)); G_hat = float(np.sum(rho * fit_diff * r)); missing = float(np.sum(rho * e * r))
assert abs((G - G_hat) - missing) < 1e-9, "second equation of the lemma"
# the critic's own readout with its TRAINED forward map, for the record
F = np.asarray(F_all[RUN], np.float64); G_trained = float(np.sum(rho * ((F[2 * b] - F[2 * b + 1]) @ B.T) * r))
print(f"rank {RANK} run {RUN} ({cfg[RUN]}): G = {G:.3f}, G_hat (best fit, frozen B) = {G_hat:.3f}, missing = {missing:.3f}, "
      f"trained forward map reads {G_trained:.3f}; max |fit_diff - proj| = {np.abs(fit_diff - proj).max():.1e}")
others = []
for i in range(len(cfg)):
    Bi = np.asarray(B_all[i], np.float64); pi_ = Bi @ np.linalg.lstsq(W * Bi, np.sqrt(rho) * dm, rcond=None)[0]
    others.append(float(np.sum(rho * pi_ * r)))
print("G_hat over all 10 runs' features:", np.round(others, 3))

cfg_d, pred_d, _, _ = fam.train_group(mdp, RANK, "density", list(fam.SEEDS), list(fam.RATES), steps=fam.TRAIN_STEPS)
pred_d = np.asarray(pred_d, np.float64)                                   # (runs, 2n, n) softmax rows, sum to 1
dfb_diff = (pred_d[:, 2 * b] - pred_d[:, 2 * b + 1]) / (1 - GAMMA)        # difference in discounted visits, per run
dfb_mean, dfb_sd = dfb_diff.mean(0), dfb_diff.std(0)
G_dfb = float(np.mean(dfb_diff @ r))
print(f"Density-FB rank {RANK}: value difference read {G_dfb:.3f} (per run {np.round(dfb_diff @ r, 2)})")
WITH_DFB = os.environ.get("WITH_DFB", "0") == "1"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
# One plot. Scale: rho * Delta m = Delta M, the difference in discounted visits (M has mass 1/(1-gamma)); the rewarded
# bars then sum to G_r directly (prelims: Q = int r dM), so no further factor is applied.
fig, ax = plt.subplots(figsize=(3.6, 2.4))
x = np.arange(n)
ax.axvspan(A[0] - 0.5, A[-1] + 0.5, color=GREEN, alpha=0.12, zorder=0)
ax.text(np.mean(A), 1.36, "Reward = 1", color=GREEN, fontsize=7.5, ha="center", va="center")
ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
ax.bar(x, rho * dm, 0.6, color=INK, zorder=2, label="true difference")
ax.scatter(x, rho * fit_diff, s=22, facecolors=ORANGE, edgecolors="white", linewidths=0.6, zorder=4, label="best fit with FB's features")
if WITH_DFB:
    ax.errorbar(x, dfb_mean, yerr=dfb_sd, fmt="none", ecolor=GREEN, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.scatter(x, dfb_mean, s=22, facecolors=GREEN, edgecolors="white", linewidths=0.6, zorder=5, label="Density-FB, same rank")
ax.set_xticks(x); ax.set_xticklabels([str(k) for k in range(n)], fontsize=7)
ax.set_xlabel("future state $s^+$"); ax.set_ylabel("difference in discounted visits", fontsize=8.5)
ax.legend(fontsize=6.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3 if WITH_DFB else 2, handlelength=1.2, columnspacing=1.2, markerscale=1.0)
ax.legend(fontsize=6.8, frameon=False, loc="upper right", handlelength=1.2)
ax.set_ylim(-1.3, 1.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_sb_features_lose_rank{RANK}{'_dfb' if WITH_DFB else ''}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FIG / f"fig_sb_features_lose_rank{RANK}{'_dfb' if WITH_DFB else ''}.pdf")
print("rewarded bars sum:", float(np.sum((rho * dm)[A])), " G:", G)
