"""Section 6.1 on the four-state control-bottleneck MDP of the failure lemma, grow-with-d.

The base process (analysis/four_state_mdp/mdp.py) is states 0,1,2,3 with the bottleneck at state 1:
a_+ -> the rewarded absorbing state 2, a_- -> state 3.  On four states alone the action-difference is
the second singular direction, so FB recovers it at rank two -- a fixed finite MDP is always fit exactly
once the rank reaches its successor rank.  Lemma 1 therefore lets the MDP depend on d: at feature
dimension d the process is the base plus (d-1) disconnected components, each a large future-visit
direction that outranks the difference and pushes it to position d+1.  FB then reads none of the
action-value difference at every d, while the softmax cross-entropy fit (Density-FB) reads it all.  We
measure the recovered fraction of the exact action gap at the bottleneck,
    (Q_hat(1, a_+) - Q_hat(1, a_-)) / (Q(1, a_+) - Q(1, a_-)),
closed form and TD-trained, for the arms A-D, sweeping d in DIMS, plus the action values and the
frozen-feature refit (the lemma's projection identity) at dimension VALUES_DIM.

    JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/four_state_mdp/run.py
    ... --smoke          (dims 2,3; one seed)
    ... --figures-only   (rebuild figures + tables from four_state_*.csv)
Outputs: analysis/four_state_mdp/four_state_{closed,trained}.csv, four_state_summary.json,
         PAPER/ICLR/figures/four-state-mdp/fig_fs_*.{pdf,png}.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
import bottleneck_family_td as bf          # noqa: E402
import family as fam                        # noqa: E402
import mdp as fsm                           # noqa: E402
single = fam.single

FIG_DIR = REPO / "PAPER" / "ICLR" / "figures" / "four-state-mdp"
GAMMA = fsm.GAMMA
SMOKE = "--smoke" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv
# Grow-with-d (Lemma 1): at feature dimension d the MDP is the four-state bottleneck plus (d-1) added
# components, and every fit uses d features.  No single fixed MDP fails at every rank -- a finite MDP is
# fit exactly once the rank reaches its successor rank -- so the faithful illustration of the lemma lets
# the MDP grow with d, and FB then reads none of the action gap at every d while Density-FB reads it all.
DIMS = [1, 2, 3] if SMOKE else [1, 2, 3, 4, 5, 6]
FIXED_RANKS = [1, 2, 3] if SMOKE else [1, 2, 3, 4, 5, 6, 7, 8]   # rank sweep on the one connected MDP
SEEDS = [0] if SMOKE else list(fam.SEEDS)
RATES = [0.01] if SMOKE else list(fam.RATES)
TRAIN_STEPS = 300 if SMOKE else fam.TRAIN_STEPS
VALUES_DIM = 2                              # dimension used for the value / feature-loss panels
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ----------------------------------------------------------------------------
# Closed-form and TD-trained arms on the fixed four-state MDP at representation rank d,
# same schema and harness as analysis/toy_ablation/family.py.
# ----------------------------------------------------------------------------
def closed_rows(mdp, d):
    n, k = mdp["n"], mdp["seed"]
    rows = []
    F, B = fam.svd_fit(n * mdp["m_exact"], d)
    rows.append(dict(arm="A", method="fb", mdp=k, d=d, seed=-1, **fam.gap_stats(mdp, fam.linear_readout(mdp, F, B)),
                     retained_mass=fam.retained_mass(mdp, (1 - GAMMA) * (F @ B.T) / n)))
    F, B = fam.svd_fit(mdp["t_exact"], d)
    Q = fam.linear_readout(mdp, F, B) * n / (1 - GAMMA)
    rows.append(dict(arm="B", method="fb_norm", mdp=k, d=d, seed=-1, **fam.gap_stats(mdp, Q),
                     retained_mass=fam.retained_mass(mdp, F @ B.T)))
    for sd in SEEDS:
        s = 1000 * sd + 17 * d + 101 * k
        for arm, method, tgt, kind in (("C", "softsq", mdp["t_exact"], "sq"),
                                        ("D", "density", mdp["t_exact"], "ce")):
            F, B, P, _ = fam.softmax_fit(tgt, n, d, s, kind)
            rows.append(dict(arm=arm, method=method, mdp=k, d=d, seed=sd,
                             **fam.gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                             retained_mass=fam.retained_mass(mdp, P)))
            if arm == "D":
                rows.append(fam.e_arm_rows(mdp, F, B, dict(mdp=k, d=d, seed=sd,
                                                           retained_mass=fam.retained_mass(mdp, P))))
    return rows


TRAIN_METHODS = ["fb", "normsq", "softsq", "density"]     # soft-target arm (density_soft) dropped


def trained_rows(mdp, d):
    n, k = mdp["n"], mdp["seed"]
    rows = []
    for method in TRAIN_METHODS:
        cfg, pred, F_all, B_all = fam.train_group(mdp, d, method, SEEDS, RATES, steps=TRAIN_STEPS)
        for i, c in enumerate(cfg):
            P = bf.readout(pred[i], fam._base_method(method), GAMMA, n)
            base = dict(mdp=k, d=d, seed=c["seed"], lr=c["lr"])
            rows.append(dict(arm=fam.ARM_OF[method], method=method, **base,
                             **fam.gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                             retained_mass=fam.retained_mass(mdp, P)))
            if method == "density":
                rows.append(fam.e_arm_rows(mdp, F_all[i], B_all[i],
                                           dict(base, retained_mass=fam.retained_mass(mdp, P))))
    return rows


# ----------------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------------
def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "text.color": single.INK,
                         "axes.edgecolor": single.INK, "axes.labelcolor": single.INK,
                         "axes.labelsize": 16, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                         "legend.fontsize": 12, "xtick.color": single.INK, "ytick.color": single.INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    return plt


# One figure size and one font scale for every four-state figure (ICLR); saved at exactly this size
# (no tight-crop) so the figures share dimensions.
FS_SINGLE = (4.6, 3.4)


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300)


def make_ablation_figure(closed, trained, ranks, xlabel, stem_prefix, legend_loc="center right"):
    plt = _rc()
    from matplotlib.lines import Line2D
    for suffix, rows, hollow in (("closed", closed, False), ("trained", trained, True)):
        fig, ax = plt.subplots(figsize=FS_SINGLE)
        fam.draw_panel(ax, [r for r in rows if r["arm"] not in ("E", "F")], hollow=hollow, ranks=ranks)
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlim(ranks[0] - 0.25, ranks[-1] + 0.25)
        ax.set_xticks(ranks)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel("recovered fraction\nof the action gap", fontsize=15)
        ax.tick_params(labelsize=13.5, length=3.0)
        short = {"A": "FB", "B": "FB, normalized target", "C": "softmax + squared error",
                 "D": "Density-FB"}
        hs = [Line2D([0], [0], color=single.ARM_COLOR[key], marker=single.ARM_MARKER[key], markersize=6,
                     linewidth=1.8, markerfacecolor="none" if hollow else single.ARM_COLOR[key], label=v)
              for key, v in short.items()]
        hs.append(Line2D([0], [0], color=single.EXACT_COLOR, linestyle="--", linewidth=1.2, label="exact"))
        ax.legend(handles=hs, loc=legend_loc, fontsize=10.5, frameon=False, handlelength=1.6, labelspacing=0.3)
        fig.tight_layout(pad=0.4)
        _save(fig, f"{stem_prefix}_{suffix}")
        plt.close(fig)


def make_values_figure(rows, mdps):
    plt = _rc()
    from matplotlib.patches import Patch
    vals = fam.values_rows(rows, mdps, d=VALUES_DIM)
    groups = [("exact", "exact", single.EXACT_COLOR), ("A", "FB", single.ARM_COLOR["A"]),
              ("D", "Density-FB", single.ARM_COLOR["D"])]
    fig, ax = plt.subplots(figsize=FS_SINGLE)
    width, gap = 0.34, 0.12
    xt, xl = [], []
    for i, (key, name, color) in enumerate(groups):
        base = i * (2 * width + gap + 0.3)
        for a in (0, 1):
            m, lo, hi = vals[key][a]
            x = base + a * (width + 0.03)
            ax.bar(x, m, width=width, color=color, alpha=1.0 if a == 0 else 0.42,
                   edgecolor=color, linewidth=0.8, zorder=2)
            if hi > lo:
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], ecolor=single.INK, elinewidth=0.9,
                            capsize=2.0, linestyle="none", zorder=3)
            ax.annotate(f"{m:.2f}", (x, max(m, hi)), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=11.5, color=single.INK)
        xt.append(base + width / 2 + 0.015)
        xl.append(name)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=14)
    ax.set_ylabel("value at the decision state\n(exact pair mapped to 1 and 0)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13.5, length=2.5)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9a9a")
    ax.set_ylim(0.0, 1.16)
    ax.legend(handles=[Patch(facecolor="#6a6a6a", label="$a_+$ (state $2$)"),
                       Patch(facecolor="#6a6a6a", alpha=0.42, label="$a_-$ (state $3$)")],
              fontsize=11.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14),
              handlelength=1.2, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig_fs_values")
    plt.close(fig)


def make_features_lose_figure(mdp, d):
    """The lemma's projection identity at the exact (closed-form) FB optimum: fit the rank-d SVD of the
    exact successor densities, refit the two forward vectors to m+/m- by rho-weighted least squares with
    those backward features held fixed, and show the fitted difference (= projection of Delta m) is zero at
    the rewarded state, while the softmax fit (Density-FB) at the same d keeps it."""
    plt = _rc()
    n, b, rho, r = mdp["n"], mdp["decision"], mdp["rho"], mdp["r"]
    A = [s for s in range(n) if r[s] > 0]
    X = n * mdp["m_exact"]
    _, B = fam.svd_fit(X, d)                                  # exact FB backward features at rank d
    W = np.sqrt(rho)[:, None]
    m_p, m_m = X[2 * b], X[2 * b + 1]
    dm = m_p - m_m
    fit_diff = B @ np.linalg.lstsq(W * B, np.sqrt(rho) * dm, rcond=None)[0]   # Pi_B Delta m
    G = float(np.sum(rho * dm * r))
    G_hat = float(np.sum(rho * fit_diff * r))
    dfbs = []
    for s in SEEDS:                                    # rank-1 softmax is seed-sensitive: show the spread
        _, _, P, _ = fam.softmax_fit(mdp["t_exact"], n, d, 1000 * s + 17 * d, "ce")
        dfbs.append((P[2 * b] - P[2 * b + 1]) / (1 - GAMMA))   # difference in discounted visits
    dfb, dfb_sd = np.mean(dfbs, 0), np.std(dfbs, 0)
    log(f"[features-lose d={d}] G={G:.3f}  FB best-fit reads {G_hat:.3f}  "
        f"Density-FB reads {float(dfb @ r):.3f} +/- {float(dfb_sd @ (r > 0)):.3f}")

    ORANGE, GREEN = single.ARM_COLOR["A"], single.ARM_COLOR["D"]
    fig, ax = plt.subplots(figsize=FS_SINGLE)
    x = np.arange(n)
    ax.axvspan(A[0] - 0.5, A[-1] + 0.5, color=GREEN, alpha=0.12, zorder=0)
    ax.text(A[0], (rho * dm).max() * 1.12 + 0.05, "reward $=1$", color=GREEN, fontsize=12, ha="center", va="bottom")
    ax.axhline(0, color="#cfcfcf", linewidth=0.6, zorder=1)
    ax.bar(x, rho * dm, 0.6, color=single.INK, zorder=2, label="true difference $\\Delta M$")
    ax.errorbar(x, dfb, yerr=dfb_sd, fmt="none", ecolor=GREEN, elinewidth=1.0, capsize=2.5, zorder=4)
    ax.scatter(x, dfb, s=95, facecolors=GREEN, edgecolors="white", linewidths=1.1, zorder=5,
               label="Density-FB, same $d$")
    ax.scatter(x, rho * fit_diff, s=95, facecolors=ORANGE, edgecolors="white", linewidths=1.1, zorder=6,
               label="best fit with FB's features")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(n)], fontsize=12.5)
    ax.set_xlabel("future state $s^+$", fontsize=16)
    ax.set_ylabel("difference in discounted visits", fontsize=14)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.legend(fontsize=11, frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    _save(fig, "fig_fs_features_lose")
    plt.close(fig)


# ----------------------------------------------------------------------------
# CSV round-trip.
# ----------------------------------------------------------------------------
def write_csv(rows, name):
    keys = sorted({k for r in rows for k in r})
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(name):
    rows = []
    with open(HERE / name) as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v == "":
                    out[k] = ""
                    continue
                try:
                    out[k] = int(v) if k in ("mdp", "d", "seed") else float(v)
                except ValueError:
                    out[k] = (v == "True") if v in ("True", "False") else v
            rows.append(out)
    return rows


def main():
    t0 = time.time()
    mdps = {d: fsm.make_mdp(d - 1) for d in DIMS}         # grow-with-d: at dim d, base + (d-1) components
    cm = fsm.make_connected_mdp()                         # the single connected shared-tail MDP (fixed)
    if FIGURES_ONLY:
        closed, trained = read_csv("four_state_closed.csv"), read_csv("four_state_trained.csv")
        fclosed, ftrained = read_csv("four_state_fixed_closed.csv"), read_csv("four_state_fixed_trained.csv")
    else:
        closed, trained = [], []
        for d in DIMS:                                    # grow-with-d family
            m = mdps[d]
            closed += closed_rows(m, d)
            trained += trained_rows(m, d)
            log(f"[grow] d={d}  n={m['n']}  done ({time.time() - t0:.0f}s)")
        write_csv(closed, "four_state_closed.csv")
        write_csv(trained, "four_state_trained.csv")
        fclosed, ftrained = [], []
        for d in FIXED_RANKS:                             # rank sweep on the one connected MDP
            fclosed += closed_rows(cm, d)
            ftrained += trained_rows(cm, d)
            log(f"[fixed] rank={d}  n={cm['n']}  done ({time.time() - t0:.0f}s)")
        write_csv(fclosed, "four_state_fixed_closed.csv")
        write_csv(ftrained, "four_state_fixed_trained.csv")

    # grow-with-d (the "MDPs with higher rank" family) -> fig_fs_ablation_{closed,trained}
    make_ablation_figure(closed, trained, DIMS, "feature dimension $d$", "fig_fs_ablation")
    # one fixed connected MDP, rank sweep -> fig_fs_ablation_fixed_{closed,trained}
    make_ablation_figure(fclosed, ftrained, FIXED_RANKS, "representation rank $d$", "fig_fs_ablation_fixed")
    make_values_figure(closed, [mdps[VALUES_DIM]])       # closed-form = the exact optimum of the lemma
    make_features_lose_figure(mdps[VALUES_DIM], VALUES_DIM)

    # headline numbers for the paper text
    def frac(rows, arm, d):
        vals = [float(r["gap_ratio"]) for r in rows if r["arm"] == arm and int(r["d"]) == d]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {"gamma": GAMMA, "dims": DIMS, "n_states": {d: mdps[d]["n"] for d in DIMS},
               "connected_n": cm["n"], "fixed_ranks": FIXED_RANKS,
               "grow": {"closed": {a: {d: frac(closed, a, d) for d in DIMS} for a in "ABCD"},
                        "trained": {a: {d: frac(trained, a, d) for d in DIMS} for a in "ABCD"}},
               "fixed": {"closed": {a: {d: frac(fclosed, a, d) for d in FIXED_RANKS} for a in "ABCD"},
                         "trained": {a: {d: frac(ftrained, a, d) for d in FIXED_RANKS} for a in "ABCD"}}}
    (HERE / "four_state_summary.json").write_text(json.dumps(summary, indent=2))
    names = {"A": "FB", "B": "FB, norm target", "C": "softmax+sq", "D": "Density-FB"}
    for tag, xs, rc, rt in (("grow-with-d", DIMS, "grow", "grow"),
                            (f"fixed connected MDP (n={cm['n']})", FIXED_RANKS, "fixed", "fixed")):
        log("")
        log(f"recovered fraction of the action gap, {tag} (closed / TD-trained):")
        log(f"{'arm':>18}  " + "  ".join(f"{d:>5}" for d in xs))
        for a in "ABCD":
            c = "  ".join(f"{summary[rc]['closed'][a][d]:+.2f}" for d in xs)
            tr = "  ".join(f"{summary[rt]['trained'][a][d]:+.2f}" for d in xs)
            log(f"{names[a]:>18}  closed {c}   trained {tr}")
    (HERE / "four_state_log.txt").write_text("\n".join(LOG))
    log(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
