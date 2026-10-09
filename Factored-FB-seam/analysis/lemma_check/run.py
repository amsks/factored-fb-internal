"""Validate the fixed-features lemma of failure.tex on the five generated bottleneck processes.

The lemma: with the backward features fixed (span U = column space of B), the least-squares fit
of every successor row is its projection onto U, so the fitted action gap at the decision state is
    gap_hat(r) = dv^T P_U r / n,   dv = X[i0] - X[i1],  X = n M  (density ratio w.r.t. rho = 1/n),
and the discarded gap is (dv - P_U dv)^T r / n.  Three tests, all exact population quantities:

  1. Trained features.  Train FB with the didactic TD harness (bootstrapped target, Adam, 50k
     steps, 5 seeds x 2 learning rates) at ranks 2..9, keep B and F.  Compare the gap FB reads
     with its trained F, (F[i0] - F[i1]) B^T r / n, to the lemma's projection prediction computed
     from B alone.  The lemma says they agree whenever F is at its least-squares optimum given B;
     the residual is the TD fixed point's distance from that optimum.  Closed-form (SVD) fits agree
     by construction and are the sanity check.
  2. Reward sweep.  Same trained B; put the unit reward on each state in turn and on random
     rewards.  The lemma says the kept fraction depends on r only through <dv - P_U dv, r>, so
     the same features keep the decision for some rewards and lose it for others, with the size
     given by the formula.
  3. Constructive.  Append one direction to U, the residual dv - P_U dv, and refit.  The lemma
     says the discarded gap is then exactly zero at rank d+1 for every reward; we also record what
     this costs in FB's objective relative to the optimal rank-(d+1) fit.

Run (CPU, ~3 min):
    JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/lemma_check/run.py
    ... --figures-only        # reuse results.npz
Outputs: analysis/lemma_check/{results.npz, summary.json, run_log.txt},
         PAPER/ICLR/figures/lemma_check/fig_lemma_check.{pdf,png} (+ per-panel files).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
import family as fam                    # noqa: E402  (sets jax x64, imports the harness)

FIG_DIR = REPO / "PAPER" / "ICLR" / "figures" / "lemma_check"
RANKS = list(range(2, 10))
SEEDS = [0, 1, 2, 3, 4]
RATES = [0.001, 0.01]
N_RANDOM_REWARDS = 50
REWARD_RANK = 3
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


# ----------------------------------------------------------------------------
# pure linear algebra of the lemma
# ----------------------------------------------------------------------------
def projector(B):
    """Orthogonal projection onto the column space of B (n x d)."""
    Q, _ = np.linalg.qr(B)
    return Q @ Q.T


def ratio_matrix(mdp):
    """X = n M: the density ratio of the successor measure w.r.t. rho = 1/n (what FB fits)."""
    return mdp["n"] * mdp["m_exact"]


def decision_rows(mdp):
    return 2 * mdp["decision"], 2 * mdp["decision"] + 1


def true_gap(mdp, r):
    X = ratio_matrix(mdp)
    i0, i1 = decision_rows(mdp)
    return float((X[i0] - X[i1]) @ r / mdp["n"])


def lemma_gap(mdp, B, r):
    """The lemma's fitted gap from the features alone: dv^T P_U r / n."""
    X = ratio_matrix(mdp)
    i0, i1 = decision_rows(mdp)
    return float((X[i0] - X[i1]) @ projector(B) @ r / mdp["n"])


def discarded_gap(mdp, B, r):
    X = ratio_matrix(mdp)
    i0, i1 = decision_rows(mdp)
    dv = X[i0] - X[i1]
    return float((dv - projector(B) @ dv) @ r / mdp["n"])


def trained_gap(mdp, F, B, r):
    """The gap FB reads with its own F: (F[i0] - F[i1]) B^T r / n, i.e. F z with z = B^T rho r."""
    i0, i1 = decision_rows(mdp)
    return float((F[i0] - F[i1]) @ B.T @ r / mdp["n"])


def augmented_features(mdp, B):
    """Append the residual of the decision row difference outside span(B)."""
    X = ratio_matrix(mdp)
    i0, i1 = decision_rows(mdp)
    dv = X[i0] - X[i1]
    res = dv - projector(B) @ dv
    if np.linalg.norm(res) < 1e-12:
        return B
    return np.concatenate([B, (res / np.linalg.norm(res))[:, None]], axis=1)


def objective(mdp, B):
    """FB's population squared objective at the least-squares F given B: mean_i (1/n)||X_i - X_i P||^2."""
    X = ratio_matrix(mdp)
    R = X - X @ projector(B)
    return float(np.mean(np.sum(R ** 2, axis=1)) / mdp["n"])


# ----------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------
def load_processes():
    stored = json.load(open(REPO / "analysis" / "toy_ablation" / "family_mdps.json"))["processes"]
    mdps = []
    for p in stored:
        m = fam.sample_mdp(p["seed"], p["variant"])
        assert m["edges"].tolist() == p["edges"] and abs(m["gap"] - p["gap"]) < 1e-9, p["seed"]
        mdps.append(m)
    return mdps


def run(mdps):
    rows = []                     # test 1 + closed-form sanity
    reward_rows = []              # test 2
    construct_rows = []           # test 3 (closed-form B at every rank)
    t0 = time.monotonic()
    for mdp in mdps:
        X = ratio_matrix(mdp)
        r_goal = mdp["r"]
        n = mdp["n"]
        for d in RANKS:
            # closed form: SVD B, sanity of the identity
            Fs, Bs = fam.svd_fit(X, d)
            g_svd = trained_gap(mdp, Fs, Bs, r_goal)
            g_lem = lemma_gap(mdp, Bs, r_goal)
            assert abs(g_svd - g_lem) < 1e-9, (mdp["seed"], d, g_svd, g_lem)
            assert abs(g_lem + discarded_gap(mdp, Bs, r_goal) - mdp["gap"]) < 1e-9
            rows.append(dict(mdp=mdp["seed"], d=d, kind="closed", seed=-1, lr=0.0,
                             lemma=g_lem / mdp["gap"], read=g_svd / mdp["gap"]))
            # constructive on the closed-form B
            Ba = augmented_features(mdp, Bs)
            g_aug = lemma_gap(mdp, Ba, r_goal)
            assert abs(g_aug - mdp["gap"]) < 1e-9, (mdp["seed"], d, g_aug, mdp["gap"])
            Fs1, Bs1 = fam.svd_fit(X, min(d + 1, n))
            construct_rows.append(dict(mdp=mdp["seed"], d=d,
                                       gap_rank_d=g_lem / mdp["gap"],
                                       gap_aug=g_aug / mdp["gap"],
                                       obj_rank_d=objective(mdp, Bs),
                                       obj_aug=objective(mdp, Ba),
                                       obj_rank_d1=objective(mdp, Bs1),
                                       aug_cols=Ba.shape[1]))
            # trained
            cfg, pred, F, B = fam.train_group(mdp, d, "fb", SEEDS, RATES)
            for i, c in enumerate(cfg):
                g_read = trained_gap(mdp, F[i], B[i], r_goal)
                g_lem = lemma_gap(mdp, B[i], r_goal)
                rows.append(dict(mdp=mdp["seed"], d=d, kind="trained", seed=c["seed"], lr=c["lr"],
                                 lemma=g_lem / mdp["gap"], read=g_read / mdp["gap"],
                                 discarded=discarded_gap(mdp, B[i], r_goal) / mdp["gap"]))
                # constructive on the trained B: refit F by least squares given [B, residual]
                Ba = augmented_features(mdp, B[i])
                assert abs(lemma_gap(mdp, Ba, r_goal) - mdp["gap"]) < 1e-9
                if d == REWARD_RANK:
                    rng = np.random.default_rng(1000 * mdp["seed"] + 10 * c["seed"] + int(c["lr"] * 1000))
                    rewards = [("unit", k, np.eye(n)[k]) for k in range(n)]
                    rewards += [("random", k, rng.uniform(0, 1, n)) for k in range(N_RANDOM_REWARDS)]
                    for kind, k, rr in rewards:
                        reward_rows.append(dict(mdp=mdp["seed"], seed=c["seed"], lr=c["lr"], kind=kind, k=k,
                                                true=true_gap(mdp, rr), lemma=lemma_gap(mdp, B[i], rr),
                                                read=trained_gap(mdp, F[i], B[i], rr),
                                                aug=lemma_gap(mdp, Ba, rr)))
            tr = [x for x in rows if x["mdp"] == mdp["seed"] and x["d"] == d and x["kind"] == "trained"]
            log(json.dumps(dict(mdp=mdp["seed"], d=d,
                                lemma=round(float(np.mean([x["lemma"] for x in tr])), 3),
                                read=round(float(np.mean([x["read"] for x in tr])), 3),
                                max_abs_diff=round(float(np.max([abs(x["lemma"] - x["read"]) for x in tr])), 4),
                                elapsed=round(time.monotonic() - t0, 1))))
    return rows, reward_rows, construct_rows


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def make_figures(rows, reward_rows, construct_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    single = fam.single
    INK = single.INK
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    cmap = plt.get_cmap("viridis")
    rank_color = {d: cmap((d - 2) / 7) for d in RANKS}
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def save(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        log(f"wrote {FIG_DIR / stem}.pdf / .png")

    # (a) projection prediction vs the gap FB reads, trained features
    def panel_a(ax):
        tr = [x for x in rows if x["kind"] == "trained"]
        for d in RANKS:
            xs = [x["lemma"] for x in tr if x["d"] == d]
            ys = [x["read"] for x in tr if x["d"] == d]
            ax.scatter(xs, ys, s=11, color=rank_color[d], alpha=0.75, linewidths=0, zorder=3)
        cl = [x for x in rows if x["kind"] == "closed"]
        ax.scatter([x["lemma"] for x in cl], [x["read"] for x in cl], s=26, facecolors="none",
                   edgecolors=INK, linewidths=0.7, zorder=4)
        lo, hi = -0.35, 1.25
        ax.plot([lo, hi], [lo, hi], color="#9a9a9a", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("lemma: $\\Delta v^\\top \\Pi_U r$ from the features (fraction of exact)")
        ax.set_ylabel("gap FB reads with its trained $F$")
        ax.set_title("(a) fixed features: lemma vs FB's readout", loc="left", fontsize=8.5)
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, color=rank_color[d],
                          label=f"rank {d}") for d in RANKS]
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor="none",
                              markeredgecolor=INK, label="closed-form fit"))
        ax.legend(handles=handles, fontsize=6.4, frameon=False, loc="upper left", ncol=2,
                  handlelength=1.0, columnspacing=0.8, labelspacing=0.3)

    # (b) reward sweep at rank 3: predicted vs read, unit rewards and random rewards
    def panel_b(ax):
        units = [x for x in reward_rows if x["kind"] == "unit"]
        rands = [x for x in reward_rows if x["kind"] == "random"]
        ax.scatter([x["lemma"] for x in rands], [x["read"] for x in rands], s=6, color="#b0b0b0",
                   alpha=0.5, linewidths=0, zorder=2, label="random rewards")
        mdps = sorted({x["mdp"] for x in units})
        pcol = plt.get_cmap("tab10")
        for m in mdps:
            u = [x for x in units if x["mdp"] == m]
            ax.scatter([x["lemma"] for x in u], [x["read"] for x in u], s=13, color=pcol(m),
                       alpha=0.8, linewidths=0, zorder=3, label=f"unit reward, process {m + 1}")
        allv = [x["lemma"] for x in reward_rows] + [x["read"] for x in reward_rows]
        lo, hi = min(allv) - 0.1, max(allv) + 0.1
        ax.plot([lo, hi], [lo, hi], color="#9a9a9a", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("lemma: $\\Delta v^\\top \\Pi_U r$ from the features")
        ax.set_ylabel("gap FB reads with its trained $F$")
        ax.set_title(f"(b) one feature set, many rewards (rank {REWARD_RANK})", loc="left", fontsize=8.5)
        ax.legend(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.3)

    # (c) constructive: closed-form rank d vs rank d + residual direction; objective cost
    def panel_c(ax):
        ds = RANKS
        def rel_cost(d):
            # relative excess of the augmented fit over the optimal rank-(d+1) fit; 0 when both are exact
            vals = [x["obj_aug"] / x["obj_rank_d1"] - 1.0 if x["obj_rank_d1"] > 1e-12 else 0.0
                    for x in construct_rows if x["d"] == d]
            return float(np.mean(vals))
        def mean_over(key, d):
            return float(np.mean([x[key] for x in construct_rows if x["d"] == d]))
        ax.plot(ds, [mean_over("gap_rank_d", d) for d in ds], marker="o", markersize=4,
                color=single.ARM_COLOR["A"], label="FB, optimal rank $d$")
        ax.plot(ds, [mean_over("gap_aug", d) for d in ds], marker="s", markersize=4,
                color=single.ARM_COLOR["D"], label="rank $d$ + residual direction  ($d{+}1$)")
        ax.axhline(1.0, color="#9a9a9a", linewidth=0.7, linestyle="--")
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlabel("rank $d$")
        ax.set_ylabel("fraction of the exact action gap")
        ax.set_title("(c) adding the discarded direction", loc="left", fontsize=8.5)
        ax2 = ax.twinx()
        cost = [rel_cost(d) for d in ds]
        ax2.plot(ds, cost, color="#6a6a6a", linestyle=":", linewidth=1.2, marker="^", markersize=3.5,
                 label="objective above the optimal rank $d{+}1$ fit")
        ax2.set_ylabel("relative objective cost", color="#6a6a6a", fontsize=7.6)
        ax2.tick_params(labelsize=7.6, colors="#6a6a6a")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color("#9a9a9a")
        ax2.set_ylim(bottom=-0.02)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.4, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.6)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.tight_layout(pad=0.4, w_pad=1.6)
    save(fig, "fig_lemma_check"); plt.close(fig)
    for name, fn in (("a", panel_a), ("b", panel_b), ("c", panel_c)):
        fig, ax = plt.subplots(figsize=(3.3, 2.8)); fn(ax); fig.tight_layout(pad=0.3)
        save(fig, f"fig_lemma_check_{name}"); plt.close(fig)


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def summarize(rows, reward_rows, construct_rows):
    out = {}
    tr = [x for x in rows if x["kind"] == "trained"]
    log("\nTest 1 -- trained features, gap fraction: lemma prediction from B | FB's readout | max |diff| | corr")
    for d in RANKS:
        a = np.array([x["lemma"] for x in tr if x["d"] == d]); b = np.array([x["read"] for x in tr if x["d"] == d])
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        out[f"rank{d}"] = dict(lemma=float(a.mean()), read=float(b.mean()), max_abs_diff=float(np.abs(a - b).max()),
                               mean_abs_diff=float(np.abs(a - b).mean()), corr=corr, n=int(len(a)))
        log(f"  d={d}: {a.mean():6.3f} | {b.mean():6.3f} | {np.abs(a-b).max():.4f} | {corr:.4f}   (n={len(a)})")
    a = np.array([x["lemma"] for x in tr]); b = np.array([x["read"] for x in tr])
    out["all"] = dict(max_abs_diff=float(np.abs(a - b).max()), mean_abs_diff=float(np.abs(a - b).mean()),
                      corr=float(np.corrcoef(a, b)[0, 1]),
                      sign_agree=float(np.mean(np.sign(a) == np.sign(b))))
    log(f"  all ranks: max |diff| {out['all']['max_abs_diff']:.4f}, mean {out['all']['mean_abs_diff']:.4f}, "
        f"corr {out['all']['corr']:.4f}, sign agreement {out['all']['sign_agree']:.3f}")

    log(f"\nTest 2 -- reward sweep at rank {REWARD_RANK} (trained B): unit rewards per state, kept fraction "
        "lemma | read  (true gap in parentheses; mean over 10 runs)")
    units = [x for x in reward_rows if x["kind"] == "unit"]
    sweep = {}
    for m in sorted({x["mdp"] for x in units}):
        line = []
        for k in sorted({x["k"] for x in units}):
            u = [x for x in units if x["mdp"] == m and x["k"] == k]
            tg = u[0]["true"]
            lem = np.mean([x["lemma"] for x in u]); rd = np.mean([x["read"] for x in u])
            frac = (lem / tg, rd / tg) if abs(tg) > 0.05 else (float("nan"), float("nan"))
            sweep[f"mdp{m}_state{k}"] = dict(true=tg, lemma=float(lem), read=float(rd), frac_lemma=frac[0], frac_read=frac[1])
            line.append(f"s{k}({tg:+.2f}) {frac[0]:+.2f}|{frac[1]:+.2f}" if abs(tg) > 0.05 else f"s{k}({tg:+.2f}) --")
        log(f"  process {m + 1}: " + "  ".join(line))
    allr = np.array([x["lemma"] for x in reward_rows]); allq = np.array([x["read"] for x in reward_rows])
    out["reward_sweep"] = dict(max_abs_diff=float(np.abs(allr - allq).max()), corr=float(np.corrcoef(allr, allq)[0, 1]),
                               n=int(len(allr)), aug_max_err=float(np.max([abs(x["aug"] - x["true"]) for x in reward_rows])))
    log(f"  all {len(allr)} (reward, run) pairs: max |lemma - read| {out['reward_sweep']['max_abs_diff']:.4f}, "
        f"corr {out['reward_sweep']['corr']:.4f}; with the residual direction appended, max |gap - true| "
        f"{out['reward_sweep']['aug_max_err']:.1e}")

    log("\nTest 3 -- closed-form FB: gap fraction at rank d | with residual appended | objective cost vs optimal d+1")
    for d in RANKS:
        c = [x for x in construct_rows if x["d"] == d]
        g = np.mean([x["gap_rank_d"] for x in c]); ga = np.mean([x["gap_aug"] for x in c])
        cost = np.mean([x["obj_aug"] / x["obj_rank_d1"] - 1 if x["obj_rank_d1"] > 1e-12 else 0.0 for x in c])
        out[f"construct_rank{d}"] = dict(gap_rank_d=float(g), gap_aug=float(ga), rel_cost=float(cost))
        log(f"  d={d}: {g:6.3f} | {ga:6.3f} | {cost:+.3f}")
    out["sweep"] = sweep
    return out


if __name__ == "__main__":
    figures_only = "--figures-only" in sys.argv
    HERE.mkdir(parents=True, exist_ok=True)
    if figures_only:
        z = np.load(HERE / "results.npz", allow_pickle=True)
        rows, reward_rows, construct_rows = z["rows"].tolist(), z["reward_rows"].tolist(), z["construct_rows"].tolist()
    else:
        mdps = load_processes()
        log(f"{len(mdps)} processes; ranks {RANKS}; seeds {SEEDS}; rates {RATES}")
        rows, reward_rows, construct_rows = run(mdps)
        np.savez(HERE / "results.npz", rows=np.array(rows, dtype=object),
                 reward_rows=np.array(reward_rows, dtype=object),
                 construct_rows=np.array(construct_rows, dtype=object))
    summary = summarize(rows, reward_rows, construct_rows)
    make_figures(rows, reward_rows, construct_rows)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    (HERE / "run_log.txt").write_text("\n".join(LOG) + "\n")
