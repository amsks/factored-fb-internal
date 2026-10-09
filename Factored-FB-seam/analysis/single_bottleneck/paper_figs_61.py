"""Section 6.1 figures, one style, all on the eleven-state process P11 (fig:didactic-mdp).

  fig_61_main      (a) action values at the bottleneck, rank 3: exact, FB, Density-FB (10 TD runs, 95% CI)
                   (b) Lemma (lem:bottleneck-loss): difference a+ - a- in discounted future visits.
                       Exact bars; least-squares refit of FB's forward vectors on its FROZEN trained
                       features (every run, all zero because the feature averages of the two futures
                       coincide); Density-FB at the same rank.
                   (c) Proposition (prop:bottleneck-preservation): KL(M || M_hat) at the bottleneck for
                       Density-FB, larger of the two actions, per run and rank, against the threshold
                       eps = (1-gamma)^2 Delta^2 / (2 R^2). Filled markers: ordering preserved.
  fig_61_rank      FB vs Density-FB, fraction of the exact action-value gap read, ranks 1..11 (TD, P11)
  fig_61_ablation  all ablation arms on P11 (appendix)

TD runs: analysis/toy_ablation/family.train_group (5 seeds x 2 learning rates, 50k steps). Ranks >= 2 of
the rank sweep come from sb_trained.csv (written by run.py); rank 1 and the KL/refit quantities are
computed here and cached in paper_figs_61_cache.npz (--recompute to redo).
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/paper_figs_61.py
"""
import csv, importlib.util, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
# SB_PROCESS=p11 (default): the eleven-state process P11, rank sweep read from sb_trained.csv.
# SB_PROCESS=p7: the seven-state process (entry 0,1; bottleneck 2; A = {3}; B = {4}; tail {5,6}),
#   built by design_single_bottleneck.build(entry=2, cluster=1, tail=2, merge=True); every arm is
#   trained here (same harness) and cached; figures go to figures/seven_state/.
PROCESS = os.environ.get("SB_PROCESS", "p11")
TAIL_REWARD = float(os.environ.get("SB_TAIL_REWARD", "0.5"))
if PROCESS == "p9":
    # pass-through: entry 0,1; bottleneck 2; A = {3}; B = {4}; tail ring {5..8}; reward 1 for
    # entering A and TAIL_REWARD on every tail state (so V(A) is high; the gap at 2 stays 1).
    FIG = REPO / "PAPER" / "ICLR" / "figures" / "nine_state"
    CACHE = HERE / "paper_figs_61_p9_cache.npz"
    GAMMA, VAL_RANK, KL_RANKS = 0.8, 4, [1, 2, 3, 4, 5, 6]
elif PROCESS == "p7":
    FIG = REPO / "PAPER" / "ICLR" / "figures" / "seven_state"
    CACHE = HERE / "paper_figs_61_p7_cache.npz"
    GAMMA, VAL_RANK, KL_RANKS = 0.8, 2, [1, 2, 3, 4]
else:
    FIG = REPO / "PAPER" / "ICLR" / "figures" / "single_bottleneck"
    CACHE = HERE / "paper_figs_61_cache.npz"
    GAMMA, VAL_RANK, KL_RANKS = 0.8, 3, [1, 2, 3, 4, 5]
FIG.mkdir(parents=True, exist_ok=True)
SWEEP_METHODS = {"fb": "A", "normsq": "B", "softsq": "C", "density": "D", "density_soft": "F"}

INK, GRID = "#222222", "#d9d9d9"
EXACT, FB, DFB = "#6f6f6f", "#DE8F05", "#029E73"
ARM = {"A": ("FB", FB, "o"), "B": ("FB, normalized target", "#CC78BC", "s"),
       "C": ("softmax + squared error", "#D55E00", "^"), "D": ("Density-FB", DFB, "D"),
       "F": ("Density-FB, soft immediate target", "#56B4E9", "v")}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "text.color": INK,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", FIG / f"{stem}.pdf")


def ci95(v):
    v = np.asarray(v, float)
    return v.mean(), (2.262 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


# ------------------------------------------------------------------ compute (cached)
def compute():
    spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [argv[0]]; spec.loader.exec_module(sb); sys.argv = argv
    fam = sb.fam
    if PROCESS == "p9":
        sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
        import design_single_bottleneck as dsb
        edges, b_, A_, B_, T_ = dsb.build_passthrough(entry=2, tail=4)
        mdp = sb.finish(0, "P9", dict(entry=2, tail=4, passthrough=True), edges,
                        np.full((len(edges), 2), 0.5), b_, A_, B_, T_)
        r9 = mdp["r"].copy(); r9[T_] = TAIL_REWARD
        q9 = mdp["m_exact"] @ r9
        mdp.update(r=r9, q_exact=q9, gap=float(q9[2 * b_] - q9[2 * b_ + 1]))
    elif PROCESS == "p7":
        sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
        import design_single_bottleneck as dsb
        edges, b_, A_, B_, T_ = dsb.build(entry=2, cluster=1, tail=2, merge=True)
        mdp = sb.finish(0, "P7", dict(entry=2, cluster=1, tail=2, merge=True), edges,
                        np.full((len(edges), 2), 0.5), b_, A_, B_, T_)
    else:
        mdp = sb.fixed_processes()[1]; assert mdp["n"] == 11
    n, b, rho, r = mdp["n"], mdp["decision"], mdp["rho"], mdp["r"]
    T = mdp["t_exact"]; X = n * mdp["m_exact"]; dm = X[2 * b] - X[2 * b + 1]; W = np.sqrt(rho)[:, None]
    out = dict(T=T, r=r, b=b, n=n, A=np.array(mdp["clusterA"]), edges=np.asarray(mdp["edges"]))
    if PROCESS in ("p7", "p9"):
        # the rank sweep, every arm, same harness as run.py's run_trained
        rows = []
        for d in range(1, n):
            for meth, arm in SWEEP_METHODS.items():
                _, pred, _, _ = fam.train_group(mdp, d, meth)
                for p in pred:
                    P = sb.bf.readout(p, fam._base_method(meth), GAMMA, n); q = P @ r / (1 - GAMMA)
                    rows.append((ord(arm), d, q[2 * b], q[2 * b + 1], (q[2 * b] - q[2 * b + 1]) / mdp["gap"]))
                print(f"sweep d={d} {meth}: {np.mean([x[4] for x in rows[-len(pred):]]):+.2f}", flush=True)
        out["rows"] = np.array(rows)
    # (b) refit on FB's frozen rank-3 features, every run
    _, _, _, B_fb = fam.train_group(mdp, VAL_RANK, "fb")
    refit, moment = [], []
    for B in np.asarray(B_fb, float):
        refit.append(rho * (B @ np.linalg.lstsq(W * B, np.sqrt(rho) * dm, rcond=None)[0]))
        moment.append(np.linalg.norm(B.T @ (rho * dm)))
    out.update(refit=np.array(refit), moment=np.array(moment))
    _, pred_d3, _, _ = fam.train_group(mdp, VAL_RANK, "density")
    out["dfb_diff"] = np.asarray(pred_d3, float)[:, 2 * b] / (1 - GAMMA) - np.asarray(pred_d3, float)[:, 2 * b + 1] / (1 - GAMMA)
    # (c) KL at the bottleneck and ordering, ranks 1..5; also rank-1 gap ratios for the sweep
    kls, order, gr1 = [], [], {}
    for d in KL_RANKS:
        _, pred, _, _ = fam.train_group(mdp, d, "density")
        P = np.asarray(pred, float)
        kls.append([max(kl(T[2 * b], p[2 * b]), kl(T[2 * b + 1], p[2 * b + 1])) for p in P])
        order.append([(p[2 * b] - p[2 * b + 1]) @ r > 0 for p in P])
        if d == 1:
            gap = (T[2 * b] - T[2 * b + 1]) @ r
            gr1["D"] = [((p[2 * b] - p[2 * b + 1]) @ r) / gap for p in P]
    _, pred_fb1, _, _ = fam.train_group(mdp, 1, "fb")
    P1 = np.stack([sb.bf.readout(p, "fb", GAMMA, n) for p in pred_fb1])
    gap = (T[2 * b] - T[2 * b + 1]) @ r
    gr1["A"] = [((p[2 * b] - p[2 * b + 1]) @ r) / gap for p in P1]
    out.update(kl=np.array(kls), order=np.array(order), gr1_A=np.array(gr1["A"]), gr1_D=np.array(gr1["D"]))
    np.savez(CACHE, **out)
    return out


def trained_rows(c=None):
    if PROCESS in ("p7", "p9"):
        return [dict(arm=chr(int(a)), d=int(d), q0=q0, q1=q1, gap_ratio=g) for a, d, q0, q1, g in c["rows"]]
    rows = [r for r in csv.DictReader(open(HERE / "sb_trained.csv")) if r["mdp"] == "0"]
    for r in rows:
        r["d"] = int(r["d"])
    return rows


# ------------------------------------------------------------------ figures
def fig_main(c, rows):
    T, r, b, n = c["T"], c["r"], int(c["b"]), int(c["n"])
    gap = float((T[2 * b] - T[2 * b + 1]) @ r) / (1 - GAMMA)
    R = float(r.max() - r.min()); eps = (1 - GAMMA) ** 2 * gap ** 2 / (2 * R ** 2)
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw=dict(width_ratios=[0.8, 1.35, 1.0]))

    # (a) values
    sub = [x for x in rows if x["d"] == VAL_RANK]
    groups = [("exact", EXACT, (T[2 * b] @ r / (1 - GAMMA), 0.0), (T[2 * b + 1] @ r / (1 - GAMMA), 0.0))]
    for arm, (name, col, _) in (("A", ARM["A"]), ("D", ARM["D"])):
        q0 = ci95([float(x["q0"]) for x in sub if x["arm"] == arm]); q1 = ci95([float(x["q1"]) for x in sub if x["arm"] == arm])
        groups.append((name, col, q0, q1))
    w = 0.36
    for i, (name, col, qp, qm) in enumerate(groups):
        ax.bar(i - w / 2, qp[0], w, color=col, yerr=qp[1], ecolor=INK, error_kw=dict(lw=0.6, capsize=1.5))
        ax.bar(i + w / 2, qm[0], w, color=col, alpha=0.4, hatch="/////", edgecolor=col, linewidth=0,
               yerr=qm[1], ecolor=INK, error_kw=dict(lw=0.6, capsize=1.5))
        print(f"(a) {name}: Q(a+) {qp[0]:.2f}+-{qp[1]:.2f}  Q(a-) {qm[0]:.2f}+-{qm[1]:.2f}")
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(f"value at state {b}"); ax.set_ylim(0, 1.3 * max(max(g[2][0], g[3][0]) for g in groups))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#888888"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#cccccc", hatch="/////", edgecolor="#888888", lw=0)],
              labels=["$a_+$", "$a_-$"], loc="upper center", ncol=2, frameon=False, handlelength=1.1,
              borderaxespad=0, columnspacing=0.8)
    ax.set_title("(a) action values", loc="left", pad=13)

    # (b) lemma
    x = np.arange(n); A = [int(v) for v in c.get("A", [3, 4])]
    bx.axvspan(A[0] - 0.5, A[-1] + 0.5, color=DFB, alpha=0.10, lw=0)
    bx.axhline(0, color=GRID, lw=0.6)
    exact_diff = (T[2 * b] - T[2 * b + 1]) / (1 - GAMMA)
    bx.bar(x, exact_diff, 0.62, color=EXACT, label="exact")
    dm, ds = c["dfb_diff"].mean(0), c["dfb_diff"].std(0)
    bx.errorbar(x + 0.17, dm, yerr=ds, fmt="D", ms=2.8, color=DFB, mec="white", mew=0.3, elinewidth=0.6, label="Density-FB")
    bx.plot(x - 0.17, c["refit"].mean(0), "o", ms=3.2, color=FB, mec="white", mew=0.3, label="FB features, best fit")
    bx.set_xticks(x); bx.set_xlabel("future state $s^+$")
    bx.set_ylabel("visits after $a_+$ minus $a_-$")
    top = 1.2 * float(np.abs(exact_diff).max())
    bx.text(float(np.mean(A)), 0.9 * top, "reward 1", color=DFB, fontsize=6.5, ha="center")
    bx.set_ylim(-top, top)
    bx.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.0,
              borderaxespad=0.1, fontsize=6.0, columnspacing=0.8, handletextpad=0.3)
    bx.set_title("(b) Lemma 1: fit on FB's features", loc="left", pad=13)
    print(f"(b) max |feature-average difference| over runs {c['moment'].max():.1e}, max |refit| {np.abs(c['refit']).max():.1e}")

    # (c) proposition: region below eps is where the proposition guarantees the ordering
    lo_y, hi_y = 3e-6, 8.0
    cx.axhspan(lo_y, eps, color=DFB, alpha=0.08, lw=0, zorder=0)
    cx.axhline(eps, color=INK, ls="--", lw=0.7, zorder=1)
    cx.text(KL_RANKS[-1] + 0.45, eps * 0.62, "ordering\nguaranteed", fontsize=6.3, color=DFB, ha="right", va="top")
    cx.text(KL_RANKS[-1] + 0.45, eps * 1.35, f"$\\varepsilon={eps:.3f}$", fontsize=6.5, ha="right", va="bottom")
    for i, d in enumerate(KL_RANKS):
        v, ok = c["kl"][i], c["order"][i]
        xs = d + np.linspace(-0.22, 0.22, len(v))
        cx.scatter(xs[ok], v[ok], s=11, color=DFB, lw=0, zorder=3)
        cx.scatter(xs[~ok], v[~ok], s=11, facecolors="white", edgecolors="#C0392B", lw=0.8, zorder=4)
        cx.text(d, hi_y * 0.75, f"{int(ok.sum())}/{len(ok)}", fontsize=6.3 if len(KL_RANKS) <= 5 else 5.2, ha="center", va="top",
                color=DFB if ok.all() else "#C0392B")
        print(f"(c) rank {d}: KL max-over-actions median {np.median(v):.4f}, below eps {np.mean(v < eps):.1f}, ordering kept {ok.mean():.1f}")
    cx.set_yscale("log"); cx.set_ylim(lo_y, hi_y); cx.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]); cx.set_xlim(0.45, KL_RANKS[-1] + 0.5)
    cx.set_xticks(KL_RANKS); cx.set_xlabel("feature dimension $d$")
    cx.set_ylabel("KL divergence at state 2")
    cx.legend(handles=[Line2D([0], [0], marker="o", ls="", color=DFB, ms=3.2, label="ordering kept"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec="#C0392B", mew=0.8, ms=3.2, label="ordering lost")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, handletextpad=0.2, borderaxespad=0.1,
              fontsize=6.3, labelspacing=0.25)
    cx.set_title("(c) Prop. 1: Density-FB accuracy", loc="left", pad=13)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    save(fig, "fig_61_main")
    plt.close(fig)
    return eps


def sweep(rows, arms, c):
    ranks = sorted({x["d"] for x in rows})
    out = {}
    for arm in arms:
        pts = {}
        if arm in ("A", "D"):
            pts[1] = ci95(c[f"gr1_{arm}"])
        for d in ranks:
            v = [float(x["gap_ratio"]) for x in rows if x["arm"] == arm and x["d"] == d]
            if v:
                pts[d] = ci95(v)
        out[arm] = pts
    return out


def draw_sweep(ax, data, arms):
    ax.axhline(1.0, color=EXACT, ls="--", lw=0.8)
    ax.axhline(0.0, color=GRID, lw=0.6)
    for arm in arms:
        name, col, mk = ARM[arm]
        ds = sorted(data[arm]); m = np.array([data[arm][d][0] for d in ds]); e = np.array([data[arm][d][1] for d in ds])
        ax.fill_between(ds, m - e, m + e, color=col, alpha=0.18, lw=0)
        ax.plot(ds, m, color=col, marker=mk, ms=3.2, mfc="white", mew=0.9, lw=1.2, label=name)
    ax.set_xlabel("feature dimension $d$"); ax.set_ylabel("fraction of action-value gap")
    ax.set_ylim(-0.1, 1.15)


def fig_rank(rows, c):
    data = sweep(rows, ["A", "D"], c)
    for arm in "AD":
        print(f"rank sweep {ARM[arm][0]}: " + " ".join(f"{d}:{data[arm][d][0]:.2f}" for d in sorted(data[arm])))
    fig, ax = plt.subplots(figsize=(2.7, 1.75))
    draw_sweep(ax, data, ["A", "D"])
    ax.set_xticks(sorted(data["A"]))
    ax.legend(**(dict(loc="center", bbox_to_anchor=(0.6, 0.5)) if PROCESS == "p9" else dict(loc="lower right")), frameon=False, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    save(fig, "fig_61_rank")
    plt.close(fig)


def fig_ablation(rows, c):
    arms = ["A", "B", "C", "D", "F"]
    data = sweep(rows, arms, c)
    for arm in arms:
        print(f"ablation {ARM[arm][0]}: " + " ".join(f"{d}:{data[arm][d][0]:.2f}" for d in sorted(data[arm])))
    fig, ax = plt.subplots(figsize=(4.4, 2.0))          # = 0.8 \\textwidth, prints at 1:1
    draw_sweep(ax, data, arms)
    ax.set_xticks(sorted(data["A"]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(pad=0.2)
    save(fig, "fig_61_ablation")
    plt.close(fig)



def fig_ablation_main(rows, c):
    """fig_61_ablation_main: the ablation sized for a 0.5 \\textwidth minipage (2.75 in), printed 1:1."""
    arms = ["A", "B", "C", "D", "F"]
    short = {"A": "FB", "B": "FB-norm", "C": "SM-sq", "D": "D-FB", "F": "D-FB-soft"}
    data = sweep(rows, arms, c)
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    ax.axhline(1.0, color=EXACT, ls="--", lw=0.8)
    ax.axhline(0.0, color=GRID, lw=0.6)
    handles = []
    for arm in arms:
        name, col, mk = ARM[arm]
        ds = sorted(data[arm]); m = np.array([data[arm][d][0] for d in ds]); e = np.array([data[arm][d][1] for d in ds])
        ax.fill_between(ds, m - e, m + e, color=col, alpha=0.16, lw=0)
        h, = ax.plot(ds, m, color=col, marker=mk, ms=3.4, mfc="white", mew=0.9, lw=1.3, label=short[arm])
        handles.append(h)
    ax.set_xlabel("feature dimension $d$"); ax.set_ylabel("fraction of value gap")
    ax.set_ylim(-0.08, 1.12); ax.set_xticks(sorted(data["A"])); ax.set_yticks([0, 0.5, 1.0])
    order = [0, 1, 2, 3, 4]
    ax.legend([handles[i] for i in order], [short[arms[i]] for i in order], loc="upper center",
              bbox_to_anchor=(0.42, -0.26), ncol=5, frameon=False, columnspacing=0.55, handlelength=1.1,
              handletextpad=0.25, borderaxespad=0.0, fontsize=7)
    fig.tight_layout(pad=0.15)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"fig_61_ablation_main.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    print("wrote", FIG / "fig_61_ablation_main.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ individual panels (subfigures)
PANEL_SIZE = (1.85, 1.6)
# identical axes box (inches) in every panel so the three subfigures align exactly
PANEL_MARGINS = dict(left=0.47, bottom=0.40, right=0.06, top=0.06)


def panel_fig():
    W, H = PANEL_SIZE; m = PANEL_MARGINS
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = fig.add_axes([m["left"] / W, m["bottom"] / H, (W - m["left"] - m["right"]) / W,
                       (H - m["bottom"] - m["top"]) / H])
    return fig, ax


def panel_style():
    style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
                         "ytick.minor.visible": False, "xtick.minor.visible": False})


def save_panel(fig, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{stem}.{ext}", **kw)          # no tight bbox: keep the fixed geometry
    print("wrote", FIG / f"{stem}.pdf")


def fig_panels(c, rows):
    """fig_61_values / fig_61_lemma / fig_61_prop: the three panels of fig_61_main as separate,
    title-free files sized for three side-by-side subfigures."""
    from matplotlib.ticker import LogLocator, NullLocator
    panel_style()
    T, r, b, n = c["T"], c["r"], int(c["b"]), int(c["n"])
    gap = float((T[2 * b] - T[2 * b + 1]) @ r) / (1 - GAMMA)
    R = float(r.max() - r.min()); eps = (1 - GAMMA) ** 2 * gap ** 2 / (2 * R ** 2)

    # values
    fig, ax = panel_fig()
    sub = [x for x in rows if x["d"] == VAL_RANK]
    groups = [("exact", EXACT, (T[2 * b] @ r / (1 - GAMMA), 0.0), (T[2 * b + 1] @ r / (1 - GAMMA), 0.0))]
    for arm in ("A", "D"):
        name, col, _ = ARM[arm]
        groups.append((name, col, ci95([float(x["q0"]) for x in sub if x["arm"] == arm]),
                       ci95([float(x["q1"]) for x in sub if x["arm"] == arm])))
    w = 0.38
    for i, (name, col, qp, qm) in enumerate(groups):
        ax.bar(i - w / 2, qp[0], w, color=col, yerr=qp[1], ecolor=INK, error_kw=dict(lw=0.7, capsize=1.8))
        ax.bar(i + w / 2, qm[0], w, color=col, alpha=0.4, hatch="////", edgecolor=col, linewidth=0,
               yerr=qm[1], ecolor=INK, error_kw=dict(lw=0.7, capsize=1.8))
    ax.set_xticks(range(3)); ax.set_xticklabels(["exact", "FB", "D-FB"]); ax.set_xlabel("method")
    ax.set_ylabel(f"value at state {b}")
    top = max(max(g[2][0], g[3][0]) for g in groups)
    ax.set_ylim(0, 1.35 * top)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#888888"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#cccccc", hatch="////", edgecolor="#888888", lw=0)],
              labels=["$a_+$", "$a_-$"], loc="upper center", ncol=2, frameon=False, handlelength=1.2,
              borderaxespad=0.1, columnspacing=1.0)
    save_panel(fig, "fig_61_values"); plt.close(fig)

    # lemma
    fig, bx = panel_fig()
    x = np.arange(n); A = [int(v) for v in c.get("A", [3, 4])]
    exact_diff = (T[2 * b] - T[2 * b + 1]) / (1 - GAMMA)
    lim = 1.25 * float(np.abs(exact_diff).max())
    bx.axvspan(A[0] - 0.5, A[-1] + 0.5, color=DFB, alpha=0.12, lw=0)
    bx.axhline(0, color=GRID, lw=0.7)
    bx.bar(x, exact_diff, 0.7, color=EXACT, label="exact", zorder=2)
    dm = c["dfb_diff"].mean(0)
    bx.plot(x, dm, "D", ms=4.2, color=DFB, mec="white", mew=0.5, label="Density-FB", zorder=4)
    bx.plot(x, c["refit"].mean(0), "o", ms=4.6, color=FB, mec=INK, mew=0.5, label="FB (best fit)", zorder=5)
    bx.set_xticks(x); bx.set_xlabel("future state $s^+$")
    bx.set_ylabel("visits: $a_+$ minus $a_-$")
    bx.set_ylim(-lim, lim)
    bx.legend(loc="upper right", frameon=False, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.3)
    save_panel(fig, "fig_61_lemma"); plt.close(fig)

    # proposition
    fig, cx = panel_fig()
    lo_y = 10 ** np.floor(np.log10(max(float(np.min(c["kl"])), 1e-7))) / 2
    hi_y = 10 ** np.ceil(np.log10(float(np.max(c["kl"])))) * 1.5
    cx.axhspan(lo_y, eps, color=DFB, alpha=0.07, lw=0, zorder=0)
    cx.axhline(eps, color=INK, ls="--", lw=0.8, zorder=1)
    cx.text(KL_RANKS[-1] + 0.45, eps * 1.3, "$\\varepsilon$", ha="right", va="bottom", fontsize=9)
    for i, d in enumerate(KL_RANKS):
        v, ok = c["kl"][i], c["order"][i]
        xs = d + np.linspace(-0.22, 0.22, len(v))
        cx.scatter(xs[ok], v[ok], s=14, color=DFB, lw=0, zorder=3)
        cx.scatter(xs[~ok], v[~ok], s=14, facecolors="white", edgecolors="#C0392B", lw=0.9, zorder=4)
    cx.set_yscale("log"); cx.set_ylim(lo_y, hi_y); cx.set_xlim(0.45, KL_RANKS[-1] + 0.55)
    cx.yaxis.set_major_locator(LogLocator(base=10, numticks=12)); cx.yaxis.set_minor_locator(NullLocator())
    # label every other decade when the range is wide
    if np.log10(hi_y / lo_y) > 5:
        cx.yaxis.set_major_locator(LogLocator(base=100, numticks=12))
    cx.set_xticks(KL_RANKS); cx.set_xlabel("feature dimension $d$")
    cx.set_ylabel("KL at state %d" % b)
    cx.legend(handles=[Line2D([0], [0], marker="o", ls="", color=DFB, ms=3.8, label="order kept"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec="#C0392B", mew=0.9, ms=3.8,
                              label="order lost")],
              loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
    save_panel(fig, "fig_61_prop"); plt.close(fig)

if __name__ == "__main__":
    style()
    c = dict(np.load(CACHE)) if CACHE.exists() and "--recompute" not in sys.argv else compute()
    rows = trained_rows(c)
    fig_main(c, rows)
    fig_rank(rows, c)
    fig_ablation(rows, c)
    fig_panels(c, rows)
    fig_ablation_main(rows, c)
