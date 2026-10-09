"""Where the action gap lives in the spectrum of X = nM, and what the cluster-A reward looks like
through rank-d features (P6 and P11, closed form on the exact successor matrix + TD-trained FB).

Left panels: singular directions v_k of X ordered by singular value; two bars per k: the share of
the matrix's squared mass s_k^2 / sum s^2 and the share of the bottleneck row difference
dv = X[2b] - X[2b+1] carried by v_k, (v_k . dv)^2 / |dv|^2.
Right panels: the reward r = 1_A projected onto the top-d singular span, P_d r, as bars over
states for d = 2, 3, 4 (cluster A shaded green, cluster B blue), and the same projection onto
span(B) of the TD-trained FB features at ranks 2 and 4 (mean over 10 runs).

    JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/gap_direction.py
Outputs: PAPER/ICLR/figures/single_bottleneck/fig_gap_direction.{pdf,png}, analysis/single_bottleneck/gap_direction.json
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)
fam, lc, single = sb.fam, sb.lc, sb.single

RANKS_PROJ = [2, 3, 4]
RANKS_TRAINED = [2, 4]


def spectrum(mdp):
    X = mdp["X"]
    b = mdp["decision"]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    dv = X[2 * b] - X[2 * b + 1]
    mass = s ** 2 / np.sum(s ** 2)
    gap_share = (Vt @ dv) ** 2 / (dv @ dv)
    return s, Vt, mass, gap_share


def projected_reward(Bspan, r):
    return lc.projector(Bspan) @ r


def main():
    fixed = sb.fixed_processes()
    out, results = {}, {}
    for mdp in fixed:
        sb.check(mdp)
        name, r, n = mdp["variant"], mdp["r"], mdp["n"]
        s, Vt, mass, gap_share = spectrum(mdp)
        proj = {d: projected_reward(Vt[:d].T, r) for d in RANKS_PROJ}
        trained = {}
        for d in RANKS_TRAINED:
            cfg, pred, F_all, B_all = fam.train_group(mdp, d, "fb", sb.SEEDS, sb.RATES, steps=sb.TRAIN_STEPS)
            trained[d] = np.mean([projected_reward(B_all[i], r) for i in range(len(cfg))], axis=0)
            frac = [lc.lemma_gap(mdp, B_all[i], r) / mdp["gap"] for i in range(len(cfg))]
            results.setdefault(name, {})[f"trained_rank{d}_gap_fraction_mean"] = float(np.mean(frac))
        out[name] = dict(mdp=mdp, s=s, mass=mass, gap_share=gap_share, proj=proj, trained=trained)
        A, B = mdp["clusterA"], mdp["clusterB"]
        results[name].update(
            singular_values=s.tolist(), mass_share=mass.tolist(), gap_share=gap_share.tolist(),
            cum_gap_share={d: float(gap_share[:d].sum()) for d in range(1, n + 1)},
            projected_reward={d: proj[d].tolist() for d in RANKS_PROJ},
            projected_reward_trained={d: trained[d].tolist() for d in RANKS_TRAINED},
            A_states=A, B_states=B)
        print(f"{name}: mass share {np.round(mass[:5], 3)}, gap share {np.round(gap_share[:5], 3)}")
        for d in RANKS_PROJ:
            print(f"   P_{d} r: A {np.round(proj[d][A], 3)} B {np.round(proj[d][B], 3)}  others {np.round(np.delete(proj[d], A + B), 3)}")
        for d in RANKS_TRAINED:
            print(f"   trained rank {d}: A {np.round(trained[d][A], 3)} B {np.round(trained[d][B], 3)} "
                  f"(gap fraction {results[name][f'trained_rank{d}_gap_fraction_mean']:.3f})")
    (HERE / "gap_direction.json").write_text(json.dumps(results, indent=2, default=float))
    make_figure(out)


def make_figure(out):
    plt = sb._rc()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    INK = single.INK
    names = ["P11"]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), gridspec_kw=dict(width_ratios=[1.0, 1.35]))
    for row, name in enumerate(names):
        o = out[name]
        mdp, n = o["mdp"], o["mdp"]["n"]
        A, B, b = mdp["clusterA"], mdp["clusterB"], mdp["decision"]
        # left: spectrum
        ax = axes[row, 0]
        k = np.arange(1, n + 1)
        w = 0.38
        ax.bar(k - w / 2, o["mass"], width=w, color="#9a9a9a", label="share of $\\|X\\|_F^2$ ($s_k^2/\\sum s^2$)", zorder=2)
        ax.bar(k + w / 2, o["gap_share"], width=w, color=single.ARM_COLOR["A"], label="share of $\\|\\Delta v\\|^2$ ($(v_k\\cdot\\Delta v)^2/\\|\\Delta v\\|^2$)", zorder=2)
        ax.set_xticks(k)
        ax.set_xlabel("singular direction $k$ of $X = nM$", fontsize=8.5)
        ax.set_ylabel("share", fontsize=8.5)
        ax.set_ylim(0, 1.0)
        pass
        ax.tick_params(labelsize=8, length=2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.legend(fontsize=6.4, frameon=False, loc="upper right", handlelength=1.2)
        # right: projected reward over states
        ax = axes[row, 1]
        for s_ in A:
            ax.axvspan(s_ - 0.5, s_ + 0.5, color="#029E73", alpha=0.12, linewidth=0, zorder=0)
        for s_ in B:
            ax.axvspan(s_ - 0.5, s_ + 0.5, color="#0173B2", alpha=0.12, linewidth=0, zorder=0)
        ax.axvspan(b - 0.5, b + 0.5, color="#DE8F05", alpha=0.10, linewidth=0, zorder=0)
        series = [(f"$d={d}$, SVD span", o["proj"][d], plt.get_cmap("Greys")(0.35 + 0.2 * i), "") for i, d in enumerate(RANKS_PROJ)]
        series += [(f"$d={d}$, trained FB", o["trained"][d], single.ARM_COLOR["A"], "///" if d == 2 else "") for d in RANKS_TRAINED]
        m = len(series)
        w = 0.8 / m
        x = np.arange(n)
        for j, (lab, vals, col, hatch) in enumerate(series):
            ax.bar(x + (j - (m - 1) / 2) * w, vals, width=w, color=col, edgecolor=col if not hatch else "white",
                   hatch=hatch, linewidth=0.4, label=lab, zorder=2, alpha=0.85 if hatch else 1.0)
        ax.plot(x, mdp["r"], linestyle="none", marker="_", color=INK, markersize=9, markeredgewidth=1.0, zorder=3, label="$r = 1_A$")
        ax.axhline(0, color="#cfcfcf", linewidth=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xlabel("state (bottleneck orange, cluster A green, cluster B blue)", fontsize=8)
        ax.set_ylabel("$\\Pi_U\\, r$", fontsize=8.5)
        pass
        ax.tick_params(labelsize=8, length=2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.legend(fontsize=6.0, frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8, labelspacing=0.3)
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.4)
    sb.save(fig, "fig_gap_direction")
    plt.close(fig)


if __name__ == "__main__":
    main()
