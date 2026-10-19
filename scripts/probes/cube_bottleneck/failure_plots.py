"""Failure figures from failure_figs.py outputs: outcome fractions, TV error per state type, and TV error vs success.

Panel data are written next to each figure as CSV.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("analysis/failure_figs")
FIG = Path("figures/failure")
ENVS = ["cube-single-play-v0"]
ENV_LABEL = {"cube-single-play-v0": "cube-single", "cube-double-play-v0": "cube-double"}
METHODS = ["fb", "density_fb_z"]
METHOD_LABEL = {"fb": "FB", "density_fb_z": "Density-FB"}
# Critic name each method's runs were saved under; probe CSVs carry it in their
# "critic" column and the run directories are named after it.
RUN_NAME = {"fb": "fb_tdj", "density_fb_z": "density_fb_z"}
OUTCOMES = ["approach", "acquisition", "maintain", "transport", "success"]
OUTCOME_LABEL = {
    "approach": "approach failure",
    "acquisition": "acquisition failure",
    "maintain": "maintain failure",
    "transport": "transport failure",
    "success": "success",
}
BANKS = ["approach", "inhand", "carrying"]
BANK_LABEL = {"approach": "approach", "inhand": "in hand", "carrying": "carrying"}

# seaborn "colorblind" palette
CB = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]
# Outcome stack order
OUTCOME_COLOR = {
    "approach": CB[4],
    "acquisition": CB[1],
    "maintain": CB[0],
    "transport": CB[3],
    "success": CB[2],
}
METHOD_COLOR = {"fb": CB[7], "density_fb_z": CB[2]}  # grey vs green
ENV_COLOR = {"cube-single-play-v0": CB[0], "cube-double-play-v0": CB[1]}
METHOD_MARKER = {"fb": "o", "density_fb_z": "^"}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.6,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name, tight=True):
    # tight=False keeps the exact figsize
    FIG.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(FIG / f"{name}.pdf", bbox_inches=bbox)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches=bbox)
    print(f"[fig] {FIG / name}.pdf/.png")


# Shared canvas size for the two failure panels
FAILURE_FIGSIZE = (3.3, 3.7)


# Figure A: outcome fractions


def load_episodes(epoch=1000000):
    dfs = []
    for env in ENVS:
        for f in sorted((ROOT / env / "rollouts").glob(f"episodes_*_{epoch}.csv")):
            dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


def outcome_table(ep):
    per_seed = (
        ep.groupby(["env", "critic", "seed"])
        .outcome.value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=OUTCOMES, fill_value=0.0)
        .reset_index()
    )
    per_seed["n_episodes"] = ep.groupby(["env", "critic", "seed"]).size().values
    mean = per_seed.groupby(["env", "critic"])[OUTCOMES].mean().reset_index()
    return per_seed, mean


def draw_a(ax, mean, legend=True):
    # One stacked column per (env, method)
    rows = [(env, m) for env in ENVS for m in METHODS]
    x = np.arange(len(rows))
    for xi, (env, m) in zip(x, rows):
        r = mean[(mean.env == env) & (mean.critic == RUN_NAME[m])]
        if r.empty:
            continue
        bottom = 0.0
        for o in OUTCOMES:
            v = float(r[o].iloc[0])
            ax.bar(
                xi,
                v,
                bottom=bottom,
                width=0.68,
                color=OUTCOME_COLOR[o],
                edgecolor="white",
                linewidth=0.8,
                label=OUTCOME_LABEL[o] if xi == x[0] else None,
            )
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{ENV_LABEL[e]}\n{METHOD_LABEL[m]}" for e, m in rows], fontsize=6
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of episodes")
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            frameon=False,
            handlelength=1.0,
            columnspacing=1.0,
        )


# Figure B: TV error per state type


def load_pred(epoch=None, banks=BANKS):
    dfs = []
    for env in ENVS:
        for f in sorted((ROOT / env / "predict").glob("pred_*.csv")):
            if f.stem.endswith("_smoke"):
                continue
            d = pd.read_csv(f)
            dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    d = d[d.bank.isin(banks)]
    if epoch is not None:
        d = d[d.epoch == epoch]
    return d


def tv_table(pred):
    """Per (env, bank, method): TV error averaged over seeds per state, then mean and s.e. over states."""
    st = (
        pred.groupby(["env", "bank", "critic", "state"])
        .agg(
            tv=("tv", "mean"),
            goal_mass_pred=("goal_mass_pred", "mean"),
            goal_mass_truth=("goal_mass_truth", "mean"),
            tv_pred_prior=("tv_pred_prior", "mean"),
            tv_truth_prior=("tv_truth_prior", "mean"),
        )
        .reset_index()
    )
    out = (
        st.groupby(["env", "bank", "critic"])
        .agg(
            tv_mean=("tv", "mean"),
            tv_sem=("tv", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            tv_sd=("tv", "std"),
            n_states=("tv", "size"),
            goal_mass_pred=("goal_mass_pred", "mean"),
            goal_mass_truth=("goal_mass_truth", "mean"),
            tv_pred_prior=("tv_pred_prior", "mean"),
            tv_truth_prior=("tv_truth_prior", "mean"),
        )
        .reset_index()
    )
    return st, out


def method_legend(fig, y=-0.02):
    from matplotlib.patches import Patch

    h = [Patch(facecolor=METHOD_COLOR[m], label=METHOD_LABEL[m]) for m in METHODS]
    fig.legend(
        handles=h,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.5,
    )


def draw_b(axes, tab, legend=False):
    w = 0.34
    x = np.arange(len(BANKS))
    for ax, env in zip(axes, ENVS):
        for j, m in enumerate(METHODS):
            r = (
                tab[(tab.env == env) & (tab.critic == RUN_NAME[m])]
                .set_index("bank")
                .reindex(BANKS)
            )
            ax.bar(
                x + (j - 0.5) * w,
                r.tv_mean.values,
                width=w * 0.92,
                color=METHOD_COLOR[m],
                label=METHOD_LABEL[m] if ax is axes[0] else None,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.errorbar(
                x + (j - 0.5) * w,
                r.tv_mean.values,
                yerr=r.tv_sem.values,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.6,
                capsize=1.5,
                capthick=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([BANK_LABEL[b] for b in BANKS])
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"state type ({ENV_LABEL[env]})")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("TV distance")
    if legend:
        axes[0].legend(
            loc="upper right",
            frameon=False,
            handlelength=1.0,
            bbox_to_anchor=(1.0, 1.02),
        )


# Figure B, action-ranking panel

ACTION_METRICS = ["tv_action", "gap_error", "rank_spearman", "regret", "sign_bestworst"]


def load_sign(epoch=1000000):
    """Per-state agreement between the predicted and true sign of the best-vs-worst action gap."""
    rows = []
    for env in ENVS:
        for f in sorted(
            (ROOT / env / "actions").glob(f"actions_long_seed*_{epoch}_*.csv")
        ):
            if f.stem.endswith("_smoke"):
                continue
            rows.append(pd.read_csv(f))
    if not rows:
        return None
    L = pd.concat(rows, ignore_index=True)

    def _sb(g):
        q = g.q_pred.to_numpy(float)
        r = g.return_true.to_numpy(float)
        ib, iw = int(np.argmax(r)), int(np.argmin(r))
        return (
            np.nan
            if r[ib] == r[iw]
            else float(np.sign(q[ib] - q[iw]) == np.sign(r[ib] - r[iw]))
        )

    key = ["env", "seed", "epoch", "bank", "state", "task", "critic"]
    return L.groupby(key).apply(_sb).reset_index(name="sign_bestworst")


def load_actions(epoch=1000000):
    dfs = []
    for env in ENVS:
        for f in sorted((ROOT / env / "actions").glob(f"actions_seed*_{epoch}_*.csv")):
            if f.stem.endswith("_smoke") or "_long_" in f.stem:
                continue
            dfs.append(pd.read_csv(f))
    if not dfs:
        return None
    act = pd.concat(dfs, ignore_index=True)
    sign = load_sign(epoch)
    if sign is not None:
        act = act.merge(
            sign,
            on=["env", "seed", "epoch", "bank", "state", "task", "critic"],
            how="left",
        )
    else:
        act["sign_bestworst"] = np.nan
    return act


def action_table(act):
    """Per (env, bank, method): metrics averaged over seeds per state, then mean and s.e. over states.

    States where every candidate has the same true return (NaN) are dropped.
    """
    st = (
        act.groupby(["env", "bank", "critic", "state"])[ACTION_METRICS]
        .mean()
        .reset_index()
    )
    agg = {}
    for m in ACTION_METRICS:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sem"] = (m, lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()))
        agg[f"{m}_n"] = (m, lambda x: int(x.notna().sum()))
    out = st.groupby(["env", "bank", "critic"]).agg(**agg).reset_index()
    return st, out


def draw_rank(
    axes,
    tab,
    metric="sign_bestworst",
    ylabel="sign agreement",
    ylim=(0.4, 1.0),
    ref=0.5,
    legend=False,
):
    w = 0.34
    x = np.arange(len(BANKS))
    for ax, env in zip(axes, ENVS):
        for j, m in enumerate(METHODS):
            r = (
                tab[(tab.env == env) & (tab.critic == RUN_NAME[m])]
                .set_index("bank")
                .reindex(BANKS)
            )
            ax.bar(
                x + (j - 0.5) * w,
                r[f"{metric}_mean"].values,
                width=w * 0.92,
                color=METHOD_COLOR[m],
                edgecolor="white",
                linewidth=0.6,
                label=METHOD_LABEL[m] if (legend and ax is axes[0]) else None,
            )
            ax.errorbar(
                x + (j - 0.5) * w,
                r[f"{metric}_mean"].values,
                yerr=r[f"{metric}_sem"].values,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.6,
                capsize=1.5,
                capthick=0.6,
            )
        # Chance level for the sign metric
        ax.axhline(ref, color="#999999", linewidth=0.5, ls="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([BANK_LABEL[b] for b in BANKS])
        ax.set_ylim(*ylim)
        ax.set_xlabel(f"state type ({ENV_LABEL[env]})")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel(ylabel)
    if legend:
        axes[0].legend(loc="upper left", frameon=False, handlelength=1.0)


# Figure C: TV error vs success


def success_at(env, critic, seed, step):
    """Return (success, eval rows used): the run's logged success at `step`, mean over tasks.

    Steps between eval points average the two neighbouring evals.
    """
    for root in ("exp", "exp/tdjtab"):
        f = Path(root) / f"{env}_{critic}_flowbc_seed_{seed}" / "eval_relabel.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.task_id.between(1, 5) & d.step.notna()]
        valid = "inference_valid" if "inference_valid" in d else "route_b_valid"
        if valid in d:
            d = d[d[valid] > 0]
        steps = sorted(d.step.unique())
        if step in steps:
            use = [step]
        else:
            lo = [x for x in steps if x < step]
            hi = [x for x in steps if x > step]
            use = ([lo[-1]] if lo else []) + ([hi[0]] if hi else [])
        sub = d[d.step.isin(use)]
        if sub.empty:
            return float("nan"), 0
        per = sub.groupby(["step", "task_id"]).success.mean().groupby("step").mean()
        return float(per.mean()), int(len(sub))
    return float("nan"), 0


def figc_table(pred):
    p = pred[pred.bank == "carrying"]
    g = (
        p.groupby(["env", "critic", "seed", "epoch"])
        .agg(
            tv=("tv", "mean"),
            n_states=("tv", "size"),
            goal_mass_pred=("goal_mass_pred", "mean"),
            goal_mass_truth=("goal_mass_truth", "mean"),
        )
        .reset_index()
    )
    succ, ntask = [], []
    for _, r in g.iterrows():
        s, n = success_at(r.env, r.critic, int(r.seed), int(r.epoch))
        succ.append(s)
        ntask.append(n)
    g["success"] = succ
    g["n_tasks"] = ntask
    return g


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def draw_c(ax, g):
    for env in ENVS:
        for m in METHODS:
            r = g[(g.env == env) & (g.critic == RUN_NAME[m])]
            if r.empty:
                continue
            ax.scatter(
                r.tv,
                r.success,
                s=22,
                marker=METHOD_MARKER[m],
                color=ENV_COLOR[env],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    ok = g.dropna(subset=["tv", "success"])
    a, b = np.polyfit(ok.tv, ok.success, 1)
    xs = np.linspace(ok.tv.min() - 0.02, ok.tv.max() + 0.02, 50)
    ax.plot(xs, a * xs + b, color="#888888", linewidth=1.0, zorder=2)
    ax.set_xlabel("prediction error at carrying states (total variation)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.0)
    from matplotlib.lines import Line2D

    h = [
        Line2D(
            [],
            [],
            marker=METHOD_MARKER[m],
            color="#555555",
            linestyle="none",
            markersize=5,
            label=METHOD_LABEL[m],
        )
        for m in METHODS
    ] + [
        Line2D(
            [],
            [],
            marker="s",
            color=ENV_COLOR[e],
            linestyle="none",
            markersize=5,
            label=ENV_LABEL[e],
        )
        for e in ENVS
    ]
    ax.legend(handles=h, frameon=False, loc="upper right", handlelength=1.0, ncol=2)
    out = dict(
        slope=float(a),
        intercept=float(b),
        spearman=spearman(ok.tv, ok.success),
        pearson=float(np.corrcoef(ok.tv, ok.success)[0, 1]),
        n_points=int(len(ok)),
    )
    for env in ENVS:
        e = ok[ok.env == env]
        out[f"spearman_{ENV_LABEL[env]}"] = spearman(e.tv, e.success)
        out[f"n_{ENV_LABEL[env]}"] = int(len(e))
    fb = ok[ok.critic == RUN_NAME["fb"]]
    out["spearman_fb_only"] = spearman(fb.tv, fb.success)
    return out


# Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["a", "b", "ab", "c"])
    ap.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        help="override FAILURE_FIGSIZE for panel a",
    )
    args = ap.parse_args()
    global FAILURE_FIGSIZE
    if args.figsize:
        FAILURE_FIGSIZE = tuple(args.figsize)
    FIG.mkdir(parents=True, exist_ok=True)

    if any(w in args.which for w in ("a", "ab")):
        ep = load_episodes()
        per_seed, mean = outcome_table(ep)
        per_seed.to_csv(FIG / "figA_outcomes_per_seed.csv", index=False)
        mean.to_csv(FIG / "figA_outcomes_mean.csv", index=False)
        print(mean.round(3).to_string())
        fig, ax = plt.subplots(figsize=FAILURE_FIGSIZE)
        draw_a(ax, mean, legend=False)
        fig.tight_layout(pad=0.3)
        save(fig, "fig_failure_where", tight=False)
        plt.close(fig)

    if any(w in args.which for w in ("b", "ab")):
        pred = load_pred(epoch=1000000)
        st, tab = tv_table(pred)
        st.to_csv(FIG / "figB_tv_per_state.csv", index=False)
        tab.to_csv(FIG / "figB_tv_summary.csv", index=False)
        print(tab.round(3).to_string())
        act = load_actions()
        atab = None
        if act is not None:
            ast, atab = action_table(act)
            ast.to_csv(FIG / "figB_actions_per_state.csv", index=False)
            atab.to_csv(FIG / "figB_actions_summary.csv", index=False)
            print(atab.round(3).to_string())
        ne = len(ENVS)
        if atab is None:
            fig, axes = plt.subplots(
                1, ne, figsize=FAILURE_FIGSIZE, sharey=True, squeeze=False
            )
            draw_b(axes[0], tab, legend=True)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            fig, axes = plt.subplots(
                2, ne, figsize=FAILURE_FIGSIZE, sharey="row", squeeze=False
            )
            draw_b(axes[0], tab)
            draw_rank(axes[1], atab)
            for ax in axes[0]:
                ax.set_xlabel("")
            fig.tight_layout(rect=[0, 0.06, 1, 1])  # leave a strip for the legend
            method_legend(fig, y=0.01)
        save(fig, "fig_failure_why", tight=False)
        plt.close(fig)

    if "ab" in args.which:
        if atab is None:
            fig = plt.figure(figsize=(11.0, 2.4))
            gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.12)
            axa = fig.add_subplot(gs[0])
            axb = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
            axb[1].sharey(axb[0])
            draw_a(axa, mean)
            draw_b(axb, tab)
            plt.setp(axb[1].get_yticklabels(), visible=False)
        else:
            fig = plt.figure(figsize=(11.0, 4.2))
            gs = fig.add_gridspec(
                2, 3, width_ratios=[1.25, 1, 1], wspace=0.28, hspace=0.45
            )
            axa = fig.add_subplot(gs[:, 0])
            axb = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
            axr = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
            axb[1].sharey(axb[0])
            axr[1].sharey(axr[0])
            draw_a(axa, mean)
            draw_b(axb, tab)
            draw_rank(axr, atab)
            for ax in axb:
                ax.set_xlabel("")
            plt.setp(axb[1].get_yticklabels(), visible=False)
            plt.setp(axr[1].get_yticklabels(), visible=False)
            method_legend(fig, y=-0.04)
        save(fig, "fig_failure_where_why")
        plt.close(fig)

    if "c" in args.which:
        pred = load_pred(banks=["carrying"])
        g = figc_table(pred)
        g.to_csv(FIG / "figC_tv_vs_success.csv", index=False)
        print(g.round(3).to_string())
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        stats = draw_c(ax, g)
        save(fig, "fig_failure_tracks")
        plt.close(fig)
        (FIG / "figC_stats.json").write_text(json.dumps(stats, indent=2))
        print("[figC]", stats)


if __name__ == "__main__":
    main()
