"""Markdown tables for docs/notes/2026-09-12-failure-figures.md, from the CSVs next to the
figures (PAPER/ICLR/figures/failure/). Run after failure_plots.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIG = Path("PAPER/ICLR/figures/failure")
ENV_LABEL = {"cube-single-play-v0": "cube-single", "cube-double-play-v0": "cube-double"}
METHOD_LABEL = {"fb_tdj": "FB", "density_fb_z": "Density-FB"}
BANK_LABEL = {"approach": "approach", "inhand": "in-hand", "carrying": "carrying"}
OUTCOMES = ["approach", "acquisition", "maintain", "transport", "success"]


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    print("### Figure A: outcome fractions (mean over 3 seeds, 100 episodes per seed)\n")
    m = pd.read_csv(FIG / "figA_outcomes_mean.csv")
    m["env"] = m.env.map(ENV_LABEL); m["method"] = m.critic.map(METHOD_LABEL)
    print(md(m[["env", "method"] + OUTCOMES]))
    print("\nPer seed:\n")
    ps = pd.read_csv(FIG / "figA_outcomes_per_seed.csv")
    ps["env"] = ps.env.map(ENV_LABEL); ps["method"] = ps.critic.map(METHOD_LABEL)
    print(md(ps[["env", "method", "seed"] + OUTCOMES + ["n_episodes"]]))

    print("\n### Figure B (top row): state-level prediction error, mean +- s.e.m. over 64 states\n")
    t = pd.read_csv(FIG / "figB_tv_summary.csv")
    t["env"] = t.env.map(ENV_LABEL); t["method"] = t.critic.map(METHOD_LABEL); t["state type"] = t.bank.map(BANK_LABEL)
    t["TV error"] = t.apply(lambda r: f"{r.tv_mean:.3f} +- {r.tv_sem:.3f}", axis=1)
    t = t.rename(columns={"goal_mass_pred": "goal mass (pred)", "goal_mass_truth": "goal mass (truth)",
                          "tv_pred_prior": "TV pred vs prior", "tv_truth_prior": "TV truth vs prior"})
    print(md(t[["env", "state type", "method", "TV error", "goal mass (pred)", "goal mass (truth)",
               "TV pred vs prior", "TV truth vs prior"]]))

    f = FIG / "figB_actions_summary.csv"
    if f.exists():
        print("\n### Figure B (bottom row) and the action-conditioned probe: mean +- s.e.m. over states (n = states with a non-degenerate true-return range)\n")
        a = pd.read_csv(f)
        a["env"] = a.env.map(ENV_LABEL); a["method"] = a.critic.map(METHOD_LABEL); a["state type"] = a.bank.map(BANK_LABEL)
        for k, lab in [("tv_action", "occupancy TV (action-conditioned)"), ("gap_error", "action-gap error"),
                       ("rank_spearman", "ranking accuracy (Spearman)"), ("regret", "decision regret")]:
            a[lab] = a.apply(lambda r: f"{r[k + '_mean']:.3f} +- {r[k + '_sem']:.3f} (n={int(r[k + '_n'])})", axis=1)
        print(md(a[["env", "state type", "method", "occupancy TV (action-conditioned)", "action-gap error",
                   "ranking accuracy (Spearman)", "decision regret"]]))

    print("\n### Figure C: carrying-state TV error vs success per checkpoint\n")
    c = pd.read_csv(FIG / "figC_tv_vs_success.csv")
    c["env"] = c.env.map(ENV_LABEL); c["method"] = c.critic.map(METHOD_LABEL)
    c = c.rename(columns={"epoch": "step", "tv": "TV error (carrying)", "success": "success (eval_relabel)"})
    print(md(c[["env", "method", "seed", "step", "TV error (carrying)", "success (eval_relabel)"]]))
    st = json.loads((FIG / "figC_stats.json").read_text())
    print("\nFit and correlations: " + ", ".join(f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}" for k, v in st.items()))


if __name__ == "__main__":
    main()
