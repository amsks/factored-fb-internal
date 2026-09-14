"""Aggregate the reward-family eval (scripts/reward_families/eval_families.py) into
one CSV, the figure fig_reward_families.{pdf,png}, the LaTeX table
PAPER/ICLR/baselines/reward_families.tex and per-seed markdown tables for the note.

  python scripts/reward_families/aggregate.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reward_families.eval_families import PRETTY, GAMMA, SHIFT, N_INFER, RADIUS, REGIONS  # noqa: E402

DATA = ROOT / "analysis/reward_families"
FIG_DIR = ROOT / "PAPER/ICLR/figures/reward_families"
TEX = ROOT / "PAPER/ICLR/baselines/reward_families.tex"
ORDER = ["fb_tdj", "density_fb_z", "hilp_tdjc", "hilp_density_reg", "rldp", "rldp_density_reg"]
FLOORS = ["random", "bc"]
ENVS = ["cube-single-play-v0", "cube-double-play-v0"]
SHORT_ENV = {"cube-single-play-v0": "cube-single", "cube-double-play-v0": "cube-double"}

# figure/table families (display name -> (env, json family key(s), metric key))
MAIN = [
    ("R1 region", "cube-single-play-v0", ["R1_region"], "mean_reward"),
    ("R2 either", "cube-single-play-v0", ["R2_either"], "mean_reward"),
    ("R4 lift", "cube-single-play-v0", ["R4_lift"], "mean_reward"),
    ("R5 dense", "cube-single-play-v0", [f"R5_dense_t{t}" for t in range(1, 6)], "mean_reward"),
    ("R6 move+keep", "cube-double-play-v0", ["R6_move_keep"], "mean_reward"),
    ("R7 either", "cube-double-play-v0", ["R7_either_cube"], "mean_reward"),
]


def load():
    """{(env, method, seed): results-json}"""
    out = {}
    for p in sorted(glob.glob(str(DATA / "*" / "*.json"))):
        if p.endswith("_smoke.json"):
            continue
        d = json.load(open(p))
        out[(d["env"], d["method"], int(d["seed"]))] = d
    return out


def per_seed_metrics(rec):
    """One family record -> {metric: scalar} (mean over the record's episodes)."""
    m = {}
    if "mean_reward" not in rec:
        m["route_b_valid"] = 0.0
        return m
    for k in ("mean_reward", "discounted_return", "undiscounted_return", "final_reward"):
        m[k] = float(np.mean(rec[k]))
    m["mean_reward_ep_std"] = float(np.std(rec["mean_reward"]))
    for k, v in rec.items():
        if k.startswith("aux_"):
            m[k[4:]] = float(np.mean(v))
    for k in ("route_b_valid", "inferred_cond_norm", "inferred_cond_cos_neutral", "label_mean",
              "label_std", "label_pos_frac", "label_raw_mean"):
        if k in rec:
            m[k] = float(rec[k])
    m["T"] = float(rec["T"])
    return m


def build_long(runs):
    """Long rows: env, family, method, seed, metric, value. R5 sub-families are also folded
    into R5_dense / R5_sparse (mean over the 5 targets, 100 episodes per seed)."""
    rows = []
    for (env, method, seed), d in runs.items():
        fold = defaultdict(list)
        for fam, rec in d["families"].items():
            ms = per_seed_metrics(rec)
            for k, v in ms.items():
                rows.append(dict(env=env, family=fam, method=method, method_label=PRETTY[method],
                                 seed=seed, metric=k, value=v, episodes=d["episodes"]))
            if fam.startswith("R5_"):
                key = "R5_dense" if "dense" in fam else "R5_sparse"
                for k, v in ms.items():
                    fold[(key, k)].append(v)
        for (key, k), vs in fold.items():
            rows.append(dict(env=env, family=key, method=method, method_label=PRETTY[method],
                             seed=seed, metric=k, value=float(np.mean(vs)),
                             episodes=d["episodes"] * len(vs)))
    return rows


def table(rows, env, family, metric, methods):
    """{method: (mean, std, [per-seed])} over seeds."""
    out = {}
    for m in methods:
        vs = [r["value"] for r in rows if r["env"] == env and r["family"] == family
              and r["method"] == m and r["metric"] == metric]
        if vs:
            out[m] = (float(np.mean(vs)), float(np.std(vs)) if len(vs) > 1 else float("nan"), vs)
    return out


def fmt(ms, m, scale=1.0, nd=2):
    if m not in ms:
        return "---"
    mu, sd, vs = ms[m]
    s = f"{mu * scale:.{nd}f}"
    if len(vs) > 1:
        s += f"\\pmse{{{sd * scale:.{nd}f}}}"
    return f"\\ckptM{{{s}}}"


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        pal = sns.color_palette("colorblind", 8)
    except ImportError:   # seaborn's "colorblind" palette, hex-for-hex (not installed in the jax venv)
        pal = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494"]
    col = {m: pal[i] for i, m in enumerate(ORDER)}
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6),
                                  gridspec_kw=dict(width_ratios=[2.5, 1.0], wspace=0.35))

    n_m = len(ORDER)
    w = 0.8 / n_m
    xt, xl = [], []
    handles = {}
    for fi, (name, env, keys, metric) in enumerate(MAIN):
        fam_key = keys[0] if len(keys) == 1 else "R5_dense"
        ms = table(rows, env, fam_key, metric, ORDER + FLOORS)
        for mi, m in enumerate(ORDER):
            if m not in ms:
                continue
            x = fi + (mi - (n_m - 1) / 2) * w
            mu, sd, vs = ms[m]
            h = ax.bar(x, mu, w * 0.92, color=col[m], yerr=(sd if np.isfinite(sd) else 0), capsize=1.2,
                       error_kw=dict(lw=0.6), zorder=3)
            handles.setdefault(PRETTY[m], h)
        if "random" in ms:
            h = ax.hlines(ms["random"][0], fi - 0.42, fi + 0.42, colors="k", linestyles="--", lw=0.8, zorder=4)
            handles.setdefault("random floor", h)
        if "bc" in ms:
            h = ax.hlines(ms["bc"][0], fi - 0.42, fi + 0.42, colors="0.45", linestyles=":", lw=0.9, zorder=4)
            handles.setdefault("BC flow (no value term)", h)
        xt.append(fi)
        xl.append(name.replace(" ", "\n", 1))
    n_single = sum(1 for _n, e, _k, _m in MAIN if "single" in e)
    ax.axvspan(n_single - 0.5, len(MAIN) - 0.5, color="0.94", zorder=0, lw=0)
    ax.text(n_single - 0.45, ax.get_ylim()[1] * 0.98, "cube-double", ha="left", va="top", fontsize=6, color="0.35")
    ax.text(-0.45, ax.get_ylim()[1] * 0.98, "cube-single", ha="left", va="top", fontsize=6, color="0.35")
    ax.axhline(0, color="0.7", lw=0.5, zorder=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_xlim(-0.5, len(MAIN) - 0.5)
    ax.set_ylabel("mean per-step reward")
    ax.set_title("non-goal state rewards, frozen 1M checkpoints", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)

    # panel 2: preference flip
    env = "cube-single-play-v0"
    xs = np.arange(len(ORDER))
    wb = 0.2
    pa, pb = pal[0], pal[3]
    h2 = {}
    for i, m in enumerate(ORDER):
        A_a = table(rows, env, "R3_pref_a", "in_A_mean", [m]).get(m)
        B_a = table(rows, env, "R3_pref_a", "in_B_mean", [m]).get(m)
        A_b = table(rows, env, "R3_pref_b", "in_A_mean", [m]).get(m)
        B_b = table(rows, env, "R3_pref_b", "in_B_mean", [m]).get(m)
        if A_a is None:
            continue
        for j, (v, c, hh, lab) in enumerate([(A_a, pa, "", "in A under $r_a$"), (B_a, pa, "////", "in B under $r_a$"),
                                              (A_b, pb, "", "in A under $r_b$"), (B_b, pb, "////", "in B under $r_b$")]):
            h = ax2.bar(xs[i] + (j - 1.5) * wb, v[0], wb * 0.95, color=c, hatch=hh, edgecolor="white", lw=0.3,
                        yerr=(v[1] if np.isfinite(v[1]) else 0), capsize=1, error_kw=dict(lw=0.5), zorder=3)
            h2.setdefault(lab, h)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([PRETTY[m].replace("Density-", "D-") for m in ORDER], rotation=45, ha="right")
    ax2.set_ylabel("fraction of steps in region")
    ax2.set_title("R3 preference flip", loc="left")
    ax2.legend(list(h2.values()), list(h2.keys()), frameon=False, handlelength=1.2, loc="upper right", fontsize=5.5, labelspacing=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax2.set_ylim(0, 1.0)

    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.2), handlelength=1.4, columnspacing=1.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_reward_families.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


TEX_ROWS = [
    # (label, env, family, metric, scale, nd)
    ("R1 region", "cube-single-play-v0", "R1_region", "mean_reward", 1, 2),
    ("R2 either of two", "cube-single-play-v0", "R2_either", "mean_reward", 1, 2),
    ("R3 pref.\\ $r_a$: time in A / B", "cube-single-play-v0", "R3_pref_a", ("in_A_mean", "in_B_mean"), 1, 2),
    ("R3 pref.\\ $r_b$: time in A / B", "cube-single-play-v0", "R3_pref_b", ("in_A_mean", "in_B_mean"), 1, 2),
    ("R4 lift", "cube-single-play-v0", "R4_lift", "mean_reward", 1, 2),
    ("R5 dense ($-$dist, m), dense $z$", "cube-single-play-v0", "R5_dense", "mean_reward", 1, 3),
    ("R5 success (\\%), dense $z$", "cube-single-play-v0", "R5_dense", "hit_any", 100, 0),
    ("R5 dense ($-$dist, m), sparse $z$", "cube-single-play-v0", "R5_sparse", "dense_mean", 1, 3),
    ("R5 success (\\%), sparse $z$", "cube-single-play-v0", "R5_sparse", "hit_any", 100, 0),
    ("R6 move one, keep other", "cube-double-play-v0", "R6_move_keep", "mean_reward", 1, 2),
    ("R7 either cube in A", "cube-double-play-v0", "R7_either_cube", "mean_reward", 1, 2),
]


def make_tex(rows):
    cols = ORDER + FLOORS
    head = " & ".join(["FB", "Density-FB", "HILP", "Density-HILP", "RLDP", "Density-RLDP", "Random", "BC"])
    lines = [
        "% Reward families on the frozen 1M cube checkpoints (evaluation only). Generated by",
        "% scripts/reward_families/aggregate.py from analysis/reward_families/; do not edit by hand.",
        "\\InputIfFileExists{baselines/colors.tex}{}{\\input{colors}}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Zero-shot control of state rewards that are not goal-reaching, on the same frozen 1M OGBench cube checkpoints as the reward-inference column of the main table. Cells are the requested reward's own mean per-step return over 20 full-length episodes from the env's own resets (R3: fraction of steps in each region; R5: also the success rate, any step within 0.04\\,m). Mean $\\pm$ std over 3 seeds. Random = uniform actions; BC = the checkpoints' behaviour-cloning flow policy with no value term.}",
        "\\label{tab:reward_families}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        " & " + head + " \\\\",
        "\\midrule",
        "\\multicolumn{" + str(len(cols) + 1) + "}{l}{\\textit{cube-single}} \\\\",
    ]
    cur_env = "cube-single-play-v0"
    for label, env, fam, metric, scale, nd in TEX_ROWS:
        if env != cur_env:
            lines.append("\\midrule")
            lines.append("\\multicolumn{" + str(len(cols) + 1) + "}{l}{\\textit{cube-double}} \\\\")
            cur_env = env
        if isinstance(metric, tuple):
            t1 = table(rows, env, fam, metric[0], cols)
            t2 = table(rows, env, fam, metric[1], cols)
            cells = []
            for m in cols:
                if m not in t1:
                    cells.append("---")
                else:
                    cells.append(f"\\ckptM{{{t1[m][0] * scale:.{nd}f} / {t2[m][0] * scale:.{nd}f}}}")
        else:
            t = table(rows, env, fam, metric, cols)
            cells = [fmt(t, m, scale, nd) for m in cols]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\par\\vspace{2pt}",
        "\\begin{minipage}{\\linewidth}\\scriptsize\\raggedright",
        f"Protocol: for every (family, method, seed) the task index $z$ is inferred from {N_INFER:,} random training transitions labelled with the family's reward on the next state (+{SHIFT} shift, as in the table protocol), through each method's own \\texttt{{infer\\_cond}}; the frozen policy is then rolled for 20 episodes of the env's standard length (200 / 500 steps, no early termination) and the family's reward is scored on every visited state. "
        f"Regions are discs of radius {RADIUS}\\,m in the table plane: cube-single A~$=({REGIONS['cube-single-play-v0']['A'][0]}, {REGIONS['cube-single-play-v0']['A'][1]})$, B~$=({REGIONS['cube-single-play-v0']['B'][0]}, {REGIONS['cube-single-play-v0']['B'][1]})$; cube-double A~$=({REGIONS['cube-double-play-v0']['A'][0]}, {REGIONS['cube-double-play-v0']['A'][1]})$, B$_0$~$=({REGIONS['cube-double-play-v0']['B0'][0]}, {REGIONS['cube-double-play-v0']['B0'][1]})$ (cube 2's start under the fixed task-2 reset used for R6). "
        "R1: $r=\\mathbf{1}[\\text{cube}\\in A]$. R2: $r=\\max(\\mathbf{1}[A],\\mathbf{1}[B])$. R3: $r_a=\\mathbf{1}[A]+0.3\\,\\mathbf{1}[B]$, $r_b=0.3\\,\\mathbf{1}[A]+\\mathbf{1}[B]$. R4: $r=\\mathrm{clip}((z_{\\text{cube}}-0.02)/0.10,0,1)$. R5: $r=-\\|\\text{cube}-g\\|$ with $g$ each eval task's goal (5 targets $\\times$ 4 episodes), against the sparse $r=\\mathbf{1}[\\|\\text{cube}-g\\|\\le 0.04]$ on the same targets. R6: $r=\\mathbf{1}[\\text{cube}_1\\in A]-0.5\\,\\mathbf{1}[\\text{cube}_2\\notin B_0]$. R7: $r=\\max(\\mathbf{1}[\\text{cube}_1\\in A],\\mathbf{1}[\\text{cube}_2\\in A])$. "
        "Checkpoints: FB = fb\\_tdj, Density-FB = density\\_fb\\_z, HILP = hilp\\_tdjc, RLDP = rldp at ortho 1 (exp/ortho1), Density-RLDP = rldp\\_density\\_reg (ortho 1), Density-HILP = hilp\\_density\\_reg; the two density retrofits have no cube-double run (``---'').",
        "\\end{minipage}",
        "\\end{table}",
    ]
    TEX.write_text("\n".join(lines) + "\n")


def md_tables(rows):
    """Per-seed markdown tables for the note."""
    out = []
    for label, env, fam, metric, scale, nd in TEX_ROWS:
        mets = metric if isinstance(metric, tuple) else (metric,)
        for met in mets:
            t = table(rows, env, fam, met, ORDER + FLOORS)
            if not t:
                continue
            out.append(f"\n**{label} -- `{fam}` / `{met}`** ({SHORT_ENV[env]}, values x{scale})\n")
            out.append("| method | seed 0 | seed 1 | seed 2 | mean | std |")
            out.append("|---|---|---|---|---|---|")
            for m in ORDER + FLOORS:
                if m not in t:
                    continue
                mu, sd, vs = t[m]
                cells = [f"{v * scale:.{nd + 1}f}" for v in vs] + [""] * (3 - len(vs))
                out.append(f"| {PRETTY[m]} | " + " | ".join(cells) + f" | {mu * scale:.{nd + 1}f} | {sd * scale:.{nd + 1}f} |")
    # inference witnesses
    out.append("\n**Reward-inference witnesses (route_b_valid, ||z||, cos to the neutral 1_d, label mean / positive fraction), seed 0**\n")
    out.append("| env | family | method | valid | norm | cos1 | label mean | pos frac |")
    out.append("|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["seed"] != 0 or r["method"] in FLOORS or r["metric"] != "inferred_cond_norm":
            continue
        key = (r["env"], r["family"], r["method"])
        if key in seen or r["family"].startswith("R5_") and r["family"] not in ("R5_dense_t1", "R5_sparse_t1"):
            continue
        seen.add(key)
        g = lambda k: next((x["value"] for x in rows if x["env"] == r["env"] and x["family"] == r["family"]
                            and x["method"] == r["method"] and x["seed"] == 0 and x["metric"] == k), float("nan"))
        out.append(f"| {SHORT_ENV[r['env']]} | {r['family']} | {PRETTY[r['method']]} | {int(g('route_b_valid'))} | "
                   f"{r['value']:.2f} | {g('inferred_cond_cos_neutral'):.2f} | {g('label_mean'):.3f} | {g('label_pos_frac'):.3f} |")
    return "\n".join(out)


def main():
    import csv
    runs = load()
    rows = build_long(runs)
    DATA.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (DATA / "reward_families.csv", FIG_DIR / "reward_families.csv"):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["env", "family", "method", "method_label", "seed", "metric", "value", "episodes"])
            w.writeheader()
            w.writerows(rows)
    make_figure(rows)
    make_tex(rows)
    (DATA / "per_seed_tables.md").write_text(md_tables(rows))
    # console summary
    print(f"{len(runs)} runs, {len(rows)} rows")
    for label, env, fam, metric, scale, nd in TEX_ROWS:
        mets = metric if isinstance(metric, tuple) else (metric,)
        for met in mets:
            t = table(rows, env, fam, met, ORDER + FLOORS)
            print(f"{label:38s} {met:12s} " + "  ".join(f"{PRETTY[m]}={t[m][0] * scale:.{nd + 1}f}±{t[m][1] * scale:.{nd + 1}f}" for m in ORDER + FLOORS if m in t))


if __name__ == "__main__":
    main()
