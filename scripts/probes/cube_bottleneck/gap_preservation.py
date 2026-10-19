"""Action-gap figures: per phase, the true Monte-Carlo gap between best and worst candidate action and how often the critic ranks them correctly.

Reads the cached action probe of failure_figs.py. Run with no flags to aggregate, --smoke for a
small end-to-end check, or --probe to regenerate probe rows first.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ENV = "cube-single-play-v0"
EPOCH = 1000000
ROOT = Path("analysis/failure_figs")
FIG = Path("figures/gap_preservation")
METHODS = ["fb", "density_fb_z"]
METHOD_LABEL = {"fb": "FB", "density_fb_z": "Density-FB"}
# Critic name each method's runs were saved under; probe CSVs carry it in their
# "critic" column and the run directories are named after it.
RUN_NAME = {"fb": "fb_tdj", "density_fb_z": "density_fb_z"}
METHOD_OF = {run: m for m, run in RUN_NAME.items()}
BANKS = ["approach", "inhand", "carrying"]
N_BINS = 5
STATE_KEYS = ["critic", "seed", "bank", "state"]


# Aggregation


def calibration_slopes(long: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """Return {(critic, seed): beta}, the slope of true return on predicted value, both centred per state."""
    out = {}
    for (c, s), g in long.groupby(["critic", "seed"]):
        qc = g["q_pred"] - g.groupby(["bank", "state"])["q_pred"].transform("mean")
        rc = g["return_true"] - g.groupby(["bank", "state"])["return_true"].transform(
            "mean"
        )
        den = float((qc**2).sum())
        out[(str(c), int(s))] = (
            float((qc * rc).sum() / den) if den > 0 else float("nan")
        )
    return out


def per_state_gaps(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (critic, seed, bank, state) with the true gap, the critic's gap at the same pair, and y_rel.

    States where all candidates tie in return get tie=1.
    """
    rows = []
    for (c, s, b, i), g in long.groupby(STATE_KEYS, sort=False):
        q = g["q_pred"].to_numpy(float)
        r = g["return_true"].to_numpy(float)
        cand = g["candidate"].to_numpy(object)
        ib, iw = int(np.argmax(r)), int(np.argmin(r))
        dmc = float(r[ib] - r[iw])
        dhat = float(q[ib] - q[iw])
        qr = float(np.ptp(q))
        rows.append(
            dict(
                critic=c,
                seed=int(s),
                bank=b,
                state=int(i),
                task=int(g["task"].iloc[0]) if "task" in g else -1,
                n_candidates=len(g),
                best=cand[ib],
                worst=cand[iw],
                return_best=float(r[ib]),
                return_worst=float(r[iw]),
                delta_mc=dmc,
                delta_hat=dhat,
                q_range=qr,
                y_rel=dhat / qr if qr > 0 else float("nan"),
                reversed=int(dhat < 0),
                tie=int(dmc <= 0),
            )
        )
    return pd.DataFrame(rows)


def attach_ratio(
    per_state: pd.DataFrame, slopes: Dict[Tuple[str, int], float]
) -> pd.DataFrame:
    """Add beta, y_raw = Delta_hat / Delta_MC and y = beta * y_raw (NaN at ties)."""
    p = per_state.copy()
    p["beta"] = [slopes[(c, int(s))] for c, s in zip(p["critic"], p["seed"])]
    with np.errstate(divide="ignore", invalid="ignore"):
        p["y_raw"] = np.where(p["delta_mc"] > 0, p["delta_hat"] / p["delta_mc"], np.nan)
    p["y"] = p["beta"] * p["y_raw"]
    return p


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bins by rank; returns ints in [0, n_bins)."""
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="stable")
    b = np.empty(n, int)
    b[order] = (np.arange(n) * n_bins) // max(n, 1)
    return b


def bin_table(
    per_state: pd.DataFrame, n_bins: int = N_BINS, cols: Sequence[str] = ("y", "y_rel")
) -> pd.DataFrame:
    """Per (critic, bin): true-gap range and mean, mean/s.e./quantiles of each column, and fraction reversed."""
    rows = []
    p = per_state[per_state["tie"] == 0]
    for c, g in p.groupby("critic", sort=False):
        k = min(n_bins, len(g))
        b = quantile_bins(g["delta_mc"].to_numpy(), k)
        for j in range(k):
            h = g[b == j]
            row = dict(
                critic=c,
                method=METHOD_LABEL.get(METHOD_OF.get(c, c), c),
                bin=j,
                n=len(h),
                gap_lo=float(h["delta_mc"].min()),
                gap_hi=float(h["delta_mc"].max()),
                gap_mean=float(h["delta_mc"].mean()),
                frac_reversed=float(h["reversed"].mean()),
            )
            for col in cols:
                v = h[col].to_numpy(float)
                v = v[np.isfinite(v)]
                row[f"{col}_mean"] = float(v.mean()) if len(v) else float("nan")
                row[f"{col}_se"] = (
                    float(v.std(ddof=1) / np.sqrt(len(v)))
                    if len(v) > 1
                    else float("nan")
                )
                if len(v):
                    q10, q50, q90 = np.quantile(v, [0.1, 0.5, 0.9])
                else:
                    q10 = q50 = q90 = float("nan")
                row[f"{col}_q10"], row[f"{col}_q50"], row[f"{col}_q90"] = (
                    float(q10),
                    float(q50),
                    float(q90),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(long: pd.DataFrame, n_bins: int = N_BINS):
    slopes = calibration_slopes(long)
    per = attach_ratio(per_state_gaps(long), slopes)
    return per, bin_table(per, n_bins), slopes


# Probe rows


def load_long(
    root: Path = ROOT,
    env: str = ENV,
    epoch: int = EPOCH,
    tags: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Load cached per-candidate rows of the action probe; all non-smoke tags unless `tags` is given."""
    fs = sorted(
        glob.glob(str(root / env / "actions" / f"actions_long_seed*_{epoch}*.csv"))
    )
    if tags is None:
        fs = [f for f in fs if "smoke" not in f]
    else:
        fs = [f for f in fs if any(f.endswith(f"_{epoch}{t}.csv") for t in tags)]
    assert fs, f"no action-probe rows under {root / env / 'actions'}"
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[df["bank"].isin(BANKS) & df["critic"].isin([RUN_NAME[m] for m in METHODS])]
    if seeds is not None:
        df = df[df["seed"].isin(list(seeds))]
    return df.reset_index(drop=True)


def run_probe(
    seeds: Sequence[int],
    banks: Sequence[str],
    max_states: int,
    tag: str,
    smoke: bool,
    env: str = ENV,
    epoch: int = EPOCH,
) -> None:
    """Run failure_figs.stage_actionprobe; `smoke` shrinks it to 4 candidates and 1 continuation."""
    from scripts.probes.cube_bottleneck import failure_figs as FF

    if smoke:
        FF.K_CONT = 1
        FF.N_PERTURB = 0
    out = ROOT / env / "actions"
    bank_file = ROOT / env / "banks" / "banks.npz"
    for seed in seeds:
        t0 = time.time()
        FF.stage_actionprobe(
            env,
            seed,
            bank_file,
            banks,
            out,
            epoch=epoch,
            tag=tag,
            max_states=max_states,
        )
        print(f"[gap] probe seed {seed} done in {time.time() - t0:.0f}s", flush=True)


# Figure


def plot(
    per: pd.DataFrame,
    bins: pd.DataFrame,
    out_dir: Path,
    name: str = "fig_gap_preservation",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.probes.cube_bottleneck.failure_plots import (  # also sets rcParams
        METHOD_COLOR,
        METHOD_MARKER,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1])
    )
    ax, ax2, ax3 = axes
    for m in METHODS:
        b = bins[bins["critic"] == RUN_NAME[m]].sort_values("bin")
        if b.empty:
            continue
        col, mk, lab = METHOD_COLOR[m], METHOD_MARKER[m], METHOD_LABEL[m]
        x = b["gap_mean"].to_numpy()
        for a, key in ((ax, "y"), (ax2, "y_rel")):
            y, se = b[f"{key}_mean"].to_numpy(), b[f"{key}_se"].to_numpy()
            a.fill_between(x, y - se, y + se, color=col, alpha=0.2, lw=0)
            a.plot(x, y, marker=mk, ms=4, lw=1.2, color=col, label=lab)
        p = per[(per["critic"] == RUN_NAME[m]) & (per["tie"] == 0)]
        ax3.scatter(
            p["delta_mc"],
            p["y"],
            s=5,
            alpha=0.35,
            color=col,
            marker=mk,
            lw=0,
            label=lab,
        )
    for a in (ax, ax2, ax3):
        a.axhline(0.0, color="#666666", lw=0.6, ls="--")
        a.set_xlabel("true action gap  $\\Delta_{MC}(s)$  (return)")
    ax.set_ylabel("preserved fraction  $\\beta\\,\\hat\\Delta/\\Delta_{MC}$")
    ax2.set_ylabel("$\\hat\\Delta\\,/\\,(\\max_a q - \\min_a q)$")
    ax3.set_ylabel("preserved fraction (per state)")
    ax3.set_xscale("log")
    ax3.set_yscale("symlog", linthresh=0.1)
    ax.set_title("calibrated fraction, binned mean $\\pm$ s.e.", fontsize=8.5)
    ax2.set_title("share of the critic's own value spread", fontsize=8.5)
    ax3.set_title("raw per-state", fontsize=8.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {out_dir / name}.pdf/.png")


BANK_LABEL = {"approach": "approach", "inhand": "in hand", "carrying": "carrying"}


# Per-phase decision figure


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95 % by default)."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def phase_table(per_state: pd.DataFrame, by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank[, seed]): n states, mean true gap and s.e., fraction ranked correctly with Wilson CI."""
    p = per_state[per_state["tie"] == 0]
    keys = ["critic", "bank"] + (["seed"] if by_seed else [])
    rows = []
    for k, g in p.groupby(keys, sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        d = g["delta_mc"].to_numpy(float)
        n = len(g)
        k_ok = int((g["reversed"] == 0).sum())
        lo, hi = wilson_ci(k_ok, n)
        row = dict(zip(keys, k))
        row.update(
            method=METHOD_LABEL.get(
                METHOD_OF.get(row["critic"], row["critic"]), row["critic"]
            ),
            n=n,
            gap_mean=float(d.mean()),
            gap_se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            n_correct=k_ok,
            frac_correct=k_ok / n,
            ci_lo=lo,
            ci_hi=hi,
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["bank"] = pd.Categorical(out["bank"], BANKS, ordered=True)
    return out.sort_values(keys).reset_index(drop=True)


def plot_decision_by_phase(
    per: pd.DataFrame,
    out_dir: Path,
    name: str = "fig_decision_by_phase",
    show_seeds: bool = True,
) -> pd.DataFrame:
    """Plot mean true action gap (top) and fraction ranked correctly (bottom) per phase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR, METHOD_MARKER

    t = phase_table(per)
    ts = phase_table(per, by_seed=True)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.3, 4.1),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.32),
    )
    x = np.arange(len(BANKS))
    w = 0.34
    for j, m in enumerate(METHODS):
        col, mk = METHOD_COLOR[m], METHOD_MARKER[m]
        b = t[t["critic"] == RUN_NAME[m]].set_index("bank").reindex(BANKS)
        xo = x + (j - 0.5) * w
        ax1.bar(
            xo,
            b["gap_mean"],
            w,
            yerr=b["gap_se"],
            color=col,
            alpha=0.75,
            lw=0,
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444444"),
            label=METHOD_LABEL[m],
        )
        if show_seeds:
            sb = ts[ts["critic"] == RUN_NAME[m]]
            for bk_i, bk in enumerate(BANKS):
                v = sb[sb["bank"] == bk]["frac_correct"].to_numpy()
                ax2.scatter(
                    np.full(len(v), xo[bk_i]),
                    v,
                    marker=mk,
                    s=9,
                    color=col,
                    alpha=0.35,
                    lw=0,
                    zorder=1,
                )
        yerr = np.vstack(
            [b["frac_correct"] - b["ci_lo"], b["ci_hi"] - b["frac_correct"]]
        )
        ax2.errorbar(
            xo,
            b["frac_correct"],
            yerr=yerr,
            fmt=mk,
            ms=6,
            color=col,
            ecolor=col,
            elinewidth=1.0,
            capsize=2.5,
            lw=0,
            label=METHOD_LABEL[m],
            zorder=3,
        )
        for xi, (_, r) in zip(xo, b.iterrows()):
            ax2.annotate(
                f"n={int(r['n'])}",
                (xi, 0.03 + 0.06 * j),
                ha="center",
                va="bottom",
                fontsize=6,
                color=col,
            )
    ax1.set_ylabel("true action gap (return)")
    ax1.set_title("How much the action choice matters", fontsize=8.5)
    ax1.legend(frameon=False, loc="upper left")
    ax2.axhline(0.5, color="#666666", lw=0.7, ls="--")
    ax2.text(
        len(BANKS) - 1 + 0.5,
        0.515,
        "chance",
        fontsize=7,
        color="#666666",
        ha="right",
        va="bottom",
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("critic ranks best above worst\n(fraction of states)")
    ax2.set_title("Does the critic pick the right action?", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([BANK_LABEL[b] for b in BANKS])
    ax2.set_xlim(-0.6, len(BANKS) - 0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {out_dir / name}.pdf/.png")
    t.to_csv(out_dir / f"{name}.csv", index=False)
    ts.to_csv(out_dir / f"{name}_per_seed.csv", index=False)
    return t


def print_tables(bins: pd.DataFrame, slopes: Dict, per: pd.DataFrame) -> None:
    print(
        "=== calibration slope beta per checkpoint (return units per critic unit) ==="
    )
    for k, v in sorted(slopes.items()):
        print(f"  {k[0]} seed {k[1]}: beta = {v:.4g}")
    ties = per.groupby("critic")["tie"].sum().to_dict()
    print(f"=== tie states (all candidates same return; excluded): {ties}")
    cols = [
        "method",
        "bin",
        "n",
        "gap_lo",
        "gap_hi",
        "gap_mean",
        "y_mean",
        "y_se",
        "y_q10",
        "y_q50",
        "y_q90",
        "y_rel_mean",
        "y_rel_se",
        "frac_reversed",
    ]
    print("=== per-bin table (bins are per method: own-policy rollouts) ===")
    print(bins[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="5 states x 4 actions x 1 rollout, then aggregate",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="re-run the full action probe for --seeds/--banks",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--banks", nargs="+", default=BANKS)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument(
        "--tag", default="", help="tag of freshly probed rows (default: campaign tags)"
    )
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--out", default=str(FIG))
    args = ap.parse_args()

    out = Path(args.out)
    tags = None
    if args.smoke:
        tag = "_gapsmoke"
        run_probe([args.seeds[0]], ["carrying"], 5, tag, smoke=True)
        tags, out = [tag], out / "smoke"
    elif args.probe:
        tag = args.tag or "_gap"
        run_probe(args.seeds, args.banks, args.max_states, tag, smoke=False)
        tags = [tag]
    elif args.tag:
        tags = [args.tag]
    long = load_long(
        tags=tags, seeds=args.seeds if (args.smoke or args.probe or args.tag) else None
    )
    long = long[long["bank"].isin(args.banks)]
    per, bins, slopes = aggregate(long, args.n_bins)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "gap_preservation_per_state.csv", index=False)
    bins.to_csv(out / "gap_preservation_bins.csv", index=False)
    (out / "gap_preservation_slopes.json").write_text(
        json.dumps({f"{c}_seed{s}": v for (c, s), v in slopes.items()}, indent=2)
    )
    print_tables(bins, slopes, per)
    plot(per, bins, out)
    pt = plot_decision_by_phase(per, out)
    print(
        "=== per-phase table (correct = critic ranks true best above true worst; 95 % Wilson) ==="
    )
    print(pt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
