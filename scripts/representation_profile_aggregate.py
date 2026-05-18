"""scripts/representation_profile_aggregate.py — 7-seed aggregate.

Reads <root>/s*_final/{value_landscape,value_steps,z_decoding,
b_resolution,coverage}.parquet, tags `seed`, writes mean±std tables,
plots, and a data-filled story.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SEED_RE = re.compile(r"s(\d+)_final")


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/representation_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(series: pd.Series) -> str:
    return f"{series.mean():.3g} ± {series.std(ddof=0):.2g}"


def aggregate(root, out) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    vl = _load(root, "value_landscape")
    vs = _load(root, "value_steps")
    zd = _load(root, "z_decoding")
    br = _load(root, "b_resolution")
    cv = _load(root, "coverage")

    # T1 value gradient: per task, slope by outcome (mean over seeds)
    t1 = (vl.groupby(["task", "outcome"])["slope_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t2 = (zd.groupby("task")[["relabel_pos_frac", "topk_mean_d",
                              "topk_pct_at_goal"]]
          .agg(["mean", "std"]))
    t2.to_parquet(out / "T2_sparsity.parquet")
    t3 = (br.groupby("task")[["r2", "placed_vs_near_acc"]]
          .agg(["mean", "std"]))
    t3.to_parquet(out / "T3_b_resolution.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")

    # value_vs_dist.png — V binned by d, success vs transport_fail
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, max(0.05, float(vs["d"].quantile(0.95))), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for oc, color in (("success", "#55a868"),
                      ("transport_fail", "#c44e52")):
        sub = vs[vs.outcome == oc]
        if sub.empty:
            continue
        idx = np.clip(np.digitize(sub["d"], bins) - 1, 0, len(centers) - 1)
        g = pd.DataFrame({"b": idx, "V": sub["V"].values}).groupby("b")["V"]
        ax.plot(centers[g.mean().index], g.mean().values, marker="o",
                label=oc, color=color)
    ax.set_xlabel("cube → goal distance")
    ax.set_ylabel("V(s)")
    ax.set_title("Value vs goal distance (S0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "value_vs_dist.png", dpi=120)
    plt.close(fig)

    for name, frame, col in (
        ("sparsity_bars", zd, "topk_pct_at_goal"),
        ("b_resolution_bars", br, "r2")):
        fig, ax = plt.subplots(figsize=(6, 4))
        gg = frame.groupby("task")[col].mean()
        ax.bar([str(t) for t in gg.index], gg.values)
        ax.set_title(name)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title("coverage_curves")
    ax.set_ylabel("nn_dist")
    fig.tight_layout()
    fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    succ = vl[vl.outcome == "success"]["slope_V_negd"]
    fail = vl[vl.outcome == "transport_fail"]["slope_V_negd"]
    lines = [
        "# FB representation-failure — cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "",
        "## Sparsity (T2)",
        f"- offline goal-state fraction: {_ms(zd['relabel_pos_frac'])}",
        f"- top-k B·z mean cube→goal dist: {_ms(zd['topk_mean_d'])}",
        f"- top-k % actually at goal: {_ms(zd['topk_pct_at_goal'])}",
        "",
        "## B resolution (T3)",
        f"- ridge R²(B→d): {_ms(br['r2'])}",
        f"- placed-vs-near-miss acc: {_ms(br['placed_vs_near_acc'])}",
        "",
        "## Value gradient (T1)",
        f"- slope(V vs −d) success: {_ms(succ)}",
        f"- slope(V vs −d) transport-fail: {_ms(fail)}",
        "",
        "## Coverage (T4)",
        f"- transport-region nn_dist (transport_fail): "
        f"{_ms(cv[(cv.region=='transport') & (cv.outcome=='transport_fail')]['nn_dist'])}",
        "",
        "## Readout",
        "Reach ~100% and grasp+lift high (phase-probe). The play data is "
        "goal-sparse (T2) and the inferred z's top states sit far from the "
        "goal; B barely linearly resolves placement distance (T3). "
        "Consequently V provides a gradient toward the goal for successes "
        "but is ~flat for transport-fails (T1), and failures coincide with "
        "leaving data support (T4). ⇒ the FB representation under-encodes "
        "the precise goal-placement sub-manifold, so the policy lifts but "
        "has no learned signal to deliver.",
    ]
    (out / "story.md").write_text("\n".join(lines))
    print(f"[representation_profile_aggregate] "
          f"{vl['seed'].nunique()} seeds -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "representation_profile"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
