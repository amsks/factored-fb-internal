"""scripts/training_value_aggregate.py — Spearman table + value-vs-distance
curves + phase-composition bar + value-over-cube-xy scenes for the
training-data value analysis. Reads fb_values/gciql_values/phase_composition
from --root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import phase_spearman_table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIONS = ["reach", "grasp", "transport"]


def _load_pooled(root: Path, name: str):
    """Read `name` from per-pair subdirs p*/ (multi-seed, tagged with `pair`)
    if present, else from `root` directly (single run)."""
    pair_dirs = sorted(d for d in root.glob("p*") if d.is_dir())
    if pair_dirs:
        frames = []
        for pd_ in pair_dirs:
            f = pd_ / name
            if f.exists():
                df = pd.read_parquet(f)
                df["pair"] = pd_.name
                frames.append(df)
        return (pd.concat(frames, ignore_index=True) if frames else None), True
    f = root / name
    return (pd.read_parquet(f) if f.exists() else None), False


def _methods(root: Path):
    """((df, value_col, label) list, group_cols). Pools per-pair subdirs."""
    fb, multi = _load_pooled(root, "fb_values.parquet")
    out = [(fb, "V_policy", "FB policy"), (fb, "V_data", "FB data")]
    gq, _ = _load_pooled(root, "gciql_values.parquet")
    if gq is not None:
        out.append((gq, "V", "GCIQL"))
    group_cols = ("pair", "task") if multi else ("task",)
    return out, group_cols


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    methods, group_cols = _methods(root)
    n_seeds = (methods[0][0]["pair"].nunique()
               if "pair" in methods[0][0].columns else 1)

    # 1. Spearman table (region x method, mean over group_cols).
    rows = []
    for df, vc, label in methods:
        tab = phase_spearman_table(df, value_col=vc, group_cols=group_cols)
        for _, r in tab.iterrows():
            rows.append({"method": label, "region": r["region"],
                         "rho_mean": r["rho_mean"], "rho_std": r["rho_std"]})
    sp = pd.DataFrame(rows)
    sp.to_parquet(out / "spearman_table.parquet")
    md = ["# Training-state value gradient: Spearman rho(V, -d) by phase",
          f"_n_seeds={n_seeds}; mean over {'seed x task' if n_seeds > 1 else 'task'}_",
          "",
          "| method | " + " | ".join(REGIONS) + " |",
          "| :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
    for label in [m[2] for m in methods]:
        cells = [label]
        for rg in REGIONS:
            s = sp[(sp.method == label) & (sp.region == rg)]
            cells.append(f"{s['rho_mean'].iloc[0]:.3f}" if len(s) else "—")
        md.append("| " + " | ".join(cells) + " |")
    (out / "spearman_table.md").write_text("\n".join(md) + "\n")

    # 2. value-vs-distance curves (faceted by phase, line per method, z-scored).
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(5 * len(REGIONS), 4),
                             squeeze=False)
    for ax, rg in zip(axes[0], REGIONS):
        for df, vc, label in methods:
            sub = df[df.region == rg]
            if sub.empty:
                continue
            v = sub[vc].to_numpy(float)
            vz = (v - v.mean()) / (v.std() + 1e-9)
            edges = np.unique(np.quantile(sub["d"], np.linspace(0, 1, 10)))
            if len(edges) < 3:
                continue
            ctr = 0.5 * (edges[:-1] + edges[1:])
            idx = np.clip(np.digitize(sub["d"], edges) - 1, 0, len(ctr) - 1)
            mv = pd.DataFrame({"b": idx, "vz": vz}).groupby("b")["vz"].mean()
            ax.plot(ctr[mv.index], mv.values, marker="o", label=label)
        ax.set_title(rg)
        ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("z-scored value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training states: value vs cube-to-goal distance, by phase")
    fig.tight_layout()
    fig.savefig(out / "value_vs_dist.png", dpi=130)
    plt.close(fig)

    # 3. dataset phase composition bar (averaged across pairs if multi-seed).
    comp_raw, _ = _load_pooled(root, "phase_composition.parquet")
    comp = (comp_raw.groupby("region")["fraction"].mean().reset_index()
            .set_index("region").reindex(REGIONS).fillna(0.0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(REGIONS, comp["fraction"].values)
    ax.set_ylabel("fraction of training transitions")
    ax.set_title("Dataset phase composition")
    fig.tight_layout()
    fig.savefig(out / "phase_composition.png", dpi=130)
    plt.close(fig)

    # 4. value-over-cube-xy scenes (phase x method): binned mean value.
    for df, vc, label in methods:
        fig, axes = plt.subplots(1, len(REGIONS),
                                 figsize=(5 * len(REGIONS), 4), squeeze=False)
        for ax, rg in zip(axes[0], REGIONS):
            sub = df[df.region == rg]
            ax.set_title(f"{label} — {rg} (n={len(sub)})")
            if len(sub) >= 20:
                h = ax.hexbin(sub["cube_x"], sub["cube_y"], C=sub[vc],
                              gridsize=20, reduce_C_function=np.mean,
                              cmap="Reds")
                fig.colorbar(h, ax=ax, shrink=0.8)
            ax.set_xlabel("cube x")
            ax.set_ylabel("cube y")
        fig.tight_layout()
        fname = f"scene_{label.replace(' ', '_').lower()}.png"
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    print(f"[training_value_aggregate] -> {out}")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/training_value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
