#!/usr/bin/env python
"""scripts/plot_repsep.py — representation-aliasing figure (rlbrew P4).

Reads the per-method grasp-state representations in analysis/repsep/*.parquet
(columns {split, task, success, rep_0..rep_k}) and renders, as the headline P4
visual, that the learned representations do NOT separate soon-to-succeed from
soon-to-fail in-hand states.

For each method we fit a 2D PCA on its pooled (train+eval) grasps and a linear
probe (logistic regression, stratified 5-fold, ROC-AUC) separately on train and
eval grasps. The figure is a 2x4 grid: rows {Train, Eval} x cols {FB, GCIQL,
CRL, RLDP}; each point is an in-hand state coloured green=success-bound /
vermillion=fail-bound, each panel annotated with its linear-probe AUC. AUC ~0.5
= no linear separation = aliasing; the train->eval AUC drop = overgeneralisation.

Outputs PAPER/rlbrew/figures/fig_repsep.png and analysis/repsep/auc_table.md.

Run under .venv (sklearn + seaborn).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
REPSEP = REPO / "analysis" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
SUCC, FAIL = _CB[2], _CB[3]          # green / vermillion

METHODS = ["fb", "gciql", "crl", "rldp"]
LABELS = {"fb": "FB", "gciql": "GCIQL", "crl": "CRL", "rldp": "RLDP"}
MAX_PTS = 600                        # per class per panel (visual clarity)
RNG = np.random.default_rng(0)


def _load(method):
    df = pd.read_parquet(REPSEP / f"{method}.parquet")
    cols = sorted((c for c in df.columns if c.startswith("rep_")),
                  key=lambda c: int(c.split("_")[1]))
    X = df[cols].to_numpy(np.float32)
    y = df["success"].to_numpy(bool)
    split = df["split"].to_numpy()
    episode = df["episode"].to_numpy(int)
    return X, y, split, episode


def _auc(X, y, groups=None):
    """ROC-AUC of a standardised logistic probe, 5-fold.

    With `groups` (eval: episode ids) use StratifiedGroupKFold so no rollout
    spans train/test folds -- otherwise per-step labels leak episode identity
    and the AUC is meaningless. Train grasps are independent -> StratifiedKFold.
    """
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        return np.nan
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            return np.nan
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=0)
        return float(cross_val_score(clf, X, y.astype(int), groups=groups,
                                     cv=cv, scoring="roc_auc").mean())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y.astype(int), cv=cv,
                                 scoring="roc_auc").mean())


def _subsample(idx, y):
    """Balanced subsample of indices for a scatter panel."""
    pos = idx[y[idx]]
    neg = idx[~y[idx]]
    pos = RNG.choice(pos, min(len(pos), MAX_PTS), replace=False)
    neg = RNG.choice(neg, min(len(neg), MAX_PTS), replace=False)
    return pos, neg


def _panel(ax, Z, y, mask, auc, title=None, ylab=None):
    idx = np.where(mask)[0]
    pos, neg = _subsample(idx, y)
    ax.scatter(Z[neg, 0], Z[neg, 1], s=6, c=FAIL, alpha=0.35,
               linewidths=0, label="Fail-bound")
    ax.scatter(Z[pos, 0], Z[pos, 1], s=6, c=SUCC, alpha=0.45,
               linewidths=0, label="Success-bound")
    # class centroids make the overlap legible
    for sub, col in ((pos, SUCC), (neg, FAIL)):
        if len(sub):
            ax.scatter(Z[sub, 0].mean(), Z[sub, 1].mean(), s=160, marker="X",
                       c=col, edgecolors="black", linewidths=1.2, zorder=5)
    txt = "AUC n/a" if np.isnan(auc) else f"AUC {auc:.2f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if ylab:
        ax.set_ylabel(ylab, fontsize=12, fontweight="bold")


def fig_auc_bars(out, auc_rows):
    """Grouped train/eval linear-probe AUC bar chart (chance = 0.5)."""
    names = [r[0] for r in auc_rows]
    tr = np.array([r[1] for r in auc_rows])
    ev = np.array([r[2] for r in auc_rows])
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x - w / 2, tr, w, color=_CB[0], edgecolor="black", linewidth=0.4,
           label="Train grasps")
    ax.bar(x + w / 2, ev, w, color=_CB[3], edgecolor="black", linewidth=0.4,
           label="Eval grasps")
    ax.axhline(0.5, color="black", ls="--", lw=1.0, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Linear-probe AUC\n(success- vs fail-bound)", fontsize=10)
    ax.set_ylim(0.4, 0.85)
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), frameon=False, columnspacing=1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    auc_rows = []
    for j, m in enumerate(METHODS):
        X, y, split, episode = _load(m)
        tr = split == "train"
        ev = split == "eval"
        # per-method PCA fit on pooled grasps so train/eval share axes
        Xs = StandardScaler().fit_transform(X)
        Z = PCA(n_components=2, random_state=0).fit_transform(Xs)
        auc_tr = _auc(X[tr], y[tr])
        auc_ev = _auc(X[ev], y[ev], groups=episode[ev])
        auc_rows.append((LABELS[m], auc_tr, auc_ev))
        _panel(axes[0, j], Z, y, tr, auc_tr,
               title=LABELS[m], ylab="Train grasps" if j == 0 else None)
        _panel(axes[1, j], Z, y, ev, auc_ev,
               ylab="Eval grasps" if j == 0 else None)
        print(f"[repsep] {LABELS[m]}: train AUC={auc_tr:.3f} "
              f"eval AUC={auc_ev:.3f} (n_tr={tr.sum()}, n_ev={ev.sum()})")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, c=SUCC,
                          label="Success-bound"),
               plt.Line2D([], [], marker="o", ls="", ms=7, c=FAIL,
                          label="Fail-bound")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("In-hand grasp representations: success-bound vs fail-bound "
                 "separability (per-method PCA; linear-probe AUC)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    out_png = OUT / "fig_repsep.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")

    fig_auc_bars(OUT / "fig_repsep_auc.png", auc_rows)

    # AUC table for the appendix
    lines = ["| Method | Train AUC | Eval AUC | Train$\\to$Eval drop |",
             "|---|---|---|---|"]
    for name, a_tr, a_ev in auc_rows:
        drop = "n/a" if (np.isnan(a_tr) or np.isnan(a_ev)) else f"{a_tr - a_ev:+.2f}"
        lines.append(f"| {name} | {a_tr:.2f} | {a_ev:.2f} | {drop} |")
    tbl = REPSEP / "auc_table.md"
    tbl.write_text("\n".join(lines) + "\n")
    print(f"[plot] wrote {tbl}\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
