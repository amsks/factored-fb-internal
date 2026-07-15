#!/usr/bin/env python
"""scripts/probes/repr_aliasing.py — outcome-separation of look-alike grasps.

Question. Among in-hand grasps that are CLOSE in raw observation but differ in
the agent's own continuation outcome (keeps control -> success vs loses it ->
fail), does the *policy-evaluated representation* the value reads --
F(s, pi(s,z), z) (PSM: psi(s,pi,z); GCIQL: penultimate value features) -- place
opposite-outcome look-alikes farther apart than equally-close same-outcome
look-alikes?

Method (addresses reviewer points 1-8):
  * Trustworthy labels (#4). Each grasp's continuation outcome is a soft label
    p_succ from K independent rollouts (the flow/FQL actor is stochastic at eval,
    so a single rollout is a noisy draw). We keep only CONFIDENT grasps:
    p_succ >= HI (success) or <= LO (fail); ambiguous grasps are dropped.
  * Caliper matching, no raw-distance division (#1). Per goal we z-score obs and
    rep. For each confident anchor we take its nearest opposite-outcome grasp and
    its nearest same-outcome grasp in raw obs, and KEEP the triplet only if the
    two raw match-distances agree to within a caliper (|log-ratio| <= CALIPER).
    The caliper makes the two matches equidistant in raw space, so we compare
    their representation distances DIRECTLY (in log space) instead of dividing by
    a noisy raw distance.
  * Both-class, class-balanced anchors (#2). Anchors are drawn from BOTH success
    and fail grasps; the per-anchor statistic Delta = log d_rep(opp) - log d_rep(same)
    is averaged within each class and the two class means are then averaged, so
    neither outcome dominates.
  * Statistic (#6). Outcome Separation Ratio  OSR = exp(E[Delta]).  OSR = 1 means
    no differential separation (opposite- and same-outcome look-alikes equally far
    in the representation); OSR > 1 means opposite outcomes are pushed apart more.
  * Hierarchical bootstrap (#3). Uncertainty resamples seeds -> goals -> anchors,
    so the CI reflects across-seed variation, not just anchor count.
  * Label-permutation null (#8). Within each goal we shuffle the confident labels
    across grasps (preserving class counts) and recompute OSR; the null band
    calibrates the OSR expected from finite-sample matching artifacts alone.

Reads per-seed analysis/value/repsep/cf_repfeat_<m>_s*.parquet (falls back to the
single cf_repfeat_<m>.parquet, treated as one seed). Run under .venv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis/value/repsep"
METHODS = ["gciql", "crl", "fb", "rldp", "psm"]

CONF_HI, CONF_LO = 0.8, 0.2      # confident-success / confident-fail thresholds on p_succ
CALIPER = np.log(1.5)            # opp and same raw match-distances must agree to 1.5x
MIN_PER_CLASS = 5               # need >= this many confident grasps of each class in a goal
RNG = np.random.default_rng(0)


def _zscore(X):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    return (X - mu) / sd


def _seed_files(m):
    fs = sorted(REPSEP.glob(f"cf_repfeat_{m}_s*.parquet"))
    if fs:
        return fs
    f = REPSEP / f"cf_repfeat_{m}.parquet"
    return [f] if f.exists() else []


def _labels(d):
    """Confident label in {+1, -1, 0(ambiguous)}. Prefers soft p_succ; falls back
    to the single-draw boolean success column (all treated as confident)."""
    if "p_succ" in d.columns:
        p = d["p_succ"].to_numpy(np.float64)
        lab = np.zeros(len(p), int)
        lab[p >= CONF_HI] = 1
        lab[p <= CONF_LO] = -1
        return lab
    y = d["success"].to_numpy(bool)
    return np.where(y, 1, -1)


def _triplets(O, R, lab):
    """Caliper-matched triplets for one (seed, goal).
    Returns list of (anchor_class, log d_rep(opp), log d_rep(same))."""
    pos, neg = np.where(lab == 1)[0], np.where(lab == -1)[0]
    if len(pos) < MIN_PER_CLASS or len(neg) < MIN_PER_CLASS:
        return []
    tp, tn = cKDTree(O[pos]), cKDTree(O[neg])
    out = []
    for cls, idx, same_tree, same_ix, opp_tree, opp_ix in (
            (1, pos, tp, pos, tn, neg), (-1, neg, tn, neg, tp, pos)):
        d_opp, i_opp = opp_tree.query(O[idx], k=1)                 # nearest opposite class
        d_sm, i_sm = same_tree.query(O[idx], k=2)                  # nearest same class (drop self)
        d_sm, i_sm = d_sm[:, 1], i_sm[:, 1]
        keep = (d_opp > 0) & (d_sm > 0) & (np.abs(np.log(d_opp / d_sm)) <= CALIPER)
        for a in np.where(keep)[0]:
            lo = np.log(np.linalg.norm(R[idx[a]] - R[opp_ix[i_opp[a]]]) + 1e-12)
            ls = np.log(np.linalg.norm(R[idx[a]] - R[same_ix[i_sm[a]]]) + 1e-12)
            out.append((cls, float(lo), float(ls)))
    return out


def _osr_from(triplets):
    """Class-balanced OSR = exp(mean over classes of mean Delta)."""
    if not triplets:
        return np.nan
    arr = np.asarray([(c, lo - ls) for c, lo, ls in triplets], float)
    deltas = []
    for c in (1, -1):
        dc = arr[arr[:, 0] == c, 1]
        if len(dc):
            deltas.append(dc.mean())
    if not deltas:
        return np.nan
    return float(np.exp(np.mean(deltas)))


def _build(m):
    """Per-seed, per-goal caliper-matched triplet tables for method m.
    Returns list over seeds of {goal_id: list-of-triplets}."""
    seeds = []
    for f in _seed_files(m):
        d = pd.read_parquet(f)
        ocols = [c for c in d.columns if c.startswith("obs_")]
        rcols = [c for c in d.columns if c.startswith("rep_")]
        goals = {}
        for t in sorted(d.task.unique()):
            dt = d[d.task == t]
            O = _zscore(dt[ocols].to_numpy(np.float64))
            R = _zscore(dt[rcols].to_numpy(np.float64))
            trip = _triplets(O, R, _labels(dt))
            if trip:
                goals[int(t)] = trip
        if goals:
            seeds.append(goals)
    return seeds


def _hier_boot(seeds, reps=3000):
    """Hierarchical bootstrap over seeds -> goals -> anchors. Returns (osr, lo, hi)."""
    if not seeds:
        return np.nan, np.nan, np.nan
    point = _osr_from([t for goals in seeds for trips in goals.values() for t in trips])
    vals = []
    sidx = np.arange(len(seeds))
    for _ in range(reps):
        pooled = []
        for s in RNG.choice(sidx, len(sidx), replace=True):
            gk = list(seeds[s].keys())
            if not gk:
                continue
            for g in RNG.choice(gk, len(gk), replace=True):
                tr = seeds[s][g]
                pooled.extend(tr[i] for i in RNG.integers(0, len(tr), len(tr)))
        v = _osr_from(pooled)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return point, float(lo), float(hi)


def _perm_null(m_seeds, reps=1000):
    """Label-permutation null: shuffle confident labels within each (seed, goal),
    rebuild caliper triplets, recompute OSR. Returns (lo, hi) 95% null band.
    m_seeds carries the raw z-scored (O, R, lab) so labels can be re-permuted."""
    vals = []
    for _ in range(reps):
        pooled = []
        for goals in m_seeds:
            for O, R, lab in goals.values():
                lp = lab.copy()
                conf = lp != 0
                perm = RNG.permutation(lp[conf])
                lp = lp.copy(); lp[conf] = perm
                pooled.extend(_triplets(O, R, lp))
        v = _osr_from(pooled)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _build_raw(m):
    """Per-seed, per-goal (O, R, lab) arrays for the permutation null."""
    seeds = []
    for f in _seed_files(m):
        d = pd.read_parquet(f)
        ocols = [c for c in d.columns if c.startswith("obs_")]
        rcols = [c for c in d.columns if c.startswith("rep_")]
        goals = {}
        for t in sorted(d.task.unique()):
            dt = d[d.task == t]
            lab = _labels(dt)
            if (lab == 1).sum() < MIN_PER_CLASS or (lab == -1).sum() < MIN_PER_CLASS:
                continue
            goals[int(t)] = (_zscore(dt[ocols].to_numpy(np.float64)),
                             _zscore(dt[rcols].to_numpy(np.float64)), lab)
        if goals:
            seeds.append(goals)
    return seeds


# display order = descending aggregate success; labels + paper colorblind palette
ORDER = [("gciql", "GCIQL", 71.8), ("fb", "FB", 52.0), ("psm", "PSM", 36.7),
         ("crl", "CRL", 25.8), ("rldp", "RLDP", 22.7)]


def _plot(res, out):
    import matplotlib.pyplot as plt, seaborn as sns
    _CB = sns.color_palette("colorblind").as_hex()
    MC = {"fb": _CB[0], "gciql": _CB[1], "crl": _CB[2], "rldp": _CB[3], "psm": _CB[6]}
    ms = [m for m, _, _ in ORDER if m in res]
    labs = [lab for m, lab, _ in ORDER if m in res]
    y = [res[m]["osr"] for m in ms]
    lo = [res[m]["osr"] - res[m]["lo"] for m in ms]
    hi = [res[m]["hi"] - res[m]["osr"] for m in ms]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.axhline(1.0, color="black", ls="--", lw=1.0, zorder=1)
    ax.text(len(ms) - 0.5, 1.0, "  no differential\n  separation", va="center",
            ha="left", fontsize=8, color="black")
    # label-permutation null band (grey) behind each bar
    for i, m in enumerate(ms):
        nlo, nhi = res[m].get("null", (np.nan, np.nan))
        if np.isfinite(nlo):
            ax.add_patch(plt.Rectangle((i - 0.31, nlo), 0.62, nhi - nlo,
                                       color="0.6", alpha=0.35, zorder=1.5))
    bars = ax.bar(range(len(ms)), y, yerr=[lo, hi], capsize=4, width=0.62,
                  color=[MC[m] for m in ms], edgecolor="black", linewidth=0.5, zorder=2)
    for i, (v, h) in enumerate(zip(y, hi)):
        ax.text(i, v + h + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(labs, fontsize=11, fontweight="bold")
    ax.set_ylabel("outcome-separation ratio\n(opposite- vs same-outcome, representation)", fontsize=10)
    ax.set_title("Does the representation separate look-alike grasps\n"
                 "with opposite continuation outcomes?", fontsize=11)
    ax.set_ylim(0.0, max(max(y) + max(hi) + 0.25, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    # legend for the null band
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="0.6", alpha=0.35, label="permutation null (95%)"))
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    res = {}
    print(f"{'method':7} {'seeds':>5} {'triplets':>9}   OSR (95% hier-CI)      null-95%")
    for m in METHODS:
        if not _seed_files(m):
            print(f"  {m}: no data"); continue
        seeds = _build(m)
        n_trip = sum(len(tr) for g in seeds for tr in g.values())
        if n_trip == 0:
            print(f"  {m}: no caliper-matched triplets"); continue
        osr, lo, hi = _hier_boot(seeds)
        nlo, nhi = _perm_null(_build_raw(m))
        res[m] = dict(osr=osr, lo=lo, hi=hi, null=(nlo, nhi),
                      nseed=len(seeds), ntrip=n_trip)
        print(f"{m:7} {len(seeds):>5} {n_trip:>9}   {osr:.2f} [{lo:.2f}, {hi:.2f}]"
              f"     [{nlo:.2f}, {nhi:.2f}]")
    _plot(res, REPSEP / "fig_repr_aliasing.png")
    return res


if __name__ == "__main__":
    main()
