"""scripts/gciql_profile_aggregate.py — 10-seed GCIQL aggregate
(T1/T4/funnel) + FB-vs-GCIQL comparison. Reuses _profile_core verdicts
so GCIQL and FB are scored identically."""

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

from evals._profile_core import (  # noqa: E402
    _synthesis, _verdict_t1, _verdict_t4)

_SEED_RE = re.compile(r"s(\d+)_final")


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(root).glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/gciql_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3g} ± {s.std(ddof=0):.2g}"


def aggregate(root, out, fb_aggregate=None, fb_seed_root=None) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vl = _load(root, "value_landscape")
    cv = _load(root, "coverage")
    fun = _load(root, "phase_funnel")

    t1 = (vl.groupby(["task", "outcome"])["rho_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")
    fun.to_parquet(out / "funnel.parquet")

    rho_s = float(vl[vl.outcome == "success"]["rho_V_negd"].mean())
    rho_f = float(vl[vl.outcome == "transport_fail"]["rho_V_negd"].mean())
    cvt = cv[cv.region == "transport"]
    fail_nn = float(cvt[cvt.outcome == "transport_fail"]["nn_dist"].mean())
    succ_nn = float(cvt[cvt.outcome == "success"]["nn_dist"].mean())
    verdicts = {"T1": _verdict_t1(rho_s, rho_f),
                "T4": _verdict_t4(fail_nn, succ_nn)}

    fig, ax = plt.subplots(figsize=(6, 4))
    g = vl.groupby("outcome")["rho_V_negd"].mean()
    ax.bar([str(i) for i in g.index], g.values)
    ax.set_title("GCIQL value gradient")
    ax.set_ylabel("Spearman rho(V, -d)")
    fig.tight_layout(); fig.savefig(out / "value_rho_bars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title("GCIQL coverage"); ax.set_ylabel("nn_dist")
    fig.tight_layout(); fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    fc = fun.groupby("furthest_phase").size()
    ax.bar([str(i) for i in fc.index], fc.values)
    ax.set_title("GCIQL phase funnel")
    fig.tight_layout(); fig.savefig(out / "funnel.png", dpi=120)
    plt.close(fig)

    lines = [
        "# GCIQL cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "", "## Value gradient (T1)",
        f"- Spearman ρ(V vs −d) success: "
        f"{_ms(vl[vl.outcome=='success']['rho_V_negd'])}",
        f"- Spearman ρ(V vs −d) transport-fail: "
        f"{_ms(vl[vl.outcome=='transport_fail']['rho_V_negd'])}",
        "", "## Coverage (T4)",
        f"- transport nn_dist (transport_fail): "
        f"{_ms(cvt[cvt.outcome=='transport_fail']['nn_dist'])}",
        "", "## Readout",
        f"- T1 value gradient: rho_success={rho_s:.3g}, "
        f"rho_fail={rho_f:.3g} -> {verdicts['T1'][0]} "
        f"({verdicts['T1'][1]})",
        f"- T4 coverage: fail={fail_nn:.3g}, success={succ_nn:.3g} "
        f"-> {verdicts['T4'][0]} ({verdicts['T4'][1]})",
        _synthesis(verdicts),
    ]
    (out / "story.md").write_text("\n".join(lines))

    if fb_aggregate is not None:
        _comparison(out / "comparison", vl, cvt, fb_aggregate,
                    rho_s, rho_f, fail_nn, succ_nn, verdicts,
                    gciql_root=root, fb_seed_root=fb_seed_root)
    print(f"[gciql_profile_aggregate] {vl['seed'].nunique()} seeds -> {out}")


def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd, gciql_root=None, fb_seed_root=None) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    fb_t1 = pd.read_parquet(
        Path(fb_aggregate) / "T1_value_gradient.parquet")

    def _fb_rho(oc):
        sub = fb_t1[fb_t1.outcome == oc]
        col = "mean" if "mean" in sub.columns else "rho_V_negd"
        return float(sub[col].mean()) if len(sub) else float("nan")

    fb_rs, fb_rf = _fb_rho("success"), _fb_rho("transport_fail")
    fbv = _verdict_t1(fb_rs, fb_rf)
    rows = [
        {"method": "FB", "rho_success": fb_rs, "rho_fail": fb_rf,
         "T1_verdict": fbv[0]},
        {"method": "GCIQL", "rho_success": g_rs, "rho_fail": g_rf,
         "T1_verdict": g_verd["T1"][0]},
    ]
    cmp = pd.DataFrame(rows)
    cmp.to_parquet(cdir / "fb_vs_gciql.parquet")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [fb_rs, g_rs], 0.4, label="success")
    ax.bar(x + 0.2, [fb_rf, g_rf], 0.4, label="transport_fail")
    ax.set_xticks(x); ax.set_xticklabels(["FB", "GCIQL"])
    ax.set_ylabel("Spearman rho(V, -d)"); ax.legend()
    ax.set_title("FB vs GCIQL: value gradient")
    fig.tight_layout(); fig.savefig(cdir / "cmp_value_rho.png", dpi=120)
    plt.close(fig)

    def _seed_concat(rt, name):
        if rt is None:
            return None
        fr = []
        for p in sorted(Path(rt).glob(f"s*_final/{name}.parquet")):
            fr.append(pd.read_parquet(p))
        return pd.concat(fr, ignore_index=True) if fr else None

    g_vs = _seed_concat(gciql_root, "value_steps")
    f_vs = _seed_concat(fb_seed_root, "value_steps")
    g_vl = _seed_concat(gciql_root, "value_landscape")
    f_vl = _seed_concat(fb_seed_root, "value_landscape")
    g_cv = _seed_concat(gciql_root, "coverage")
    f_cv = _seed_concat(fb_seed_root, "coverage")

    # 1. value vs cube->goal distance (FB & GCIQL x success/fail)
    if g_vs is not None and f_vs is not None:
        pool = np.concatenate([g_vs["d"].to_numpy(),
                               f_vs["d"].to_numpy()])
        edges = np.quantile(pool, np.linspace(0, 1, 10))
        edges = np.unique(edges)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(6, 4))
        for nm, df in (("FB", f_vs), ("GCIQL", g_vs)):
            for oc, ls in (("success", "-"),
                           ("transport_fail", "--")):
                sub = df[df.outcome == oc]
                if sub.empty:
                    continue
                idx = np.clip(np.digitize(sub["d"], edges) - 1, 0,
                              len(ctr) - 1)
                mv = (pd.DataFrame({"b": idx, "V": sub["V"].to_numpy()})
                      .groupby("b")["V"].mean())
                ax.plot(ctr[mv.index], mv.values, ls, marker="o",
                        label=f"{nm} {oc}")
        ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("V(s)")
        ax.set_title("Value vs goal distance: FB vs GCIQL")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_value_vs_dist.png", dpi=120)
        plt.close(fig)
    else:
        print("  [warn] cmp_value_vs_dist skipped (missing value_steps)")

    # 2. per-episode rho distribution (box) for the four groups
    if g_vl is not None and f_vl is not None:
        groups, labels = [], []
        for nm, df in (("FB", f_vl), ("GCIQL", g_vl)):
            for oc in ("success", "transport_fail"):
                v = df[df.outcome == oc]["rho_V_negd"].dropna()
                groups.append(v.to_numpy() if len(v) else np.array([0.0]))
                labels.append(f"{nm}\n{oc}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_ylabel("Spearman rho(V, -d)")
        ax.set_title("Per-episode value gradient: FB vs GCIQL")
        ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_rho_box.png", dpi=120)
        plt.close(fig)

    # 3. outcome funnel (normalized fraction) FB vs GCIQL
    if g_vl is not None and f_vl is not None:
        order = ["success", "transport_fail", "other"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(order))
        for i, (nm, df) in enumerate((("FB", f_vl), ("GCIQL", g_vl))):
            fr = (df["outcome"].value_counts(normalize=True)
                  .reindex(order).fillna(0.0))
            ax.bar(x + (i - 0.5) * 0.4, fr.values, 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel("episode fraction")
        ax.set_title("Outcome funnel: FB vs GCIQL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cdir / "cmp_funnel.png", dpi=120)
        plt.close(fig)

    # 4. coverage nn_dist by region FB vs GCIQL
    if g_cv is not None and f_cv is not None:
        regs = sorted(set(g_cv["region"]) | set(f_cv["region"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(regs))
        for i, (nm, df) in enumerate((("FB", f_cv), ("GCIQL", g_cv))):
            m = df.groupby("region")["nn_dist"].mean()
            ax.bar(x + (i - 0.5) * 0.4,
                   [float(m.get(r, 0.0)) for r in regs], 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(regs)
        ax.set_ylabel("nn_dist"); ax.legend()
        ax.set_title("Off-support coverage: FB vs GCIQL")
        fig.tight_layout()
        fig.savefig(cdir / "cmp_coverage.png", dpi=120)
        plt.close(fig)

    (cdir / "comparison.md").write_text("\n".join([
        "# FB vs GCIQL — shared-axis comparison", "",
        f"- FB    value gradient: rho_s={fb_rs:.3g}, rho_f={fb_rf:.3g} "
        f"-> {fbv[0]}",
        f"- GCIQL value gradient: rho_s={g_rs:.3g}, rho_f={g_rf:.3g} "
        f"-> {g_verd['T1'][0]}",
        f"- GCIQL coverage transport nn_dist: fail={g_fnn:.3g}, "
        f"success={g_snn:.3g} -> {g_verd['T4'][0]}",
        "",
        "Both methods are compared only on axes that transfer "
        "(value gradient, coverage, phase funnel); FB-specific "
        "z-decoding (T2) and B-resolution (T3) have no GCIQL analog.",
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "gciql_profile"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fb-aggregate",
                    default=str(REPO_ROOT / "analysis"
                               / "representation_profile" / "aggregate"))
    ap.add_argument("--fb-seed-root",
                    default=str(REPO_ROOT / "analysis"
                               / "representation_profile"))
    args = ap.parse_args()
    root = Path(args.root)
    fb = Path(args.fb_aggregate)
    fbsr = Path(args.fb_seed_root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None,
              fb_seed_root=fbsr if fbsr.exists() else None)


if __name__ == "__main__":
    main()
