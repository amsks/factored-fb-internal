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


def aggregate(root, out, fb_aggregate=None) -> None:
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
                    rho_s, rho_f, fail_nn, succ_nn, verdicts)
    print(f"[gciql_profile_aggregate] {vl['seed'].nunique()} seeds -> {out}")


def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd) -> None:
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
    args = ap.parse_args()
    root = Path(args.root)
    fb = Path(args.fb_aggregate)
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None)


if __name__ == "__main__":
    main()
