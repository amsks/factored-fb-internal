"""Master CSV + figure for the trajectory value profile (run in .venv).

Combines the per-step FB/GCIQL trajectory parquets (from traj_value_profile, whole
training dataset, hindsight goal) into one master CSV, and renders:
  - aggregate value-vs-distance + value-vs-progress over ALL relabelled trajectories,
  - workspace overlays of example trajectories (success: value rises near goal;
    failure: it does not), value-colored cube paths on the MuJoCo top-down + goal ★.

  .venv/bin/python -m scripts.traj_value_master \
    --fb analysis/traj_value/fb.parquet --gciql analysis/traj_value/gciql.parquet \
    --out-dir analysis/traj_value
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).resolve().parent.parent


def _norm(df):
    g = df.groupby(["method", "traj"])["V"]
    vmin, vmax = g.transform("min"), g.transform("max")
    r = vmax - vmin
    df["Vn"] = np.where(r > 1e-9, (df["V"] - vmin) / r.where(r > 1e-9, 1.0), 0.5)
    return df


def _binned(x, y, bins):
    idx = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    m = np.full(len(bins) - 1, np.nan); s = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        yb = y[idx == b]
        if len(yb) > 20:
            m[b] = yb.mean(); s[b] = yb.std() / np.sqrt(len(yb))
    return ctr, m, s


def _bg_ext():
    sx = REPO_ROOT / "analysis" / "scene"
    cal = json.loads((sx / "calib.json").read_text())
    bg = (mpimg.imread(str(sx / "topdown.png"))
          if (sx / "topdown.png").exists() and cal.get("photoreal") else None)
    ws = cal["workspace"]
    return bg, [ws["xmin"], ws["xmax"], ws["ymin"], ws["ymax"]]


def _pick(df, summ, method, kind, min_span=0.12, min_n=20):
    s = summ[summ.method == method].copy()
    # require a real approach (cube covers distance) and enough points
    span = df[df.method == method].groupby("traj")["d"].agg(lambda x: x.max() - x.min())
    s = s[s.traj.map(span).fillna(0) >= min_span]
    s = s[s.n >= min_n]
    if s.empty:
        return None, np.nan
    row = s.loc[s.goal_peak_rho.idxmax() if kind == "success" else s.goal_peak_rho.idxmin()]
    g = df[(df.method == method) & (df.traj == row.traj)].sort_values("t")
    return g, float(row.goal_peak_rho)


def _overlay(ax, g, bg, ext, title):
    if bg is not None:
        ax.imshow(bg, extent=ext, aspect="auto", zorder=0, interpolation="bilinear")
    else:
        ax.set_facecolor("0.9")
    ax.plot(g["cube_x"], g["cube_y"], color="white", lw=1.0, alpha=0.5, zorder=2)
    sc = ax.scatter(g["cube_x"], g["cube_y"], c=g["Vn"], cmap="viridis", s=22,
                    vmin=0, vmax=1, zorder=3, edgecolors="black", linewidths=0.2)
    gp = g.loc[g["d"].idxmin()]
    ax.scatter([gp["cube_x"]], [gp["cube_y"]], marker="*", s=320, c="gold",
               edgecolors="black", linewidths=1.2, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", required=True); ap.add_argument("--gciql", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fb = pd.read_parquet(args.fb); gq = pd.read_parquet(args.gciql)
    master = _norm(pd.concat([fb, gq], ignore_index=True))
    master.to_csv(out / "master.csv", index=False)
    summ = pd.concat([pd.read_csv(Path(args.fb).with_name(Path(args.fb).stem + "_summary.csv")),
                      pd.read_csv(Path(args.gciql).with_name(Path(args.gciql).stem + "_summary.csv"))],
                     ignore_index=True)
    summ.to_csv(out / "summary.csv", index=False)

    bg, ext = _bg_ext()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.28, wspace=0.12)
    axd = fig.add_subplot(gs[0, 0:2]); axp = fig.add_subplot(gs[0, 2:4])
    series = [("fb", "FB", "tab:red"), ("gciql", "GCIQL", "tab:blue")]

    # Aggregate value vs distance-to-goal (whole dataset).
    dmax = float(np.percentile(master["d"], 97)); dbins = np.linspace(0, dmax, 18)
    for mth, lab, c in series:
        d = master[master.method == mth]
        ctr, m, s = _binned(d["d"].to_numpy(), d["Vn"].to_numpy(), dbins)
        axd.plot(ctr, m, color=c, lw=2, label=lab); axd.fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axd.invert_xaxis(); axd.set_xlabel("cube→goal distance  (→ approaching goal)")
    axd.set_ylabel("normalized value (per-traj)"); axd.legend()
    axd.set_title("Value vs distance-to-goal — ALL training trajectories (hindsight)\n"
                  "rise toward d→0 = value peaks at goal")

    # Aggregate value vs progress.
    pbins = np.linspace(0, 1, 18)
    for mth, lab, c in series:
        d = master[master.method == mth]
        ctr, m, s = _binned(d["progress"].to_numpy(), d["Vn"].to_numpy(), pbins)
        axp.plot(ctr, m, color=c, lw=2, label=lab); axp.fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axp.set_xlabel("progress to goal step (t / t_goal)"); axp.set_ylabel("normalized value")
    axp.legend(); axp.set_title("Value vs progress — ALL training trajectories")

    # Example overlays: success (value rises) + failure (it doesn't) per method.
    cells = [("fb", "success"), ("fb", "failure"), ("gciql", "success"), ("gciql", "failure")]
    sc = None
    for j, (mth, kind) in enumerate(cells):
        ax = fig.add_subplot(gs[1, j])
        g, rho = _pick(master, summ, mth, kind)
        lab = dict(fb="FB", gciql="GCIQL")[mth]
        if g is None:
            ax.set_title(f"{lab} {kind}: none"); ax.set_xticks([]); ax.set_yticks([]); continue
        sc = _overlay(ax, g, bg, ext, f"{lab} — {kind} example  (ρ(V,−d)={rho:.2f}, ★=goal)")
    if sc is not None:
        fig.colorbar(sc, ax=fig.axes[2:], fraction=0.015, pad=0.01,
                     label="normalized value (per-traj)")

    fig.suptitle("Goal-conditioned value along real training trajectories (hindsight goal): "
                 "GCIQL value rises toward the goal; FB stays flat (B can't resolve the goal)",
                 fontsize=12)
    fig.savefig(out / "traj_value_master.png", dpi=130, bbox_inches="tight")
    print(f"[master] {master['traj'].nunique()} trajs/method; wrote {out}/master.csv, "
          f"summary.csv, traj_value_master.png")
    print("  goal_peak_rho mean:  " +
          "  ".join(f"{m}={summ[summ.method==m].goal_peak_rho.mean():.3f}" for m in ["fb", "gciql"]))


if __name__ == "__main__":
    main()
