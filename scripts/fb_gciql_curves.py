"""scripts/fb_gciql_curves.py — FB vs GCIQL eval curves + IQM table.

Raw training-eval success for FB (wandb cache) and GCIQL (local eval.csv),
per cube task and aggregated across all 5 tasks, with rliable IQM and
95% stratified-bootstrap CIs for both the sample-efficiency curves and a
final success-rate table.

FB seeds with a failed run (no 1M eval) are forward-filled to the common
grid by np.interp (holds the last logged value), per the agreed policy.
GCIQL's step-1 row is treated as ~step 0 on the shared grid.

Usage:
    python scripts/fb_gciql_curves.py \
        --fb analysis/wandb_data_fbgciql \
        --gciql results/gciql_20260518_201030/factored-fb/factored-fb-gciql \
        --out analysis/fb_gciql
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Canonical 5 cube tasks, in the order GCIQL's eval.csv lists them.
TASKS: List[str] = ["task1", "task2", "task3", "task4", "task5"]
TASK_LABELS = {
    "task1": "task1 (horizontal)", "task2": "task2 (vertical1)",
    "task3": "task3 (vertical2)", "task4": "task4 (diagonal1)",
    "task5": "task5 (diagonal2)",
}

FB_OVERALL = "eval/reward/eval/success"
def _fb_task_key(t: str) -> str:
    n = t[-1]
    return f"eval/reward/cube-single-play-singletask-task{n}-v0/success"

GCIQL_OVERALL = "evaluation/overall_success"
GCIQL_TASK_COLS = {
    "task1": "evaluation/task1_horizontal_success",
    "task2": "evaluation/task2_vertical1_success",
    "task3": "evaluation/task3_vertical2_success",
    "task4": "evaluation/task4_diagonal1_success",
    "task5": "evaluation/task5_diagonal2_success",
}

# Shared 100k-resolution grid (GCIQL's native resolution; FB is 50k).
COMMON_GRID = np.arange(0, 1_000_001, 100_000)


def load_fb(cache_dir: Path | str) -> Dict[int, pd.DataFrame]:
    """seed -> DataFrame[step, task1..5, overall] from a wandb_pull cache."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "_meta.json").read_text())
    out: Dict[int, pd.DataFrame] = {}
    for m in meta:
        h = pd.read_parquet(cache_dir / f"{m['id']}.parquet")
        cols = {"step": h["_step"], "overall": h.get(FB_OVERALL)}
        for t in TASKS:
            cols[t] = h.get(_fb_task_key(t))
        df = pd.DataFrame(cols).dropna(subset=["overall"]).reset_index(drop=True)
        out[int(m["config"]["seed"])] = df.sort_values("step")
    return out


def load_gciql(root: Path | str) -> Dict[int, pd.DataFrame]:
    """seed -> DataFrame[step, task1..5, overall] from per-seed eval.csv."""
    out: Dict[int, pd.DataFrame] = {}
    for p in sorted(glob.glob(str(Path(root) / "sd*/eval.csv"))):
        seed = int(Path(p).parent.name.split("_")[0][2:])  # sd007_... -> 7
        g = pd.read_csv(p)
        df = pd.DataFrame({
            "step": g["step"],
            "overall": g[GCIQL_OVERALL],
            **{t: g[GCIQL_TASK_COLS[t]] for t in TASKS},
        }).sort_values("step").reset_index(drop=True)
        out[seed] = df
    return out


def interp_tensor(data: Dict[int, pd.DataFrame], cols: List[str],
                  grid: np.ndarray = COMMON_GRID) -> np.ndarray:
    """[n_seeds, n_cols, n_frames] via per-series np.interp onto `grid`.

    np.interp holds endpoints: values past a seed's last logged step are
    the last value (forward-fill for failed FB seeds); steps before the
    first are the first value (GCIQL's step-1 row -> ~step 0).
    """
    seeds = sorted(data)
    ten = np.empty((len(seeds), len(cols), len(grid)), dtype=float)
    for i, s in enumerate(seeds):
        df = data[s]
        xp = df["step"].to_numpy(dtype=float)
        for j, c in enumerate(cols):
            ten[i, j] = np.interp(grid, xp, df[c].to_numpy(dtype=float))
    return ten


def final_scores(data: Dict[int, pd.DataFrame],
                  cols: List[str]) -> np.ndarray:
    """[n_seeds, n_cols] = each seed's last logged value per column.

    For failed FB seeds this is the 950k eval (their latest checkpoint);
    for finished seeds / GCIQL it is the 1M eval.
    """
    seeds = sorted(data)
    out = np.empty((len(seeds), len(cols)), dtype=float)
    for i, s in enumerate(seeds):
        last = data[s].iloc[-1]
        out[i] = [float(last[c]) for c in cols]
    return out


# ---------------------------------------------------------------------------
# rliable aggregation
# ---------------------------------------------------------------------------

def _patch_rliable_for_arch8() -> None:
    """arch>=8 renamed IIDBootstrap's RNG kwarg ``random_state`` -> ``seed``;
    rliable still passes ``random_state=``, so arch treats it as bootstrap
    data and raises. arch<8 can't be used (it fails to import under
    pandas>=3). rliable's ``StratifiedBootstrap.update_indices`` uses the
    global numpy RNG, not arch's generator, so translating just the
    constructor kwarg is a complete fix."""
    from rliable import library as rly
    import arch.bootstrap as ab

    SB = rly.StratifiedBootstrap
    if getattr(SB, "_arch8_patched", False):
        return

    def __init__(self, *args, random_state=None, task_bootstrap=False,
                 **kwargs):
        ab.IIDBootstrap.__init__(self, *args, seed=random_state, **kwargs)
        self._args_shape = args[0].shape
        self._num_tasks = self._args_shape[1]
        self._parameters = [self._num_tasks, task_bootstrap]
        self._task_bootstrap = task_bootstrap
        self._strata_indices = self._get_strata_indices()

    SB.__init__ = __init__
    SB._arch8_patched = True


def _iqm_fn(x: np.ndarray) -> np.ndarray:
    from rliable import metrics
    return np.array([metrics.aggregate_iqm(x)])


def iqm_curve(tensor_by_method: Dict[str, np.ndarray], reps: int = 2000):
    """Per-frame IQM + 95% stratified-bootstrap CI across (seeds x tasks).

    Input tensors are [n_seeds, n_tasks, n_frames]. Returns
    (point[method] -> [n_frames], ci[method] -> [2, n_frames]).
    """
    from rliable import library as rly
    _patch_rliable_for_arch8()
    frames = list(range(tensor_by_method[next(iter(tensor_by_method))].shape[-1]))
    pt, ci = rly.get_interval_estimates(
        tensor_by_method,
        lambda scores: np.array(
            [_iqm_fn(scores[..., f])[0] for f in frames]),
        reps=reps,
    )
    return pt, ci


def iqm_table_estimates(final_by_method: Dict[str, np.ndarray],
                        reps: int = 5000):
    """IQM + 95% CI per task and overall. final tensors are
    [n_seeds, n_tasks(+overall)]; returns (point, ci) keyed by method,
    each value an array over the columns."""
    from rliable import library as rly
    _patch_rliable_for_arch8()
    n = final_by_method[next(iter(final_by_method))].shape[1]
    return rly.get_interval_estimates(
        final_by_method,
        lambda s: np.array([_iqm_fn(s[:, [k]])[0] for k in range(n)]),
        reps=reps,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# rliable-style fixed method colors.
COLORS = {"FB": "#d62728", "GCIQL": "#1f77b4"}


def _plot_curve(ax, pt, ci, title: str) -> None:
    for method in pt:
        c = COLORS.get(method, None)
        ax.plot(COMMON_GRID, pt[method], label=method, color=c, lw=2)
        ax.fill_between(COMMON_GRID, ci[method][0], ci[method][1],
                        color=c, alpha=0.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("env steps")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def render(fb: Dict[int, pd.DataFrame], gc: Dict[int, pd.DataFrame],
           out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task sample-efficiency curves (IQM across the 10 seeds per task).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), TASKS):
        pt, ci = iqm_curve({
            "FB": interp_tensor(fb, [t])[:, :, :],
            "GCIQL": interp_tensor(gc, [t])[:, :, :],
        })
        _plot_curve(ax, pt, ci, TASK_LABELS[t])
        ax.set_ylabel("IQM success")
    # 6th panel: aggregate across all 5 tasks.
    pt_all, ci_all = iqm_curve({
        "FB": interp_tensor(fb, TASKS),
        "GCIQL": interp_tensor(gc, TASKS),
    })
    _plot_curve(axes.ravel()[5], pt_all, ci_all, "aggregate (all 5 tasks)")
    axes.ravel()[5].set_ylabel("IQM success")
    axes.ravel()[5].legend(loc="lower right", fontsize=9)
    fig.suptitle("FB vs GCIQL — cube-single eval success (IQM, 95% CI)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_pertask.png", dpi=130)
    plt.close(fig)

    # Standalone aggregate curve.
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve(ax, pt_all, ci_all, "FB vs GCIQL — aggregate across 5 tasks")
    ax.set_ylabel("IQM success (95% stratified bootstrap CI)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_aggregate.png", dpi=130)
    plt.close(fig)

    # Final IQM table (last eval per seed; failed FB seeds = 950k).
    cols = TASKS + ["overall"]
    final = {"FB": final_scores(fb, cols), "GCIQL": final_scores(gc, cols)}
    pt, ci = iqm_table_estimates(final)
    hdr = ["Method"] + [TASK_LABELS[t] for t in TASKS] + ["overall"]
    md = ["# FB vs GCIQL — final eval success (IQM, 95% CI)", "",
          "Final eval per seed (failed FB seeds use their 950k checkpoint; "
          "n=10 seeds each). IQM with 95% stratified-bootstrap CI.", "",
          "| " + " | ".join(hdr) + " |",
          "| " + " | ".join([":---"] * len(hdr)) + " |"]
    for method in ("FB", "GCIQL"):
        row = [method]
        for k in range(len(cols)):
            lo, hi = ci[method][0][k], ci[method][1][k]
            row.append(f"{pt[method][k]*100:.1f} "
                       f"[{lo*100:.1f}, {hi*100:.1f}]")
        md.append("| " + " | ".join(row) + " |")
    (out_dir / "iqm_table.md").write_text("\n".join(md) + "\n")

    # LaTeX booktabs version for the paper.
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & ".join(["Method"] + [t for t in TASKS] + ["Overall"])
           + r" \\", r"\midrule"]
    for method in ("FB", "GCIQL"):
        cells = [method]
        for k in range(len(cols)):
            lo, hi = ci[method][0][k] * 100, ci[method][1][k] * 100
            cells.append(f"{pt[method][k]*100:.1f} "
                         f"\\scriptsize[{lo:.1f}, {hi:.1f}]")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "iqm_table.tex").write_text("\n".join(tex) + "\n")

    print(f"[fb_gciql_curves] wrote curves + table to {out_dir}")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fb", default=str(REPO_ROOT / "analysis"
                                        / "wandb_data_fbgciql"))
    ap.add_argument("--gciql", default=str(
        REPO_ROOT / "results" / "gciql_20260518_201030"
        / "factored-fb" / "factored-fb-gciql"))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "fb_gciql"))
    args = ap.parse_args()
    fb = load_fb(args.fb)
    gc = load_gciql(args.gciql)
    print(f"[fb_gciql_curves] FB seeds={sorted(fb)} "
          f"GCIQL seeds={sorted(gc)}")
    render(fb, gc, Path(args.out))


if __name__ == "__main__":
    main()
