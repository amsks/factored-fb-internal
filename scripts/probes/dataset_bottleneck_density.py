"""Appendix figure: how rare are control-bottleneck transitions in the offline datasets?

A control bottleneck is operationalized as a MOTION ONSET: the object goes from at-rest at t to
in-motion at t+1 -- the decisive commitment point where the successful action sends the object
onto a thin spread of transient futures and the unsuccessful one leaves it on the persistent
at-rest set (the sparsity the argument in Sec. failure rests on). We also report the broader
"object in motion" fraction as a loose upper bound.

Two dataset regimes, kept visually separate because they are NOT the same kind of data:
  OGBench (play)    -- unlabeled play; object blocks are the measure hit-spec slices.
  MetaWorld (expert)-- scripted/TD-MPC2 experts that finish early and hold at the goal (hence the
                       0.76-0.95 at-goal fraction reported in data-gen-mw-ms.tex); object = obs[4:7]
                       (V2 convention). Local packs TommyShen/Factor-FB-Offline (N=1000), which are
                       NOT the exact packs the main table used -- a structural probe, not a table.
ManiSkill is omitted: its object columns are not documented here (achieved_goal is a zeros
placeholder per the pack meta.json), so its object motion cannot be measured without guessing.

    MUJOCO_GL=egl python scripts/probes/dataset_bottleneck_density.py
"""
import os
import numpy as np

EPS = 0.005
OUT = "PAPER/ICLR/figures/appendix"
MW_ROOT = "/home/adityamohan/git/datasets/Factor-FB-Offline/data/metaworld"

# OGBench play datasets: object position block(s) = the measure hit-spec slices
OGB = {
    "cube-single-play-v0": [(19, 22)],
    "cube-double-play-v0": [(19, 22), (28, 31)],
    "scene-play-v0": [(19, 22), (28, 30), (32, 34), (36, 37), (38, 39)],
}
# MetaWorld expert packs: object xyz = obs[4:7] (goal-observable V2, previous-frame half at 18:36)
MW = ["mw-assembly", "mw-basketball", "mw-bin-picking", "mw-box-close", "mw-button-press"]


def _onset_inmotion_flat(disp, valids):
    """disp, valids: 1-D over transitions in dataset order. Onset uses consecutive valid pairs."""
    d = disp
    inmotion = float((d[valids] >= EPS).mean())
    pair = valids[:-1] & valids[1:]                       # t and t+1 both real transitions
    onset = ((d[:-1] < EPS) & (d[1:] >= EPS) & pair)
    return inmotion, float(onset.sum() / max(pair.sum(), 1))


def ogbench_stats(name, blocks):
    import ogbench
    _, train, _ = ogbench.make_env_and_datasets(name, compact_dataset=True)
    obs = np.asarray(train["observations"], np.float64)
    valids = (np.asarray(train.get("valids", np.ones(len(obs)))) > 0)
    idx = np.arange(len(obs)); nxt = np.minimum(idx + 1, len(obs) - 1)
    disp = np.zeros(len(obs))
    for (a, b) in blocks:
        disp += np.linalg.norm(obs[nxt, a:b] - obs[:, a:b], axis=-1)
    inm, ons = _onset_inmotion_flat(disp, valids)
    return dict(dataset=name.replace("-play-v0", ""), regime="OGBench (play)",
                median=float(np.median(disp[valids])), inmotion=inm, onset=ons)


def metaworld_stats(task):
    o = np.load(f"{MW_ROOT}/{task}.npz")["observation"].astype(np.float64)[:, :, 4:7]  # [N,T,3]
    disp = np.linalg.norm(o[:, 1:] - o[:, :-1], axis=-1)                                # [N,T-1]
    onset = (disp[:, :-1] < EPS) & (disp[:, 1:] >= EPS)
    return dict(dataset=task.replace("mw-", ""), regime="MetaWorld (expert)",
                median=float(np.median(disp)), inmotion=float((disp >= EPS).mean()),
                onset=float(onset.mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    rows = [ogbench_stats(n, b) for n, b in OGB.items()] + [metaworld_stats(t) for t in MW]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/dataset_bottleneck_density.csv", index=False)
    for r in rows:
        print(f"{r['dataset']:14s} [{r['regime']:>18}]  median|dObj|={r['median']:.4f}  "
              f"in-motion={r['inmotion']:.3f}  onset={r['onset']:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows); x = np.arange(n); w = 0.4
    fig, ax = plt.subplots(figsize=(6.6, 2.7), dpi=200)
    ax.bar(x - w / 2, [r["inmotion"] * 100 for r in rows], w, label="object in motion", color="#c9c9c9")
    ax.bar(x + w / 2, [r["onset"] * 100 for r in rows], w, label="motion onset (bottleneck)", color="#c44e52")
    for xi, r in zip(x + w / 2, rows):
        ax.text(xi, r["onset"] * 100 + 0.4, f"{r['onset']*100:.1f}", ha="center", fontsize=6)
    # divider + group labels between the two regimes
    n_ogb = sum(r["regime"].startswith("OGBench") for r in rows)
    ax.set_ylim(0, 58)
    ax.axvline(n_ogb - 0.5, color="0.6", lw=0.7, ls=":")
    ax.text((n_ogb - 1) / 2, 53, "OGBench (play)", ha="center", fontsize=7.5, color="0.35")
    ax.text(n_ogb + 0.6, 53, "MetaWorld (expert)", ha="left", fontsize=7.5, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels([r["dataset"] for r in rows], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("% of transitions", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.86))
    ax.set_title("Control-bottleneck transitions are rare across datasets", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/dataset_bottleneck_density.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/dataset_bottleneck_density.png", bbox_inches="tight")
    print("wrote", f"{OUT}/dataset_bottleneck_density.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
