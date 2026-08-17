#!/usr/bin/env python3
"""scripts/goal_coverage.py — E4 of docs/specs/2026-08-10-deconfound-size-coverage.md:
is the eval-goal set inside the training goal marginal, and by how much?

The committed `PAPER/decoding-bilinear/figs/goal_coverage.png` was built from MW v1 and the
48k upstream MS set, and it is a picture. This replaces it with (a) the same picture from v3
MW / the E2 MS data / cube and (b) the number the picture was standing in for: for each eval
goal, the distance to the nearest training achieved goal, in units of that env's success
radius. C2 is controlled when that number is small; Arm D is the causal anchor for what
happens when it is not.

Two passes, because the three families need three different venvs (Metaworld and cube under
.venv-jax-fb, ManiSkill under .venv-ms) and neither can import the other's backend:

    # one extract per task, each under its own venv, on a COMPUTE node (envs are needed)
    $SCRATCH/.venv-jax-fb/bin/python scripts/goal_coverage.py extract --task cube-single \
        --out results/deconfound/cov-cube.npz
    $SCRATCH/.venv-jax-fb/bin/python scripts/goal_coverage.py extract --task mw-push \
        --data-root $SCRATCH/fb-data/play-resample --out results/deconfound/cov-mw-push.npz
    $SCRATCH/.venv-ms/bin/python  scripts/goal_coverage.py extract --task ms-pick-cube \
        --data-root $SCRATCH/fb-data/ms-play-resample --out results/deconfound/cov-ms-pick.npz

    # then merge, on a login node (numpy + matplotlib only)
    python scripts/goal_coverage.py report --in results/deconfound/cov-*.npz \
        --fig PAPER/decoding-bilinear/figs/goal_coverage.png \
        --json-out results/deconfound/goal-coverage.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Success radius in the env's OWN metric, and how many leading components that metric uses
# (PushCube scores planar XY distance to a goal region; everything else is 3-D). Distances are
# reported in units of this radius so the three families are commensurable.
RADIUS = {"cube-single": (0.04, 3), "mw-push": (0.05, 3), "mw-pick-place": (0.07, 3),
          "ms-push-cube": (0.10, 2), "ms-pick-cube": (0.025, 3)}
# OGBench cube-single obs (28-d): block xyz is centred on (0.425,0,0) and scaled x10.
CUBE_BLOCK = slice(19, 22)
MAX_TRAIN = 200_000        # subsample of training achieved goals for the nearest-neighbour


def _cube_metres(a):
    return np.asarray(a, np.float64)[..., CUBE_BLOCK] / 10.0 + np.array([0.425, 0.0, 0.0])


def extract(task, data_root, episodes, out):
    """-> npz with training achieved goals, per-episode eval targets, and the `fixed`
    protocol's single conditioning goal. Needs the env, so: compute node."""
    radius, k = RADIUS[task]

    if task == "cube-single":
        # The OGBench path draws its eval goals from the env's fixed task_infos, so there is
        # no `fixed` conditioning frame to compare against and no synth mode.
        from impls.utils.env_utils import make_env_and_datasets
        env, train_ds, _ = make_env_and_datasets("cube-single-play-v0")
        train = _cube_metres(np.asarray(train_ds["observations"]))
        goals = []
        for tid in range(1, len(env.unwrapped.task_infos) + 1):
            _obs, info = env.reset(options=dict(task_id=tid, render_goal=False))
            goals.append(_cube_metres(np.asarray(info["goal"], np.float64)))
        eval_goals = np.stack(goals)
        fixed = None
    else:
        from env.unified import goal_layout, load_packed, seeded_reset, make_task_env
        lay = goal_layout(task)
        family = "metaworld" if task.startswith("mw-") else "maniskill"
        data = load_packed(os.path.join(data_root, "data", family, f"{task}.npz"))
        nobs = data["next_observations"]
        train = np.asarray(nobs[:, lay["object"][0]:lay["object"][1]], np.float64)
        # The `fixed` protocol's conditioning frame: the highest-reward next_obs, exactly what
        # UnifiedAdapter caches. Its OBJECT slot is the goal a critic is actually pointed at.
        fixed = np.asarray(nobs[int(np.argmax(data["rewards"])),
                                lay["object"][0]:lay["object"][1]], np.float64)
        env, _h = make_task_env(task, render=False)
        goals = []
        for i in range(episodes):
            obs, _info = seeded_reset(env, task, i)
            goals.append(np.asarray(obs, np.float64)[lay["target"][0]:lay["target"][1]])
        eval_goals = np.stack(goals)
        env.close()

    if len(train) > MAX_TRAIN:
        train = train[np.random.RandomState(0).choice(len(train), MAX_TRAIN, replace=False)]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = dict(task=np.array(task), train=train.astype(np.float32),
                   eval_goals=eval_goals.astype(np.float32),
                   radius=np.array(radius), ndim=np.array(k))
    if fixed is not None:
        payload["fixed"] = fixed.astype(np.float32)
    np.savez_compressed(out, **payload)
    print(f"[coverage] {task}: {len(train)} training achieved goals, "
          f"{len(eval_goals)} eval goals -> {out}")


def _nn_dist(query, ref, k):
    """min over `ref` of ||query - ref||, in the env's own metric (k leading components)."""
    q, r = np.asarray(query)[:, :k], np.asarray(ref)[:, :k]
    out = np.empty(len(q))
    for i in range(0, len(q), 64):                    # chunked: ref is up to 200k rows
        d = np.linalg.norm(r[None, :, :] - q[i:i + 64, None, :], axis=-1)
        out[i:i + 64] = d.min(1)
    return out


def report(paths, fig_path, json_out):
    rows = []
    for p in paths:
        z = np.load(p)
        task = str(z["task"])
        radius, k = float(z["radius"]), int(z["ndim"])
        train, ev = z["train"], z["eval_goals"]
        ins = _nn_dist(ev, train, k) / radius
        row = dict(task=task, radius=radius, n_train=int(len(train)), n_eval=int(len(ev)),
                   # E4's scalar: under eval_goal_mode=synth this should be ~in-support
                   # everywhere, which is the point -- it SHOWS C2 is controlled rather than
                   # asserting it.
                   insupport_median=float(np.median(ins)), insupport_p95=float(np.percentile(ins, 95)),
                   train=train, eval_goals=ev, k=k)
        if "fixed" in z.files:
            # What the `fixed` protocol actually conditioned on, against what the env scored:
            # the distance between the ONE cached dataset goal and each episode's true target.
            # This is C4 as a number. Under synth it is 0 by construction.
            err = np.linalg.norm(np.asarray(z["fixed"])[:k] - ev[:, :k], axis=-1) / radius
            row["fixed_err_median"] = float(np.median(err))
            row["fixed_err_p95"] = float(np.percentile(err, 95))
        rows.append(row)
    rows.sort(key=lambda r: r["task"])

    print(f"\n{'task':<16}{'radius':>8}{'n_train':>10}{'n_eval':>8}"
          f"{'in-support med':>16}{'p95':>8}{'fixed-err med':>16}{'p95':>8}")
    print("-" * 90)
    for r in rows:
        fm = r.get("fixed_err_median")
        print(f"{r['task']:<16}{r['radius']:>8.3f}{r['n_train']:>10}{r['n_eval']:>8}"
              f"{r['insupport_median']:>16.3f}{r['insupport_p95']:>8.3f}"
              + (f"{fm:>16.2f}{r['fixed_err_p95']:>8.2f}" if fm is not None
                 else f"{'--':>16}{'--':>8}"))
    print("\n(distances are in units of each env's success radius; 'fixed-err' is the gap "
          "between the ONE cached dataset goal the fixed protocol conditions on and the "
          "episode's true target -- 0 by construction under eval_goal_mode=synth)")

    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        with open(json_out, "w") as f:
            json.dump([{k: v for k, v in r.items() if k not in ("train", "eval_goals")}
                       for r in rows], f, indent=2)
        print(f"[coverage] wrote {json_out}")
    if fig_path:
        _plot(rows, fig_path)


def _plot(rows, fig_path):
    """Small multiples: the training goal marginal as a single-hue density, the eval goals as
    points on top. The question is spatial containment -- does the eval set sit inside the
    training marginal -- so a 2-D density with the eval set overlaid is the form; a summary
    bar would throw away the only thing worth looking at.

    Two encodings, so two colors: a sequential blue ramp for density (magnitude, one hue,
    light to dark) and categorical slot 2 orange for the eval goals (identity). Blue/orange
    are slots 1-2 of the reference palette, which validate all-pairs in both modes (CVD
    dE 9.2 light). Each eval marker carries a surface-colored ring so it reads on top of any
    density level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    ramp = LinearSegmentedColormap.from_list("cov", [SURFACE, "#a9c9ee", BLUE, "#123a68"])

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), facecolor=SURFACE, squeeze=False)
    for ax, r in zip(axes[0], rows):
        tr, ev = r["train"], r["eval_goals"]
        ax.set_facecolor(SURFACE)
        ax.hexbin(tr[:, 0], tr[:, 1], gridsize=44, cmap=ramp, bins="log", linewidths=0)
        ax.scatter(ev[:, 0], ev[:, 1], s=34, c=ORANGE, marker="o",
                   edgecolors=SURFACE, linewidths=1.2, zorder=3, label="eval goals")
        ax.set_title(f"{r['task']}\nin-support {r['insupport_median']:.2f}"
                     f" / p95 {r['insupport_p95']:.2f} radii",
                     fontsize=9, color=INK, pad=6)
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        for s in ax.spines.values():
            s.set_color("#dcdbd6")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0][0].set_ylabel("y (m)", fontsize=8, color=MUTED)
    # Two encodings only, so the legend is two entries and lives once, outside the panels.
    axes[0][-1].legend(loc="upper right", fontsize=7, frameon=False, labelcolor=MUTED)
    fig.suptitle("Training achieved-goal marginal (density) vs eval goals (points)",
                 fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"[coverage] wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--task", required=True, choices=sorted(RADIUS))
    e.add_argument("--data-root", default=None, help="unified tasks only")
    e.add_argument("--episodes", type=int, default=50, help="eval episodes to draw goals for")
    e.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--in", dest="inputs", nargs="+", required=True)
    r.add_argument("--fig", default=None)
    r.add_argument("--json-out", default=None)
    a = ap.parse_args()

    if a.cmd == "extract":
        if a.task != "cube-single" and not a.data_root:
            raise SystemExit(f"--data-root is required for {a.task}")
        extract(a.task, a.data_root, a.episodes, a.out)
    else:
        paths = [p for pat in a.inputs for p in sorted(glob.glob(pat))]
        if not paths:
            raise SystemExit(f"no extract files matched {a.inputs}")
        report(paths, a.fig, a.json_out)


if __name__ == "__main__":
    main()
