"""Is Density-FB's concentration a PREDICTION or an ECHO of its conditioning?

fig_sharpness_map.py measures 13.6x the dataset prior's mass inside the goal's 4cm success
radius, from pre-grasp starts. The objection that figure cannot answer: the prediction is
conditioned on the goal, so a critic that always replied "the cube ends up exactly where
you asked" scores the same 13.6x while knowing nothing.

The test. Sweep the REQUESTED cube target across the table and watch the mass at that
target as a function of how far it is from the cube's current position. Distance is the
feasibility axis: a far target needs a longer transport, so more discounting and more
chances to fail, and a calibrated measure must put LESS mass there. An echo does not decay
-- it reports whatever it was asked for at every distance.

    decays with distance  -> the concentration is a prediction
    flat in distance      -> the concentration is an echo, and the sharpness figure is
                             measuring wishful thinking rather than knowledge

Everything is reported as LIFT over the dataset prior at the same target, because the pool's
own cube density varies across the table; raw mass would confound "the critic likes this
target" with "the data visits this target". That correction is what turned the earlier
gripper-relative reach map from an apparent result into the null it actually was.

Targets and goals are drawn from real dataset rows, never fabricated, for the reason
fig_outcome_map documents: the env's goal draw is not reproducible and a hand-edited goal
row is off-manifold.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck.fig_outcome_map import CM
from scripts.probes.cube_bottleneck.fig_reach_map import CUBE_OBS, batched_mass

ANALYSIS = Path("analysis/cube_bottleneck")
NEAR = 0.4          # 4 cm in scaled units: the benchmark's own success radius


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=["fb_ref", "density_fb_state"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--targets", type=int, default=12)
    ap.add_argument("--goal-draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    root = ANALYSIS / args.env
    banks = dict(np.load(root / "phase_banks.npz"))
    pool_obs = dict(np.load(root / "pool.npz"))["obs"]
    pool_xy = pool_obs[:, CUBE_OBS][:, :2]

    from scripts.probes.cube_bottleneck import banks as BK
    raw = BK.load_raw(args.env)
    ds_obs = np.asarray(raw["observations"], np.float32)
    ds_xy = ds_obs[:, CUBE_OBS][:, :2]

    # Targets spread over the table, each with enough pool support that a lift is meaningful.
    rng = np.random.default_rng(args.seed)
    cand = pool_xy[rng.choice(len(pool_xy), size=400, replace=False)]
    targets, prior_mass = [], []
    for t in cand:
        pm = float((np.linalg.norm(pool_xy - t, axis=1) < NEAR).mean())
        if pm > 0.005 and all(np.linalg.norm(t - u) > 0.5 for u in targets):
            targets.append(t), prior_mass.append(pm)
        if len(targets) >= args.targets:
            break
    targets, prior_mass = np.stack(targets), np.asarray(prior_mass)
    print(f"{len(targets)} targets; prior mass within 4cm ranges "
          f"{prior_mass.min():.3f}-{prior_mass.max():.3f}\n")

    goals_per_target = []
    for t in targets:
        near = np.flatnonzero(np.linalg.norm(ds_xy - t, axis=1) < NEAR)
        sel = np.random.default_rng(args.seed).choice(near, size=args.goal_draws,
                                                      replace=len(near) < args.goal_draws)
        goals_per_target.append(ds_obs[sel])

    obs_b = banks[f"{args.phase}_obs"][:args.states]
    act_b = banks[f"{args.phase}_action"][:args.states]
    state_xy = obs_b[:, CUBE_OBS][:, :2]

    edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0])          # scaled units -> 0-10-20-30-50 cm
    print(f"{'method':20s} " + "  ".join(f"{a*CM:.0f}-{b*CM:.0f}cm"
                                         for a, b in zip(edges[:-1], edges[1:])))
    for a in args.critics:
        lifts, dists = [], []
        for s in args.seeds:
            run = Path("exp") / f"{args.env}_{a}_flowbc_seed_{s}"
            if not (run / "params_500000.pkl").exists():
                continue
            arm = A.load_arm(run)
            for ti, t in enumerate(targets):
                near_t = np.linalg.norm(pool_xy - t, axis=1) < NEAR
                for i in range(len(obs_b)):
                    p, _ = batched_mass(arm, obs_b[i], act_b[i], goals_per_target[ti],
                                        pool_obs)
                    pm = p.mean(0)
                    lifts.append(float(pm[near_t].sum()) / max(prior_mass[ti], 1e-9))
                    dists.append(float(np.linalg.norm(t - state_xy[i])))
        lifts, dists = np.asarray(lifts), np.asarray(dists)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dists >= lo) & (dists < hi)
            cells.append(f"{lifts[m].mean():7.2f}x" if m.sum() > 5 else "     n/a")
        print(f"{a:20s} " + "  ".join(cells) + f"   [n={len(lifts)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
