#!/usr/bin/env python3
"""scripts/compare_play_resample_to_ogbench.py — is the Metaworld play-resample data
structurally the same object as OGBench's cube-single-play-v0?

Both are "play": long task-agnostic trajectories that chain target relocations. The claim
this script tests is that they agree on the properties that actually determine how
hindsight relabeling behaves, once each is expressed in units of its own eval horizon
(cube 200 steps, Metaworld 100 wrapper steps) rather than in raw steps.

    $SCRATCH/.venv-jax-fb/bin/python scripts/compare_play_resample_to_ogbench.py \
        --mw-root $SCRATCH/fb-data/play-resample \
        --cube ~/.ogbench/data/cube-single-play-v0.npz

Everything is recomputed from the arrays. Nothing is read from a meta sidecar or a doc.
"""
import argparse
import json
import os

import numpy as np

# Metaworld unified obs layout (39-d): [curr 18 | prev 18 | goal 3].
MW_OBJ = slice(4, 7)      # object xyz -- what success is measured on
MW_GOAL = slice(36, 39)   # current _target_pos
MW_EVAL_HORIZON = 100     # wrapper steps (env/registry.py)
# OGBench cube-single obs (28-d); block xyz is centred on (0.425,0,0) and scaled x10.
CUBE_BLOCK = slice(19, 22)
CUBE_EVAL_HORIZON = 200   # gym registration max_episode_steps
CUBE_SUCCESS_R = 0.04     # metres, ||block - target||
MW_SUCCESS_R = {"mw-push": 0.05, "mw-pick-place": 0.07}


def _segments_from_jumps(sig):
    """[N,T,·] signal that is piecewise-constant -> per-trajectory segment lengths."""
    chg = (np.abs(np.diff(sig, axis=1)) > 1e-9).any(-1)     # [N, T-1]
    out = []
    for i in range(chg.shape[0]):
        idx = np.flatnonzero(chg[i]) + 1
        bnds = np.concatenate([[0], idx, [sig.shape[1]]])
        out.append(np.diff(bnds))
    return out


def _bouts(pos, thr, min_run=3):
    """Contiguous runs where the object is moving -> (n_bouts, rest_fraction)."""
    mv = np.linalg.norm(np.diff(pos, axis=1), axis=-1) > thr   # [N, T-1]
    n = 0
    for i in range(mv.shape[0]):
        d = np.diff(np.concatenate([[0], mv[i].astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        n += int(((ends - starts) >= min_run).sum())
    return n / mv.shape[0], float((~mv).mean())


def _disp_vs_dt(pos, dt_frac, horizon, rng, n_anchor=60):
    """median ||Δobject|| and P(within radius) vs temporal offset.

    Offsets are given as fractions of the benchmark's OWN eval horizon, so cube (200 steps)
    and Metaworld (100 wrapper steps) are compared at matched effective lookaheads rather
    than at matched raw step counts."""
    N, T = pos.shape[:2]
    rows = []
    for flo, fhi in dt_frac:
        lo, hi = max(1, int(flo * horizon)), min(int(fhi * horizon), T - 1)
        if lo >= hi:
            rows.append((f"{flo:g}-{fhi:g} x H", float("nan"), np.array([np.nan])))
            continue
        d = []
        for i in rng.choice(N, size=min(N, 300), replace=False):
            t = rng.randint(0, T - hi, size=n_anchor)
            off = rng.randint(lo, hi, size=n_anchor)
            d.append(np.linalg.norm(pos[i, t + off] - pos[i, t], axis=-1))
        d = np.concatenate(d)
        rows.append((f"{flo:g}-{fhi:g} x H  ({lo}-{hi} steps)", float(np.median(d)), d))
    return rows


def _uniformity(xy, bounds, bins=(5, 6)):
    """entropy of the achieved-goal marginal / entropy of the uniform grid."""
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=bounds)
    p = h.ravel() / h.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()), float(np.log(len(p)))


def analyse_mw(npz, task, rng):
    z = np.load(npz)
    O, NO, S = z["observation"], z["next_observation"], z["success"]
    N, T = O.shape[:2]
    obj, goal = NO[..., MW_OBJ], NO[..., MW_GOAL]
    seg = np.concatenate(_segments_from_jumps(O[..., MW_GOAL]))
    bouts, rest = _bouts(obj, 5e-4)
    r = MW_SUCCESS_R[task]

    # per-segment: did the object ever get inside the success radius?
    chg = (np.abs(np.diff(O[..., MW_GOAL], axis=1)) > 1e-9).any(-1)
    hit, tot = 0, 0
    for i in range(N):
        b = np.concatenate([[0], np.flatnonzero(chg[i]) + 1, [T]])
        for a, c in zip(b[:-1], b[1:]):
            tot += 1
            hit += int(S[i, a:c].any())
    gxy = goal.reshape(-1, 3)[:, :2]
    ent, ent_u = _uniformity(gxy, [[gxy[:, 0].min(), gxy[:, 0].max()],
                                   [gxy[:, 1].min(), gxy[:, 1].max()]])
    return dict(
        name=f"MW {task}", n_traj=N, traj_len=T, transitions=N * T,
        eval_horizon=MW_EVAL_HORIZON, radius=r, obj=obj,
        seg_per_traj=len(seg) / N, seg_len=float(seg.mean()), seg_med=float(np.median(seg)),
        seg_frac_len1=float((seg == 1).mean()),
        bouts_per_traj=bouts, rest_frac=rest,
        dwell_success=float(S.mean()), seg_completion=hit / tot,
        goal_entropy_ratio=ent / ent_u,
        stale=float(((np.abs(O[..., MW_GOAL] - NO[..., MW_GOAL]) > 1e-9).any(-1)).mean()),
        success_rule_agreement=float(
            (S == (np.linalg.norm(obj - goal, axis=-1) < r)).mean()),
        goal_in_obs=True,
    )


# ManiSkill unified obs layouts (confirmed against the packed npz; see the spec's §8.1):
#   PushCube (35) = qpos9 | qvel9 | tcp_pose7 | goal_pos3 | obj_pose7
#   PickCube (42) = qpos9 | qvel9 | is_grasped1 | tcp_pose7 | goal_pos3 | obj_pose7
#                   | tcp_to_obj3 | obj_to_goal3
# `dist_dims` is the number of components the env's own success rule measures: PushCube scores
# planar XY distance to a goal region, PickCube full 3-D distance to a goal point. Every
# distance in this comparison is taken in the env's own metric, or the P(goal satisfied at dt)
# column would not be commensurable across benchmarks.
MS_LAYOUT = {
    "ms-push-cube": dict(goal=slice(25, 28), obj=slice(28, 31), dist_dims=2, radius=0.10),
    "ms-pick-cube": dict(goal=slice(26, 29), obj=slice(29, 32), dist_dims=3, radius=0.025),
}
MS_EVAL_HORIZON = 25      # env/maniskill_env.py TASKS


def analyse_ms(npz, task, rng):
    lay = MS_LAYOUT[task]
    z = np.load(npz)
    O, NO, S = z["observation"], z["next_observation"], z["success"]
    N, T = O.shape[:2]
    k = lay["dist_dims"]
    obj_full, goal_full = NO[..., lay["obj"]], NO[..., lay["goal"]]
    obj, goal = obj_full[..., :k], goal_full[..., :k]
    seg = np.concatenate(_segments_from_jumps(O[..., lay["goal"]]))
    bouts, rest = _bouts(obj, 1e-3)
    r = lay["radius"]

    chg = (np.abs(np.diff(O[..., lay["goal"]], axis=1)) > 1e-9).any(-1)
    hit, tot = 0, 0
    for i in range(N):
        b = np.concatenate([[0], np.flatnonzero(chg[i]) + 1, [T]])
        for a, c in zip(b[:-1], b[1:]):
            tot += 1
            hit += int(S[i, a:c].any())
    gxy = goal_full.reshape(-1, 3)[:, :2]
    ent, ent_u = _uniformity(gxy, [[gxy[:, 0].min(), gxy[:, 0].max()],
                                   [gxy[:, 1].min(), gxy[:, 1].max()]])
    # `success_rule_agreement` is a LOWER bound for ManiSkill: both tasks AND extra predicates
    # onto the distance test (PushCube requires the cube still on the table, PickCube requires
    # a static robot), which the distance alone cannot reproduce. A value near 1 means the
    # recorded flag and the goal dims agree; a low one means the goal channel is lying.
    placed = np.linalg.norm(obj - goal, axis=-1) < r
    return dict(
        name=f"MS {task}", n_traj=N, traj_len=T, transitions=N * T,
        eval_horizon=MS_EVAL_HORIZON, radius=r, obj=obj,
        seg_per_traj=len(seg) / N, seg_len=float(seg.mean()), seg_med=float(np.median(seg)),
        seg_frac_len1=float((seg == 1).mean()),
        bouts_per_traj=bouts, rest_frac=rest,
        dwell_success=float(S.mean()), seg_completion=hit / tot,
        goal_entropy_ratio=ent / ent_u,
        stale=float(((np.abs(O[..., lay["goal"]] - NO[..., lay["goal"]]) > 1e-9).any(-1)).mean()),
        success_rule_agreement=float((S <= placed).mean()),
        goal_in_obs=True,
    )


def analyse_cube(npz, rng):
    z = np.load(os.path.expanduser(npz))
    obs, term = z["observations"], z["terminals"]
    ti = np.flatnonzero(term)
    T = int(ti[0]) + 1
    N = len(ti)
    obs = obs[: N * T].reshape(N, T, -1)
    blk = obs[..., CUBE_BLOCK] / 10.0 + np.array([0.425, 0.0, 0.0])   # -> metres
    bouts, rest = _bouts(blk, 1e-3)
    # achieved goals = block pose while at rest
    mv = np.linalg.norm(np.diff(blk, axis=1), axis=-1) > 1e-3
    at_rest = blk[:, 1:][~mv]
    ent, ent_u = _uniformity(at_rest[:, :2], [[0.30, 0.55], [-0.30, 0.30]])
    return dict(
        name="OGBench cube-single-play", n_traj=N, traj_len=T, transitions=N * T,
        eval_horizon=CUBE_EVAL_HORIZON, radius=CUBE_SUCCESS_R, obj=blk,
        seg_per_traj=bouts, seg_len=T / max(bouts, 1), seg_med=float("nan"),
        seg_frac_len1=0.0,
        bouts_per_traj=bouts, rest_frac=rest,
        dwell_success=float("nan"), seg_completion=float("nan"),
        goal_entropy_ratio=ent / ent_u,
        stale=0.0, success_rule_agreement=float("nan"),
        goal_in_obs=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mw-root", default=None)
    ap.add_argument("--ms-root", default=None,
                    help="root of a ManiSkill collection; reads <root>/data/maniskill/<task>.npz")
    ap.add_argument("--cube", default="~/.ogbench/data/cube-single-play-v0.npz")
    ap.add_argument("--tasks", nargs="*", default=["mw-push", "mw-pick-place"])
    ap.add_argument("--ms-tasks", nargs="*", default=["ms-push-cube", "ms-pick-cube"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    rows = [analyse_cube(args.cube, rng)]
    if args.mw_root:
        for t in args.tasks:
            rows.append(analyse_mw(os.path.join(args.mw_root, "data", "metaworld", f"{t}.npz"),
                                   t, rng))
    if args.ms_root:
        for t in args.ms_tasks:
            p = os.path.join(args.ms_root, "data", "maniskill", f"{t}.npz")
            if os.path.exists(p):
                rows.append(analyse_ms(p, t, rng))
            else:
                print(f"[compare] skipping {t}: no {p}")

    def line(label, key, fmt="{:.3f}"):
        vals = []
        for r in rows:
            v = r[key] if not callable(key) else key(r)
            vals.append("--" if isinstance(v, float) and np.isnan(v) else
                        (fmt.format(v) if isinstance(v, float) else str(v)))
        print(f"  {label:<44}" + "".join(f"{v:>22}" for v in vals))

    print("\n" + "=" * 110)
    print("STRUCTURAL COMPARISON  (units of each benchmark's own eval horizon where marked *)")
    print("=" * 110)
    print(f"  {'':<44}" + "".join(f"{r['name']:>22}" for r in rows))
    print("\n-- scale --")
    line("trajectories", "n_traj")
    line("steps per trajectory", "traj_len")
    line("transitions", "transitions")
    line("eval horizon", "eval_horizon")
    line("* traj_len / eval_horizon", lambda r: r["traj_len"] / r["eval_horizon"], "{:.2f}")
    print("\n-- segment / relocation structure --")
    line("relocations per trajectory", "seg_per_traj", "{:.2f}")
    line("segment length (steps)", "seg_len", "{:.1f}")
    line("* segment length / eval horizon", lambda r: r["seg_len"] / r["eval_horizon"], "{:.3f}")
    line("object-at-rest fraction", "rest_frac")
    line("degenerate length-1 segments", "seg_frac_len1")
    line("segments reaching the target", "seg_completion")
    line("dwell (frac steps success firing)", "dwell_success")
    print("\n-- goal marginal --")
    line("achieved-goal entropy / uniform", "goal_entropy_ratio")
    print("\n-- data integrity --")
    line("goal-channel stale transitions", "stale")
    line("success rule reproduces flag", "success_rule_agreement")
    line("GOAL PRESENT IN OBSERVATION", "goal_in_obs", "{}")

    print("\n-- temporal geometry: P(||object(t+dt) - object(t)|| < success radius) --")
    dts = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    print(f"  {'dt window (H = that benchmark eval horizon)':<44}"
          + "".join(f"{r['name']:>22}" for r in rows))
    per = [_disp_vs_dt(r["obj"], dts, r["eval_horizon"], rng) for r in rows]
    for k in range(len(per[0])):
        lbl = per[0][k][0].split("  (")[0]
        cells = []
        for r, p in zip(rows, per):
            _, med, d = p[k]
            cells.append("--" if np.isnan(med) else
                         f"{med:.3f} m P={float((d < r['radius']).mean()):.3f}")
        print(f"  {lbl:<44}" + "".join(f"{c:>22}" for c in cells))

    if args.json_out:
        dump = [{k: v for k, v in r.items() if k != "obj"} for r in rows]
        with open(args.json_out, "w") as f:
            json.dump(dump, f, indent=2, default=float)
        print(f"\n[compare] wrote {args.json_out}")


if __name__ == "__main__":
    main()
