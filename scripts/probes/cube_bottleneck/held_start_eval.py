"""Test 1 -- held-start evaluation (spec section 2).

Remove the control bottleneck at test time by starting the episode with the cube already
secured, and ask whether plain FB recovers toward Density-FB.

Three start conditions:
  H0  standard `env.reset(options=task_id)`. This IS the table protocol and must reproduce
      the run's own logged eval (gate G1).
  H1  in-distribution held start: a RECORDED held state from the play dataset the arm
      trained on, restored via qpos/qvel (see restore.py), policy takes over toward the
      task's own goal.
  H2  the old S2 clamp, ported verbatim from [torch] evals/phase_probe.py, at reduced
      episode count. Continuity with the recorded PSM/RLDP/GCIQL S2 rows only; the spec
      marks it explicitly non-deciding.

Every arm, seed, task and start walks the SAME bank rows in the SAME order with the SAME
rollout seeds, so H1-vs-H0 and FB-vs-Density-FB are paired comparisons.

Output: one row per EPISODE to `held_start.csv`. Aggregation and the pre-registered
readings live in readings.py -- this file only produces evidence.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.phases import (
    PhaseThresholds, apply_scenario, bucket_row, classify_phases, episode_goal,
    episode_table_z, step_signals, transport_mask,
)

STARTS = ["H0", "H1", "H2"]
RETENTION_WINDOW = 50      # spec 2.4: hold_retention over the first 50 steps
EPS_NEAR = 0.05            # transport_buckets default: "near goal"


# ── G2 filtering of the shared bank ─────────────────────────────────────────────────────

def validate_bank(env, bank: Dict[str, np.ndarray], task_id: int = 1,
                  thr: Optional[PhaseThresholds] = None) -> Dict[str, Any]:
    """Apply gate G2 to every bank row ONCE, before any arm sees the bank (spec 2.2).

    Runs at a single task_id: the restore writes the whole simulator state, so the only
    thing task_id changes is the goal mocap, which no G2 quantity reads.
    """
    keep, rows = [], []
    for i in range(len(bank["qpos"])):
        R.restore(env, task_id, bank["qpos"][i], bank["qvel"][i])
        g = R.g2_check(env, bank["qpos"][i], thr=thr)
        rows.append({**g.as_row(), "bank_row": int(bank["row"][i])})
        if g.passes:
            keep.append(i)
    keep = np.asarray(keep, int)
    return dict(keep=keep, per_row=rows, n=len(bank["qpos"]), n_kept=int(len(keep)),
                drop_frac=float(1.0 - len(keep) / max(1, len(bank["qpos"]))))


# ── rollout ─────────────────────────────────────────────────────────────────────────────

def _signals_now(env) -> Dict[str, Any]:
    return step_signals(env.unwrapped.compute_ob_info())


def _episode_record(ep, thr, task_id, start, consumed) -> Dict[str, Any]:
    """Phase metrics + retention diagnostics for one finished episode."""
    import pandas as pd

    ph = classify_phases(ep, thr)
    tm = transport_mask(ep, thr)
    ep["transport_mask"] = tm

    win = tm[:RETENTION_WINDOW]
    hold_retention = float(win.mean()) if len(win) else float("nan")
    ended = ~tm & np.concatenate([[False], tm[:-1]]) if len(tm) else np.zeros(0, bool)
    first_drop = int(np.flatnonzero(ended)[0]) if ended.any() else -1

    rec = dict(task=task_id, start=start, steps_consumed_by_restore=consumed,
               success=int(ph["success"]), reached=int(ph["reached"]),
               secured=int(ph["secured"]), furthest_phase=ph["furthest_phase"],
               fail_phase=ph["fail_phase"], length=ph["length"],
               final_cube_lift=ph["final_cube_lift"],
               final_cube_goal_dist=ph["final_cube_goal_dist"],
               final_grip=ph["final_grip"], max_cube_lift=ph["max_cube_lift"],
               min_eff_cube_dist=ph["min_eff_cube_dist"],
               hold_retention_50=hold_retention,
               held_frac=float(tm.mean()) if len(tm) else float("nan"),
               time_to_first_drop=first_drop, drop_censored=int(first_drop < 0))
    rec["bucket"] = (bucket_row(pd.Series(rec), thr.delta_lift, EPS_NEAR)
                     if not ph["success"] else "success")
    rec["maintain_fail"] = int((not ph["success"]) and rec["final_cube_lift"] < thr.delta_lift)
    rec["transport_fail"] = int((not ph["success"]) and rec["final_cube_lift"] >= thr.delta_lift)
    return rec


def rollout_cell(actor_fn, envs, task_id: int, start: str, *, bank, keep, n_episodes: int,
                 thr: PhaseThresholds, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All `n_episodes` of one (task, start) cell, stepped in LOCKSTEP.

    Why batched: an arm that fails runs every episode to the step limit, so a per-episode
    loop pays one actor call per step per episode. Stepping the cell together pays one
    batched call per step for the whole cell -- the same trick Pass A of the measure probe
    uses, and the reason plain FB's cells are tractable at all (measured: a 96-row batch
    costs about what a single row costs).

    Bank rows are walked in a fixed order shared by every arm, seed and task, so H1 is
    paired across arms row by row.
    """
    assert config.get("measure_qpos_slices") is None, (
        "batched rollout is state-only; a pixel arm needs the per-episode goal-channel dict")

    obs, goals, eff, cube, grip, succ, alive, consumed = [], [], [], [], [], [], [], 0
    ep_goal, table_z = [], []
    for m in range(n_episodes):
        env = envs[m]
        ob, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        if start == "H1":
            idx = int(keep[m % len(keep)])
            R.set_state(env, bank["qpos"][idx], bank["qvel"][idx])
            R.settle(env)                      # 2 env steps; see restore.N_SETTLE
            consumed = R.N_SETTLE
            ob = R.observation(env)
        elif start == "H2":
            apply_scenario(env, "S2")          # vendored verbatim: raw ctrl=255, 5 mj_steps
            ob = R.observation(env)
        obs.append(np.asarray(ob, np.float32))
        goals.append(np.asarray(info.get("goal"), np.float32))
        ep_goal.append(episode_goal(env)); table_z.append(episode_table_z(env))
        sig = step_signals(env.unwrapped.compute_ob_info())
        eff.append([sig["eff"]]); cube.append([sig["cube"]]); grip.append([sig["grip"]])
        succ.append(False); alive.append(True)

    obs = np.stack(obs); goal_b = np.stack(goals); alive = np.asarray(alive)
    while alive.any():
        acts = np.clip(np.asarray(actor_fn(observations=obs, goals=goal_b, temperature=0)),
                       -1, 1)
        for m in range(n_episodes):
            if not alive[m]:
                continue
            ob, _r, term, trunc, info = envs[m].step(acts[m].astype(np.float32))
            obs[m] = np.asarray(ob, np.float32)
            s = step_signals(info)
            eff[m].append(s["eff"]); cube[m].append(s["cube"]); grip[m].append(s["grip"])
            succ[m] = succ[m] or bool(info.get("success", False))
            if term or trunc:
                alive[m] = False

    out = []
    for m in range(n_episodes):
        ep = dict(eff=np.asarray(eff[m]), cube=np.asarray(cube[m]), grip=np.asarray(grip[m]),
                  goal=ep_goal[m], table_z=table_z[m], success=succ[m])
        rec = _episode_record(ep, thr, task_id, start, consumed)
        rec["bank_row"] = (int(bank["row"][int(keep[m % len(keep)])]) if start == "H1" else -1)
        rec["episode"] = m
        out.append(rec)
    return out


# ── driver ──────────────────────────────────────────────────────────────────────────────

def run_arm(run_dir: Path, bank, keep, *, starts: List[str], n_episodes: int,
            n_episodes_h2: int, epoch: int, thr: PhaseThresholds) -> List[Dict[str, Any]]:
    import jax
    from impls.utils.evaluation import supply_rng

    arm = A.load_arm(run_dir, epoch)
    R.ensure_cube_env(arm.env)
    tasks = A.eval_tasks(arm)
    # One rng stream per arm, seeded identically for every arm -> paired rollouts.
    actor_fn = supply_rng(arm.agent.sample_actions, rng=jax.random.PRNGKey(0))
    envs = [R.make_env(arm.env_name, seed=i) for i in range(max(n_episodes, n_episodes_h2))]
    print(f"  [{arm.label}] env pool: {len(envs)}", flush=True)

    out = []
    for task in tasks:
        for start in starts:
            n = n_episodes_h2 if start == "H2" else n_episodes
            recs = rollout_cell(actor_fn, envs, task, start, bank=bank, keep=keep,
                                n_episodes=n, thr=thr, config=arm.config)
            for r in recs:
                r.update(env=arm.env_name, critic=arm.critic, seed=arm.seed, epoch=epoch,
                         run_dir=str(run_dir))
            out += recs
            print(f"  [{arm.label}] task {task} {start}: "
                  f"success {np.mean([r['success'] for r in recs]):.3f} (n={len(recs)})",
                  flush=True)
    return out


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=["fb_ref", "density_fb_state"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--starts", nargs="+", default=STARTS, choices=STARTS)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--episodes-h2", type=int, default=10)
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--exp-root", default="exp")
    ap.add_argument("--out-root", default="analysis/cube_bottleneck")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.env
    bank = dict(np.load(out_dir / "held_bank.npz"))

    # G2 once, on a throwaway env, before any arm is built.
    env = R.make_env(args.env, seed=0)
    R.ensure_cube_env(env)
    thr = PhaseThresholds()
    g2 = validate_bank(env, bank, thr=thr)
    print(f"[G2] {args.env}: kept {g2['n_kept']}/{g2['n']} "
          f"(dropped {g2['drop_frac']:.1%}; spec ceiling 25%)", flush=True)
    (out_dir / "g2_held_bank.json").write_text(json.dumps(
        dict(n=g2["n"], n_kept=g2["n_kept"], drop_frac=g2["drop_frac"],
             n_settle=R.N_SETTLE, drift_bound=R.G2_DRIFT_M, contact_min=R.G2_CONTACT_MIN,
             per_row=g2["per_row"]), indent=2))
    if g2["drop_frac"] > 0.25:
        print("[G2] FLOOR CLAUSE: >25% of the bank dropped; Test 1 is not read.", flush=True)
    if g2["n_kept"] == 0 and "H1" in args.starts:
        raise SystemExit("[G2] no bank row survived; H1 cannot run. Inspect g2_held_bank.json.")
    if hasattr(env, "close"):
        env.close()

    runs = A.discover(args.exp_root, args.env, args.critics, args.seeds, args.epoch)
    labels = [f"{r['critic']}_s{r['seed']}" for r in runs]
    print(f"[test1] {len(runs)} runs: {labels}", flush=True)

    # ONE CSV PER ARM, so arms can be run as separate processes in parallel without
    # racing on a shared file. readings.py globs them back together.
    part_dir = out_dir / "held_start"
    part_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for r in runs:
        print(f"[test1] {r['run_dir']}", flush=True)
        rows = run_arm(r["run_dir"], bank, g2["keep"], starts=args.starts,
                       n_episodes=args.episodes, n_episodes_h2=args.episodes_h2,
                       epoch=args.epoch, thr=thr)
        out = part_dir / f"{r['critic']}_seed{r['seed']}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        total += len(rows)
        print(f"[test1] wrote {len(rows)} rows -> {out}", flush=True)
    print(f"[test1] done: {total} episode rows across {len(runs)} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
