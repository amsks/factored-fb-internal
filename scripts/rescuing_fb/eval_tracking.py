#!/usr/bin/env python
"""scripts/rescuing_fb/eval_tracking.py -- spec 26 R0b(ii): the held-out demo
TRACKING eval OGBench lacks (PAPER/Multistep-FB/specs/26-bfm-roadmap.md sec.2).

Protocol (every piece routed through existing audited code -- the repo's hard
rule; the only new logic is the pure, unit-tested waypoint machinery imported
from bfm_act.py):

  SPLIT      hold out --num-demos (default 50) play trajectories by EPISODE
             index, seeded (`held_out_episode_indices`, pure + unit-tested):
             a sorted no-replacement draw of episode indices under
             default_rng(--split-seed). These episodes were never used as
             demos before (no prior harness in this repo conditions on raw
             dataset episodes); the split is recorded in the output JSON.
  STARTS     teleport, exactly as the G4 probe does
             (eval_measure_g4_spearman.py "MC ground truth" block /
             torchfb_full_measure.py part B): env.reset(), then
             data.qpos[:] = qpos[row0]; data.qvel[:] = qvel[row0];
             mujoco.mj_forward. The t=0 observation is the dataset's own
             obs[row0] (the audited convention -- that IS what the recorded
             qpos/qvel produced). Dataset = `env_geometry.resolve_dataset`'s
             npz (the jax-side ogbench dataset has no qpos/qvel).
  POLICY     sliding-waypoint conditioning: `bfm_act.WaypointTracker`
             (delta default 25, monotone nearest-progress -- the demo-prompt
             branch of the BFM interface, same class, not a copy), actor =
             the checkpoint's own `sample_actions(obs, goals=waypoint_obs,
             temperature=0.0)` batched across demos exactly like
             `batched_goal_rollout_density` batches episodes.
  WINDOW     play episodes are 1001 rows but the singletask env's TimeLimit
             is 200, so each demo is its episode's FIRST --track-len
             (default 200) transitions. CONTRACT DEVIATION, stated: full-
             episode tracking would require stepping past the env's
             TimeLimit; the 200-step window matches the audited G4 horizon.

  METRICS (per demo, then mean over demos; `tracking_metrics` is pure +
  unit-tested; membership = the hit-spec's own conjunctive `contains`, the
  SAME test every region metric in the repo uses):
    path_error   mean time-aligned object-path error: mean_t
                 || coords(rollout_t) - coords(demo_t) ||_2 (raw hit-spec
                 coords, meters on cube), rollout step t aligned to demo row
                 t+1 (both are "the state after t+1 actions from row 0").
    hit_rate     fraction of demo waypoints (rows delta, 2*delta, ..., T-1;
                 `bfm_act.waypoint_schedule`) reached within eps: any rollout
                 state inside spec.contains(., demo_coords[w], mult=--eps-mult).
    completion   final rollout state within eps of the demo's final row.

  CONTROLS (the gate needs BOTH):
    BC-only ckpt      run this script on the spec-19 BCOnly checkpoint --
                      same tracking arm, actor floor.
    shuffled arm      --arms includes 'shuffled': same checkpoint, same
                      demos, same schedule, but the actor is conditioned on
                      demo row perm[k] instead of row k
                      (`bfm_act.conditioning_permutation`, seeded,
                      non-identity, per-demo). Progress and ALL metrics still
                      run against the TRUE demo -- the control isolates
                      whether tracking is real or drift.

GATE R0b (spec 26 sec.2): density beats BOTH controls on waypoint hit-rate.

Run (CPU-only jax, GPUs carry training jobs; always inside tmux):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_tracking.py \\
    --run-dir /home/adityamohan/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0 \\
    --checkpoint-step 500000 --num-demos 50 --arms tracking,shuffled \\
    --out results/spec26_r0b/track_d18_cs_dens_s0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

# The single waypoint implementation (BFM demo branch) + the single geometry
# definition. Imported, never copied.
from bfm_act import WaypointTracker, conditioning_permutation, waypoint_schedule  # noqa: E402
from env_geometry import episode_bounds, geometry_for, resolve_dataset  # noqa: E402
from eval_zeroshot_rewards import eval_tasks_for  # noqa: E402

ARMS = ("tracking", "shuffled")


# ======================= pure logic (jax-free, unit-tested) ========================

def held_out_episode_indices(n_episodes: int, n_holdout: int, seed: int) -> np.ndarray:
    """Episode-level held-out split: a sorted, no-replacement, seeded draw of
    episode indices. Deterministic in (n_episodes, n_holdout, seed)."""
    assert 0 < n_holdout <= n_episodes, \
        f"cannot hold out {n_holdout} of {n_episodes} episodes"
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_episodes, n_holdout, replace=False))


def tracking_metrics(roll_coords: np.ndarray, demo_coords: np.ndarray, spec,
                     delta: int, eps_mult: float = 1.0) -> dict:
    """Per-demo metrics; pure numpy. roll_coords: [L, d_c] hit-spec coords of
    the rollout states AFTER each action (step t <-> demo row t+1);
    demo_coords: [T, d_c] with T = L + 1 (row 0 = the shared start state).
    Membership = spec.contains (conjunctive per-block L2, the repo's single
    region test). Waypoints/metrics ALWAYS use the true demo -- the shuffled
    control never enters here (its permutation touches conditioning only)."""
    roll_coords = np.asarray(roll_coords, np.float64)
    demo_coords = np.asarray(demo_coords, np.float64)
    L, T = len(roll_coords), len(demo_coords)
    assert T == L + 1, f"expected demo rows = rollout steps + 1, got T={T} L={L}"
    errs = np.linalg.norm(roll_coords - demo_coords[1:], axis=1)       # [L]
    wps = waypoint_schedule(T, delta)
    hits = np.array([bool(spec.contains(roll_coords, demo_coords[w],
                                        mult=eps_mult).any()) for w in wps])
    completion = bool(spec.contains(roll_coords[-1:], demo_coords[T - 1],
                                    mult=eps_mult)[0])
    return {"path_error": float(errs.mean()),
            "path_error_final": float(errs[-1]),
            "n_waypoints": int(len(wps)),
            "waypoints": wps.tolist(),
            "waypoint_hits": hits.astype(int).tolist(),
            "hit_rate": float(hits.mean()),
            "completion": completion}


def summarize_arm(per_demo: list) -> dict:
    """Mean/std over demos for the three headline metrics."""
    out = {}
    for k in ("path_error", "hit_rate"):
        v = np.array([d[k] for d in per_demo], np.float64)
        out[k] = {"mean": float(v.mean()), "std": float(v.std())}
    comp = np.array([float(d["completion"]) for d in per_demo])
    out["completion"] = {"mean": float(comp.mean()), "std": float(comp.std())}
    out["n_demos"] = len(per_demo)
    return out


# ======================= the rollout (jax-dependent) ===============================

def run_arm(arm: str, agent, envs, spec, demos: list, qpos_all, qvel_all,
            track_len: int, delta: int, eps_mult: float, seed: int) -> dict:
    """One batched rollout of all demos under `arm`. demos: list of dicts with
    'episode', 'row0', 'obs' [T, obs_dim]. Returns per-demo metrics + summary."""
    import jax
    import jax.numpy as jnp
    import mujoco

    n = len(demos)
    assert len(envs) == n
    trackers = []
    for i, d in enumerate(demos):
        perm = None
        if arm == "shuffled":
            perm = conditioning_permutation(len(d["obs"]),
                                            seed * 1_000_003 + i)
        trackers.append(WaypointTracker(d["obs"], spec, delta=delta, perm=perm))

    obs_dim = int(envs[0].observation_space.shape[0])
    obs = np.zeros((n, obs_dim), np.float32)
    for i, (e, d) in enumerate(zip(envs, demos)):
        e.reset(seed=i)
        u = e.unwrapped
        u.data.qpos[:] = qpos_all[d["row0"]]
        u.data.qvel[:] = qvel_all[d["row0"]]
        mujoco.mj_forward(u.model, u.data)
        obs[i] = d["obs"][0]        # the dataset's own obs at this (qpos,qvel)

    d_c = spec.dim
    roll_coords = np.zeros((n, track_len, d_c))
    progress_log = np.zeros((n, track_len), np.int64)
    active = np.ones(n, bool)
    t0 = time.time()
    for t in range(track_len):
        goals = np.zeros((n, obs_dim), np.float32)
        for i in range(n):
            g, info = trackers[i].goal(obs[i])
            goals[i] = g
            progress_log[i, t] = info["progress"]
        key = jax.random.PRNGKey(seed * 1_000_003 + t)
        acts = np.clip(np.asarray(agent.sample_actions(
            observations=jnp.asarray(obs), goals=jnp.asarray(goals),
            seed=key, temperature=0.0)), -1, 1)
        for i, e in enumerate(envs):
            if not active[i]:
                roll_coords[i, t] = roll_coords[i, t - 1]
                continue
            o, _r, term, trunc, _inf = e.step(acts[i])
            obs[i] = o
            roll_coords[i, t] = spec.coords(o[None])[0]
            if term or (trunc and t < track_len - 1):
                active[i] = False       # freeze; TimeLimit shorter than window
        if (t + 1) % 50 == 0:
            print(f"[tracking][{arm}] step {t+1}/{track_len} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    per_demo = []
    for i, d in enumerate(demos):
        m = tracking_metrics(roll_coords[i], spec.coords(d["obs"]), spec,
                             delta, eps_mult=eps_mult)
        m.update({"episode": int(d["episode"]), "row0": int(d["row0"]),
                  "final_progress": int(progress_log[i, -1]),
                  "froze_early": bool(not active[i])})
        per_demo.append(m)
    summary = summarize_arm(per_demo)
    summary["frozen_early"] = int(np.sum([m["froze_early"] for m in per_demo]))
    print(f"[tracking][{arm}] hit_rate={summary['hit_rate']['mean']:.3f} "
          f"path_error={summary['path_error']['mean']:.3f} "
          f"completion={summary['completion']['mean']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"per_demo": per_demo, "summary": summary}


# ======================= CLI / main ================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--num-demos", type=int, default=50,
                    help="held-out play episodes to track (contract N=50)")
    ap.add_argument("--track-len", type=int, default=200,
                    help="tracked window per demo (env TimeLimit bound)")
    ap.add_argument("--delta", type=int, default=25,
                    help="waypoint lookahead (contract default)")
    ap.add_argument("--eps-mult", type=float, default=1.0,
                    help="hit/completion tolerance as a multiple of the "
                         "hit-spec's own per-block eps")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="episode-level held-out split seed (document with "
                         "the gate; same split across every checkpoint)")
    ap.add_argument("--seed", type=int, default=0,
                    help="rollout/action + shuffled-permutation seed")
    ap.add_argument("--arms", default="tracking,shuffled",
                    help=f"comma-sep subset of {ARMS}")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec26_r0b/"
                         "track_<run>.json)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    assert not unknown, f"unknown arms {unknown}; choose from {ARMS}"

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    env_name = flags["env_name"]
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec

    # dataset WITH qpos/qvel (teleport protocol needs exact states -- G4/torch
    # precedent; the jax-side ogbench dataset carries no qpos/qvel).
    npz_path = resolve_dataset(str(cfg.get("data_path") or ""), geo.dataset_name)
    npz = np.load(npz_path)
    obs_all = np.asarray(npz["observations"], np.float32)
    qpos_all, qvel_all = np.asarray(npz["qpos"]), np.asarray(npz["qvel"])
    bounds = episode_bounds(np.asarray(npz["terminals"]))

    T = args.track_len + 1
    held = held_out_episode_indices(len(bounds), args.num_demos, args.split_seed)
    demos = []
    for ep in held:
        lo, hi = int(bounds[ep, 0]), int(bounds[ep, 1])
        assert hi - lo >= T, \
            f"episode {ep} has {hi-lo} rows < window {T} -- shrink --track-len"
        demos.append({"episode": int(ep), "row0": lo, "obs": obs_all[lo:lo + T]})

    env_id = eval_tasks_for(env_name)[1]
    envs = [ogbench.make_env_and_datasets(env_id, env_only=True)
            for _ in range(len(demos))]
    max_steps = int(envs[0].spec.max_episode_steps)
    assert args.track_len <= max_steps, \
        f"--track-len {args.track_len} > env TimeLimit {max_steps}"
    obs_dim = int(envs[0].observation_space.shape[0])
    act_dim = int(envs[0].action_space.shape[0])

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    print(f"[tracking] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} hit_spec={cfg.get('measure_hit_spec')!r} "
          f"demos={len(demos)} track_len={args.track_len} delta={args.delta} "
          f"eps_mult={args.eps_mult} split_seed={args.split_seed} "
          f"seed={args.seed} arms={arms}", flush=True)

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "dataset": str(npz_path),
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in spec.blocks),
        "num_demos": len(demos), "track_len": args.track_len, "delta": args.delta,
        "eps_mult": args.eps_mult, "split_seed": args.split_seed,
        "seed": args.seed, "arms": arms,
        "held_out_episodes": held.tolist(),
        "actor_bc_only": bool(cfg.get("actor_bc_only", False)),
        "measure_density": bool(cfg.get("measure_density", False)),
        "measure_density_hard": bool(cfg.get("measure_density_hard", False)),
        "protocol": {
            "spec": "PAPER/Multistep-FB/specs/26-bfm-roadmap.md sec.2 (R0b ii)",
            "split": "episode-level, sorted seeded no-replacement draw "
                     "(held_out_episode_indices)",
            "starts": "teleport to the demo's row-0 (qpos,qvel), mj_forward; "
                      "t=0 obs = dataset obs[row0] (G4 / torchfb part B "
                      "convention)",
            "policy": "bfm_act.WaypointTracker sliding waypoint (monotone "
                      "nearest-progress + delta lookahead), actor = "
                      "sample_actions(obs, goals=waypoint_obs, temperature=0), "
                      "batched over demos",
            "window": "CONTRACT DEVIATION: demos are each held-out episode's "
                      "FIRST track_len transitions (play episodes are 1001 "
                      "rows; the singletask env TimeLimit is 200)",
            "metrics": "tracking_metrics: time-aligned L2 path error on raw "
                       "hit-spec coords; waypoint hit-rate + completion via "
                       "spec.contains at eps_mult",
            "shuffled_control": "conditioning row k -> perm[k] "
                                "(seeded non-identity permutation per demo); "
                                "progress + metrics vs the TRUE demo",
        },
    }, "arms": {}}

    for arm in arms:
        results["arms"][arm] = run_arm(
            arm, agent, envs, spec, demos, qpos_all, qvel_all,
            args.track_len, args.delta, args.eps_mult, args.seed)

    for e in envs:
        if hasattr(e, "close"):
            e.close()

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec26_r0b" / f"track_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[tracking] DONE -> {outp}", flush=True)
    for arm in arms:
        s = results["arms"][arm]["summary"]
        print(f"[tracking]   {arm}: hit_rate={s['hit_rate']['mean']:.3f} "
              f"path_error={s['path_error']['mean']:.3f} "
              f"completion={s['completion']['mean']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
