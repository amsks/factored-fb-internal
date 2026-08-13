#!/usr/bin/env python
"""scripts/rescuing_fb/eval_fb_zeroshot_rewards.py -- the PLAIN-FB column of
`tab:zeroshot` in PAPER/Multistep-FB/density-fb.tex.

The other two columns of that table come from `eval_density_zeroshot.py`, which
reads a reward out of the normalized successor DENSITY (a pooled softmax over
query coordinates) and then deploys a decoded goal proxy. Plain FB cannot use
that readout -- it has its OWN reward interface, and this script uses it:

    z_r = project( sum_i r(x_i) B(s'_i) )        # FBModel.reward_inference
    Q_r(s,a) = F(s,a,z_r)^T z_r                  # implicit: the trained actor
                                                 #   pi(s, z_r) already maximizes it

so the deployed policy is `FBAgent.act(obs, z_r)` -- no goal decode, no
retrieval, because FB's task index IS the reward moment. That is the one and
only intended difference from the density column.

EVERYTHING ELSE IS THE DENSITY PROTOCOL, REUSED NOT REIMPLEMENTED:
  * the three reward families come from the SAME functions the density harness
    calls -- `reward_conj_dirac` / `reward_conj_dense` / `region_success_from_obs`
    / `eval_tasks_for` / `print_table` (eval_zeroshot_rewards.py) and
    `draw_region_centers_with_obs` (eval_partition_zeroshot.py). Rebuilding the
    families here would silently break comparability, so nothing is rebuilt.
  * the geometry (hit spec = region rule) comes from `env_geometry.geometry_for`,
    the single definition both stacks share. The density checkpoints' recorded
    `measure_hit_spec` equals this registry entry on all three envs (verified
    against the archived JSONs -- see --ref-json below).
  * the pool is the SAME draw: one `default_rng(seed)` consumed in the SAME
    ORDER as `eval_density_zeroshot.main()` (relabel subsample first, then the
    P-row pool), and the region centers from `default_rng(seed+2)`. `--ref-json`
    re-reads an archived density result and asserts the region centers, hit
    spec, horizon, episode count and pool size match EXACTLY; a mismatch means
    the column is not comparable and the run refuses to continue.
  * rollout protocol: 25 independent envs of the task's own singletask id,
    `reset(seed=i)`, the env's own `max_episode_steps`, success = the env's own
    `info['success']` at ANY step (dirac/dense) or conjunctive region
    membership at the FINAL recorded step (region) -- the same reduction
    `batched_goal_rollout_density` uses.

CONTROLS (both are run by default; a table without them is not quotable):
  * `native`: FB's dirac cells are the five native OGBench tasks with indicator
    rewards at the native radius (obs cube xyz = (metres - CEN)*10, so the
    hit-spec eps 0.4 IS the 0.04 m OGBench threshold), so zero-shot dirac must
    land near the checkpoint's OWN native goal-conditioned success (wandb
    `success/overall`). `--native-success` records that number in the output for
    the comparison; a wild disagreement means the harness is wrong, not FB.
  * `randz` floor: the SAME rollout with z drawn from the model's own prior
    (`model.sample_z`), which is what "no task information" scores under this
    exact success criterion. Without it a low cell cannot be told from a floor.

Run (CPU only; the GPUs carry training jobs):
  taskset -c 24-27 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MUJOCO_GL=egl \\
    .venv/bin/python scripts/rescuing_fb/eval_fb_zeroshot_rewards.py \\
      --run /home/adityamohan/exp_missbase/fbref_cdbl_s42 \\
      --checkpoint-step 500000 --pool_size 16384 --episodes 25 --seed 0 \\
      --ref-json results/xenv_eval/dzs_cd_dens_s0_RET_k1.json \\
      --out results/fbzs_cdbl_s42.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# --- helper imports FIRST (they prepend third_party/ogbench/impls to sys.path,
#     which shadows the repo's own top-level `agents` package), then that path
#     is dropped again before any torch-side repo import happens.
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))
from eval_zeroshot_rewards import (  # noqa: E402
    eval_tasks_for,
    print_table,
    region_success_from_obs,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for  # noqa: E402
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402

# The helper modules put `third_party/ogbench/impls` (its own `agents` package)
# and `scripts` (its own `train` package) ahead of the repo root; both shadow
# the torch-side modules this script needs, so the repo root is forced back to
# the front and the impls path is dropped.
sys.path[:] = [p for p in sys.path if "third_party/ogbench/impls" not in p
               and p != str(REPO)]
sys.path.insert(0, str(REPO))

FAMILIES = ("dirac", "dense", "region")


class AgentMismatchError(RuntimeError):
    """The checkpoint is not the plain flow-actor FB this column needs."""


def check_agent_consistency(cfg_agent: str, cfg_obs_type: str, force: bool = False,
                            run_dir: str = "") -> list[str]:
    """Refuse, loudly, anything that is not `fb_flowbc` + state observations.

    The zero-shot table's density columns are flow-actor state cells, so the
    plain-FB column must be the flow actor on states too. `agent: fb` is the
    Gaussian+BC actor (a different row), `fb_hitdirac_flowbc` / `fb_measure_*`
    are different methods, and `obs_type: pixels` is a different observation
    space. Directory names are not evidence -- only the recorded config is.
    """
    warns: list[str] = []
    ok_agent = str(cfg_agent) == "fb_flowbc"
    ok_obs = str(cfg_obs_type) == "state"
    if ok_agent and ok_obs:
        return warns
    msg = (f"{run_dir or 'checkpoint'} records agent={cfg_agent!r} "
           f"obs_type={cfg_obs_type!r}; this column needs agent='fb_flowbc' and "
           f"obs_type='state'")
    if not force:
        raise AgentMismatchError(msg)
    warns.append(f"WARNING --force: {msg} -- numbers are NOT the plain-FB column")
    return warns


# ======================= comparability check ======================================

def check_reference(ref_path: str, meta: dict, centers: np.ndarray) -> dict:
    """Assert this run's protocol matches an archived density result exactly.

    Compares the fields that define the reward families and the rollout, not
    the readout: hit spec, pool size, episodes, horizon, task ids and -- the
    strongest check -- the drawn region centers, which depend on the dataset,
    the rng seed AND the order the rng was consumed in.
    """
    ref = json.loads(Path(ref_path).read_text())
    rm = ref["meta"]
    out: dict = {"ref": str(ref_path), "checks": {}, "ok": True}
    for key in ("env", "hit_spec", "pool_size", "episodes", "horizon",
                "checkpoint_step", "seed", "relabel_size"):
        same = rm.get(key) == meta.get(key)
        out["checks"][key] = {"ref": rm.get(key), "ours": meta.get(key), "match": bool(same)}
        out["ok"] &= bool(same)
    ref_centers = np.asarray(
        [ref["families"]["region"][k]["center_coord"]
         for k in sorted(ref["families"].get("region", {}))
         if k.startswith("c") and isinstance(ref["families"]["region"][k], dict)],
        np.float64)
    ours = np.asarray(centers, np.float64)
    same_shape = ref_centers.shape == ours.shape
    maxdiff = float(np.abs(ref_centers - ours).max()) if same_shape and ours.size else float("inf")
    out["checks"]["region_centers"] = {"shape_ref": list(ref_centers.shape),
                                       "shape_ours": list(ours.shape),
                                       "max_abs_diff": maxdiff,
                                       "match": bool(same_shape and maxdiff == 0.0)}
    out["ok"] = bool(out["ok"] and same_shape and maxdiff == 0.0)
    return out


# ======================= FB readout ===============================================

def fb_infer_z(agent, pool_obs: np.ndarray, r_pool: np.ndarray):
    """FB's own reward interface: z_r = project(sum_i r_i B(x_i)).

    `pool_obs` are raw dataset observations (the backward map normalizes them
    itself, exactly as in training and in the native evaluator's `_infer_z`).
    Returns (z [z_dim] torch tensor, diag dict). A pool with no reward mass
    would make z_r the projection of the zero vector (nan under norm_z), so
    that case is caught and reported instead of silently producing garbage.
    """
    import torch

    r = np.asarray(r_pool, np.float64).reshape(-1, 1)
    diag = {"pool_r_sum": float(r.sum()),
            "pool_r_nonzero": int(np.count_nonzero(r)),
            "pool_size": int(len(r))}
    if not np.isfinite(r).all() or float(np.abs(r).sum()) == 0.0:
        diag["degenerate"] = True
        return None, diag
    obs_t = torch.as_tensor(np.asarray(pool_obs, np.float32))
    r_t = torch.as_tensor(r, dtype=torch.float32)
    z = agent.infer_z(obs_t, r_t).detach().float().reshape(-1)
    diag["degenerate"] = bool(not bool(torch.isfinite(z).all()))
    diag["z_norm"] = float(torch.linalg.norm(z))
    return z, diag


def batched_z_rollout(agent, envs, z, H: int):
    """Fixed-z rollout, one z for the whole episode -- FB's deployment step.

    Mirrors `eval_density_zeroshot.batched_goal_rollout_density` exactly in
    everything the success numbers depend on: per-env `reset(seed=i)`, the
    env's own horizon, per-step action clipping, the env's own `info['success']`
    latched over the episode, and the recorded observation trajectory used by
    the region criterion.
    """
    import torch

    n = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    agent.reset()
    z_b = torch.as_tensor(np.asarray(z, np.float32)).reshape(1, -1).repeat(n, 1)
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    succ_env = np.zeros(n)
    active = np.ones(n, bool)
    for t in range(H):
        acts = np.clip(np.asarray(
            agent.act(torch.as_tensor(obs), z_b, eval_mode=True).cpu().numpy()), -1, 1)
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, _r, term, trunc, inf = e.step(acts[i])
            obs[i] = o
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if inf.get("success"):
                succ_env[i] = 1.0
            if term or trunc:
                active[i] = False
        if not active.any():
            break
    return obs_traj, lengths, succ_env


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="hydra run dir (.hydra/config.yaml + checkpoints/)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--ckpt", default="", help="explicit checkpoint path (overrides --checkpoint-step)")
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--region-min-pos", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=0, help="0 = the env's own max_episode_steps")
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--pool_size", type=int, default=16384)
    ap.add_argument("--pool_cap", type=int, default=0,
                    help="rows drawn for the pool (density used max(pool_size, sweep "
                         "sizes) = pool_size for the archived P16384 runs); 0 = pool_size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randz-floor", type=int, default=1,
                    help="1 = also run the prior-z floor arm (chance reference)")
    ap.add_argument("--native-z", type=int, default=0,
                    help="1 = also run the NATIVE-READOUT arm: z from the run's own "
                         "OGBenchEvaluator._infer_z (relabelled offline transitions, "
                         "the run's recorded eval_relabel_size / eval_shift_reward) "
                         "rolled out through THIS harness. Same rollout code, "
                         "different z: it separates a harness error from a "
                         "reward-definition difference. Needs the offline buffer.")
    ap.add_argument("--native-success", type=float, default=float("nan"),
                    help="the checkpoint's own wandb success/overall, recorded for "
                         "the dirac-vs-native control")
    ap.add_argument("--ref-json", default="",
                    help="archived density result to assert protocol equality against")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"

    import ogbench
    import torch
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))

    from evals.analysis import load_cfg, load_checkpoint
    from train import make_agent

    run_dir = Path(args.run).expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    assert cfg_path.exists(), f"{run_dir} has no .hydra/config.yaml (arch would be guessed)"
    cfg = load_cfg(str(cfg_path), device="cpu")
    for w in check_agent_consistency(cfg.agent, cfg.get("obs_type", "state"),
                                     force=args.force, run_dir=str(run_dir)):
        print(f"[fb-zs] {w}", flush=True)

    env_name = str(cfg.domain)
    geo = geometry_for(env_name, "")
    spec = geo.hit_spec
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    H = args.horizon if args.horizon > 0 else int(_e0.spec.max_episode_steps)
    obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    act_dim = int(np.asarray(ds1["actions"]).shape[-1])

    # ---- agent restore (architecture from the run's OWN recorded config) ----
    probe_env = ogbench.make_env_and_datasets(eval_tasks[1], env_only=True)
    agent = make_agent(cfg, probe_env.observation_space, act_dim)
    ckpt = args.ckpt or str(run_dir / "checkpoints" / f"step_{args.checkpoint_step}.pt")
    assert Path(ckpt).exists(), f"missing checkpoint {ckpt}"
    load_checkpoint(agent, ckpt, map_location="cpu")
    agent.model.eval()
    # FB's flow actor draws fresh action noise EVERY step (FBFlowBCModel.actor;
    # `mean=True` is ignored by design), so the deployed policy is stochastic --
    # that is FB's own eval-time behaviour and is kept, but seeded, so a cell is
    # reproducible instead of drifting by a couple of episodes between reruns.
    torch.manual_seed(args.seed)
    probe_env.close()
    print(f"[fb-zs] restored {run_dir.name} agent={cfg.agent} ckpt={Path(ckpt).name} "
          f"env={env_name} z_dim={cfg.z_dim} ortho_coef={cfg.ortho_coef} "
          f"obs_dim={obs_dim} H={H} P={args.pool_size} ({time.time()-t00:.0f}s)", flush=True)

    # ---- the density harness's rng stream, consumed in the SAME order ----
    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    idx = rng_np.choice(len(coords_all), size=min(args.relabel_size, len(coords_all)),
                        replace=False)
    relabel_coords = coords_all[idx]
    pool_cap = args.pool_cap or args.pool_size
    n_ds = len(obs_all)
    pool_idx = rng_np.choice(n_ds, size=min(pool_cap, n_ds), replace=(pool_cap > n_ds))
    pool_obs = obs_all[pool_idx][:args.pool_size]
    pool_coords = coords_all[pool_idx][:args.pool_size]

    rng_reg = np.random.default_rng(args.seed + 2)
    centers, _centers_obs, npos_c, rejected = draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel_coords, rng_reg, args.regions,
        min_pos=args.region_min_pos)

    goal_coord = {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_coord[i] = spec.coords(np.asarray(info["goal"], np.float32)[None])[0]
        env.close()

    meta = {
        "run_dir": str(run_dir), "checkpoint": ckpt, "checkpoint_step": args.checkpoint_step,
        "agent": str(cfg.agent), "obs_type": str(cfg.get("obs_type", "state")),
        "z_dim": int(cfg.z_dim), "ortho_coef": float(cfg.ortho_coef),
        "env": env_name, "d_c": spec.dim, "gamma": float(cfg.discount),
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in spec.blocks),
        "pool_size": args.pool_size, "pool_cap": pool_cap, "seed": args.seed,
        "episodes": args.episodes, "horizon": H, "relabel_size": int(len(relabel_coords)),
        "tasks": task_ids, "families": families,
        "readout": {
            "z_r": "project(sum_i r(x_i) B(s'_i)) -- FBModel.reward_inference, FB's "
                   "own reward interface (agents/fb/agent.py infer_z)",
            "policy": "FBAgent.act(obs, z_r): the trained flow actor conditioned on "
                      "z_r; Q_r(s,a)=F(s,a,z_r)^T z_r is what it was trained to "
                      "maximize. No goal decode and no retrieval -- that is the one "
                      "intended difference from the density columns",
            "families": "reward_conj_dirac / reward_conj_dense / region centers "
                        "imported from eval_zeroshot_rewards.py + "
                        "eval_partition_zeroshot.py, the same functions "
                        "eval_density_zeroshot.py calls",
            "success": "env info['success'] latched over the episode (dirac/dense); "
                       "conjunctive region membership at the final step (region)",
        },
        "native_success_wandb": (None if np.isnan(args.native_success)
                                 else float(args.native_success)),
        "goal_coords": {f"t{i}": goal_coord[i].tolist() for i in task_ids},
    }

    results = {"meta": meta, "families": {}, "controls": {}}

    if args.ref_json:
        ref = check_reference(args.ref_json, meta, centers)
        results["meta"]["reference_check"] = ref
        print(f"[fb-zs] reference check vs {Path(args.ref_json).name}: "
              f"{'PASS' if ref['ok'] else 'FAIL'}", flush=True)
        for k, v in ref["checks"].items():
            if not v.get("match", False):
                print(f"[fb-zs]   MISMATCH {k}: {v}", flush=True)
        assert ref["ok"] or args.force, \
            "protocol does not match the archived density run; the column would not " \
            "be comparable (pass --force only if you know why)"

    pools: dict[int, list] = {}

    def pool_envs(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
                        for _ in range(args.episodes)]
        return pools[i]

    table = []

    def run_task_cell(fam_name: str, i: int, r_pool: np.ndarray):
        z, diag = fb_infer_z(agent, pool_obs, r_pool)
        if z is None or diag.get("degenerate"):
            print(f"[fb-zs][{fam_name}] t{i}: DEGENERATE z_r {diag}", flush=True)
            return {"success": None, "z_diag": diag}
        _ot, _ln, senv = batched_z_rollout(agent, pool_envs(i), z, H)
        rec = {"success": float(senv.mean()), "z_diag": diag}
        print(f"[fb-zs][{fam_name}] t{i}: succ={rec['success']:.3f} "
              f"r_nonzero={diag['pool_r_nonzero']} ({time.time()-t00:.0f}s)", flush=True)
        return rec

    # ================= 1. dirac =================
    if "dirac" in families:
        fam = {}
        for i in task_ids:
            fam[f"t{i}"] = run_task_cell("dirac", i, reward_conj_dirac(pool_coords, goal_coord[i], spec))
            table.append(("dirac", f"t{i}", "success", str(fam[f"t{i}"]["success"])))
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dirac"] = fam
        table.append(("dirac", "MEAN", "success", str(fam["mean"])))

    # ================= 2. dense =================
    if "dense" in families:
        fam = {}
        for i in task_ids:
            fam[f"t{i}"] = run_task_cell("dense", i, reward_conj_dense(pool_coords, goal_coord[i], spec))
            table.append(("dense", f"t{i}", "success", str(fam[f"t{i}"]["success"])))
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dense"] = fam
        table.append(("dense", "MEAN", "success", str(fam["mean"])))

    # ================= 3. region =================
    if "region" in families:
        v0 = task_ids[0]
        fam = {"n_centers": int(len(centers)), "centers_rejected": int(rejected),
               "center_min_pos": int(args.region_min_pos)}
        print(f"[fb-zs][region] {len(centers)}/{args.regions} hindsight centers "
              f"({rejected} rejected at min_pos={args.region_min_pos})", flush=True)
        for k_i, c in enumerate(centers):
            r_pool = reward_conj_dirac(pool_coords, c, spec)
            z, diag = fb_infer_z(agent, pool_obs, r_pool)
            if z is None or diag.get("degenerate"):
                fam[f"c{k_i}"] = {"success_final": None, "success_any": None,
                                  "z_diag": diag, "center_coord": np.asarray(c).tolist()}
                print(f"[fb-zs][region] c{k_i}: DEGENERATE z_r {diag}", flush=True)
                continue
            ot, ln, _se = batched_z_rollout(agent, pool_envs(v0), z, H)
            fin, anys = region_success_from_obs(ot, ln, c, spec)
            fam[f"c{k_i}"] = {"success_final": float(fin.mean()),
                              "success_any": float(anys.mean()),
                              "center_coord": np.asarray(c).tolist(),
                              "center_n_pos": int(npos_c[k_i]), "z_diag": diag}
            table.append(("region", f"c{k_i}", "success_final", str(fam[f"c{k_i}"]["success_final"])))
            print(f"[fb-zs][region] c{k_i}: fin={fam[f'c{k_i}']['success_final']:.3f} "
                  f"any={fam[f'c{k_i}']['success_any']:.3f} "
                  f"r_nonzero={diag['pool_r_nonzero']} ({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        vals = [v["success_final"] for v in cells if v["success_final"] is not None]
        anyv = [v["success_any"] for v in cells if v["success_any"] is not None]
        fam["mean_final"] = float(np.mean(vals)) if vals else None
        fam["mean_any"] = float(np.mean(anyv)) if anyv else None
        results["families"]["region"] = fam
        table.append(("region", "MEAN", "success_final", str(fam["mean_final"])))

    # ================= control: prior-z floor =================
    if args.randz_floor:
        torch.manual_seed(1000 + args.seed)
        floor = {"note": "z ~ model.sample_z (the model's own prior over task "
                         "indices): what this success criterion scores with no "
                         "task information at all",
                 "tasks": {}, "regions": {}}
        for i in task_ids:
            z = agent.model.sample_z(1).reshape(-1)
            _ot, _ln, senv = batched_z_rollout(agent, pool_envs(i), z, H)
            floor["tasks"][f"t{i}"] = float(senv.mean())
            print(f"[fb-zs][randz] t{i}: succ={float(senv.mean()):.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        if len(centers):
            v0 = task_ids[0]
            for k_i, c in enumerate(centers):
                z = agent.model.sample_z(1).reshape(-1)
                ot, ln, _se = batched_z_rollout(agent, pool_envs(v0), z, H)
                fin, _anys = region_success_from_obs(ot, ln, c, spec)
                floor["regions"][f"c{k_i}"] = float(fin.mean())
                print(f"[fb-zs][randz] c{k_i}: fin={float(fin.mean()):.3f} "
                      f"({time.time()-t00:.0f}s)", flush=True)
        floor["task_mean"] = float(np.mean(list(floor["tasks"].values()))) if floor["tasks"] else None
        floor["region_mean"] = float(np.mean(list(floor["regions"].values()))) if floor["regions"] else None
        results["controls"]["randz_floor"] = floor
        table.append(("randz", "TASKS", "success", str(floor["task_mean"])))
        table.append(("randz", "REGIONS", "success_final", str(floor["region_mean"])))

    # ================= control: FB's NATIVE readout, same rollout code =========
    if args.native_z:
        from data.ogbench import load_ogbench_dataset
        from evals.ogbench import OGBenchEvaluator

        buf = load_ogbench_dataset(
            domain=env_name, data_path=cfg.data_path,
            load_n_episodes=cfg.load_n_episodes, device="cpu",
            n_transitions=cfg.n_transitions, obs_type=cfg.get("obs_type", "state"))
        ev = OGBenchEvaluator(
            domain=env_name, agent=agent, offline_buffer=buf,
            relabel_size=int(cfg.eval_relabel_size), n_episodes=args.episodes,
            shift_reward=float(cfg.eval_shift_reward),
            obs_type=str(cfg.get("obs_type", "state")), seed=args.seed, device="cpu")
        nat = {"note": "z from evals/ogbench.py OGBenchEvaluator._infer_z -- the run's "
                       "OWN native task-inference (relabelled offline transitions, "
                       "reward shifted by eval_shift_reward), rolled out through THIS "
                       "script's rollout. Differs from the dirac family ONLY in how "
                       "the reward fed to the SAME estimator is defined.",
               "relabel_size": int(cfg.eval_relabel_size),
               "shift_reward": float(cfg.eval_shift_reward), "tasks": {}}
        torch.manual_seed(2000 + args.seed)
        for i in task_ids:
            z, rmet = ev._infer_z(eval_tasks[i])
            _ot, _ln, senv = batched_z_rollout(agent, pool_envs(i), z.reshape(-1), H)
            nat["tasks"][f"t{i}"] = {"success": float(senv.mean()),
                                     "relabel": {k: float(v) for k, v in rmet.items()}}
            print(f"[fb-zs][nativez] t{i}: succ={float(senv.mean()):.3f} "
                  f"relabel_mean={rmet.get('relabel_reward#mean')} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        nat["mean"] = float(np.mean([v["success"] for v in nat["tasks"].values()]))
        results["controls"]["native_z"] = nat
        table.append(("nativez", "TASKS", "success", str(nat["mean"])))

    if not np.isnan(args.native_success):
        d = results["families"].get("dirac", {}).get("mean")
        results["controls"]["native_vs_dirac"] = {
            "native_success_overall_wandb": float(args.native_success),
            "zeroshot_dirac_mean": d,
            "abs_diff": (None if d is None else abs(d - float(args.native_success))),
            "note": "dirac = the five native tasks with indicator rewards at the "
                    "native radius (obs cube xyz = (m - CEN)*10 so eps 0.4 = the "
                    "0.04 m OGBench threshold); a wild disagreement means the "
                    "harness is wrong, not that FB transfers",
        }
        table.append(("control", "native", "success", str(args.native_success)))

    for i in list(pools):
        for e in pools.pop(i):
            if hasattr(e, "close"):
                e.close()

    outp = Path(args.out) if args.out else (REPO / "results" / f"fbzs_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[fb-zs] DONE ({time.time()-t00:.0f}s) -> {outp}\n", flush=True)
    print_table(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
