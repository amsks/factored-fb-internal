"""scripts/gciql_profile.py — per-checkpoint GCIQL profiler.

Runs ONLY in the isolated JAX venv (it imports jax/flax/vendored
OGBench inside main()/_jax_*). The pure seams below (flags parsing,
episode-records -> frames) are JAX-free and unit-tested.

T1 (Spearman rho(V(s,g), -d)), T4 (off-support coverage) and the phase
funnel are computed with evals._profile_core / evals.phase_probe so the
FB and GCIQL numbers are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import _spearman, probe_coverage, transport_mask
from evals.phase_probe import Thresholds, classify_phases


def parse_flags(run_dir) -> Dict[str, Any]:
    """Read OGBench flags.json from a GCIQL seed dir."""
    data = json.loads((Path(run_dir) / "flags.json").read_text())
    return {"env_name": data.get("env_name", "cube-single-play-v0"),
            "seed": int(data.get("seed", 0)),
            "agent": data.get("agent", "agents/gciql.py")}


def episodes_to_frames(episodes: List[Dict[str, Any]], thr: Thresholds,
                       ref_obs: np.ndarray):
    """Episode records -> (value_landscape, coverage, funnel) frames.

    Each record: obs[T,O], d[T], V[T], grip[T], cube[T,3], table_z,
    success(bool), task(str), episode(int). Pure; no jax.
    """
    vl_rows, fun_rows, cov_eps = [], [], []
    for ep in episodes:
        d = np.asarray(ep["d"], np.float64)
        V = np.asarray(ep["V"], np.float64)
        tm = transport_mask(ep, thr)
        sig = {
            "eff": np.zeros((len(d), 3)),
            "cube": np.asarray(ep["cube"], np.float64).reshape(-1, 3),
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "goal": np.asarray(ep["cube"], np.float64).reshape(-1, 3)[-1],
            "success": bool(ep["success"]),
            "table_z": float(ep["table_z"]),
            "length": len(d),
        }
        cls = classify_phases(sig, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        n_t = int(tm.sum())
        if n_t >= 5 and np.ptp(d[tm]) > 1e-9 and np.ptp(V[tm]) > 1e-9:
            rho = _spearman(V[tm], -d[tm])
            v_secure, v_end = float(V[tm][0]), float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        vl_rows.append({"task": ep["task"], "episode": ep["episode"],
                        "outcome": outcome, "rho_V_negd": rho,
                        "V_at_secure": v_secure, "V_at_end": v_end,
                        "n_transport_steps": n_t})
        fun_rows.append({"task": ep["task"], "episode": ep["episode"],
                         "furthest_phase": cls["fail_phase"]
                         if not cls["success"] else "success",
                         "outcome": outcome})
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
    cov = probe_coverage(cov_eps, ref_obs, thr)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="a GCIQL seed dir (contains flags.json, "
                         "params_<step>.pkl)")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--scenario", default="S0", choices=["S0"])
    ap.add_argument("--dataset-path",
                    default="datasets/cube-single-play-v0")
    ap.add_argument("--max-steps", type=int, default=1000)
    return ap


def _load_ref_obs(dataset_path: str, max_n: int = 5000) -> np.ndarray:
    """Stack next/observation from the OGBench cube npz buffer (same
    data the FB coverage probe used)."""
    buf = Path(dataset_path) / "buffer"
    files = sorted(buf.glob("episode_*.npz"))[:200]
    obs = []
    for f in files:
        with np.load(f) as z:
            key = "observations" if "observations" in z else "obs"
            obs.append(np.asarray(z[key], np.float32))
    arr = np.concatenate(obs, 0) if obs else np.zeros((1, 1), np.float32)
    if len(arr) > max_n:
        rng = np.random.default_rng(0)
        arr = arr[rng.choice(len(arr), max_n, replace=False)]
    return arr


def _jax_rollout(run_dir: str, step: int, task_ids: List[int],
                 n_episodes: int, max_steps: int):
    """Load the GCIQL checkpoint and roll out the goal-conditioned
    policy. Imports jax/flax/vendored OGBench — JAX venv only."""
    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb_impls) not in sys.path:
        sys.path.insert(0, str(ogb_impls))
    import ogbench  # vendored
    from agents.gciql import GCIQLAgent
    from utils.flax_utils import restore_agent
    import jax

    flags = parse_flags(run_dir)
    episodes: List[Dict[str, Any]] = []
    for task_id in task_ids:
        env = ogbench.make_env_and_datasets(
            flags["env_name"], compact_dataset=True, env_only=True) \
            if hasattr(ogbench, "make_env_and_datasets") else \
            ogbench.make_env(flags["env_name"])
        ex_obs, info = env.reset(options=dict(task_id=task_id))
        goal = info["goal"]
        ex_act = env.action_space.sample()
        agent = GCIQLAgent.create(
            flags["seed"],
            np.asarray(ex_obs)[None],
            np.asarray(goal)[None],
            np.asarray(ex_act)[None],
            GCIQLAgent.get_config(),
        )
        agent = restore_agent(agent, str(run_dir), step)
        value_fn = jax.jit(lambda o, g: agent.network.select("value")(
            o[None], g[None])[0])
        for ep_i in range(n_episodes):
            obs, info = env.reset(options=dict(task_id=task_id))
            goal = info["goal"]
            O, A, EFF, CUBE, GRIP, VV, D = [], [], [], [], [], [], []
            succ = False
            for _ in range(max_steps):
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(obs), goals=np.asarray(goal),
                    temperature=0.0))
                v = float(np.asarray(value_fn(
                    np.asarray(obs, np.float32),
                    np.asarray(goal, np.float32))))
                obs, _, term, trunc, info = env.step(a)
                eff = np.asarray(info["proprio/effector_pos"], np.float64)
                cube = np.asarray(info["privileged/block_0_pos"],
                                  np.float64)
                grip = float(np.asarray(
                    info["proprio/gripper_opening"]).reshape(-1)[0])
                gxyz = np.asarray(info.get("privileged/target_block_pos",
                                           cube), np.float64)
                O.append(np.asarray(obs, np.float32)); A.append(a)
                EFF.append(eff); CUBE.append(cube); GRIP.append(grip)
                VV.append(v)
                D.append(float(np.linalg.norm(cube - gxyz)))
                succ = succ or bool(info.get("success", False))
                if term or trunc:
                    break
            episodes.append({
                "obs": np.asarray(O, np.float32),
                "d": np.asarray(D), "V": np.asarray(VV),
                "grip": np.asarray(GRIP),
                "cube": np.asarray(CUBE),
                "table_z": float(CUBE[0][2]) if CUBE else 0.02,
                "success": succ,
                "task": f"task{task_id}", "episode": ep_i})
        env.close()
    return episodes, flags


def main() -> None:
    args = build_argparser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    episodes, flags = _jax_rollout(
        args.run_dir, args.step, task_ids, args.n_episodes,
        args.max_steps)
    ref = _load_ref_obs(args.dataset_path)
    vl, cov, fun = episodes_to_frames(episodes, Thresholds(), ref)
    vl.to_parquet(out / "value_landscape.parquet")
    cov.to_parquet(out / "coverage.parquet")
    fun.to_parquet(out / "phase_funnel.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "step": args.step,
        "tasks": task_ids, "n_episodes": args.n_episodes,
        "env_name": flags["env_name"],
        "value_fn": "GCValue V(s,g)",
    }, indent=2))
    print(f"[gciql_profile] done -> {out}")


if __name__ == "__main__":
    main()
