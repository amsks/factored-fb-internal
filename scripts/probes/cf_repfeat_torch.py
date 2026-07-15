#!/usr/bin/env python
"""scripts/probes/cf_repfeat_torch.py — dump per-grasp value-input representation
for FB / RLDP / PSM on the counterfactual grasp records.

For each in-hand grasp in cf_records_<records>.parquet, extract F(s, pi(s,z), z)
-- the representation the bilinear value reads (PSM: sf_psi) -- so the matched-pair
aliasing analysis (opposite- vs same-outcome representation distances) can run
under .venv (sklearn), method-agnostically. Emits
analysis/value/repsep/cf_repfeat_<method>.parquet with {task, success, value,
obs_*, rep_*}.

Run under .venv (torch). macOS: MUJOCO_GL=glfw; Linux: egl.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_TASKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True, help="output tag: cf_repfeat_<method>.parquet")
    ap.add_argument("--records", required=True, help="cf_records_<records>.parquet basename tag")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--mujoco-gl", default="glfw")
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS
    from data.ogbench import load_ogbench_dataset
    from scripts.probes.repsep_forward_probe import F_rep

    repsep = REPO / "analysis/value/repsep"
    rec = pd.read_parquet(repsep / f"cf_records_{args.records}.parquet")
    obs_cols = [c for c in rec.columns if c.startswith("obs_")]

    cfg = load_cfg(args.config, device="cpu"); cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    model = agent.model
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size, n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                                 seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    zs = {ti: evaluator._infer_z(t)[0] for ti, t in enumerate(tasks, start=1)}

    out_rows = []
    for ti in range(1, N_TASKS + 1):
        d = rec[rec["task"] == ti]
        if len(d) == 0:
            continue
        obs = d[obs_cols].to_numpy(np.float32)
        Frep = np.asarray(F_rep(model, agent, obs, zs[ti]), np.float32)   # [N, z_dim]
        frame = pd.DataFrame({"task": ti,
                              "success": d["success"].to_numpy(bool),
                              "value": d["value"].to_numpy(np.float32)})
        if "p_succ" in d.columns:                        # multi-continuation soft label
            frame["p_succ"] = d["p_succ"].to_numpy(np.float32)
        for j in range(obs.shape[1]):
            frame[f"obs_{j}"] = obs[:, j]
        for j in range(Frep.shape[1]):
            frame[f"rep_{j}"] = Frep[:, j]
        out_rows.append(frame)

    out = pd.concat(out_rows, ignore_index=True)
    dst = repsep / f"cf_repfeat_{args.method}.parquet"
    out.to_parquet(dst)
    print(f"[cf_repfeat_torch:{args.method}] {len(out)} grasps, rep_dim="
          f"{sum(c.startswith('rep_') for c in out.columns)} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
