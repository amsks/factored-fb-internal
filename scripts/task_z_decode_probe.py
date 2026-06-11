#!/usr/bin/env python
"""scripts/task_z_decode_probe.py — 4th angle: WHAT does each training-z encode?

Reuses the task-distribution goal pool (z = B(s')/phi(s') for buffer next-states s')
and asks which state factors are linearly recoverable from z (held-out R^2 / AUC):
  cube_xyz (3d), cube lift (height), hand-cube distance, gripper contact (binary).
High R^2 => the task embedding "carries" that factor => the z's correspond to tasks
that distinguish states along it. CPU-only. Run with PYTHONPATH=. .venv/bin/python.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data.ogbench import load_ogbench_dataset
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.task_distribution import estimate_table_z, EFF_SLICE, CUBE_OBS, CONTACT_IDX, CUBE_PHYSICS
from scripts.task_distribution_probe import embed_pool, METHODS, RUNS, DATA, DOMAIN

OUT = Path("analysis/task_distribution")


def build_pool_with_factors(m):
    torch.manual_seed(0)
    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state", load_n_episodes=1000)
    batch = buf.sample(min(m, len(buf)))
    next_obs = torch.as_tensor(batch["next"]["observation"], dtype=torch.float32)
    o = next_obs.numpy()
    cube_phys_z = np.asarray(batch["next"]["physics"])[:, CUBE_PHYSICS][:, 2]
    lift = cube_phys_z - estimate_table_z(cube_phys_z)
    factors = {
        "cube_xyz": o[:, CUBE_OBS],                                   # 3d (obs frame)
        "cube_lift": lift.reshape(-1, 1),                             # height above table
        "hand_cube_dist": np.linalg.norm(o[:, EFF_SLICE] - o[:, CUBE_OBS], axis=1).reshape(-1, 1),
    }
    gripper = (o[:, CONTACT_IDX] > 0.5).astype(int)                   # binary contact
    return buf, next_obs, factors, gripper


def decode(Z, factors, gripper, seed=0):
    out = {}
    Ztr, Zte, idx_tr, idx_te = train_test_split(Z, np.arange(len(Z)), test_size=0.3, random_state=seed)
    for name, Y in factors.items():
        m = Ridge(alpha=1.0).fit(Ztr, Y[idx_tr])
        out[name] = float(r2_score(Y[idx_te], m.predict(Zte)))           # held-out R^2
    # gripper: AUC of a linear logistic readout
    gtr, gte = gripper[idx_tr], gripper[idx_te]
    if gtr.min() != gtr.max():
        lr = LogisticRegression(max_iter=500).fit(Ztr, gtr)
        out["gripper_auc"] = float(roc_auc_score(gte, lr.predict_proba(Zte)[:, 1]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=50000)
    ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    m = 4000 if args.quick else args.m
    max_seeds = 2 if args.quick else 10
    buf, next_obs, factors, gripper = build_pool_with_factors(m)
    print(f"pool {len(next_obs)} states; factors {list(factors)} + gripper")
    summary = {}
    for method, (prefix, kind) in METHODS.items():
        rows = []
        for d in sorted(Path(RUNS).glob(f"{prefix}__s*"))[:max_seeds]:
            cfg = load_cfg(str(d / ".hydra/config.yaml"), device="cpu")
            _, agent, _, _ = build_env_and_agent(cfg); agent.model.train(False)
            load_checkpoint(agent, str(d / "checkpoints/final.pt"))
            Z = embed_pool(agent.model, next_obs, kind)
            r = decode(Z, factors, gripper)
            rows.append(r)
            desc = ", ".join(f"{k}={v:.2f}" for k, v in r.items())
            print(f"  {method} {d.name.split('__s')[-1]}: {desc}")
        keys = rows[0].keys()
        summary[method] = {k: {"mean": float(np.mean([x[k] for x in rows])),
                               "std": float(np.std([x[k] for x in rows]))} for k in keys}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "z_decode.json").write_text(json.dumps(summary, indent=2))
    # markdown table
    keys = list(next(iter(summary.values())).keys())
    lines = ["# What each training-z encodes (held-out R^2 / AUC, mean±std over seeds)", "",
             "| method | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for mth, d in summary.items():
        lines.append("| " + mth + " | " + " | ".join(f"{d[k]['mean']:.2f}±{d[k]['std']:.2f}" for k in keys) + " |")
    (OUT / "z_decode_summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines)); print(f"\nwrote {OUT}/z_decode_summary.md")


if __name__ == "__main__":
    main()
