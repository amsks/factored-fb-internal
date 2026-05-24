"""Factorization-ablation + readout-ceiling probe on a frozen FB checkpoint.

Answers "why does FB fail at transport: data (B1), FB's use of data (B2), or the
bilinear FᵀB factorization (B3)?" by measuring, on frozen embeddings, the
readout ceiling R^2(feature -> cube-to-goal distance d) — the paper's R^2(B->d)
— per manipulation phase, for raw obs vs the forward branch's left-encoder vs
the backward map B, with linear vs MLP readouts.

Static geometry (d) is the target: in PLAY data, future-outcome targets
(success_bound / mc_return) are exploration-driven and ~unpredictable, so they
are deliberately not used here (see evals.factorization_probe for those helpers,
kept for an on-policy value-calibration probe).

Goal-independent embeddings (B, left_enc, raw) are extracted once; only d depends
on the task goal, so all 5 tasks are swept cheaply. Reuses the extraction
plumbing from scripts/training_value_profile.py (no existing files modified).
Run under .venv (torch). macOS: prefix MUJOCO_GL=glfw. Example:

  MUJOCO_GL=glfw .venv/bin/python -m scripts.factorization_probe \
    --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt \
    --data-path datasets --out analysis/factorization_probe/s3 \
    --n-states 6000 --tasks all --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evals.training_value import region_labels, cube_to_goal_dist
from evals.factorization_probe import readout_ceiling_table, classify_bucket_regression
# Reuse the tested extraction helpers + verified column constants (read-only).
from scripts.training_value_profile import (
    _sample_states, _episode_files, CUBE_SLICE, GRIP_QPOS_IDX, TASK_TMPL,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--tasks", default="all", help="comma-sep task ids (1-N) or 'all'")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from envs.ogbench import create_ogbench_env, ALL_TASKS
    from evals.phase_probe import Thresholds

    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.model.eval()
    if hasattr(env, "close"):
        env.close()

    thr = Thresholds()
    # table-z (block init height; shared across tasks) from the task1 env.
    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed, obs_type=cfg.obs_type)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][
        int(getattr(e0.unwrapped, "_target_block", 0) or 0)][2])
    e0.close()

    # Sample states once; phase labels + frozen GOAL-INDEPENDENT embeddings once.
    files = _episode_files(cfg.data_path, cfg.domain)
    obs, phys, act, ep_id, step_id, cube_by_ep = _sample_states(files, args.n_states, seed=cfg.seed)
    cube = phys[:, CUBE_SLICE].astype(np.float64)
    grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
    region = [str(r) for r in region_labels(grip, phys[:, 16] - table_z, thr)]
    m = agent.model
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        B = m.backward_map(obs_t).cpu().numpy()
        left = m._left_encoder(m._fw_encoder(m._normalize(obs_t))).cpu().numpy()
    feats = {"raw": obs.astype(np.float64), "B": B, "left_enc": left}

    all_tasks = list(ALL_TASKS[cfg.domain])
    task_ids = (list(range(1, len(all_tasks) + 1)) if args.tasks == "all"
                else [int(t) for t in args.tasks.split(",")])

    # Per task: static cube->goal distance d -> readout ceiling R^2(feature -> d).
    ceilings, per_task = [], {}
    for ti in task_ids:
        e, _ = create_ogbench_env(TASK_TMPL.format(n=ti), seed=cfg.seed, obs_type=cfg.obs_type)
        g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][
            int(getattr(e.unwrapped, "_target_block", 0) or 0)], np.float64)
        e.close()
        frame = pd.DataFrame({"region": region, "d": cube_to_goal_dist(cube, g)})
        c = readout_ceiling_table(frame, feats, targets=("d",), seed=cfg.seed)
        c["task"] = f"task{ti}"
        ceilings.append(c)
        per_task[f"task{ti}"] = classify_bucket_regression(c, target="d")["bucket"]

    ceiling = pd.concat(ceilings, ignore_index=True)
    pooled = (ceiling.groupby(["feature", "target", "region", "kind"])["score"]
              .mean().reset_index())
    verdict = classify_bucket_regression(pooled, target="d")

    ceiling.to_parquet(out / "readout_ceiling.parquet")        # per-task rows (has 'task')
    pooled.to_parquet(out / "readout_ceiling_pooled.parquet")  # mean over tasks
    (out / "verdict.json").write_text(json.dumps({**verdict, "per_task": per_task}, indent=2))
    _write_story(out, pooled, verdict, per_task)
    print(f"[factorization_probe] verdict={verdict['bucket']} | per-task={per_task} :: {verdict['why']}")


def _write_story(out, pooled, verdict, per_task):
    tr = pooled[(pooled.region == "transport") & (pooled.target == "d")]
    lines = ["# Factorization probe — R^2(feature -> cube-to-goal d), pooled over tasks", "",
             f"**Verdict: {verdict['bucket']}** — {verdict['why']}", "",
             f"Per-task verdicts: {per_task}", "",
             "## Transport-phase readout ceilings (mean R^2 over tasks)", "",
             "| feature | kind | mean R^2 |", "| :-- | :-- | :-- |"]
    for _, r in tr.sort_values(["feature", "kind"]).iterrows():
        lines.append(f"| {r.feature} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
