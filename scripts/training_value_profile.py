"""scripts/training_value_profile.py — value vs cube-to-goal-distance on the
offline TRAINING states, per phase, for FB (torch) and GCIQL (jax).

--method fb    : samples aligned (obs, physics, action) from the buffer npz,
                 classifies phases, scores FB policy/data values per task, and
                 writes the shared training_states.npz (run under .venv).
--method gciql : scores GCIQL (or GCIVL) V(s,g) on the SAME shared states
                 (run under .venv-jax-cpu).

Usage (FB):
    .venv/bin/python scripts/training_value_profile.py --method fb \
        --config RESULTS/Factored-FB-cube-run/<run>/.hydra/config.yaml \
        --checkpoint RESULTS/Factored-FB-cube-run/<run>/checkpoints/final.pt \
        --data-path datasets --mujoco-gl glfw --n-states 20000 \
        --out analysis/training_value

Usage (GCIQL):
    .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
        --run-dir RESULTS/gciql_.../sd003_... --step 1000000 \
        --out analysis/training_value --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import region_labels, cube_to_goal_dist

CUBE_SLICE = slice(14, 17)
GRIP_QPOS_IDX = 6      # ur5e/robotiq/right_driver_joint id (verified)
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"


def _episode_files(data_path: str, domain: str) -> List[str]:
    return sorted(glob.glob(str(Path(data_path) / domain / "buffer"
                                / "episode_*.npz")))


def _sample_states(files, n_states, seed=0):
    """Aligned (obs, physics, action) sampled across episodes."""
    rng = np.random.default_rng(seed)
    obs_l, phys_l, act_l = [], [], []
    for f in files:
        z = np.load(f)
        obs_l.append(np.asarray(z["observation"], np.float32))
        phys_l.append(np.asarray(z["physics"], np.float32))
        act_l.append(np.asarray(z["action"], np.float32))
    obs = np.concatenate(obs_l)
    phys = np.concatenate(phys_l)
    act = np.concatenate(act_l)
    if len(obs) > n_states:
        idx = rng.choice(len(obs), n_states, replace=False)
        obs, phys, act = obs[idx], phys[idx], act[idx]
    return obs, phys, act


def _phase_composition(files, thr, table_z):
    """Fraction reach/grasp/transport over ALL transitions (trajectory pass)."""
    import pandas as pd
    grip_l, lift_l = [], []
    for f in files:
        z = np.load(f)
        phys = np.asarray(z["physics"], np.float32)
        grip_l.append(np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1))
        lift_l.append(phys[:, 16] - table_z)
    reg = region_labels(np.concatenate(grip_l), np.concatenate(lift_l), thr)
    vc = pd.Series(reg).value_counts(normalize=True)
    return pd.DataFrame({"region": vc.index, "fraction": vc.values})


def _fb(args) -> None:
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import q_values, v_values
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.phase_probe import Thresholds
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    if hasattr(env, "close"):
        env.close()

    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed,
                               obs_type=cfg.obs_type)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    e0.close()
    thr = Thresholds()

    files = _episode_files(cfg.data_path, cfg.domain)
    obs, phys, act = _sample_states(files, args.n_states, seed=cfg.seed)
    cube = phys[:, CUBE_SLICE]
    grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
    region = region_labels(grip, phys[:, 16] - table_z, thr)

    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes,
                                  device=args.device,
                                  n_transitions=cfg.n_transitions,
                                  obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent,
                                 offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size,
                                 n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward,
                                 obs_type=cfg.obs_type, seed=cfg.seed,
                                 device=args.device, use_wandb=False)

    tasks = list(ALL_TASKS.get(cfg.domain, []))
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    rows, goals = [], {}
    for ti, task in enumerate(tasks, start=1):
        z, _ = evaluator._infer_z(task)
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        tbk = int(getattr(e.unwrapped, "_target_block", 0) or 0)
        g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tbk], np.float64)
        e.close()
        goals[f"task{ti}"] = g
        d = cube_to_goal_dist(cube, g)
        v_pol = v_values(agent.model, agent, obs_t, z)
        v_dat = q_values(agent.model, obs_t, act_t, z)
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": str(region[i]),
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]),
                         "V_policy": float(v_pol[i]),
                         "V_data": float(v_dat[i])})
    pd.DataFrame(rows).to_parquet(out / "fb_values.parquet")
    _phase_composition(files, thr, table_z).to_parquet(
        out / "phase_composition.parquet")
    np.savez(out / "training_states.npz", obs=obs, cube=cube,
             region=region.astype("U12"),
             goals=np.stack([goals[f"task{i}"] for i in range(1, 6)]))
    print(f"[training_value:fb] {len(obs)} states x {len(tasks)} tasks -> {out}")


def _gciql(args) -> None:
    raise NotImplementedError("GCIQL backend added in Task 3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["fb", "gciql"])
    ap.add_argument("--out", default="analysis/training_value")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--mujoco-gl", default=None)
    # fb
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path", default="datasets")
    # gciql
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int, default=1000000)
    args = ap.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.method == "fb":
        assert args.config and args.checkpoint, "fb needs --config/--checkpoint"
        _fb(args)
    else:
        assert args.run_dir, "gciql needs --run-dir"
        _gciql(args)


if __name__ == "__main__":
    main()
