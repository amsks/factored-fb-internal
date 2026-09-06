"""Re-score EXISTING cube checkpoints under td_jepa's reward-relabel zero-shot protocol.

No retraining. Only how the task is communicated at test time changes.

WHAT THIS MEASURES, and why it is not what our tables currently report. Every cube cell in
`table.tex` / `table_privcoord.tex` / `table_objcoords.tex` specifies the task with a single
GOAL OBSERVATION -- the critic's `map_eval_cond` turns it into z = project_z(B(goal)), a
Dirac. td_jepa specifies the task with a REWARD FUNCTION sampled over the data, and reports
the policy inferred from those samples. On the same `fb_tdj` checkpoints that difference is
0.197 -> 0.378, i.e. the readout, not the training stack, is what put our FB numbers a factor
of two below td_jepa's published 0.420.

It is also the interface the project actually requires: rewards in, policy out. A
goal-conditioned-only number cannot stand in for it.

PER TASK, PER CHECKPOINT:
  1. draw `--relabel-size` random transitions from the run's own TRAIN dataset;
  2. relabel each by its NEXT state's simulator qpos -- 1 if the cube is within 4 cm of that
     task's goal, else 0 (`cube_relabel_reward`, transcribed from td_jepa's `cube_reward_fn`;
     `shift_reward=1.0` turns `successes - num_cubes` into that 0/1 indicator). The dataset's
     own `rewards` column is NEVER used: FBDataset zeroes it for unlabelled play data
     (datasets.py:126-128), so inferring from it would silently solve for a constant reward;
  3. `agent.infer_eval_cond(obs, rewards)` -> the task vector, through the critic's own
     `infer_cond`. Every critic implements it, so this works for FB, RLDP, HILP, PSM and
     Density-FB alike;
  4. roll `--episodes` episodes wrapped in `InferredCondAgent`, which REFUSES the env's goal
     frame -- without it `sample_actions` would merge the goal back in and `map_eval_cond`
     would overwrite the vector we just inferred. `witness()` records how many goal frames
     were intercepted, so a row with `inferred_cond_goals_dropped == 0` is not trustworthy;
  5. score success as ANY step reaching the goal (td_jepa's accounting).

This is `impls/main.py:512-546` lifted out of the training loop and run over checkpoints. It
is deliberately a copy of that call sequence rather than a reimplementation.

CAVEAT ON VARIANCE, stated up front: the relabel rows are redrawn at every invocation, so
this eval carries sampling noise on top of policy noise. `fb_tdj` seed 0 moved 0.60 -> 0.356
between its 800k and 1M eval points. Quote the seed count and the episode count with any
number this produces.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np


def rescore_run(run_dir: Path, epoch: int, relabel_size: int, episodes: int, shift: float):
    """One checkpoint -> a list of per-task rows."""
    import jax.numpy as jnp

    from env import get_adapter
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.eval_cond import InferredCondAgent
    from impls.utils.flax_utils import restore_agent

    meta = json.loads((run_dir / "config.json").read_text())
    # eval_relabel_size MUST be in cfg BEFORE make_env_and_datasets: it is what gates loading
    # the dataset's `qpos` column (env/ogbench.py:114-119), and relabel_infer_batch needs it.
    cfg = {**meta["config"], "eval_relabel_size": relabel_size,
           "eval_relabel_shift": shift, "eval_relabel_episodes": episodes}
    np.random.seed(meta["seed"])

    adapter = get_adapter(cfg)
    env, train_ds, _val = adapter.make_env_and_datasets(cfg)
    if not hasattr(adapter, "relabel_infer_batch"):
        raise NotImplementedError(f"{type(adapter).__name__} has no relabel_infer_batch")
    ex = make_dataset(train_ds, cfg).sample(1)
    agent = Agent.create(meta["critic"], meta["actor"], meta["seed"],
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, str(run_dir), epoch)

    rows = []
    for tid in range(1, len(env.unwrapped.task_infos) + 1):
        obs, rewards = adapter.relabel_infer_batch(train_ds, env, tid, relabel_size,
                                                   shift_reward=shift)
        ra = agent.infer_eval_cond(jnp.asarray(obs), jnp.asarray(rewards))
        ra = InferredCondAgent(ra)
        stats, _frames = adapter.evaluate(ra, env, tid, cfg, episodes, 0,
                                          any_step_success=True)
        row = {**stats, **ra.witness(), "task_id": tid, "critic": meta["critic"],
               "actor": meta["actor"], "seed": meta["seed"], "epoch": epoch,
               "env_name": cfg["env_name"], "relabel_size": relabel_size,
               "episodes": episodes, "run_dir": str(run_dir),
               "reward_frac_positive": float(np.mean(np.asarray(rewards) > 0))}
        rows.append(row)
        print(f"    task {tid}: success={row.get('success'):.3f} "
              f"goals_dropped={row.get('inferred_cond_goals_dropped')} "
              f"pos_reward_frac={row['reward_frac_positive']:.4f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=10, help="td_jepa's protocol is 10")
    ap.add_argument("--relabel-size", type=int, default=10000)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--exp-root", default="exp")
    ap.add_argument("--out", default="analysis/relabel_rescore.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for critic in args.critics:
        for seed in args.seeds:
            d = Path(args.exp_root) / f"{args.env}_{critic}_flowbc_seed_{seed}"
            if not (d / f"params_{args.epoch}.pkl").exists():
                print(f"[skip] {d.name}: no params_{args.epoch}.pkl", flush=True)
                continue
            print(f"[rescore] {d.name}", flush=True)
            try:
                rows = rescore_run(d, args.epoch, args.relabel_size, args.episodes,
                                   args.shift)
            except Exception as e:                       # one bad arm must not kill the sweep
                print(f"[FAIL] {d.name}: {type(e).__name__}: {e}", flush=True)
                continue
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer))
                                            else v) for k, v in r.items()}) + "\n")
            n += len(rows)
            print(f"    -> mean over tasks: "
                  f"{np.mean([r['success'] for r in rows]):.3f}", flush=True)
    print(f"[rescore] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
