"""fb_zeroshot_l2.py — the MISSING paper cell: naive/tuned FB on the 48
held-out rewards (Table 1 right group, "Tuned FB" row; ledger §9.12 flagged
that "stock FB [has] NEVER been run through this harness").

FB's torch model (agents/fb/model.py) natively exposes the full readout API the
spec-12 harness asserts — reward_inference (z_r = Σ r_i B(s'_i), projected),
backward_map, project_z, sample_z — and FBAgent.act(obs, z, eval_mode) is the
z-conditioned actor, so NO adapter and NO retrieval decode are needed. Stages
A/B therefore DELEGATE to torchfb_zeroshot_l2.main() unmodified: reward
instances (make_rewards, seed 0 ⇒ verified to reproduce the archived
L2_behavior_baseline.json to 6 decimals), rng call order, episode counts,
horizon, discount, drives (zr / zgoal / zrand) and reporting are bit-identical
to the archived protocol that produced results/spec10_torchfb/L2_B_*.json.

`--stage success` additionally runs the run's NATIVE OGBench 5-task eval
(evals/ogbench.OGBenchEvaluator, same relabel-z inference as training-time
eval) on the same checkpoint, so the goal-reaching and reward-transfer numbers
in the paper row come from one set of weights.

Usage (GPU box, tmux):
  .venv/bin/python scripts/rescuing_fb/fb_zeroshot_l2.py \
      --run /home/adityamohan/exp_spec10/torchfb/fb_stock_s0 --stage B \
      --out results/spec10_torchfb/L2_B_fbstock_s0.json
  .venv/bin/python scripts/rescuing_fb/fb_zeroshot_l2.py \
      --run /home/adityamohan/exp_spec10/torchfb/fb_stock_s0 --stage success \
      --episodes 50 --out results/spec10_torchfb/success_fbstock_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_success(a) -> int:
    """Native OGBench eval: 5 tasks, z from relabelled offline transitions."""
    import numpy as np
    import torch

    from data.ogbench import load_ogbench_dataset
    from envs.ogbench import create_ogbench_env
    from evals.analysis import load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from train import make_agent

    cfg = load_cfg(f"{a.run}/.hydra/config.yaml", device=a.device)
    print(f"[cfg] agent={cfg.agent} domain={cfg.domain} "
          f"goal_cond={cfg.get('goal_cond')} fixed_b={cfg.get('fixed_b')} "
          f"z_dim={cfg.z_dim} ortho_coef={cfg.ortho_coef}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=a.seed)
    agent = make_agent(cfg, env.observation_space,
                       int(env.action_space.shape[0]))
    load_checkpoint(agent, f"{a.run}/{a.ckpt}", map_location=a.device)
    agent.model.eval()
    env.close()

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=cfg.device,
        n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    print(f"[success] buffer size: {len(buffer):,}")

    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=a.episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        seed=a.seed, device=cfg.device)
    metrics = evaluator.run()
    for k in sorted(metrics):
        if k.endswith("/success") or k == "eval/success":
            print(f"[success] {k} = {metrics[k]:.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"run": a.run, "ckpt": a.ckpt, "episodes": a.episodes,
                   "seed": a.seed, "metrics": metrics}
        json.dump(payload, open(a.out, "w"), indent=1)
        print(f"[out] {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--stage", choices=["A", "B", "success"], required=True)
    ap.add_argument("--episodes", type=int, default=10,
                    help="stage B: episodes/drive; success: episodes/task")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a, passthrough = ap.parse_known_args()

    if a.stage == "success":
        return run_success(a)

    # Stages A/B: delegate to the archived spec-12 harness verbatim. FB passes
    # its readout-API asserts natively, so the protocol is bit-identical to the
    # runs behind results/spec10_torchfb/L2_*.json.
    import torchfb_zeroshot_l2 as harness
    argv = [sys.argv[0], "--run", a.run, "--ckpt", a.ckpt, "--stage", a.stage,
            "--episodes", str(a.episodes), "--device", a.device,
            "--seed", str(a.seed)]
    if a.out:
        argv += ["--out", a.out]
    argv += passthrough
    sys.argv = argv
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
