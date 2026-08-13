"""Render benchmark-task rollouts of a TORCH checkpoint as GIFs.

Reuses the trained-and-tested OGBenchEvaluator end to end (relabel z-inference,
rollout, gif writer); the only change is the render resolution, injected by
wrapping `create_ogbench_env` before the evaluator builds its envs.

    .venv/bin/python scripts/rescuing_fb/render_policies_torch.py \
        --run <hydra run dir> [--ckpt checkpoints/final.pt] --tag revisitFB \
        --out results/videos [--episodes 1] [--res 400]

Writes <out>/<tag>/step_0/<task>.gif for each of the domain's 5 tasks.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/videos")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--relabel-size", type=int, default=10_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import envs.ogbench as EN
    import evals.ogbench as EV
    from data.ogbench import load_ogbench_dataset
    from evals.analysis import load_cfg, load_checkpoint
    from train import make_agent

    # Resolution injection: the evaluator calls create_ogbench_env(task, seed=...,
    # obs_type=..., frame_stack=...) and leaves render size at the 64x64 default.
    EV.create_ogbench_env = partial(EN.create_ogbench_env,
                                    render_height=a.res, render_width=a.res)

    cfg = load_cfg(f"{a.run}/.hydra/config.yaml", device=a.device)
    env, _ = EN.create_ogbench_env(cfg.domain, obs_type=cfg.obs_type)
    agent = make_agent(cfg, env.observation_space, int(env.action_space.shape[0]))
    env.close()
    load_checkpoint(agent, f"{a.run}/{a.ckpt}", map_location=a.device)
    agent.model.eval()

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=a.device,
        n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
        frame_stack=cfg.get("frame_stack", 1), with_index=False,
    )
    print(f"[render] buffer {len(buffer):,} rows; domain {cfg.domain}")

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ev = EV.OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=a.relabel_size, n_episodes=a.episodes,
        obs_type=cfg.obs_type, seed=a.seed, device=a.device,
        save_videos=True, video_dir=out,
    )
    with torch.no_grad():
        metrics = ev.run(step=0)
    for k in sorted(metrics):
        if k.endswith("/success") or k == "eval/success":
            print(f"[render] {k} = {metrics[k]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
