"""Render benchmark-task rollouts of a JAX checkpoint (gciql_flow / crl_flowbc)
as GIFs, using the impls' own `evaluate` (goal plumbing + rendering).

    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/render_policies_jax.py \
        --run <run dir with flags.json + params_<step>.pkl> --step 500000 \
        --agent gciql_flow --tag GCIQL --out results/videos [--episodes 1]

Writes <out>/<tag>/task<i>.gif for the domain's 5 tasks (goal frame stacked on
top of each frame, the impls convention).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "third_party" / "ogbench" / "impls",
          REPO / "tools" / "wandb_mode_shim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--agent", required=True, choices=["gciql_flow", "crl_flowbc"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/videos")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--frame-skip", type=int, default=1)
    a = ap.parse_args()

    import imageio
    import jax.numpy as jnp
    import ogbench
    from utils.evaluation import evaluate
    from utils.flax_utils import restore_agent

    if a.agent == "gciql_flow":
        from gciql_flow import GCIQLFlowAgent as AgentCls, get_config
    else:
        from crl_flowbc import CRLFlowBCAgent as AgentCls, get_config

    run_dir = Path(a.run)
    flags = json.loads((run_dir / "flags.json").read_text())
    base_env = flags["env_name"]

    cfg = dict(get_config())
    cfg.update(flags["agent"])
    cfg["encoder"] = flags["agent"].get("encoder")
    cfg["frame_stack"] = flags["agent"].get("frame_stack")

    tmp_env = ogbench.make_env_and_datasets(
        base_env.replace("-v0", "-singletask-task1-v0"), env_only=True)
    ex_obs_np, _ = tmp_env.reset()
    ex_obs = jnp.asarray(ex_obs_np, dtype=jnp.float32)[None]
    ex_act = jnp.zeros((1, *tmp_env.action_space.shape), dtype=jnp.float32)
    tmp_env.close()

    agent = AgentCls.create(int(flags.get("seed", 0)), ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), a.step)
    print(f"[render] restored {a.agent} from {run_dir.name} @ {a.step}")

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for task_id in range(1, 6):
        env = ogbench.make_env_and_datasets(
            base_env.replace("-v0", f"-singletask-task{task_id}-v0"), env_only=True)
        try:
            stats, _trajs, renders = evaluate(
                agent, env, task_id=task_id, config=cfg,
                num_eval_episodes=0, num_video_episodes=a.episodes,
                video_frame_skip=a.frame_skip)
        finally:
            env.close()
        for ep, frames in enumerate(renders):
            arr = np.stack(frames).astype(np.uint8)
            gif = out / (f"task{task_id}.gif" if a.episodes == 1
                         else f"task{task_id}_ep{ep}.gif")
            imageio.mimsave(gif, arr, format="GIF", fps=30, loop=0)
            print(f"[render] wrote {gif}  ({arr.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
