"""campaign/eval/eval_jax.py — evaluate a trained JAX baseline (gciql / crl / crl_flowbc)
on a testbed task with FIXED-SEED episodes, reporting success rate (+ optional video grid).

Runs in **.venv-jax** (the exact jax that trained the agents -> no param-version risk),
with rlmini/metaworld/mani_skill added for the env rollout. Mirrors eval_fb but for the
goal-conditioned JAX agents: the goal is a high-reward offline observation (OGBench's
_goal_observation recipe), and actions come from agent.sample_actions(obs, goals, temp=0).

    JAX_PLATFORMS=cpu PYTHONPATH=<impls>:<shim> .venv-jax/bin/python campaign/eval/eval_jax.py \
        --save_dir <jax-run-dir> --params params_500000.pkl \
        --agent_file agents/gciql.py --task b2d-goal-ctrl \
        --data_dir <collected>/Box2D__b2d-goal-ctrl \
        --n_episodes 100 --seed_base 0 --n_video 4 --out <eval-out>
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

FB_ROOT = "/raid/tongle/rl/Factored-FB"
IMPLS = f"{FB_ROOT}/ogbench/impls"
SHIM = f"{FB_ROOT}/tools/wandb_mode_shim"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval_common
for p in (IMPLS, SHIM):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_common import make_eval_env, seeded_reset, save_video_grid, load_offline  # noqa: E402
import jax  # noqa: E402
import flax  # noqa: E402
import agents  # noqa: E402


def load_agent_config(agent_file):
    path = agent_file if os.path.isabs(agent_file) else os.path.join(IMPLS, agent_file)
    if not os.path.exists(path):
        path = os.path.join(FB_ROOT, agent_file)
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def build_agent(agent_file, params_pkl, obs_dim, act_dim):
    cfg = load_agent_config(agent_file)              # arch = get_config defaults (matches training)
    ex_obs = np.zeros((1, obs_dim), np.float32)
    ex_act = np.zeros((1, act_dim), np.float32)
    agent = agents.agents[cfg["agent_name"]].create(0, ex_obs, ex_act, cfg)
    with open(params_pkl, "rb") as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
    return agent, cfg


def compute_goal(data_dir):
    """GCRL goal = the highest-reward offline next_obs (a canonical success state)."""
    nobs, rew = load_offline(data_dir)
    return nobs[int(np.argmax(rew))].astype(np.float32)


def rollout(agent, env, task, cond, horizon, n_episodes, seed_base, n_video):
    """cond is the conditioning kwargs for sample_actions: {'goals': g} (GCRL) or {'z': z} (FB)."""
    successes, frames = [], []
    for i in range(n_episodes):
        obs, info = seeded_reset(env, task, seed_base + i)
        key = jax.random.PRNGKey(seed_base + i)
        ep_success = bool(info.get("success", 0.0))
        rec = i < n_video
        ep_frames = [env.render()] if rec else None
        for _ in range(horizon):
            key, sub = jax.random.split(key)
            a = np.asarray(agent.sample_actions(observations=np.asarray(obs, np.float32),
                                                seed=sub, temperature=0, **cond))
            obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
            ep_success = ep_success or bool(info.get("success", 0.0))
            if rec:
                ep_frames.append(env.render())
            if term or trunc:
                break
        successes.append(float(ep_success))
        if rec:
            frames.append(ep_frames)
    return np.array(successes, dtype=np.float32), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--params", default="params_500000.pkl")
    ap.add_argument("--agent_file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--n_video", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env, horizon = make_eval_env(args.task, render=args.n_video > 0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent, cfg = build_agent(args.agent_file, os.path.join(args.save_dir, args.params), obs_dim, act_dim)
    if cfg["agent_name"] in ("fb", "fb_flowbc"):
        nobs, rew = load_offline(args.data_dir)                    # FB: z = infer from offline reward
        if len(nobs) > 10000:
            idx = np.random.default_rng(0).choice(len(nobs), 10000, replace=False)
            nobs, rew = nobs[idx], rew[idx]
        cond = {"z": np.asarray(agent.infer_z(nobs, rew))}
    else:
        cond = {"goals": compute_goal(args.data_dir)}             # GCRL: high-reward goal obs

    successes, frames = rollout(agent, env, args.task, cond, horizon,
                                args.n_episodes, args.seed_base, args.n_video)
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_path = None
    if args.n_video > 0 and frames:
        video_path = str(out / "eval_grid.mp4")
        save_video_grid(frames, video_path)

    metrics = {
        "task": args.task, "agent": cfg["agent_name"], "params": args.params,
        "n_episodes": args.n_episodes, "seed_base": args.seed_base,
        "success_rate": float(successes.mean()),
        "success_per_episode": successes.tolist(),
        "video": video_path,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval_jax] {args.task} {cfg['agent_name']} {args.params}: "
          f"success_rate={metrics['success_rate']:.3f} over {args.n_episodes} eps"
          f"{' + video' if video_path else ''}")


if __name__ == "__main__":
    main()
