"""campaign/eval/eval_fb.py — evaluate a trained native FB / FB+FlowBC checkpoint on a
testbed task with FIXED-SEED episodes, reporting success rate (+ optional video grid).

Runs in **newt's venv** (torch + rlmini/metaworld/mani_skill + video). It:
  1. rebuilds the agent from the run's saved .hydra/config.yaml and loads the checkpoint,
  2. infers the FB task latent z = reward_inference(offline next_obs, offline reward)
     from the unified dataset (the zero-shot FB recipe; same for every FB checkpoint),
  3. rolls out N episodes each seeded with a FIXED seed (seed_base + i) so the exact
     initial states are identical across checkpoints / algorithms (fair comparison),
  4. reports success rate (mean over episodes) and writes metrics.json,
  5. optionally renders the first `n_video` episodes into one 2x2 grid mp4.

    /raid/tongle/rl/newt/.venv/bin/python campaign/eval/eval_fb.py \
        --run_dir <native-run-dir> --ckpt step_500000.pt \
        --task b2d-goal-ctrl --data_dir <collected>/Box2D__b2d-goal-ctrl \
        --n_episodes 100 --seed_base 0 --n_video 4 --out <eval-out-dir>
"""
import argparse
import glob
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

FB_ROOT = "/raid/tongle/rl/Factored-FB"
NEWT_TD = "/raid/tongle/rl/newt/tdmpc2"

# `envs` must resolve to newt's backend package (for envs.rlmini) — register a stub
# BEFORE importing it, so it never clashes with Factored-FB's own envs/ package.
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(NEWT_TD, "envs")]
    sys.modules["envs"] = _pkg
for p in (NEWT_TD, FB_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


# ----------------------------------------------------------------------- agent
def build_fb_agent(run_dir, ckpt_path, obs_dim, act_dim, device):
    from nn_models import (ForwardArchiConfig, BackwardArchiConfig, SimpleActorArchiConfig,
                           NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig)
    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    obs_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
    shared = dict(
        obs_space=obs_space, action_dim=act_dim, batch_size=cfg.batch_size,
        z_dim=cfg.z_dim, L_dim=cfg.L_dim, actor_encode_obs=cfg.actor_encode_obs,
        forward_cfg=ForwardArchiConfig(hidden_dim=cfg.forward.hidden_dim, hidden_layers=cfg.forward.hidden_layers,
                                       embedding_layers=cfg.forward.embedding_layers, num_parallel=cfg.forward.num_parallel),
        backward_cfg=BackwardArchiConfig(hidden_dim=cfg.backward.hidden_dim, hidden_layers=cfg.backward.hidden_layers, norm=cfg.backward.norm),
        left_encoder_cfg=BackwardArchiConfig(hidden_dim=cfg.left_encoder.hidden_dim, hidden_layers=cfg.left_encoder.hidden_layers, norm=cfg.left_encoder.norm),
        discount=cfg.discount, lr_f=cfg.lr_f, lr_b=cfg.lr_b, lr_actor=cfg.lr_actor,
        weight_decay=cfg.weight_decay, clip_grad_norm=cfg.clip_grad_norm, ortho_coef=cfg.ortho_coef,
        train_goal_ratio=cfg.train_goal_ratio, actor_std=cfg.actor_std, stddev_clip=cfg.stddev_clip,
        f_target_tau=cfg.f_target_tau, b_target_tau=cfg.b_target_tau, bc_coeff=cfg.get("bc_coeff", 0.0),
        goal_cond=cfg.get("goal_cond", False), device=device,
    )
    if cfg.agent == "fb_flowbc":
        from agents.fb.flow_bc.agent import FBFlowBCAgent
        agent = FBFlowBCAgent(
            actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=cfg.actor.hidden_dim, hidden_layers=cfg.actor.hidden_layers, embedding_layers=cfg.actor.embedding_layers),
            actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=cfg.actor_vf.hidden_dim, hidden_layers=cfg.actor_vf.hidden_layers),
            flow_steps=cfg.flow_steps, lr_actor_vf=cfg.lr_actor_vf, **shared)
    else:
        from agents.fb.agent import FBAgent
        agent = FBAgent(actor_cfg=SimpleActorArchiConfig(hidden_dim=cfg.actor.hidden_dim, hidden_layers=cfg.actor.hidden_layers, embedding_layers=cfg.actor.embedding_layers), **shared)
    agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    return agent, cfg


def infer_z(agent, data_dir, n, device):
    """z = reward_inference(offline next_obs, offline reward) over an n-sample subset."""
    eps = sorted(glob.glob(f"{data_dir}/buffer/episode_*.npz"),
                 key=lambda p: int(re.findall(r"\d+", Path(p).stem)[-1]))
    nobs, rew = [], []
    for ep in eps:
        z = np.load(ep)
        nobs.append(z["next_observation"].astype(np.float32))
        rew.append(z["reward"].astype(np.float32).reshape(-1))
    nobs = np.concatenate(nobs)
    rew = np.concatenate(rew)
    if len(nobs) > n:
        idx = np.random.default_rng(0).choice(len(nobs), n, replace=False)
        nobs, rew = nobs[idx], rew[idx]
    nobs_t = torch.as_tensor(nobs, dtype=torch.float32, device=device)
    rew_t = torch.as_tensor(rew, dtype=torch.float32, device=device).reshape(-1, 1)
    return agent.model.reward_inference(next_obs=nobs_t, reward=rew_t)


# ------------------------------------------------------------------------- env
class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def make_eval_env(task, render=False):
    """Build the raw-obs eval env (obs identical to the training dataset)."""
    if task.startswith(("b2d-", "p2d-")):
        from envs.rlmini import RLMiniWrapper, _PREFIX_TO_ENV, episode_length_for
        prefix = "b2d-" if task.startswith("b2d-") else "p2d-"
        env_name = _PREFIX_TO_ENV[prefix]
        variant = task[len(prefix):]
        env = RLMiniWrapper(env_name, variant, _Cfg(), seed=0)
        if render:  # swap in a render-enabled raw env (obs/keys already frozen, unchanged)
            import rlmini_envs
            env._raw = rlmini_envs.make(env_name, variant=variant, horizon=1_000_000_000, seed=0, render=True)
            if hasattr(env._raw, "max_steps"):
                env._raw.max_steps = 1_000_000_000
        return env, episode_length_for(task)
    raise ValueError(f"eval_fb env builder not yet wired for task {task!r}")


def rollout_fixed_seeds(agent, env, z, horizon, n_episodes, seed_base, n_video, device):
    successes, frames = [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + i)
        if hasattr(agent, "reset"):
            agent.reset()
        ep_success = bool(info.get("success", 0.0))
        rec = i < n_video
        ep_frames = [env.render()] if rec else None
        for _ in range(horizon):
            a = agent.act(obs=torch.as_tensor(obs, dtype=torch.float32, device=device)[None], z=z)
            obs, _r, term, trunc, info = env.step(a.cpu().numpy()[0])
            ep_success = ep_success or bool(info.get("success", 0.0))
            if rec:
                ep_frames.append(env.render())
            if term or trunc:
                break
        successes.append(float(ep_success))
        if rec:
            frames.append(ep_frames)
    return np.array(successes, dtype=np.float32), frames


def save_video_grid(frames, out_path, fps=30):
    """Tile up to 4 episodes into a single 2x2 grid mp4 (pad shorter episodes)."""
    import imageio
    frames = [f for f in frames if f]
    if not frames:
        return None
    T = max(len(f) for f in frames)
    tiles = []
    for f in frames[:4]:
        arr = np.stack(f)                                  # [t,H,W,3]
        if len(arr) < T:
            arr = np.concatenate([arr, np.repeat(arr[-1:], T - len(arr), 0)])
        tiles.append(arr)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    H, W = tiles[0].shape[1:3]
    tiles = [t[:, :H, :W] for t in tiles]
    top = np.concatenate([tiles[0], tiles[1]], axis=2)
    bot = np.concatenate([tiles[2], tiles[3]], axis=2)
    grid = np.concatenate([top, bot], axis=1)              # [T,2H,2W,3]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, grid.astype(np.uint8), fps=fps, macro_block_size=1)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--n_video", type=int, default=4)
    ap.add_argument("--relabel_n", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt_path = os.path.join(args.run_dir, "checkpoints", args.ckpt)
    env, horizon = make_eval_env(args.task, render=args.n_video > 0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent, cfg = build_fb_agent(args.run_dir, ckpt_path, obs_dim, act_dim, args.device)
    z = infer_z(agent, args.data_dir, args.relabel_n, args.device)

    successes, frames = rollout_fixed_seeds(agent, env, z, horizon, args.n_episodes,
                                            args.seed_base, args.n_video, args.device)
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_path = None
    if args.n_video > 0 and frames:
        video_path = str(out / "eval_grid.mp4")
        save_video_grid(frames, video_path)

    metrics = {
        "task": args.task, "agent": str(cfg.agent), "ckpt": args.ckpt,
        "n_episodes": args.n_episodes, "seed_base": args.seed_base,
        "success_rate": float(successes.mean()),
        "success_per_episode": successes.tolist(),
        "video": video_path,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval_fb] {args.task} {cfg.agent} {args.ckpt}: "
          f"success_rate={metrics['success_rate']:.3f} over {args.n_episodes} eps"
          f"{' + video' if video_path else ''}")


if __name__ == "__main__":
    main()
