"""Collect offline episodes from trained CRL checkpoints (no training).

Loads one or more checkpoints, rolls each policy out for --episodes-per-checkpoint
episodes, and writes all episodes to --output as .npz files for offline training.

Usage
-----
    # From a checkpoint directory (all checkpoints, sorted by step):
    python scripts/data/run_crl_iws_collect.py \\
        --checkpoint-dir runs/crl-iws-air-hockey/checkpoints \\
        --episodes-per-checkpoint 100 \\
        --output datasets/crl-iws-air-hockey/buffer

    # From specific checkpoint files:
    python scripts/data/run_crl_iws_collect.py \\
        --checkpoints runs/.../checkpoint_00500000.pt \\
        --episodes-per-checkpoint 500 \\
        --output datasets/crl-iws-air-hockey/buffer

    # Then train offline:
    python train.py domain=crl_iws_air_hockey data_path=datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from agents.crl_iws.agent_torch import CRLIWSAgentTorch


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _make_env(env_name: str, seed: int):
    try:
        import rlmini_envs
        return rlmini_envs.make(
            env_name, seed=seed,
            gym_to_gymnasium=True, select_obs=True,
            obs_flat=True, obs_include_metadata=False,
        )
    except Exception:
        import gymnasium as gym
        return gym.make(env_name)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_agent(ckpt_path: Path, device: torch.device) -> CRLIWSAgentTorch:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = CRLIWSAgentTorch(
        state_dim=int(payload["state_dim"]),
        goal_dim=int(payload["goal_dim"]),
        action_dim=int(payload["action_dim"]),
        action_low=np.array(payload["action_low"], dtype=np.float32),
        action_high=np.array(payload["action_high"], dtype=np.float32),
        device=device,
        hidden_dims=tuple(int(d) for d in payload["hidden_dims"]),
        repr_dim=int(payload["repr_dim"]),
    )
    agent.load_state_dict(payload["agent"])
    return agent


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
    elif args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        paths = sorted(d.glob("checkpoint_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No checkpoint_*.pt files found in {d}")
    else:
        raise ValueError("Provide --checkpoints or --checkpoint-dir")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return paths


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------


def _collect_episodes(
    agent: CRLIWSAgentTorch,
    env_name: str,
    num_episodes: int,
    seed: int,
    *,
    deterministic: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Roll out agent for num_episodes complete episodes; return list of episode dicts."""
    env = _make_env(env_name, seed)
    episodes = []
    ep_seed = seed

    while len(episodes) < num_episodes:
        obs, _ = env.reset(seed=ep_seed)
        ep_seed += 1
        obs = np.asarray(obs, dtype=np.float32)

        obs_list, act_list = [obs.copy()], []
        done = False

        while not done:
            goal = obs  # self-goal at collection time; agent uses random goal internally
            action = agent.sample_actions(obs[None], goal[None])[0]
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = bool(terminated or truncated)

            act_list.append(action.astype(np.float32))
            obs_list.append(next_obs.copy())
            obs = next_obs

        obs_arr = np.stack(obs_list, axis=0)           # (T+1, S)
        act_arr = np.stack(act_list, axis=0)            # (T,   A)
        T = act_arr.shape[0]

        pad_action = np.zeros_like(act_arr[-1])[None]
        action_arr = np.concatenate([act_arr, pad_action], axis=0)  # (T+1, A)

        episodes.append({
            "observation": obs_arr,
            "physics": obs_arr.copy(),
            "action": action_arr,
            "reward": np.zeros((T + 1, 1), dtype=np.float32),
            "discount": np.ones((T + 1, 1), dtype=np.float32),
            "terminated": np.array([[False] * T + [True]], dtype=bool).reshape(T + 1, 1),
            "achieved_goal": obs_arr.copy(),
        })

    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_collect(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _resolve_checkpoints(args)
    print(f"[crl-iws-collect] {len(checkpoints)} checkpoint(s), "
          f"{args.episodes_per_checkpoint} episodes each → {out_dir}")

    ep_idx = 0
    written: list[dict] = []

    for ckpt_path in checkpoints:
        print(f"[crl-iws-collect] Loading {ckpt_path.name}")
        agent = _load_agent(ckpt_path, device)

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_steps = int(payload.get("total_steps", 0))

        episodes = _collect_episodes(
            agent, args.env, args.episodes_per_checkpoint,
            seed=args.seed + ep_idx,
        )

        for ep in episodes:
            T = ep["action"].shape[0] - 1
            path = out_dir / f"episode_{ep_idx:06d}_{T}.npz"
            np.savez_compressed(path, **ep)
            written.append({"path": str(path.resolve()), "checkpoint_steps": ckpt_steps})
            ep_idx += 1

        print(f"[crl-iws-collect]   wrote {len(episodes)} episodes (total so far: {ep_idx})")

    manifest = {
        "source": "crl_iws_collect",
        "env": args.env,
        "episodes_per_checkpoint": args.episodes_per_checkpoint,
        "num_episodes": ep_idx,
        "episodes": written,
    }
    (out_dir.parent / "collect_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[crl-iws-collect] Done. {ep_idx} total episodes → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)

    ckpt_group = ap.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint-dir", default=None,
                            help="Directory of checkpoint_*.pt files (all are used, sorted by step)")
    ckpt_group.add_argument("--checkpoints", nargs="+", default=None,
                            help="Explicit checkpoint file(s)")

    ap.add_argument("--env", default="AirHockeyPuckGoalPosition-v0")
    ap.add_argument("--episodes-per-checkpoint", type=int, default=100)
    ap.add_argument("--output", default="datasets/crl-iws-air-hockey/buffer")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")

    return ap.parse_args()


if __name__ == "__main__":
    run_collect(_parse_args())
