"""Generate small random-action offline datasets from rlmini_envs.

Example:
    python scripts/data/export_rlmini_random.py \
        --domain rlmini-box2d-goal-target \
        --env Box2D --variant goal-ctrl-target \
        --output datasets --episodes 3 --steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.rlmini import create_rlmini_env


def _json_arg(value: str | None, default):
    if value is None:
        return default
    return json.loads(value)


def _make_cfg(args):
    return {
        "rlmini_env": args.env,
        "rlmini_variant": args.variant,
        "rlmini_horizon": args.horizon,
        "rlmini_render": args.render,
        "rlmini_obs_keys": _json_arg(args.obs_keys, None),
        "rlmini_obs_slices": _json_arg(args.obs_slices, None),
        "rlmini_extra": _json_arg(args.extra, {}),
    }


def collect_episode(env, max_steps: int):
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    observations = [obs]
    actions = []
    rewards = [0.0]
    discounts = [1.0]
    terminated = [False]

    for _ in range(max_steps):
        action = np.asarray(env.action_space.sample(), dtype=np.float32)
        next_obs, reward, term, trunc, _info = env.step(action)
        done = bool(term or trunc)

        actions.append(action)
        observations.append(np.asarray(next_obs, dtype=np.float32))
        rewards.append(float(reward))
        discounts.append(0.0 if done else 1.0)
        terminated.append(done)
        if done:
            break

    # Episode arrays use T observations and T actions. The final action is padding
    # because there is no transition starting from the terminal/final observation.
    if actions:
        pad_action = np.zeros_like(actions[-1], dtype=np.float32)
    else:
        pad_action = np.zeros(env.action_space.shape, dtype=np.float32)
    action_arr = np.stack(actions + [pad_action], axis=0).astype(np.float32)
    obs_arr = np.stack(observations, axis=0).astype(np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    discount_arr = np.asarray(discounts, dtype=np.float32).reshape(-1, 1)
    terminated_arr = np.asarray(terminated, dtype=bool).reshape(-1, 1)

    return {
        "observation": obs_arr,
        "physics": obs_arr.copy(),
        "action": action_arr,
        "reward": reward_arr,
        "discount": discount_arr,
        "terminated": terminated_arr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, help="Output domain folder name under --output")
    ap.add_argument("--env", required=True, help="rlmini env name, e.g. Box2D")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--output", default="datasets")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--obs-keys", default=None, help="JSON list of factored keys to select")
    ap.add_argument("--obs-slices", default=None, help="JSON mapping of key -> indices")
    ap.add_argument("--extra", default=None, help="JSON object forwarded to rlmini_envs.make")
    args = ap.parse_args()

    out_dir = Path(args.output) / args.domain / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(args)
    env, _ = create_rlmini_env(args.domain, cfg=cfg, seed=args.seed)
    try:
        for ep_idx in range(args.episodes):
            env.reset(seed=args.seed + ep_idx)
            ep = collect_episode(env, args.steps)
            length = ep["observation"].shape[0] - 1
            path = out_dir / f"episode_{ep_idx:06d}_{length}.npz"
            np.savez_compressed(path, **ep)
            print(f"[rlmini] wrote {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
