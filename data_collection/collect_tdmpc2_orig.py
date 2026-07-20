"""data_collection/collect_tdmpc2_orig.py — collect Metaworld push / pick-place with the
ORIGINAL single-task TD-MPC2 checkpoints (nicklashansen/tdmpc2, `_Qs.params` format).

The original tdmpc2 env uses old `gym`; ours uses gymnasium. So we combine:
  * env    = newt's Metaworld backend (gymnasium, obs 39 — matches our datasets + eval)
  * model  = tdmpc2-orig's WorldModel + TDMPC2 planner (loads the single-task checkpoint)
single-task (task_dim=0), default model_size (latent 512 / mlp 512, matches the checkpoint).

Run in newt's venv (has torch + metaworld + tensordict/hydra):
    MUJOCO_GL=egl /raid/tongle/rl/newt/.venv/bin/python \
        data_collection/collect_tdmpc2_orig.py --task mw-push --checkpoint <mw-push-1.pt>
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
from omegaconf import OmegaConf

TDMPC2_ORIG = "/raid/tongle/rl/tdmpc2-orig/tdmpc2"
NEWT_TD = "/raid/tongle/rl/newt/tdmpc2"
FB_ROOT = "/raid/tongle/rl/Factored-FB"

os.environ.setdefault("MUJOCO_GL", "egl")
# `envs` -> newt's backend package (for the metaworld env); common/tdmpc2/config -> tdmpc2-orig.
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(NEWT_TD, "envs")]
    sys.modules["envs"] = _pkg
for p in (TDMPC2_ORIG, FB_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.parser import parse_cfg               # tdmpc2-orig
from tdmpc2 import TDMPC2                          # tdmpc2-orig
from data.unified import write_episode, write_meta, dataset_dir  # Factored-FB

import hydra.utils as _hu                          # parse_cfg calls get_original_cwd(); we run
_hu.get_original_cwd = lambda: os.getcwd()         # standalone (not under @hydra.main).


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)


def make_mw_env(task, seed=0):
    from envs.metaworld import MetaWorldWrapper
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
    raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed, render_mode="rgb_array")
    return MetaWorldWrapper(raw, _C())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)               # mw-push / mw-pick-place
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/collected")
    ap.add_argument("--num_episodes", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--success_only", type=int, default=1)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(TDMPC2_ORIG, "config.yaml"))
    cfg.task, cfg.checkpoint, cfg.obs = args.task, args.checkpoint, "state"
    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": [39]}
    cfg.action_dim = 4
    cfg.episode_length = args.horizon              # normally set by make_env
    agent = TDMPC2(cfg)
    agent.load(args.checkpoint)

    env = make_mw_env(args.task)
    root = dataset_dir(args.out, f"Metaworld/{args.task}")
    buf = root / "buffer"
    buf.mkdir(parents=True, exist_ok=True)

    written, rolled, n_success = 0, 0, 0
    while written < args.num_episodes:
        obs, info = env.reset()
        obs_seq = [np.asarray(obs, np.float32).reshape(-1)]
        acts, rews, succs = [], [], []
        ep_success = 0.0
        for t in range(args.horizon):
            ot = torch.as_tensor(np.asarray(obs, np.float32).reshape(-1))
            action = agent.act(ot, t0=(t == 0), eval_mode=True, task=None).cpu().numpy().reshape(-1)
            obs, reward, term, trunc, info = env.step(action)
            obs_seq.append(np.asarray(obs, np.float32).reshape(-1))
            acts.append(action.astype(np.float32))
            rews.append(float(reward))
            s = float(info.get("success", 0.0))
            succs.append(s)
            ep_success = max(ep_success, s)
            if term or trunc:
                break
        rolled += 1
        n_success += int(ep_success >= 1.0)
        if args.success_only and ep_success < 1.0:
            continue
        oa = np.stack(obs_seq)
        T = len(acts)
        traj = dict(observation=oa[:-1], action=np.stack(acts), next_observation=oa[1:],
                    achieved_goal=np.zeros((T, 3), np.float32), reward=np.array(rews, np.float32),
                    terminal=np.zeros(T, bool), timeout=np.zeros(T, bool),
                    success=np.array(succs) > 0.5)
        write_episode(buf, written, traj)
        written += 1
        if written % 50 == 0:
            print(f"[collect_orig] {args.task} written={written}/{args.num_episodes} "
                  f"(rolled={rolled}, yield={n_success/max(rolled,1):.2f})", flush=True)

    write_meta(root, dict(env_family="metaworld", task=args.task, checkpoint=args.checkpoint,
                          policy="tdmpc2-orig-expert", success_only=bool(args.success_only),
                          num_episodes=written, episode_length=int(T), obs_dim=39, act_dim=4,
                          note="collected with original single-task TD-MPC2 + newt metaworld env."))
    print(f"[collect_orig] DONE {args.task}: {written} eps -> {root} (yield {n_success/max(rolled,1):.2f})")


if __name__ == "__main__":
    main()
