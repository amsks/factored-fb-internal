"""data_collection/collect_mw_scripted.py — collect Metaworld demonstrations with
Metaworld's built-in SCRIPTED ORACLE policies (metaworld.policies).

Unlike collect_tdmpc2_orig.py (which needs a per-task TD-MPC2 checkpoint, and we only
have a handful), the scripted policies exist for all 50 MT50 tasks, so this covers the
whole benchmark with no checkpoints. Episodes are written in exactly the same unified
schema as the TD-MPC2 collections (obs 39, act 4, achieved_goal 3), so the two are
drop-in compatible downstream.

Env is newt's MetaWorldWrapper (gymnasium, raw 39-dim obs) so the data matches what the
eval envs produce.

    MUJOCO_GL=egl /raid/tongle/rl/newt/.venv/bin/python \
        data_collection/collect_mw_scripted.py --task mw-door-open --num_episodes 1000
"""
import argparse
import os
import sys
import types

import numpy as np

NEWT_TD = "/raid/tongle/rl/newt/tdmpc2"
FB_ROOT = "/raid/tongle/rl/Factored-FB"

os.environ.setdefault("MUJOCO_GL", "egl")
# `envs` -> newt's backend package (its __init__ pulls dm_control, so shim the namespace).
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(NEWT_TD, "envs")]
    sys.modules["envs"] = _pkg
if FB_ROOT not in sys.path:
    sys.path.insert(0, FB_ROOT)

from data.unified import write_episode, write_meta, dataset_dir  # noqa: E402


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)


# env base name -> policy class, where the CamelCase rule doesn't hold
POLICY_OVERRIDES = {"peg-insert-side": "SawyerPegInsertionSideV2Policy"}


def policy_name(base):
    if base in POLICY_OVERRIDES:
        return POLICY_OVERRIDES[base]
    return "Sawyer" + "".join(p.capitalize() for p in base.split("-")) + "V2Policy"


def make_policy(base):
    import metaworld.policies as P
    name = policy_name(base)
    if not hasattr(P, name):
        raise KeyError(f"no scripted policy for {base!r} (tried {name})")
    return getattr(P, name)()


def make_mw_env(task, seed=0):
    from envs.metaworld import MetaWorldWrapper
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
    raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed, render_mode="rgb_array")
    return MetaWorldWrapper(raw, _C())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)                  # mw-door-open
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/collected")
    ap.add_argument("--num_episodes", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--success_only", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_attempts_mult", type=int, default=10,
                    help="give up after num_episodes*mult rollouts (low-yield scripted policy)")
    args = ap.parse_args()

    base = args.task.split("-", 1)[-1]
    policy = make_policy(base)
    env = make_mw_env(args.task, seed=args.seed)
    np.random.seed(args.seed)

    root = dataset_dir(args.out, f"Metaworld/{args.task}")
    buf = root / "buffer"
    buf.mkdir(parents=True, exist_ok=True)

    max_rollouts = args.num_episodes * args.max_attempts_mult
    written, rolled, n_success, T = 0, 0, 0, args.horizon
    while written < args.num_episodes and rolled < max_rollouts:
        obs, info = env.reset()
        obs_seq = [np.asarray(obs, np.float32).reshape(-1)]
        acts, rews, succs = [], [], []
        ep_success = 0.0
        for _ in range(args.horizon):
            a = np.asarray(policy.get_action(np.asarray(obs, np.float64).reshape(-1)), np.float32)
            a = np.clip(a, -1.0, 1.0).reshape(-1)
            obs, reward, term, trunc, info = env.step(a)
            obs_seq.append(np.asarray(obs, np.float32).reshape(-1))
            acts.append(a.astype(np.float32))
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
        if written % 100 == 0:
            print(f"[mw_scripted] {args.task} {written}/{args.num_episodes} "
                  f"(rolled={rolled}, yield={n_success/max(rolled,1):.2f})", flush=True)

    yield_rate = n_success / max(rolled, 1)
    write_meta(root, dict(env_family="metaworld", task=args.task,
                          checkpoint=None, policy=f"metaworld-scripted:{policy_name(base)}",
                          success_only=bool(args.success_only), num_episodes=written,
                          episode_length=int(T), obs_dim=39, act_dim=4,
                          rollout_yield=round(yield_rate, 4),
                          note="collected with Metaworld's scripted oracle policy + newt metaworld env."))
    status = "DONE" if written >= args.num_episodes else "PARTIAL"
    print(f"[mw_scripted] {status} {args.task}: {written} eps -> {root} (yield {yield_rate:.2f})",
          flush=True)
    if written < args.num_episodes:
        sys.exit(2)


if __name__ == "__main__":
    main()
