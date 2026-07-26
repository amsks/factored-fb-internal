"""data_collection/collect_play_resample.py — OGBench-style DIVERSE-GOAL play data.

Unlike collect_play.py (single task goal + action noise = diverse *states*), this chains many
*goals* in ONE long continuous episode: reach the goal, hold it briefly, then resample a new
valid goal and keep going -- no env reset between goals. That gives goal-relabeling a broad
goal distribution within a trajectory, which is the property the goal-conditioned critics
(gciql / iql_bilinear) actually need (and the thing plain action-noise can't manufacture).

Mechanism (Box2D / rlmini): the Goal factor is part of the observation, and the TD-MPC2
checkpoints for the goal-VARIED tasks were trained to reach whatever goal the obs specifies --
so moving the goal mid-episode redirects the policy (validated: ~9-15 goals chained / 600 steps).
Tasks whose training goal was FIXED (center-wall, goal-ctrl-maze) don't redirect and are skipped
by the redirection gate below. Box2D goal resampling honors maze walls and a min-distance from
the previous goal ("another, not too close").

Writes the same unified schema as collect_play.py, one LONG episode per env (terminal=0
throughout, timeout=1 only at the end, so the whole chain is a single trajectory).

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 /raid/tongle/rl/newt/.venv/bin/python \
        data_collection/collect_play_resample.py --task b2d-goal-ctrl-target --episodes 200
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_FB = "/raid/tongle/rl/Factored-FB"
sys.path.insert(0, _FB)
import data_collection.collect_play as cp  # noqa: E402  (build_cfg/make_vec_env/load_agent/resolve_checkpoint)
import torch  # noqa: E402
from envs.rlmini import RLMiniWrapper  # noqa: E402


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)


def make_b2d(task, n, render=False):
    variant = task[len("b2d-"):] if task.startswith("b2d-") else task[len("p2d-"):]
    fam = "Box2D" if task.startswith("b2d-") else "Pusher2D"
    envs = [RLMiniWrapper(fam, variant, _C(), seed=1000 + i) for i in range(n)]
    return envs


def _reset(e, seed):
    r = e.reset(seed=seed)
    return r[0] if isinstance(r, tuple) else r


def _goal_xy(core):
    return np.asarray(core.goal.attribute, dtype=float).reshape(-1)[:2].copy()


def resample_goal(e, min_dist):
    """Sample a new VALID goal (bounds via sample_goal, not in a maze wall) at least `min_dist`
    from the previous goal. Returns True if a genuinely new goal was set (False => fixed-goal
    task where sample_goal never moves)."""
    core = e._raw
    prev = _goal_xy(core)
    walls = len(getattr(core.cfg, "maze_walls", []) or []) > 0
    for _ in range(96):
        core.goal.sample_goal()
        g = _goal_xy(core)
        ok = (not core._goal_overlaps_maze_wall(g, core._goal_clearance_radius())) if walls else True
        if ok and np.linalg.norm(g - prev) > min_dist:
            return True
    return False   # sample_goal never produced a far-enough new goal -> not resamplable


def redirection_gate(task, agent, cfg, ro, ra, steps=400, hold=8, min_dist=0.8):
    """Quick check that resampling actually redirects the policy. Returns mean goals/episode.
    Must feed exactly cfg.num_envs envs -- the TD-MPC2 planner tensors are sized to it."""
    K = cfg.num_envs
    envs = make_b2d(task, K)
    obs = [_reset(e, 100 + i) for i, e in enumerate(envs)]
    done = torch.ones(K, dtype=torch.bool)
    tasks = torch.arange(K, dtype=torch.int32)
    gr = [0] * K
    hc = [0] * K
    for _ in range(steps):
        a = agent(_pad(obs, ro, K), t0=done, task=tasks, eval_mode=True)[:, :ra].detach().cpu().numpy()
        done = torch.zeros(K, dtype=torch.bool)
        for i, e in enumerate(envs):
            r = e.step(np.clip(a[i], -1, 1))
            obs[i] = r[0]
            if float(r[4].get("success", 0)) > 0.5:
                hc[i] += 1
                if hc[i] >= hold:
                    if resample_goal(e, min_dist):
                        gr[i] += 1
                    hc[i] = 0
                    done[i] = True
            else:
                hc[i] = 0
    for e in envs:
        try:
            e.close()
        except Exception:
            pass
    return float(np.mean(gr))


def _pad(obs_list, ro, K):
    a = np.zeros((K, 128), np.float32)
    for i, o in enumerate(obs_list):
        a[i, :ro] = np.asarray(o, np.float32)
    return torch.tensor(a)


def rollout(task, agent, cfg, ro, ra, episodes, horizon, hold, min_dist, noise, stuck, seed0,
            capture=0, render_size=128):
    """Long continuous goal-chaining episodes. Returns (episode dicts, frames, mean_goals).
    Always runs exactly cfg.num_envs envs (the planner is sized to it); extra episodes past the
    request are trimmed."""
    K = cfg.num_envs
    tasks = torch.arange(K, dtype=torch.int32)
    out, frames, goals = [], [], []
    n_batches = (episodes + K - 1) // K
    for b in range(n_batches):
        envs = make_b2d(task, K)
        rndr = capture and b == 0
        obs = [_reset(e, seed0 + b * K + i) for i, e in enumerate(envs)]
        O = np.empty((horizon, K, ro), np.float32)
        A = np.empty((horizon, K, ra), np.float32)
        NO = np.empty((horizon, K, ro), np.float32)
        R = np.empty((horizon, K), np.float32)
        SU = np.zeros((horizon, K), np.float32)
        SW = np.zeros((horizon, K), np.float32)   # goal-switch marker (info only)
        ff = [[] for _ in range(min(capture, K))] if rndr else None
        done = torch.ones(K, dtype=torch.bool)
        gc = [0] * K
        hc = [0] * K
        since = [0] * K   # steps on the current goal (for stuck-resample)
        for t in range(horizon):
            a = agent(_pad(obs, ro, K), t0=done, task=tasks, eval_mode=(noise <= 0))
            a = a[:, :ra].detach().cpu().numpy()
            if noise > 0:
                a = np.clip(a + noise * np.random.randn(*a.shape).astype(np.float32), -1, 1)
            done = torch.zeros(K, dtype=torch.bool)
            for i, e in enumerate(envs):
                st = np.asarray(obs[i], np.float32)
                r = e.step(np.clip(a[i], -1, 1))
                nst = np.asarray(r[0], np.float32)
                O[t, i] = st[:ro]
                A[t, i] = a[i]
                NO[t, i] = nst[:ro]
                R[t, i] = float(r[1])
                s = float(r[4].get("success", 0))
                SU[t, i] = s
                obs[i] = r[0]
                since[i] += 1
                switch = False
                if s > 0.5:
                    hc[i] += 1
                    if hc[i] >= hold:
                        switch = resample_goal(e, min_dist)
                        hc[i] = 0
                else:
                    hc[i] = 0
                if not switch and since[i] >= stuck:      # stuck on a goal too long -> move on
                    switch = resample_goal(e, min_dist)
                if switch:
                    gc[i] += 1
                    since[i] = 0
                    done[i] = True                        # t0 -> re-plan for the new goal
                    SW[t, i] = 1.0
                if rndr and i < len(ff):
                    ff[i].append(np.asarray(e.render()))
        for i in range(K):
            if len(out) >= episodes:
                break
            out.append(dict(observation=O[:, i], action=A[:, i], next_observation=NO[:, i],
                            reward=R[:, i], success=SU[:, i], goal_switch=SW[:, i]))
            goals.append(gc[i])
        if rndr:
            for fr in ff:
                if fr:
                    frames.append(np.stack(fr))
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
    return out[:episodes], frames, float(np.mean(goals)) if goals else 0.0


def write_episodes(out_root, family, task, eps):
    d = os.path.join(out_root, f"{family}__{task}", "buffer")
    os.makedirs(d, exist_ok=True)
    for j, ep in enumerate(eps):
        H = len(ep["action"])
        term = np.zeros(H, np.float32)
        timeout = np.zeros(H, np.float32)
        timeout[-1] = 1.0                                 # one long continuous trajectory
        np.savez_compressed(os.path.join(d, f"episode_{j:06d}_{H:05d}.npz"),
                            observation=ep["observation"], action=ep["action"],
                            next_observation=ep["next_observation"], reward=ep["reward"],
                            terminal=term, timeout=timeout, success=ep["success"],
                            goal_switch=ep["goal_switch"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/collected_play_resample")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=1000, help="long episode length (chains many goals)")
    ap.add_argument("--hold", type=int, default=10, help="steps held in goal before resampling (~1s)")
    ap.add_argument("--min_dist", type=float, default=0.8, help="new goal must be this far from the last")
    ap.add_argument("--noise", type=float, default=0.1, help="light action noise (0 = deterministic)")
    ap.add_argument("--stuck", type=int, default=200, help="resample if a goal isn't reached in N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_k", type=int, default=4)
    ap.add_argument("--video_dir", default="/raid/tongle/rl/fb-data/collected_play_resample/_videos")
    ap.add_argument("--force", action="store_true", help="skip the redirection gate")
    args = ap.parse_args()

    assert args.task.startswith(("b2d-", "p2d-")), \
        "play-resample currently supports Box2D/Pusher2D (goal in obs, resamplable)."
    family = "Box2D" if args.task.startswith("b2d-") else "Pusher2D"
    cfg = cp.build_cfg(args.task, args.num_envs)
    _v = cp.make_vec_env(cfg)
    _v.close()
    ro, ra = cp.real_dims(args.task, cfg)
    agent = cp.load_agent(cfg, cp.resolve_checkpoint(args.task))
    print(f"[resample] {args.task}: obs={ro} act={ra} horizon={args.horizon}", flush=True)

    mg = redirection_gate(args.task, agent, cfg, ro, ra, min_dist=args.min_dist)
    print(f"[resample] redirection gate: {mg:.1f} goals/episode", flush=True)
    if mg < 1.5 and not args.force:
        print(f"[resample] SKIP {args.task}: policy does not redirect to resampled goals "
              f"(fixed training goal). Use --force to override.", flush=True)
        return

    eps, frames, meang = rollout(args.task, agent, cfg, ro, ra, args.episodes, args.horizon,
                                 args.hold, args.min_dist, args.noise, args.stuck, args.seed * 7000,
                                 capture=args.video_k)
    d = write_episodes(args.out, family, args.task, eps)
    print(f"[resample] wrote {len(eps)} long episodes -> {d}  "
          f"(mean {meang:.1f} goals/episode, {len(eps)*args.horizon:,} transitions)", flush=True)

    if frames:
        spec = importlib.util.spec_from_file_location("_v", os.path.join(_FB, "env", "video.py"))
        vmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vmod)
        os.makedirs(args.video_dir, exist_ok=True)
        p = vmod.save_video_grid(frames, os.path.join(args.video_dir, f"{args.task}_resample.mp4"),
                                 max_episodes=args.video_k)
        print(f"[resample] wrote video -> {p}", flush=True)

    meta = dict(task=args.task, family=family, policy="tdmpc2-goal-resample",
                episodes=len(eps), horizon=args.horizon, obs_dim=int(ro), action_dim=int(ra),
                mean_goals_per_episode=meang, hold=args.hold, min_dist=args.min_dist,
                noise=args.noise, stuck=args.stuck, checkpoint=cp.resolve_checkpoint(args.task))
    with open(os.path.join(args.out, f"{family}__{args.task}", "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[resample] DONE {args.task}", flush=True)


if __name__ == "__main__":
    main()
