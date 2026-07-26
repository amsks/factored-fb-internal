"""data_collection/collect_play_resample_ms.py — ManiSkill diverse-GOAL play (goal chaining).

Same idea as the Box2D collect_play_resample.py, adapted to ManiSkill: the goal is a SAPIEN
actor (push-cube: `goal_region` on the table; pick-cube: `goal_site` floating in 3D). Moving
that actor mid-episode redirects the goal-conditioned TD-MPC2 checkpoint (validated: push-cube
cube follows a moved goal). Reach is the env's own info['success']; on hold (or if stuck) we
move the goal to a new valid pose (not too close) and continue -- one long continuous episode,
many goals, no reset.

newt's ManiSkill wrapper forces num_envs=1, so we run K single-env instances and batch their
obs for the (num_envs=K) planner, exactly like the Box2D collector does with rlmini envs.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 /raid/tongle/rl/newt/.venv/bin/python \
        data_collection/collect_play_resample_ms.py --task ms-push-cube --episodes 120
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_FB = "/raid/tongle/rl/Factored-FB"
sys.path.insert(0, _FB)
import data_collection.collect_play as cp  # noqa: E402
import torch  # noqa: E402

# goal actor / manipulated-object attr + sampling bounds (workspace-relative) per task
MS = {
    "ms-push-cube": dict(obj="obj", goal="goal_region", z=0.02,
                         xr=(-0.10, 0.10), yr=(-0.20, 0.20), zr=None, min_dist=0.10),
    "ms-pick-cube": dict(obj="cube", goal="goal_site", z=None,
                         xr=(-0.10, 0.10), yr=(-0.15, 0.15), zr=(0.08, 0.32), min_dist=0.10),
}


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)


def make_ms(task, n, render=False):
    from envs.maniskill import make_env
    return [make_env(_C(task=task, seed=2000 + i, obs="state", render_size=128, num_envs=1))
            for i in range(n)]


def _stt(o):
    s = (o["state"] if hasattr(o, "keys") else o)
    s = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
    return s.reshape(-1)


def _pad(obs_list, ro, K):
    a = np.zeros((K, 128), np.float32)
    for i, o in enumerate(obs_list):
        a[i, :ro] = _stt(o)[:ro]
    return torch.tensor(a)


def _succ(info):
    s = info.get("success", 0)
    if torch.is_tensor(s):
        return float(s.float().mean())
    return float(np.mean(s)) if hasattr(s, "__len__") else float(s)


def move_goal(env, spec, min_dist):
    """Move the goal actor to a new valid pose at least min_dist from the current one."""
    u = env.unwrapped
    g = getattr(u, spec["goal"])
    PoseCls = type(g.pose)
    prev = g.pose.p.clone()
    dev = prev.device
    p = prev.clone()
    for _ in range(64):
        p = prev.clone()
        p[:, 0] = torch.empty(p.shape[0], device=dev).uniform_(*spec["xr"])
        p[:, 1] = torch.empty(p.shape[0], device=dev).uniform_(*spec["yr"])
        if spec["zr"] is not None:
            p[:, 2] = torch.empty(p.shape[0], device=dev).uniform_(*spec["zr"])
        elif spec["z"] is not None:
            p[:, 2] = spec["z"]
        if float(torch.linalg.norm((p - prev)[..., :2])) > min_dist:
            break
    g.set_pose(PoseCls.create_from_pq(p=p, q=g.pose.q))


def _reset(e, seed):
    try:
        r = e.reset(seed=seed)
    except TypeError:
        r = e.reset()
    return r[0] if isinstance(r, tuple) else r


def rollout(task, agent, cfg, ro, ra, episodes, horizon, hold, stuck, noise, seed0, capture=0):
    spec = MS[task]
    K = cfg.num_envs
    tasks = torch.arange(K, dtype=torch.int32)
    out, frames, goals = [], [], []
    n_batches = (episodes + K - 1) // K
    for b in range(n_batches):
        envs = make_ms(task, K)
        obs = [_reset(e, seed0 + b * K + i) for i, e in enumerate(envs)]
        O = np.empty((horizon, K, ro), np.float32)
        A = np.empty((horizon, K, ra), np.float32)
        NO = np.empty((horizon, K, ro), np.float32)
        R = np.empty((horizon, K), np.float32)
        SU = np.zeros((horizon, K), np.float32)
        SW = np.zeros((horizon, K), np.float32)
        rndr = capture and b == 0
        ff = [[] for _ in range(min(capture, K))] if rndr else None
        done = torch.ones(K, dtype=torch.bool)
        gc = [0] * K
        hc = [0] * K
        since = [0] * K
        for t in range(horizon):
            a = agent(_pad(obs, ro, K), t0=done, task=tasks, eval_mode=(noise <= 0))
            a = a[:, :ra].detach().cpu().numpy()
            if noise > 0:
                a = np.clip(a + noise * np.random.randn(*a.shape).astype(np.float32), -1, 1)
            done = torch.zeros(K, dtype=torch.bool)
            for i, e in enumerate(envs):
                st = _stt(obs[i])
                r = e.step(a[i].reshape(-1))
                nst = _stt(r[0])
                O[t, i] = st[:ro]
                A[t, i] = a[i]
                NO[t, i] = nst[:ro]
                R[t, i] = float(r[1].mean()) if torch.is_tensor(r[1]) else float(r[1])
                s = _succ(r[4])
                SU[t, i] = s
                obs[i] = r[0]
                since[i] += 1
                switch = False
                if s > 0.5:
                    hc[i] += 1
                    if hc[i] >= hold:
                        move_goal(e, spec, spec["min_dist"]); switch = True; hc[i] = 0
                else:
                    hc[i] = 0
                if not switch and since[i] >= stuck:
                    move_goal(e, spec, spec["min_dist"]); switch = True
                if switch:
                    gc[i] += 1; since[i] = 0; done[i] = True; SW[t, i] = 1.0
                if rndr and i < len(ff):
                    fr = e.render()
                    ff[i].append(np.asarray(fr.detach().cpu().numpy() if torch.is_tensor(fr) else fr).squeeze())
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


def redirection_gate(task, agent, cfg, ro, ra, steps=120, hold=3, stuck=40):
    eps, _, mg = rollout(task, agent, cfg, ro, ra, cfg.num_envs, steps, hold, stuck, 0.0, 999)
    return mg


def write_episodes(out_root, task, eps):
    d = os.path.join(out_root, f"ManiSkill__{task}", "buffer")
    os.makedirs(d, exist_ok=True)
    for j, ep in enumerate(eps):
        H = len(ep["action"])
        term = np.zeros(H, np.float32)
        timeout = np.zeros(H, np.float32)
        timeout[-1] = 1.0
        np.savez_compressed(os.path.join(d, f"episode_{j:06d}_{H:05d}.npz"),
                            observation=ep["observation"], action=ep["action"],
                            next_observation=ep["next_observation"], reward=ep["reward"],
                            terminal=term, timeout=timeout, success=ep["success"],
                            goal_switch=ep["goal_switch"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(MS))
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/collected_play_resample")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--stuck", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_k", type=int, default=4)
    ap.add_argument("--video_dir", default="/raid/tongle/rl/fb-data/collected_play_resample/_videos")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = cp.build_cfg(args.task, args.num_envs)
    cp.make_vec_env(cfg).close()
    ro, ra = cp.real_dims(args.task, cfg)
    agent = cp.load_agent(cfg, cp.resolve_checkpoint(args.task))
    print(f"[ms-resample] {args.task}: obs={ro} act={ra} horizon={args.horizon} num_envs={args.num_envs}",
          flush=True)

    mg = redirection_gate(args.task, agent, cfg, ro, ra)
    print(f"[ms-resample] redirection gate: {mg:.1f} goals/episode", flush=True)
    if mg < 1.2 and not args.force:
        print(f"[ms-resample] SKIP {args.task}: policy does not redirect. Use --force to override.",
              flush=True)
        return

    eps, frames, meang = rollout(args.task, agent, cfg, ro, ra, args.episodes, args.horizon,
                                 args.hold, args.stuck, args.noise, args.seed * 5000,
                                 capture=args.video_k)
    d = write_episodes(args.out, args.task, eps)
    print(f"[ms-resample] wrote {len(eps)} episodes -> {d}  (mean {meang:.1f} goals/ep, "
          f"{len(eps)*args.horizon:,} transitions)", flush=True)

    if frames:
        spec = importlib.util.spec_from_file_location("_v", os.path.join(_FB, "env", "video.py"))
        vmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vmod)
        os.makedirs(args.video_dir, exist_ok=True)
        p = vmod.save_video_grid(frames, os.path.join(args.video_dir, f"{args.task}_resample.mp4"),
                                 max_episodes=args.video_k)
        print(f"[ms-resample] wrote video -> {p}", flush=True)

    meta = dict(task=args.task, family="ManiSkill", policy="tdmpc2-goal-resample",
                episodes=len(eps), horizon=args.horizon, obs_dim=int(ro), action_dim=int(ra),
                mean_goals_per_episode=meang, hold=args.hold, stuck=args.stuck, noise=args.noise,
                checkpoint=cp.resolve_checkpoint(args.task))
    with open(os.path.join(args.out, f"ManiSkill__{args.task}", "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[ms-resample] DONE {args.task}", flush=True)


if __name__ == "__main__":
    main()
