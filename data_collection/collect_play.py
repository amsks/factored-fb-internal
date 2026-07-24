"""data_collection/collect_play.py — OGBench-style PLAY data from a newt/TD-MPC2 checkpoint.

Motivation: our Box2D/Metaworld/ManiSkill datasets are expert-only, success-narrow demos.
cube-single (OGBench) is diverse *play* data, and that is exactly the regime where the
goal-conditioned critics (gciql / iql_bilinear) shine. This collector produces the play-data
analogue for the other tasks: a trained TD-MPC2 agent rolled out with heavy exploration so the
data covers the state space broadly instead of one optimal corridor.

PLAY = exploratory, NOT expert:
  * stochastic policy prior (eval_mode=False), planning off by default (faster + more diverse);
  * epsilon-random action mixing + Gaussian action noise;
  * fixed-horizon episodes run to the end (no stop-at-success), so post-success wandering and
    failures are all kept -- that broad coverage is the whole point.

The TD-MPC2 checkpoints live in the multitask-padded space (obs 128 / action 16); the newt
vectorized wrapper right-zero-pads the raw env obs and truncates the action. rlmini/ms envs
here are the SAME envs env/unified.py evaluates on, so slicing the padded obs/action back to
each task's real dims yields rows byte-compatible with load_unified() -- no conversion.

Writes one unified .npz per episode under <out>/<Family>__<task>/buffer/ with keys
observation, action, next_observation, reward, terminal, timeout, success (env/unified.py
load_unified reads observation/action/next_observation/reward; terminals come from the episode
boundary). Also renders a grid mp4 of the play rollout via env/video.py.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 /raid/tongle/rl/newt/.venv/bin/python \
        data_collection/collect_play.py --task b2d-center-ctrl --episodes 200 --video_k 4
"""
import argparse
import glob
import json
import os
import sys
import types

import numpy as np

_TD = os.environ.get("NEWT_TDMPC2", "/raid/tongle/rl/newt/tdmpc2")
_FB = "/raid/tongle/rl/Factored-FB"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

# Register a stub `envs` package (newt's backend) BEFORE any `import envs.*`, so newt's
# envs/__init__.py never eagerly imports every backend. This is newt's own pattern
# (rollout_expert.py / train_rlmini.py). Factored-FB's package is `env` (singular) -- no clash.
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(_TD, "envs")]
    _pkg.__package__ = "envs"
    sys.modules["envs"] = _pkg
for _p in (_TD, _FB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import hydra.utils as _hu  # noqa: E402
_hu.get_original_cwd = lambda: os.getcwd()          # parse_cfg needs it outside @hydra.main

_FAMILY = {"b2d": "Box2D", "p2d": "Box2D", "mw": "Metaworld", "ms": "maniskill"}


def resolve_checkpoint(task):
    """Newest TD-MPC2 checkpoint for `task`, or None. b2d/p2d -> logs_rlmini; ms -> HF snapshot."""
    if task.startswith(("b2d-", "p2d-")):
        named = glob.glob(f"{_TD}/logs_rlmini/{task}/*/models/{task}.pt")   # final (task-named)
        if named:
            return sorted(named)[-1]
        steps = glob.glob(f"{_TD}/logs_rlmini/{task}/*/models/*.pt")        # else latest step
        steps = [p for p in steps if os.path.basename(p)[0].isdigit()]
        return max(steps, key=lambda p: int(os.path.basename(p).split(".")[0].replace("_", ""))) \
            if steps else None
    if task.startswith("ms-"):
        hits = glob.glob(f"{_TD}/../checkpoints/**/{task}.pt", recursive=True)
        return hits[0] if hits else None
    return None                                                            # mw-: no TD-MPC2 ckpt


def build_cfg(task, num_envs, render=False, render_size=128):
    # Runtime extras (num_demos) aren't in the structured Config schema, so pass everything as
    # compose overrides -- `++` adds-or-overrides regardless of whether the key exists. Mirrors
    # rollout_expert.py's `+num_demos=` CLI. num_demos>0 makes make_env apply the Render wrapper.
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from config import Config, parse_cfg
    ConfigStore.instance().store(name="config", node=Config)
    with initialize_config_dir(version_base=None, config_dir=_TD):
        cfg = compose(config_name="config", overrides=[
            f"task={task}", f"num_envs={num_envs}", "model_size=B",
            "enable_wandb=false", "env_mode=sync", "compile=false",
            f"save_video={'true' if render else 'false'}",
            f"++num_demos={num_envs if render else 0}", f"render_size={render_size}"])
    return parse_cfg(cfg)


def make_vec_env(cfg):
    """Vectorized single-backend env (avoids newt's eager all-backend import)."""
    import gymnasium as gym
    from envs.wrappers.vectorized_multitask import make_vectorized_multitask_env
    from envs.wrappers.render import Render
    gym.logger.set_level(40)

    def _mk(c):
        if not c.child_env:
            return make_vectorized_multitask_env(c, _mk)
        t = c.task
        if t.startswith(("b2d-", "p2d-")):
            from envs.rlmini import make_env as backend
        elif t.startswith("ms-"):
            from envs.maniskill import make_env as backend
        elif t.startswith("mw-"):
            from envs.metaworld import make_env as backend
        else:
            raise ValueError(f"no backend for {t!r}")
        e = backend(c)
        if c.save_video and c.get("num_demos", 0) > 0:
            e = Render(e, c)
        return e

    env = _mk(cfg)
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    return env


def real_dims(task, cfg):
    """(real_obs_dim, real_act_dim) before multitask padding. act from cfg.action_dims; obs
    from one raw child-env reset (newt right-zero-pads, so this is exact)."""
    real_act = int(cfg.action_dims[0])
    from copy import deepcopy
    c = deepcopy(cfg)
    c.child_env = True
    c.save_video = False
    c.num_demos = 0
    if task.startswith(("b2d-", "p2d-")):
        from envs.rlmini import make_env as backend
    elif task.startswith("ms-"):
        from envs.maniskill import make_env as backend
    elif task.startswith("mw-"):
        from envs.metaworld import make_env as backend
    raw = backend(c)
    sp = raw.observation_space
    real_obs = int(sp.spaces["state"].shape[0] if hasattr(sp, "spaces") else sp.shape[0])
    try:
        raw.close()
    except Exception:
        pass
    return real_obs, real_act


def load_agent(cfg, ckpt):
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    sd = torch.load(ckpt, map_location=f"cuda:{cfg.rank}", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    sd = {k: v for k, v in sd.items() if k != "_action_masks"}   # buffer; ours is right for num_envs
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "_action_masks" not in k]
    assert not bad and not res.unexpected_keys, \
        f"checkpoint mismatch missing={bad} unexpected={res.unexpected_keys}"
    return agent


def _state(obs):
    return obs["state"] if hasattr(obs, "keys") else obs


def play_actions(agent, obs, done, tasks, mpc, eps_v, noise_v):
    """One play step. eps_v/noise_v are PER-ENV [N,1] exploration levels (a mixture across
    episodes = a spectrum from near-expert to exploratory, which is what makes it play data)."""
    a = agent(obs, t0=done, task=tasks, eval_mode=False, mpc=mpc)
    eps_v, noise_v = eps_v.to(a.device), noise_v.to(a.device)   # ms actions may come back on cpu
    a = a + noise_v * torch.randn_like(a)
    rand = torch.empty_like(a).uniform_(-1.0, 1.0)
    m = (torch.rand(a.shape[0], 1, device=a.device) < eps_v).float()
    a = m * rand + (1.0 - m) * a
    return torch.clamp(a, -1.0, 1.0)


def _explore_levels(N, eps_max, noise_max, expert_frac, device, rng):
    """Per-env exploration: `expert_frac` of envs get 0 (competent, guarantees success coverage);
    the rest draw eps ~ U(0,eps_max), noise ~ U(0,noise_max)."""
    eps = torch.rand(N, 1, generator=rng, device=device) * eps_max
    noise = torch.rand(N, 1, generator=rng, device=device) * noise_max
    expert = torch.rand(N, 1, generator=rng, device=device) < expert_frac
    eps = torch.where(expert, torch.zeros_like(eps), eps)
    noise = torch.where(expert, torch.zeros_like(noise), noise)
    return eps, noise


def rollout(agent, env, cfg, ro, ra, episodes, eps_max, noise_max, expert_frac, mpc, seed0,
            capture_frames=0):
    """Collect `episodes` full-horizon play episodes; return list of episode dicts + frames.
    Fixed-horizon vectorized envs finish together, so each batch yields num_envs episodes."""
    H, N = cfg.episode_length, cfg.num_envs
    dev = f"cuda:{cfg.rank}"
    tasks = torch.arange(N, dtype=torch.int32)
    rng = torch.Generator(device=dev).manual_seed(seed0)
    episodes_out, frames_out = [], []
    n_batches = (episodes + N - 1) // N
    for b in range(n_batches):
        try:
            obs, info = env.reset(seed=seed0 + b)
        except TypeError:
            obs, info = env.reset()
        eps_v, noise_v = _explore_levels(N, eps_max, noise_max, expert_frac, dev, rng)
        want_frames = capture_frames if b == 0 else 0
        fr = [info["frame"].clone().cpu().numpy()] if (want_frames and "frame" in info) else None
        O = np.empty((H, N, ro), np.float32)
        A = np.empty((H, N, ra), np.float32)
        NO = np.empty((H, N, ro), np.float32)
        R = np.empty((H, N), np.float32)
        SU = np.zeros((H, N), np.float32)
        done = torch.ones(N, dtype=torch.bool)
        st = _state(obs)
        for t in range(H):
            a = play_actions(agent, obs, done, tasks, mpc, eps_v, noise_v)
            nobs, r, term, trunc, info = env.step(a)
            nst = _state(nobs)
            O[t] = st[:, :ro].detach().cpu().numpy()
            A[t] = a[:, :ra].detach().cpu().numpy()
            NO[t] = nst[:, :ro].detach().cpu().numpy()
            R[t] = r.detach().cpu().numpy()
            s = info.get("success")
            if s is not None:
                SU[t] = (s.float() if torch.is_tensor(s) else torch.as_tensor(s)).cpu().numpy()
            if fr is not None and "frame" in info:
                f = info["frame"].clone()
                fr.append(f.cpu().numpy())
            obs, st, done = nobs, nst, (term | trunc)
        for i in range(N):
            if len(episodes_out) >= episodes:
                break
            episodes_out.append(dict(
                observation=O[:, i], action=A[:, i], next_observation=NO[:, i],
                reward=R[:, i], success=SU[:, i]))
        if fr is not None:
            frames = np.stack(fr, 0)                       # [T+1, N, 3, Hh, Ww]
            for i in range(min(want_frames, N)):
                frames_out.append(np.moveaxis(frames[:, i], 1, -1))   # [T, Hh, Ww, 3]
    return episodes_out[:episodes], frames_out


# ------------------------------------------------------------------------------------------
# Metaworld: no TD-MPC2 checkpoint ships for mw-push / mw-pick-place, so drive the built-in
# SCRIPTED ORACLE (metaworld.policies) with the SAME per-episode exploration mixture. Same
# unified schema + video path as the TD-MPC2 tasks, so the two are drop-in downstream.
# ------------------------------------------------------------------------------------------
def _mw_policy(task):
    import metaworld.policies as P
    base = task.split("-", 1)[-1]
    overrides = {"peg-insert-side": "SawyerPegInsertionSideV2Policy"}
    name = overrides.get(base, "Sawyer" + "".join(p.capitalize() for p in base.split("-")) + "V2Policy")
    if not hasattr(P, name):
        raise KeyError(f"no scripted policy for {base!r} (tried {name})")
    return getattr(P, name)()


def _mw_env(task, render=False):
    from envs.metaworld import MetaWorldWrapper
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

    class _C(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

        def __getattr__(self, k):
            return dict.get(self, k, None)

    env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
    raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=0, render_mode="rgb_array")
    return MetaWorldWrapper(raw, _C())


def rollout_mw(task, episodes, horizon, eps_max, noise_max, expert_frac, seed0, capture_frames=0):
    """Scripted-oracle play rollout (single env, sequential episodes). Full horizon, no early
    stop; per-episode exploration mixture just like the TD-MPC2 path."""
    policy = _mw_policy(task)
    env = _mw_env(task, render=capture_frames > 0)
    rng = np.random.RandomState(seed0)
    out, frames = [], []
    for e in range(episodes):
        np.random.seed(seed0 + e)                          # MW draws rand_vec from global np
        obs, info = env.reset()
        ex = rng.rand() < expert_frac
        eps = 0.0 if ex else rng.rand() * eps_max
        noise = 0.0 if ex else rng.rand() * noise_max
        want = capture_frames and e < capture_frames
        ff = [np.asarray(env.render())] if want else None
        O, A, NO, R, SU = [], [], [], [], []
        st = np.asarray(obs, np.float32).reshape(-1)
        for _ in range(horizon):
            a = np.asarray(policy.get_action(st.astype(np.float64)), np.float32).reshape(-1)
            a = a + noise * rng.randn(*a.shape).astype(np.float32)
            if rng.rand() < eps:
                a = rng.uniform(-1.0, 1.0, size=a.shape).astype(np.float32)
            a = np.clip(a, -1.0, 1.0)
            nobs, r, term, trunc, info = env.step(a)
            nst = np.asarray(nobs, np.float32).reshape(-1)
            O.append(st); A.append(a); NO.append(nst); R.append(float(r))
            SU.append(float(info.get("success", 0.0)))
            if want:
                ff.append(np.asarray(env.render()))
            st = nst
            if term or trunc:
                break
        out.append(dict(observation=np.stack(O), action=np.stack(A),
                        next_observation=np.stack(NO), reward=np.asarray(R, np.float32),
                        success=np.asarray(SU, np.float32)))
        if want:
            frames.append(np.stack(ff))
    try:
        env.close()
    except Exception:
        pass
    return out, frames


def write_episodes(out_root, family, task, eps_list, start_idx=0):
    d = os.path.join(out_root, f"{family}__{task}", "buffer")
    os.makedirs(d, exist_ok=True)
    for j, ep in enumerate(eps_list):
        H = len(ep["action"])
        term = np.zeros(H, np.float32)
        timeout = np.zeros(H, np.float32)
        timeout[-1] = 1.0
        np.savez_compressed(
            os.path.join(d, f"episode_{start_idx + j:06d}_{H:05d}.npz"),
            observation=ep["observation"], action=ep["action"],
            next_observation=ep["next_observation"], reward=ep["reward"],
            terminal=term, timeout=timeout, success=ep["success"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: auto-resolve for the task")
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/collected_play")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=20)
    # Exploration is a per-episode MIXTURE (near-expert .. exploratory). Defaults differ by
    # family: Box2D is robust over 200 steps so it takes heavy noise; ManiSkill's 25-step
    # manipulation is fragile, so gentler noise + planning keeps real goal-reaching in the data.
    ap.add_argument("--eps_max", type=float, default=None, help="max uniform-random action fraction")
    ap.add_argument("--noise_max", type=float, default=None, help="max Gaussian action-noise std")
    ap.add_argument("--expert_frac", type=float, default=None, help="fraction of no-noise episodes")
    ap.add_argument("--mpc", type=int, default=None, help="1 = plan (competent), 0 = policy prior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_k", type=int, default=4, help="episodes rendered into the play mp4")
    ap.add_argument("--render_size", type=int, default=128)
    ap.add_argument("--video_dir", default="/raid/tongle/rl/fb-data/collected_play/_videos")
    args = ap.parse_args()

    family = _FAMILY[args.task.split("-")[0]]
    is_mw = family == "Metaworld"                          # scripted oracle: no TD-MPC2 ckpt ships
    # per-family exploration defaults (CLI overrides win)
    DEFAULTS = {"Box2D": dict(eps_max=0.5, noise_max=0.4, expert_frac=0.2, mpc=0),
                "maniskill": dict(eps_max=0.25, noise_max=0.15, expert_frac=0.3, mpc=1),
                "Metaworld": dict(eps_max=0.3, noise_max=0.2, expert_frac=0.3, mpc=0)}[family]
    eps_max = DEFAULTS["eps_max"] if args.eps_max is None else args.eps_max
    noise_max = DEFAULTS["noise_max"] if args.noise_max is None else args.noise_max
    expert_frac = DEFAULTS["expert_frac"] if args.expert_frac is None else args.expert_frac
    mpc = DEFAULTS["mpc"] if args.mpc is None else args.mpc

    frames = []
    if is_mw:
        # --- Metaworld: scripted-oracle play (no checkpoint) ---
        H = 100
        print(f"[play] {args.task}: scripted oracle (no TD-MPC2 checkpoint)", flush=True)
        print(f"[play] explore: eps_max={eps_max} noise_max={noise_max} "
              f"expert_frac={expert_frac}", flush=True)
        eps_list, _ = rollout_mw(args.task, args.episodes, H, eps_max, noise_max, expert_frac,
                                 args.seed * 1000)
        ro, ra = eps_list[0]["observation"].shape[-1], eps_list[0]["action"].shape[-1]
        ckpt = "metaworld-scripted-oracle"
        policy_tag = "scripted-play"
        if args.video_k > 0:
            _, frames = rollout_mw(args.task, args.video_k, H, eps_max, noise_max, expert_frac,
                                   args.seed * 1000 + 777, capture_frames=args.video_k)
    else:
        # --- Box2D / ManiSkill: TD-MPC2 play ---
        from common import set_seed
        ckpt = args.checkpoint or resolve_checkpoint(args.task)
        assert ckpt and os.path.exists(ckpt), f"no checkpoint for {args.task} ({ckpt})"
        print(f"[play] {args.task}: checkpoint={ckpt}", flush=True)
        print(f"[play] explore: eps_max={eps_max} noise_max={noise_max} "
              f"expert_frac={expert_frac} mpc={mpc}", flush=True)
        cfg = build_cfg(args.task, args.num_envs, render=False)
        set_seed(cfg.seed if hasattr(cfg, "seed") else args.seed)
        env = make_vec_env(cfg)
        ro, ra = real_dims(args.task, cfg)
        print(f"[play] real dims obs={ro} act={ra}; H={cfg.episode_length} "
              f"num_envs={cfg.num_envs}", flush=True)
        agent = load_agent(cfg, ckpt)
        eps_list, _ = rollout(agent, env, cfg, ro, ra, args.episodes,
                              eps_max, noise_max, expert_frac, bool(mpc), args.seed * 1000)
        try:
            env.close()
        except Exception:
            pass
        policy_tag = "tdmpc2-play"
        if args.video_k > 0:
            vcfg = build_cfg(args.task, max(args.video_k, 1), render=True,
                             render_size=args.render_size)
            venv = make_vec_env(vcfg)
            vagent = load_agent(vcfg, ckpt)
            _, frames = rollout(vagent, venv, vcfg, ro, ra, args.video_k,
                                eps_max, noise_max, expert_frac, bool(mpc),
                                args.seed * 1000 + 777, capture_frames=args.video_k)
            try:
                venv.close()
            except Exception:
                pass

    d = write_episodes(args.out, family, args.task, eps_list)
    succ = float(np.mean([ep["success"].max() for ep in eps_list]))
    cov = float(np.mean([ep["success"].mean() for ep in eps_list]))
    print(f"[play] wrote {len(eps_list)} episodes -> {d}  (ep-success={succ:.2f}, "
          f"at-goal-frac={cov:.2f})", flush=True)

    if frames:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_v", os.path.join(_FB, "env", "video.py"))
        vmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vmod)
        os.makedirs(args.video_dir, exist_ok=True)
        p = vmod.save_video_grid(frames, os.path.join(args.video_dir, f"{args.task}_play.mp4"),
                                 max_episodes=args.video_k)
        print(f"[play] wrote video -> {p}", flush=True)

    meta = dict(task=args.task, family=family, checkpoint=ckpt, episodes=len(eps_list),
                horizon=int(len(eps_list[0]["action"])), obs_dim=int(ro), action_dim=int(ra),
                eps_max=eps_max, noise_max=noise_max, expert_frac=expert_frac, mpc=bool(mpc),
                policy=policy_tag, ep_success=succ, at_goal_frac=cov)
    with open(os.path.join(args.out, f"{family}__{args.task}", "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[play] DONE {args.task}", flush=True)


if __name__ == "__main__":
    main()
