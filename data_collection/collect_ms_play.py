"""data_collection/collect_ms_play.py — ManiSkill PLAY data from a newt TD-MPC2 checkpoint,
rolled out in the working mani_skill 3.0.1 env (env/maniskill_env.py) with obs padding.

The newt checkpoints live in a multitask-padded space (obs 128 / action 16); the raw 3.0.1
env obs (42 for pick-cube, 35 for push-cube) is right-zero-padded to 128 before the model,
and the model's action is truncated to the task's real dim — exactly what newt's vectorized
wrapper did, so the recorded rows are byte-compatible with env/unified.py::load_packed.

play-noise = a per-episode exploration mixture (a fraction of clean episodes; the rest get
eps-random action mixing + Gaussian action noise), so coverage spreads beyond the one optimal
corridor. Defaults are newt's ManiSkill row (gentler than Box2D: the 25-step manipulation is
fragile).

    VK_ICD_FILENAMES=$SCRATCH/vulkan/nvidia_icd.json .venv-gen/bin/python \
        data_collection/collect_ms_play.py --task ms-pick-cube --episodes 1000 \
        --out $SCRATCH/fb-data/play-noise
"""
import argparse
import json
import os
import sys
import types

import numpy as np

_TD = "/work/11708/amsks/vista/git/newt/tdmpc2"
_FB = "/work/11708/amsks/vista/git/Factored-FB"
PLAY_DEFAULTS = dict(eps_max=0.25, noise_max=0.15, expert_frac=0.3)


def _load_agent(task):
    """-> (agent, obs_pad_dim, real_act_dim, task_idx). Loads the newt WorldModel + checkpoint."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("VK_ICD_FILENAMES", "/scratch/11708/amsks/vulkan/nvidia_icd.json")
    if "envs" not in sys.modules:
        p = types.ModuleType("envs"); p.__path__ = [os.path.join(_TD, "envs")]
        sys.modules["envs"] = p
    for x in (_TD, _FB):
        if x not in sys.path:
            sys.path.insert(0, x)
    import torch
    import hydra.utils as _hu
    _hu.get_original_cwd = lambda: os.getcwd()
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from config import Config, parse_cfg
    ConfigStore.instance().store(name="config", node=Config)
    with initialize_config_dir(version_base=None, config_dir=_TD):
        cfg = compose(config_name="config", overrides=[
            f"task={task}", "num_envs=1", "model_size=B",
            "enable_wandb=false", "env_mode=sync", "compile=false", "save_video=false"])
    cfg = parse_cfg(cfg)
    cfg.rank = 0
    if not getattr(cfg, "obs_shape", None):
        cfg.obs_shape = {"state": (128,)}       # model_size=B multitask padded dims
    if not cfg.action_dim:
        cfg.action_dim = 16
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    model = WorldModel(cfg).to("cuda:0")
    ck = f"/scratch/11708/amsks/newt-ckpt/{task}.pt"
    sd = torch.load(ck, map_location="cuda:0", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    sd = {k: v for k, v in sd.items() if k != "_action_masks"}
    res = model.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if "_action_masks" not in k] and not res.unexpected_keys
    if _TD in sys.path:
        sys.path.remove(_TD)                    # so `import env.*` finds Factored-FB, not newt
    return TDMPC2(model, cfg), cfg.obs_shape["state"][0], int(cfg.action_dims[0]), 0


def collect(task, episodes, regime, eps_max, noise_max, expert_frac, seed0):
    import torch
    agent, pad_dim, real_act, task_idx = _load_agent(task)
    from env.maniskill_env import make_maniskill_env
    env, H = make_maniskill_env(task)
    rng = np.random.RandomState(seed0)
    tt = torch.tensor([task_idx], device="cuda:0")
    O, A, NO, R, SU = [], [], [], [], []
    for e in range(episodes):
        obs, info = env.reset(seed=seed0 + e)
        if regime == "expert":
            eps = noise = 0.0
        else:
            clean = rng.rand() < expert_frac
            eps = 0.0 if clean else rng.rand() * eps_max
            noise = 0.0 if clean else rng.rand() * noise_max
        o, a, no, r, su = [], [], [], [], []
        st = np.asarray(obs, np.float32).reshape(-1)
        for t in range(H):
            po = np.concatenate([st, np.zeros(pad_dim - st.shape[0], np.float32)])
            with torch.no_grad():
                act = agent(torch.tensor(po, device="cuda:0").unsqueeze(0), t0=(t == 0),
                            eval_mode=False, task=tt, mpc=False)
            act = np.asarray(act.squeeze(0).cpu(), np.float32)[:real_act]
            act = act + noise * rng.randn(*act.shape).astype(np.float32)
            if rng.rand() < eps:
                act = rng.uniform(-1.0, 1.0, size=act.shape).astype(np.float32)
            act = np.clip(act, -1.0, 1.0)
            nobs, rew, term, trunc, info = env.step(act)
            nst = np.asarray(nobs, np.float32).reshape(-1)
            o.append(st); a.append(act); no.append(nst)
            r.append(float(rew)); su.append(float(_flag(info.get("success", 0))))
            st = nst
        O.append(np.stack(o)); A.append(np.stack(a)); NO.append(np.stack(no))
        R.append(np.asarray(r, np.float32)); SU.append(np.asarray(su, np.float32))
    env.close()
    n, T = len(O), H
    return dict(observation=np.stack(O), action=np.stack(A), next_observation=np.stack(NO),
                achieved_goal=np.zeros((n, T, 3), np.float32), reward=np.stack(R),
                terminal=np.zeros((n, T), bool),
                timeout=np.stack([np.eye(T, dtype=bool)[-1]] * n),
                success=np.stack(SU).astype(bool))


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="ms-pick-cube | ms-push-cube")
    ap.add_argument("--regime", choices=["expert", "play-noise"], default="play-noise")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps_max", type=float, default=PLAY_DEFAULTS["eps_max"])
    ap.add_argument("--noise_max", type=float, default=PLAY_DEFAULTS["noise_max"])
    ap.add_argument("--expert_frac", type=float, default=PLAY_DEFAULTS["expert_frac"])
    args = ap.parse_args()
    d = collect(args.task, args.episodes, args.regime,
                args.eps_max, args.noise_max, args.expert_frac, args.seed * 100000)
    outdir = os.path.join(args.out, "data", "maniskill")
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"{args.task}.npz")
    np.savez_compressed(fp, **d)
    meta = dict(task=args.task, family="maniskill", regime=args.regime, policy="tdmpc2-play",
                num_episodes=int(d["action"].shape[0]), episode_length=int(d["action"].shape[1]),
                obs_dim=int(d["observation"].shape[-1]), act_dim=int(d["action"].shape[-1]),
                ep_success=float(d["success"].any(1).mean()),
                at_goal_frac=float(d["success"].mean()),
                eps_max=args.eps_max, noise_max=args.noise_max, expert_frac=args.expert_frac)
    json.dump(meta, open(os.path.join(outdir, f"{args.task}.meta.json"), "w"), indent=1)
    print(f"[ms-play] {args.task} [{args.regime}]: {meta['num_episodes']} eps, "
          f"ep-success={meta['ep_success']:.3f}, at-goal-frac={meta['at_goal_frac']:.3f} -> {fp}",
          flush=True)


if __name__ == "__main__":
    main()
