"""data_collection/collect_expert.py — collect episodes from a newt/TD-MPC2 checkpoint
into the unified data format (Metaworld / ManiSkill).

Runs in a mani_skill-capable venv (newt's .venv). It reuses newt's TD-MPC2 agent +
env-making (via an ``envs``-package stub that avoids newt's eager all-backend imports
AND avoids clashing with Factored-FB's own ``envs`` package), rolls the planner over
``num_envs`` parallel episodes, and writes one unified ``.npz`` per episode
(data.unified format) that FB training consumes offline (train.py build_env=false).

Per-step recorded fields: observation, action, next_observation, reward, terminal,
timeout, success. ``achieved_goal`` is a placeholder here (zeros) — FB *training*
does not use it; the (future) goal evaluator / relabeler fills it properly.

    # in newt's venv, from the Factored-FB repo root:
    /raid/tongle/rl/newt/.venv/bin/python data_collection/collect_expert.py \
        task=ms-pick-cube +num_episodes=1000 +num_envs=50 \
        +checkpoint_dir=<dir> +out=/raid/tongle/rl/fb-data/collected \
        +success_only=true
"""

import os
import sys
import types

os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")  # headless Metaworld rendering-free rollout still inits GL

_HERE = os.path.dirname(os.path.abspath(__file__))
_FB_ROOT = os.path.dirname(_HERE)                          # Factored-FB repo root (for data.unified)
_NEWT_TD = os.environ.get("NEWT_TDMPC2_ROOT", "/raid/tongle/rl/newt/tdmpc2")

# Register the newt envs-package stub BEFORE any `import envs.*` (bypasses newt's
# eager backend imports and shadows Factored-FB's envs/ package).
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(_NEWT_TD, "envs")]
    _pkg.__package__ = "envs"
    sys.modules["envs"] = _pkg

# Path order: newt/tdmpc2 first (config/common/tdmpc2/envs), then FB root (data.unified).
for p in (_NEWT_TD, _FB_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import hydra
import gymnasium as gym
from hydra.core.config_store import ConfigStore

from common import set_seed
from common.world_model import WorldModel
from config import Config, parse_cfg
from tdmpc2 import TDMPC2
from envs.wrappers.vectorized_multitask import make_vectorized_multitask_env

from data.unified import write_episode, write_meta, dataset_dir  # Factored-FB (no envs import)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Real (unpadded) obs/action dims of the backend env — newt pads to a common
# multitask size (e.g. 128/16); we strip the trailing zero-padding for a clean
# single-task dataset. Captured when the first child env is built.
_REAL = {}


def make_env(cfg):
    """Single-backend env-maker (ms-/mw-), mirroring envs/__init__.make_env."""
    gym.logger.set_level(40)
    if not cfg.child_env:
        env = make_vectorized_multitask_env(cfg, make_env)
    else:
        task = cfg.task
        if task.startswith("ms-"):
            from envs.maniskill import make_env as backend
        elif task.startswith("mw-"):
            from envs.metaworld import make_env as backend
        else:
            raise ValueError(f"collect_expert supports ms-/mw- tasks only, got {task!r}")
        env = backend(cfg)
        _REAL.setdefault("obs", int(env.observation_space.shape[0]))
        _REAL.setdefault("act", int(env.action_space.shape[0]))
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    return env


def _robust_load(agent, ckpt):
    """Load a newt checkpoint keeping the current model's derived/frozen buffers
    (_action_masks / _task_emb). Avoids api_model_conversion's div-by-zero repeat
    when num_envs isn't a multiple of the checkpoint's action_masks row count."""
    from common.layers import api_model_conversion

    sd = torch.load(ckpt, map_location=torch.get_default_device(), weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    for k in list(sd.keys()):
        if k.endswith("_action_masks") or k.endswith("_task_emb.weight"):
            sd.pop(k)
    sd = api_model_conversion(agent.model.state_dict(), sd)
    missing, unexpected = agent.model.load_state_dict(sd, strict=False)
    crit = [m for m in missing if not (m.endswith("_action_masks") or "_task_emb" in m)]
    assert not crit, f"missing critical checkpoint keys: {crit[:8]}"


def _state(obs):
    """Flatten one env-step observation to a [num_envs, obs_dim] float32 numpy array."""
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs = obs["state"]
    return np.asarray(obs.cpu() if hasattr(obs, "cpu") else obs, dtype=np.float32)


@hydra.main(version_base=None, config_name="config")
def main(cfg):
    assert torch.cuda.is_available()
    ck_dir = cfg.get("checkpoint_dir")
    out = cfg.get("out", "/raid/tongle/rl/fb-data/collected")
    n_episodes = int(cfg.get("num_episodes", 1000))
    n_envs = int(cfg.get("num_envs", 50))
    success_only = bool(cfg.get("success_only", True))

    cfg.enable_wandb = False
    cfg.env_mode = "sync"
    cfg.checkpoint = f"{ck_dir}/{cfg.task}.pt"
    cfg.num_envs = n_envs
    cfg.model_size = "B"
    cfg.save_video = False
    cfg.compile = False
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    assert os.path.exists(cfg.checkpoint), f"missing checkpoint {cfg.checkpoint}"

    env = make_env(cfg)
    tasks = torch.arange(cfg.num_envs, dtype=torch.int32)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    _robust_load(agent, cfg.checkpoint)

    root = dataset_dir(out, f"{'Metaworld' if cfg.task.startswith('mw-') else 'ManiSkill'}/{cfg.task}")
    buffer_dir = root / "buffer"
    buffer_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    n_success_total = 0
    batch = 0
    while written < n_episodes:
        batch += 1
        obs, info = env.reset()
        obs_seq = [_state(obs)]
        act_seq, rew_seq, term_seq, trunc_seq, succ_seq = [], [], [], [], []
        done = torch.ones(cfg.num_envs, dtype=torch.bool)
        while True:
            action = agent(obs, t0=done, task=tasks, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_seq.append(_state(obs))
            act_seq.append(np.asarray(action.cpu(), dtype=np.float32))
            rew_seq.append(np.asarray(reward.cpu(), dtype=np.float32).reshape(-1))
            term_seq.append(np.asarray(terminated.cpu(), dtype=bool).reshape(-1))
            trunc_seq.append(np.asarray(truncated.cpu(), dtype=bool).reshape(-1))
            succ_seq.append(np.asarray(info["success"].cpu() if hasattr(info["success"], "cpu")
                                       else info["success"], dtype=np.float32).reshape(-1))
            done = terminated | truncated
            if done.any():
                assert done.all(), "fixed-horizon envs finish together"
                ep_success = np.asarray(info["final_info"]["success"].cpu()
                                        if hasattr(info["final_info"]["success"], "cpu")
                                        else info["final_info"]["success"], dtype=np.float32).reshape(-1)
                break

        obs_arr = np.stack(obs_seq, axis=0)              # [T+1, N, obs_dim(padded)]
        act_arr = np.stack(act_seq, axis=0)              # [T, N, act_dim(padded)]
        ro = _REAL.get("obs", obs_arr.shape[-1])
        ra = _REAL.get("act", act_arr.shape[-1])
        obs_arr = obs_arr[..., :ro]                      # strip newt multitask zero-padding
        act_arr = act_arr[..., :ra]
        rew_arr = np.stack(rew_seq, axis=0)              # [T, N]
        term_arr = np.stack(term_seq, axis=0)
        trunc_arr = np.stack(trunc_seq, axis=0)
        succ_arr = np.stack(succ_seq, axis=0)
        T, N = act_arr.shape[0], act_arr.shape[1]
        n_success_total += int(ep_success.sum())

        for i in range(N):
            if written >= n_episodes:
                break
            if success_only and ep_success[i] < 1.0:
                continue
            traj = {
                "observation": obs_arr[:-1, i],           # s_t
                "action": act_arr[:, i],
                "next_observation": obs_arr[1:, i],       # s_{t+1}
                "achieved_goal": np.zeros((T, 3), dtype=np.float32),  # placeholder (unused by training)
                "reward": rew_arr[:, i],
                "terminal": term_arr[:, i],
                "timeout": trunc_arr[:, i],
                "success": succ_arr[:, i] > 0.5,
            }
            write_episode(buffer_dir, written, traj)
            written += 1
        print(f"[collect] batch {batch}: episode_success={ep_success.mean():.2f} "
              f"written={written}/{n_episodes}", flush=True)

    write_meta(root, {
        "env_family": "maniskill" if cfg.task.startswith("ms-") else "metaworld",
        "task": cfg.task,
        "checkpoint": cfg.checkpoint,
        "policy": "tdmpc2-expert",
        "success_only": success_only,
        "num_episodes": written,
        "episode_length": int(T),
        "obs_dim": int(obs_arr.shape[-1]),
        "act_dim": int(act_arr.shape[-1]),
        "note": "achieved_goal is a placeholder (zeros); training does not use it.",
    })
    print(f"[collect] DONE {cfg.task}: wrote {written} episodes ({T} steps each) to {root}")


if __name__ == "__main__":
    main()
