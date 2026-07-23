"""campaign/eval/eval_common.py — venv-agnostic eval helpers shared by the PyTorch
(eval_fb, newt venv) and JAX (eval_jax, .venv-jax) evaluators. numpy-only + envs; no
torch/jax here so it imports in either venv.

Provides: make_eval_env (raw-obs env matching the training data, per family),
seeded_reset (per-family fixed-seed reset), save_video_grid (2x2 mp4 of 4 episodes),
load_offline (concatenated next_obs + reward for z / goal inference).
"""
import glob
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

NEWT_TD = "/raid/tongle/rl/newt/tdmpc2"

# `envs` must resolve to newt's backend package (envs.rlmini / .metaworld / .maniskill).
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(NEWT_TD, "envs")]
    sys.modules["envs"] = _pkg
if NEWT_TD not in sys.path:
    sys.path.insert(0, NEWT_TD)


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)

    def __setattr__(self, k, v):
        self[k] = v


def make_eval_env(task, render=False):
    """Return (env, horizon). Obs is the RAW factored obs identical to the collected data."""
    if task.startswith(("b2d-", "p2d-")):
        from envs.rlmini import RLMiniWrapper, _PREFIX_TO_ENV, episode_length_for
        prefix = "b2d-" if task.startswith("b2d-") else "p2d-"
        env_name = _PREFIX_TO_ENV[prefix]
        variant = task[len(prefix):]
        env = RLMiniWrapper(env_name, variant, _Cfg(), seed=0)
        if render:
            import rlmini_envs
            env._raw = rlmini_envs.make(env_name, variant=variant, horizon=1_000_000_000, seed=0, render=True)
            if hasattr(env._raw, "max_steps"):
                env._raw.max_steps = 1_000_000_000
        return env, episode_length_for(task)

    if task.startswith("mw-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from envs.metaworld import MetaWorldWrapper
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
        raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=0, render_mode="rgb_array")
        return MetaWorldWrapper(raw, _Cfg()), 100

    if task.startswith("ms-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from envs.maniskill import make_env as ms_make
        cfg = _Cfg(task=task, seed=0, obs="state", render_size=128, num_envs=1)
        return ms_make(cfg), 25
    raise ValueError(f"eval env builder not wired for task {task!r}")


def seeded_reset(env, task, seed):
    """A given seed => a fixed initial state where the env supports it. rlmini honors
    reset(seed=); Metaworld draws its rand_vec from the GLOBAL np.random; ManiSkill's
    newt wrapper doesn't accept seed -> best-effort (global seed + plain reset)."""
    if task.startswith("mw-"):
        np.random.seed(seed)
        return env.reset()
    if task.startswith("ms-"):
        np.random.seed(seed)
        try:
            return env.reset(seed=seed)
        except TypeError:
            return env.reset()
    return env.reset(seed=seed)


def load_offline(data_dir):
    """Concatenate unified next_observation + reward across a task's episodes."""
    eps = sorted(glob.glob(f"{data_dir}/buffer/episode_*.npz"),
                 key=lambda p: int(re.findall(r"\d+", Path(p).stem)[-1]))
    nobs, rew = [], []
    for ep in eps:
        z = np.load(ep)
        nobs.append(z["next_observation"].astype(np.float32))
        rew.append(z["reward"].astype(np.float32).reshape(-1))
    return np.concatenate(nobs), np.concatenate(rew)


def save_video_grid(frames, out_path, fps=30):
    """Tile up to 4 episodes into one 2x2 grid mp4 (pad shorter episodes to the longest)."""
    import imageio
    frames = [f for f in frames if f]
    if not frames:
        return None
    T = max(len(f) for f in frames)
    tiles = []
    for f in frames[:4]:
        arr = np.stack(f)
        if len(arr) < T:
            arr = np.concatenate([arr, np.repeat(arr[-1:], T - len(arr), 0)])
        tiles.append(arr)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    H, W = tiles[0].shape[1:3]
    tiles = [t[:, :H, :W] for t in tiles]
    top = np.concatenate([tiles[0], tiles[1]], axis=2)
    bot = np.concatenate([tiles[2], tiles[3]], axis=2)
    grid = np.concatenate([top, bot], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, grid.astype(np.uint8), fps=fps, macro_block_size=1)
    return out_path
