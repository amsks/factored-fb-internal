"""Unified adapter — our collected per-episode datasets (Box2D / Metaworld / ManiSkill)
and their envs, behind the DomainAdapter seam.

Data layout (written by data_collection/*):
    <data_dir>/buffer/episode_XXXXXX_LLLLL.npz  with observation, action, next_observation,
                                                reward, terminal, timeout, success
We flatten those into one Dataset and mark each episode's LAST step in `terminals`, which is
what GCDataset uses as trajectory boundaries for goal sampling. Real `rewards` are kept
(unlike OGBench play data, which is unlabeled) so FB-style z-inference has a signal.

Eval is our protocol, not OGBench's: a fixed seed per episode (so the same seed is the same
initial state across algorithms/checkpoints) and TWO metrics --
    success      -- did info['success'] ever fire  ("did it reach the goal")
    at_goal_frac -- fraction of steps with info['success'] ("did it STAY there")
The second one matters: a policy can score success 1.00 while holding the goal for 27% of
the episode (gciql on b2d-goal-ctrl-maze) versus 84% for a flow actor.
"""
import glob
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

from impls.utils.datasets import FBDataset
from env import registry

# newt's env backends (rlmini / metaworld / maniskill). Its envs/__init__ imports dm_control
# for dmcontrol tasks we don't use, so bind the namespace directly instead of importing it.
NEWT_TD = os.environ.get("NEWT_TDMPC2", "/raid/tongle/rl/newt/tdmpc2")
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [os.path.join(NEWT_TD, "envs")]
    sys.modules["envs"] = _pkg
if NEWT_TD not in sys.path:
    sys.path.insert(0, NEWT_TD)


class _Cfg(dict):
    """Attribute-style config; the ManiSkill backend reads cfg.task, cfg.obs, ..."""

    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return dict.get(self, k, None)

    def __setattr__(self, k, v):
        self[k] = v


def _ep_index(p):
    return int(re.findall(r"\d+", Path(p).stem)[-1])


def load_unified(data_dir):
    """Flatten <data_dir>/buffer/episode_*.npz into one flat dataset dict."""
    eps = sorted(glob.glob(os.path.join(data_dir, "buffer", "episode_*.npz")), key=_ep_index)
    assert eps, f"no episodes under {data_dir}"
    obs, act, nobs, rew, term = [], [], [], [], []
    for ep in eps:
        z = np.load(ep)
        T = len(z["action"])
        t = np.zeros(T, np.float32)
        t[-1] = 1.0                       # episode boundary -> GCDataset trajectory split
        obs.append(z["observation"].astype(np.float32))
        act.append(np.clip(z["action"], -1.0, 1.0).astype(np.float32))
        nobs.append(z["next_observation"].astype(np.float32))
        rew.append(z["reward"].astype(np.float32).reshape(-1))
        term.append(t)
    return dict(observations=np.concatenate(obs), actions=np.concatenate(act),
                next_observations=np.concatenate(nobs), rewards=np.concatenate(rew),
                terminals=np.concatenate(term))


def make_task_env(task, render=False):
    """-> (env, horizon). Obs is the RAW factored obs, identical to the collected data."""
    if task.startswith(("b2d-", "p2d-")):
        from envs.rlmini import RLMiniWrapper, _PREFIX_TO_ENV, episode_length_for
        prefix = "b2d-" if task.startswith("b2d-") else "p2d-"
        variant = task[len(prefix):]
        env = RLMiniWrapper(_PREFIX_TO_ENV[prefix], variant, _Cfg(), seed=0)
        if render:
            import rlmini_envs
            env._raw = rlmini_envs.make(_PREFIX_TO_ENV[prefix], variant=variant,
                                        horizon=1_000_000_000, seed=0, render=True)
            if hasattr(env._raw, "max_steps"):
                env._raw.max_steps = 1_000_000_000
        return env, episode_length_for(task)
    if task.startswith("mw-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from envs.metaworld import MetaWorldWrapper
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task.split("-", 1)[-1] + "-v2-goal-observable"](
            seed=0, render_mode="rgb_array")
        return MetaWorldWrapper(raw, _Cfg()), 100
    if task.startswith("ms-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from envs.maniskill import make_env as ms_make
        return ms_make(_Cfg(task=task, seed=0, obs="state", render_size=128, num_envs=1)), 25
    raise ValueError(f"no env builder for task {task!r}")


def seeded_reset(env, task, seed):
    """A given seed => a fixed initial state wherever the backend allows it. rlmini honors
    reset(seed=); Metaworld draws its rand_vec from the GLOBAL np.random; ManiSkill's wrapper
    does not accept a seed, so it is best-effort."""
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


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


class UnifiedAdapter:
    name = "unified"

    def make_env_and_datasets(self, cfg):
        data = load_unified(cfg.get("data_dir") or registry.data_dir(cfg["task"]))
        # FROZEN TEACHER as the BC regularizer: relabel once, offline. Actors regress toward
        # data["teacher_actions"] instead of co-training their own BC head.
        tck = cfg.get("teacher_ckpt")
        if tck:
            from env.teacher import load_teacher
            data["teacher_actions"] = load_teacher(tck).relabel(data["observations"])
            print(f"[env] relabeled with frozen teacher {tck}", flush=True)
        env, horizon = make_task_env(cfg["task"], render=bool(cfg.get("render", True)))
        self._horizon = horizon
        n = len(data["observations"])
        # Hold out the tail as validation. The split lands mid-episode only if val_frac is
        # tiny; terminals still mark every boundary inside each side.
        n_val = int(n * float(cfg.get("val_frac", 0.05)))
        if n_val > 0:
            cut = n - n_val
            tr = {k: v[:cut] for k, v in data.items()}
            va = {k: v[cut:] for k, v in data.items()}
            tr["terminals"][-1] = 1.0     # GCDataset asserts the last index is terminal
            va["terminals"][-1] = 1.0
        else:
            tr, va = data, data
        return env, FBDataset.create(**tr), FBDataset.create(**va)

    def eval_tasks(self, env, cfg):
        return [None]                      # single task per domain

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        import jax
        slug = cfg["task"]
        horizon = getattr(self, "_horizon", None) or int(cfg.get("horizon", 100))
        is_box2d = slug.startswith(("b2d-", "p2d-"))   # at-goal ticks are Box2D-only
        # z-critics condition on agent.cond_eval (goals=None); goal-conditioned critics need
        # an explicit goal, for which we use the highest-reward next_obs in the offline data.
        provides = set(getattr(agent.critic, "PROVIDES_COND", ()) or ())
        goal = None
        if "goal" in provides and "z" not in provides:
            d = load_unified(cfg.get("data_dir") or registry.data_dir(cfg["task"]))
            goal = d["next_observations"][int(np.argmax(d["rewards"]))].astype(np.float32)

        successes, ticks, frames = [], [], []
        for i in range(num_episodes):
            obs, info = seeded_reset(env, slug, i)
            key = jax.random.PRNGKey(i)
            ep_success = _flag(info.get("success", 0))
            at_goal = int(_flag(info.get("success", 0)))
            record = i < num_video
            ff = [env.render()] if record else None
            for _ in range(horizon):
                key, sub = jax.random.split(key)
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(obs, np.float32), goals=goal,
                    seed=sub, temperature=0))
                obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                now = _flag(info.get("success", 0))
                ep_success = ep_success or now
                at_goal += int(now)
                if record:
                    ff.append(env.render())
                if _flag(term) or _flag(trunc):
                    break
            successes.append(float(ep_success))
            ticks.append(at_goal)                       # integer steps spent at goal
            if record:
                frames.append(np.array(ff))
        out = {"success": float(np.mean(successes))}
        if is_box2d:                                    # at-goal ticks: Box2D only, integer scale
            out["at_goal_ticks"] = float(np.mean(ticks))
        return out, frames
