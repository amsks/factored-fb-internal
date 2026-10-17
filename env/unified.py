"""Unified adapter for ExoRL, MetaWorld and ManiSkill.

ExoRL tasks are reward functions scored by episode return; MetaWorld and ManiSkill tasks
are scored by success. Eval episode i uses seed i.
"""
import contextlib
import glob
import os
import re
from pathlib import Path

import numpy as np

from impls.utils.datasets import FBDataset
from env import registry
from env.base import column_backed_dataset, episode_horizon, needs_goal_frame

def _ep_index(p):
    return int(re.findall(r"\d+", Path(p).stem)[-1])


def load_packed(npz_path):
    """Flatten a packed [N, T, ...] per-task npz; `terminals` marks each episode's last step."""
    z = np.load(npz_path, allow_pickle=True)
    obs, act, nobs = z["observation"], z["action"], z["next_observation"]
    assert obs.ndim == 3, (
        f"{npz_path}: expected packed [N,T,obs_dim], got {obs.shape} -- a ragged (object) "
        f"pack means the task is not fixed-horizon and needs the per-episode path")
    n_ep, T = obs.shape[0], obs.shape[1]
    term = np.zeros((n_ep, T), np.float32)
    term[:, -1] = 1.0

    def flat(a):
        a = np.asarray(a, np.float32)
        return a.reshape(n_ep * T, a.shape[-1]) if a.ndim == 3 else a.reshape(n_ep * T)

    out = dict(observations=flat(obs), actions=np.clip(flat(act), -1.0, 1.0),
               next_observations=flat(nobs), rewards=flat(z["reward"]),
               terminals=flat(term))
    # Per-step success flag, used only by reward-inference eval
    if "success" in z.files:
        out["success"] = flat(np.asarray(z["success"], np.float32))
    return out


def load_unified(data_dir):
    """Flatten <data_dir>/buffer/episode_*.npz into one flat dataset dict."""
    eps = sorted(glob.glob(os.path.join(data_dir, "buffer", "episode_*.npz")), key=_ep_index)
    assert eps, f"no episodes under {data_dir}"
    obs, act, nobs, rew, term = [], [], [], [], []
    for ep in eps:
        z = np.load(ep)
        T = len(z["action"])
        t = np.zeros(T, np.float32)
        t[-1] = 1.0
        obs.append(z["observation"].astype(np.float32))
        act.append(np.clip(z["action"], -1.0, 1.0).astype(np.float32))
        nobs.append(z["next_observation"].astype(np.float32))
        rew.append(z["reward"].astype(np.float32).reshape(-1))
        term.append(t)
    return dict(observations=np.concatenate(obs), actions=np.concatenate(act),
                next_observations=np.concatenate(nobs), rewards=np.concatenate(rew),
                terminals=np.concatenate(term))


def load_domain_data(cfg):
    """Load offline data from `data_path` (packed npz), `data_dir`, or the ExoRL loader."""
    task = cfg.get("task") or ""
    if task.startswith("dmc-"):
        from env.exorl_env import load_domain_data as load_exorl_domain
        return load_exorl_domain(cfg)
    path = cfg.get("data_path")
    if path:
        return load_packed(path)
    return load_unified(cfg.get("data_dir") or registry.data_dir(cfg["task"]))


def make_task_env(task, reward_task=None):
    """Return (env, horizon); `reward_task` selects the ExoRL reward function."""
    if task.startswith("mw-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from env.metaworld_env import make_metaworld_env
        return make_metaworld_env(task)
    if task.startswith("ms-"):
        from env.maniskill_env import make_maniskill_env
        return make_maniskill_env(task)
    if task.startswith("dmc-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from env.exorl_env import make_exorl_env
        return make_exorl_env(task, reward_task=reward_task)
    raise ValueError(f"no env builder for task {task!r}")


# Goal slots in the raw observation, for synthesized goals and goal-distance logging.
# target: goal written by the env; object: achieved goal; copy: slots set to target in a
# synthesized goal; grasp_offset: copy slot that keeps its offset from the object;
# derived_offset: obj - tcp, recomputed; zero/one: slots set to 0/1; ndim: dims the success
# rule measures.
_GOAL_LAYOUT = {
    "mw": dict(target=(36, 39), object=(4, 7),
               copy=((0, 3), (4, 7), (18, 21), (22, 25)), zero=(), one=(), ndim=3),
    "ms-push-cube": dict(target=(25, 28), object=(28, 31),
                         copy=((18, 21), (28, 31)), zero=(), one=(), ndim=2),
    "ms-pick-cube": dict(target=(26, 29), object=(29, 32),
                         copy=((19, 22), (29, 32)), zero=((39, 42),),
                         one=((18, 19),), ndim=3,
                         grasp_offset=(19, 22), derived_offset=(36, 39)),
}


def goal_layout(task):
    """Return the _GOAL_LAYOUT entry for a task slug, or None."""
    if task in _GOAL_LAYOUT:
        return _GOAL_LAYOUT[task]
    return _GOAL_LAYOUT.get(task.split("-", 1)[0])


def synth_goal_frame(task, raw_obs):
    """Goal observation built from the reset observation with end effector and object on the target."""
    lay = goal_layout(task)
    if lay is None:
        raise ValueError(f"eval_goal_mode=synth: no goal-in-obs layout registered for {task!r}")
    g = np.array(raw_obs, np.float32, copy=True)
    tgt = g[slice(*lay["target"])].copy()
    grasp = lay.get("grasp_offset")
    obj0 = g[slice(*lay["object"])].copy()
    grasp_vec = None
    for a, b in lay["copy"]:
        if grasp is not None and (a, b) == grasp:
            # Keep the gripper offset; fall back to the median grasp if it is not a grasp
            grasp_vec = g[a:b] - obj0
            if np.linalg.norm(grasp_vec) > 0.1:
                grasp_vec = np.array([0.0, 0.0, 0.018], np.float32)
            g[a:b] = tgt + grasp_vec
        else:
            g[a:b] = tgt
    for a, b in lay["zero"]:
        g[a:b] = 0.0
    for a, b in lay["one"]:
        g[a:b] = 1.0
    derived = lay.get("derived_offset")
    if derived is not None and grasp_vec is not None:
        a, b = derived
        g[a:b] = -grasp_vec
    return g


def goal_distance(task, raw_obs):
    """Distance from object to target in the raw observation, or None."""
    lay = goal_layout(task)
    if lay is None:
        return None
    o = np.asarray(raw_obs, np.float32)
    n = lay["ndim"]
    obj = o[lay["object"][0]:lay["object"][0] + n]
    tgt = o[lay["target"][0]:lay["target"][0] + n]
    return float(np.linalg.norm(obj - tgt))


@contextlib.contextmanager
def _global_np_seed(seed):
    """Seed the global np.random inside the block and restore its state afterwards."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def seeded_reset(env, task, seed):
    """Reset to the initial state fixed by `seed`."""
    if task.startswith("mw-"):
        with _global_np_seed(seed):
            return env.reset()
    if task.startswith("ms-"):
        with _global_np_seed(seed):
            try:
                return env.reset(seed=seed)
            except TypeError:
                return env.reset()
    return env.reset(seed=seed)


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


def cfg_get(adapter, key, default=None):
    """Run-config value stored by make_env_and_datasets, or `default`."""
    return (getattr(adapter, "_cfg", None) or {}).get(key, default)


class UnifiedAdapter:
    name = "unified"

    def make_env_and_datasets(self, cfg):
        self._cfg = dict(cfg)
        data = load_domain_data(cfg)
        env, horizon = make_task_env(cfg["task"], reward_task=cfg.get("exorl_task"))
        self._horizon = horizon
        # Fixed eval goal: the highest-reward next observation
        self._goal = data["next_observations"][int(np.argmax(data["rewards"]))].astype(np.float32)
        n = len(data["observations"])
        # Hold out the tail as validation
        n_val = int(n * float(cfg.get("val_frac", 0.05)))
        if n_val > 0:
            cut = n - n_val
            tr = {k: v[:cut] for k, v in data.items()}
            va = {k: v[cut:] for k, v in data.items()}
            tr["terminals"][-1] = 1.0
            va["terminals"][-1] = 1.0
        else:
            tr, va = data, data
        return env, FBDataset.create(**tr), FBDataset.create(**va)

    def eval_tasks(self, env, cfg):
        return [None]

    def relabel_infer_batch(self, train_ds, env, task_id, num_samples, shift_reward=1.0):
        """Sample transitions labeled with the task reward of their next state.

        The reward is the success flag minus 1 or the stored dense reward, plus `shift_reward`.
        Observations are masked as in training.

        :return: (next_observations, rewards) for `agent.infer_eval_cond`.
        """
        ds = column_backed_dataset(train_ds)
        mode = str(cfg_get(self, "eval_relabel_reward", "success"))
        idxs = ds.get_random_idxs(int(num_samples))
        next_obs = np.array(ds["next_observations"][idxs], np.float32, copy=True)
        if mode == "success":
            if "success" not in ds:
                raise ValueError(
                    "eval_relabel_reward=success needs the dataset's per-step `success` "
                    "column; this pack has none (load_packed keeps it when the npz carries "
                    "it; the per-episode `load_unified` layout never does). Pass "
                    "--override eval_relabel_reward=dense to relabel with the stored reward.")
            rewards = np.asarray(ds["success"][idxs], np.float32) - 1.0 + float(shift_reward)
        elif mode == "dense":
            rewards = np.asarray(ds["rewards"][idxs], np.float32) + float(shift_reward)
        else:
            raise ValueError(f"eval_relabel_reward must be 'success' or 'dense', got {mode!r}")
        dims = sorted(set(cfg_get(self, "obs_mask_dims") or ()) |
                      set(cfg_get(self, "goal_mask_dims") or ()))
        if dims:
            next_obs[:, dims] = 0.0
        return next_obs, rewards.astype(np.float32)

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video=0, any_step_success=None):
        if num_video:
            raise ValueError("video rendering is not supported; pass num_video=0")
        import jax
        slug = cfg["task"]
        horizon = episode_horizon(self)
        is_dmc = slug.startswith("dmc-")
        needs_goal = needs_goal_frame(agent)

        # Observation masking as in training
        obs_mask = list(cfg.get("obs_mask_dims") or ())
        goal_mask = list(cfg.get("goal_mask_dims") or ())

        def _mask_obs(o):
            if not obs_mask:
                return np.asarray(o, np.float32)
            o = np.array(o, np.float32, copy=True)
            o[..., obs_mask] = 0.0
            return o

        def _prep_goal(g):
            """Mask a goal observation as in training."""
            g = np.array(g, np.float32, copy=True)
            if goal_mask:
                g[goal_mask] = 0.0
            if obs_mask:
                g[obs_mask] = 0.0
            return g

        # Goal: fixed from the data, or synthesized per episode
        goal_mode = str(cfg.get("eval_goal_mode", "fixed"))
        if goal_mode not in ("fixed", "synth"):
            raise ValueError(f"eval_goal_mode must be 'fixed' or 'synth', got {goal_mode!r}")
        if needs_goal and goal_mode == "synth" and goal_layout(slug) is None:
            raise ValueError(
                f"eval_goal_mode=synth is not available for task {slug!r}: no goal-in-obs "
                f"layout is registered for it in env/unified.py::_GOAL_LAYOUT")

        goal = None
        if needs_goal and goal_mode == "fixed":
            goal = getattr(self, "_goal", None)
            if goal is None:
                d = load_domain_data(cfg)
                goal = d["next_observations"][int(np.argmax(d["rewards"]))].astype(np.float32)
            goal = _prep_goal(goal)

        successes, finals, returns = [], [], []
        saw_success = False
        dist_min, dist_final = [], []
        # Episode i uses seed i + eval_seed_offset
        seed_off = int(cfg.get("eval_seed_offset", 0) or 0)
        for i in range(num_episodes):
            obs, info = seeded_reset(env, slug, i + seed_off)
            # Synthesize from the unmasked reset observation
            ep_goal = goal
            if needs_goal and goal_mode == "synth":
                ep_goal = _prep_goal(synth_goal_frame(slug, obs))
            key = jax.random.PRNGKey(i + seed_off)
            saw_success = saw_success or ("success" in info)
            ep_success = _flag(info.get("success", 0))
            ep_return = 0.0
            now = _flag(info.get("success", 0))
            d = goal_distance(slug, obs)
            d_min, d_last = d, d
            for _ in range(horizon):
                key, sub = jax.random.split(key)
                a = np.asarray(agent.sample_actions(
                    observations=_mask_obs(obs), goals=ep_goal,
                    seed=sub, temperature=0))
                obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                ep_return += float(_r)
                saw_success = saw_success or ("success" in info)
                now = _flag(info.get("success", 0))
                ep_success = ep_success or now
                d = goal_distance(slug, obs)
                if d is not None:
                    d_min, d_last = min(d_min, d), d
                if _flag(term) or _flag(trunc):
                    break
            successes.append(float(ep_success))
            finals.append(float(now))
            returns.append(ep_return)
            if d_last is not None:
                dist_min.append(d_min)
                dist_final.append(d_last)
        out = {}
        if saw_success:
            out["success"] = float(np.mean(successes))
            out["success_final"] = float(np.mean(finals))
        if is_dmc:
            out["episode_return"] = float(np.mean(returns))
        if dist_min:
            out["goal_dist_min"] = float(np.mean(dist_min))
            out["goal_dist_final"] = float(np.mean(dist_final))
        # Per-episode values, written to eval_episodes.jsonl
        per_ep = {}
        if is_dmc:
            per_ep["episode_return"] = [float(x) for x in returns]
        if dist_min:
            per_ep.update({"goal_dist_min": [float(x) for x in dist_min],
                           "goal_dist_final": [float(x) for x in dist_final],
                           "success": [float(x) for x in successes]})
        if per_ep:
            out["_per_episode"] = per_ep
        return out, []
