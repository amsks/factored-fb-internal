"""Unified adapter — ExoRL (dm_control), MetaWorld and ManiSkill datasets and their envs
behind the DomainAdapter interface.

ExoRL (`dmc-` tasks) differs in kind from the other two: its tasks are reward functions, not
goals, so its data comes from env/exorl_env.py (which relabels rewards from the stored
mujoco physics) and its eval metric is `episode_return` -- walker/cheetah have no binary
success rule.

MetaWorld/ManiSkill data is a packed per-task npz (`data_path`) or a directory of
per-episode files:
    <data_dir>/buffer/episode_XXXXXX_LLLLL.npz  with observation, action, next_observation,
                                                reward, terminal, timeout, success
Both are flattened into one Dataset with each episode's last step marked in `terminals`,
which the goal samplers use as trajectory boundaries.

Eval uses a fixed seed per episode (the same seed is the same initial state across
checkpoints and methods) and reports `success` (did the success flag ever fire) and
`success_final` (the flag on the last step).
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
    """Flatten a packed per-task npz (`data/<family>/<task>.npz`, the unified fields
    stacked as [N, T, ...]) into the same flat dict `load_unified` returns.

    `terminals` marks each episode's last step (the trajectory split for goal sampling),
    not the env's `terminal` flag, which for fixed-horizon collections never fires.
    """
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
    # The env's own per-step success flag: the 0/1 task reward the reward-inference eval
    # relabels with (`relabel_infer_batch`). No training code reads it.
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
        t[-1] = 1.0                       # episode boundary -> GCDataset trajectory split
        obs.append(z["observation"].astype(np.float32))
        act.append(np.clip(z["action"], -1.0, 1.0).astype(np.float32))
        nobs.append(z["next_observation"].astype(np.float32))
        rew.append(z["reward"].astype(np.float32).reshape(-1))
        term.append(t)
    return dict(observations=np.concatenate(obs), actions=np.concatenate(act),
                next_observations=np.concatenate(nobs), rewards=np.concatenate(rew),
                terminals=np.concatenate(term))


def load_domain_data(cfg):
    """Offline data for a domain, either layout: a PACKED per-task npz when the config names
    one (`data_path`, the HuggingFace export), else the per-episode collected directory
    (`data_dir`, defaulting to the registry's path for the task).

    ExoRL (`dmc-`) has its own layout, and its episode files' `reward` column is not the
    task reward, so env/exorl_env.py relabels from the stored mujoco `physics` states."""
    task = cfg.get("task") or ""
    if task.startswith("dmc-"):
        from env.exorl_env import load_domain_data as load_exorl_domain
        return load_exorl_domain(cfg)
    path = cfg.get("data_path")
    if path:
        return load_packed(path)
    return load_unified(cfg.get("data_dir") or registry.data_dir(cfg["task"]))


def make_task_env(task, reward_task=None):
    """-> (env, horizon). Obs is the raw observation, identical to the collected data.

    `reward_task` is ExoRL-only: which dm_control reward function the eval env scores. It
    must be the one the offline data was relabelled with (the domain yaml's `exorl_task`
    feeds both)."""
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


# Raw-observation layout per task, used by synthesised eval goals (`eval_goal_mode=synth`)
# and by per-episode goal-distance logging. Every slot indexes the RAW obs, before any
# obs_mask_dims/goal_mask_dims blanking.
#
#   target -- the episode's goal, written into the obs by the env at reset (the dims the
#             MW/MS domain configs mask out of the observation)
#   object -- the achieved goal the env's success rule compares against `target`
#   copy   -- slots overwritten with `target` when synthesising a goal FRAME: the end
#             effector and the object, plus their prev-frame copies where the layout carries
#             them.
#   grasp_offset -- (pick-cube only) which `copy` slot must NOT collapse onto `target`: real
#             success frames never have tcp == obj (grasp offset mean/median 0.021/0.018 --
#             the same scale as the ~0.02-0.025 success radius), so this slot instead keeps
#             the base frame's own tcp-to-obj offset, added onto `target`. See synth_goal_frame.
#   derived_offset -- (pick-cube only) tcp_to_obj_pos (obj - tcp): 0 under the old collapse,
#             but once grasp_offset is preserved this channel is no longer algebraically zero
#             and is NOT one of the run's masked dims, so it must be recomputed rather than
#             zeroed, or it would contradict the rest of the synthesised frame.
#   zero   -- difference channels that stay algebraically zero once every OTHER `copy` slot
#             holds `target` (ManiSkill's obj_to_goal_pos).
#   one    -- flags that every achieved-goal frame carries. PickCube's success rule requires
#             a grasp and is_grasped == 1.0 on 100% of success frames in the collected data.
#   ndim   -- components of (object - target) the success rule measures. PushCube scores the
#             planar XY distance to its goal region; everything else is 3-D.
#
# ManiSkill slots are the ManiSkill 3 state layouts, confirmed against the packed upstream
# npz: PushCube (35) = qpos9 | qvel9 | tcp_pose7 | goal_pos3 | obj_pose7; PickCube (42) =
# qpos9 | qvel9 | is_grasped1 | tcp_pose7 | goal_pos3 | obj_pose7 | tcp_to_obj3 | obj_to_goal3.
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
    """-> the _GOAL_LAYOUT entry for a task slug, or None when the task has no goal-in-obs
    layout."""
    if task in _GOAL_LAYOUT:
        return _GOAL_LAYOUT[task]
    return _GOAL_LAYOUT.get(task.split("-", 1)[0])


def synth_goal_frame(task, raw_obs):
    """Per-episode goal FRAME built from this episode's own raw reset observation: a copy of
    it with the end effector and object moved onto the episode's true target.

    The alternative -- one goal frame cached from the dataset and reused for every episode
    (`eval_goal_mode=fixed`) -- conditions the critic on a (s, g) pair whose context dims
    disagree, since MW/MS redraw the target on every reset. Synthesising per episode keeps the context dims the episode's own, so the pair is
    consistent, and the eval goal is in-support by construction (the env drew the target from
    the same goal space the collector sampled).
    """
    lay = goal_layout(task)
    if lay is None:
        raise ValueError(f"eval_goal_mode=synth: no goal-in-obs layout registered for {task!r}")
    g = np.array(raw_obs, np.float32, copy=True)
    tgt = g[slice(*lay["target"])].copy()
    grasp = lay.get("grasp_offset")           # copy slot exempt from the target-collapse below
    obj0 = g[slice(*lay["object"])].copy()    # pre-overwrite object pos, to measure that offset
    grasp_vec = None
    for a, b in lay["copy"]:
        if grasp is not None and (a, b) == grasp:
            # Preserve THIS frame's own grasp geometry instead of collapsing tcp onto the
            # target -- tcp == obj never occurs in the data and sits at success-radius scale.
            # Degenerate base offsets (not actually a grasp) fall back to the dataset-median
            # grasp: straight above the object.
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
    derived = lay.get("derived_offset")       # e.g. tcp_to_obj_pos = obj - tcp, no longer 0
    if derived is not None and grasp_vec is not None:
        a, b = derived
        g[a:b] = -grasp_vec
    return g


def goal_distance(task, raw_obs):
    """‖object - target‖ read off the raw observation, under the env's own success metric.
    Returns None for tasks with no goal-in-obs layout."""
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
    """Seed the global np.random for the duration of the block, then put the previous state
    back. Metaworld/ManiSkill draw their reset randomness from the global RNG and give us no
    way to inject a Generator, so seeding it is the only way to fix an eval initial state.
    Restoring afterwards keeps eval from perturbing training: the datasets draw every batch
    index from the same global RNG, so without the restore every seed would replay one
    identical batch stream after each eval."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def seeded_reset(env, task, seed):
    """A given seed => a fixed initial state wherever the backend allows it. Metaworld draws
    its rand_vec from the global np.random; ManiSkill takes reset(seed=) where supported."""
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
    """A run-config value stashed by make_env_and_datasets, or `default` when the adapter is
    used without it (eval-only harnesses build the adapter and call evaluate directly)."""
    return (getattr(adapter, "_cfg", None) or {}).get(key, default)


class UnifiedAdapter:
    name = "unified"

    def make_env_and_datasets(self, cfg):
        self._cfg = dict(cfg)              # read by relabel_infer_batch (masks, reward mode)
        data = load_domain_data(cfg)
        env, horizon = make_task_env(cfg["task"], reward_task=cfg.get("exorl_task"))
        self._horizon = horizon
        # The eval goal for goal-conditioned critics (see `evaluate`), cached here so eval
        # never re-reads the dataset -- for a packed npz that reload costs tens of seconds.
        self._goal = data["next_observations"][int(np.argmax(data["rewards"]))].astype(np.float32)
        n = len(data["observations"])
        # Hold out the tail as validation. The split lands mid-episode only if val_frac is
        # tiny; terminals still mark every boundary inside each side.
        n_val = int(n * float(cfg.get("val_frac", 0.05)))
        if n_val > 0:
            cut = n - n_val
            tr = {k: v[:cut] for k, v in data.items()}
            va = {k: v[cut:] for k, v in data.items()}
            tr["terminals"][-1] = 1.0     # the goal samplers assert the last index is terminal
            va["terminals"][-1] = 1.0
        else:
            tr, va = data, data
        return env, FBDataset.create(**tr), FBDataset.create(**va)

    def eval_tasks(self, env, cfg):
        return [None]                      # single task per domain

    def relabel_infer_batch(self, train_ds, env, task_id, num_samples, shift_reward=1.0):
        """(next_observations, rewards) for `agent.infer_eval_cond` -- the reward-inference
        eval (eval_relabel.csv), mirroring env/ogbench.py::OGBenchAdapter.relabel_infer_batch:
        `num_samples` random transitions from the train dataset, each labelled with the task
        reward of its next state.

        `eval_relabel_reward` picks the task signal (default `success`):
          success -- the env's per-step success flag, mapped to the OGBench convention
                     (success - 1) + shift_reward, so at shift_reward=1.0 the reward is a 0/1
                     reached-the-goal indicator, as the OGBench cube relabel produces;
          dense   -- the stored per-step env reward + shift_reward (MetaWorld's shaped
                     reward in [0, 10] per raw step, summed over action repeat 2; ManiSkill's
                     normalized dense reward), a different, shaped task specification.
        Observations are blanked with obs_mask_dims | goal_mask_dims, the same union the
        datasets apply in training, so the rows handed to B / phi live in the space those
        nets were fitted on. `task_id` is ignored: a unified domain is one task."""
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
        # `any_step_success` is accepted for signature parity with OGBenchAdapter.evaluate:
        # this evaluator always reports any-step `success` and last-step `success_final`.
        if num_video:
            raise ValueError("video rendering is not supported; pass num_video=0")
        import jax
        slug = cfg["task"]
        horizon = episode_horizon(self)
        # ExoRL is reward-specified: undiscounted episode return is the benchmark metric.
        is_dmc = slug.startswith("dmc-")
        # Goal-conditioned critics, and z-critics with a map_eval_cond hook, need an explicit
        # goal frame; an agent rolling an inferred task vector must not get one. The rule is
        # env/base.py::needs_goal_frame.
        needs_goal = needs_goal_frame(agent)

        # Observation masking must match training, or the policy sees inputs it never saw --
        # and on goal-observable envs would get back the target it was denied in training.
        obs_mask = list(cfg.get("obs_mask_dims") or ())
        goal_mask = list(cfg.get("goal_mask_dims") or ())

        def _mask_obs(o):
            if not obs_mask:
                return np.asarray(o, np.float32)
            o = np.array(o, np.float32, copy=True)
            o[..., obs_mask] = 0.0
            return o

        def _prep_goal(g):
            """Blank a raw goal frame the way training goals are blanked: goal_mask_dims,
            then obs_mask_dims (the goal frame is an observation too)."""
            g = np.array(g, np.float32, copy=True)
            if goal_mask:
                g[goal_mask] = 0.0
            if obs_mask:
                g[obs_mask] = 0.0
            return g

        # `fixed` (default): one goal frame, the highest-reward next_obs in the offline data,
        # reused for every episode. `synth`: rebuilt from each episode's own reset obs.
        goal_mode = str(cfg.get("eval_goal_mode", "fixed"))
        if goal_mode not in ("fixed", "synth"):
            raise ValueError(f"eval_goal_mode must be 'fixed' or 'synth', got {goal_mode!r}")
        # Only checked when a goal frame is needed: for other agents the knob is inert.
        if needs_goal and goal_mode == "synth" and goal_layout(slug) is None:
            raise ValueError(
                f"eval_goal_mode=synth is not available for task {slug!r}: no goal-in-obs "
                f"layout is registered for it in env/unified.py::_GOAL_LAYOUT")

        goal = None
        if needs_goal and goal_mode == "fixed":
            goal = getattr(self, "_goal", None)
            if goal is None:                      # adapter used for eval only
                d = load_domain_data(cfg)
                goal = d["next_observations"][int(np.argmax(d["rewards"]))].astype(np.float32)
            goal = _prep_goal(goal)

        successes, finals, returns = [], [], []
        # Domains with no binary success rule (ExoRL walker/cheetah) emit no `success` key;
        # the row then carries no success column rather than a fabricated 0.0.
        saw_success = False
        dist_min, dist_final = [], []           # per-episode goal distances
        # Episode seeds are the loop index, so every checkpoint is scored on the same initial
        # states. `eval_seed_offset` shifts the block to a disjoint set of initial states.
        seed_off = int(cfg.get("eval_seed_offset", 0) or 0)
        for i in range(num_episodes):
            obs, info = seeded_reset(env, slug, i + seed_off)
            # Build the goal from the raw reset obs, before _mask_obs blanks the target dims.
            ep_goal = goal
            if needs_goal and goal_mode == "synth":
                ep_goal = _prep_goal(synth_goal_frame(slug, obs))
            key = jax.random.PRNGKey(i + seed_off)      # action noise follows the same shift
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
        # Per-episode values; impls/main.py moves them to eval_episodes.jsonl. The DMC returns
        # are kept because the mean hides bimodal outcomes (rare total failures).
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
