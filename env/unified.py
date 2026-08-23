"""Unified adapter — our collected per-episode datasets (Box2D / Metaworld / ManiSkill)
and their envs, plus the ExoRL/dm_control family, behind the DomainAdapter seam.

ExoRL (`dmc-` tasks) rides this adapter but differs in kind from the other three: its tasks
are REWARD functions, not goals, so its data comes from env/exorl_env.py (which relabels
rewards off the stored mujoco physics) and its eval metric is `episode_return` rather than
`success` — walker/cheetah have no binary success rule at all.

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
import contextlib
import glob
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

from impls.utils.datasets import FBDataset
from env import registry
from env.base import needs_goal_frame

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


def load_packed(npz_path):
    """Flatten a PACKED per-task npz -- the HuggingFace hosting layout written by
    hf_dataset/pack_and_manifest.py, `data/<family>/<task>.npz` with the 8 unified fields
    stacked as [N, T, ...] -- into the same flat dict `load_unified` returns.

    Same contract: `terminals` marks each episode's LAST step (the GCDataset trajectory
    split), not the env's `terminal` flag, which for fixed-horizon collections never fires.
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

    return dict(observations=flat(obs), actions=np.clip(flat(act), -1.0, 1.0),
                next_observations=flat(nobs), rewards=flat(z["reward"]),
                terminals=flat(term))


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

    ExoRL (`dmc-`) has a THIRD layout and, more importantly, a different reward story: its
    episode files' `reward` column is not the task reward, so the loader relabels from the
    stored mujoco `physics` states. See env/exorl_env.py."""
    task = cfg.get("task") or ""
    if task.startswith("dmc-"):
        from env.exorl_env import load_domain_data as load_exorl_domain
        return load_exorl_domain(cfg)
    path = cfg.get("data_path")
    if path:
        return load_packed(path)
    return load_unified(cfg.get("data_dir") or registry.data_dir(cfg["task"]))


def make_task_env(task, render=False, reward_task=None):
    """-> (env, horizon). Obs is the RAW factored obs, identical to the collected data.

    `reward_task` is ExoRL-only: which dm_control reward function the eval env scores. It
    MUST be the same one the offline data was relabelled with (the domain yaml's
    `exorl_task` feeds both), or training and eval are on two different tasks."""
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
        from env.metaworld_env import make_metaworld_env    # newt's wrapper, torch-free
        return make_metaworld_env(task)
    if task.startswith("ms-"):
        from env.maniskill_env import make_maniskill_env   # self-contained, no newt backend
        return make_maniskill_env(task)
    if task.startswith("dmc-"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from env.exorl_env import make_exorl_env           # ExoRL / dm_control, reward-specified
        return make_exorl_env(task, reward_task=reward_task)
    raise ValueError(f"no env builder for task {task!r}")


# Raw-observation layout per task, used by synth eval goals (`eval_goal_mode=synth`) and by
# per-episode goal-distance logging. Every slot indexes the RAW obs, before any
# obs_mask_dims/goal_mask_dims blanking.
#
#   target -- the episode's goal, written into the obs by the env at reset (the dims the
#             MW/MS domain configs mask out of the observation)
#   object -- the achieved goal the env's success rule compares against `target`
#   copy   -- slots overwritten with `target` when synthesising a goal FRAME: the end
#             effector and the object, plus their prev-frame copies where the layout carries
#             them. Metaworld's list is bon_probe.py's verified construction, unchanged.
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
    """-> the _GOAL_LAYOUT entry for a task slug, or None when the family has no goal-in-obs
    layout (Box2D/rlmini, whose targets are not carried in the observation)."""
    if task in _GOAL_LAYOUT:
        return _GOAL_LAYOUT[task]
    return _GOAL_LAYOUT.get(task.split("-", 1)[0])


def synth_goal_frame(task, raw_obs):
    """Per-episode goal FRAME built from this episode's own raw reset observation: a copy of
    it with the end effector and object moved onto the episode's true target.

    This is the repair for C4. The alternative -- one goal frame cached from the dataset and
    reused for every episode (`eval_goal_mode=fixed`) -- conditions a goal-conditioned critic
    on a (s, g) pair whose context dims disagree, while MW/MS redraw the target on every
    reset. Synthesising per episode keeps the context dims the episode's own, so the pair is
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
    """‖object - target‖ read straight off the RAW observation, under the env's own success
    metric (E0.2). Returns None for families with no goal-in-obs layout."""
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
    Restoring afterwards is what keeps eval from perturbing training: GCDataset draws every
    batch index, goal index and relabel coin from that same global RNG (impls/utils/datasets.py),
    so without the restore the post-eval state is a pure function of the last episode index --
    every seed then replays one identical batch stream, periodic in eval_interval, and the
    reported seed-to-seed sd stops being a seed estimate."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def seeded_reset(env, task, seed):
    """A given seed => a fixed initial state wherever the backend allows it. rlmini honors
    reset(seed=); Metaworld draws its rand_vec from the GLOBAL np.random; ManiSkill's wrapper
    does not accept a seed, so it is best-effort."""
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


class UnifiedAdapter:
    name = "unified"

    def make_env_and_datasets(self, cfg):
        data = load_domain_data(cfg)
        env, horizon = make_task_env(cfg["task"], render=bool(cfg.get("render", True)),
                                     reward_task=cfg.get("exorl_task"))
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
        # ExoRL/dm_control is REWARD-specified: undiscounted episode return is the
        # benchmark's own metric, and for walker/cheetah it is the ONLY meaningful one
        # (no binary success rule exists, so `success` would be a constant 0.0).
        is_dmc = slug.startswith("dmc-")
        # z-critics condition on agent.cond_eval (goals=None); goal-conditioned critics need
        # an explicit goal, for which we use the highest-reward next_obs in the offline data.
        # A z-critic exposing map_eval_cond (fb) also needs the goal frame: the hook embeds
        # it into z at eval — without it the stored eval z is the zero vector (goal-blind).
        # An agent rolling an INFERRED conditioning (--eval_on_inferred_cond) needs none, and
        # must not be handed one. env/base.py::needs_goal_frame is the whole rule, and is the
        # same function env/online.py's collector calls.
        needs_goal = needs_goal_frame(agent)

        # Observation masking must match training (GCDataset obs_mask_dims) or the policy sees
        # a distribution it was never trained on -- and, on goal-observable envs, would get the
        # target handed back to it at eval after being denied it during training.
        obs_mask = list(cfg.get("obs_mask_dims") or ())
        goal_mask = list(cfg.get("goal_mask_dims") or ())

        def _mask_obs(o):
            if not obs_mask:
                return np.asarray(o, np.float32)
            o = np.array(o, np.float32, copy=True)
            o[..., obs_mask] = 0.0
            return o

        def _prep_goal(g):
            """Put a raw goal frame into the space the critic's TRAINING goals live in:
            goal_mask_dims (GCDataset blanks the emitted goals) then obs_mask_dims (the goal
            frame is an observation too, and must be blanked the same way)."""
            g = np.array(g, np.float32, copy=True)
            if goal_mask:
                g[goal_mask] = 0.0
            if obs_mask:
                g[obs_mask] = 0.0
            return g

        # `fixed` (default) reproduces the historical behaviour: ONE goal frame, the
        # highest-reward next_obs in the offline data, reused for every episode. `synth`
        # rebuilds the goal from each episode's own reset obs -- see synth_goal_frame.
        goal_mode = str(cfg.get("eval_goal_mode", "fixed"))
        if goal_mode not in ("fixed", "synth"):
            raise ValueError(f"eval_goal_mode must be 'fixed' or 'synth', got {goal_mode!r}")
        # Only checked for goal-conditioned critics: a z-critic never reads a goal frame, so
        # the knob is inert for it and must not turn into a hard error on a task with no
        # goal-in-obs layout.
        if needs_goal and goal_mode == "synth" and goal_layout(slug) is None:
            raise ValueError(
                f"eval_goal_mode=synth is not available for task {slug!r}: no goal-in-obs "
                f"layout is registered for it in env/unified.py::_GOAL_LAYOUT")

        goal = None
        if needs_goal and goal_mode == "fixed":
            goal = getattr(self, "_goal", None)
            if goal is None:                      # adapter used for eval only (campaign/eval)
                d = load_domain_data(cfg)
                goal = d["next_observations"][int(np.argmax(d["rewards"]))].astype(np.float32)
            goal = _prep_goal(goal)

        successes, ticks, frames = [], [], []
        returns = []                                # dmc: undiscounted episode return
        # E0.2: per-episode goal distances, so E5's matched-radius re-scoring is an analysis
        # pass over existing logs rather than a re-run. Costs one norm per step.
        dist_min, dist_final = [], []
        # Metric episodes first, then num_video EXTRA rendered ones. Rendering costs ~7 ms a
        # frame and must not perturb the episodes we report, so the reported set is never
        # rendered — the same split OGBench's evaluate() uses.
        # Episode seeds are the loop index, so every checkpoint of every run is scored on the
        # SAME initial states — that is what makes two checkpoints comparable. It also means a
        # re-eval cannot draw a fresh sample: repeating it reproduces the number exactly, so a
        # 50-episode fluke is indistinguishable from real behaviour. `eval_seed_offset` shifts
        # the whole block to a disjoint set of initial states, which is the only way to ask
        # "was that 0.02 real, or was it the episode draw?" (edge-attribution spec W4).
        # Leave it 0 for anything that will be compared against existing numbers.
        seed_off = int(cfg.get("eval_seed_offset", 0) or 0)
        for i in range(num_episodes + num_video):
            record = i >= num_episodes
            obs, info = seeded_reset(env, slug, i + seed_off)
            # The goal must be built from the RAW reset obs, before _mask_obs blanks the very
            # target dims it is built from.
            ep_goal = synth_goal_frame(slug, obs) if (needs_goal and goal_mode == "synth") \
                else goal
            if needs_goal and goal_mode == "synth":
                ep_goal = _prep_goal(ep_goal)
            key = jax.random.PRNGKey(i + seed_off)      # action noise follows the same shift
            ep_success = _flag(info.get("success", 0))
            ep_return = 0.0
            at_goal = int(_flag(info.get("success", 0)))
            d = goal_distance(slug, obs)
            d_min, d_last = d, d
            ff = [env.render()] if record else None
            for _ in range(horizon):
                key, sub = jax.random.split(key)
                a = np.asarray(agent.sample_actions(
                    observations=_mask_obs(obs), goals=ep_goal,
                    seed=sub, temperature=0))
                obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                ep_return += float(_r)
                now = _flag(info.get("success", 0))
                ep_success = ep_success or now
                at_goal += int(now)
                d = goal_distance(slug, obs)
                if d is not None:
                    d_min, d_last = min(d_min, d), d
                if record:
                    ff.append(env.render())
                if _flag(term) or _flag(trunc):
                    break
            if record:
                frames.append(np.array(ff))
            else:
                successes.append(float(ep_success))
                returns.append(ep_return)
                ticks.append(at_goal)                   # integer steps spent at goal
                if d_last is not None:
                    dist_min.append(d_min)
                    dist_final.append(d_last)
        out = {"success": float(np.mean(successes))}
        if is_dmc:                                  # the benchmark metric for ExoRL
            out["episode_return"] = float(np.mean(returns))
        if is_box2d:                                    # at-goal ticks: Box2D only, integer scale
            out["at_goal_ticks"] = float(np.mean(ticks))
        if dist_min:
            out["goal_dist_min"] = float(np.mean(dist_min))
            out["goal_dist_final"] = float(np.mean(dist_final))
            # The per-episode arrays are what E5 re-scores at a sweep of radii; the means
            # above are only the at-a-glance summary that fits in eval.csv. impls/main.py
            # pops this key out of the row before logging and writes it to its own file.
            out["_per_episode"] = {"goal_dist_min": [float(x) for x in dist_min],
                                   "goal_dist_final": [float(x) for x in dist_final],
                                   "success": [float(x) for x in successes]}
        return out, frames
