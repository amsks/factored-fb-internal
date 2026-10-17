"""ExoRL (dm_control) envs and datasets.

Rewards are relabeled for the requested task from the stored physics states, as in ExoRL;
the stored reward column is not used. Episode files hold T+1 rows with a dummy first action.
"""
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

# Import numpy before any mujoco overlay is put on sys.path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASKS_ROOT = str(_REPO_ROOT)

# Default ExoRL exploration buffer
DEFAULT_ALGO = "rnd"

DATA_ROOT = os.environ.get("EXORL_DATA_ROOT", os.path.expanduser("~/exorl_data/exorl/datasets"))
# Relabeled-reward cache
CACHE_ROOT = os.environ.get("EXORL_CACHE_ROOT", os.path.expanduser("~/exorl_data/relabel_cache"))

EPISODE_LEN = 1000

# task slug -> (dm_control domain, default reward task, all reward tasks)
TASKS = {
    "dmc-point-mass-maze": ("point_mass_maze", "reach_top_left",
                            ("reach_top_left", "reach_top_right",
                             "reach_bottom_left", "reach_bottom_right")),
    "dmc-walker":          ("walker", "walk", ("stand", "walk", "run", "flip", "spin")),
    "dmc-cheetah":         ("cheetah", "run", ("run", "run_backward", "walk",
                                               "walk_backward", "flip", "flip_backward")),
    "dmc-quadruped":       ("quadruped", "walk", ("stand", "walk", "run", "jump")),
}


def _overlay_candidates():
    """site-packages directories that may hold a usable mujoco, in probe order."""
    env = os.environ.get("FB_MUJOCO_SITE_PACKAGES")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    out = []
    for root in [_REPO_ROOT, *_REPO_ROOT.parents][:6]:
        out.extend(sorted(str(p) for p in root.glob(".venv*/lib/python*/site-packages")
                          if (p / "mujoco").is_dir()))
    return out


def domain_of(task):
    """Return the dm_control domain for a `dmc-*` task slug."""
    if task not in TASKS:
        raise ValueError(f"unknown ExoRL task {task!r} (known: {sorted(TASKS)})")
    return TASKS[task][0]


def reward_tasks(task):
    return TASKS[task][2]


def default_reward_task(task):
    return TASKS[task][1]


def resolve_reward_task(task, reward_task=None):
    """Validate the reward task, or return the domain's default."""
    known = reward_tasks(task)
    if reward_task is None:
        return default_reward_task(task)
    if reward_task not in known:
        raise ValueError(f"unknown reward task {reward_task!r} for {task!r} (known: {known})")
    return reward_task


def _purge(*prefixes):
    for name in [k for k in sys.modules
                 if any(k == pre or k.startswith(pre + ".") for pre in prefixes)]:
        del sys.modules[name]


def _dm_control_works():
    """Whether dm_control can compile a model with the current mujoco."""
    try:
        from dm_control import mujoco as dm_mujoco
        dm_mujoco.Physics.from_xml_string("<mujoco/>")
        return True
    except Exception:                                              # noqa: BLE001
        return False


def _mujoco_overlay():
    """Import a mujoco that dm_control can use, from another site-packages if needed.

    Set FB_MUJOCO_SITE_PACKAGES to choose the directories. No-op if the current one works.
    """
    if _dm_control_works():
        return
    tried = []
    for sp in _overlay_candidates():
        if not os.path.isdir(sp):
            continue
        _purge("mujoco", "dm_control")
        sys.path.insert(0, sp)
        try:
            import mujoco as _mj
            ver = getattr(_mj, "__version__", "?")
            ok = _dm_control_works()
        except Exception as e:                                     # noqa: BLE001
            ver, ok = f"import failed: {type(e).__name__}", False
        finally:
            if sp in sys.path:
                sys.path.remove(sp)
        if ok:
            return
        tried.append(f"{sp} (mujoco {ver})")
    _purge("mujoco", "dm_control")
    raise RuntimeError(
        "no usable mujoco/dm_control pairing found: the running interpreter's own mujoco "
        "cannot compile a dm_control model, and no overlay site-packages could "
        f"(tried: {tried or 'none found'}). Set FB_MUJOCO_SITE_PACKAGES (colon-separated) "
        "to a site-packages directory whose mujoco matches this dm_control, or install a "
        "matching pair in the running venv.")


def _load_dmc(domain, reward_task, seed=0):
    """Build the dm_control env, preferring suite tasks over ExoRL custom tasks."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    _mujoco_overlay()
    if _TASKS_ROOT not in sys.path:
        sys.path.insert(0, _TASKS_ROOT)
    from dm_control import suite
    if (domain, reward_task) in suite.ALL_TASKS:
        return suite.load(domain, reward_task, task_kwargs={"random": seed})
    from env import exorl_dmc_tasks
    return exorl_dmc_tasks.make(domain, reward_task, task_kwargs={"random": seed})


def _flat_obs(observation):
    """Concatenate observation entries in emission order, as ExoRL stores them."""
    return np.concatenate([np.asarray(v, np.float64).ravel()
                           for v in observation.values()]).astype(np.float32)


class ExoRLWrapper:
    """Gym-style wrapper over a dm_control env.

    info["success"] is reported only for point_mass_maze (target reached at any step).
    """

    def __init__(self, env, domain, reward_task, seed=0):
        self.env = env
        self.domain = domain
        self.reward_task = reward_task
        self._seed = seed
        self._target = None
        self._success = False
        if domain == "point_mass_maze":
            if _TASKS_ROOT not in sys.path:
                sys.path.insert(0, _TASKS_ROOT)
            from env.exorl_dmc_tasks import point_mass_maze as _pmm
            self._target = np.asarray(
                dict(_pmm.TASKS)[reward_task][:2], np.float64)
            # Maze target size
            self._eps = 0.03

    def _check(self, obs):
        if self._target is not None and float(
                np.linalg.norm(np.asarray(obs[:2], np.float64) - self._target)) <= self._eps:
            self._success = True

    def _info(self, extra=None):
        info = dict(extra or {})
        if self._target is not None:
            info["success"] = float(self._success)
        return info

    def reset(self, seed=None, **_kw):
        """Reset to the initial state fixed by `seed`.

        Seeds both the task RNG and the global np.random, then restores the global state.
        """
        if seed is not None:
            self._seed = int(seed)
        state = np.random.get_state()
        try:
            np.random.seed(self._seed)
            task_rng = getattr(self.env, "task", None)
            task_rng = getattr(task_rng, "_random", None)
            if task_rng is not None and hasattr(task_rng, "seed"):
                task_rng.seed(self._seed)
            self._success = False
            ts = self.env.reset()
        finally:
            np.random.set_state(state)
        obs = _flat_obs(ts.observation)
        self._check(obs)
        return obs, self._info()

    def step(self, action):
        ts = self.env.step(np.clip(np.asarray(action, np.float64), -1.0, 1.0))
        obs = _flat_obs(ts.observation)
        self._check(obs)
        reward = float(np.asarray(ts.reward if ts.reward is not None else 0.0).mean())
        # Episodes end only at the time limit
        return obs, reward, False, bool(ts.last()), self._info({"reward": reward})

    def render(self, *_a, **_k):
        try:
            return np.asarray(self.env.physics.render(height=64, width=64, camera_id=0))
        except Exception:                                          # noqa: BLE001
            return np.zeros((64, 64, 3), np.uint8)

    def close(self):
        try:
            self.env.close()
        except Exception:                                          # noqa: BLE001
            pass


def make_exorl_env(task, reward_task=None, seed=0):
    """Return (env, horizon) for a `dmc-*` task slug scored by `reward_task`."""
    domain = domain_of(task)
    rt = resolve_reward_task(task, reward_task)
    return ExoRLWrapper(_load_dmc(domain, rt, seed=seed), domain, rt, seed=seed), EPISODE_LEN


# Dataset

def _ep_index(p):
    """Episode index from `episode_<idx>_<length>.npz`."""
    nums = re.findall(r"\d+", Path(p).stem)
    if not nums:
        raise ValueError(f"cannot read an episode index out of {p!r}")
    return int(nums[0])


def episode_files(buffer_dir, max_episodes=None):
    eps = sorted(glob.glob(os.path.join(str(buffer_dir), "episode_*.npz")), key=_ep_index)
    if not eps:
        raise FileNotFoundError(f"no ExoRL episode .npz under {buffer_dir!r}")
    return eps[:max_episodes] if max_episodes else eps


def read_episode(path):
    """Read one episode file.

    :return: (observations, actions, next_observations) for the T transitions, and all
        T+1 physics states.
    """
    with np.load(path) as f:
        obs = np.asarray(f["observation"])
        act = np.asarray(f["action"])
        phys = np.asarray(f["physics"]) if "physics" in f.files else None
    assert obs.shape[0] == act.shape[0], (path, obs.shape, act.shape)
    n_declared = _ep_index_len(path)
    n = obs.shape[0] - 1
    if n_declared is not None and n != n_declared:
        raise ValueError(f"{path}: filename declares {n_declared} transitions but the "
                         f"observation array implies {n} (shape {obs.shape})")
    return (obs[:-1].astype(np.float32), act[1:].astype(np.float32),
            obs[1:].astype(np.float32), None if phys is None else phys)


def _ep_index_len(path):
    parts = Path(path).stem.split("_")
    return int(parts[-1]) if parts and parts[-1].isdigit() else None


def relabel_rewards(physics, domain, reward_task, env=None):
    """Task reward of each stored physics state, [T+1]; control is left at zero as in ExoRL."""
    if env is None:
        env = _load_dmc(domain, reward_task)
    physics = np.asarray(physics)
    out = np.empty(physics.shape[0], np.float32)
    for i in range(physics.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(physics[i])
        out[i] = float(np.asarray(env.task.get_reward(env.physics)).mean())
    return out


def _cache_key(files, domain, reward_task, algo):
    h = hashlib.sha1()
    h.update(f"{domain}|{reward_task}|{algo}|{len(files)}".encode())
    h.update(("|".join(Path(f).name for f in files[:8] + files[-8:])).encode())
    return h.hexdigest()[:16]


def load_exorl(buffer_dir, task, reward_task=None, algo=DEFAULT_ALGO, max_episodes=None,
               cache_dir=None, use_cache=True):
    """Flatten an ExoRL buffer into a dataset dict with relabeled rewards.

    `rewards` is the task reward at the next state; `terminals` marks each episode's last
    row. Relabeled rewards are cached under `cache_dir`.
    """
    domain = domain_of(task)
    rt = resolve_reward_task(task, reward_task)
    files = episode_files(buffer_dir, max_episodes)

    cache_dir = cache_dir or CACHE_ROOT
    cache_path = os.path.join(cache_dir, f"{domain}_{algo}_{rt}_{_cache_key(files, domain, rt, algo)}.npy")
    cached = None
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)

    obs, act, nobs, rew, term = [], [], [], [], []
    env = None
    off = 0
    for fn in files:
        o, a, no, phys = read_episode(fn)
        T = o.shape[0]
        if cached is not None:
            r = cached[off:off + T]
            if r.shape[0] != T:
                raise ValueError(f"reward cache {cache_path!r} is stale for {buffer_dir!r} "
                                 f"(expected {T} rows at offset {off}, got {r.shape[0]}); "
                                 f"delete it and re-run")
        else:
            if phys is None:
                raise ValueError(
                    f"{fn} has no 'physics' array, so its rewards cannot be relabelled for "
                    f"{rt!r}. The stored 'reward' column is the reward of the task the data "
                    f"was COLLECTED under, not of this one (see this module's docstring), "
                    f"so it is deliberately not used as a fallback: it would be silently "
                    f"right on one task per domain and silently wrong on the rest.")
            if env is None:
                env = _load_dmc(domain, rt)
            r = relabel_rewards(phys, domain, rt, env=env)[1:]      # reward at the next state
        t = np.zeros(T, np.float32)
        t[-1] = 1.0
        obs.append(o)
        act.append(np.clip(a, -1.0, 1.0))
        nobs.append(no)
        rew.append(np.asarray(r, np.float32).reshape(-1))
        term.append(t)
        off += T

    data = dict(observations=np.concatenate(obs), actions=np.concatenate(act),
                next_observations=np.concatenate(nobs),
                rewards=np.concatenate(rew), terminals=np.concatenate(term))
    if cached is None and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, data["rewards"])
    return data


def load_domain_data(cfg):
    """Load offline ExoRL data for a domain config."""
    task = cfg["task"]
    algo = str(cfg.get("exorl_algo") or DEFAULT_ALGO)
    d = cfg.get("data_dir") or os.path.join(DATA_ROOT, domain_of(task), algo)
    d = os.path.expanduser(str(d))
    if os.path.isdir(os.path.join(d, "buffer")):
        d = os.path.join(d, "buffer")
    return load_exorl(d, task, reward_task=cfg.get("exorl_task"), algo=algo,
                      max_episodes=cfg.get("exorl_max_episodes"),
                      cache_dir=cfg.get("exorl_cache_dir"))
