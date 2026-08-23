"""env/exorl_env.py — the ExoRL (dm_control) family behind the same backend surface as
`env/metaworld_env.py` / `env/maniskill_env.py`.

WHY THIS FAMILY EXISTS HERE. Every other domain in this tree is *goal-specified*: the task
is "reach this observation", and `GCDataset` synthesises the reward from goal proximity.
ExoRL tasks are *reward-specified* — `walker run`, `cheetah run_backward`,
`point_mass_maze reach_top_left` are dense reward functions, not goals. That is precisely
what the successor-measure family (`density_fb` with `measure_density_z=true`, `fb`,
`td_fb`, `psm`, `rldp`) claims to handle zero-shot, via
`Critic.infer_cond(net, obs, rewards)` -> a task vector z. No goal-conditioned critic in
this repo (gciql / gcivl / hiql / iql_bilinear / ...) has a reward interface at all and
none of them can run these domains. This backend is what makes the z-from-rewards claim
testable on something that was never a goal-reaching problem.

THE REWARD TRAP (read before touching the loader). ExoRL's episode buffers DO carry a
`reward` array, and it is the reward of the task the data was COLLECTED under — one task,
fixed at collection time — not of the task you are asking about. On the downloaded
`point_mass_maze/rnd` buffers it is shape `(T+1, 4)`: four identical copies of the
`reach_top_left` reward (measured 2026-08-22; corr 0.9994 against this module's
relabelling for that target, the residual being the `small_control` factor, and max 0.87
over 30 episodes, so it is a real signal — just the wrong one for three of the four
targets). Passing it through would therefore be silently right on exactly one task per
domain and silently wrong on every other, with nothing failing anywhere:
`impls/main.py::infer_cond_noop_reason` only flags an ALL-ZERO reward vector, and this
column is not zero.

ExoRL's own offline protocol never reads that column at all —
`exorl/utils.py::relabel_episode` replays each stored `physics` state into the *target
task's* env and calls `task.get_reward(physics)`. `relabel_rewards` below reproduces that
exactly, which is also what makes numbers from this tree comparable with published ExoRL
results.

DATA LAYOUT. ExoRL ships `<root>/<domain>/<algo>/buffer/episode_<idx>_<T>.npz` with

    observation [T+1, obs_dim]   -- observation[0] is the true reset obs
    action      [T+1, act_dim]   -- action[0] is a DUMMY zero padding row
    reward      [T+1, ...]       -- see above; not used
    discount    [T+1, 1]
    physics     [T+1, state_dim] -- mujoco qpos|qvel, the relabelling input

so the T real transitions are `(observation[t], action[t+1], observation[t+1])` for
t = 0..T-1 (the "dummy first transition" convention of `exorl/replay_buffer.py`). We emit
`next_observations` explicitly rather than inferring them at sample time.

MUJOCO OVERLAY. `/dev/shm/.venv-jax` -- the interpreter everything JAX in this repo runs
under -- has a `mujoco` its own `dm_control` cannot drive (`MjModel has no attribute
'B_colind'` on any `Physics.from_xml_string`). Upgrading a shared venv with live training
on it was rejected, so `_mujoco_overlay()` finds a site-packages directory whose mujoco
dm_control CAN drive, imports under it, and drops the path again. It probes by compiling a
model rather than comparing versions -- there is a ceiling as well as a floor, and both
neighbouring venvs sit on the wrong side of one of them. Override with
`FB_MUJOCO_SITE_PACKAGES` (colon-separated). Nothing is installed; nothing on disk changes.
"""
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

# Import position is load-bearing: numpy must resolve and cache from THIS interpreter's own
# site-packages before _mujoco_overlay() ever prepends another one to sys.path, or a later
# import could pull numpy from the overlay venv too. (Verified empirically: with numpy
# imported first, only the mujoco extension comes from the overlay.)
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# exorl_dmc_tasks lives under env/ now; the repo root is what must be importable.
_TASKS_ROOT = str(_REPO_ROOT)

# ExoRL exploration algorithms whose buffers are published. `rnd` is the T3 default.
DEFAULT_ALGO = "rnd"

# Where the raw ExoRL `<domain>/<algo>/buffer` trees live. Overridable so a box that keeps
# them elsewhere needs no code edit.
DATA_ROOT = os.environ.get("EXORL_DATA_ROOT", os.path.expanduser("~/exorl_data/exorl/datasets"))
# Relabelled-reward cache (one .npy per domain x algo x task x episode-set).
CACHE_ROOT = os.environ.get("EXORL_CACHE_ROOT", os.path.expanduser("~/exorl_data/relabel_cache"))

EPISODE_LEN = 1000  # every T3 domain, action_repeat 1. Measured, not assumed: each
                    # env truncates after exactly 1000 steps, and the point_mass_maze
                    # RND buffers are `episode_<idx>_1000.npz`.

# task slug -> (dm_control domain, default reward task, all scored reward tasks).
# The task lists are ExoRL's own benchmark tasks for each domain (README table); the
# default is the one a domain yaml gets if it names no `exorl_task`.
TASKS = {
    "dmc-point-mass-maze": ("point_mass_maze", "reach_top_left",
                            ("reach_top_left", "reach_top_right",
                             "reach_bottom_left", "reach_bottom_right")),
    "dmc-walker":          ("walker", "walk", ("stand", "walk", "run", "flip")),
    # cheetah: `run` is the dm_control suite task; the other three are ExoRL additions
    # (env/exorl_dmc_tasks/cheetah.py). There is no `walk` for cheetah upstream.
    "dmc-cheetah":         ("cheetah", "run", ("run", "run_backward", "flip",
                                               "flip_backward")),
}


def _overlay_candidates():
    """site-packages directories that might hold a usable mujoco, in probe order.

    `FB_MUJOCO_SITE_PACKAGES` (colon-separated, like PATH) wins outright. Otherwise we look
    for a `.venv*` beside the repo AND beside each of its ancestors -- a git worktree lives
    at `<checkout>/.claude/worktrees/<id>` and has no `.venv` of its own, so searching only
    `_REPO_ROOT` would fail on exactly the trees this gets developed in."""
    env = os.environ.get("FB_MUJOCO_SITE_PACKAGES")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    out = []
    for root in [_REPO_ROOT, *_REPO_ROOT.parents][:6]:
        out.extend(sorted(str(p) for p in root.glob(".venv*/lib/python*/site-packages")
                          if (p / "mujoco").is_dir()))
    return out


def domain_of(task):
    """-> the dm_control domain name for a `dmc-*` task slug."""
    if task not in TASKS:
        raise ValueError(f"unknown ExoRL task {task!r} (known: {sorted(TASKS)})")
    return TASKS[task][0]


def reward_tasks(task):
    return TASKS[task][2]


def default_reward_task(task):
    return TASKS[task][1]


def resolve_reward_task(task, reward_task=None):
    """Validate + default the reward task for a domain slug. A typo must fail loudly: a
    silently-defaulted reward task would relabel the whole dataset for the WRONG task and
    every downstream number would be a number for a task nobody asked about."""
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
    """Can dm_control build ANY model against the currently-resolved mujoco?

    A version floor is not the right test and quietly picks the wrong venv. dm_control's
    `index.struct_indexer` walks a hardcoded list of `MjModel` field names, so it breaks
    against a mujoco that is too OLD (`'MjModel' object has no attribute 'B_colind'`,
    mujoco 3.3.0) AND against one that is too NEW (`... has no attribute
    'flex_bandwidth'`, mujoco 3.10.0) -- both of which are installed in venvs next to this
    repo. The only criterion that is actually true-or-false is "does a Physics compile",
    so that is what we test, on the smallest possible model."""
    try:
        from dm_control import mujoco as dm_mujoco
        dm_mujoco.Physics.from_xml_string("<mujoco/>")
        return True
    except Exception:                                              # noqa: BLE001
        return False


def _mujoco_overlay():
    """Make `import mujoco` resolve to a build dm_control can actually use, via a temporary
    sys.path overlay.

    Why an overlay at all: `/dev/shm/.venv-jax` (the interpreter this repo's JAX code runs
    under) has a mujoco that its own dm_control cannot drive. Upgrading it venv-wide was
    rejected -- it is a shared venv with live training on it. So we prepend a site-packages
    directory that has a workable mujoco, import under it, and drop the path again. numpy
    is imported at the top of this module, before any overlay, so it (and everything else)
    keeps resolving to the running venv; only the mujoco C-extension comes from elsewhere.
    Nothing is installed and nothing on disk changes.

    Idempotent, and a no-op when the running venv is already fine."""
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
    _purge("mujoco", "dm_control")                                 # leave no half-overlay
    raise RuntimeError(
        "no usable mujoco/dm_control pairing found: the running interpreter's own mujoco "
        "cannot compile a dm_control model, and no overlay site-packages could "
        f"(tried: {tried or 'none found'}). Set FB_MUJOCO_SITE_PACKAGES (colon-separated) "
        "to a site-packages directory whose mujoco matches this dm_control, or install a "
        "matching pair in the running venv.")


def _load_dmc(domain, reward_task, seed=0):
    """Build the raw dm_control `control.Environment` for (domain, reward_task).

    Resolution order is ExoRL's own (`exorl/dmc.py`): the dm_control suite first, its
    vendored custom tasks second. Getting this backwards would silently pick a different
    reward function for `walker walk` than the benchmark uses."""
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
    """Flatten a dm_control observation the way ExoRL's collector did.

    ExoRL builds its envs with `flat_observation=True`, which concatenates the task's
    OrderedDict in emission order; doing the same concatenation here is what makes the eval
    observation the SAME vector as the stored `observation` arrays (verified per domain:
    point_mass_maze 4, walker 24, cheetah 17). Order therefore comes from the task, not from
    a list here -- the three domains do not share one (walker emits orientations, height,
    velocity; the other two emit position, velocity)."""
    return np.concatenate([np.asarray(v, np.float64).ravel()
                           for v in observation.values()]).astype(np.float32)


class ExoRLWrapper:
    """Gym-like single-env numpy adapter over a dm_control `control.Environment`, matching
    the surface `env/unified.py::evaluate` consumes: `reset(seed=)` -> (obs, info) and
    `step(a)` -> (obs, reward, terminated, truncated, info).

    `info["reward"]` carries the task reward and `info["success"]` is present only where
    the domain HAS a binary success rule (point_mass_maze: within `GOAL_EPS` of the target
    at any point). For walker/cheetah there is no such rule — the benchmark metric is
    episode return — so no `success` key is emitted rather than a fabricated 0.0 that
    would read as "the policy failed" in eval.csv."""

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
            # MazeMultiGoal's `target_size` in the published zero-shot-RL maze protocol
            # (controllable_agent @ 5a9950b, url_benchmark/goals.py). Binary success is a
            # DERIVED metric reusing that full-credit radius; the benchmark itself scores
            # the dense reward, which is what `episode_return` reports.
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
        """A given seed => a fixed initial state, WITHOUT recompiling the model.

        Two RNGs have to be pinned, not one: dm_control tasks draw from `task.random` (the
        RandomState handed to the constructor), but `point_mass_maze`'s own
        `initialize_episode` reaches for the GLOBAL `np.random.uniform` for the start qpos
        (upstream quirk, env/exorl_dmc_tasks/point_mass_maze.py). Seeding only the
        first would leave that domain's initial state uncontrolled, i.e. two checkpoints
        scored on different start states. The global state is restored afterwards for the
        same reason env/unified.py::_global_np_seed restores it: the dataset classes draw
        every batch index from that same global RNG, and leaving it pinned would make the
        post-eval batch stream a pure function of the last episode index."""
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
        # dm_control episodes end only by time limit; `evaluate` runs the fixed horizon.
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
    """-> (env, horizon) for a `dmc-*` task slug. `reward_task` names WHICH reward function
    the env scores; it must match the one the dataset was relabelled with, or training and
    eval are on two different tasks."""
    domain = domain_of(task)
    rt = resolve_reward_task(task, reward_task)
    return ExoRLWrapper(_load_dmc(domain, rt, seed=seed), domain, rt, seed=seed), EPISODE_LEN


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

def _ep_index(p):
    """Sort key: the EPISODE index out of `episode_<idx>_<length>.npz`, not the length.

    Sorting on the wrong number (or lexicographically, once indices pass 6 digits) would
    reorder episodes, which silently changes both the train/val split boundary and which
    prefix `max_episodes` selects."""
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
    """-> (observations, actions, next_observations, physics) for one ExoRL episode file.

    The first three are the T REAL transitions under the dummy-first-transition convention:
    `observation[t]`, `action[t+1]`, `observation[t+1]` for t = 0..T-1. `physics` is
    returned UNSLICED, all T+1 stored states, because it is the relabeller's input and the
    caller chooses the alignment: `relabel_rewards(physics)[1:]` is the reward at s_{t+1}
    (what `load_exorl` emits) and `[:-1]` is the reward at s_t. The `[1:]` choice matters --
    the successor-measure family's index is solved against the NEXT-state marginal (see the
    "WHICH ROWS" paragraph in `impls/critics/density_fb.py::infer_cond`), so the reward has
    to be the reward of the state the measure is over.
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
    """ExoRL's `relabel_episode`, for ONE episode's stored physics states.

    `physics` is the raw `[T+1, state_dim]` array. Returns `[T+1]` float32 rewards, one per
    stored state; the caller slices `[1:]` to align with `next_observations` (reward at
    s_{t+1}) or `[:-1]` to align with `observations`.

    Faithfulness note: upstream sets ONLY the physics state, so `physics.control()` is
    whatever `reset_context` leaves it (zeros). For `point_mass_maze` that makes the
    `small_control` factor exactly 1.0, i.e. the relabelled reward is the pure
    `near_target` term. We reproduce upstream rather than "fixing" it: a different control
    term would make our numbers incomparable with every published ExoRL result."""
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
    """Flatten an ExoRL `<domain>/<algo>/buffer` tree into the flat dataset dict
    `env/unified.py` builds an `FBDataset` from.

    Returns keys `observations, actions, next_observations, rewards, terminals` with
    `rewards` the RELABELLED task reward at the NEXT state (see `read_episode`), and
    `terminals` marking each episode's LAST row -- the trajectory split every dataset class
    in `impls/utils/datasets.py` reads, not an absorbing-state flag (dm_control episodes end
    on the time limit and never terminate).

    Relabelling is one mujoco `set_state` + `get_reward` per stored state, so it is cached
    to `<cache_dir>/<domain>_<algo>_<reward_task>_<hash>.npy`. The hash covers the episode
    COUNT and the first/last filenames, so pointing `max_episodes` at a different prefix
    gets its own cache entry rather than silently reusing another one's rewards."""
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
            r = relabel_rewards(phys, domain, rt, env=env)[1:]      # reward at s_{t+1}
        t = np.zeros(T, np.float32)
        t[-1] = 1.0                                                # trajectory boundary
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
    """Offline ExoRL data for a domain config. `data_dir` may point at either the
    `<domain>/<algo>/buffer` directory or its parent; `exorl_task` selects the reward
    function, `exorl_max_episodes` bounds the relabelling cost."""
    task = cfg["task"]
    algo = str(cfg.get("exorl_algo") or DEFAULT_ALGO)
    d = cfg.get("data_dir") or os.path.join(DATA_ROOT, domain_of(task), algo)
    d = os.path.expanduser(str(d))
    if os.path.isdir(os.path.join(d, "buffer")):
        d = os.path.join(d, "buffer")
    return load_exorl(d, task, reward_task=cfg.get("exorl_task"), algo=algo,
                      max_episodes=cfg.get("exorl_max_episodes"),
                      cache_dir=cfg.get("exorl_cache_dir"))
