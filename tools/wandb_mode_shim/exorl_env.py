"""Spec 23 P0 -- the ONE blocker closer: an extension point that lets
`run_gciql.py` / vendored `main.py` load a converted ExoRL dataset + the
dm_control eval adapter for `point_mass_maze`, without editing
`third_party/ogbench/` (must stay byte-for-byte upstream, see
`third_party/ogbench/PROVENANCE.md`) and without editing
`tools/wandb_mode_shim/gciql_flow.py`.

Mechanism: exactly the same monkeypatch-via-sitecustomize pattern already
used in this file's neighbors (agent injection, DrQ encoder registration,
etc. -- see `sitecustomize.py`). `install()` below wraps
`utils.env_utils.make_env_and_datasets` so that:

  - env_name starting with EXORL_PREFIX ("exorl:") routes to the ExoRL
    loader in this module.
  - every OTHER env_name calls straight through to the ORIGINAL vendored
    function, with the same arguments, same return value -- BYTE-IDENTICAL
    default path (see tests/test_exorl_env_extension_point.py, which stubs
    `ogbench.make_env_and_datasets` and asserts the non-ExoRL branch reaches
    it unchanged).

Everything ExoRL-specific (imports of `point_mass_maze_env`/`dm_control`,
the mujoco-version overlay below) is LAZY -- deferred to
`make_exorl_env_and_datasets`, never done at module import time -- so
`install()` running on every single run_gciql.py launch (any env_name, via
sitecustomize) has zero import/behavioral footprint for non-ExoRL configs.

Second, independent blocker found only at verification time (not part of
the original env_utils gap): `/dev/shm/.venv-jax` has `mujoco==3.3.0`, but
`dm_control==1.0.41` (already installed there) declares `Requires-Dist:
mujoco>=3.8.1` -- unrelated to this extension point, a pre-existing unmet
dependency in the shared jax venv. Upgrading mujoco venv-wide was rejected
as out of contained scope (shared venv, live GPU training). Instead,
`_mujoco_overlay()` below temporarily prepends this repo's OWN `.venv`
(which already carries a working `mujoco==3.8.1` + the same
`dm_control==1.0.41`) to `sys.path`, imports only `point_mass_maze_env`
(and transitively `dm_control`/`mujoco`) under that overlay, then removes
the overlay path immediately -- so only the `mujoco` C-extension resolves
to the newer build; numpy/jax/everything else in the process keeps
resolving to the jax venv's own copies (numpy is pre-imported before the
overlay is added specifically to guarantee this -- verified empirically,
see task-report.md). No package is installed or modified anywhere; nothing
on disk changes; this path only ever executes for `exorl:`-prefixed
dataset_names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401  (forces numpy to resolve/cache BEFORE any mujoco overlay is applied)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXORL_PREFIX = "exorl:"

_VENV_SITE_PACKAGES = str(REPO_ROOT / ".venv" / "lib" / "python3.11" / "site-packages")


_MIN_MUJOCO = (3, 8, 1)  # dm_control==1.0.41's declared Requires-Dist floor


def _parse_version(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def _import_point_mass_maze_env():
    """Import the eval adapter (and, transitively, dm_control/mujoco) under
    the repo .venv's mujoco==3.8.1 overlay, restoring sys.path immediately
    after. Only called from the ExoRL registry builders, never eagerly.

    `import mujoco` SUCCEEDS in the jax venv today (mujoco==3.3.0 is
    installed there, just too old for dm_control's own declared floor) --
    checking for ImportError is not enough; the version must be checked
    explicitly and the stale module purged from sys.modules before the
    overlay import, or Python's import cache would silently keep serving
    the already-imported 3.3.0 module."""
    scripts_exorl = str(REPO_ROOT / "scripts" / "exorl")
    if scripts_exorl not in sys.path:
        sys.path.insert(0, scripts_exorl)

    import mujoco as _mj

    if _parse_version(_mj.__version__) < _MIN_MUJOCO:
        for name in list(sys.modules):
            if name == "mujoco" or name.startswith("mujoco."):
                del sys.modules[name]
        sys.path.insert(0, _VENV_SITE_PACKAGES)
        try:
            import mujoco as _mj2

            assert _parse_version(_mj2.__version__) >= _MIN_MUJOCO, (
                f"overlay mujoco {_mj2.__version__} still below {_MIN_MUJOCO}"
            )
        finally:
            if _VENV_SITE_PACKAGES in sys.path:
                sys.path.remove(_VENV_SITE_PACKAGES)

    import point_mass_maze_env as _pmm_env  # noqa: E402

    return _pmm_env


class _ExoRLPointMassMazeEnv:
    """Gym-like shim so the vendored `evaluate()` (utils/evaluation.py) and
    `main.py`'s eval-task loop run unmodified against the ExoRL
    point_mass_maze dm_control adapter.

    Goal-conditioned success (spec 23 4.4): the (x, y) block (dims 0:2) of
    the observation coming within GOAL_EPS of one of the 20 published maze
    goals, at any point in the episode. `info['goal']` is a full 4-dim
    observation-shaped vector `[gx, gy, 0, 0]` (target position, zero
    velocity) -- GCDataset-trained actors condition on full-observation
    goals (HER-style relabeling), not on the raw 2-dim coordinate; the
    success check itself still only reads the verified 0:2 block.
    """

    def __init__(self, pmm_module, task: str = "reach_top_left", seed: int = 0):
        self._pmm = pmm_module
        self._inner = pmm_module.PointMassMazeEnv(task=task, seed=seed)
        self.task_infos = [{"task_name": f"goal{i + 1}"} for i in range(len(pmm_module.MAZE_20_GOALS))]
        self._goal_xy = pmm_module.MAZE_20_GOALS[0]
        self._success_any_time = False

    @property
    def unwrapped(self):
        return self

    def _goal_obs(self) -> np.ndarray:
        return np.array([self._goal_xy[0], self._goal_xy[1], 0.0, 0.0], dtype=np.float32)

    def _check_success(self, obs: np.ndarray) -> None:
        dist = float(np.linalg.norm(obs[:2] - self._goal_xy))
        if dist <= self._pmm.GOAL_EPS:
            self._success_any_time = True

    def reset(self, *, seed=None, options=None):
        options = options or {}
        task_id = options.get("task_id")
        goals = self._pmm.MAZE_20_GOALS
        idx = 0 if task_id is None else (int(task_id) - 1) % len(goals)
        self._goal_xy = goals[idx]
        self._success_any_time = False
        obs = self._inner.reset()
        self._check_success(obs)
        info = {"goal": self._goal_obs(), "success": float(self._success_any_time)}
        return obs, info

    def step(self, action):
        obs, _terminated, truncated, _inner_info = self._inner.step(action)
        self._check_success(obs)
        reward = -float(np.linalg.norm(obs[:2] - self._goal_xy))  # eval bookkeeping only; unused in training
        info = {"goal": self._goal_obs(), "success": float(self._success_any_time)}
        return obs, reward, False, truncated, info

    def render(self):
        try:
            return self._inner._env.physics.render(height=64, width=64, camera_id=0)
        except Exception:
            return np.zeros((64, 64, 3), dtype=np.uint8)


def _load_npz_dict(path: Path) -> dict:
    with np.load(path) as f:
        return {k: np.array(f[k]) for k in f.files}


def _make_point_mass_maze_env_and_datasets(entry: dict, frame_stack=None):
    if frame_stack is not None:
        raise NotImplementedError(
            "frame_stack is not supported for ExoRL envs (state-only P0); "
            f"got frame_stack={frame_stack!r} for {entry['prefix']!r}"
        )
    data_dir = Path(entry["data_dir"]).expanduser()
    train = _load_npz_dict(data_dir / f"{entry['prefix']}.npz")
    val = _load_npz_dict(data_dir / f"{entry['prefix']}-val.npz")
    pmm_module = _import_point_mass_maze_env()
    env = _ExoRLPointMassMazeEnv(pmm_module, task=entry.get("task", "reach_top_left"), seed=entry.get("seed", 0))
    env.reset()
    return env, train, val


# Registry: env_name = f"{EXORL_PREFIX}{key}" -> loader entry. Add new ExoRL
# domains/buffers here as spec 23 progresses past P0 -- never by branching on
# ad hoc string patterns inside the loader itself.
EXORL_REGISTRY = {
    "point_mass_maze_rnd_seed0": {
        "data_dir": "~/exorl_data/converted/point_mass_maze_rnd_seed0",
        "prefix": "point_mass_maze_rnd",
        "task": "reach_top_left",
        "seed": 0,
        "builder": _make_point_mass_maze_env_and_datasets,
    },
}


def make_exorl_env_and_datasets(dataset_name: str, frame_stack=None):
    """Dispatch on the registry. Raises (does not silently fall through) on
    an exorl: name that isn't registered -- a typo must fail loudly, not
    quietly re-route to the OGBench path."""
    key = dataset_name[len(EXORL_PREFIX):]
    if key not in EXORL_REGISTRY:
        raise ValueError(
            f"unknown ExoRL dataset_name {dataset_name!r}; registered keys: "
            f"{sorted(EXORL_REGISTRY)}"
        )
    entry = EXORL_REGISTRY[key]
    return entry["builder"](entry, frame_stack=frame_stack)


def install():
    """Wrap utils.env_utils.make_env_and_datasets. Idempotent (checks a
    marker attribute) and inert if utils.env_utils can't be imported (e.g.
    from a non-jax process). Does NOT import anything ExoRL-specific --
    that stays lazy inside make_exorl_env_and_datasets."""
    import utils.env_utils as _env_utils

    if getattr(_env_utils.make_env_and_datasets, "_exorl_patched", False):
        return
    _orig = _env_utils.make_env_and_datasets

    def _patched(dataset_name, frame_stack=None):
        if dataset_name.startswith(EXORL_PREFIX):
            return make_exorl_env_and_datasets(dataset_name, frame_stack=frame_stack)
        return _orig(dataset_name, frame_stack=frame_stack)

    _patched._exorl_patched = True
    _patched._exorl_orig = _orig
    _env_utils.make_env_and_datasets = _patched
