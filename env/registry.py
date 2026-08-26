"""env/registry.py — the generic environment registry.

Adding an environment to the unified adapter is a data entry here, not code scattered across
adapters + yaml. A task's family (which backend builds it), its default offline data_dir, and
its eval horizon are all resolved from the task slug's prefix; per-task overrides live in
`EXPLICIT`. The domain yaml then only needs `adapter: unified` + `task: <slug>`.
"""
import os

COLLECTED = os.environ.get("FB_COLLECTED", "/raid/tongle/rl/fb-data/collected")

# task-slug prefix -> (family, eval horizon, family folder used in the data path)
_PREFIX = {
    "b2d": ("box2d", 200, "Box2D"),
    "p2d": ("box2d", 200, "Box2D"),
    "mw":  ("metaworld", 100, "Metaworld"),
    "ms":  ("maniskill", 25, "ManiSkill"),
    # ExoRL / dm_control. REWARD-specified, not goal-specified: these are the domains the
    # successor-measure family's `infer_cond(net, obs, rewards)` interface is actually for,
    # and the only family here that no goal-conditioned critic can run at all. Horizon 1000
    # is the ExoRL episode length for every T3 domain (action_repeat 1). Data does NOT live
    # under COLLECTED -- ExoRL ships its own `<domain>/<algo>/buffer` trees, so the domain
    # yamls set `data_dir` explicitly (env/exorl_env.py::DATA_ROOT is the default root).
    "dmc": ("exorl", 1000, "ExoRL"),
    # HumEnv / Meta Motivo SMPL humanoid. Data is the RELEASED REPLAY BUFFER of a trained
    # FB-CPR checkpoint (agent-replay, not play data -- spec 23 §6.1 requires saying so), so
    # it ships no `buffer/episode_*.npz` tree of its own: scripts/convert_humenv_buffer.py
    # writes that layout and the domain yaml sets `data_dir` explicitly. Horizon 300 is
    # HumEnv's episode length (30 Hz decisions), which the buffer confirms -- 16,600 of its
    # 16,650 episodes are exactly 300 steps.
    "hum": ("humenv", 300, "HumEnv"),
}

# per-task overrides (data_dir / horizon / family), for anything the prefix rule doesn't cover
EXPLICIT = {}


def _prefix(task):
    key = task.split("-", 1)[0]
    if key not in _PREFIX:
        raise KeyError(f"no env registered for task {task!r} (known prefixes: {sorted(_PREFIX)})")
    return _PREFIX[key]


def family(task):
    return EXPLICIT.get(task, {}).get("family") or _prefix(task)[0]


def horizon(task):
    ov = EXPLICIT.get(task, {}).get("horizon")
    return int(ov) if ov is not None else _prefix(task)[1]


def data_dir(task):
    ov = EXPLICIT.get(task, {}).get("data_dir")
    if ov:
        return ov
    return os.path.join(COLLECTED, f"{_prefix(task)[2]}__{task}")


def known_tasks():
    return sorted(EXPLICIT) or list(_PREFIX)
