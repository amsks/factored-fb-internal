"""env/registry.py — task registry for the unified adapter.

A task's family (which backend builds it), its default offline data_dir and its eval horizon
are resolved from the task slug's prefix; per-task overrides live in `EXPLICIT`. A domain
yaml then only needs `adapter: unified` + `task: <slug>`.
"""
import os

COLLECTED = os.environ.get("FB_COLLECTED", "/raid/tongle/rl/fb-data/collected")

# task-slug prefix -> (family, eval horizon, family folder used in the data path)
_PREFIX = {
    "mw":  ("metaworld", 100, "Metaworld"),
    "ms":  ("maniskill", 25, "ManiSkill"),
    # ExoRL / dm_control: reward-specified tasks, 1000-step episodes. ExoRL ships its own
    # `<domain>/<algo>/buffer` trees, so the domain yamls set `data_dir` explicitly.
    "dmc": ("exorl", 1000, "ExoRL"),
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
