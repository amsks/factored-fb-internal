"""Task registry for the unified adapter, keyed by task-slug prefix."""

import os

COLLECTED = os.environ.get("FB_COLLECTED", os.path.expanduser("~/fb-data/collected"))

# task-slug prefix -> (family, eval horizon, family folder used in the data path)
_PREFIX = {
    "mw": ("metaworld", 100, "Metaworld"),
    "ms": ("maniskill", 25, "ManiSkill"),
    # ExoRL domain yamls set data_dir explicitly
    "dmc": ("exorl", 1000, "ExoRL"),
}

# Per-task overrides of data_dir, horizon or family
EXPLICIT = {}


def _prefix(task):
    key = task.split("-", 1)[0]
    if key not in _PREFIX:
        raise KeyError(
            f"no env registered for task {task!r} (known prefixes: {sorted(_PREFIX)})"
        )
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
