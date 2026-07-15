"""envs/testbed.py — read the rlmini factoredfb_testbed registry from Factored-FB.

The task set + per-env adapter specs live in the rlmini repo
(``factoredfb_testbed/`` on branch tl-factoredfb-env), the source of truth. This
module locates that package (from the installed ``rlmini_envs`` location, or an
explicit ``FACTOREDFB_TESTBED_ROOT`` env var) and exposes its registry so
``envs/rlmini_adapter.make_rlmini_env`` can auto-configure obs_keys / goal_key /
achieved_goal per task.

Import is lazy and best-effort: if the testbed package is not found, adapter
construction falls back to the RLMiniAdapter defaults (no hard dependency).
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


@functools.lru_cache(maxsize=1)
def _registry():
    root = os.environ.get("FACTOREDFB_TESTBED_ROOT")
    if root is None:
        import rlmini_envs

        # .../src/rlmini_envs/__init__.py -> repo root two levels up from the package dir
        root = str(Path(rlmini_envs.__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from factoredfb_testbed import registry  # type: ignore

    return registry


def get_task_spec(task_id: str) -> Optional[Dict[str, Any]]:
    """Merged task+env spec for ``Env/variant``, or None if unknown / testbed absent."""
    try:
        return _registry().get_task(task_id)
    except (ImportError, ModuleNotFoundError, KeyError, FileNotFoundError):
        return None


def list_task_ids(manipulation: Optional[int] = None, env: Optional[str] = None) -> List[str]:
    return _registry().task_ids(manipulation=manipulation, env=env)
