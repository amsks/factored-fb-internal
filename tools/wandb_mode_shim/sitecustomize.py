"""Auto-imported (via PYTHONPATH) shim that makes OGBench's hardcoded
``wandb.init(mode='online')`` honor an offline/disabled request.

Active ONLY when ``GCIQL_WANDB_FORCE_MODE`` is set to ``offline``/``disabled``;
otherwise a complete no-op, so default online runs are untouched. This lives in
the launcher's environment, NOT in third_party/ogbench (which stays byte-for-byte
upstream per its PROVENANCE rules).
"""
import functools
import os

_forced = os.environ.get("GCIQL_WANDB_FORCE_MODE")
if _forced in ("offline", "disabled"):
    try:
        import wandb as _wandb

        _orig_init = _wandb.init

        @functools.wraps(_orig_init)
        def _init(*args, **kwargs):  # force mode regardless of caller's kwarg
            kwargs["mode"] = _forced
            return _orig_init(*args, **kwargs)

        _wandb.init = _init
    except Exception:
        pass
