"""Auto-imported (via PYTHONPATH) shim for the vendored OGBench child process.

Two independent, opt-in behaviors (each a no-op unless its env var is set), so
this file is inert for any unrelated Python process and never edits
third_party/ (which must stay byte-for-byte upstream per its PROVENANCE rules):

1. GCIQL_WANDB_FORCE_MODE = offline|disabled
   OGBench's setup_wandb hardcodes wandb.init(mode='online'); force the mode
   (wandb >= 0.27 lets the explicit kwarg override WANDB_MODE/WANDB_DISABLED).

2. GCIQL_FB_ENV = <ogbench env_name, e.g. cube-single-play-v0>
   Mirror GCIQL's eval metrics into FB's wandb key scheme, ADDING keys
   alongside OGBench's originals so FB and GCIQL overlay on the same panels:
     evaluation/overall_success           -> eval/reward/eval/success
     evaluation/task<N>_<desc>_success    -> eval/reward/<fb_task>/success
   where <fb_task> = "<env_base>-singletask-task<N>-v0" and env_base is
   GCIQL_FB_ENV with any trailing "-v0" removed.
"""
import functools
import os
import re

_forced = os.environ.get("GCIQL_WANDB_FORCE_MODE")
_fb_env = os.environ.get("GCIQL_FB_ENV")

if _forced in ("offline", "disabled") or _fb_env:
    try:
        import wandb as _wandb

        if _forced in ("offline", "disabled"):
            _orig_init = _wandb.init

            @functools.wraps(_orig_init)
            def _init(*args, **kwargs):  # force mode regardless of caller kwarg
                kwargs["mode"] = _forced
                return _orig_init(*args, **kwargs)

            _wandb.init = _init

        if _fb_env:
            _env_base = _fb_env[:-3] if _fb_env.endswith("-v0") else _fb_env
            _task_re = re.compile(r"^evaluation/task(\d+)_.*_success$")
            _orig_log = _wandb.log

            def _fb_keys(data):
                extra = {}
                for k, v in list(data.items()):
                    if k == "evaluation/overall_success":
                        extra["eval/reward/eval/success"] = v
                        continue
                    m = _task_re.match(k)
                    if m:
                        n = m.group(1)
                        extra[
                            f"eval/reward/{_env_base}-singletask-task{n}-v0/success"
                        ] = v
                return extra

            @functools.wraps(_orig_log)
            def _log(data=None, *args, **kwargs):
                try:
                    if isinstance(data, dict):
                        data = {**data, **_fb_keys(data)}
                except Exception:
                    pass
                return _orig_log(data, *args, **kwargs)

            _wandb.log = _log
    except Exception:
        pass
