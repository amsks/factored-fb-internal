"""exorl_dmc_tasks — ExoRL's `custom_dmc_tasks`, vendored.

Upstream: https://github.com/denisyarats/exorl @ a3fb07a (MIT). See PROVENANCE.md for the
exact scope and the two deliberate patches.

`make(domain, task)` mirrors upstream's dispatcher. Note that it only covers the tasks
ExoRL ADDS on top of the dm_control suite (cheetah run_backward/flip/flip_backward,
walker flip, all four point_mass_maze reach targets); the standard suite tasks
(walker stand/walk/run, cheetah run) still come from `dm_control.suite.load`, exactly as
ExoRL's own `dmc.py` resolves them (`suite.ALL_TASKS` first, custom tasks second).
Sub-modules are imported LAZILY: importing this package must not pull in dm_control /
mujoco, because env/registry.py's prefix table is read by processes that have neither.
"""
import importlib

_DOMAINS = ("cheetah", "walker", "point_mass_maze")


def _module(domain):
    if domain not in _DOMAINS:
        raise ValueError(f"no vendored ExoRL task module for domain {domain!r} "
                         f"(vendored: {_DOMAINS})")
    return importlib.import_module(f"{__name__}.{domain}")


def make(domain, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    return _module(domain).make(task, task_kwargs=task_kwargs,
                                environment_kwargs=environment_kwargs,
                                visualize_reward=visualize_reward)
