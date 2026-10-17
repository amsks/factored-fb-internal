"""Custom DMC tasks from ExoRL (https://github.com/denisyarats/exorl, MIT).

Covers only the tasks ExoRL adds to the dm_control suite. Domain modules are imported
lazily so this package does not require dm_control.
"""
import importlib

_DOMAINS = ("cheetah", "walker", "point_mass_maze", "quadruped")


def _module(domain):
    if domain not in _DOMAINS:
        raise ValueError(f"no vendored ExoRL task module for domain {domain!r} "
                         f"(vendored: {_DOMAINS})")
    return importlib.import_module(f"{__name__}.{domain}")


def make(domain, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    return _module(domain).make(task, task_kwargs=task_kwargs,
                                environment_kwargs=environment_kwargs,
                                visualize_reward=visualize_reward)
