"""envs/registry.py — family-dispatched construction of env / dataset / evaluator.

``train.py`` calls exactly three factories — :func:`make_env`, :func:`make_dataset`,
:func:`make_evaluator` — and this module routes each to the right env family based
on a ``family:name`` prefix in the domain string (or an explicit ``env_family``):

    "cube-single-play-v0"          -> family "ogbench", name "cube-single-play-v0"
    "ogbench:cube-single-play-v0"  -> family "ogbench", name "cube-single-play-v0"
    "rlmini:Metaworld/pick-place"  -> family "rlmini",  name "Metaworld/pick-place"

The ``ogbench`` family delegates verbatim to the existing
``envs.ogbench`` / ``data.ogbench`` / ``evals.ogbench`` code paths, so bare domain
names (no prefix) behave byte-identically to before this refactor (P0). The
``rlmini`` family is wired in P1 (``envs.rlmini_adapter`` / ``data.unified`` /
``evals.goal_eval``); referencing it before those modules exist raises a clear
error rather than silently falling back.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

DEFAULT_FAMILY = "ogbench"
KNOWN_FAMILIES = ("ogbench", "rlmini")


def parse_domain(domain: str, env_family: Optional[str] = None) -> Tuple[str, str]:
    """Split a domain string into ``(family, name)``.

    An explicit ``env_family`` wins; otherwise a ``family:name`` prefix is used;
    otherwise the family defaults to ``ogbench`` and the whole string is the name.
    A prefix present alongside an explicit ``env_family`` is stripped from name.
    """
    prefix_family, _, prefix_name = domain.partition(":")
    if _ and prefix_family in KNOWN_FAMILIES:
        family, name = prefix_family, prefix_name
    else:
        family, name = DEFAULT_FAMILY, domain
    if env_family is not None:
        family = env_family
    return family, name


def _unknown(family: str) -> ValueError:
    return ValueError(
        f"unknown env family {family!r}; expected one of {KNOWN_FAMILIES}. "
        f"Use a 'family:name' domain (e.g. 'rlmini:Metaworld/reach') or set env_family."
    )


def make_env(
    domain: str,
    *,
    seed: Optional[int] = None,
    obs_type: str = "state",
    frame_stack: int = 1,
    env_family: Optional[str] = None,
    **extra: Any,
):
    """Construct the eval/rollout env. Returns ``(env, info_dict)`` (OGBench shape)."""
    family, name = parse_domain(domain, env_family)
    if family == "ogbench":
        from envs.ogbench import create_ogbench_env

        return create_ogbench_env(name, seed=seed, obs_type=obs_type, frame_stack=frame_stack)
    if family == "rlmini":
        from envs.rlmini_adapter import make_rlmini_env  # P1

        return make_rlmini_env(name, seed=seed, obs_type=obs_type, frame_stack=frame_stack, **extra)
    raise _unknown(family)


def make_dataset(
    domain: str,
    *,
    data_path,
    load_n_episodes: int,
    device: str,
    n_transitions: Optional[int],
    obs_type: str = "state",
    frame_stack: int = 1,
    env_family: Optional[str] = None,
):
    """Load the offline dataset into a ``DictBuffer``."""
    family, name = parse_domain(domain, env_family)
    if family == "ogbench":
        from data.ogbench import load_ogbench_dataset

        return load_ogbench_dataset(
            domain=name,
            data_path=data_path,
            load_n_episodes=load_n_episodes,
            device=device,
            n_transitions=n_transitions,
            obs_type=obs_type,
            frame_stack=frame_stack,
        )
    if family == "rlmini":
        from data.unified import load_dataset  # P1

        return load_dataset(
            name,
            data_path=data_path,
            load_n_episodes=load_n_episodes,
            device=device,
            n_transitions=n_transitions,
        )
    raise _unknown(family)


def make_evaluator(domain: str, *, agent, offline_buffer, env_family: Optional[str] = None, **kw):
    """Construct the per-family evaluator (relabel -> infer z / goal -> rollout)."""
    family, name = parse_domain(domain, env_family)
    if family == "ogbench":
        from evals.ogbench import OGBenchEvaluator

        return OGBenchEvaluator(domain=name, agent=agent, offline_buffer=offline_buffer, **kw)
    if family == "rlmini":
        from evals.goal_eval import GoalEvaluator  # P1

        return GoalEvaluator(env_name=name, agent=agent, offline_buffer=offline_buffer, **kw)
    raise _unknown(family)
