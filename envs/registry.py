"""Backend registry for environment families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EnvBackend:
    create_env: Callable
    load_dataset: Callable
    make_evaluator: Callable


def _ogbench_backend() -> EnvBackend:
    from data.ogbench import load_ogbench_dataset
    from envs.ogbench import create_ogbench_env
    from evals.ogbench import OGBenchEvaluator

    def make_evaluator(**kwargs):
        kwargs.pop("cfg", None)
        return OGBenchEvaluator(**kwargs)

    return EnvBackend(
        create_env=lambda domain, cfg=None, **kwargs: create_ogbench_env(domain, **kwargs),
        load_dataset=load_ogbench_dataset,
        make_evaluator=make_evaluator,
    )


def _rlmini_backend() -> EnvBackend:
    from data.rlmini import load_rlmini_dataset
    from envs.rlmini import create_rlmini_env
    from evals.rlmini import RLMiniEvaluator

    return EnvBackend(
        create_env=create_rlmini_env,
        load_dataset=load_rlmini_dataset,
        make_evaluator=lambda **kwargs: RLMiniEvaluator(**kwargs),
    )


def get_backend(env_family: str) -> EnvBackend:
    match env_family:
        case "ogbench":
            return _ogbench_backend()
        case "rlmini":
            return _rlmini_backend()
        case _:
            raise ValueError(
                f"Unknown env_family {env_family!r}; expected 'ogbench' or 'rlmini'"
            )
