"""The seam between the ALGORITHMS (impls/critics, impls/actors) and the ENVIRONMENTS +
DATASETS they run on.

Before adapters, impls/main.py talked to OGBench directly: `make_env_and_datasets(env_name)`
for data and `env.unwrapped.task_infos` + OGBench's `evaluate()` for eval. That hard-wired
the algorithm tree to one benchmark. A DomainAdapter owns all three of those decisions, so
adding a simulator is a new adapter rather than an edit to the training loop.

Implementations:
    ogbench_adapter.OGBenchAdapter   -- OGBench play datasets + its multi-task eval protocol
    unified_adapter.UnifiedAdapter   -- our collected per-episode npz (Box2D / Metaworld /
                                        ManiSkill) + fixed-seed eval with the at-goal dwell metric
"""
from typing import Any, Protocol


class DomainAdapter(Protocol):
    """What the training loop needs from a domain, and nothing more."""

    name: str

    def make_env_and_datasets(self, cfg) -> tuple:
        """-> (env, train_dataset, val_dataset). Datasets are impls.utils.datasets.Dataset
        subclasses; `make_dataset(train, cfg)` then applies the critic's dataset_class."""
        ...

    def eval_tasks(self, env, cfg) -> list:
        """Task handles to evaluate. OGBench returns its task_ids; single-task domains
        return [None]."""
        ...

    def evaluate(self, agent, env, task, cfg, num_episodes: int, num_video: int) -> tuple:
        """-> (stats: dict[str, float], frames: list). `stats` is logged verbatim under
        eval/, so metric names are the adapter's choice (ours: success, at_goal_frac)."""
        ...


def get_adapter(cfg) -> DomainAdapter:
    """Resolve the adapter named by the domain config (`adapter:` key, default 'ogbench')."""
    name = (cfg.get("adapter") or "ogbench") if hasattr(cfg, "get") else "ogbench"
    if name == "ogbench":
        from env.ogbench import OGBenchAdapter
        return OGBenchAdapter()
    if name == "unified":
        from env.unified import UnifiedAdapter
        return UnifiedAdapter()
    raise ValueError(f"unknown adapter {name!r} (known: ogbench, unified)")
