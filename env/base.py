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


def needs_goal_frame(agent):
    """Does the rollout have to hand this agent an explicit goal FRAME?

    THE one definition of the rule, shared by `UnifiedAdapter.evaluate` and by
    `env/online.py::OnlineTrainer._needs_goal` (they used to carry two copies of it).

      * a purely goal-conditioned critic (gciql: cond["goal"], no z) obviously needs one;
      * so does a z-critic that exposes `map_eval_cond` (fb, psm, rldp, density_fb): the hook
        embeds the goal frame INTO z, and without it eval runs on eval_cond0's zero z, for
        which Q(s,a;0) == 0 and the policy is goal-blind (the 2026-08-17 FB-zero bug);
      * but NOT an agent that declares `ROLLS_INFERRED_COND`. That is impls/main.py's
        `InferredCondAgent`, the --eval_on_inferred_cond route: it is evaluating a task
        specified by REWARD SAMPLES, so an env goal is not merely unnecessary, it is the
        thing that must not be supplied. `Agent.sample_actions` merges any goal it is given
        over the stored cond_eval, and `map_eval_cond` then recomputes the conditioning from
        that goal, so a goal handed in here silently overwrites the inferred index -- which
        is precisely why --infer_cond_episodes had no observable effect before.

    Deliberately a `getattr` on the AGENT, not on the critic: which conditioning a run rolls
    is a property of that run, while the critic's ability to consume a goal is unchanged.
    """
    if getattr(agent, "ROLLS_INFERRED_COND", False):
        return False
    provides = set(getattr(agent.critic, "PROVIDES_COND", ()) or ())
    return ("goal" in provides and "z" not in provides) \
        or hasattr(agent.critic, "map_eval_cond")


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
