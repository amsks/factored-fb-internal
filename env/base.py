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


def column_backed_dataset(ds):
    """The raw column-backed Dataset under any per-critic sampler wrapper.

    impls/main.py hands eval-time code the dataset AFTER `make_dataset` wrapped it, and the
    wrappers (HILPFutureDataset, GCDataset, WindowedDataset, ...) implement `sample` but not
    the `"col" in ds` / `ds["col"]` column access a reward relabel needs; each keeps its base
    in `.dataset`. Same walk as env/ogbench.py::OGBenchAdapter._column_backed, shared so the
    unified adapter's relabel path does not carry a second copy."""
    cur = ds
    for _ in range(8):
        try:
            if "observations" in cur:
                return cur
        except TypeError:
            pass
        nxt = getattr(cur, "dataset", None)
        if nxt is None:
            break
        cur = nxt
    raise TypeError(f"no column-backed dataset under {type(ds).__name__}")


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


def episode_horizon(adapter, explicit=None):
    """The EPISODE LENGTH a rollout runs for, taken from the ADAPTER.

    Deliberately NOT read from `cfg["horizon"]`, and deliberately not defaulted. That key is
    already taken: `impls/configs/critic/rldp.yaml` sets `horizon: 5` as RLDP's
    self-predictive WINDOW length, and no domain yaml in this tree sets an episode length
    under that name at all. Both rollout sites used to fall back to
    `cfg.get("horizon", 100)`, so `--critic rldp --online` on an OGBench domain (whose
    adapter publishes no `_horizon`) collected 5-STEP "episodes" -- the env reset every five
    steps, and `GCDataset` handed trajectories far too short for its goal sampling to mean
    anything -- with nothing failing anywhere. The `100` default was the same failure with a
    less obvious number.

    The episode length belongs to the env builder: it comes back alongside the env
    (`env/unified.py::make_task_env` -> `(env, horizon)`) and is published as
    `adapter._horizon`. An adapter that does not publish one cannot be rolled out here, and
    says so rather than guessing.
    """
    if explicit is not None:
        return int(explicit)
    h = getattr(adapter, "_horizon", None)
    if h:
        return int(h)
    raise ValueError(
        f"{type(adapter).__name__} publishes no `_horizon`, so the episode length for this "
        f"rollout is unknown. It is NOT taken from config['horizon'] -- that key is RLDP's "
        f"self-predictive window length (impls/configs/critic/rldp.yaml), and reading it "
        f"here silently cut online episodes to five steps. Set `_horizon` on the adapter "
        f"(env/unified.py does, from make_task_env) or pass the horizon explicitly."
    )


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
