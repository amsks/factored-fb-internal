"""The interface between the algorithms (impls/critics, impls/actors) and the environments
and datasets they run on. A DomainAdapter owns data loading, the list of eval tasks and the
eval rollout, so adding a simulator is a new adapter rather than an edit to the training loop.

Implementations:
    env.ogbench.OGBenchAdapter   -- OGBench play datasets + its multi-task eval protocol
    env.unified.UnifiedAdapter   -- ExoRL, MetaWorld and ManiSkill datasets + fixed-seed eval
"""
from typing import Protocol


def column_backed_dataset(ds):
    """The raw column-backed Dataset under any per-critic sampler wrapper.

    impls/main.py hands eval-time code the dataset after `make_dataset` wrapped it, and the
    wrappers (HILPFutureDataset, GCDataset, WindowedDataset) implement `sample` but not the
    column access a reward relabel needs; each keeps its base in `.dataset`."""
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

    def evaluate(self, agent, env, task, cfg, num_episodes: int, num_video: int = 0) -> tuple:
        """-> (stats: dict[str, float], frames: list). `stats` is logged verbatim under
        eval/. Video rendering is not supported: num_video must be 0 and frames is []."""
        ...


def needs_goal_frame(agent):
    """Does the rollout have to hand this agent an explicit goal frame?

      * a purely goal-conditioned critic (cond["goal"], no z) needs one;
      * so does a z-critic that exposes `map_eval_cond` (fb, rldp, hilp, density_fb): the hook
        embeds the goal frame into z, and without it eval runs on eval_cond0's zero z, for
        which Q(s,a;0) == 0 and the policy ignores the task;
      * an agent that declares `ROLLS_INFERRED_COND` (InferredCondAgent, the reward-inference
        route) must NOT get one: `Agent.sample_actions` merges a supplied goal over cond_eval,
        and `map_eval_cond` would then overwrite the task vector inferred from rewards.

    A `getattr` on the agent, not the critic: which conditioning a run rolls is a property
    of that run.
    """
    if getattr(agent, "ROLLS_INFERRED_COND", False):
        return False
    provides = set(getattr(agent.critic, "PROVIDES_COND", ()) or ())
    return ("goal" in provides and "z" not in provides) \
        or hasattr(agent.critic, "map_eval_cond")


def episode_horizon(adapter, explicit=None):
    """The episode length a rollout runs for, published by the adapter as `_horizon`.

    Not read from `cfg["horizon"]`: that key is RLDP's self-predictive window length
    (impls/configs/critic/rldp.yaml)."""
    if explicit is not None:
        return int(explicit)
    h = getattr(adapter, "_horizon", None)
    if h:
        return int(h)
    raise ValueError(
        f"{type(adapter).__name__} publishes no `_horizon`, so the episode length for this "
        f"rollout is unknown (config['horizon'] is RLDP's window length, not an episode "
        f"length). Set `_horizon` on the adapter or pass the horizon explicitly.")


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
