"""Domain adapter protocol: data loading, eval tasks and eval rollouts."""

from typing import Protocol


def column_backed_dataset(ds):
    """Return the column-backed Dataset under any sampler wrapper."""
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
    """What the training loop needs from a domain."""

    name: str

    def make_env_and_datasets(self, cfg) -> tuple:
        """Return (env, train_dataset, val_dataset)."""
        ...

    def eval_tasks(self, env, cfg) -> list:
        """Task handles to evaluate; [None] for single-task domains."""
        ...

    def evaluate(
        self, agent, env, task, cfg, num_episodes: int, num_video: int = 0
    ) -> tuple:
        """Return (stats, frames); video is unsupported, so frames is []."""
        ...


def needs_goal_frame(agent):
    """Whether the rollout must pass this agent an explicit goal observation.

    True for goal-conditioned critics and for critics that embed a goal into z. False for
    reward-inference agents, whose inferred z a goal would overwrite.
    """
    if getattr(agent, "ROLLS_INFERRED_COND", False):
        return False
    provides = set(getattr(agent.critic, "PROVIDES_COND", ()) or ())
    return ("goal" in provides and "z" not in provides) or hasattr(
        agent.critic, "map_eval_cond"
    )


def episode_horizon(adapter, explicit=None):
    """Episode length from the adapter's `_horizon` (cfg["horizon"] is RLDP's window)."""
    if explicit is not None:
        return int(explicit)
    h = getattr(adapter, "_horizon", None)
    if h:
        return int(h)
    raise ValueError(
        f"{type(adapter).__name__} publishes no `_horizon`, so the episode length for this "
        f"rollout is unknown (config['horizon'] is RLDP's window length, not an episode "
        f"length). Set `_horizon` on the adapter or pass the horizon explicitly."
    )


def get_adapter(cfg) -> DomainAdapter:
    """Return the adapter named by the domain config (default 'ogbench')."""
    name = (cfg.get("adapter") or "ogbench") if hasattr(cfg, "get") else "ogbench"
    if name == "ogbench":
        from env.ogbench import OGBenchAdapter

        return OGBenchAdapter()
    if name == "unified":
        from env.unified import UnifiedAdapter

        return UnifiedAdapter()
    raise ValueError(f"unknown adapter {name!r} (known: ogbench, unified)")
