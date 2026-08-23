"""OGBench adapter — the behaviour impls/main.py had before adapters existed, moved behind
the DomainAdapter seam.

By DEFAULT it is byte-for-byte the original OGBench behaviour: one eval row per task_id
(the collaborator's runs are unaffected). Set `eval_aggregate: true` in the domain config to
instead fold the task_infos into a single mean-success row (what our unified campaign +
agg-jsonl reporting expect, so cube-single lines up with the Box2D/MW/MS tasks).
"""
import numpy as np

from impls.utils.datasets import FBDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate as ogbench_evaluate

# Subset draw for `dataset_num_transitions`. Fixed, not seeded off --seed: the size ladder
# compares critics at a given dataset size, so every run at that size must see the SAME data
# or the critic contrast is confounded with a data draw.
_SUBSET_SEED = 0


def truncate_whole_trajectories(dataset, num_transitions, seed=_SUBSET_SEED):
    """Keep a random subset of WHOLE trajectories totalling >= `num_transitions` rows.

    Whole trajectories, because GCDataset's goal sampling is trajectory-relative: cutting
    mid-trajectory would shorten the horizon over which trajectory goals are drawn, which is
    a change to the goal distribution, not just to the sample count -- exactly the confound
    the size ladder (spec E1) exists to remove.

    Rows, not transitions, are counted: OGBench's compact layout stores T+1 states per
    length-T trajectory, and it is that row count (1,001,000 for cube-single-play-v0) the
    spec's size table quotes.

    The draw is one fixed permutation truncated at the requested count, so the sizes nest
    (48k ⊂ 100k ⊂ 250k) and every run at a size sees identical data. Selected trajectories
    are re-emitted in their original order, which makes the full-size case a bit-for-bit
    no-op.
    """
    fields = {k: np.asarray(v) for k, v in dataset.items()}
    size = len(fields["observations"])
    if num_transitions is None or int(num_transitions) >= size:
        return dataset
    num_transitions = int(num_transitions)
    assert num_transitions > 0, f"dataset_num_transitions must be positive, got {num_transitions}"

    # A trajectory's LAST row is the one with valids == 0 (compact layout); without `valids`
    # fall back to `terminals`, which in the non-compact layout marks the same rows.
    flag = fields.get("valids")
    ends = np.nonzero(np.asarray(flag) == 0)[0] if flag is not None \
        else np.nonzero(np.asarray(fields["terminals"]) > 0)[0]
    assert len(ends) and ends[-1] == size - 1, \
        "dataset does not end on a trajectory boundary; cannot split it into trajectories"
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1

    order = np.random.RandomState(seed).permutation(len(ends))
    take = order[:int(np.searchsorted(np.cumsum(lens[order]), num_transitions) + 1)]
    take = np.sort(take)
    keep = np.concatenate([np.arange(starts[i], ends[i] + 1) for i in take])

    out = {k: v[keep] for k, v in fields.items() if len(v) == size}
    print(f"[ogbench] dataset_num_transitions={num_transitions}: kept {len(take)} of "
          f"{len(ends)} trajectories, {len(keep)} of {size} rows (seed {seed})", flush=True)
    return FBDataset.create(**out)


class OGBenchAdapter:
    name = "ogbench"

    def make_env_and_datasets(self, cfg):
        # `measure_qpos_slices` is density_fb's PRIVILEGED MEASURE CHANNEL on pixels; it is
        # the only thing in this tree that needs the npz's raw 'qpos' column, so it is also
        # what gates loading it (state runs keep their exact batch keys).
        env, train_ds, val_ds = make_env_and_datasets(
            cfg["env_name"], cfg.get("frame_stack"),
            add_info=cfg.get("measure_qpos_slices") is not None)
        # Publish the episode length for env/base.py::episode_horizon. Online rollouts and
        # evaluate() need it, and it must NOT be inferred from config['horizon'] -- that key
        # is RLDP's self-predictive window. OGBench carries it on the TimeLimit wrapper.
        self._horizon = int(getattr(env, "_max_episode_steps", None)
                            or (env.spec.max_episode_steps if env.spec is not None else 0)
                            or 0) or None
        # Size ladder (spec E1). Train split only: the validation split is the fixed yardstick
        # every rung is measured against, so shrinking it too would move the ruler as well as
        # the data.
        n = cfg.get("dataset_num_transitions")
        if n is not None:
            train_ds = truncate_whole_trajectories(train_ds, n)
        return env, train_ds, val_ds

    def eval_tasks(self, env, cfg):
        if cfg.get("eval_aggregate"):
            return [None]                                    # one folded mean-success row
        return list(range(1, len(env.unwrapped.task_infos) + 1))  # DEFAULT: per-task (original)

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        if task is not None:
            # Original per-task_id behaviour, unchanged.
            stats, _, renders = ogbench_evaluate(
                agent, env, task_id=task, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video)
            return {k: float(v) for k, v in dict(stats).items()}, renders
        # Aggregate: mean success over all task_infos, one scalar (no at-goal ticks: Box2D-only).
        import numpy as np
        n_tasks = len(env.unwrapped.task_infos)
        succ, renders = [], []
        for tid in range(1, n_tasks + 1):
            stats, _, r = ogbench_evaluate(
                agent, env, task_id=tid, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video if tid == 1 else 0)
            succ.append(float(dict(stats).get("success", next(iter(dict(stats).values())))))
            renders += list(r)
        return {"success": float(np.mean(succ))}, renders
