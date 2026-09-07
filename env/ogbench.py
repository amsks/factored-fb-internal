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


# --- OGBench reward-relabel + reward-inference eval (fb_tdj parity arm) --------------------
# docs/notes/2026-09-03-fb-vs-tdjepa.md #5/#7: td_jepa's OGBench FB numbers are a zero-shot
# reward-inference readout, not the env's goal-Dirac z. This is the missing half: relabel a
# sample of TRAIN transitions by their next observation's simulator state, feed
# (next_observations, rewards) to `agent.infer_eval_cond`, and roll the result. Four OGBench
# families: cube, antmaze, scene, puzzle -- one reward function each, dispatched on the
# task_info's goal key by `relabel_reward_for_task`.
#
# Offsets transcribed from td_jepa's cube_reward_fn
# (~/git/baselines/td_jepa/metamotivo/envs/ogbench.py:30-39): the free joint
# `object_joint_<i>` is 7 qpos wide (3 position + 4 quaternion) and starts at column 14 --
# the same fact impls/utils/datasets.py::parse_qpos_slices's docstring records and
# tests/impls/test_density_fb_pixels.py pins for cube 0.
CUBE_QPOS_OFFSETS = (14, 21, 28, 35)


def cube_relabel_reward(qpos, goal_xyzs, threshold=0.04):
    """td_jepa's zero-shot OGBench cube reward (`cube_reward_fn`), transcribed bit-for-bit:
    the count of cubes within `threshold` metres of their goal position, minus the number of
    cubes -- 0 when every cube is placed, down to -num_cubes when none are. Callers add
    `shift_reward` (1.0 in the fb_tdj protocol) to turn this into a 0/1 "all cubes placed"
    indicator on cube-single.

    qpos: [N, D] raw simulator qpos rows (the dataset's 'qpos' field; add_info=True).
    goal_xyzs: [num_cubes, 3] target cube positions in metres
        (env.unwrapped.task_infos[task_id - 1]['goal_xyzs']).
    Returns: [N] float64, UNSHIFTED (callers add shift_reward themselves).
    """
    qpos = np.asarray(qpos, np.float64)
    goal_xyzs = np.asarray(goal_xyzs, np.float64).reshape(-1, 3)
    num_cubes = goal_xyzs.shape[0]
    assert 1 <= num_cubes <= len(CUBE_QPOS_OFFSETS), (
        f"cube_relabel_reward supports 1-{len(CUBE_QPOS_OFFSETS)} cubes, got {num_cubes}")
    need = CUBE_QPOS_OFFSETS[num_cubes - 1] + 3
    assert qpos.ndim == 2 and qpos.shape[-1] >= need, (
        f"qpos is {qpos.shape} wide, need at least {need} columns for {num_cubes} cube(s)")
    successes = np.zeros(qpos.shape[0], np.float64)
    for i in range(num_cubes):
        off = CUBE_QPOS_OFFSETS[i]
        dist = np.linalg.norm(qpos[:, off:off + 3] - goal_xyzs[i], axis=-1)
        successes += (dist < threshold).astype(np.float64)
    return successes - num_cubes


def antmaze_relabel_reward(qpos, goal_xy, threshold=0.5):
    """td_jepa's `ant_reward_fn`, transcribed: 1[||qpos[:, :2] - goal_xy|| <= threshold] - 1,
    i.e. 0 at the goal and -1 elsewhere. `threshold` 0.5 is ogbench's own ant goal tolerance
    (locomaze/maze.py: `_goal_tol = 0.5` for the ant body); note the INCLUSIVE `<=`, unlike
    the cube's strict `<`.

    goal_xy comes from `task_infos[t]['goal_xy']`, the maze cell CENTRE. td_jepa instead
    calls `env.reset()` and reads `get_oracle_rep()`, and ogbench's maze reset adds noise to
    the goal by default (`add_noise_to_goal=True`), so td_jepa relabels against one random
    jitter of the same cell; the cell centre is the deterministic version of that.

    qpos: [N, D] rows (antmaze's npz qpos is 15 wide; only the first 2 columns are read).
    Returns: [N] float64, UNSHIFTED.
    """
    qpos = np.asarray(qpos, np.float64)
    goal_xy = np.asarray(goal_xy, np.float64).reshape(2)
    assert qpos.ndim == 2 and qpos.shape[-1] >= 2, f"qpos is {qpos.shape}, need >= 2 columns"
    dist = np.linalg.norm(qpos[:, :2] - goal_xy, axis=-1)
    return (dist <= threshold).astype(np.float64) - 1.0


def puzzle_relabel_reward(physics, goal_button_states):
    """td_jepa's `puzzle_reward_fn`, transcribed: (physics[:, -k:] == goal).sum(-1) - k for
    k = number of buttons -- 0 when every button matches, down to -k.

    physics: [N, D] td_jepa "physics" rows = qpos ++ button_states (see `_physics_rows`);
        the LAST k columns are the discrete button states. It is NOT the raw qpos: the
        buttons' physical joint positions (qpos[-9:] on puzzle-3x3) are never equal to the
        integer goal states, so a raw-qpos row scores -k everywhere.
    goal_button_states: [k] ints, `task_infos[t]['goal_button_states']`.
    Returns: [N] float64, UNSHIFTED. With the fb_tdj shift of 1.0 the shifted reward is
        (#matched - k + 1): 1 only when ALL buttons match, 0 for one miss, negative below --
        a graded count, not a 0/1 success indicator.
    """
    physics = np.asarray(physics)
    goal = np.asarray(goal_button_states).reshape(-1)
    k = goal.shape[0]
    assert physics.ndim == 2 and physics.shape[-1] >= k, (
        f"physics is {physics.shape}, need >= {k} columns for {k} buttons")
    return (physics[:, -k:] == goal).sum(-1).astype(np.float64) - k


# ogbench scene_env.py:72-77 -- td_jepa flattens `task_info['goal'].values()` in this order
# into its 7-vector target (cube xyz, button 1, button 2, drawer, window).
SCENE_GOAL_KEYS = ("block_xyzs", "button_states", "drawer_pos", "window_pos")


def scene_relabel_reward(physics, goal, threshold=0.04):
    """td_jepa's `scene_reward_fn`, transcribed: five conditions on the physics row --
    cube `physics[14:17]` within `threshold` of the goal cube xyz (inclusive), buttons
    `physics[-2]`, `physics[-1]` EQUAL to the goal button states, drawer `physics[-4]` and
    window `physics[-3]` within `threshold` of their goal positions (inclusive) -- minus 5.

    physics: [N, D] td_jepa "physics" rows = qpos ++ button_states (see `_physics_rows`).
        On scene-play-v0 that is qpos (25: arm 0:14, cube 14:21, fingers 21:23, drawer 23,
        window 24) ++ button_states (2), so -4/-3 are drawer/window and -2/-1 the buttons.
    goal: either `task_infos[t]['goal']` (dict in SCENE_GOAL_KEYS order) or the flattened
        7-vector td_jepa builds from it.
    Returns: [N] float64, UNSHIFTED. With the fb_tdj shift of 1.0 the shifted reward is
        (#satisfied - 4): 1 only when all five hold, 0 for one miss, negative below -- a
        graded count, not a 0/1 success indicator.
    """
    physics = np.asarray(physics, np.float64)
    if isinstance(goal, dict):
        assert tuple(goal.keys()) == SCENE_GOAL_KEYS, (
            f"scene goal dict must be ordered {SCENE_GOAL_KEYS} (ogbench scene_env.py); "
            f"got {tuple(goal.keys())}")
        target = np.concatenate([np.array([v]).ravel() for v in goal.values()])
    else:
        target = np.asarray(goal).reshape(-1)
    target = target.astype(np.float64)
    assert target.shape == (7,), f"scene target must flatten to 7 values, got {target.shape}"
    assert physics.ndim == 2 and physics.shape[-1] >= 17 + 4, (
        f"physics is {physics.shape}, need cube at 14:17 and 4 trailing object columns")
    cube = (np.linalg.norm(physics[:, 14:17] - target[:3], axis=-1) <= threshold).astype(np.float64)
    button1 = (physics[:, -2] == target[3]).astype(np.float64)
    button2 = (physics[:, -1] == target[4]).astype(np.float64)
    drawer = (np.abs(physics[:, -4] - target[5]) <= threshold).astype(np.float64)
    window = (np.abs(physics[:, -3] - target[6]) <= threshold).astype(np.float64)
    return cube + button1 + button2 + drawer + window - 5.0


# Family dispatch: (task_info goal key, needs button_states, reward fn). Mirrors td_jepa's
# get_relabel_fn (envs/ogbench.py:113-127), keyed on what ogbench's task_infos actually carry
# instead of on the env name: cube -> 'goal_xyzs', antmaze -> 'goal_xy',
# puzzle -> 'goal_button_states', scene -> 'goal' (a dict).
_RELABEL_FAMILIES = (
    ("goal_xyzs", False, lambda info, phys, thr: cube_relabel_reward(phys, info["goal_xyzs"], threshold=thr)),
    ("goal_xy", False, lambda info, phys, thr: antmaze_relabel_reward(phys, info["goal_xy"])),
    ("goal_button_states", True, lambda info, phys, thr: puzzle_relabel_reward(phys, info["goal_button_states"])),
    ("goal", True, lambda info, phys, thr: scene_relabel_reward(phys, info["goal"], threshold=thr)),
)


def _relabel_family(task_info):
    for key, needs_buttons, fn in _RELABEL_FAMILIES:
        if key in task_info:
            return key, needs_buttons, fn
    raise AssertionError(
        f"relabel_infer_batch: task_info has none of the supported goal keys "
        f"{[k for k, _, _ in _RELABEL_FAMILIES]} (cube goal_xyzs / antmaze goal_xy / puzzle "
        f"goal_button_states / scene goal); keys: {sorted(task_info)} -- this domain is unsupported")


def relabel_reward_for_task(task_info, physics, threshold=0.04):
    """UNSHIFTED relabel reward of `physics` rows under `task_info`'s family. `threshold` is
    the cube/scene distance tolerance (antmaze uses its own 0.5, puzzle is exact-match)."""
    _, _, fn = _relabel_family(task_info)
    return fn(task_info, physics, threshold)


class OGBenchAdapter:
    name = "ogbench"

    def make_env_and_datasets(self, cfg):
        # `measure_qpos_slices` is density_fb's PRIVILEGED MEASURE CHANNEL on pixels; it is
        # the only thing in this tree that needs the npz's raw 'qpos' column, so it is also
        # what gates loading it (state runs keep their exact batch keys). `eval_relabel_size`
        # (fb_tdj parity arm, above) needs the same column to relabel a state run's play data.
        env, train_ds, val_ds = make_env_and_datasets(
            cfg["env_name"], cfg.get("frame_stack"),
            add_info=(cfg.get("measure_qpos_slices") is not None
                      or bool(cfg.get("eval_relabel_size"))))
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

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video, any_step_success=False):
        if task is not None:
            # Original per-task_id behaviour, unchanged (any_step_success defaults False).
            stats, _, renders = ogbench_evaluate(
                agent, env, task_id=task, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video,
                any_step_success=any_step_success)
            return {k: float(v) for k, v in dict(stats).items()}, renders
        # Aggregate: mean success over all task_infos, one scalar (no at-goal ticks: Box2D-only).
        import numpy as np
        n_tasks = len(env.unwrapped.task_infos)
        succ, renders = [], []
        for tid in range(1, n_tasks + 1):
            stats, _, r = ogbench_evaluate(
                agent, env, task_id=tid, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video if tid == 1 else 0,
                any_step_success=any_step_success)
            succ.append(float(dict(stats).get("success", next(iter(dict(stats).values())))))
            renders += list(r)
        return {"success": float(np.mean(succ))}, renders

    @staticmethod
    def _column_backed(ds):
        """The raw column-backed dataset under any per-critic sampler wrapper.

        `impls/main.py` hands the eval the dataset AFTER `make_dataset` has wrapped it, and
        the wrappers (HILPFutureDataset, GCDataset, WindowedDataset, ...) implement `sample`
        but NOT the `"col" in ds` / `ds["col"]` column access this relabel needs -- each keeps
        its base in `.dataset` and reaches through it the same way (datasets.py:33 does
        `self.dataset['terminals']`). Without this, `eval_relabel_size` crashed every HILP
        arm with `TypeError: argument of type 'HILPFutureDataset' is not iterable`, while
        fb_tdj worked only because its dataset_class leaves the base unwrapped.
        """
        cur = ds
        for _ in range(8):
            try:
                if "observations" in cur:
                    return cur
            except TypeError:
                pass
            nxt = getattr(cur, "dataset", None)
            if nxt is None or nxt is cur:
                break
            cur = nxt
        return cur

    @staticmethod
    def _physics_rows(train_ds, idxs, needs_buttons):
        """td_jepa's "physics" row for dataset rows `idxs`: the npz `qpos`, CONCATENATED with
        `button_states` when the family needs it (scene, puzzle) -- exactly how td_jepa
        builds the array its reward functions index from the end
        (scripts/data_processing/ogbench/extract_all.py:56-58). Cube and antmaze read plain
        qpos, so their rows are byte-identical to before."""
        qpos = np.asarray(train_ds["qpos"][idxs], np.float64)
        if not needs_buttons:
            return qpos
        assert "button_states" in train_ds, (
            "this family's relabel reward reads discrete 'button_states', which ogbench's "
            "loader ships alongside 'qpos' under add_info=True for scene/puzzle npz files -- "
            "a dataset without it here means the add_info wiring did not run")
        buttons = np.asarray(train_ds["button_states"][idxs], np.float64)
        return np.concatenate([qpos, buttons], axis=-1)

    def relabel_infer_batch(self, train_ds, env, task_id, num_samples, shift_reward=1.0,
                            threshold=0.04):
        """(next_observations, rewards) ready for `agent.infer_eval_cond` -- td_jepa's OGBench
        reward-relabel protocol (docs/notes/2026-09-03-fb-vs-tdjepa.md #7). Draws
        `num_samples` random transitions from the TRAIN dataset and relabels each by its NEXT
        observation's stored simulator state against task `task_id`'s goal -- never the
        dataset's own 'rewards' column, which FBDataset zeroes for unlabelled play data
        (impls/utils/datasets.py:125-128).

        The family is picked from the task_info's goal key (`relabel_reward_for_task`):
        cube 'goal_xyzs', antmaze 'goal_xy', puzzle 'goal_button_states', scene 'goal'.
        `shift_reward` is added verbatim, as td_jepa does (`rewards += shift_reward`): with
        the fb_tdj value 1.0 the shifted reward is (#conditions satisfied - k + 1), a 0/1
        success indicator ONLY for the single-condition families (cube-single, antmaze); on
        scene (k=5) and puzzle (k=9) it is a graded count that goes negative.
        """
        train_ds = self._column_backed(train_ds)
        assert "qpos" in train_ds, (
            "eval_relabel_size requires the dataset to carry 'qpos' -- "
            "OGBenchAdapter.make_env_and_datasets forces add_info=True whenever "
            "cfg['eval_relabel_size'] is set, so a dataset without 'qpos' here means that "
            "wiring did not run (e.g. a hand-built cfg in a test)")
        task_info = env.unwrapped.task_infos[task_id - 1]
        _, needs_buttons, reward_fn = _relabel_family(task_info)
        idxs = train_ds.get_random_idxs(int(num_samples))
        next_idxs = np.minimum(idxs + 1, train_ds.size - 1)
        next_obs = np.asarray(train_ds["observations"][next_idxs], np.float32)
        next_physics = self._physics_rows(train_ds, next_idxs, needs_buttons)
        rewards = reward_fn(task_info, next_physics, threshold)
        rewards = (rewards + float(shift_reward)).astype(np.float32)
        return next_obs, rewards
