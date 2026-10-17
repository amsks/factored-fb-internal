"""OGBench adapter with goal-conditioned eval and reward relabeling for reward-inference eval.

Relabel rewards follow the TD-JEPA OGBench protocol.
"""

import numpy as np

from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate as ogbench_evaluate

# First qpos column of each cube's 7-wide free joint
CUBE_QPOS_OFFSETS = (14, 21, 28, 35)


def cube_relabel_reward(qpos, goal_xyzs, threshold=0.04):
    """Cube reward: number of cubes within `threshold` of their goal, minus the number of cubes.

    :param qpos: [N, D] simulator qpos rows.
    :param goal_xyzs: [num_cubes, 3] goal cube positions.
    :return: [N] unshifted rewards.
    """
    qpos = np.asarray(qpos, np.float64)
    goal_xyzs = np.asarray(goal_xyzs, np.float64).reshape(-1, 3)
    num_cubes = goal_xyzs.shape[0]
    assert 1 <= num_cubes <= len(CUBE_QPOS_OFFSETS), (
        f"cube_relabel_reward supports 1-{len(CUBE_QPOS_OFFSETS)} cubes, got {num_cubes}"
    )
    need = CUBE_QPOS_OFFSETS[num_cubes - 1] + 3
    assert qpos.ndim == 2 and qpos.shape[-1] >= need, (
        f"qpos is {qpos.shape} wide, need at least {need} columns for {num_cubes} cube(s)"
    )
    successes = np.zeros(qpos.shape[0], np.float64)
    for i in range(num_cubes):
        off = CUBE_QPOS_OFFSETS[i]
        dist = np.linalg.norm(qpos[:, off : off + 3] - goal_xyzs[i], axis=-1)
        successes += (dist < threshold).astype(np.float64)
    return successes - num_cubes


# Flattening order of the scene goal: cube xyz, button 1, button 2, drawer, window
SCENE_GOAL_KEYS = ("block_xyzs", "button_states", "drawer_pos", "window_pos")


def scene_relabel_reward(physics, goal, threshold=0.04):
    """Scene reward: number of satisfied conditions (cube, two buttons, drawer, window) minus 5.

    :param physics: [N, D] qpos rows with the two button states appended.
    :param goal: Goal dict in SCENE_GOAL_KEYS order, or the flattened 7-vector.
    :return: [N] unshifted rewards.
    """
    physics = np.asarray(physics, np.float64)
    if isinstance(goal, dict):
        assert tuple(goal.keys()) == SCENE_GOAL_KEYS, (
            f"scene goal dict must be ordered {SCENE_GOAL_KEYS} (ogbench scene_env.py); "
            f"got {tuple(goal.keys())}"
        )
        target = np.concatenate([np.array([v]).ravel() for v in goal.values()])
    else:
        target = np.asarray(goal).reshape(-1)
    target = target.astype(np.float64)
    assert target.shape == (7,), (
        f"scene target must flatten to 7 values, got {target.shape}"
    )
    assert physics.ndim == 2 and physics.shape[-1] >= 17 + 4, (
        f"physics is {physics.shape}, need cube at 14:17 and 4 trailing object columns"
    )
    cube = (
        np.linalg.norm(physics[:, 14:17] - target[:3], axis=-1) <= threshold
    ).astype(np.float64)
    button1 = (physics[:, -2] == target[3]).astype(np.float64)
    button2 = (physics[:, -1] == target[4]).astype(np.float64)
    drawer = (np.abs(physics[:, -4] - target[5]) <= threshold).astype(np.float64)
    window = (np.abs(physics[:, -3] - target[6]) <= threshold).astype(np.float64)
    return cube + button1 + button2 + drawer + window - 5.0


# (task_info goal key, needs button_states, reward fn) per task family
_RELABEL_FAMILIES = (
    (
        "goal_xyzs",
        False,
        lambda info, phys, thr: cube_relabel_reward(
            phys, info["goal_xyzs"], threshold=thr
        ),
    ),
    (
        "goal",
        True,
        lambda info, phys, thr: scene_relabel_reward(phys, info["goal"], threshold=thr),
    ),
)


def _relabel_family(task_info):
    for key, needs_buttons, fn in _RELABEL_FAMILIES:
        if key in task_info:
            return key, needs_buttons, fn
    raise AssertionError(
        f"relabel_infer_batch: task_info has none of the supported goal keys "
        f"{[k for k, _, _ in _RELABEL_FAMILIES]} (cube goal_xyzs / scene goal); keys: "
        f"{sorted(task_info)} -- this domain is unsupported"
    )


def relabel_reward_for_task(task_info, physics, threshold=0.04):
    """Unshifted relabel reward of `physics` rows for this task."""
    _, _, fn = _relabel_family(task_info)
    return fn(task_info, physics, threshold)


class OGBenchAdapter:
    name = "ogbench"

    def make_env_and_datasets(self, cfg):
        # Load qpos only for reward-inference eval
        env, train_ds, val_ds = make_env_and_datasets(
            cfg["env_name"], add_info=bool(cfg.get("eval_relabel_size"))
        )
        # Episode length from the TimeLimit wrapper
        self._horizon = (
            int(
                getattr(env, "_max_episode_steps", None)
                or (env.spec.max_episode_steps if env.spec is not None else 0)
                or 0
            )
            or None
        )
        return env, train_ds, val_ds

    def eval_tasks(self, env, cfg):
        return list(range(1, len(env.unwrapped.task_infos) + 1))

    def evaluate(
        self, agent, env, task, cfg, num_episodes, num_video=0, any_step_success=False
    ):
        if num_video:
            raise ValueError("video rendering is not supported; pass num_video=0")
        stats, _ = ogbench_evaluate(
            agent,
            env,
            task_id=task,
            config=cfg,
            num_eval_episodes=num_episodes,
            any_step_success=any_step_success,
        )
        return {k: float(v) for k, v in dict(stats).items()}, []

    @staticmethod
    def _column_backed(ds):
        """Return the column-backed dataset under any sampler wrapper."""
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
        """qpos rows, with button states appended for scene."""
        qpos = np.asarray(train_ds["qpos"][idxs], np.float64)
        if not needs_buttons:
            return qpos
        assert "button_states" in train_ds, (
            "this family's relabel reward reads discrete 'button_states', which ogbench's "
            "loader ships alongside 'qpos' under add_info=True for scene npz files"
        )
        buttons = np.asarray(train_ds["button_states"][idxs], np.float64)
        return np.concatenate([qpos, buttons], axis=-1)

    def relabel_infer_batch(
        self, train_ds, env, task_id, num_samples, shift_reward=1.0, threshold=0.04
    ):
        """Sample transitions and relabel their next states with the task reward.

        :return: (next_observations, rewards + shift_reward) for `agent.infer_eval_cond`.
        """
        train_ds = self._column_backed(train_ds)
        assert "qpos" in train_ds, (
            "eval_relabel_size requires the dataset to carry 'qpos' -- "
            "make_env_and_datasets loads it whenever cfg['eval_relabel_size'] is set"
        )
        task_info = env.unwrapped.task_infos[task_id - 1]
        _, needs_buttons, reward_fn = _relabel_family(task_info)
        idxs = train_ds.get_random_idxs(int(num_samples))
        next_idxs = np.minimum(idxs + 1, train_ds.size - 1)
        next_obs = np.asarray(train_ds["observations"][next_idxs], np.float32)
        next_physics = self._physics_rows(train_ds, next_idxs, needs_buttons)
        rewards = reward_fn(task_info, next_physics, threshold)
        rewards = (rewards + float(shift_reward)).astype(np.float32)
        return next_obs, rewards
