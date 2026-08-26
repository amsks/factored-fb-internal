"""eval/humenv_goal.py — HumEnv's goal metric, in numpy.

WHY A PORT AND NOT `humenv.bench`. The benchmark's own implementation lives in
`humenv/bench/utils/metrics.py`, which imports **torch** at module scope. Our agents are jax
and live in `.venv-jax-fb`, which has `humenv` and `mujoco` but no torch; the torch stack is
in the separate `.venv-humenv`. Rather than install torch into the venv every training run
depends on, or marshal a policy across two interpreters, the metric itself is ported -- it is
pure tensor arithmetic with no learned parts, so a faithful port is cheap and checkable.

`tests/impls/test_humenv_goal_parity.py` asserts this file agrees with the torch original on
random inputs, run inside `.venv-humenv` where both are importable. Treat that test as the
definition of "faithful"; do not edit the constants here without re-running it.

THE METRIC (metrics.py::distance_smpl / get_episode_goal_stats, defaults bound=2.0,
margin=2.0):

    dist       = || next_obs[..., :214] - goal[..., :214] ||_2
    success    = dist <= bound at ANY step of the episode          (latching, like ExoRL's)
    proximity  = per-step mean of a shaped credit that is 1.0 inside `bound`, ramps linearly
                 to 0.0 at `bound + margin`, and is 0.0 beyond
    distance   = per-step mean of dist

`:214` is the SMPL pose block. env/humenv_env.py::POSE_BLOCK carries the same constant so a
domain's measure_hit_spec and this metric cannot silently disagree about which columns "the
pose" means.
"""
import numpy as np

POSE_DIM = 214
BOUND = 2.0
MARGIN = 2.0


def distance_smpl(next_obs, goal):
    """|| next_obs[:214] - goal[:214] ||_2, over the trailing axis."""
    a = np.asarray(next_obs)[..., :POSE_DIM]
    b = np.asarray(goal)[..., :POSE_DIM]
    return np.linalg.norm(a - b, axis=-1)


def episode_goal_stats(next_obs, goal, bound=BOUND, margin=MARGIN):
    """-> dict(success, proximity, distance) for ONE episode.

    `next_obs` is [T, obs_dim] (the observations AFTER each step, i.e. the rollout's
    observation array with the reset frame dropped) and `goal` is [obs_dim] or [T, obs_dim].
    """
    dist = distance_smpl(next_obs, goal)
    in_b = dist <= bound
    out_b = dist > bound + margin
    ramp = ((bound + margin - dist) / margin) * (~in_b) * (~out_b)
    return {
        "success": float(np.any(in_b)),
        "proximity": float(np.mean(in_b.astype(np.float64) + ramp)),
        "distance": float(np.mean(dist)),
    }


def aggregate(stats_list):
    """Mean of each key over episodes, plus the seed-level spread."""
    out = {}
    for k in ("success", "proximity", "distance"):
        v = np.asarray([s[k] for s in stats_list], dtype=np.float64)
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
    out["n_episodes"] = len(stats_list)
    return out
