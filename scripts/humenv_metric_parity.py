"""Check eval/humenv_goal.py against humenv.bench's torch metric.

Runs as a PLAIN SCRIPT because `.venv-humenv` -- the only venv with torch -- has no pytest:

    PYTHONPATH=$PWD $SCRATCH/.venv-humenv/bin/python scripts/humenv_metric_parity.py

tests/impls/test_humenv_goal_parity.py wraps these same functions so the checks also run
under pytest wherever torch happens to be importable. One implementation, two entry points.
"""
import numpy as np

from eval.humenv_goal import distance_smpl, episode_goal_stats, BOUND, MARGIN, POSE_DIM

OBS, T = 358, 40


def _episode(rng, scale):
    goal = rng.normal(size=(OBS,)).astype(np.float32)
    return goal[None] + scale * rng.normal(size=(T, OBS)).astype(np.float32), goal


def check_distance(metrics, torch, scale):
    rng = np.random.default_rng(0)
    next_obs, goal = _episode(rng, scale)
    theirs = metrics.distance_smpl(
        torch.tensor(next_obs), torch.tensor(np.broadcast_to(goal, next_obs.shape).copy()))
    np.testing.assert_allclose(distance_smpl(next_obs, goal), theirs.numpy(),
                               rtol=1e-6, atol=1e-6)


def check_episode_stats(metrics, torch, scale):
    """Their get_episode_goal_stats folds success/proximity through cumsum + done-mask
    differencing over a CONCATENATED multi-episode buffer; on one episode that reduces to
    any()/mean(), which is what we assert against the same shaped-credit expression."""
    rng = np.random.default_rng(1)
    next_obs, goal = _episode(rng, scale)
    ours = episode_goal_stats(next_obs, goal)
    g = torch.tensor(np.broadcast_to(goal, next_obs.shape).copy())
    dist = metrics.distance_smpl(torch.tensor(next_obs), g)
    in_b = dist <= BOUND
    out_b = dist > BOUND + MARGIN
    prox = in_b + ((BOUND + MARGIN - dist) / MARGIN) * (~in_b) * (~out_b)
    assert ours["success"] == float(in_b.any()), (ours["success"], float(in_b.any()))
    # TOLERANCE. Both sides carry float32 observations through a 358-dim norm and then a
    # T-element mean, so the two differ by float32 accumulation order, not by formula: the
    # observed gap is 3e-9 absolute / 3.6e-6 relative on a proximity of 8.4e-4, against a
    # float32 eps of ~1.2e-7 amplified by the reduction. rtol 1e-5 is that noise floor with
    # margin; it is NOT slack for a formula difference, which would show up as a gap orders
    # of magnitude larger (and does, if you change bound/margin by a hair).
    np.testing.assert_allclose(ours["proximity"], float(prox.double().mean()),
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(ours["distance"], float(dist.double().mean()),
                               rtol=1e-5, atol=1e-8)


def check_published_defaults(metrics):
    """A silent upstream change to bound/margin would move every number we report."""
    import inspect
    sig = inspect.signature(metrics.get_episode_goal_stats)
    assert sig.parameters["bound"].default == BOUND
    assert sig.parameters["margin"].default == MARGIN


def main():
    import torch
    from humenv.bench.utils import metrics
    n = 0
    for s in (0.0, 0.05, 0.2, 1.0):
        check_distance(metrics, torch, s); n += 1
    for s in (0.02, 0.1, 0.3, 2.0):
        check_episode_stats(metrics, torch, s); n += 1
    check_published_defaults(metrics); n += 1
    # The POSE_BLOCK-agrees-with-env check lives in tests/impls/test_humenv_goal.py instead:
    # it needs neither torch nor humenv, and importing env.humenv_env from here would pull
    # env/__init__.py -> omegaconf, which .venv-humenv does not have.
    print(f"PARITY OK: {n} checks -- eval/humenv_goal.py matches humenv.bench")


if __name__ == "__main__":
    main()
