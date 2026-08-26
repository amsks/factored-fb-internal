"""pytest wrapper over scripts/humenv_metric_parity.py.

Skipped wherever torch is absent -- which includes `.venv-jax-fb`, the venv the rest of this
suite runs in, and the very reason eval/humenv_goal.py is a numpy port in the first place.
Run the checks there with the script directly (see its docstring).
"""
import pytest

torch = pytest.importorskip("torch", reason="parity needs the humenv venv's torch")
metrics = pytest.importorskip("humenv.bench.utils.metrics")

from scripts.humenv_metric_parity import (check_distance, check_episode_stats,
                                          check_published_defaults,
                                          check_pose_block_agrees_with_env)


@pytest.mark.parametrize("scale", [0.0, 0.05, 0.2, 1.0])
def test_distance(scale):
    check_distance(metrics, torch, scale)


@pytest.mark.parametrize("scale", [0.02, 0.1, 0.3, 2.0])
def test_episode_stats(scale):
    check_episode_stats(metrics, torch, scale)


def test_published_defaults():
    check_published_defaults(metrics)


def test_pose_block():
    check_pose_block_agrees_with_env()
