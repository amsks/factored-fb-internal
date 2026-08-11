"""Spec 23 P0 blocker close: the ExoRL dataset-loading extension point
(`tools/wandb_mode_shim/exorl_env.py`, installed by
`tools/wandb_mode_shim/sitecustomize.py`).

The extension point wraps `utils.env_utils.make_env_and_datasets` (vendored,
third_party/ogbench/impls/utils/env_utils.py -- NEVER edited, see
PROVENANCE.md). The one property that matters most: every env_name NOT
starting with "exorl:" must reach the ORIGINAL vendored function with
UNCHANGED arguments -- byte-identical default path for every existing
OGBench config. That's exercised here by stubbing
`ogbench.make_env_and_datasets` (the function the original vendored loader
calls) and recording what it was called with, entirely inside the isolated
jax venv (GCDataset/utils.env_utils live under third_party/ogbench/impls,
jax-only) via the repo's existing jax-subprocess test helper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exorl"))

from tests.jax_subprocess import run_jax  # noqa: E402


def test_non_exorl_env_name_reaches_original_function_unchanged():
    """The default path: any OGBench env_name must call through to the
    ORIGINAL ogbench.make_env_and_datasets with dataset_name and
    compact_dataset=True exactly as the untouched vendored code does."""
    proc = run_jax(
        """
        import numpy as np
        import ogbench
        import utils.env_utils as env_utils

        calls = []

        class _DummyEnv:
            def reset(self, *a, **k):
                return None, {}

        def _stub_make_env_and_datasets(dataset_name, compact_dataset=None):
            calls.append((dataset_name, compact_dataset))
            dummy = {"observations": np.array([0.0])}
            return _DummyEnv(), dummy, dict(dummy)

        ogbench.make_env_and_datasets = _stub_make_env_and_datasets

        env, train, val = env_utils.make_env_and_datasets("cube-single-play-v0", frame_stack=None)
        assert calls == [("cube-single-play-v0", True)], calls
        assert "observations" in train
        print("NON_EXORL_ROUTES_TO_ORIGINAL_OK")
        """
    )
    assert "NON_EXORL_ROUTES_TO_ORIGINAL_OK" in proc.stdout, (proc.stdout, proc.stderr)


def test_exorl_prefixed_name_never_reaches_ogbench_make_env_and_datasets():
    """An exorl: name must NOT touch gymnasium.make/ogbench at all (it has
    no registered gymnasium id) -- assert the stub is never called for it,
    even though the real load fails (dataset not necessarily on this box /
    an intentionally unregistered key), by using an unregistered key so we
    only need to check dispatch, not the full data-loading path."""
    proc = run_jax(
        """
        import ogbench
        import utils.env_utils as env_utils

        calls = []
        ogbench.make_env_and_datasets = lambda *a, **k: calls.append((a, k))

        try:
            env_utils.make_env_and_datasets("exorl:not_a_registered_key", frame_stack=None)
            raised = False
        except ValueError as e:
            raised = True
            msg = str(e)

        assert raised, "unregistered exorl: key must raise, not silently fall through"
        assert "not_a_registered_key" in msg
        assert calls == [], f"ogbench.make_env_and_datasets must never be called for exorl: names, got {calls}"
        print("EXORL_NEVER_TOUCHES_OGBENCH_OK")
        """
    )
    assert "EXORL_NEVER_TOUCHES_OGBENCH_OK" in proc.stdout, (proc.stdout, proc.stderr)


def test_frame_stack_unsupported_for_exorl_raises_clearly():
    proc = run_jax(
        """
        import utils.env_utils as env_utils
        try:
            env_utils.make_env_and_datasets("exorl:point_mass_maze_rnd_seed0", frame_stack=4)
            raised = False
        except NotImplementedError as e:
            raised = True
        assert raised
        print("EXORL_FRAME_STACK_REJECTED_OK")
        """
    )
    assert "EXORL_FRAME_STACK_REJECTED_OK" in proc.stdout, (proc.stdout, proc.stderr)


REAL_DATA_DIR = Path.home() / "exorl_data" / "converted" / "point_mass_maze_rnd_seed0"


def test_registered_exorl_key_loads_real_dataset_and_a_working_eval_env():
    """End-to-end (skips if the converted dataset isn't on this box): the
    registered point_mass_maze_rnd_seed0 key must load the real converted
    GCDataset-format arrays and produce an env whose reset/step contract
    matches what vendored utils/evaluation.py::evaluate() needs."""
    if not (REAL_DATA_DIR / "point_mass_maze_rnd.npz").exists():
        import pytest

        pytest.skip("converted ExoRL dataset not present on this box")
    proc = run_jax(
        """
        import numpy as np
        import utils.env_utils as env_utils

        env, train, val = env_utils.make_env_and_datasets("exorl:point_mass_maze_rnd_seed0", frame_stack=None)
        assert "observations" in train and "actions" in train and "terminals" in train
        assert train["observations"].shape[1] == 4
        assert len(env.task_infos) == 20

        obs, info = env.reset(options=dict(task_id=1, render_goal=False))
        assert obs.shape == (4,)
        assert info["goal"].shape == (4,)
        assert info["success"] in (0.0, 1.0)

        obs2, reward, terminated, truncated, info2 = env.step(np.zeros(2, dtype=np.float32))
        assert obs2.shape == (4,)
        assert np.isfinite(reward)
        assert terminated is False
        assert isinstance(truncated, bool)
        print("EXORL_REAL_LOAD_AND_ENV_OK")
        """
    )
    assert "EXORL_REAL_LOAD_AND_ENV_OK" in proc.stdout, (proc.stdout, proc.stderr)
