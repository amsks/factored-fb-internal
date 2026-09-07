import os
import numpy as np
import pytest

# x64 MUST be enabled before jax is first used anywhere in the test session.
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "fb_reference.npz")


@pytest.fixture(scope="session")
def fb_fixture():
    assert os.path.exists(FIXTURE), f"missing {FIXTURE}; copy it in Task 2"
    return dict(np.load(FIXTURE))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ogbench_dataset: needs a real OGBench npz under ~/.ogbench/data (skipped when absent)")
