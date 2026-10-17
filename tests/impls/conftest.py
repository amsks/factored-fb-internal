import os

# Enable x64 before jax is first used.
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ogbench_dataset: needs a real OGBench npz under ~/.ogbench/data (skipped when absent)")
