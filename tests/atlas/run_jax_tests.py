"""Runs the jax-venv atlas tests (mu_flow, critics, build_atlas) as plain functions,
because /dev/shm/.venv-jax has no pytest. Non-jax tests use pytest under the repo .venv.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="" \
    /dev/shm/.venv-jax/bin/python tests/atlas/run_jax_tests.py [module ...]
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODULES = sys.argv[1:] or ['test_mu_flow', 'test_critics', 'test_build_atlas']

failed = 0
for modname in MODULES:
    try:
        mod = importlib.import_module(f'tests.atlas.{modname}')
    except ModuleNotFoundError:
        print(f'SKIP  {modname} (not present yet)')
        continue
    for name in dir(mod):
        if name.startswith('test_'):
            try:
                getattr(mod, name)()
                print(f'PASS  {modname}.{name}')
            except Exception:
                failed += 1
                print(f'FAIL  {modname}.{name}')
                traceback.print_exc()

print(f'\n{"ALL PASS" if failed == 0 else str(failed) + " FAILED"}')
sys.exit(1 if failed else 0)
