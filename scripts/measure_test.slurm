#!/bin/bash
# Run the impls/ pytest suite for the successor-measure critics (density_fb, td_fb) plus
# the full existing tests/impls/ suite, on a compute node -- login nodes cannot run JAX
# (Vista's ~100 thread-per-user cap aborts the Eigen thread pool before the first step).
#
#   sbatch -A ASC26008 scripts/measure_test.slurm
#
#SBATCH -J fb-measure-test
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
mkdir -p logs
echo "== host: $(hostname)  code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
PY="$VENV/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu

$PY -c 'import jax; print("jax", jax.__version__, "devices:", jax.devices())'

echo "---- new tests: density_fb / td_fb ----"
$PY -m pytest tests/impls/test_density_fb.py tests/impls/test_td_fb.py -q
new_status=$?

echo "---- full tests/impls/ suite ----"
$PY -m pytest tests/impls/ -q
full_status=$?

echo "== new tests exit=$new_status  full suite exit=$full_status =="
exit $(( new_status != 0 || full_status != 0 ))
