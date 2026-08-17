#!/bin/bash
# E4 of docs/specs/2026-08-10-deconfound-size-coverage.md — regenerate the goal-coverage
# figure from v3 MW / the E2 MS data / cube, and produce the scalar in-supportness metric the
# figure was standing in for.
#
#   sbatch -A ASC26008 scripts/goal_coverage.slurm
#
# Needs a compute node: the extract pass builds each env to draw its eval goals the way
# evaluate() does, and the three families need three different venvs.
#
#SBATCH -J fb-goal-coverage
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
mkdir -p logs results/deconfound

PY_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}/bin/python"
PY_MS="${VENV_MS:-$SCRATCH/.venv-ms}/bin/python"
MW_ROOT="${MW_ROOT:-$SCRATCH/fb-data/play-resample}"
MS_ROOT="${MS_ROOT:-$SCRATCH/fb-data/ms-play-resample}"
EPISODES="${EPISODES:-50}"          # must match the square's --eval_episodes
OUT="${OUT:-results/deconfound}"
FIG="${FIG:-PAPER/decoding-bilinear/figs/goal_coverage.png}"

export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# The v1-based figure is archived, not overwritten: the paper currently cites it, and a
# silent replacement would leave no way to tell which data a printed figure came from.
if [ -f "$FIG" ] && [ ! -f "${FIG%.png}-v1.png" ]; then
    cp "$FIG" "${FIG%.png}-v1.png"
    echo "== archived $FIG -> ${FIG%.png}-v1.png =="
fi

fail=0
$PY_JAX scripts/goal_coverage.py extract --task cube-single \
    --episodes "$EPISODES" --out "$OUT/cov-cube-single.npz" || fail=1
for t in mw-push mw-pick-place; do
    $PY_JAX scripts/goal_coverage.py extract --task "$t" --data-root "$MW_ROOT" \
        --episodes "$EPISODES" --out "$OUT/cov-$t.npz" || fail=1
done
for t in ms-push-cube ms-pick-cube; do
    if [ -f "$MS_ROOT/data/maniskill/$t.npz" ]; then
        $PY_MS scripts/goal_coverage.py extract --task "$t" --data-root "$MS_ROOT" \
            --episodes "$EPISODES" --out "$OUT/cov-$t.npz" || fail=1
    else
        echo "SKIP $t: no $MS_ROOT/data/maniskill/$t.npz (E2 collection has not run)"
    fi
done

$PY_JAX scripts/goal_coverage.py report --in "$OUT/cov-*.npz" \
    --fig "$FIG" --json-out "$OUT/goal-coverage.json" || fail=1

echo "== goal coverage done: $fail failure(s) =="
exit $fail
