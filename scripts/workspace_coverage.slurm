#!/bin/bash
# Top-down orthographic renders for the workspace/coverage figure. Needs a GPU (EGL for
# Metaworld+OGBench, Vulkan for ManiSkill), hence a compute node; the report pass that
# assembles the figure is numpy+matplotlib and runs on a login node.
#
#   sbatch -A ASC26008 scripts/workspace_coverage.slurm
#
#SBATCH -J fb-workspace
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"

PY_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}/bin/python"
PY_MS="${VENV_MS:-$SCRATCH/.venv-ms}/bin/python"
OUT="${OUT:-results/deconfound}"
export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES=0

# Metaworld + OGBench share the jax venv; ManiSkill needs its own and may not expose a raw
# mujoco model at all (SAPIEN backend) -- it is allowed to fail without failing the job, and
# the report pass simply omits any env with no render.
$PY_JAX scripts/workspace_coverage.py render --out "$OUT" \
    --tasks mw-push mw-pick-place cube-single cube-double scene antmaze-medium
$PY_MS  scripts/workspace_coverage.py render --out "$OUT" \
    --tasks ms-push-cube ms-pick-cube

echo "== renders present =="
ls -la "$OUT"/ws-*.npz 2>/dev/null || echo "none"
