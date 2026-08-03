#!/bin/bash
# Collect the PLAY-RESAMPLE regime (phenomenon.tex Pred. kernel(b)): OGBench-style long
# no-dwell trajectories that chain target relocations, via
# data_collection/collect_mw_play_resample.py. Equalizes E7 (horizon, dwell, goal
# diversity, kernel range) between Metaworld and cube-play; the eval-side goal channel
# (E8/Ph.II) is deliberately untouched.
#
#   sbatch -A ASC26008 scripts/collect_play_resample.slurm
#
#SBATCH -J collect-presample
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
OUT="${OUT:-$SCRATCH/fb-data/play-resample}"
EPS_MAX="${EPS_MAX:-0.1}"
NOISE_MAX="${NOISE_MAX:-0.1}"
EXPERT_FRAC="${EXPERT_FRAC:-0.0}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

FAIL=0
for task in mw-push mw-pick-place; do
  npz="$OUT/data/metaworld/$task.npz"
  if [ -f "$npz" ]; then echo "== skip $task (exists)"; continue; fi
  echo "== collect $task (play-resample eps=$EPS_MAX noise=$NOISE_MAX clean=$EXPERT_FRAC) =="
  $PY data_collection/collect_mw_play_resample.py \
      --task "$task" --episodes 200 --seed 0 --out "$OUT" \
      --eps_max "$EPS_MAX" --noise_max "$NOISE_MAX" --expert_frac "$EXPERT_FRAC" \
      || { echo "!! FAILED collect $task"; FAIL=1; }
done
echo "== DONE (FAIL=$FAIL) out=$OUT =="
exit $FAIL
