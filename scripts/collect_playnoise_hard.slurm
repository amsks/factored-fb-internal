#!/bin/bash
# Collect the HARD play-noise regime for the headroom kill-test (phenomenon.tex
# Pred. headroom): same scripted-oracle collector as play-noise, but expert_frac=0
# (no clean episodes) and a hotter exploration mixture, so the BC floor drops and the
# benchmark measures guidance quality again. Everything else (horizon, per-episode
# target, dwell) is untouched ON PURPOSE — this arm isolates Ph.I from Ph.II/III.
#
#   sbatch -A ASC26008 scripts/collect_playnoise_hard.slurm
#
#SBATCH -J collect-pnh
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
OUT="${OUT:-$SCRATCH/fb-data/play-noise-hard}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

EPS_MAX="${EPS_MAX:-0.5}"
NOISE_MAX="${NOISE_MAX:-0.3}"
EXPERT_FRAC="${EXPERT_FRAC:-0.0}"

FAIL=0
for task in mw-push mw-pick-place; do
  npz="$OUT/data/metaworld/$task.npz"
  if [ -f "$npz" ]; then echo "== skip $task (exists)"; continue; fi
  echo "== collect $task (eps_max=$EPS_MAX noise_max=$NOISE_MAX expert_frac=$EXPERT_FRAC) =="
  $PY data_collection/collect_mw_standalone.py \
      --task "$task" --regime play-noise --episodes "${EPISODES:-1000}" --seed 0 \
      --eps_max "$EPS_MAX" --noise_max "$NOISE_MAX" --expert_frac "$EXPERT_FRAC" \
      --out "$OUT" || { echo "!! FAILED collect $task"; FAIL=1; }
done

# Dataset audit — the numbers Ph.I/Ph.III read off the data (success, dwell fraction).
$PY - <<EOF
import numpy as np
for task in ("mw-push", "mw-pick-place"):
    p = "$OUT/data/metaworld/%s.npz" % task
    try:
        z = np.load(p, allow_pickle=True)
    except FileNotFoundError:
        continue
    su = z["success"]
    print(f"{task}: episodes {su.shape[0]}, ep-success {su.any(1).mean():.3f}, "
          f"dwell(frac steps at success) {su.mean():.3f}")
EOF
echo "== DONE (FAIL=$FAIL) out=$OUT =="
exit $FAIL
