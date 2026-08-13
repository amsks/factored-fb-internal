#!/usr/bin/env bash
# Spec 07 items 2+3 driver: fit renewal heads (psi + cube inputs, + FP_mu, + fidelity
# eval) on the frozen FP-NoOrtho checkpoints, then the INFERENCE-ONLY zero-shot pass
# (renewal vs plain z-weighting, with GPI). CPU-only: zero GPU contention with training.
set -uo pipefail
cd "$(dirname "$0")/../.."
# jax scripts MUST use the /dev/shm jax venv (repo .venv has no jax — the e3 gotcha;
# first launch 2026-07-22 died on ModuleNotFoundError: jax under repo .venv).
PY=${PY:-/dev/shm/.venv-jax/bin/python}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50
DATA=/dev/shm/factored-fb/datasets
OUT=analysis/rescuing-fb; mkdir -p "$OUT"

for s in 0 1 2; do
  RUN=$BASE/m_firstpassage_noortho_s$s
  for inp in psi cube; do
    echo "[spec07-renewal] $(date +%H:%M:%S) fit renewal head input=$inp seed=$s"
    "$PY" scripts/rescuing_fb/renewal_head.py --run-dir "$RUN" --dataset-path "$DATA" \
      --input $inp --max-goals 2000 --iters 4000 --fid-queries 2000 --holdout 100 \
      --fit-fp-mu --eval --out "$OUT/renewal_${inp}_s$s.npz" \
      --eval-out "$OUT/renewal_${inp}_s${s}_eval.json" \
      || { echo "[spec07-renewal] FIT FAILED inp=$inp s=$s"; continue; }
  done
  for w in renewal plain; do
    EXTRA=""
    [ "$w" = renewal ] && EXTRA="--renewal-head $OUT/renewal_psi_s$s.npz"
    echo "[spec07-renewal] $(date +%H:%M:%S) zero-shot inference-only seed=$s weighting=$w"
    "$PY" scripts/rescuing_fb/eval_zeroshot_rewards.py --run-dir "$RUN" \
      --gpi --z-weighting $w $EXTRA --episodes 25 --horizon 200 \
      --out "$OUT/zeroshot_infonly_s${s}_${w}.json" \
      || echo "[spec07-renewal] EVAL FAILED s=$s w=$w"
  done
done
echo "[spec07-renewal] DONE"
