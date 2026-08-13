#!/usr/bin/env bash
# Spec 08 E3 (2026-07-24): dense-inference forensic on the EXISTING FP-ZX phase-1 ckpts.
# The dense family had a 0.90/0.13 seed-split under the moment readout; test whether ridge
# (GLS/whitened moment) closes it. moment (reproduce) + ridge lambda-sweep, both seeds.
# Direct-z, no goal decode (actor_cond=z). Spec: specs/08-two-phase-next-experiments.md (E3)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/dev/shm/.venv-jax/bin/python
ZXRUNS=/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50
OUT=analysis/rescuing-fb; mkdir -p "$OUT"
EP=${EP:-25}; CORES=${CORES:-0-15}

for s in 0 1; do
  RUN="$ZXRUNS/zx_s$s"
  # moment baseline (reproduce the split)
  echo "=== s$s moment ==="
  taskset -c "$CORES" env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MUJOCO_GL=egl "$PY" \
    scripts/rescuing_fb/eval_zeroshot_rewards.py --run-dir "$RUN" --families dense \
    --z-inference moment --tasks 1,2,3,4,5 --episodes "$EP" --horizon 200 \
    --out "$OUT/e3_dense_s${s}_moment.json"
  # ridge lambda sweep
  for lam in 0.001 0.01 0.1; do
    echo "=== s$s ridge lam=$lam ==="
    taskset -c "$CORES" env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MUJOCO_GL=egl "$PY" \
      scripts/rescuing_fb/eval_zeroshot_rewards.py --run-dir "$RUN" --families dense \
      --z-inference ridge --ridge-lambda "$lam" --tasks 1,2,3,4,5 --episodes "$EP" --horizon 200 \
      --out "$OUT/e3_dense_s${s}_ridge_lam${lam}.json"
  done
done
echo "[e3] dense-inference forensic complete -> $OUT/e3_dense_*.json"
