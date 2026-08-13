#!/bin/bash
# Spec 14 §7 — first multi-env measure-axis table on the phase-2 jax checkpoints.
# Usage: run_spec14_jax_table.sh {cd|sc}
# Runs the zero-shot harness (dirac + region, retrieval-k1 decode — the gate
# protocol) on the three 500k checkpoints of one env, sequentially.
# CPU-only rollouts (GPUs 0-7 carry training jobs); horizon 0 = env native.
set -uo pipefail
cd /home/adityamohan/git/Factored-FB

PY=/dev/shm/.venv-jax/bin/python
E8=/home/adityamohan/exp_spec08/factored-fb
OUT=results/spec14_jax
mkdir -p "$OUT"

if [ "$1" = cd ]; then
  CORES=0-23
  RUNS=(
    "GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0"
    "GCIQL-CubeDouble-Flow-Measure-BallVisit-FrozenPsi-d50/p2_cd_s0"
    "GCIQL-CubeDouble-Flow-Measure-BallVisit-Hindsight-d50/p2h_cd_s0"
  )
else
  CORES=24-47
  RUNS=(
    "GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0"
    "GCIQL-Scene-Flow-Measure-BallVisit-FrozenPsi-d50/p2_sc_s0"
    "GCIQL-Scene-Flow-Measure-BallVisit-Hindsight-d50/p2h_sc_s0"
  )
fi

for r in "${RUNS[@]}"; do
  name=$(basename "$r")
  echo "=== [$(date +%H:%M:%S)] $name ==="
  taskset -c $CORES env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    $PY scripts/rescuing_fb/eval_zeroshot_rewards.py \
      --run-dir "$E8/$r" --checkpoint-step 500000 \
      --families dirac,region --z-inference retrieval --retrieval-k 1 \
      --episodes 25 --horizon 0 --seed 0 \
      --out "$OUT/zsr_${name}_RET_k1.json" \
    || echo "!!! $name FAILED"
done
echo "=== [$(date +%H:%M:%S)] $1 chain DONE ==="
