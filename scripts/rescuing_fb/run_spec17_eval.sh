#!/usr/bin/env bash
# Spec 17 §6 "post-wave eval battery", multi-reward axis, for the cube-single wave
# (scripts/rescuing_fb/launch_spec17_cs.sh -> /home/adityamohan/exp_spec17).
#
# The goal-based axis needs nothing here -- it is the training loop's own
# evaluation/overall_success over cube-single's 5 tasks at eval_episodes=50.
# This script covers the OTHER axis: zero-shot multi-reward readout.
#
#   delta arms (17a)     eval_zeroshot_rewards.py --delta
#                        The critic learned J, not M. --delta applies
#                        Mhat = (h_g(x0) + Jhat)/(1-gamma) with h the EXACT
#                        weighted hit sum_j w_j h_{g_j}(x0) (same w that built
#                        z_r). Quoting a delta checkpoint without it is refused
#                        by the harness, which is the point -- spec 17 §1:
#                        "quote nothing raw".
#   partition arms (17b) eval_partition_zeroshot.py --decode retrieval
#                        Q_r = (r(x0) + (p - phi(x0)).z_r)/(1-gamma), z_r,k =
#                        r(proto_k). retrieval decode (NOT qr_greedy) so the
#                        numbers are comparable with the spec-14 table, which
#                        was built with --z-inference retrieval --retrieval-k 1.
#                        The prototypes are NOT in the checkpoint (static field,
#                        see gciql_flow.py's delta_protos comment), so the digest
#                        is pinned here: a regenerated or permuted .npy would
#                        change the readout with no error.
#
# CPU-only on purpose: GPUs may still carry training. Run in tmux:
#   tmux new -d -s s17ev 'scripts/rescuing_fb/run_spec17_eval.sh 2>&1 | tee -a /dev/shm/spec17_logs/eval.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/run_spec17_eval.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-/dev/shm/.venv-jax/bin/python}
ROOT=${ROOT:-/home/adityamohan/exp_spec17/factored-fb}
OUT=${OUT:-results/spec17_eval}
CORES=${CORES:-24-47}
EPOCH=${EPOCH:-500000}
EPISODES=${EPISODES:-25}
DRY_RUN=${DRY_RUN:-0}
PROTOS_SHA=b4e13e9d9e0a0b0b18deedf5fdce66c1d3f9a17e02175608dd1679064494cbf5
mkdir -p "$OUT"

if [ ! -x "$PY" ]; then
  echo "FATAL: jax venv missing at $PY (it lives on /dev/shm and is volatile)." >&2
  exit 1
fi

DELTA_RUNS=(
  "GCIQL-CubeSingle-Flow-Measure-Delta-d50/d17_cs_delta_s0"
  "GCIQL-CubeSingle-Flow-Measure-Delta-d50/d17_cs_delta_s1"
  "GCIQL-CubeSingle-Flow-Measure-Delta-d50/d17_cs_delta_s2"
)
PART_RUNS=(
  "GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50/d17_cs_part_s0"
  "GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50/d17_cs_part_s1"
  "GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50/d17_cs_part_s2"
)

run() {  # run <label> <out-json> <cmd...>
  local label=$1 out=$2; shift 2
  if [ -f "$out" ]; then echo "[s17ev] skip $label - $out exists"; return; fi
  echo "=== [$(date +%H:%M:%S)] $label ==="
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: $*"; return; fi
  taskset -c $CORES env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    "$@" || echo "!!! $label FAILED"
}

for r in "${DELTA_RUNS[@]}"; do
  name=$(basename "$r")
  [ -f "$ROOT/$r/params_$EPOCH.pkl" ] || { echo "[s17ev] skip $name - no params_$EPOCH.pkl yet"; continue; }
  run "$name (delta, --delta conversion)" "$OUT/zsr_${name}_RET_k1.json" \
    "$PY" scripts/rescuing_fb/eval_zeroshot_rewards.py \
      --run-dir "$ROOT/$r" --checkpoint-step "$EPOCH" \
      --families dirac,region --z-inference retrieval --retrieval-k 1 \
      --episodes "$EPISODES" --horizon 0 --seed 0 --delta \
      --out "$OUT/zsr_${name}_RET_k1.json"
done

for r in "${PART_RUNS[@]}"; do
  name=$(basename "$r")
  [ -f "$ROOT/$r/params_$EPOCH.pkl" ] || { echo "[s17ev] skip $name - no params_$EPOCH.pkl yet"; continue; }
  run "$name (partition, retrieval decode)" "$OUT/pzs_${name}_RET_k1.json" \
    "$PY" scripts/rescuing_fb/eval_partition_zeroshot.py \
      --run-dir "$ROOT/$r" --checkpoint-step "$EPOCH" \
      --decode retrieval --retrieval-k 1 \
      --episodes "$EPISODES" --horizon 0 --seed 0 \
      --expected-protos-sha256 "$PROTOS_SHA" \
      --out "$OUT/pzs_${name}_RET_k1.json"
done

echo "=== [$(date +%H:%M:%S)] spec17 eval chain DONE -> $OUT ==="
