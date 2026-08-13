#!/usr/bin/env bash
# Concurrency-capped GPU queue: backfills as the current wave completes, without
# recreating the CPU-eval contention meltdown (24 concurrent evals @50k = 20 s/it).
#
# Gate = TOTAL training procs on the box (the eval-contention bottleneck is GLOBAL across
# sessions). New jobs launch only while total < MAX, onto the least-loaded eligible GPU.
# eval_interval bumped 50k->100k halves each run's eval load. GPUs 1/5/6 = other track (never).
#
# Job format (pipe-delimited): "tag|config-name|seed|extra hydra overrides"
#
#   tmux new -d -s queue 'DRY_RUN=0 scripts/rescuing_fb/queue_runner.sh 2>&1 | tee /dev/shm/queue_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
MAX=${MAX:-12}                     # target total concurrent training procs (box-wide)
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
STEPS=${STEPS:-500000}
STAGGER=${STAGGER:-45}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_queue}
GPUS=(${GPUS:-0 2 3 4 7})
LOGDIR=/dev/shm/queue_logs
mkdir -p "$LOGDIR" "$OUTROOT"

# FIRST: the successor-measure offdiag_weight sweep (the experiment that decides how much
# genuine occupancy structure the measure can carry before control falls off the 0.90 cliff).
# Endpoints w=0 (CellW0) and w~1 (GCIQLCorner) run in the other session; these are the middle.
# Then: v1d50 + ogbench-sota cells (all calibrated).
SWEEP=""
for w in 0.1:010 0.25:025 0.3:030 0.5:050; do
  val=${w%%:*}; tag=${w##*:}
  for s in 0 1 2; do
    SWEEP+="offdiag_w${tag}_s${s}|cube_single_state_measure_cellw0_d50|${s}|agent_overrides.measure_offdiag_weight=${val} run_group=GCIQL-State-Flow-Measure-CellW${tag}-d50"$'\n'
  done
done

JOBS=()
while IFS= read -r line; do [ -n "$line" ] && JOBS+=("$line"); done <<<"$SWEEP"
JOBS+=(
  "v1d50_s0|cube_single_state_rgfb_v1_taskprior_d50|0|"
  "v1d50_s1|cube_single_state_rgfb_v1_taskprior_d50|1|"
  "v1d50_s2|cube_single_state_rgfb_v1_taskprior_d50|2|"
  "latefusion_s0|cube_single_state_latefusion_d50|0|"
  "latefusion_s1|cube_single_state_latefusion_d50|1|"
  "latefusion_s2|cube_single_state_latefusion_d50|2|"
  "pmatched_s0|cube_single_state_mlp_pmatched|0|"
  "pmatched_s1|cube_single_state_mlp_pmatched|1|"
  "pmatched_s2|cube_single_state_mlp_pmatched|2|"
  "phinormcap_bc_s0|cube_single_state_phinormcap_d50_ddpgbc|0|"
  "phinormcap_bc_s1|cube_single_state_phinormcap_d50_ddpgbc|1|"
  "phinormcap_bc_s2|cube_single_state_phinormcap_d50_ddpgbc|2|"
  "phinormcap_fl_s0|cube_single_state_phinormcap_d50_flow|0|"
  "phinormcap_fl_s1|cube_single_state_phinormcap_d50_flow|1|"
  "phinormcap_fl_s2|cube_single_state_phinormcap_d50_flow|2|"
  "bounded_s0|cube_single_state_bilinear_sg_bounded_d50|0|"
  "bounded_s1|cube_single_state_bilinear_sg_bounded_d50|1|"
  "bounded_s2|cube_single_state_bilinear_sg_bounded_d50|2|"
  "scenesplit_fl_s0|scene_state_bilinear_split_flow|0|"
  "scenesplit_fl_s1|scene_state_bilinear_split_flow|1|"
  "scenesplit_fl_s2|scene_state_bilinear_split_flow|2|"
  "scenesplit_bc_s0|scene_state_bilinear_split_ddpgbc|0|"
  "scenesplit_bc_s1|scene_state_bilinear_split_ddpgbc|1|"
  "scenesplit_bc_s2|scene_state_bilinear_split_ddpgbc|2|"
)

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

echo "[queue] ${#JOBS[@]} jobs (sweep first), MAX=$MAX concurrent, eval_interval=$EVAL_INTERVAL, GPUs ${GPUS[*]} -> $OUTROOT"
if [ "$DRY_RUN" = "1" ]; then
  for j in "${JOBS[@]}"; do IFS='|' read -r t c s x <<<"$j"; echo "   $t  ($c seed $s) ${x:+[$x]}"; done
  echo "[queue] active procs: $(active) (holds new launches until < $MAX)"; exit 0
fi

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[queue] $(date +%H:%M:%S) active=$(active) >= $MAX — waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[queue] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g ${extra:+[$extra]} (active was $(active))"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" train_steps="$STEPS" \
      eval_interval="$EVAL_INTERVAL" output_root="$OUTROOT" exp_name="$tag" $extra \
      >"$LOGDIR/${tag}.log" 2>&1 &
  sleep "$STAGGER"
done
echo "[queue] all ${#JOBS[@]} jobs dispatched; waiting for completion..."
wait
echo "[queue] ALL QUEUE JOBS DONE"
