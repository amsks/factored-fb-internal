#!/usr/bin/env bash
# Spec 18 section 7 (PAPER/Multistep-FB/specs/18-density-measure.md) -- the cube-single wave.
# All cells FROM SCRATCH, 500k, standing d50 recipe (bc_coeff 3.0, eval_interval 50000,
# eval_episodes 50). No warm start anywhere (the lattice refuses it for density).
#
#   d18_cs_dens s0/s1/s2   PRIMARY: measure_density. Two-tower score f(s,a,g,u)=F.B/tau over a
#                          query support resampled from the batch's own next-state object coords
#                          EVERY update, softmax-normalized; CE fit to sg[(1-g)k + g*w_bar+].
#                          No protos file, no checkpoint<->file coupling. The query count is a
#                          Monte-Carlo sample size, not a spatial resolution.
#   d18_cs_hard s0         ABLATION: measure_density_hard -- immediate term k = I (one-hot at the
#                          row's own successor) instead of the eps-kernel softmax. Attributes
#                          smoothing vs simplex-TD.
#   d18_cs_partK{64,512,1024} s0
#                          CONTROL: 17b partition at other K. Existing code, only the protos file
#                          changes. Answers "is the discretization arbitrary?" -- flat across K
#                          means the prototype count is not load-bearing.
#
# Gates (spec section 7): G1 density success >= 0.5 at 500k (real target ~= part's 0.884);
#   G2 median density_ess >= 4 and density_w_entropy not -> 0 while success holds;
#   G3 PRIMARY readout stable across the eval pool sweep (<= 0.05 per-task from P=4096 to 16384);
#   G4 Spearman of MC region masses >= 0.65 (owed for part K256 too).
# Pre-registered interpretations, including the ones that are RESULTS rather than bugs
# (ESS -> 1 = nearest-neighbour degeneration; density floors while part holds at every K =
# the finite support is load-bearing), are in spec section 7 -- read them BEFORE reacting to a number.
#
# Anchors: part K256 = 0.884/0.884 (17b, gate G3 passed); delta = zero-flux collapse 0.15-0.26;
#   fp-noortho 0.967; visitation-from-scratch unstable in 0.004-0.5.
#
# Run in tmux:
#   tmux new -d -s s18cs 'scripts/rescuing_fb/launch_spec18_cs.sh 2>&1 | tee -a /dev/shm/spec18_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec18_cs.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec18}
LOG=/dev/shm/spec18_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
# GPU0 is reserved: the spec17 d17_cs_part_s1 re-run lives there until it reaches 500k.
GPUS=(${GPUS:-1 2 3 4 5 6 7})
FP=$OUTROOT/factored-fb
DATASET=${DATASET:-$HOME/.ogbench/data/cube-single-play-v0.npz}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# --- prototypes for the 17b K-sweep control (density needs none) -------------------
# 17b prototypes are NOT serialized into the checkpoint (static field, so pre-17b ckpts still
# restore), so every later eval re-reads these exact files. Generate once, log the digest, and
# leave them read-only -- a regenerated or permuted file silently changes the q-term with no error.
for K in 64 512 1024; do
  P=$(pwd)/results/delta_protos/cube_single_K${K}_s0.npy
  if [ ! -f "$P" ]; then
    echo "[s18] generating K=$K prototypes -> $P"
    mkdir -p "$(dirname "$P")"
    "$PY" scripts/rescuing_fb/make_delta_prototypes.py \
      --dataset "$DATASET" --hit_spec '19:22:0.4' --k "$K" --seed 0 --out "$P" || exit 1
    chmod 444 "$P"
  fi
  echo "[s18] protos K=$K sha256: $(sha256sum "$P")"
done

# tag|config|seed
JOBS=(
  "d18_cs_dens_s0|cube_single_state_measure_density_d50|0"
  "d18_cs_dens_s1|cube_single_state_measure_density_d50|1"
  "d18_cs_dens_s2|cube_single_state_measure_density_d50|2"

  "d18_cs_hard_s0|cube_single_state_measure_density_hard_d50|0"

  "d18_cs_partK64_s0|cube_single_state_measure_delta_partition_K64_d50|0"
  "d18_cs_partK512_s0|cube_single_state_measure_delta_partition_K512_d50|0"
  "d18_cs_partK1024_s0|cube_single_state_measure_delta_partition_K1024_d50|0"
)

echo "[s18] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX, GPUS=${GPUS[*]}"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s18] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s18] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s18] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  # XLA_FLAGS: disable CUDA-graph/command-buffer capture. The spec17 cell d17_cs_part_s1 died
  # TWICE (70k on GPU4, 230k on GPU0) with CUDA_ERROR_STREAM_CAPTURE_INVALIDATED raised from a
  # device_get at logging time -- different GPUs, so not a bad card. Costs a little speed; a
  # crash 200k into a 7-job wave costs much more.
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s18] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
