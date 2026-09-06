#!/bin/bash
# The missing cube-double seeds for the Density-FB with/without-object-coordinates control.
#
# scripts/launch_density_state_control.sh gave cube-SINGLE 3 seeds per arm but cube-DOUBLE
# only seed 0. This adds seeds 1 and 2 for both arms so every cell in that wave is n=3.
#
# PACKED onto GPUs already hosting cube-single cells. Safe here for a specific reason: the
# cube-double cells are EVAL-bound, not train-bound -- their eval rollouts run at ~19 s/
# episode on CPU/EGL, which is why those GPUs sit at ~1% utilisation for long stretches.
# Memory is ~0.6 GB per cell against 23 GB, and the ExoRL campaign ran three per GPU.
# 192 CPUs, load average ~10 at launch.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/dstate_logs
SESSION=dstatecd

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000
CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'

tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }
mkdir -p "$LOGDIR"

# name           critic            seed  gpu
ARMS=(
  "cd-dfb-s1     density_fb        1     0"
  "cd-dfb-s2     density_fb        2     1"
  "cd-dstate-s1  density_fb_state  1     2"
  "cd-dstate-s2  density_fb_state  2     3"
)

first=1
for spec in "${ARMS[@]}"; do
  read -r name critic seed gpu <<< "$spec"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
WANDB_MODE=offline \
$PY -m impls.main --critic $critic --actor flowbc --domain cube_double --seed $seed \
  --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY \
  --override $CD_HIT \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "$name" "$cmd"; first=0
  else tmux new-window -t "=${SESSION}" -n "$name" "$cmd"; fi
  echo "  $name -> GPU $gpu (packed)   ($critic / cube_double / seed $seed)"
done
echo "launched '$SESSION', ${#ARMS[@]} windows"
