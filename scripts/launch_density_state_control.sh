#!/bin/bash
# Density-FB WITH vs WITHOUT privileged object coordinates in the B tower.
#
#   bash scripts/launch_density_state_control.sh
#
# WHY THIS EXISTS. Every other method in the with/without table has a matched pair
# (fb/fb_oc, rldp/rldp_oc, psm/psm_oc, hilp_cube/hilp_oc). Density-FB had none, because its
# B tower reads the eps-scaled object-coordinate slice by construction. So its cube-single
# 0.92 could never be attributed between the categorical ESTIMATOR and the
# object-coordinate OUTCOME SPACE. `backward_input: state` (impls/critics/density_fb.py::
# _b_query) is that control: B reads the full observation row, while `u_raw` -- the kernel
# reward's operand -- is untouched, so the hit kernel, the hard immediate target and the
# goal semantics are bit-identical between the two arms.
#
# THE SECOND PURPOSE, and it may matter more. The `density_fb` seeds here are a
# REPRODUCTION of a published number (0.915+-0.098 reference / 0.916+-0.016 port, 500k,
# 3 seeds) inside THIS harness at THIS protocol (20 eval episodes, not 50). We currently
# have plain FB at 0.11 against a published 0.420 on the same env, and no way to tell a
# broken FB config from a broken shared harness. If density_fb reproduces ~0.9 here, the
# harness is sound and the FB gap is FB's own configuration; if it also lands low, every
# state-input cell in the table is measuring the harness, not the method.
#
# CUDA GRAPHS OFF: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED killed cd-fb_oc at 465k in the
# previous wave, and has taken a cell in ~25% of waves historically (commit 8010207). It
# faults inside the logging line's device->host copy, never with a numerical symptom.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/dstate_logs
SESSION=dstate

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000

read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'

# name           critic             domain       seed  extra-override
ARMS=(
  "cs-dfb-s0     density_fb         cube_single  0     -"
  "cs-dfb-s1     density_fb         cube_single  1     -"
  "cs-dfb-s2     density_fb         cube_single  2     -"
  "cs-dstate-s0  density_fb_state   cube_single  0     -"
  "cs-dstate-s1  density_fb_state   cube_single  1     -"
  "cs-dstate-s2  density_fb_state   cube_single  2     -"
  "cd-dfb-s0     density_fb         cube_double  0     $CD_HIT"
  "cd-dstate-s0  density_fb_state   cube_double  0     $CD_HIT"
)
[ "${#GPU_LIST[@]}" -eq "${#ARMS[@]}" ] || { echo "need ${#ARMS[@]} GPUs"; exit 1; }

first=1
for i in "${!ARMS[@]}"; do
  read -r name critic dom seed ov <<< "${ARMS[$i]}"
  gpu="${GPU_LIST[$i]}"
  ovflag=""; [ "$ov" = "-" ] || ovflag="--override $ov"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
WANDB_MODE=offline \
$PY -m impls.main --critic $critic --actor flowbc --domain $dom --seed $seed \
  --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY $ovflag \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "$name" "$cmd"; first=0
  else tmux new-window -t "=${SESSION}" -n "$name" "$cmd"; fi
  echo "  $name -> GPU $gpu   ($critic / $dom / seed $seed)"
done
echo "launched '$SESSION', ${#ARMS[@]} windows on GPUs ${GPU_LIST[*]}"
