#!/bin/bash
# RLDP with the PAPER implementation's self-predictive window off-by-one reinstated.
#
# `Is Manipulation Hard for Successor Measures?` reports RLDP at 0.227 +- 0.060 on
# cube-single (1M steps, 10 seeds, IQM, goal-conditioned eval). Its implementation
# (agents/rldp/agent.py:67) slices `batch["observation"][:, 1:]` for the SP rollout targets
# where td_jepa's sampler reads the NEXT-observation sequence, so from rollout step 2 the
# predictor is trained that the action does not move the embedding and the rollout never
# reaches s_{t+h}. Our jax port corrected that on 2026-08-25 and gets 0.163 +- 0.064.
#
# The two numbers are therefore not the same algorithm. This wave runs the paper's version
# at the PAPER'S budget -- 1M steps, matching tab:pertask -- so they can be compared
# like for like.
#
# PACKED onto GPUs already hosting the hilpfb wave: ~0.7 GB per cell against 23 GB, 192 CPUs.
set -euo pipefail
REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/portalign_logs
SESSION=portalign

tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }
mkdir -p "$LOGDIR"
first=1
for s in 0 1 2; do
  gpu=$((s+4))
  name="cs-portalign-s$s"
  cmd="cd $REPO && CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline \
$PY -m impls.main --critic rldp_portalign --actor flowbc --domain cube_single --seed $s \
  --train_steps 1000000 --eval_interval 50000 --save_interval 500000 \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "$name" "$cmd"; first=0
  else tmux new-window -t "=${SESSION}" -n "$name" "$cmd"; fi
  echo "  $name -> GPU $gpu (packed)"
done
echo "launched '$SESSION', 3 windows at 1M steps"
