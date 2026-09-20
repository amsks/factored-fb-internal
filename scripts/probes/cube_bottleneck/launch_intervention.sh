#!/usr/bin/env bash
# Fresh evaluation rollouts of the intervention probe: 9 jobs (seed x bank) spread over GPUs 0-2.
# usage: launch_intervention.sh  (then: intervention.py aggregate)
set -euo pipefail
cd /home/adityamohan/git/Factored-FB-seam
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline MPLCONFIGDIR=/tmp/mpl
PY=/dev/shm/.venv-jax/bin/python
S=scripts/probes/cube_bottleneck/intervention.py
L=logs/intervention; mkdir -p $L
i=0
for SEED in 0 1 2; do
  for BANK in approach inhand carrying; do
    GPU=$((i % 3))
    CUDA_VISIBLE_DEVICES=$GPU $PY $S roll --seeds $SEED --banks $BANK 2>&1 \
      | grep -v "Warn\|gym.logger\|UserWarning\|warnings.warn\|jnp.zeros" > $L/roll_seed${SEED}_${BANK}.log &
    i=$((i + 1))
  done
done
wait
echo "ROLL DONE" >> $L/roll_seed0_approach.log
