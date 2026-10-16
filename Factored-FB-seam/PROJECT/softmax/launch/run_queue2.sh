#!/bin/bash
# Like run_queue.sh but the queue file and step budget are arguments; the actor is in the line.
# Usage: run_queue2.sh cmds_file train_steps eval_interval
cd /home/adityamohan/git/Factored-FB-seam
Q=$1; STEPS=${2:-1000000}; EV=${3:-500000}; MAXJ=${MAXJ:-24}; i=${GPU0:-0}
while IFS='|' read -r tag args; do
  [ -z "$tag" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 30; done
  gpu=$((i % 8)); i=$((i+1)); mkdir -p exp/softmax/$tag
  eval "CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --train_steps $STEPS --eval_interval $EV --save_interval $EV --eval_episodes 10 --seed 0 --save_dir exp/softmax/$tag $args" > exp/softmax/$tag/stdout.log 2>&1 &
  echo "$(date +%H:%M) launched $tag on gpu $gpu"
  sleep 20
done < $Q
wait; echo QUEUE_DONE
