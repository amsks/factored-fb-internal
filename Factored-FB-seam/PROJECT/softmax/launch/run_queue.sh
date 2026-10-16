#!/bin/bash
# Runs every line of cmds.txt (tag|args) at 500k steps, eval once at the end, 3 jobs per GPU.
cd /home/adityamohan/git/Factored-FB-seam
Q=PROJECT/softmax/launch/cmds.txt; MAXJ=${MAXJ:-24}; i=0
while IFS='|' read -r tag args; do
  [ -z "$tag" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 30; done
  gpu=$((i % 8)); i=$((i+1)); mkdir -p exp/softmax/$tag
  eval "CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --train_steps 500000 --eval_interval 500000 --save_interval 500000 --eval_episodes 10 --seed 0 --save_dir exp/softmax/$tag $args" > exp/softmax/$tag/stdout.log 2>&1 &
  echo "$(date +%H:%M) launched $tag on gpu $gpu"
  sleep 20
done < $Q
wait; echo QUEUE_DONE
