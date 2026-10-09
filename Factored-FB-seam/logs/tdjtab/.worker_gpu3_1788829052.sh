#!/usr/bin/env bash
cd /home/adityamohan/git/Factored-FB-seam
export GPU=3
mkdir -p /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/rldp
echo ">>> cube_double/rldp/s0  [GPU 3] $(date)"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic rldp --domain cube_double --seed 0 --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 2>&1 | tee -a /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/rldp/s0.log
echo ALL_DONE_GPU_3; sleep 86400
