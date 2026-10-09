#!/usr/bin/env bash
cd /home/adityamohan/git/Factored-FB-seam
export GPU=1
mkdir -p /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/fb_tdj
echo ">>> cube_double/fb_tdj/s1  [GPU 1] $(date)"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic fb_tdj --domain cube_double --seed 1 --save_dir exp/tdjtab --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 2>&1 | tee -a /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/fb_tdj/s1.log
mkdir -p /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/rldp_density
echo ">>> cube_double/rldp_density/s1  [GPU 1] $(date)"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic rldp_density --domain cube_double --seed 1 --override measure_density_hard=true --override 'measure_hit_spec="19:22:0.4,28:31:0.4"' --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 2>&1 | tee -a /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/cube_double/rldp_density/s1.log
echo ALL_DONE_GPU_1; sleep 86400
