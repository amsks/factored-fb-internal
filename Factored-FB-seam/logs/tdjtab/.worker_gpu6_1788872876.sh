#!/usr/bin/env bash
cd /home/adityamohan/git/Factored-FB-seam
export GPU=6
mkdir -p /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/antmaze_large_navigate/hilp_density_tdjc
echo ">>> antmaze_large_navigate/hilp_density_tdjc/s2  [GPU 6] $(date)"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic hilp_density_tdjc --domain antmaze_large_navigate --seed 2 --save_dir exp/tdjtab --override measure_density_hard=true --override 'measure_hit_spec="0:2:0.5"' --override actor.bc_coeff=0.3 --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 2>&1 | tee -a /home/adityamohan/git/Factored-FB-seam/logs/tdjtab/antmaze_large_navigate/hilp_density_tdjc/s2.log
echo ALL_DONE_GPU_6; sleep 86400
