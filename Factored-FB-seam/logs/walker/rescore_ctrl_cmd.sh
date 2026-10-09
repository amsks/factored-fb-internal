#!/bin/bash
cd /home/adityamohan/git/Factored-FB-seam
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
/dev/shm/.venv-jax/bin/python -m scripts.exorl_paper_eval --root exp/exorl-tdj --critic rldp_tdj --actor ddpgbc \
    --domains dmc-walker --seeds 0 1 2 --steps 500000 --episodes 10 --task-set tdjepa --jobs 3 \
    --out exp/paper_eval_tdj_rescore_20260914.jsonl 2>&1 | tee -a logs/walker/rescore_rldp_tdj_par.log
echo "== CTRL RESCORE DONE =="
