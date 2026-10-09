#!/bin/bash
cd /home/adityamohan/git/Factored-FB-seam
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
for C in rldp_density_tdj rldp_tdj; do
  /dev/shm/.venv-jax/bin/python -m scripts.exorl_paper_eval --root exp/exorl-tdj --critic $C --actor ddpgbc \
    --domains dmc-walker --seeds 0 1 2 --steps 500000 --episodes 10 --task-set tdjepa --jobs 2 \
    --out exp/paper_eval_tdj_rescore_20260914.jsonl 2>&1 | tee -a logs/walker/rescore_$C.log
done
echo "== RESCORE DONE =="
