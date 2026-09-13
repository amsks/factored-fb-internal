#!/usr/bin/env bash
# Plain RLDP at the td_jepa manipulation grid's HIGHER ortho cell (ortho_coef=0.1; campaign ran 0.01),
# 4 manipulation envs x 3 seeds, 1M, same protocol as tdjtab. One tmux window per GPU, 2 runs each.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
GPUS=(0 1 4 5 6 7)
JOBS=(cube_single:0 cube_single:1 cube_single:2 cube_double:0 cube_double:1 cube_double:2 scene_play:0 scene_play:1 scene_play:2 puzzle_3x3:0 puzzle_3x3:1 puzzle_3x3:2)
tmux new-session -d -s rldp_ortho01 -n init || true
for gi in "${!GPUS[@]}"; do
  gpu=${GPUS[$gi]}; cmds=""
  for j in $gi $((gi+6)); do
    dom=${JOBS[$j]%%:*}; seed=${JOBS[$j]##*:}
    cmds+="CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic rldp --domain $dom --seed $seed --save_dir exp/rldp_ortho01 --override ortho_coef=0.1 --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 > logs/rldp_ortho01/${dom}_s${seed}.log 2>&1; "
  done
  tmux new-window -t rldp_ortho01 -n "gpu$gpu" "cd $HOME/git/Factored-FB-seam; $cmds echo DONE_GPU$gpu; sleep 999999"
done
tmux kill-window -t rldp_ortho01:init || true
