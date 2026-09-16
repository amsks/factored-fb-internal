#!/usr/bin/env bash
# Route B of the transfer table: reward inference. Uses scripts/relabel_rescore.py, NOT
# reeval_checkpoint: the task reward must be RE-LABELLED per task from the next state's
# simulator qpos. The dataset's own reward column is the goal-reaching one (zeroed for play
# data), so inferring from it solves for a constant and scores near zero -- which is exactly
# what a first attempt with reeval_checkpoint produced.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p analysis/transfer_table logs/transfer
OUT=analysis/transfer_table/routeB.jsonl
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python scripts/relabel_rescore.py --epoch 1000000 --episodes 10 --relabel-size 10000 --out $OUT"
GPUS=(0 1 2 3 5 6 7)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
# (env, critic, exp-root) triples: the roots the paper's tables quote.
while read -r env crit root; do
  gi=$((i % ${#GPUS[@]})); gpu=${GPUS[$gi]}
  tag="routeB_${env}_${crit}"
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE --env $env --critics $crit --exp-root $root --seeds 0 1 2 > logs/transfer/${tag}.log 2>&1 || echo FAIL $tag >> logs/transfer/failures.txt; "
  i=$((i+1))
done < /tmp/cellsB.txt
tmux new-session -d -s transferB -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t transferB -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo TRANSFER_B_DONE >> logs/transfer/done.txt; sleep 999999"; done
tmux kill-window -t transferB:init || true
echo "queued $i (env,critic) groups on ${#GPUS[@]} gpus"
