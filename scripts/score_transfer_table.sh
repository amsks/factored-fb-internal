#!/usr/bin/env bash
# Transfer table (sec:reward-transfer): score EVERY method on BOTH task routes from the same
# 1M checkpoints, under one protocol, so the two columns are comparable.
#   route A = goal conditioning (the env's goal frame -> the critic's own map_eval_cond)
#   route B = reward inference (10k relabelled transitions -> infer_cond -> InferredCondAgent)
# Route A for the HILP family predates impls/critics/hilp.py::map_eval_cond (added 2026-09-08),
# so every published HILP goal cell was goal-blind; this re-scores it with the fix.
# Checkpoints: exp/tdjtab, exp/, and exp/hub/exp (pulled from amsks/factored-fb-checkpoints).
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p analysis/transfer_table logs/transfer
OUT=analysis/transfer_table/routes.jsonl
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m scripts.reeval_checkpoint --actor flowbc --restore_epoch 1000000 --eval_episodes 10 --out_jsonl $OUT"
ROUTE=${1:-A}
if [ "$ROUTE" = "B" ]; then EXTRA="--infer_cond_episodes 10000 --eval_on_inferred_cond"; else EXTRA=""; fi
GPUS=(0 1 2 3 5 6 7)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
while read -r dom crit seed path; do
  gi=$((i % ${#GPUS[@]})); gpu=${GPUS[$gi]}
  tag="route${ROUTE}_${dom}_${crit}_s${seed}"
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE --critic $crit --domain $dom --seed $seed --restore_path $path --tag $tag $EXTRA > logs/transfer/${tag}.log 2>&1 || echo FAIL $tag >> logs/transfer/failures.txt; "
  i=$((i+1))
done < /tmp/cells.txt
tmux new-session -d -s transfer${ROUTE} -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t transfer${ROUTE} -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo TRANSFER_${ROUTE}_DONE >> logs/transfer/done.txt; sleep 999999"; done
tmux kill-window -t transfer${ROUTE}:init || true
echo "queued $i cells on ${#GPUS[@]} gpus"
