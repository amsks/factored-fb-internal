#!/usr/bin/env bash
# td_jepa's own HILP code on cube-double: 2x2 sweep x 3 seeds = 12 runs, all concurrent,
# 4 per GPU on GPUs 5-7 (alongside our version's grid). Logs in /dev/shm/tdj_cd_logs.
set -euo pipefail
cd /home/adityamohan/git/baselines/td_jepa
mkdir -p /dev/shm/tdj_cd_logs /home/adityamohan/tdj_ogbench/datasets
# 1. dataset in td_jepa's episode format (once)
if [ ! -d /home/adityamohan/tdj_ogbench/datasets/cube-double-play-v0/buffer ]; then
  MUJOCO_GL=egl .venv/bin/python -c "from scripts.data_processing.ogbench.extract_all import extract; \
extract('cube-double-play-v0', '/home/adityamohan/tdj_ogbench/datasets', '/home/adityamohan/.ogbench/data')"
fi
.venv/bin/python run_cd_grid.py --list
# 2. one tmux window per trial
read -r -a GPUS <<< "${GPUS:-5 6 7}"
tmux has-session -t '=tdjcd' 2>/dev/null && { echo "session tdjcd exists"; exit 1; }
for i in $(seq 0 11); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  cmd="cd $PWD; CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl .venv/bin/python run_cd_grid.py --index $i 2>&1 | tee /dev/shm/tdj_cd_logs/trial_$i.log; echo DONE_$i; sleep 86400"
  if [ "$i" -eq 0 ]; then tmux new-session -d -s tdjcd -n t$i "$cmd"; else tmux new-window -d -t '=tdjcd' -n t$i "$cmd"; fi
done
echo "launched tmux session tdjcd"
