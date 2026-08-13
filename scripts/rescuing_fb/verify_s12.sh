#!/usr/bin/env bash
# verify seeds 1/2 of the FP-NoOrtho cell at the native 50-episode protocol (fresh seeds).
set -uo pipefail
cd "$(dirname "$0")/../.."
FPRUNS=/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50
for s in 1 2; do
  echo "[verify2] $(date +%H:%M:%S) seed=$s"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=2 \
    .venv/bin/python run_gciql.py --config-name cube_single_state_measure_firstpassage_noortho_d50 \
      seed=$((100 + s)) train_steps=100 log_interval=100 eval_interval=100 \
      save_interval=1000000 eval_episodes=50 \
      restore_path="$FPRUNS/m_firstpassage_noortho_s$s" restore_epoch=500000 \
      output_root=/home/adityamohan/exp_spec07 exp_name="verify_s$s" \
      run_group=GCIQL-State-Flow-Measure-FP-NoOrtho-VERIFY-d50 \
      > "/dev/shm/spec07_logs/verify_s$s.log" 2>&1 || echo "[verify2] FAILED seed=$s"
done
echo "[verify2] DONE"
