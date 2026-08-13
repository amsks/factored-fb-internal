#!/usr/bin/env bash
# 2026-07-23 CPU/light-GPU gates for the FP successor-measure claim (8h deliverable):
#   verify — independent re-eval of the 0.967 FP-NoOrtho cells: restore params_500000,
#            train 100 throwaway steps (loop constraint), eval 100 episodes at a FRESH
#            seed (100+s). Sequential to avoid the eval-contention segfault.
#   ret    — retrieval-decode k-sweep on seeds 1/2 (k=1,16; s0 already has 0.888/0.904):
#            the zero-shot robustness read. CPU-only.
# Usage: run_gates_20260723.sh {verify|ret}
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
JPY=/dev/shm/.venv-jax/bin/python
FPRUNS=/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50
OUT=analysis/rescuing-fb
LOG=/dev/shm/spec07_logs; mkdir -p "$LOG"

case "${1:?verify|ret}" in
verify)
  for s in 0 1 2; do
    echo "[gate-verify] $(date +%H:%M:%S) seed=$s"
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=2 \
      "$PY" run_gciql.py --config-name cube_single_state_measure_firstpassage_noortho_d50 \
        seed=$((100 + s)) train_steps=100 log_interval=100 eval_interval=100 \
        save_interval=1000000 eval_episodes=100 \
        restore_path="$FPRUNS/m_firstpassage_noortho_s$s" restore_epoch=500000 \
        output_root=/home/adityamohan/exp_spec07 exp_name="verify_s$s" \
        run_group=GCIQL-State-Flow-Measure-FP-NoOrtho-VERIFY-d50 \
        > "$LOG/verify_s$s.log" 2>&1 \
      || echo "[gate-verify] FAILED seed=$s"
  done
  echo "[gate-verify] DONE"
  ;;
ret)
  for s in 1 2; do
    for k in 1 16; do
      echo "[gate-ret] $(date +%H:%M:%S) seed=$s k=$k"
      CUDA_VISIBLE_DEVICES= "$JPY" scripts/rescuing_fb/eval_zeroshot_rewards.py \
        --run-dir "$FPRUNS/m_firstpassage_noortho_s$s" \
        --gpi --z-inference retrieval --retrieval-k $k --episodes 25 --horizon 200 \
        --out "$OUT/zeroshot_infonly_s${s}_RET_k${k}.json" \
        > "$LOG/ret_s${s}_k${k}.log" 2>&1 \
        || echo "[gate-ret] FAILED s=$s k=$k"
    done
  done
  echo "[gate-ret] DONE"
  ;;
esac
