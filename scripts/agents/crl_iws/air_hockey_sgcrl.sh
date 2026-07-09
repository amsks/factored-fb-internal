#!/usr/bin/env bash
# Online SGCRL air-hockey training — defaults aligned with Unsupervised_Manipulation
# scripts/train_air_hockey_task.sh (plain SGCRL, no IWS).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/data/run_crl_iws_train.py"

TASK_ID="${TASK_ID:-AirHockeyPuckGoalPosition-v0}"
AIR_HOCKEY_CONFIG="${AIR_HOCKEY_CONFIG:-configs/air_hockey/puck_goal_real_transfer_s4.yaml}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-20000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-256}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"

module load gcc/14.2.0 2>/dev/null || true

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  RESUME_ARGS+=(--resume-run-dir "${RESUME_RUN_DIR}")
fi

uv run python "${TRAIN_SCRIPT}" \
  --algorithm sgcrl \
  --env "${TASK_ID}" \
  --air-hockey-config "${AIR_HOCKEY_CONFIG}" \
  --num-envs "${TRAIN_NUM_ENVS}" \
  --total-steps "${TOTAL_ENV_STEPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --eval-num-envs "${EVAL_NUM_ENVS}" \
  --wandb-group air_hockey \
  --run-name "air_hockey_${TASK_ID//-/_}_sgcrl" \
  "${RESUME_ARGS[@]}" \
  "$@"
