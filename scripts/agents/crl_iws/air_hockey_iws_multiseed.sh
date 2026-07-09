#!/usr/bin/env bash
# Launch IWS air-hockey training for multiple seeds with auto-resume.
#
# ── Where outputs go (per seed <SEED>, default seeds: 7 11 42 43) ───────────
#
#   Training artifacts (fixed path — safe to re-run / resume):
#     runs/air_hockey/iws_seed<SEED>/
#       config.json              run config
#       metrics.jsonl            training + eval metrics (appended on resume)
#       metrics.csv              written at end of each training process
#       wandb_run.json           W&B run id (used to resume the same online run)
#       checkpoints/             checkpoint_<STEPS>.pt every CHECKPOINT_EVERY (500k)
#       eval_videos/             eval rollout GIFs
#
#   Launcher logs (timestamped each invocation):
#     logs/air_hockey_iws_multiseed_<YYYYMMDD_HHMMSS>/
#       seed_<SEED>.log          stdout/stderr for that seed's training process
#
#   W&B (online):
#     project: air_hockey
#     group:   air_hockey_iws_multiseed
#     filter:  https://wandb.ai/calcharles/air_hockey?group=air_hockey_iws_multiseed
#
# Recommended: run inside screen so SSH disconnect does not kill training:
#   screen -S iws_fb
#   bash scripts/agents/crl_iws/air_hockey_iws_multiseed.sh 2>&1 | tee logs/iws_multiseed.log
#   # Ctrl-a d to detach; screen -r iws_fb to reattach
#
set -euo pipefail
trap '' HUP

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/data/run_crl_iws_train.py"

# ── Campaign defaults (match UM iws.yaml air_hockey backend) ────────────────
TASK_ID="${TASK_ID:-AirHockeyPuckGoalPosition-v0}"
AIR_HOCKEY_CONFIG="${AIR_HOCKEY_CONFIG:-configs/air_hockey/puck_goal_real_transfer_s4.yaml}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-20000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-256}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500000}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-air_hockey}"
WANDB_GROUP="${WANDB_GROUP:-air_hockey_iws_multiseed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"

SEEDS="${SEEDS:-7 11 42 43}"
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
STAGGER_SEC="${STAGGER_SEC:-5}"
DRY_RUN="${DRY_RUN:-0}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-0}"
FORCE_FRESH="${FORCE_FRESH:-0}"

module load gcc/14.2.0 2>/dev/null || true

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/air_hockey_iws_multiseed_${TS}}"
mkdir -p "${LOG_ROOT}"

campaign_run_dir() {
  echo "${REPO_ROOT}/${OUTPUT_ROOT}/air_hockey/iws_seed${1}"
}

find_latest_checkpoint() {
  local ckpt_dir="${1}/checkpoints"
  [[ -d "${ckpt_dir}" ]] || return 0
  local latest=""
  local f
  for f in "${ckpt_dir}"/checkpoint_*.pt; do
    [[ -f "${f}" ]] || continue
    if [[ -z "${latest}" || "${f}" > "${latest}" ]]; then
      latest="${f}"
    fi
  done
  printf '%s' "${latest}"
}

checkpoint_steps() {
  local ckpt_path="$1"
  [[ -n "${ckpt_path}" ]] || { echo 0; return; }
  local stem
  stem="$(basename "${ckpt_path}" .pt)"
  stem="${stem#checkpoint_}"
  echo $((10#${stem}))
}

seed_progress_steps() {
  local run_dir="$1"
  local ckpt steps metrics last_step
  ckpt="$(find_latest_checkpoint "${run_dir}")"
  steps="$(checkpoint_steps "${ckpt}")"
  metrics="${run_dir}/metrics.jsonl"
  if [[ -f "${metrics}" ]]; then
    last_step="$(python3 - <<PY
import json
from pathlib import Path
path = Path(${metrics@Q})
lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
print(int(json.loads(lines[-1]).get("step", 0)) if lines else 0)
PY
)"
    if [[ "${last_step}" -gt "${steps}" ]]; then
      steps="${last_step}"
    fi
  fi
  echo "${steps}"
}

build_resume_args() {
  local seed="$1"
  local run_dir
  run_dir="$(campaign_run_dir "${seed}")"
  mkdir -p "${run_dir}/checkpoints"

  local -a args=(--resume-run-dir "${run_dir}")

  if [[ "${FORCE_FRESH}" == 1 || "${NO_AUTO_RESUME}" == 1 ]]; then
    printf '%s\n' "${args[@]}"
    return 0
  fi

  local ckpt progress
  ckpt="$(find_latest_checkpoint "${run_dir}")"
  progress="$(seed_progress_steps "${run_dir}")"
  if [[ "${progress}" -ge "${TOTAL_ENV_STEPS}" ]]; then
    echo "__COMPLETE__"
    return 0
  fi
  if [[ -n "${ckpt}" ]]; then
    args+=(--resume-from "${ckpt}")
  fi
  printf '%s\n' "${args[@]}"
}

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local run_dir log_file
  run_dir="$(campaign_run_dir "${seed}")"
  log_file="${LOG_ROOT}/seed_${seed}.log"

  mapfile -t resume_args < <(build_resume_args "${seed}")
  if [[ "${#resume_args[@]}" -eq 1 && "${resume_args[0]}" == "__COMPLETE__" ]]; then
    echo "[iws-multiseed] seed=${seed} already at ${TOTAL_ENV_STEPS} steps — skip"
    return 0
  fi

  local kind="FRESH"
  if [[ " ${resume_args[*]} " == *" --resume-from "* ]]; then
    kind="RESUME"
  fi

  local -a cmd=(
    uv run python "${TRAIN_SCRIPT}"
    --algorithm iws
    --env "${TASK_ID}"
    --air-hockey-config "${AIR_HOCKEY_CONFIG}"
    --num-envs "${TRAIN_NUM_ENVS}"
    --total-steps "${TOTAL_ENV_STEPS}"
    --seed "${seed}"
    --device "${DEVICE}"
    --eval-every-steps "${EVAL_EVERY_STEPS}"
    --eval-episodes "${EVAL_EPISODES}"
    --eval-num-envs "${EVAL_NUM_ENVS}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --output-root "${OUTPUT_ROOT}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --run-name "air_hockey_iws_seed${seed}"
    --wandb-tag "iws"
    --wandb-tag "multiseed"
    --wandb-tag "seed_${seed}"
    "${resume_args[@]}"
  )

  echo "[iws-multiseed] $(date +%T) seed=${seed} gpu=${gpu} [${kind}] run_dir=${run_dir}"
  echo "[iws-multiseed] log=${log_file}"

  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[DRY] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    return 0
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${cmd[@]}"
  ) > "${log_file}" 2>&1
}

read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR <<< "${GPUS}"
NUM_SEEDS="${#SEED_ARR[@]}"
NUM_GPUS="${#GPU_ARR[@]}"
if [[ "${MAX_PARALLEL}" -eq 0 ]]; then
  MAX_PARALLEL="${NUM_SEEDS}"
fi
if [[ "${MAX_PARALLEL}" -gt "${NUM_SEEDS}" ]]; then
  MAX_PARALLEL="${NUM_SEEDS}"
fi

echo "[iws-multiseed] repo=${REPO_ROOT}"
echo "[iws-multiseed] seeds=[${SEEDS}] gpus=[${GPUS}] max_parallel=${MAX_PARALLEL} total_steps=${TOTAL_ENV_STEPS}"
echo "[iws-multiseed] logs=${LOG_ROOT}  wandb_group=${WANDB_GROUP}  dry_run=${DRY_RUN}"
echo ""

if [[ "${MAX_PARALLEL}" -le 1 ]]; then
  gpu="${GPU_ARR[0]:-0}"
  for seed in "${SEED_ARR[@]}"; do
    run_one_seed "${seed}" "${gpu}"
  done
else
  declare -a SLOT_PIDS=()
  for ((i = 0; i < MAX_PARALLEL; i++)); do
    SLOT_PIDS+=("")
  done

  for seed in "${SEED_ARR[@]}"; do
    slot=""
    while [[ -z "${slot}" ]]; do
      for ((i = 0; i < MAX_PARALLEL; i++)); do
        if [[ -z "${SLOT_PIDS[$i]}" ]] || ! kill -0 "${SLOT_PIDS[$i]}" 2>/dev/null; then
          slot="${i}"
          break
        fi
      done
      [[ -z "${slot}" ]] && sleep 5
    done
    gpu="${GPU_ARR[$((slot % NUM_GPUS))]}"
    run_one_seed "${seed}" "${gpu}" &
    SLOT_PIDS[$slot]=$!
    sleep "${STAGGER_SEC}"
  done

  for pid in "${SLOT_PIDS[@]}"; do
    [[ -n "${pid}" ]] && wait "${pid}" || true
  done
fi

echo ""
echo "[iws-multiseed] ALL DONE. logs=${LOG_ROOT}"
