#!/usr/bin/env bash
# Spec 27 Q4 / spec 29 section 2 -- offline supervised topline.
# 3 envs x 5 tasks x 3 seeds = 45 offline SAC-singletask runs (native task
# reward, standing 500k recipe -- see configs/gciql/q4_*_singletask_sac.yaml
# for full rationale). Fill-in compute: R-A owns the GPUs when they
# conflict, so this defers when the box is busy (same MAX_PROCS gate as
# scripts/launch_gciql_awrflow.sh) and is capped at a low concurrency so it
# never saturates GPUs 1-7. ALWAYS in tmux:
#   tmux new -s q4sac 'DRY_RUN=0 scripts/launch_q4_singletask_sac.sh 2>&1 | tee /dev/shm/q4sac_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
MAX_PROCS=${MAX_PROCS:-6}
LOGDIR=/dev/shm/q4sac_logs
mkdir -p "$LOGDIR"

# base config -> env_name template (task suffix inserted below)
declare -A BASE_CFG=(
  [cube-single]=q4_cube_single_singletask_sac
  [cube-double]=q4_cube_double_singletask_sac
  [scene]=q4_scene_singletask_sac
)

JOBS=()
for env in cube-single cube-double scene; do
  cfg=${BASE_CFG[$env]}
  for task in 1 2 3 4 5; do
    for s in 0 1 2; do
      tag="${env}_task${task}_s${s}"
      # dataset_name format per third_party/ogbench/README.md: "<env>-play-
      # singletask-task[n]-v0" -- 'play' names the underlying dataset file
      # (cube-single-play-v0.npz etc, the SAME file the zero-shot cells use);
      # 'singletask-task[n]' selects which of the 5 eval goals fixes
      # _reward_task_id (see ogbench/utils.make_env_and_datasets's splits
      # arithmetic, which derives BOTH the gym env id and the dataset file
      # from this one string).
      env_name="${env}-play-singletask-task${task}-v0"
      JOBS+=("${tag}|||--config-name $cfg env_name=$env_name seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=q4_${tag}")
    done
  done
done

# GPUs 1-7 only -- GPU 0 is in use by another job on this shared box.
GPUS=(${GPUS:-1 2 3 4 5 6 7})
echo "Q4 SAC-singletask topline: ${#JOBS[@]} runs @${STEPS} (defer gate: procs <= $MAX_PROCS, GPUs ${GPUS[*]})"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

for j in "${JOBS[@]}"; do
  while [ "$(pgrep -fc run_gciql.py)" -gt "$MAX_PROCS" ]; do
    echo "[defer] $(date +%H:%M) box busy ($(pgrep -fc run_gciql.py) procs > $MAX_PROCS), waiting..."
    sleep 300
  done
  tag=${j%%|||*}; args=${j#*|||}
  # rotate GPU assignment among currently-running slots (not job index) so
  # a deferred queue still spreads evenly once it resumes
  running=$(pgrep -fc run_gciql.py)
  g=${GPUS[$((running % ${#GPUS[@]}))]}
  echo "[launch] GPU$g  $tag"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  sleep 5
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
