#!/bin/bash
# Overnight PLAY-data generation for the 9 non-cube dev tasks (cube-single already has OGBench
# play data). One process per task, GPU-pinned, thread-capped. Logs -> $LOGD/<task>.log.
# Box2D/ManiSkill roll TD-MPC2 checkpoints; Metaworld uses the scripted oracle (no ckpt).
set -u
FB=/raid/tongle/rl/Factored-FB
PY=/raid/tongle/rl/newt/.venv/bin/python
OUT=/raid/tongle/rl/fb-data/collected_play
LOGD=$OUT/_logs
VID=$OUT/_videos
mkdir -p "$LOGD" "$VID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# task  gpu  episodes  num_envs  extra
JOBS=(
  "b2d-center-ctrl         0 1000 50"
  "b2d-center-ctrl-target  1 1000 50"
  "b2d-center-wall         2 1000 50"
  "b2d-goal-ctrl-maze      3 1000 50"
  "b2d-goal-ctrl-target    4 1000 50"
  "ms-pick-cube            5 3000 50"
  "ms-push-cube            6 3000 50"
  "mw-push                 7 1000  1"
  "mw-pick-place           7 1000  1"
)

for j in "${JOBS[@]}"; do
  set -- $j; task=$1; gpu=$2; eps=$3; nenv=$4
  echo "[launch] $task -> gpu $gpu, $eps episodes"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu nohup \
    "$PY" "$FB/data_collection/collect_play.py" \
      --task "$task" --episodes "$eps" --num_envs "$nenv" \
      --out "$OUT" --video_dir "$VID" --video_k 4 --render_size 128 \
      > "$LOGD/$task.log" 2>&1 &
  sleep 8   # stagger CUDA/env init
done
echo "[launch] all 9 play-collection jobs started; logs in $LOGD"
wait
echo "[launch] ALL DONE"
