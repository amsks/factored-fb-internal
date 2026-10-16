#!/bin/bash
# OGBench battery: 6 envs x {task coords, full obs}, K=512, gamma=0.99. GPU jax venv, one GPU per job.
cd "$(dirname "$0")/.."
PY=/dev/shm/.venv-jax/bin/python
D=$HOME/.ogbench/data
K=${K:-512}
declare -A COORDS=(
 [cube-single-play-v0]="19:22"
 [cube-double-play-v0]="19:22,28:31"
 [scene-play-v0]="19:22,28:30,32:34,36:37,38:39"
 [puzzle-3x3-play-v0]="19:21,23:25,27:29,31:33,35:37,39:41,43:45,47:49,51:53"
 [antmaze-medium-navigate-v0]="0:2"
 [antmaze-large-navigate-v0]="0:2"
)
i=0
for env in "${!COORDS[@]}"; do
  for rep in task full; do
    c=${COORDS[$env]}; [ $rep = full ] && c=full
    gpu=$((i % 8)); i=$((i+1))
    tag="${env}_${rep}_K${K}"
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false $PY scripts/occupancy_rank.py \
      --npz $D/$env.npz --coords "$c" --K $K --gamma 0.99 --ranks 2,4,8,16,32,64,128,256 \
      --fit_steps 3000 ${EXTRA} --tag "$tag" --out results/ogbench/$tag.json > logs/$tag.log 2>&1 &
  done
done
wait
echo BATTERY_DONE
