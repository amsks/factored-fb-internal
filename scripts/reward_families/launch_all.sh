#!/usr/bin/env bash
# Every (env, method, seed) job of the reward-family eval, 8 workers over GPUs 0/1.
# Reproduce: bash scripts/reward_families/launch_all.sh   (from the repo root, inside tmux)
set -u
cd "$(dirname "$0")/../.."
PY=/dev/shm/.venv-jax/bin/python
mkdir -p logs/rewardfam analysis/reward_families
JOBS=()
i=0
for env in cube-single-play-v0 cube-double-play-v0; do
  if [ "$env" = cube-single-play-v0 ]; then
    methods="fb_tdj density_fb_z hilp_tdjc hilp_density_reg rldp rldp_density_reg bc"
  else
    methods="fb_tdj density_fb_z hilp_tdjc rldp bc"   # no cube-double run of either density retrofit
  fi
  for m in $methods; do for s in 0 1 2; do JOBS+=("$env $m $s"); done; done
  JOBS+=("$env random 0")
done
run_one() {
  env=$1; m=$2; s=$3; idx=$4
  gpu=$((idx % 2))
  log=logs/rewardfam/${env}_${m}_seed${s}.log
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline \
    $PY scripts/reward_families/eval_families.py --env "$env" --method "$m" --seed "$s" --episodes 20 > "$log" 2>&1
  echo "[exit $?] $env $m $s" >> logs/rewardfam/launch_all.status
}
export -f run_one; export PY
: > logs/rewardfam/launch_all.status
for j in "${JOBS[@]}"; do echo "$j $i"; i=$((i+1)); done | xargs -P 8 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "[all done]" >> logs/rewardfam/launch_all.status
