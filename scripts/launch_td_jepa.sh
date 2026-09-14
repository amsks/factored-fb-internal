#!/usr/bin/env bash
# Validation campaign for the TD-JEPA port (impls/critics/td_jepa.py).
#
# The port reproduces arXiv:2510.00739's own agent, so the validation target is their
# published OGBench proprio column (Table 1): cube-single 34.2, cube-double 3.6,
# scene 38.4, puzzle-3x3 15.6. Same protocol as every other 1M cell in this tree --
# state input, flow actor, reward-inference evaluation with 10^4 relabelled samples,
# last-3-checkpoint estimator, 3 seeds.
#
# Order matters: cube-single runs first, all three seeds, so a broken port shows up in one
# run-length rather than after the whole campaign. Published 34.2 there, and our own FB and
# HILP reproduce their columns to within a few points, so a port that lands near 34 is
# working and one that lands near 0 is not.
#
# lr_actor: the reference sets 1e-4 (TDJEPAAgentTrainConfig.lr_actor) but config merge order
# is (domain, critic, actor) with the actor winning, and actor/flowbc.yaml also declares
# 1e-4 -- so the flow-actor pairs below already match. The override is passed anyway so the
# line is explicit and so a switch to ddpgbc (which declares 3e-4) cannot silently run hot.
# tests/impls/test_td_jepa.py::test_actor_learning_rate_must_be_overridden_at_launch pins it.
set -u
cd "$(dirname "$0")/.."

ROOT=exp/td_jepa
LOGS=logs/td_jepa
mkdir -p "$LOGS"

# "<domain>|<seed>"  -- cube-single first (the early read), then the rest.
PENDING=()
for s in 0 1 2; do PENDING+=("cube_single|$s"); done
for s in 0 1 2; do PENDING+=("cube_double|$s"); done
for s in 0 1 2; do PENDING+=("scene_play|$s"); done
for s in 0 1 2; do PENDING+=("puzzle_3x3|$s"); done

GPUS=(0 1 2 3 4 5 6 7)

gpu_free () {
  [ -n "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 1
  sleep 30
  [ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

launch () {
  local gpu="$1" dom="$2" seed="$3"
  echo "$(date -u +%H:%M:%SZ) launching ${dom}_s${seed} on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  WANDB_MODE=offline MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python -m impls.main \
      --actor flowbc --auto_resume \
      --train_steps 1000000 --eval_interval 100000 --save_interval 250000 \
      --critic td_jepa --domain "$dom" --seed "$seed" --save_dir "$ROOT" \
      --override lr_actor=1.0e-4 \
      --override eval_relabel_size=10000 \
      --override eval_relabel_shift=1.0 \
      --override eval_relabel_episodes=10 \
      > "$LOGS/${dom}_s${seed}.log" 2>&1 &
}

# The full-state cube runs (tmux dfbstate) and the walker actor diagnostic (tmux actorfix)
# were queued first; wait for both to be fully launched before competing for GPUs.
echo "$(date -u +%H:%M:%SZ) waiting for the dfb_state and walker_actorfix queues to launch"
while [ "$(ls -1d exp/dfb_state/*seed_* 2>/dev/null | wc -l)" -lt 6 ] \
   || [ "$(ls -1d exp/walker_actorfix/*/*seed_* 2>/dev/null | wc -l)" -lt 9 ]; do sleep 300; done
echo "$(date -u +%H:%M:%SZ) both queues launched; ${#PENDING[@]} TD-JEPA runs pending"

i=0
while [ "$i" -lt "${#PENDING[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$i" -lt "${#PENDING[@]}" ] || break
    if gpu_free "$g"; then
      IFS='|' read -r dom seed <<< "${PENDING[$i]}"
      launch "$g" "$dom" "$seed"
      i=$((i+1))
      sleep 120
    fi
  done
  [ "$i" -lt "${#PENDING[@]}" ] && sleep 300
done
echo "$(date -u +%H:%M:%SZ) all ${#PENDING[@]} runs started"
wait
echo "$(date -u +%H:%M:%SZ) TDJEPA_DONE"
