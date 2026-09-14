#!/usr/bin/env bash
# Density-FB at FULL STATE (density_fb_z_state) on the two manipulation envs that only have
# object-coordinate runs: scene and puzzle-3x3. These fill the Density-FB column of the main
# table (PAPER/ICLR/baselines/td_jepa_density.tex), which is otherwise not input-matched to the
# baselines -- every other column reads the full state.
#
# The launch line is the campaign's density_fb_z line (scripts/tdjtab_manifest.sh) with the
# critic swapped: the hub configs of the two cube runs differ from density_fb_z in exactly one
# key, config.backward_input=state, which the critic yaml sets. Nothing else changes, so
# scene/puzzle at full state sit in the same protocol as every other 1M cell.
#
# The box is full, so this is a QUEUE: it polls for a GPU with no compute process and starts the
# next pending run there. Six runs, two envs x three seeds. Run it under tmux (session dfbstate).
#
# Gotchas this script encodes:
#   - measure_hit_spec MUST be quoted (OmegaConf reads a bare a:b:c as a sexagesimal number).
#   - XLA_PYTHON_CLIENT_PREALLOCATE=false or the packed GPUs OOM each other.
#   - --auto_resume so a restart picks up the last checkpoint instead of starting over.
set -u
cd "$(dirname "$0")/.."

ROOT=exp/dfb_state
LOGS=logs/dfb_state
mkdir -p "$LOGS"

SCENE_SPEC='19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'
PUZZLE_SPEC='19:21:0.5,23:25:0.5,27:29:0.5,31:33:0.5,35:37:0.5,39:41:0.5,43:45:0.5,47:49:0.5,51:53:0.5'

# pending runs: "<domain>|<seed>|<hit spec>"
PENDING=()
for s in 0 1 2; do PENDING+=("scene_play|$s|$SCENE_SPEC"); done
for s in 0 1 2; do PENDING+=("puzzle_3x3|$s|$PUZZLE_SPEC"); done

# GPUs 3 and 4 are excluded: the walker_feat windows run three seeds back to back on each of
# them, so they go briefly idle between seeds and will reclaim themselves. Taking one would put
# two trainings on one card. Everything else (sqloss on 0/1/5, sqloss_retro on 6/7/2) frees for
# good when its chain ends.
GPUS=(0 1 2 5 6 7)

gpu_free () {  # free = no compute process, twice in a row (a single poll can catch a gap
               # between two chained runs on someone else's GPU)
  [ -n "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 1
  sleep 30
  [ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

launch () {
  local gpu="$1" dom="$2" seed="$3" spec="$4"
  local tag="${dom}_s${seed}"
  echo "$(date -u +%H:%M:%SZ) launching $tag on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  WANDB_MODE=offline MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python -m impls.main \
      --actor flowbc --auto_resume \
      --train_steps 1000000 --eval_interval 100000 --save_interval 250000 \
      --critic density_fb_z_state --domain "$dom" --seed "$seed" --save_dir "$ROOT" \
      --override measure_density_z_actor_reward=latent_centered \
      --override measure_density_hard=true \
      --override "measure_hit_spec=\"$spec\"" \
      --override eval_relabel_size=10000 \
      --override eval_relabel_shift=1.0 \
      --override eval_relabel_episodes=10 \
      > "$LOGS/$tag.log" 2>&1 &
  echo "$!" > "$LOGS/$tag.pid"
}

echo "$(date -u +%H:%M:%SZ) queue start: ${#PENDING[@]} runs pending, waiting for a free GPU"
i=0
while [ "$i" -lt "${#PENDING[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$i" -lt "${#PENDING[@]}" ] || break
    if gpu_free "$g"; then
      IFS='|' read -r dom seed spec <<< "${PENDING[$i]}"
      launch "$g" "$dom" "$seed" "$spec"
      i=$((i+1))
      sleep 120   # let the new process claim the GPU before polling again
    fi
  done
  [ "$i" -lt "${#PENDING[@]}" ] && sleep 300
done
echo "$(date -u +%H:%M:%SZ) all ${#PENDING[@]} runs started; waiting for them to finish"
wait
echo "$(date -u +%H:%M:%SZ) DFB_STATE_DONE"
