#!/usr/bin/env bash
# Route 3 -- GOAL-CONDITIONED training + eval -- for the three baselines, so they line up
# with the density_fb goal arm as a four-method GCRL comparison on cube.
#
#   fb_gc    plain FB,  hindsight-goal mixing fraction 1.0 (impls/configs/critic/fb_gc.yaml)
#   rldp_gc  RLDP,      hindsight-goal mixing fraction 1.0 (impls/configs/critic/rldp_gc.yaml)
#   hilp_gc  HILP,      mix ratio 1.0, full-state phi       (impls/configs/critic/hilp_gc.yaml)
#
# All three: actor conditioned on the goal (z=B(g) for fb/rldp, phi(g) for hilp), full state
# (NO privileged object coordinates), flowbc actor, 1M steps, 3 seeds, discount 0.99. Eval is
# the native goal-conditioned success (eval.csv) -- NOT reward inference, so NO eval_relabel
# flags (the goal-conditioned checkpoints have no reward-inference path). Idempotent:
# --auto_resume, and a run whose final checkpoint exists is skipped.
#
#   DRY=1 bash scripts/launch_goalcond_gc.sh                 # print the job list only
#   JOBS_ONLY=1 bash scripts/launch_goalcond_gc.sh           # one shell command per line
#   ENVS="cube_single cube_double" SEEDS="0 1 2" GPUS="0 1 2 3" bash scripts/launch_goalcond_gc.sh
#
# ENVS defaults to the two cube envs, where the density_fb goal arm already has route-3 runs.
# scene_play / puzzle_3x3 are valid here too, but their density_fb goal-arm counterpart has
# not been run, so the four-method comparison is only complete on cube.
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR="${LOGDIR:-/dev/shm/goalcond_gc_logs}"
SESSION="${SESSION:-goalcond_gc}"
SAVE="${SAVE:-exp/goalcond_gc}"
STEPS="${STEPS:-1000000}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
SAVE_EVERY="${SAVE_EVERY:-250000}"
ENVS="${ENVS:-cube_single cube_double}"
SEEDS="${SEEDS:-0 1 2}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
cd "$REPO"

# domain (short) -> env-dir name in the run directory
declare -A ENVDIR=( [cube_single]="cube-single-play-v0" [cube_double]="cube-double-play-v0"
                    [scene_play]="scene-play-v0" [puzzle_3x3]="puzzle-3x3-play-v0" )
# hilp_gc's phi is full state, but its yaml carries measure_hit_spec; cube_double needs the
# two-block override (comma-joined). fb_gc/rldp_gc read no hit spec. Empty = pass nothing.
declare -A HILP_HIT=( [cube_single]="" [cube_double]="19:22:0.4,28:31:0.4"
                      [scene_play]="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"
                      [puzzle_3x3]="19:21:0.5,23:25:0.5,27:29:0.5,31:33:0.5,35:37:0.5,39:41:0.5,43:45:0.5,47:49:0.5,51:53:0.5" )

# build the job list: "critic domain seed hilp_hit_or_-"
JOBS=""
for critic in fb_gc rldp_gc hilp_gc; do
  for dom in $ENVS; do
    envn="${ENVDIR[$dom]}"
    hit="-"; [ "$critic" = "hilp_gc" ] && [ -n "${HILP_HIT[$dom]:-}" ] && hit="${HILP_HIT[$dom]}"
    for seed in $SEEDS; do
      [ -f "$SAVE/${envn}_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
      JOBS+="$critic $dom $seed $hit"$'\n'
    done
  done
done

emit_cmd() { # critic dom seed hit  -> full python command
  local critic=$1 dom=$2 seed=$3 hit=$4 ov=""
  [ "$hit" = "-" ] || ov="--override 'measure_hit_spec=\"$hit\"'"
  echo "$PY -m impls.main --critic $critic --actor flowbc --domain $dom --seed $seed" \
       "--auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY" \
       "--save_interval $SAVE_EVERY --save_dir $SAVE $ov"
}

N=$(printf '%s' "$JOBS" | grep -c . || true)
if [ "${JOBS_ONLY:-0}" = "1" ]; then
  printf '%s' "$JOBS" | while read -r critic dom seed hit; do
    [ -n "${critic:-}" ] || continue; emit_cmd "$critic" "$dom" "$seed" "$hit"
  done
  exit 0
fi
echo "$N jobs:"; printf '%s' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0
[ -x "$PY" ] || { echo "missing python at $PY"; exit 1; }
mkdir -p "$LOGDIR"
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }

NG=${#GPU_LIST[@]}; first=1
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"
  slice=$(printf '%s' "$JOBS" | awk -v w="$w" -v n="$NG" 'NF && (NR-1)%n==w')
  [ -n "$slice" ] || continue
  body="cd $REPO; "
  while read -r critic dom seed hit; do
    [ -n "${critic:-}" ] || continue
    tag="${dom}-${critic}-s${seed}"
    cmd=$(emit_cmd "$critic" "$dom" "$seed" "$hit")
    body+="echo '>>> $tag'; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
$cmd 2>&1 | tee $LOGDIR/$tag.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s' "$slice" | grep -c .) jobs"
done
echo "launched session '$SESSION'  (watch: tmux attach -t $SESSION)"
