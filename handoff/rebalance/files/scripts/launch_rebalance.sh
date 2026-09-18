#!/usr/bin/env bash
# REBALANCE the cube-single play data toward its rare bottleneck rows and ask whether Density-FB
# still beats FB once the rarity is gone. The point of the experiment: mitigate the
# bottleneck-row rarity WITHOUT changing the data -- no expert data is added, no rows are
# dropped, the state-action SUPPORT is unchanged; only the index sampler's per-row mass moves
# (impls/utils/datasets.py::Dataset.enable_persist_oversample with lam < 1, i.e. the
# persistence lever run in reverse). Every critic draws through the same base sampler, so FB
# and Density-FB see the identical rebalanced stream.
#
# Levels (LEVELS, "pool[window]:transient_mass"):
#   onset5:0.25  rows within +-5 steps of a cube MOTION ONSET (rest -> moving at thresh 0.005)
#                get 25% of the sampling mass (measured 13.4% of play transitions -> ~2x)
#   onset5:0.5   same window, 50% mass (~4x)
#   motion:0.5   all rows where the cube is moving get 50% mass; the cube is already moving in
#                47.8% of play transitions, so this is a NEAR NO-OP control
# Two MODES (MODE, default "sample"; MODE="sample loss" runs both):
#   sample  resample the whole batch (above). Its conclusion is "this sampling change helps":
#           the actor's training distribution and the candidate next-state sets of both
#           critics move too.
#   loss    sampling UNCHANGED (uniform); the same transient/persistent split becomes a
#           per-row CRITIC loss weight (batch["row_weight"], transient rows W, persistent 1,
#           dataset-mean 1; W*ntr/(W*ntr+npst) = transient_mass). Only the FB TD row terms and
#           the density CE rows are weighted; actor, reward-inference and eval untouched. Its
#           conclusion is "giving these decisions more critic weight fixes the approximation".
# The lambda=1 (uniform) baselines ALREADY EXIST and are not re-run here: fb_tdj / density_fb_z
# / density_fb from the tdjtab campaign (exp/tdjtab), fb_gc from the goalcond handoff
# (exp/goalcond_gc). 4 critics x 3 levels x 3 seeds = 36 runs, cube_single, 1M steps, flowbc.
#
# Two pairs, two eval routes (state the route in every table):
#   z pair    fb_tdj vs density_fb_z -- reward-inference eval, Route B (eval_relabel.csv)
#   goal pair fb_gc  vs density_fb   -- native goal-conditioned eval (eval.csv), no relabel flags
#
#   DRY=1 bash scripts/launch_rebalance.sh          # print the job list
#   JOBS_ONLY=1 bash scripts/launch_rebalance.sh    # one shell command per line (36)
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_rebalance.sh
#   LEVELS="onset5:0.5" SEEDS="0" bash scripts/launch_rebalance.sh
#   MODE="sample loss" JOBS_ONLY=1 bash scripts/launch_rebalance.sh   # 72 commands
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR="${LOGDIR:-/dev/shm/rebalance_logs}"
SESSION="${SESSION:-rebalance}"
SAVE="${SAVE:-exp/rebalance}"
STEPS="${STEPS:-1000000}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
SAVE_EVERY="${SAVE_EVERY:-250000}"
LEVELS="${LEVELS:-onset5:0.25 onset5:0.5 motion:0.5}"
MODE="${MODE:-sample}"
SEEDS="${SEEDS:-0 1 2}"
CRITICS="${CRITICS:-fb_tdj density_fb_z fb_gc density_fb}"
THRESH="${THRESH:-0.005}"
BLOCK="${BLOCK:-19:22}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
cd "$REPO"
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
HIT="--override measure_density_hard=true --override 'measure_hit_spec=\"19:22:0.4\"'"

level_tag()    { local ps="${1%%:*}" tm="${1#*:}"; echo "${ps}_tm${tm}"; }
level_pool()   { local ps="${1%%:*}"; echo "${ps//[0-9]/}"; }
level_window() { local ps="${1%%:*}" w; w="${ps//[a-z]/}"; echo "${w:-0}"; }
level_tm()     { echo "${1#*:}"; }

emit() { # critic mode level seed -> full command
  local critic=$1 mode=$2 lvl=$3 seed=$4 ov="" sd
  sd="$SAVE/${mode}_$(level_tag "$lvl")"
  case "$critic" in
    fb_tdj)       ov="$RELABEL" ;;
    density_fb_z) ov="--override measure_density_z_actor_reward=latent_centered $HIT $RELABEL" ;;
    fb_gc)        ov="" ;;
    density_fb)   ov="$HIT" ;;
    *) echo "unknown critic $critic" >&2; return 1 ;;
  esac
  echo "$PY -m impls.main --critic $critic --actor flowbc --domain cube_single --seed $seed" \
       "--auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY" \
       "--save_dir $sd --override persist_mode=$mode --override persist_pool=$(level_pool "$lvl") --override persist_window=$(level_window "$lvl")" \
       "--override persist_transient_mass=$(level_tm "$lvl") --override persist_vel_thresh=$THRESH" \
       "--override 'persist_block=\"$BLOCK\"' $ov"
}

JOBS=""
for mode in $MODE; do
  case "$mode" in sample|loss) ;; *) echo "MODE must be sample and/or loss, got '$mode'" >&2; exit 1 ;; esac
  for critic in $CRITICS; do
    for lvl in $LEVELS; do
      for seed in $SEEDS; do
        [ -f "$SAVE/${mode}_$(level_tag "$lvl")/cube-single-play-v0_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
        JOBS+="$critic $mode $lvl $seed"$'\n'
      done
    done
  done
done

N=$(printf '%s' "$JOBS" | grep -c . || true)
if [ "${JOBS_ONLY:-0}" = "1" ]; then
  printf '%s' "$JOBS" | while read -r c m l s; do [ -n "${c:-}" ] && emit "$c" "$m" "$l" "$s"; done
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
  while read -r c m l s; do
    [ -n "${c:-}" ] || continue
    tag="${m}_$(level_tag "$l")-${c}-s${s}"
    body+="echo '>>> $tag'; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl $(emit "$c" "$m" "$l" "$s") 2>&1 | tee $LOGDIR/$tag.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s' "$slice" | grep -c .) jobs"
done
echo "launched '$SESSION'"
