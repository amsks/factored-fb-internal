#!/usr/bin/env bash
# Train FB, HILP and RLDP on MetaWorld and ManiSkill (1M steps, goal and reward-inference eval).
#
#   FB_DATA_ROOT=/path/to/Factor-FB-Offline bash scripts/launch_mwms_baselines.sh            # dry run
#   MODE=tmux  GPUS="0 1 2 3" FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh           # local
#   MODE=slurm ALLOC=ASC26008 FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh          # array job
#
# Select with CRITICS, TASKS, SEEDS. Finished runs are skipped; partial runs resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-dry}"                       # dry | tmux | slurm
DRYRUN="${DRYRUN:-0}"
[ "$MODE" = "dry" ] && DRYRUN=1
: "${FB_DATA_ROOT:?set FB_DATA_ROOT to the dataset root holding data/<family>/<task>.npz}"
FB_DATA_ROOT="$(cd "$FB_DATA_ROOT" && pwd)"
CRITICS="${CRITICS:-fb hilp rldp}"
TASKS="${TASKS:-mw-push mw-bin-picking mw-box-close ms-push-cube ms-pick-cube ms-stack-cube ms-poke-cube ms-lift-peg}"
SEEDS="${SEEDS:-0 1 2}"
ACTOR="${ACTOR:-flowbc}"                   # bc_coeff 3.0 is the yaml default (OGBench manipulation)
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
RELABEL_SIZE="${RELABEL_SIZE:-10000}"
RELABEL_SHIFT="${RELABEL_SHIFT:-1.0}"
RELABEL_EPISODES="${RELABEL_EPISODES:-10}"
EXTRA_OV="${EXTRA_OV:-}"
SAVE_ROOT="${SAVE_ROOT:-exp/mwms_baselines}"
LOGROOT="${LOGROOT:-logs/mwms_baselines}"
PY="${PY:-python}"
SESSION="${SESSION:-mwms_baselines}"
GPUS="${GPUS:-0}"
ALLOC="${ALLOC:-ASC26008}"
MAX_PARALLEL="${MAX_PARALLEL:-20}"          # slurm: array throttle (%N)
VENV="${VENV:-${SCRATCH:-$HOME}/.venv-jax-fb}"   # slurm only: vista_train.slurm's venv
mkdir -p "$LOGROOT"

# Per-task protocol overrides, identical for every critic
# MetaWorld: target position dims 36:39 masked from observation and goal
MW_PROTO="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
# ManiSkill: goal position dims masked; stack/poke/lift-peg have no goal in the observation
declare -A PROTO=(
  [mw-push]="$MW_PROTO" [mw-bin-picking]="$MW_PROTO" [mw-box-close]="$MW_PROTO"
  [ms-push-cube]="obs_mask_dims=[25,26,27] goal_mask_dims=[25,26,27] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
  [ms-pick-cube]="obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
  [ms-stack-cube]="" [ms-poke-cube]="" [ms-lift-peg]=""
)
# Per-critic overrides
declare -A CRITIC_OV=( [fb]="" [hilp]="" [rldp]="ortho_coef=1.0" )
RELABEL="eval_relabel_size=$RELABEL_SIZE eval_relabel_shift=$RELABEL_SHIFT eval_relabel_episodes=$RELABEL_EPISODES"

overrides_for() {  # <critic> <task> -> space-separated k=v list
  local c=$1 t=$2
  [ -n "${PROTO[$t]+x}" ] || { echo "no protocol entry for task $t (add it to PROTO)" >&2; exit 2; }
  [ -n "${CRITIC_OV[$c]+x}" ] || { echo "no critic entry for $c (add it to CRITIC_OV)" >&2; exit 2; }
  echo "${PROTO[$t]} ${CRITIC_OV[$c]} $RELABEL $EXTRA_OV" | tr -s ' ' | sed 's/^ //;s/ $//'
}
run_dir() { echo "$SAVE_ROOT/$1/${2}_${1}_${ACTOR}_seed_$3"; }   # critic task seed

# Enumerate runs
IDS=(); CMDS=(); JOBLINES=()
ENVBASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl FB_DATA_ROOT=$FB_DATA_ROOT FB_HF_EXPORT=$FB_DATA_ROOT"
n_skip=0
for t in $TASKS; do for c in $CRITICS; do for s in $SEEDS; do
  ov="$(overrides_for "$c" "$t")"
  rd="$(run_dir "$c" "$t" "$s")"
  if [ -f "$rd/params_${TRAIN_STEPS}.pkl" ]; then n_skip=$((n_skip+1)); echo "  skip (done): $t/$c/s$s"; continue; fi
  ovflags=""; for kv in $ov; do ovflags+=" --override $(printf '%q' "$kv")"; done
  cmd="CUDA_VISIBLE_DEVICES=\$GPU $ENVBASE $PY -m impls.main --critic $c --actor $ACTOR --domain $t --seed $s"
  cmd+=" --train_steps $TRAIN_STEPS --eval_interval $EVAL_INTERVAL --save_interval $SAVE_INTERVAL"
  cmd+=" --eval_episodes $EVAL_EPISODES --log_interval $LOG_INTERVAL --auto_resume"
  cmd+=" --save_dir $SAVE_ROOT/$c$ovflags"
  IDS+=("$t/$c/s$s"); CMDS+=("$cmd")
  # Slurm job line, read by mwms_array.slurm
  JOBLINES+=("CRITIC=$c ACTOR=$ACTOR DOMAIN=$t SEED=$s TRAIN_STEPS=$TRAIN_STEPS EVAL_INTERVAL=$EVAL_INTERVAL SAVE_INTERVAL=$SAVE_INTERVAL EVAL_EPISODES=$EVAL_EPISODES AUTO_RESUME=1 SAVE_DIR=$SAVE_ROOT/$c FB_DATA_ROOT=$FB_DATA_ROOT FB_HF_EXPORT=$FB_DATA_ROOT VENV=$VENV OVERRIDE=$(printf '%q' "$ov")")
done; done; done
N=${#IDS[@]}
echo "== $N runs to launch ($n_skip already done) | mode=$MODE steps=$TRAIN_STEPS eval@$EVAL_INTERVAL x$EVAL_EPISODES eps | data=$FB_DATA_ROOT"
[ "$N" -gt 0 ] || exit 0

if [ "$DRYRUN" = "1" ]; then
  for i in "${!IDS[@]}"; do
    echo "# ${IDS[$i]}"
    if [ "$MODE" = "slurm" ]; then echo "  ${JOBLINES[$i]}"; else echo "  ${CMDS[$i]}"; fi
  done
  if [ "$MODE" = "slurm" ]; then
    echo "# would submit: sbatch -A $ALLOC --array=0-$((N-1))%$MAX_PARALLEL --export=ALL,JOBS_FILE=<jobs file> scripts/mwms_array.slurm"
  fi
  exit 0
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
if [ "$MODE" = "slurm" ]; then
  JOBS_FILE="$LOGROOT/jobs-$STAMP.txt"
  printf '%s\n' "${JOBLINES[@]}" > "$JOBS_FILE"
  mkdir -p logs
  sbatch -A "$ALLOC" --array="0-$((N-1))%$MAX_PARALLEL" -J mwms \
         --export="ALL,JOBS_FILE=$PWD/$JOBS_FILE,CODE_DIR=$PWD" scripts/mwms_array.slurm
  echo "== submitted $N array tasks from $JOBS_FILE (logs/mwms.o<job>_<task>) =="
  exit 0
fi

# tmux: one window per GPU, its runs sequential
[ "$MODE" = "tmux" ] || { echo "MODE must be dry, tmux or slurm (got $MODE)" >&2; exit 2; }
[ -x "$PY" ] || { echo "missing $PY (volatile venv; recreate)" >&2; exit 1; }
read -r -a GPU_LIST <<< "$GPUS"; NG=${#GPU_LIST[@]}
if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "tmux session '$SESSION' exists; adding windows"; first=0; else first=1; fi
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"
  wf="$LOGROOT/.worker_gpu${gpu}_$STAMP.sh"
  {
    echo "#!/usr/bin/env bash"; echo "cd $PWD"; echo "export GPU=$gpu"
    for i in "${!IDS[@]}"; do
      [ $((i % NG)) -eq "$w" ] || continue
      log="$LOGROOT/${IDS[$i]}.log"
      printf 'mkdir -p %q\n' "$(dirname "$log")"
      printf 'echo ">>> %s  [GPU %s] $(date)"\n' "${IDS[$i]}" "$gpu"
      printf '%s 2>&1 | tee -a %q\n' "${CMDS[$i]}" "$log"
    done
    echo "echo ALL_DONE_GPU_$gpu; sleep 86400"
  } > "$wf"; chmod +x "$wf"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "bash $wf"; first=0
  else tmux new-window -d -t "=$SESSION" -n "gpu$gpu" "bash $wf"; fi
  echo "  GPU $gpu <- $(grep -c '^echo ">>>' "$wf") runs ($wf)"
done
echo "== launched in tmux session '$SESSION'; logs under $LOGROOT/<task>/<critic>/s<seed>.log =="
