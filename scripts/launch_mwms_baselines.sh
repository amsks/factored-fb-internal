#!/usr/bin/env bash
# MetaWorld / ManiSkill baseline table: {fb_tdj, hilp_tdjc, rldp@ortho_coef=1} x 8 tasks x seeds,
# the protocol of PAPER/ICLR tab:manipulation's MetaWorld/ManiSkill blocks (1M steps, eval every
# 100k, 50 goal-conditioned episodes, plus the reward-inference route) with the SAME per-task
# protocol overrides for every critic. Handoff: docs/notes/2026-09-13-mwms-baselines-handoff.md.
#
#   FB_DATA_ROOT=/path/to/Factor-FB-Offline bash scripts/launch_mwms_baselines.sh            # dry run
#   MODE=tmux  GPUS="0 1 2 3" FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh           # local
#   MODE=slurm ALLOC=ASC26008 FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh          # array job
#   MODE=tmux DRYRUN=1 ...   /   MODE=slurm DRYRUN=1 ...     print every command line, launch nothing
#
# Selection: CRITICS="fb_tdj hilp_tdjc rldp"  TASKS="mw-push ... ms-lift-peg"  SEEDS="0 1 2"
# Budget:    TRAIN_STEPS=1000000 EVAL_INTERVAL=100000 SAVE_INTERVAL=250000 EVAL_EPISODES=50
#            RELABEL_SIZE=10000 RELABEL_SHIFT=1.0 RELABEL_EPISODES=10  (Route B, eval_relabel.csv)
# Extras:    EXTRA_OV="k=v k2=v2" appended to every run; LOG_INTERVAL=5000
# Layout:    exp/mwms_baselines/<critic>/<task>_<critic>_flowbc_seed_<s>/{eval.csv,eval_relabel.csv,
#            config.json,train.csv}; logs under logs/mwms_baselines/<task>/<critic>/s<seed>.log
# A run whose params_<TRAIN_STEPS>.pkl already exists is skipped; partial runs resume (--auto_resume).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-dry}"                       # dry | tmux | slurm
DRYRUN="${DRYRUN:-0}"
[ "$MODE" = "dry" ] && DRYRUN=1
: "${FB_DATA_ROOT:?set FB_DATA_ROOT to the dataset root holding data/<family>/<task>.npz}"
FB_DATA_ROOT="$(cd "$FB_DATA_ROOT" && pwd)"
CRITICS="${CRITICS:-fb_tdj hilp_tdjc rldp}"
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

# ---- per-task PROTOCOL overrides (not tuning): identical for every critic ------------------
# MetaWorld: target dims 36:39 blanked from obs and goal (goal-observable envs), synth per-episode
# eval goal frames; p_randomgoal=0's mass moves to trajgoal (GCDataset sum-to-1). These GC keys are
# read only by goal-sampling dataset classes; FBDataset/WindowedDataset/HILPFutureDataset ignore
# them, and the masks are applied by all three (scripts/launch_density_table.sh MW_PROTO).
MW_PROTO="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
# ManiSkill: goal_pos dims per task (launch_density_table.sh MS_MASK; pick-cube also carries
# obj_to_goal_pos at 39:42). stack/poke/lift-peg carry no goal position in the observation and
# have no goal-in-obs layout in env/unified.py::_GOAL_LAYOUT, so: no mask, eval_goal_mode=fixed.
declare -A PROTO=(
  [mw-push]="$MW_PROTO" [mw-bin-picking]="$MW_PROTO" [mw-box-close]="$MW_PROTO"
  [ms-push-cube]="obs_mask_dims=[25,26,27] goal_mask_dims=[25,26,27] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
  [ms-pick-cube]="obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
  [ms-stack-cube]="" [ms-poke-cube]="" [ms-lift-peg]=""
)
# ---- per-critic overrides: RLDP at its own default orthonormality coefficient (OGBench finding:
# ortho_coef=1 beats the transcribed 0.01). fb_tdj / hilp_tdjc run their yamls unchanged.
declare -A CRITIC_OV=( [fb_tdj]="" [hilp_tdjc]="" [rldp]="ortho_coef=1.0" )
RELABEL="eval_relabel_size=$RELABEL_SIZE eval_relabel_shift=$RELABEL_SHIFT eval_relabel_episodes=$RELABEL_EPISODES"

overrides_for() {  # <critic> <task> -> space-separated k=v list
  local c=$1 t=$2
  [ -n "${PROTO[$t]+x}" ] || { echo "no protocol entry for task $t (add it to PROTO)" >&2; exit 2; }
  [ -n "${CRITIC_OV[$c]+x}" ] || { echo "no critic entry for $c (add it to CRITIC_OV)" >&2; exit 2; }
  echo "${PROTO[$t]} ${CRITIC_OV[$c]} $RELABEL $EXTRA_OV" | tr -s ' ' | sed 's/^ //;s/ $//'
}
run_dir() { echo "$SAVE_ROOT/$1/${2}_${1}_${ACTOR}_seed_$3"; }   # critic task seed

# ---- enumerate ----------------------------------------------------------------------------
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
  # slurm job line: the vista_train.slurm env-var contract, one run per line (read by mwms_array.slurm)
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

# ---- tmux: one window per GPU, its runs sequential (scripts/launch_tdjtab.sh convention) ----
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
