#!/usr/bin/env bash
# Train goal-conditioned Density-FB on the five goal-observable MetaWorld / ManiSkill tasks.
#
#   FB_DATA_ROOT=/path bash scripts/launch_mwms_gc.sh                 # dry run (print commands)
#   MODE=tmux GPUS="0 1 2 3" FB_DATA_ROOT=/path bash scripts/launch_mwms_gc.sh
#   JOBS_ONLY=1 FB_DATA_ROOT=/path bash scripts/launch_mwms_gc.sh     # one command per line
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-dry}"                        # dry | tmux
JOBS_ONLY="${JOBS_ONLY:-0}"
: "${FB_DATA_ROOT:?set FB_DATA_ROOT to the dataset root holding data/<family>/<task>.npz}"
FB_DATA_ROOT="$(cd "$FB_DATA_ROOT" && pwd)"
TASKS="${TASKS:-mw-push mw-bin-picking mw-box-close ms-push-cube ms-pick-cube}"
SEEDS="${SEEDS:-0 1 2}"
ACTOR="${ACTOR:-flowbc}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SAVE_ROOT="${SAVE_ROOT:-exp/mwms_gc}"
PY="${PY:-python}"
SESSION="${SESSION:-mwms_gc}"
GPUS="${GPUS:-0}"

# Per-task goal masks
MW_PROTO="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
declare -A PROTO=(
  [mw-push]="$MW_PROTO" [mw-bin-picking]="$MW_PROTO" [mw-box-close]="$MW_PROTO"
  [ms-push-cube]="obs_mask_dims=[25,26,27] goal_mask_dims=[25,26,27] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
  [ms-pick-cube]="obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
)
# Per-task goal kernel: object slice and success radius; fill REQUIRED before launching
declare -A HIT=(
  [mw-push]="4:7:0.05"
  [mw-bin-picking]="4:7:REQUIRED"
  [mw-box-close]="4:7:REQUIRED"
  [ms-push-cube]="28:30:0.10"
  [ms-pick-cube]="29:32:0.025"
)

ENVBASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl FB_DATA_ROOT=$FB_DATA_ROOT FB_HF_EXPORT=$FB_DATA_ROOT"
IDS=(); CMDS=()
for t in $TASKS; do
  [ -n "${PROTO[$t]+x}" ] || { echo "no protocol for $t (goal-observable tasks only)" >&2; exit 2; }
  case "${HIT[$t]:-}" in ""|*REQUIRED*) echo "hit spec for $t is unset/REQUIRED -- fill HIT[$t] first" >&2; exit 2;; esac
  for s in $SEEDS; do
    ov="${PROTO[$t]} measure_density_hard=true goal_spec=\"${HIT[$t]}\""
    ovflags=""; for kv in $ov; do ovflags+=" --override $kv"; done
    cmd="CUDA_VISIBLE_DEVICES=\$GPU $ENVBASE $PY -m impls.main --critic density_fb --actor $ACTOR --domain $t --seed $s"
    cmd+=" --train_steps $TRAIN_STEPS --eval_interval $EVAL_INTERVAL --save_interval $SAVE_INTERVAL"
    cmd+=" --eval_episodes $EVAL_EPISODES --auto_resume --save_dir $SAVE_ROOT"
    cmd+="$ovflags"
    IDS+=("$t/s$s"); CMDS+=("$cmd")
  done
done
N=${#IDS[@]}

if [ "$JOBS_ONLY" = "1" ]; then printf '%s\n' "${CMDS[@]}"; exit 0; fi
if [ "$MODE" = "dry" ]; then
  echo "== $N goal-conditioned Density-FB runs (native goal eval), save_dir $SAVE_ROOT =="
  for i in "${!IDS[@]}"; do echo "# ${IDS[$i]}"; echo "  ${CMDS[$i]}"; done
  exit 0
fi
[ "$MODE" = "tmux" ] || { echo "MODE must be dry or tmux" >&2; exit 2; }
read -r -a GPU_LIST <<< "$GPUS"; NG=${#GPU_LIST[@]}; STAMP="$(date +%Y%m%d-%H%M%S)"
tmux new-session -d -s "$SESSION" -n init "sleep 1"
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"; wf="logs/mwms_gc/.worker_gpu${gpu}_$STAMP.sh"; mkdir -p logs/mwms_gc
  { echo "#!/usr/bin/env bash"; echo "cd $PWD"; echo "export GPU=$gpu"
    for i in "${!IDS[@]}"; do [ $((i % NG)) -eq "$w" ] || continue
      log="logs/mwms_gc/${IDS[$i]}.log"; printf 'mkdir -p %q\n' "$(dirname "$log")"
      printf '%s > %q 2>&1\n' "${CMDS[$i]}" "$log"; done
    echo "echo ALL_DONE_GPU_$gpu; sleep 86400"; } > "$wf"; chmod +x "$wf"
  tmux new-window -t "=$SESSION" -n "gpu$gpu" "bash $wf"
done
tmux kill-window -t "$SESSION:init" 2>/dev/null || true
echo "== launched $N runs in tmux '$SESSION' =="
