#!/usr/bin/env bash
# FB -> bilinear-GCIQL knob ladder: 5 variants x 3 seeds @500k, cube-single state.
# Base = FB-IQL critic with LEARNED B (fixed_b=none), flow actor, ortho 0, lr 3e-4.
# goal_match reward needs the cur-goal in the mix, so R-runs use critic_goal_mix=[.5,.2,.3].
# Plan/notes: memory fb-to-gciql-bridge-knobs. ALWAYS run inside tmux.
#   tmux new -s bridgeknobs 'DRY_RUN=0 scripts/launch_bridge_knobs.sh 2>&1 | tee /dev/shm/bridge_knobs_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-$REPO/.venv/bin/python}
DATA=/dev/shm/factored-fb/datasets
NGPU=${NGPU:-8}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/bridge_knobs_logs
OUTDIR=/dev/shm/bridge_knobs_out
mkdir -p "$LOGDIR" "$OUTDIR"

BASE="agent=fb_flowbc domain=cube_single device=cuda data_path=$DATA \
use_wandb=true wandb_entity=amsks save_eval_videos=false \
fixed_b=none critic_mode=iql train_goal_mode=traj ortho_coef=0 \
lr_f=3e-4 lr_b=3e-4 lr_actor=3e-4 bc_coeff=3.0 num_train_steps=$STEPS"

GMIX="critic_goal_mix=[0.5,0.2,0.3] actor_goal_mix=[1.0,0.0,0.0]"

# tag|extra overrides
variants=(
  "a0_cubexyz|iql_reward=cube_xyz critic_goal_mix=[1.0,0.0,0.0]"
  "r0_base|iql_reward=goal_match $GMIX"
  "r1_decond|iql_reward=goal_match $GMIX decondition_F=true"
  "r2_dnorm|iql_reward=goal_match $GMIX d_norm=true"
  "r3_allon|iql_reward=goal_match $GMIX decondition_F=true d_norm=true"
)

JOBS=()
for v in "${variants[@]}"; do
  tag=${v%%|*}; extra=${v#*|}
  for s in 0 1 2; do
    JOBS+=("${tag}_s${s}|||$BASE $extra seed=$s wandb_group=fb_bridge_${tag} wandb_run_name=fb_bridge_${tag}_s${s} hydra.run.dir=$OUTDIR/${tag}_s${s}")
  done
done

echo "Variants: ${#variants[@]}  |  total runs: ${#JOBS[@]}  |  steps: $STEPS  |  ngpu: $NGPU"
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN (set DRY_RUN=0 to launch) ==="
  i=0; for j in "${JOBS[@]}"; do printf 'GPU%d  %s\n' $((i % NGPU)) "${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

i=0
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; cmd=${j#*|||}
  g=$((i % NGPU))
  echo "[launch] GPU$g  $tag"
  CUDA_VISIBLE_DEVICES=$g $PY train.py $cmd >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
  sleep 3   # stagger so data-load / GPU-alloc don't collide
done
echo "Launched ${#JOBS[@]} runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
