#!/usr/bin/env bash
# Launch/queue the 10-seed x 1M FB<->GCIQL bridge-table campaign across all GPUs.
# Plan: PAPER/specs/2026-06-22-bridge-table-10seed-1M-campaign.md
#
# SAFE BY DEFAULT: DRY_RUN=1 prints the full job list + estimate and exits.
# Set DRY_RUN=0 to actually launch. ALWAYS run inside tmux so it survives:
#   tmux new -s bridge1m \
#     'DRY_RUN=0 scripts/launch_bridge_10seed_1M.sh 2>&1 | tee /dev/shm/bridge1m_dispatch.log'
#
# Knobs:  NGPU(8)  SLOTS_PER_GPU(3)  DRY_RUN(1)  LOGDIR  PY
#
# Per-ablation policy:
#   GCIQL bilinear ablations  : resume seeds 0-2 (remaining 500k) + fresh 3-9 (@1M)
#   d=512                     : reuse 5 existing @1M, add fresh seeds 5-9
#   CRL Gaussian (no ckpt)    : 10 fresh @1M
#   FB torch ablations        : resume seeds 0-2 (@1M total) + fresh 3-9
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

NGPU=${NGPU:-8}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-3}
CAP=$((NGPU * SLOTS_PER_GPU))
DRY_RUN=${DRY_RUN:-1}
LOGDIR=${LOGDIR:-/dev/shm/bridge1m_logs}
PY=${PY:-$REPO/.venv/bin/python}
DATA=/dev/shm/factored-fb/datasets
GOUT=/dev/shm/gciql_outputs/factored-fb
FOUT=/dev/shm/factored-fb_outputs
mkdir -p "$LOGDIR"

JOBS=()                                  # entries: "TAG|||CMD" (CMD has no CUDA_VISIBLE_DEVICES)
add_job(){ JOBS+=("$1|||$2"); }

# ---------------- GCIQL bilinear ablations (resume 0-2 + fresh 3-9) -------------
# Each row: <config-name> <run_group>
gciql_rows=(
  "cube_single_state_gciql_flow_bilinear_d50         GCIQL-State-Flow-Bilinear-d50"
  "cube_single_state_gciql_flow_bilinear_d128        GCIQL-State-Flow-Bilinear-d128"
  "cube_single_state_gciql_flow_bilinear_d5          GCIQL-State-Flow-Bilinear-d5"
  "cube_single_state_gciql_flow_bilinear_d50_tau05   GCIQL-State-Flow-Bilinear-d50-tau05"
  "cube_single_state_gciql_flow_bilinear_d50_tau095  GCIQL-State-Flow-Bilinear-d50-tau095"
  "cube_single_state_gciql_flow_bilinear_d50_singleq GCIQL-State-Flow-Bilinear-d50-singleq"
  "cube_single_state_gciql_flow_bilinear_d50_ddpgbc  GCIQL-State-Flow-Bilinear-d50-ddpgbc"
  "cube_single_state_gciql_flow_g4                   gciql_g4_zcond"
)
for row in "${gciql_rows[@]}"; do
  cfg=${row%% *}; grp=${row##* }
  for s in 0 1 2; do
    d=$(ls -d "$GOUT/$grp/sd00${s}_"* 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -f "$d/params_500000.pkl" ]; then
      exp=$(basename "$d")
      add_job "${grp}__s${s}__resume" \
        "XLA_PYTHON_CLIENT_PREALLOCATE=false $PY run_gciql.py --config-name $cfg seed=$s train_steps=500000 save_interval=500000 restore_path=$d restore_epoch=500000 exp_name=$exp"
    else
      add_job "${grp}__s${s}__fresh" \
        "XLA_PYTHON_CLIENT_PREALLOCATE=false $PY run_gciql.py --config-name $cfg seed=$s train_steps=1000000 save_interval=1000000"
    fi
  done
  for s in 3 4 5 6 7 8 9; do
    add_job "${grp}__s${s}__fresh" \
      "XLA_PYTHON_CLIENT_PREALLOCATE=false $PY run_gciql.py --config-name $cfg seed=$s train_steps=1000000 save_interval=1000000"
  done
done

# ---------------- d=512: reuse 5 existing @1M, add fresh seeds 5-9 -------------
for s in 5 6 7 8 9; do
  add_job "GCIQL-State-Flow-Bilinear__s${s}__fresh" \
    "XLA_PYTHON_CLIENT_PREALLOCATE=false $PY run_gciql.py --config-name cube_single_state_gciql_flow_bilinear_d512 seed=$s train_steps=1000000 save_interval=1000000"
done

# ---------------- CRL Gaussian: no checkpoint exists -> 10 fresh @1M ------------
for s in 0 1 2 3 4 5 6 7 8 9; do
  add_job "crl_state_ddpgbc__s${s}__fresh" \
    "XLA_PYTHON_CLIENT_PREALLOCATE=false $PY run_gciql.py --config-name cube_single_state_crl seed=$s train_steps=1000000 save_interval=1000000 run_group=crl_state_ddpgbc"
done

# ---------------- FB torch ablations (resume 0-2 + fresh 3-9) -------------------
fb_common="agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 device=cuda data_path=$DATA use_wandb=true wandb_entity=amsks save_eval_videos=false"
# row: <group>|<extra agent flags>
fb_rows=(
  "fb_state_zher|train_goal_mode=traj"
  "fb_state_iql|train_goal_mode=traj q_loss_coef=1.0"
  "fb_state_iqlcritic_fix|train_goal_mode=traj critic_mode=iql"
)
for row in "${fb_rows[@]}"; do
  grp=${row%%|*}; flags=${row#*|}
  for s in 0 1 2 3 4 5 6 7 8 9; do
    dir="$FOUT/${grp}_s${s}"
    base="$PY train.py $fb_common $flags seed=$s num_train_steps=1000000 wandb_group=$grp wandb_run_name=${grp}_s${s} hydra.run.dir=$dir"
    if [ "$s" -le 2 ] && [ -f "$dir/checkpoints/final.pt" ]; then
      add_job "${grp}__s${s}__resume" "$base resume=true resume_from=$dir/checkpoints/final.pt"
    else
      add_job "${grp}__s${s}__fresh" "$base force=true"
    fi
  done
done

# ----------------------------- summary / dry-run -------------------------------
nres=$(printf '%s\n' "${JOBS[@]}" | grep -c '__resume|||') || true
nfresh=$(( ${#JOBS[@]} - nres ))
echo "Jobs: ${#JOBS[@]}  (resume=$nres  fresh=$nfresh)   cap=$CAP (${NGPU}x${SLOTS_PER_GPU})"
echo "Rough est: ~$(( (nfresh*1 + nres/2) * 3 / CAP ))h wallclock (very rough; GCIQL ~2.5h/1M, FB ~5.5h/1M)"

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN (set DRY_RUN=0 to launch) ==="
  for j in "${JOBS[@]}"; do printf '%-44s  %s\n' "${j%%|||*}" "${j#*|||}"; done
  exit 0
fi

# ----------------------------- GPU-balanced dispatcher -------------------------
declare -A PID2GPU
declare -a GLOAD; for ((g=0; g<NGPU; g++)); do GLOAD[$g]=0; done
reap(){ local pid g; for pid in "${!PID2GPU[@]}"; do
          if ! kill -0 "$pid" 2>/dev/null; then g=${PID2GPU[$pid]}; GLOAD[$g]=$((GLOAD[$g]-1)); unset 'PID2GPU[$pid]'; fi
        done; }
pick_gpu(){ local g best=-1 m=9999; for ((g=0; g<NGPU; g++)); do
              if [ "${GLOAD[$g]}" -lt "$m" ]; then m=${GLOAD[$g]}; best=$g; fi; done; echo "$best"; }

echo "Launching ${#JOBS[@]} jobs into $LOGDIR ..."
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; cmd=${j#*|||}
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 5; reap; done
  reap
  g=$(pick_gpu)
  echo "[$(date +%H:%M:%S)] gpu$g  $tag"
  CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" >"$LOGDIR/${tag}.log" 2>&1 &
  PID2GPU[$!]=$g; GLOAD[$g]=$((GLOAD[$g]+1))
done
wait
echo "ALL DONE. Logs in $LOGDIR"
