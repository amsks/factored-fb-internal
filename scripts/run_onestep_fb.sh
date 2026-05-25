#!/usr/bin/env bash
# scripts/run_onestep_fb.sh — full one-step FB + FlowBC run (state cube-single), N seeds.
# One-step FB (github.com/chongyi-zheng/onestep-fb): F is z-independent + SARSA next-action
# (behavior successor features); the FlowBC actor does one improvement step on <F(s,a),z>.
# Domain cube_single => fb_flowbc; onestep=true selects the one-step variant. Run in tmux:
#
#   tmux new -s onestep
#   bash scripts/run_onestep_fb.sh 2>&1 | tee /dev/shm/onestep_fb.log
#   DRY_RUN=1 bash scripts/run_onestep_fb.sh        # print the plan only
#
# Everything in /dev/shm (durable / is full); wandb online -> amsks (configs/train.yaml).
# Uses the extracted dataset (next.action is derived at load). PyTorch FB stack (train.py).
set -uo pipefail
trap '' HUP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GPUS="${GPUS:-2 3 4 5 6 7}"              # seed i -> GPUS[i % nGPU]
ORTHO_COEF="${ORTHO_COEF:-1.0}"         # standard FB ortho (one-step FB is not the high-ortho regime)
LR_B="${LR_B:-1e-4}"
STEPS="${STEPS:-1000000}"
DATA="${DATA:-/dev/shm/factored-fb/datasets}"
RUN_GROUP="${RUN_GROUP:-factored-fb-onestep-fb}"
WANDB_ENTITY="${WANDB_ENTITY:-amsks}"
RUNROOT="${RUNROOT:-/dev/shm/factored-fb/runs}"
STAGGER="${STAGGER:-3}"
DRY_RUN="${DRY_RUN:-0}"

export TMPDIR="${TMPDIR:-/dev/shm/tmp_fb}";   mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_fb}"; mkdir -p "$WANDB_DIR"
TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUNROOT"
LOG_DIR="/dev/shm/onestep_fb_$TS"; mkdir -p "$LOG_DIR"

read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
read -ra SEED_ARR <<< "$SEEDS"

[[ -d "$DATA/cube-single-play-v0/buffer" ]] || { echo "[onestep] MISSING dataset $DATA/cube-single-play-v0/buffer — run scripts/extract_ogbench.py"; exit 1; }

echo "[onestep] one-step FB + FlowBC  ortho=$ORTHO_COEF steps=$STEPS seeds=[$SEEDS] gpus=[$GPUS] (i%${NG})"
echo "[onestep] data=$DATA  group=$RUN_GROUP  entity=$WANDB_ENTITY"
echo "[onestep] runs=$RUNROOT  logs=$LOG_DIR  (VOLATILE /dev/shm)"

declare -a PIDS=(); i=0
for s in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$(( i % NG ))]}"
    run_dir="$RUNROOT/${TS}__cube__onestep_fb__s${s}"
    log="$LOG_DIR/onestep_fb_s${s}.log"
    echo "[onestep] $(date +%T) seed=$s -> GPU $gpu  (log: ${log##*/})"
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" train.py \
            domain=cube_single onestep=true \
            ortho_coef="$ORTHO_COEF" lr_b="$LR_B" num_train_steps="$STEPS" \
            data_path="$DATA" seed="$s" device=cuda \
            use_wandb=true wandb_entity="$WANDB_ENTITY" wandb_group="$RUN_GROUP" \
            wandb_run_name="onestep_fb_s${s}" \
            hydra.job.chdir=true "hydra.run.dir=$run_dir" \
            > "$log" 2>&1 &
        PIDS+=("$!")
        sleep "$STAGGER"
    fi
    i=$((i+1))
done

echo "[onestep] launched ${#PIDS[@]} runs; waiting (Ctrl-b d detaches tmux)..."
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
echo "[onestep] ALL DONE $(date +%T). runs=$RUNROOT logs=$LOG_DIR"
