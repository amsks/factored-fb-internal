#!/usr/bin/env bash
# scripts/run_state_crl.sh — STATE-ONLY plain CRL (stock OGBench contrastive RL, DDPG+BC
# actor) on cube-single. The NON-FlowBC CRL baseline (config cube_single_state_crl.yaml,
# agent=agents/crl.py, alpha=3.0). Meant to run inside tmux:
#
#   tmux new -s crl_plain
#   bash scripts/run_state_crl.sh 2>&1 | tee /dev/shm/crl_plain.log
#   # detach: Ctrl-b d   reattach: tmux attach -t crl_plain
#   DRY_RUN=1 bash scripts/run_state_crl.sh        # print the plan, launch nothing
#
# GPUs: this is a JAX/OGBench run (it preallocates a GPU-memory fraction), so run it on
# GPUs that are FREE of other JAX jobs. Set GPUS to whatever is available, e.g.
#   GPUS="0 1" bash scripts/run_state_crl.sh
# (the PyTorch FB sweep on GPUs 2-7 uses little VRAM but contends on compute; the
# CRL+FlowBC state seeds on 0,1 free those up when they finish).
#
# wandb online; the shim forces entity=amsks, project=factored-fb. Everything stays in
# /dev/shm (durable / is full); TMPDIR/WANDB_DIR/logs pinned there.
set -uo pipefail
trap '' HUP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-cube_single_state_crl}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"          # seed i -> GPUS[i % nGPU]
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-crl}"
STAGGER="${STAGGER:-2}"                  # seconds between launches (stagger JAX init)
DRY_RUN="${DRY_RUN:-0}"

OUT=/dev/shm/gciql_outputs
export TMPDIR="${TMPDIR:-/dev/shm/tmp_crl}";   mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_crl}"; mkdir -p "$WANDB_DIR"
LOG_DIR="/dev/shm/crl_plain_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"

read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
read -ra SEED_ARR <<< "$SEEDS"

echo "[crl] python=$PYTHON config=$CONFIG wandb=$WANDB_MODE steps=$TRAIN_STEPS group=$RUN_GROUP"
echo "[crl] seeds=[$SEEDS] over ${NG} GPUs [$GPUS]  (seed i -> GPU i%${NG})"
echo "[crl] logs=$LOG_DIR  out=$OUT  (VOLATILE /dev/shm — copy out before reboot)"

declare -a PIDS=(); i=0
for s in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$(( i % NG ))]}"
    log="$LOG_DIR/crl_s${s}.log"
    echo "[crl] $(date +%T) seed=$s -> GPU $gpu  (log: ${log##*/})"
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" run_gciql.py \
            --config-name "$CONFIG" \
            seed="$s" train_steps="$TRAIN_STEPS" \
            save_interval="$SAVE_INTERVAL" eval_interval="$EVAL_INTERVAL" \
            wandb_mode="$WANDB_MODE" run_group="$RUN_GROUP" \
            "hydra.run.dir=$LOG_DIR/hydra_s${s}" > "$log" 2>&1 &
        PIDS+=("$!")
        sleep "$STAGGER"
    fi
    i=$((i+1))
done

echo "[crl] launched ${#PIDS[@]} runs; waiting for completion (Ctrl-b d detaches tmux)..."
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
echo "[crl] ALL DONE $(date +%T). logs=$LOG_DIR  out=$OUT"
