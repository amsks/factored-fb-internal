#!/usr/bin/env bash
# scripts/run_state_crl_flowbc.sh — relaunch STATE-ONLY CRL+FlowBC, FRESH, with the
# critic-shield fix (commit f7b623d). No pixel track. Meant to run inside tmux:
#
#   tmux new -s crl_state
#   bash scripts/run_state_crl_flowbc.sh 2>&1 | tee /dev/shm/crl_state.log
#   # detach: Ctrl-b d   reattach: tmux attach -t crl_state
#
#   DRY_RUN=1     bash scripts/run_state_crl_flowbc.sh   # print the plan, launch nothing
#   CLEAN_BUGGY=1 bash scripts/run_state_crl_flowbc.sh   # delete pre-fix sd* dirs first
#
# Pre-fix checkpoints are invalid (0% — see f7b623d), so these are fresh 0..TRAIN_STEPS
# runs; the fix is picked up automatically (the shim re-imports crl_flowbc.py at child
# startup). Everything stays in /dev/shm (durable / is full); TMPDIR/WANDB_DIR/logs are
# pinned there so the full / is never written to. wandb_mode MUST be online: main.py
# builds save_dir from wandb.run.project, which the shim sets to 'factored-fb' only when
# online (offline/disabled -> 'dummy' -> wrong dir).
set -uo pipefail
trap '' HUP                       # survive SSH disconnect even outside tmux
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"          # seed i -> GPUS[i % nGPU]; 10 seeds/8 GPUs => <=2/GPU
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-crl-flowbc}"
CLEAN_BUGGY="${CLEAN_BUGGY:-0}"          # 1 = rm pre-fix sd* dirs before launching
STAGGER="${STAGGER:-2}"                  # seconds between launches (stagger JAX init)
DRY_RUN="${DRY_RUN:-0}"

OUT=/dev/shm/gciql_outputs
export TMPDIR="${TMPDIR:-/dev/shm/tmp_crl}";   mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_crl}"; mkdir -p "$WANDB_DIR"
LOG_DIR="/dev/shm/crl_state_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"

read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
read -ra SEED_ARR <<< "$SEEDS"

echo "[state] python=$PYTHON wandb=$WANDB_MODE steps=$TRAIN_STEPS group=$RUN_GROUP"
echo "[state] seeds=[$SEEDS] over ${NG} GPUs [$GPUS]  (seed i -> GPU i%${NG})"
echo "[state] logs=$LOG_DIR  out=$OUT  (VOLATILE /dev/shm — copy out before reboot)"

if [[ "$CLEAN_BUGGY" == 1 ]]; then
    echo "[state] CLEAN_BUGGY=1 -> rm $OUT/factored-fb/$RUN_GROUP/sd*"
    [[ "$DRY_RUN" == 1 ]] || rm -rf "$OUT/factored-fb/$RUN_GROUP"/sd*
fi

declare -a PIDS=(); i=0
for s in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$(( i % NG ))]}"
    log="$LOG_DIR/state_s${s}.log"
    echo "[state] $(date +%T) seed=$s -> GPU $gpu  (log: ${log##*/})"
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" run_gciql.py \
            --config-name cube_single_state_crl_flowbc \
            seed="$s" train_steps="$TRAIN_STEPS" \
            save_interval="$SAVE_INTERVAL" eval_interval="$EVAL_INTERVAL" \
            wandb_mode="$WANDB_MODE" run_group="$RUN_GROUP" \
            "hydra.run.dir=$LOG_DIR/hydra_s${s}" > "$log" 2>&1 &
        PIDS+=("$!")
        sleep "$STAGGER"
    fi
    i=$((i+1))
done

echo "[state] launched ${#PIDS[@]} runs; waiting for completion (Ctrl-b d detaches tmux)..."
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
echo "[state] ALL DONE $(date +%T). logs=$LOG_DIR  out=$OUT"
