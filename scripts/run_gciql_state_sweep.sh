#!/usr/bin/env bash
# scripts/run_gciql_state_sweep.sh — Parallel multi-seed GCIQL state sweep.
#
# Same ergonomics/scheduler as scripts/run_sweep.sh (the FB sweep), but for the
# vendored OGBench GCIQL track: cube-single-play-v0, 1M steps, OGBench's EXACT
# hyperparameters (alpha=1.0 from the config; nothing else overridden). 10 seeds
# (1..10) — matching td_jepa/FB — run in parallel across N_GPUS via a GPU-slot
# scheduler. wandb online by default so GCIQL eval metrics overlay with FB runs
# (mirrored to FB's wandb key scheme by tools/wandb_mode_shim).
#
# Checkpoints: save_interval=100000 (10 checkpoints/run, not just the final
# step). Output lives on tmpfs (/dev/shm, ephemeral on reboot); a background
# rsync mirrors it to a durable home-dir directory every SYNC_INTERVAL seconds,
# with a final sync after all seeds finish.
#
# Usage:
#   bash scripts/run_gciql_state_sweep.sh                       # 10 seeds, 8 GPUs
#   N_GPUS=4 bash scripts/run_gciql_state_sweep.sh              # 4 GPUs
#   SEEDS="1 2 3" bash scripts/run_gciql_state_sweep.sh         # fewer seeds
#   SYNC_INTERVAL=300 bash scripts/run_gciql_state_sweep.sh     # sync every 5m
#
# Run inside tmux so it survives SSH disconnect:
#   tmux new -s gciql
#   bash scripts/run_gciql_state_sweep.sh 2>&1 | tee logs/gciql_state_master.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/gciql_state_$TIMESTAMP"

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
N_GPUS="${N_GPUS:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
N_PARALLEL=$(( N_GPUS * JOBS_PER_GPU ))
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-gciql}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
STORAGE="${STORAGE:-shm}"
SHM_OUT="/dev/shm/gciql_outputs"
DURABLE_OUT="${DURABLE_OUT:-$HOME/gciql_runs/$TIMESTAMP}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"

mkdir -p "$LOG_DIR" "$DURABLE_OUT"

read -ra SEED_ARR <<< "$SEEDS"
N_JOBS="${#SEED_ARR[@]}"

echo "[gciql-sweep] $N_JOBS seeds | $N_GPUS GPUs | $JOBS_PER_GPU job(s)/GPU ($N_PARALLEL parallel)"
echo "[gciql-sweep] seeds:      $SEEDS"
echo "[gciql-sweep] env:        cube-single-play-v0 (state), 1M steps, alpha=1.0"
echo "[gciql-sweep] wandb:      $WANDB_MODE  group=$RUN_GROUP"
echo "[gciql-sweep] save_intvl: $SAVE_INTERVAL"
echo "[gciql-sweep] shm out:    $SHM_OUT"
echo "[gciql-sweep] durable:    $DURABLE_OUT  (rsync every ${SYNC_INTERVAL}s)"
echo "[gciql-sweep] logs:       $LOG_DIR"
echo ""

# ── Background durable-sync loop ───────────────────────────────────────────
sync_once() { rsync -a --delete "$SHM_OUT/" "$DURABLE_OUT/" 2>/dev/null || true; }
(
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_once
        echo "[gciql-sweep] $(date +%T)  synced -> $DURABLE_OUT"
    done
) &
SYNC_PID="$!"

# ── Graceful shutdown ─────────────────────────────────────────────────────
declare -a ACTIVE_PIDS=()
cleanup() {
    echo ""
    echo "[gciql-sweep] Caught signal — terminating jobs + sync..."
    kill "$SYNC_PID" 2>/dev/null || true
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    sync_once
    exit 1
}
trap cleanup INT TERM

# ── GPU slot tracker (slot i -> GPU i % N_GPUS) ───────────────────────────
declare -a SLOT_PIDS=()
for (( s=0; s<N_PARALLEL; s++ )); do SLOT_PIDS+=(""); done

find_free_slot() {
    while true; do
        for (( s=0; s<N_PARALLEL; s++ )); do
            pid="${SLOT_PIDS[$s]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                if [[ -n "$pid" ]]; then
                    wait "$pid" && status=0 || status=$?
                    echo "[gciql-sweep] $(date +%T)  pid=$pid slot=$s finished (exit=$status)"
                fi
                echo "$s"
                return
            fi
        done
        sleep 3
    done
}

# ── Launch one job per seed ───────────────────────────────────────────────
for idx in "${!SEED_ARR[@]}"; do
    seed="${SEED_ARR[$idx]}"
    log="$LOG_DIR/gciql_state_s${seed}.log"

    slot=$(find_free_slot)
    gpu=$(( slot % N_GPUS ))

    echo "[gciql-sweep] $(date +%T)  launching $((idx+1))/$N_JOBS seed=$seed on GPU $gpu (slot $slot)"
    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$gpu" python run_gciql.py \
            --config-name cube_single_state \
            storage="$STORAGE" \
            seed="$seed" \
            save_interval="$SAVE_INTERVAL" \
            wandb_mode="$WANDB_MODE" \
            run_group="$RUN_GROUP" \
            hydra.run.dir="$LOG_DIR/hydra_s${seed}" \
        > "$log" 2>&1
        echo "[gciql-sweep] $(date +%T)  DONE seed=$seed (exit=$?)"
    ) &
    pid="$!"
    SLOT_PIDS[$slot]="$pid"
    ACTIVE_PIDS+=("$pid")
done

# ── Wait for stragglers ───────────────────────────────────────────────────
echo "[gciql-sweep] $(date +%T)  all $N_JOBS seeds launched — waiting..."
for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    wait "$pid" && status=0 || status=$?
    echo "[gciql-sweep] $(date +%T)  pid=$pid finished (exit=$status)"
done

kill "$SYNC_PID" 2>/dev/null || true
echo "[gciql-sweep] final sync -> $DURABLE_OUT"
sync_once

echo ""
echo "[gciql-sweep] ══════════════════════════════════════════════════"
echo "[gciql-sweep] Complete: $N_JOBS seeds. Logs: $LOG_DIR"
echo "[gciql-sweep] Durable checkpoints: $DURABLE_OUT"
for seed in "${SEED_ARR[@]}"; do
    log="$LOG_DIR/gciql_state_s${seed}.log"
    last="$(tail -1 "$log" 2>/dev/null || echo '(no output)')"
    printf "  seed %-3s  %s\n" "$seed" "$last"
done
echo "[gciql-sweep] ══════════════════════════════════════════════════"
