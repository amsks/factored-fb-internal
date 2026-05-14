#!/usr/bin/env bash
# scripts/run_sweep.sh — Antmaze-medium + Cube-single sweep, N_PARALLEL at a time.
#
# Runs all (domain, ortho_coef, seed) combos in a parallel queue, logging each
# to logs/TIMESTAMP/{domain}_ortho{o}_s{seed}.log.
#
# Usage:
#   bash scripts/run_sweep.sh                    # full sweep, 6 parallel
#   N_PARALLEL=4 bash scripts/run_sweep.sh       # fewer parallel jobs
#   SEEDS="1 2" bash scripts/run_sweep.sh        # subset of seeds
#   DOMAINS="antmaze_medium" bash scripts/run_sweep.sh   # single domain
#
# Run inside tmux/screen so the SSH session can disconnect:
#   tmux new -s sweep
#   bash scripts/run_sweep.sh 2>&1 | tee logs/sweep_master.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/$TIMESTAMP"
N_PARALLEL="${N_PARALLEL:-6}"
SEEDS="${SEEDS:-1 2 3}"
DOMAINS="${DOMAINS:-antmaze_medium cube_single}"
ORTHOS="${ORTHOS:-100 1000}"

mkdir -p "$LOG_DIR"

# ── Build job list ────────────────────────────────────────────────────────
declare -a JOB_DOMAINS JOB_ORTHOS JOB_SEEDS JOB_GROUPS

for domain in $DOMAINS; do
    case "$domain" in
        antmaze_medium) group_base="antmaze-medium" ;;
        cube_single)    group_base="cube-single"    ;;
        *)              group_base="$domain"        ;;
    esac
    for ortho in $ORTHOS; do
        for seed in $SEEDS; do
            JOB_DOMAINS+=("$domain")
            JOB_ORTHOS+=("$ortho")
            JOB_SEEDS+=("$seed")
            JOB_GROUPS+=("${group_base}-ortho${ortho}")
        done
    done
done

N_JOBS="${#JOB_DOMAINS[@]}"
echo "[sweep] $N_JOBS jobs  |  up to $N_PARALLEL parallel  |  logs → $LOG_DIR"
echo "[sweep] domains: $DOMAINS"
echo "[sweep] ortho_coef: $ORTHOS"
echo "[sweep] seeds: $SEEDS"
echo ""

# ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
declare -a ACTIVE_PIDS=()
cleanup() {
    echo ""
    echo "[sweep] Caught signal — terminating running jobs..."
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}
trap cleanup INT TERM

# ── Parallel queue ────────────────────────────────────────────────────────
wait_for_slot() {
    while [ "${#ACTIVE_PIDS[@]}" -ge "$N_PARALLEL" ]; do
        local alive=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            else
                wait "$pid" && status=0 || status=$?
                echo "[sweep] $(date +%T)  pid=$pid finished (exit=$status)"
            fi
        done
        ACTIVE_PIDS=("${alive[@]+"${alive[@]}"}")
        [ "${#ACTIVE_PIDS[@]}" -ge "$N_PARALLEL" ] && sleep 3
    done
}

# ── Launch jobs ───────────────────────────────────────────────────────────
COMPLETED=0
for i in "${!JOB_DOMAINS[@]}"; do
    domain="${JOB_DOMAINS[$i]}"
    ortho="${JOB_ORTHOS[$i]}"
    seed="${JOB_SEEDS[$i]}"
    group="${JOB_GROUPS[$i]}"

    log="$LOG_DIR/${domain}_ortho${ortho}_s${seed}.log"
    run_name="${domain}__s${seed}__ortho${ortho}"

    wait_for_slot

    echo "[sweep] $(date +%T)  launching job $((i+1))/$N_JOBS: $run_name"
    (
        cd "$REPO"
        python train.py \
            domain="$domain" \
            ortho_coef="$ortho" \
            seed="$seed" \
            use_wandb=true \
            wandb_entity=amsks \
            wandb_group="$group" \
            "wandb_tags=[sweep-${TIMESTAMP}]" \
            wandb_run_name="$run_name" \
        > "$log" 2>&1
        echo "[sweep] $(date +%T)  DONE: $run_name"
    ) &
    ACTIVE_PIDS+=("$!")
done

# ── Wait for stragglers ───────────────────────────────────────────────────
echo "[sweep] $(date +%T)  all jobs launched — waiting for stragglers..."
for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    wait "$pid" && status=0 || status=$?
    echo "[sweep] $(date +%T)  pid=$pid finished (exit=$status)"
done

echo ""
echo "[sweep] ══════════════════════════════════════════════════"
echo "[sweep] Sweep complete: $N_JOBS runs in $LOG_DIR"
echo "[sweep] Quick tail of each log (last success/error line):"
for i in "${!JOB_DOMAINS[@]}"; do
    log="$LOG_DIR/${JOB_DOMAINS[$i]}_ortho${JOB_ORTHOS[$i]}_s${JOB_SEEDS[$i]}.log"
    last="$(tail -1 "$log" 2>/dev/null || echo '(no output)')"
    printf "  %-45s  %s\n" "${JOB_DOMAINS[$i]}_s${JOB_SEEDS[$i]}_ortho${JOB_ORTHOS[$i]}" "$last"
done
echo "[sweep] ══════════════════════════════════════════════════"
