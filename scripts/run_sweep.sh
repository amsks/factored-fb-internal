#!/usr/bin/env bash
# scripts/run_sweep.sh — Antmaze-medium + Cube-single sweep, N_PARALLEL at a time.
#
# Mirrors td_jepa's sweep grid exactly:
#   ortho_coef ∈ {100, 1000}  x  lr_b ∈ {1e-4, 1e-5}  =  4 combos per domain
#
# td_jepa ran 10 seeds; we default to 3. Adjust SEEDS to taste.
#
# Usage:
#   bash scripts/run_sweep.sh                          # full grid, 6 parallel
#   N_PARALLEL=4 bash scripts/run_sweep.sh             # fewer parallel jobs
#   SEEDS="1 2" bash scripts/run_sweep.sh              # fewer seeds
#   DOMAINS="antmaze_medium" bash scripts/run_sweep.sh # single domain
#
# Run inside tmux so the session survives SSH disconnect:
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
LR_BS="${LR_BS:-1e-4 1e-5}"

mkdir -p "$LOG_DIR"

# ── Build job list ────────────────────────────────────────────────────────
declare -a JOB_DOMAINS JOB_ORTHOS JOB_LR_BS JOB_SEEDS JOB_GROUPS

for domain in $DOMAINS; do
    case "$domain" in
        antmaze_medium) group_base="antmaze-medium" ;;
        cube_single)    group_base="cube-single"    ;;
        *)              group_base="$domain"        ;;
    esac
    for ortho in $ORTHOS; do
        for lr_b in $LR_BS; do
            for seed in $SEEDS; do
                JOB_DOMAINS+=("$domain")
                JOB_ORTHOS+=("$ortho")
                JOB_LR_BS+=("$lr_b")
                JOB_SEEDS+=("$seed")
                JOB_GROUPS+=("${group_base}-ortho${ortho}-lrb${lr_b}")
            done
        done
    done
done

N_JOBS="${#JOB_DOMAINS[@]}"
echo "[sweep] $N_JOBS jobs  |  up to $N_PARALLEL parallel  |  logs → $LOG_DIR"
echo "[sweep] domains:    $DOMAINS"
echo "[sweep] ortho_coef: $ORTHOS"
echo "[sweep] lr_b:       $LR_BS"
echo "[sweep] seeds:      $SEEDS"
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
for i in "${!JOB_DOMAINS[@]}"; do
    domain="${JOB_DOMAINS[$i]}"
    ortho="${JOB_ORTHOS[$i]}"
    lr_b="${JOB_LR_BS[$i]}"
    seed="${JOB_SEEDS[$i]}"
    group="${JOB_GROUPS[$i]}"

    # shorten lr_b for filenames: 1e-4 → 1e4, 1e-5 → 1e5
    lr_tag="${lr_b//-/}"
    log="$LOG_DIR/${domain}_ortho${ortho}_lrb${lr_tag}_s${seed}.log"
    run_name="${domain}__s${seed}__ortho${ortho}__lrb${lr_b}"

    wait_for_slot

    echo "[sweep] $(date +%T)  launching job $((i+1))/$N_JOBS: $run_name"
    (
        cd "$REPO"
        python train.py \
            domain="$domain" \
            ortho_coef="$ortho" \
            lr_b="$lr_b" \
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
echo "[sweep] Quick tail of each log (last line):"
for i in "${!JOB_DOMAINS[@]}"; do
    lr_tag="${JOB_LR_BS[$i]//-/}"
    log="$LOG_DIR/${JOB_DOMAINS[$i]}_ortho${JOB_ORTHOS[$i]}_lrb${lr_tag}_s${JOB_SEEDS[$i]}.log"
    last="$(tail -1 "$log" 2>/dev/null || echo '(no output)')"
    tag="${JOB_DOMAINS[$i]}_s${JOB_SEEDS[$i]}_ortho${JOB_ORTHOS[$i]}_lrb${JOB_LR_BS[$i]}"
    printf "  %-52s  %s\n" "$tag" "$last"
done
echo "[sweep] ══════════════════════════════════════════════════"
