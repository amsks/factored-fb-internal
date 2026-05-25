#!/usr/bin/env bash
# scripts/run_fb_reweight.sh — coverage-balanced FB (reweighted) sweep over seeds.
# Implements docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md. Run in tmux:
#
#   tmux new -s fb_rw
#   bash scripts/run_fb_reweight.sh 2>&1 | tee /dev/shm/fb_reweight.log
#   DRY_RUN=1 bash scripts/run_fb_reweight.sh        # print the plan only
#
# Prereqs (already staged on this box):
#   - dataset: /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/*.npz
#     (scripts/extract_ogbench.py --env cube-single-play-v0 --dataset_path ~/.ogbench/data
#      --output_folder /dev/shm/factored-fb/datasets)
#   - density: analysis/cube_density.npz  (scripts/cube_density.py)
#
# Everything (runs, checkpoints, wandb, tmp) goes to /dev/shm — the durable / is full.
# wandb is online. ALPHA=0 reproduces the baseline exactly (plumbing sanity).
set -uo pipefail
trap '' HUP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
ALPHA="${ALPHA:-0.5}"                       # coverage-balance strength; spec: start mild (0.5)
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"             # seed i -> GPUS[i % nGPU]
ORTHO_COEF="${ORTHO_COEF:-1000}"            # paper's state run
LR_B="${LR_B:-1e-4}"
STEPS="${STEPS:-1000000}"
CLIP="${CLIP:-10.0}"
WANDB_ENTITY="${WANDB_ENTITY:-amsks}"       # ALWAYS amsks, never the al-laq-fb team default
WEIGHT_DIAG="${WEIGHT_DIAG:-true}"          # true=coverage-PRIORITIZED (diag weighted); false=pure rho'
WEIGHT_Z="${WEIGHT_Z:-true}"               # true=resample goal z by w (rare-goal exposure for F/actor)
# variant tag for wandb: prio/pure x +wz
_vtag=$([ "$WEIGHT_DIAG" = true ] && echo prio || echo pure)$([ "$WEIGHT_Z" = true ] && echo _wz || echo "")
WANDB_GROUP="${WANDB_GROUP:-covbal_${_vtag}_a${ALPHA}}"  # group the seeds in wandb (was null)
DATA="${DATA:-/dev/shm/factored-fb/datasets}"
DENSITY="${DENSITY:-$REPO/analysis/cube_density.npz}"
RUNROOT="${RUNROOT:-/dev/shm/factored-fb/runs}"
STAGGER="${STAGGER:-3}"
DRY_RUN="${DRY_RUN:-0}"

export TMPDIR="${TMPDIR:-/dev/shm/tmp_fb}";   mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_fb}"; mkdir -p "$WANDB_DIR"
TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUNROOT"
LOG_DIR="/dev/shm/fb_reweight_$TS"; mkdir -p "$LOG_DIR"

read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
read -ra SEED_ARR <<< "$SEEDS"

[[ -d "$DATA/cube-single-play-v0/buffer" ]] || { echo "[fb] MISSING dataset $DATA/cube-single-play-v0/buffer — run extract_ogbench.py"; exit 1; }
[[ -f "$DENSITY" || "$ALPHA" == "0" ]] || { echo "[fb] MISSING density $DENSITY — run scripts/cube_density.py (or ALPHA=0)"; exit 1; }

echo "[fb] alpha=$ALPHA ortho=$ORTHO_COEF steps=$STEPS seeds=[$SEEDS] gpus=[$GPUS] (i%${NG})"
echo "[fb] data=$DATA  density=$DENSITY"
echo "[fb] runs=$RUNROOT  logs=$LOG_DIR  (VOLATILE /dev/shm)"

declare -a PIDS=(); i=0
for s in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$(( i % NG ))]}"
    run_dir="$RUNROOT/${TS}__cube__covbal_${_vtag}_a${ALPHA}__s${s}"
    log="$LOG_DIR/fb_${_vtag}_a${ALPHA}_s${s}.log"
    echo "[fb] $(date +%T) seed=$s -> GPU $gpu  (log: ${log##*/})"
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" train.py \
            domain=cube_single \
            reweight_alpha="$ALPHA" reweight_clip="$CLIP" reweight_density_path="$DENSITY" \
            weight_diag="$WEIGHT_DIAG" weight_z="$WEIGHT_Z" \
            ortho_coef="$ORTHO_COEF" lr_b="$LR_B" num_train_steps="$STEPS" \
            data_path="$DATA" seed="$s" device=cuda \
            use_wandb=true wandb_entity="$WANDB_ENTITY" wandb_group="$WANDB_GROUP" \
            wandb_run_name="covbal_${_vtag}_a${ALPHA}_s${s}" \
            hydra.job.chdir=true "hydra.run.dir=$run_dir" \
            > "$log" 2>&1 &
        PIDS+=("$!")
        sleep "$STAGGER"
    fi
    i=$((i+1))
done

echo "[fb] launched ${#PIDS[@]} runs; waiting (Ctrl-b d detaches tmux)..."
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
echo "[fb] ALL DONE $(date +%T). runs=$RUNROOT logs=$LOG_DIR"
