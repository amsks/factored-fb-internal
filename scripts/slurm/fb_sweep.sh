#!/usr/bin/env bash
# scripts/slurm/fb_sweep.sh — Submit FB state sweep to the ai,tnt SLURM cluster
# via Hydra submitit. Each (domain × seed × ortho_coef × lr_b) becomes one job.
#
# Cluster defaults (partition, GPUs, mem, time) come from
# configs/hydra/launcher/slurm.yaml. WANDB_MODE=offline is set in that file's
# setup commands so it propagates to compute nodes.
#
# Prerequisites:
#   make install           # PyTorch .venv/ in repo root (on Lustre)
#   python scripts/extract_ogbench.py --output_folder /bigwork/nhwpmoha/Factored-FB/datasets
#
# Usage:
#   bash scripts/slurm/fb_sweep.sh
#   bash scripts/slurm/fb_sweep.sh --group my-run --domains antmaze_medium --seeds 1,2,3
#   bash scripts/slurm/fb_sweep.sh --mem-gb 24      # override memory per job
#   bash scripts/slurm/fb_sweep.sh --dry-run        # print command, do not submit

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# ── Defaults ──────────────────────────────────────────────────────────────────
DOMAINS="antmaze_medium,cube_single"
SEEDS="1,2,3,4,5,6,7,8,9,10"
ORTHOS="100,1000"
LR_BS="1e-4,1e-5"
GROUP="fb-state-sweep"
DATA_PATH="/bigwork/nhwpmoha/Factored-FB/datasets"
MEM_GB=""
DRY_RUN=0

usage() { grep '^#' "$0" | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domains)  DOMAINS="$2";   shift 2 ;;
        --seeds)    SEEDS="$2";     shift 2 ;;
        --orthos)   ORTHOS="$2";    shift 2 ;;
        --lr-bs)    LR_BS="$2";     shift 2 ;;
        --group)    GROUP="$2";     shift 2 ;;
        --data)     DATA_PATH="$2"; shift 2 ;;
        --mem-gb)   MEM_GB="$2";    shift 2 ;;
        --dry-run)  DRY_RUN=1;      shift   ;;
        -h|--help)  usage ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "$REPO/.venv/bin/activate"

LAUNCHER_OVERRIDES=("hydra/launcher=slurm")
[[ -n "$MEM_GB" ]] && LAUNCHER_OVERRIDES+=("hydra.launcher.mem_gb=${MEM_GB}")

CMD=(
    python train.py
    --multirun
    "${LAUNCHER_OVERRIDES[@]}"
    "domain=${DOMAINS}"
    "seed=${SEEDS}"
    "ortho_coef=${ORTHOS}"
    "lr_b=${LR_BS}"
    "use_wandb=true"
    "wandb_entity=amsks"
    "wandb_project=factored-fb"
    "wandb_group=${GROUP}"
    "data_path=${DATA_PATH}"
)

echo "[fb-sweep] domains=${DOMAINS}  seeds=${SEEDS}  orthos=${ORTHOS}  lr_bs=${LR_BS}"
echo "[fb-sweep] group=${GROUP}  data=${DATA_PATH}"
echo "[fb-sweep] cmd: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[fb-sweep] DRY RUN — nothing submitted."
    exit 0
fi

"${CMD[@]}"
