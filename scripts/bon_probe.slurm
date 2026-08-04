#!/bin/bash
# Best-of-N re-ranking probe (eval/bon_probe.py) — one (domain, critic) curve per job.
#
#   DOMAIN=cube-single CRITIC=iql_bilinear CRITIC_OVERRIDE=value_latent_dim=50 \
#   CRITIC_PATH=... CRITIC_EPOCH=500000 PROPOSAL_CRITIC=qrl ... \
#   sbatch -A ASC26008 scripts/bon_probe.slurm
#
#SBATCH -J bon-probe
#SBATCH -p gh-dev                 # short probe: fits the 2h dev queue, separate from gh cap
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/expert}"
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
export WANDB_MODE=offline

echo "== bon: $CRITIC on $DOMAIN (proposal $PROPOSAL_CRITIC/${PROPOSAL_ACTOR:-bcflow}) =="
$PY -m eval.bon_probe \
  --domain "$DOMAIN" \
  --proposal_critic "$PROPOSAL_CRITIC" --proposal_actor "${PROPOSAL_ACTOR:-bcflow}" \
  --proposal_override "${PROPOSAL_OVERRIDE:-}" \
  --proposal_path "$PROPOSAL_PATH" --proposal_epoch "${PROPOSAL_EPOCH:-500000}" \
  --critic "$CRITIC" --actor "${ACTOR:-flowbc}" \
  --critic_override "${CRITIC_OVERRIDE:-}" \
  --critic_path "$CRITIC_PATH" --critic_epoch "${CRITIC_EPOCH:-500000}" \
  --goal_mode "${GOAL_MODE:-fixed}" \
  --n_list "${N_LIST:-1,8,64}" --episodes "${EPISODES:-10}" --seed "${SEED:-0}" \
  --out "$OUT"
