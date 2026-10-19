#!/bin/bash
# One-time setup on a TACC Vista login node: JAX venv and MetaWorld datasets on $SCRATCH.
#
#   HF_TOKEN=hf_xxx bash scripts/vista_setup.sh
#
# Login nodes cap threads per user, so tools are pinned to few threads.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
HF_REPO="${HF_REPO:-TommyShen/Factor-FB-Offline}"
TASKS="${TASKS:-mw-bin-picking mw-button-press}"

export RAYON_NUM_THREADS=4 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_BUILDS=1
# Hub cache on $SCRATCH; the token stays in ~/.cache/huggingface
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/hf-cache}"

echo "[setup] repo   = $REPO"
echo "[setup] venv   = $VENV"
echo "[setup] data   = $HF_EXPORT"

# Venv (uv venv has no pip, so install through uv pip)
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"

# jax[cuda12] wheels bundle the CUDA runtime
echo "[setup] verifying imports (JAX shows CPU here; the GPU appears in the job)"
"$VENV/bin/python" -c 'import jax, metaworld; print("jax", jax.__version__)' || true
"$VENV/bin/python" - <<'PY'
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as E
print("[setup] metaworld ok:", len(E), "goal-observable envs")
PY

# Datasets (gated repo, needs HF_TOKEN)
echo "[setup] downloading packed datasets from $HF_REPO"
mkdir -p "$HF_EXPORT"
FB_HF_EXPORT="$HF_EXPORT" HF_REPO="$HF_REPO" TASKS="$TASKS" "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import hf_hub_download

dest, repo = os.environ["FB_HF_EXPORT"], os.environ["HF_REPO"]
files = ["manifest.json"]
for task in os.environ["TASKS"].split():
    files += [f"data/metaworld/{task}.npz", f"data/metaworld/{task}.meta.json"]
for f in files:
    p = hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
    print("[setup]  ", p)
PY

echo
echo "[setup] done."
echo "[setup] Submit training with:  sbatch -A <alloc> scripts/vista_train.slurm"
