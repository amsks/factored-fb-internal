#!/bin/bash
# One-time environment setup for TACC Vista (GH200 / aarch64).
# Run this ON A LOGIN NODE (it needs internet for pip + the offline datasets).
#
#   HF_TOKEN=hf_xxx bash scripts/vista_setup.sh
#
# It builds a PERSISTENT JAX venv on $SCRATCH and downloads the packed Metaworld datasets
# there, because compute nodes have no outbound internet and cannot see a login node's
# /dev/shm — the venv and the data must both be materialized here, up front.
#
# NOTE on threads: Vista login nodes cap each user at ~100 threads/processes. uv's rayon
# pool and JAX's Eigen pool both size themselves to the core count and get killed by that
# cap, so every tool below is pinned to a small thread count. (JAX itself is only *verified*
# here; it is exercised for real in the job, on a compute node.)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
HF_REPO="${HF_REPO:-TommyShen/Factor-FB-Offline}"
TASKS="${TASKS:-mw-bin-picking mw-button-press}"

export RAYON_NUM_THREADS=4 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_BUILDS=1
# Blobs go to $SCRATCH (/home1 is a 23 GB quota); the credential stays at its default
# ~/.cache/huggingface/token, which is where `hf auth login` writes it.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/hf-cache}"

echo "[setup] repo   = $REPO"
echo "[setup] venv   = $VENV"
echo "[setup] data   = $HF_EXPORT"

# ── venv ────────────────────────────────────────────────────────────────────
# `uv venv` does not install pip, so drive installs through `uv pip --python` rather than
# the venv's own pip.
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"

# jax[cuda12] (pulled by ogbench[train]) ships aarch64+CUDA wheels — no system CUDA module
# needed; the wheels bundle the CUDA runtime and only use the driver.
echo "[setup] verifying imports (JAX shows CPU here; the GPU appears in the job)"
"$VENV/bin/python" -c 'import jax, metaworld; print("jax", jax.__version__)' || true
"$VENV/bin/python" - <<'PY'
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as E
print("[setup] metaworld ok:", len(E), "goal-observable envs")
PY

# ── datasets ────────────────────────────────────────────────────────────────
# The HF repo is GATED: request access on the dataset page and export HF_TOKEN (a read
# token) before running, or the download 401s.
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
