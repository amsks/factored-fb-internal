#!/usr/bin/env bash
# Run the FB pixel diagnostic profiling over all pixel seeds, then aggregate.
# The GCIQL-pixel branch activates only when GCIQL_ROOT is set (checkpoints
# arrive later). Set DRY_RUN=1 to print the commands without executing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
# GCIQL profiling rolls out a JAX checkpoint and needs the isolated JAX venv
# (the torch .venv has no jax). The GCIQL *aggregate* is numpy/matplotlib only
# and runs under PY.
JAX_PY="${JAX_PY:-.venv-jax-cpu/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/fb-pixel-results}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
N_EPISODES="${N_EPISODES:-10}"
BUFFER_SAMPLE="${BUFFER_SAMPLE:-20000}"
REPR_OUT="${REPR_OUT:-analysis/representation_profile_pixel}"
PHASE_OUT="${PHASE_OUT:-analysis/phase_probe_pixel}"
GCIQL_ROOT="${GCIQL_ROOT:-}"
GCIQL_STEP="${GCIQL_STEP:-500000}"
GCIQL_OUT="${GCIQL_OUT:-analysis/gciql_profile_pixel}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; [ "$DRY_RUN" = "1" ] || "$@"; }

shopt -s nullglob
for d in "$RESULTS_ROOT"/*__s*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/.*__s([0-9]+)$/\1/')
  cfg="$d/.hydra/config.yaml"
  # Prefer final.pt; fall back to the latest step_*.pt (not every seed saves
  # a final checkpoint). Skip the seed if neither exists.
  ckpt="$d/checkpoints/final.pt"
  if [ ! -f "$ckpt" ]; then
    ckpt=$(ls "$d/checkpoints/"step_*.pt 2>/dev/null | sort -V | tail -1)
  fi
  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    echo "[warn] no checkpoint for seed $s under $d/checkpoints; skipping"
    continue
  fi
  echo "[run_pixel_profile] seed $s checkpoint: $ckpt"
  run "$PY" scripts/representation_profile.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$REPR_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --n-episodes "$N_EPISODES" --buffer-sample "$BUFFER_SAMPLE" \
      || echo "[warn] representation_profile failed for seed $s"
  # Pixel phase funnel uses S0 only; S1/S2 counterfactuals require the pixel
  # wrapper to re-render after a physics mutation (not supported yet).
  run "$PY" scripts/phase_probe.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$PHASE_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --scenarios S0 --n-episodes "$N_EPISODES" \
      || echo "[warn] phase_probe failed for seed $s"
done

run "$PY" scripts/representation_profile_aggregate.py --root "$REPR_OUT"

if [ -n "$GCIQL_ROOT" ]; then
  for d in "$GCIQL_ROOT"/sd*; do
    [ -d "$d" ] || continue
    s=$(basename "$d")
    run "$JAX_PY" scripts/gciql_profile.py \
        --run-dir "$d" --step "$GCIQL_STEP" \
        --out "$GCIQL_OUT/${s}_final" \
        --obs-type pixels --dataset-path "$DATA_PATH/cube-single-play-v0"
  done
  # FB-vs-GCIQL comparison (numpy/matplotlib only -> PY).
  run "$PY" scripts/gciql_profile_aggregate.py \
      --root "$GCIQL_OUT" \
      --fb-aggregate "$REPR_OUT/aggregate" --fb-seed-root "$REPR_OUT"
fi
echo "[run_pixel_profile] done"
