#!/usr/bin/env bash
# Run the FB pixel diagnostic profiling over all pixel seeds, then aggregate.
# The GCIQL-pixel branch activates only when GCIQL_ROOT is set (checkpoints
# arrive later). Set DRY_RUN=1 to print the commands without executing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/fb-pixel-results}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
N_EPISODES="${N_EPISODES:-10}"
BUFFER_SAMPLE="${BUFFER_SAMPLE:-20000}"
REPR_OUT="${REPR_OUT:-analysis/representation_profile_pixel}"
PHASE_OUT="${PHASE_OUT:-analysis/phase_probe_pixel}"
GCIQL_ROOT="${GCIQL_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; [ "$DRY_RUN" = "1" ] || "$@"; }

shopt -s nullglob
for d in "$RESULTS_ROOT"/*__s*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/.*__s([0-9]+)$/\1/')
  cfg="$d/.hydra/config.yaml"
  ckpt="$d/checkpoints/final.pt"
  run "$PY" scripts/representation_profile.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$REPR_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --n-episodes "$N_EPISODES" --buffer-sample "$BUFFER_SAMPLE"
  # Pixel phase funnel uses S0 only; S1/S2 counterfactuals require the pixel
  # wrapper to re-render after a physics mutation (not supported yet).
  run "$PY" scripts/phase_probe.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$PHASE_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --scenarios S0 --n-episodes "$N_EPISODES"
done

run "$PY" scripts/representation_profile_aggregate.py --root "$REPR_OUT"

if [ -n "$GCIQL_ROOT" ]; then
  for d in "$GCIQL_ROOT"/sd*; do
    [ -d "$d" ] || continue
    s=$(basename "$d")
    run "$PY" scripts/gciql_profile.py \
        --run-dir "$d" --out "analysis/gciql_profile_pixel/${s}_final" \
        --obs-type pixels --dataset-path "$DATA_PATH/cube-single-play-v0"
  done
fi
echo "[run_pixel_profile] done"
