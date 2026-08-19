#!/bin/bash
# Regenerate the play-resample / play-resample-hard Metaworld datasets (v2 collector).
#
#   sbatch -A ASC26008 scripts/collect_play_resample_v2.slurm
#
# All four datasets collect in PARALLEL on one node -- collection is single-threaded MuJoCo
# on CPU, so four processes fit comfortably and the job finishes in about the time one
# dataset takes rather than four. Sized for the gh-dev 2h limit.
#
#SBATCH -J fb-collect-pr2
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -euo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export MUJOCO_GL=egl
OUT="${OUT_ROOT:-$SCRATCH/fb-data}"
EPISODES="${EPISODES:-200}"

# v1 is kept on disk as <regime>-v1 so the published v2 files can be diffed against the
# numbers already in the paper rather than silently replacing them.
for r in play-resample play-resample-hard; do
    if [ -d "$OUT/$r" ] && [ ! -d "$OUT/$r-v1" ]; then
        mv "$OUT/$r" "$OUT/$r-v1"
        echo "== archived $OUT/$r -> $OUT/$r-v1 =="
    fi
done

# v3: one timer for both tasks. seg_max=44 = 0.44 x the 100-step Metaworld eval horizon,
# matching cube's 0.435, and 500/44 = 11.4 relocations against cube's 11.56. No per-task
# tuning and no dwell: the timer alone is OGBench's trigger.


pids=()
for task in mw-push mw-pick-place; do
    $PY data_collection/collect_mw_play_resample.py \
        --task "$task" --episodes "$EPISODES" --seed 0 \
        --seg_max 44 \
        --out "$OUT/play-resample" &
    pids+=($!)
    # The hard variant differs only in the exploration mixture. It keeps eps_max at 0 --
    # the old 0.5 was the single most destructive setting in the collector, sending the puck
    # out of the workspace on 36% of steps with no recovery -- and raises the Gaussian only.
    # Note that opening BC headroom is now better served by obs_mask_dims at TRAINING time
    # than by degrading the data: masking removes the shortcut without inventing dead states.
    $PY data_collection/collect_mw_play_resample.py \
        --task "$task" --episodes "$EPISODES" --seed 0 \
        --seg_max 44 --noise_max 0.1 \
        --out "$OUT/play-resample-hard" &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "== a collection failed =="; exit 1; }

echo "== done =="
ls -la "$OUT/play-resample/data/metaworld" "$OUT/play-resample-hard/data/metaworld"
