#!/bin/bash
# Arm E: BC / GCIQL x FlowBC / biGCIQL x FlowBC on the UPSTREAM play-resample data
# (TommyShen/factored-fb-play), ManiSkill tasks, with the goal dims masked out of the
# observation.
#
#   bash scripts/launch_upstream_resample.sh
#   MASKED=0 bash scripts/launch_upstream_resample.sh     # unmasked control
#
# Why the mask. Every task in that upstream subset is goal-observable: the target sits in
# the observation and jumps at exactly goal_switch+1 (P(jump|switch+1) = 1.0000), at dims
# 25:28 for ms-push-cube and 26:29 for ms-pick-cube. Without masking, an unconditional BC
# actor reads the target off its own input, so BC == GCIQL == biGCIQL by construction and
# the arm measures nothing -- the same censoring already established on Metaworld.
#
# NOTE goal_mask_dims is NOT sufficient: it blanks the emitted goals only
# (impls/utils/datasets.py), leaving batch['observations'] intact. obs_mask_dims is the knob
# that actually closes the shortcut; both are set so the goal frame and the observation live
# in the same space.
set -euo pipefail
cd "$(dirname "$0")/.."

ACCOUNT="${ACCOUNT:-ASC26008}"
SEEDS="${SEEDS:-0 1 2}"
MASKED="${MASKED:-1}"
# ManiSkill needs its own environment (the ms backend is not installed in .venv-jax-fb).
VENV="${VENV:-$SCRATCH/.venv-ms}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/upstream-play-resample-packed}"

declare -A MASKDIMS=( [ms-push-cube]="[25,26,27]" [ms-pick-cube]="[26,27,28]" )

if [ "$MASKED" = "1" ]; then
    TAG_BASE="aE-upres"; SD="$SCRATCH/factored-fb-armE-masked"
else
    TAG_BASE="aE-upresNM"; SD="$SCRATCH/factored-fb-armE-nomask"
fi
MANIFEST="logs/armE-$(date +%Y%m%d-%H%M%S).txt"
mkdir -p logs

n=0
for task in ms-push-cube ms-pick-cube; do
    d="${MASKDIMS[$task]}"
    OV=""
    [ "$MASKED" = "1" ] && OV="obs_mask_dims=$d goal_mask_dims=$d"
    for combo in "gciql bcflow BC" "gciql flowbc GCIQL" "iql_bilinear flowbc biGCIQL"; do
        set -- $combo
        critic=$1; actor=$2; label=$3
        for s in $SEEDS; do
            name="${TAG_BASE}-${task#ms-}-${label}-s${s}"
            if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
                echo "SKIP(queued) $name" | tee -a "$MANIFEST"; continue
            fi
            id=$(CRITIC=$critic ACTOR=$actor DOMAIN=$task SEED=$s OVERRIDE="$OV" \
                 VENV=$VENV SAVE_DIR="$SD" FB_DATA_ROOT="$FB_DATA_ROOT" \
                 sbatch -A "$ACCOUNT" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
                 | grep -oE '^[0-9]+' | tail -1)
            if [ -n "$id" ]; then
                echo "$id $name  critic=$critic actor=$actor override='$OV'" | tee -a "$MANIFEST"
                n=$((n+1))
            else
                echo "FAILED  $name (submit limit?)" | tee -a "$MANIFEST"
            fi
        done
    done
done
echo "submitted $n jobs; FB_DATA_ROOT=$FB_DATA_ROOT SAVE_DIR=$SD"
echo "manifest: $MANIFEST"
