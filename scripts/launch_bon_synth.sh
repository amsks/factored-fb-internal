#!/bin/bash
# Goal-synthesis rerank arm (Pred-context b + bon): rerun the MW play-noise-hard bon rows
# with GOAL_MODE=synth (per-episode goal built from the episode's own target) instead of
# the fixed dataset frame. Pre-registered prediction: curves flip from monotone-decreasing
# to cube-shaped (selection value turns positive).
#
#   bash scripts/launch_bon_synth.sh              # submit missing rows (idempotent)
#   DRY_RUN=1 bash scripts/launch_bon_synth.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$SCRATCH/factored-fb-code/baselines-20260804-013505}"
PNH=$SCRATCH/factored-fb-exp-play-noise-hard
OUTD=$SCRATCH/factored-fb-metric-probe/bon
MANIFEST="logs/bon-synth-$(date +%Y%m%d-%H%M%S).txt"

for DOM in mw-push mw-pick-place; do
  DSH=${DOM#mw-}
  for CR in iql_bilinear gciql; do
    CRS=$([ "$CR" = iql_bilinear ] && echo biql || echo gciql)
    for S in 0 1 2; do
      NAME=bons-$DSH-$CRS-s$S
      OUT=$OUTD/bon_pnh_${DSH}_${CRS}_synth_s$S.json
      [ -f "$OUT" ] && { echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; }
      if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
        echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue
      fi
      if [ "${DRY_RUN:-0}" = "1" ]; then echo "would submit: $NAME"; continue; fi
      until id=$(DOMAIN=$DOM FB_DATA_ROOT=$SCRATCH/fb-data/play-noise-hard \
          PROPOSAL_CRITIC=iql_bilinear PROPOSAL_PATH=$PNH/${DOM}_iql_bilinear_bcflow_seed_0 PROPOSAL_EPOCH=300000 \
          CRITIC=$CR CRITIC_PATH=$PNH/${DOM}_${CR}_flowbc_seed_$S CRITIC_EPOCH=300000 \
          GOAL_MODE=synth EPISODES=20 CODE_DIR=$CODE_DIR OUT=$OUT \
          sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/bon_probe.slurm 2>/dev/null \
          | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
        echo "cap hit, waiting 300s ($NAME)"; sleep 300
      done
      echo "$id $NAME" | tee -a "$MANIFEST"
    done
  done
done
echo "== bon-synth: manifest $MANIFEST =="
