#!/bin/bash
# Throttled campaign launcher for the benchmark × data_type × critic × actor table.
#
# The gh QOS rejects (DenyOnLimit) submissions past 40 queued, so a 200-job campaign can't be
# fired in one shot. This drip-feeds: it keeps the queue filled to MAX_QUEUE and blocks when
# full, so the whole matrix drains without any submission being refused.
#
#   REGIME=expert DATA_ROOT=$SCRATCH/fb-data/expert DOMAINS="mw-pick-place mw-push" \
#     VENV=$SCRATCH/.venv-ms bash scripts/launch_campaign.sh
#
# One SAVE_DIR per regime (so the table's --root DTYPE=DIR mapping stays clean). Code is frozen
# to one snapshot for the whole campaign (jobs drain over hours; the tree stays editable).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
REGIME="${REGIME:?set REGIME (expert|play-noise|play-resample)}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (holds data/<family>/<env>.npz)}"
CRITICS="${CRITICS:-fb gciql iql_bilinear crl}"
ACTORS="${ACTORS:-flowbc ddpgbc bcflow}"
DOMAINS="${DOMAINS:?set DOMAINS}"
SEEDS="${SEEDS:-0 1 2}"
VENV="${VENV:-$SCRATCH/.venv-ms}"                  # universal venv (jax + sapien + metaworld)
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-exp-$REGIME}"
MAX_QUEUE="${MAX_QUEUE:-38}"                        # stay under the 40 DenyOnLimit ceiling

STAMP="$(date +%Y%m%d-%H%M%S)"
CODE_DIR="$SCRATCH/factored-fb-code/$STAMP"
mkdir -p "$CODE_DIR"
rsync -a --delete --exclude '.git' --exclude 'logs' --exclude 'exp' ./ "$CODE_DIR/"
git rev-parse HEAD > "$CODE_DIR/GIT_HEAD" 2>/dev/null || true
echo "== frozen snapshot: $CODE_DIR =="

MANIFEST="logs/campaign-$REGIME-$STAMP.txt"
n=0
for DOMAIN in $DOMAINS; do
  for CRITIC in $CRITICS; do
    for ACTOR in $ACTORS; do
      for SEED in $SEEDS; do
        # block while the queue is full
        while [ "$(squeue -h -u "$USER" 2>/dev/null | wc -l)" -ge "$MAX_QUEUE" ]; do
          sleep 30
        done
        name="$REGIME-$CRITIC-$ACTOR-${DOMAIN}-s$SEED"
        id=$(CRITIC="$CRITIC" ACTOR="$ACTOR" DOMAIN="$DOMAIN" SEED="$SEED" \
             CODE_DIR="$CODE_DIR" VENV="$VENV" TRAIN_STEPS="$TRAIN_STEPS" \
             SAVE_DIR="$SAVE_DIR" FB_DATA_ROOT="$DATA_ROOT" \
             sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm \
             | grep -oE '^[0-9]+$' | tail -1)
        echo "${id:-FAIL} $name" | tee -a "$MANIFEST"
        n=$((n + 1))
      done
    done
  done
done
echo "== submitted $n jobs for regime=$REGIME; manifest=$MANIFEST; save_dir=$SAVE_DIR =="
