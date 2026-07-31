#!/bin/bash
# Submit the Metaworld run matrix: 3 critics x 2 actors x 2 tasks x 3 seeds = 36 jobs.
#
#   ALLOC=ASC26008 bash scripts/launch_matrix.sh              # submit
#   DRY_RUN=1 bash scripts/launch_matrix.sh                   # print what would be submitted
#   SEEDS="0 1" DOMAINS=mw-bin-picking bash scripts/launch_matrix.sh   # a slice
#
# One job per (critic, actor, domain, seed) so a single failure costs one run, not the grid.
# Job ids are appended to logs/matrix-<stamp>.txt for later `sacct`/`squeue` lookups.
#
# The pairing: `fb` is the z-conditioned forward-backward critic, `iql_bilinear` the bilinear
# GCIQL bridge, `gciql` the plain goal-conditioned IQL baseline; each runs with both the
# flow-matching (`flowbc`) and DDPG+BC (`ddpgbc`) actors.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CRITICS="${CRITICS:-fb iql_bilinear gciql}"
ACTORS="${ACTORS:-flowbc ddpgbc}"
DOMAINS="${DOMAINS:-mw-bin-picking mw-button-press}"
SEEDS="${SEEDS:-0 1 2}"
export TRAIN_STEPS="${TRAIN_STEPS:-300000}"
export SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-exp}"
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"

STAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="logs/matrix-$STAMP.txt"
n=0

# Freeze the code the whole matrix runs against. Jobs drain over hours; pinning them to one
# snapshot means the tree stays editable without splitting the grid across two versions.
if [ "${FREEZE_CODE:-1}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    export CODE_DIR="${CODE_DIR:-$SCRATCH/factored-fb-code/$STAMP}"
    mkdir -p "$CODE_DIR"
    rsync -a --delete --exclude '.git' --exclude 'logs' --exclude 'exp' ./ "$CODE_DIR/"
    git rev-parse HEAD > "$CODE_DIR/GIT_HEAD" 2>/dev/null || true
    git status --porcelain > "$CODE_DIR/GIT_DIRTY" 2>/dev/null || true
    echo "== frozen code snapshot: $CODE_DIR =="
fi

for DOMAIN in $DOMAINS; do
  for CRITIC in $CRITICS; do
    for ACTOR in $ACTORS; do
      for SEED in $SEEDS; do
        name="$CRITIC-$ACTOR-${DOMAIN#mw-}-s$SEED"
        if [ "${DRY_RUN:-0}" = "1" ]; then
          echo "would submit: $name"
        else
          # TACC's submit filter prints a banner to stdout ahead of --parsable's job id, so
          # take the last bare-numeric line rather than the whole capture.
          id=$(CRITIC="$CRITIC" ACTOR="$ACTOR" DOMAIN="$DOMAIN" SEED="$SEED" \
               sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm \
               | grep -oE '^[0-9]+$' | tail -1)
          echo "${id:-SUBMIT_FAILED} $name" | tee -a "$MANIFEST"
        fi
        n=$((n + 1))
      done
    done
  done
done

echo "== $n job(s); manifest: $MANIFEST =="
