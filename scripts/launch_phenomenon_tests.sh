#!/bin/bash
# Launch the phenomenon.tex mechanism kill-test TRAINING arms (Pred. mech.1/2 + the d-sweep
# gap point), 3 seeds x 500k on OGBench cube-single, into the metric-probe run tree so the
# probe battery (scripts/metric_probe.slurm) picks the checkpoints up unchanged.
#
#   bash scripts/launch_phenomenon_tests.sh
#   DRY_RUN=1 bash scripts/launch_phenomenon_tests.sh
#
# Arms (label|critic|actor|override):
#   gciql-qb75    — bounded monolithic: kappa*tanh(x/kappa), kappa=75 = the bilinear class
#                   bound ||phi||*||psi||/sqrt(d) observed on biql-d50 (max|V| ~ 62, hard
#                   cap ~ 75, vs the true V* range [-100, 0]). Pred. mech.1: recovers a
#                   large fraction of the 73->93 gap iff boundedness is the asset.
#   biql-d50-nopsiln — de-LayerNormed psi tower (phi untouched). Pred. mech.2: norm
#                   concentration is load-bearing => success drops toward gciql.
#   biql-d15      — the one gap left in the d-sweep curve (d10 0.62 -> d20 0.92).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-ms}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-metric-probe}"
SEEDS="${SEEDS:-0 1 2}"
MAX_QUEUE="${MAX_QUEUE:-38}"

ROWS=(
  "gciql-qb75|gciql|flowbc|q_bound=75.0"
  "biql-d50-nopsiln|iql_bilinear|flowbc|value_latent_dim=50 psi_layer_norm=false"
  "biql-d15|iql_bilinear|flowbc|value_latent_dim=15"
)

STAMP="$(date +%Y%m%d-%H%M%S)"
CODE_DIR="$SCRATCH/factored-fb-code/phen-$STAMP"
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "$CODE_DIR"
  rsync -a --delete --exclude '.git' --exclude 'logs' --exclude 'exp' ./ "$CODE_DIR/"
  git rev-parse HEAD > "$CODE_DIR/GIT_HEAD" 2>/dev/null || true
  echo "== frozen snapshot: $CODE_DIR =="
fi
MANIFEST="logs/phen-tests-$STAMP.txt"
n=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r label critic actor override <<< "$row"
  for SEED in $SEEDS; do
    name="ph-$label-s$SEED"
    if [ "${DRY_RUN:-0}" = "1" ]; then
      echo "would submit: $name  [critic=$critic actor=$actor override='$override']"
    else
      while [ "$(squeue -h -u "$USER" 2>/dev/null | wc -l)" -ge "$MAX_QUEUE" ]; do sleep 30; done
      id=$(CRITIC="$critic" ACTOR="$actor" DOMAIN="cube-single" SEED="$SEED" \
           TRAIN_STEPS=500000 EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 \
           NUM_VIDEO=0 OVERRIDE="$override" CODE_DIR="$CODE_DIR" VENV="$VENV" \
           SAVE_DIR="$SAVE_DIR/$label" \
           sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm \
           | grep -oE '^[0-9]+$' | tail -1)
      echo "${id:-FAIL} $name" | tee -a "$MANIFEST"
    fi
    n=$((n + 1))
  done
done
echo "== phenomenon tests: $n runs; manifest=$MANIFEST; save_dir=$SAVE_DIR =="
