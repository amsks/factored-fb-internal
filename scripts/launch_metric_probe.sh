#!/bin/bash
# The metric-probe run matrix (docs/specs/2026-07-31-bilinear-gciql-metric-probe.md §5):
# does bilinear GCIQL's embedding encode a temporal metric? Trains the subject + VIP/QRL/CRL
# references on OGBench cube-single-play, 500k steps, checkpoint every 100k so probes can run
# across training. Actor: value-driven flowbc for critics with a real Q (iql_bilinear/gciql/
# crl), pure-BC bcflow for the representation-only ones (vip/qrl).
#
#   bash scripts/launch_metric_probe.sh            # submit (throttled under the 40-cap)
#   DRY_RUN=1 bash scripts/launch_metric_probe.sh  # print the matrix
#
# SANITY GATE (spec §5): run #1 (iql_bilinear d50) must reach >= ~85% success by 500k. Watch it
# before trusting the rest; if it floors, stop and debug.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-ms}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-metric-probe}"
SEEDS="${SEEDS:-0 1 2}"
NUM_VIDEO="${NUM_VIDEO:-4}"
MAX_QUEUE="${MAX_QUEUE:-38}"

# One row per spec §5: "label|critic|actor|override". Seeds expanded below (row 10 is 1 seed).
ROWS=(
  "biql-d50|iql_bilinear|flowbc|value_latent_dim=50"
  "biql-d512|iql_bilinear|flowbc|value_latent_dim=512"
  "gciql|gciql|flowbc|"
  "crl-d50|crl|flowbc|latent_dim=50"
  "crl-d512|crl|flowbc|latent_dim=512"
  "qrl-d64|qrl|bcflow|qrl_components=8 qrl_dim_per_component=8"
  "qrl-d512|qrl|bcflow|qrl_components=64 qrl_dim_per_component=8"
  "vip-d50|vip|bcflow|latent_dim=50 discount=0.98"
  "vip-d512|vip|bcflow|latent_dim=512 discount=0.98"
  "vip-d50-g99|vip|bcflow|latent_dim=50 discount=0.99"     # single-seed sensitivity (row 10)
)

STAMP="$(date +%Y%m%d-%H%M%S)"
CODE_DIR="$SCRATCH/factored-fb-code/metric-$STAMP"
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "$CODE_DIR"
  rsync -a --delete --exclude '.git' --exclude 'logs' --exclude 'exp' ./ "$CODE_DIR/"
  git rev-parse HEAD > "$CODE_DIR/GIT_HEAD" 2>/dev/null || true
  echo "== frozen snapshot: $CODE_DIR =="
fi
MANIFEST="logs/metric-probe-$STAMP.txt"
n=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r label critic actor override <<< "$row"
  seeds="$SEEDS"; [ "$label" = "vip-d50-g99" ] && seeds="0"    # row 10: 1 seed
  for SEED in $seeds; do
    name="mp-$label-s$SEED"
    if [ "${DRY_RUN:-0}" = "1" ]; then
      echo "would submit: $name  [critic=$critic actor=$actor override='$override']"
    else
      while [ "$(squeue -h -u "$USER" 2>/dev/null | wc -l)" -ge "$MAX_QUEUE" ]; do sleep 30; done
      id=$(CRITIC="$critic" ACTOR="$actor" DOMAIN="cube-single" SEED="$SEED" \
           TRAIN_STEPS=500000 EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 \
           NUM_VIDEO="$NUM_VIDEO" OVERRIDE="$override" CODE_DIR="$CODE_DIR" VENV="$VENV" \
           SAVE_DIR="$SAVE_DIR/$label" \
           sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm \
           | grep -oE '^[0-9]+$' | tail -1)
      echo "${id:-FAIL} $name" | tee -a "$MANIFEST"
    fi
    n=$((n + 1))
  done
done
echo "== metric-probe: $n runs; manifest=$MANIFEST; save_dir=$SAVE_DIR =="
