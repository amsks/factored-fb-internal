#!/bin/bash
# Best-of-N rerank probe matrix (phenomenon.tex Remark "bon"; supersedes bridge Exp 1).
#
# Critic-as-selector: frozen BC proposal, execute argmax_i Q(s, a_i, g), N in {1,8,64}.
# Rows: cube (3 critics x 3 seeds x 2 proposal sources) + MW play-noise-hard
# (2 envs x 2 critics x 3 seeds x 2 proposal sources). Skips rows whose OUT json already
# exists or whose job name is already in the queue (so re-running is idempotent while the
# seed-0 pilots are in flight). Submission waits out the per-user cap instead of failing.
#
#   bash scripts/launch_bon_probe.sh                # submit everything missing
#   DRY_RUN=1 bash scripts/launch_bon_probe.sh      # print the matrix
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$SCRATCH/factored-fb-code/baselines-20260804-013505}"
MP=$SCRATCH/factored-fb-metric-probe
PNH=$SCRATCH/factored-fb-exp-play-noise-hard
OUTD=$MP/bon
mkdir -p "$OUTD"
MANIFEST="logs/bon-$(date +%Y%m%d-%H%M%S).txt"

QRL_PROP="qrl|qrl_components=8 qrl_dim_per_component=8|$MP/qrl-d64/cube-single-play-v0_qrl_bcflow_seed_0|500000"
VIP_PROP="vip|latent_dim=50 discount=0.98|$MP/vip-d50/cube-single-play-v0_vip_bcflow_seed_0|500000"

submit() { # NAME then env-assignments via environment
  local name=$1; shift
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "would submit: $name"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return
  fi
  local id
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/bon_probe.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name" | tee -a "$MANIFEST"
}

# ---- cube: critics {biql-d50, gciql, crl-d50} x seeds {0,1,2} x proposals {qrl, vip(s0 only)}
for PROPROW in "$QRL_PROP" "$VIP_PROP"; do
  IFS='|' read -r PC PO PP PE <<< "$PROPROW"
  SEEDS="0 1 2"; [ "$PC" = vip ] && SEEDS="0"     # vip source: seed-0 crossing check only
  for S in $SEEDS; do
    for ROW in \
      "iql_bilinear|value_latent_dim=50|$MP/biql-d50/cube-single-play-v0_iql_bilinear_flowbc_seed_$S|biql" \
      "gciql||$MP/gciql/cube-single-play-v0_gciql_flowbc_seed_$S|gciql" \
      "crl|latent_dim=50|$MP/crl-d50/cube-single-play-v0_crl_flowbc_seed_$S|crl"; do
      IFS='|' read -r CR CO CP TAG <<< "$ROW"
      OUT=$OUTD/bon_cube_${TAG}_p${PC}_s$S.json
      [ -f "$OUT" ] && { echo "SKIP(done) bon-cube-$TAG-p$PC-s$S" | tee -a "$MANIFEST"; continue; }
      # keep the seed-0 qrl pilots' names/outputs (launched 2026-08-04 pre-launcher)
      NAME=bon-cube-$TAG; OUTX=$OUT
      if [ "$PC" = qrl ] && [ "$S" = 0 ]; then OUTX=$OUTD/bon_cube_${TAG}_s0.json; else NAME=bon-cube-$TAG-p$PC-s$S; fi
      [ -f "$OUTX" ] && { echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; }
      DOMAIN=cube-single PROPOSAL_CRITIC=$PC PROPOSAL_OVERRIDE="$PO" PROPOSAL_PATH=$PP PROPOSAL_EPOCH=$PE \
      CRITIC=$CR CRITIC_OVERRIDE="$CO" CRITIC_PATH=$CP CRITIC_EPOCH=500000 \
      EPISODES=10 CODE_DIR=$CODE_DIR OUT=$OUTX \
      submit "$NAME"
    done
  done
done

# ---- MW play-noise-hard: envs x critics {biql,gciql} x seeds x proposals {biql-bcflow, gciql-bcflow (s0 only)}
for DOM in mw-push mw-pick-place; do
  DSH=${DOM#mw-}
  for PSRC in iql_bilinear gciql; do
    PSH=$([ "$PSRC" = iql_bilinear ] && echo biql || echo gciql)
    SEEDS="0 1 2"; [ "$PSRC" = gciql ] && SEEDS="0"
    for S in $SEEDS; do
      for CR in iql_bilinear gciql; do
        CRS=$([ "$CR" = iql_bilinear ] && echo biql || echo gciql)
        OUT=$OUTD/bon_pnh_${DSH}_${CRS}_p${PSH}_s$S.json
        NAME=bon-$DSH-$CRS-p$PSH-s$S
        if [ "$PSRC" = iql_bilinear ] && [ "$S" = 0 ]; then
          NAME=bon-$DSH-$CRS; OUT=$OUTD/bon_pnh_${DSH}_${CRS}_s0.json   # seed-0 pilot names
        fi
        [ -f "$OUT" ] && { echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; }
        DOMAIN=$DOM FB_DATA_ROOT=$SCRATCH/fb-data/play-noise-hard \
        PROPOSAL_CRITIC=$PSRC PROPOSAL_PATH=$PNH/${DOM}_${PSRC}_bcflow_seed_0 PROPOSAL_EPOCH=300000 \
        CRITIC=$CR CRITIC_PATH=$PNH/${DOM}_${CR}_flowbc_seed_$S CRITIC_EPOCH=300000 \
        EPISODES=20 CODE_DIR=$CODE_DIR OUT=$OUT \
        submit "$NAME"
      done
    done
  done
done
echo "== bon matrix: manifest $MANIFEST =="
