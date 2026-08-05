#!/bin/bash
# Arms A (rank scaling), D (OOD-goal split), C (MW goal-channel repair) — one drip-fed
# matrix (docs/plans/2026-08-04-iclr-bilinear-roadmap.md). Idempotent: skips rows whose
# run dir already has a final eval.csv or whose job name is queued; waits out the
# per-user submit cap. Run AFTER the armprep job has downloaded the new OGBench datasets
# and passed the GCDataset knob smokes.
#
#   CODE_DIR=<frozen snapshot> bash scripts/launch_arms_acd.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:?set CODE_DIR to the armACD snapshot}"
VENV=$SCRATCH/.venv-ms
MANIFEST="logs/armACD-$(date +%Y%m%d-%H%M%S).txt"

submit() { # NAME [sbatch extra args via SBEXTRA]; knobs via environment
  local name=$1
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return
  fi
  local id
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name" | tee -a "$MANIFEST"
}

# ---- Arm A: rank scaling --------------------------------------------------------------
# antmaze replaced pointmaze (OGBench flags pointmaze as behaviorally anomalous).
# ARMA_ENVS override lets a resume pass run before the antmaze dataset is downloaded.
for DOM in ${ARMA_ENVS:-cube-double scene antmaze}; do
  SD=$SCRATCH/factored-fb-armA/$DOM
  for ROW in "iql_bilinear|flowbc|value_latent_dim=50|biql-d50" \
             "iql_bilinear|flowbc|value_latent_dim=512|biql-d512" \
             "gciql|flowbc||gciql" \
             "iql_bilinear|bcflow|value_latent_dim=50|bc"; do
    IFS='|' read -r CR AC OV TAG <<< "$ROW"
    for S in 0 1 2; do
      NAME=aA-$DOM-$TAG-s$S
      # File-based skip: a finished 500k run at either the per-TAG path (new layout) or
      # the SAVE_DIR root (pre-dirclash-fix layout, collision-free rows only) counts done.
      ENVN=$(grep -oE 'env_name: \S+' env/domains/$DOM.yaml | cut -d' ' -f2)
      done=0
      for base in "$SD/$TAG" "$SD"; do
        f="$base/${ENVN}_${CR}_${AC}_seed_${S}/eval.csv"
        [ -f "$f" ] && tail -1 "$f" | tr -d '\0' | grep -q ",500000$" && done=1
      done
      [ "$done" = 1 ] && { echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; }
      # SAVE_DIR gets a per-TAG subdir: the run-dir name is {domain}_{critic}_{actor}_seed
      # and does NOT encode overrides, so d50/d512 rows would clobber each other otherwise.
      CRITIC=$CR ACTOR=$AC DOMAIN=$DOM SEED=$S OVERRIDE="$OV" \
      TRAIN_STEPS=500000 EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=2 \
      CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR=$SD/$TAG \
      submit "$NAME"
    done
  done
done

# ---- Arm A: ground-truth kernel spectra (cheap, gh-dev) --------------------------------
for DOM in cube-double-play-v0 scene-play-v0 antmaze-medium-navigate-v0; do
  NAME=aA-spec-${DOM%%-play*}; NAME=${NAME%%-navigate*}
  if ! squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q . && \
     [ ! -f "$SCRATCH/factored-fb-armA/spectrum_$DOM.json" ]; then
    sbatch -A "$ALLOC" -p gh-dev -t 01:30:00 -N 1 -n 1 -J "$NAME" -o logs/%x.o%j -e logs/%x.e%j \
      --wrap "cd $CODE_DIR && $VENV/bin/python -m eval.kernel_spectrum --data ~/.ogbench/data/${DOM}-val.npz --out $SCRATCH/factored-fb-armA/spectrum_$DOM.json" \
      | tee -a "$MANIFEST"
  fi
done

# ---- Arm D: OOD-goal split on cube ----------------------------------------------------
FILT="goal_filter_dim=19 goal_filter_thresh=0.0 goal_filter_side=below"
SD=$SCRATCH/factored-fb-armD
for ROW in "iql_bilinear|flowbc|value_latent_dim=50 $FILT|biql-d50" \
           "gciql|flowbc|$FILT|gciql" \
           "iql_bilinear|bcflow|value_latent_dim=50 $FILT|bc"; do
  IFS='|' read -r CR AC OV TAG <<< "$ROW"
  for S in 0 1 2; do
    NAME=aD-$TAG-s$S
    CRITIC=$CR ACTOR=$AC DOMAIN=cube-single SEED=$S OVERRIDE="$OV" \
    TRAIN_STEPS=500000 EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=2 \
    CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR=$SD \
    submit "$NAME"
  done
done

# ---- Arm C: MW train-side goal-channel repair (play-noise-hard) -----------------------
MASK="goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8"
for DOM in mw-push mw-pick-place; do
  DSH=${DOM#mw-}
  SD=$SCRATCH/factored-fb-armC
  for CR in iql_bilinear gciql; do
    CRS=$([ "$CR" = iql_bilinear ] && echo biql || echo gciql)
    for AC in bcflow flowbc; do
      for S in 0 1 2; do
        NAME=aC-$DSH-$CRS-$AC-s$S
        CRITIC=$CR ACTOR=$AC DOMAIN=$DOM SEED=$S OVERRIDE="$MASK" \
        FB_DATA_ROOT=$SCRATCH/fb-data/play-noise-hard \
        CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR=$SD \
        submit "$NAME"
      done
    done
  done
done
echo "== armACD launcher done; manifest $MANIFEST =="
