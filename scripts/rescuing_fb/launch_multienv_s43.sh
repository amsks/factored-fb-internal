#!/bin/bash
# Spec 14 §4 — second seed for the torch multi-env measure checkpoints.
#
# WHY. Every torch scene / cube-double checkpoint is seed 42 only
# (MULTIENV-RESULTS.md open action #2: add a second seed before quoting
# anything from them). Spec 14 scores those checkpoints on the measure axis;
# a single-seed measure table is not quotable and would have to be re-run.
# GPUs 3-6 were idle at launch time, so this costs idle GPU hours only.
#
# WHAT. Two independent chains, phase 1 -> phase 2, byte-matching the s42
# `.hydra/overrides.yaml` in every field except `seed` and the warm-start path:
#
#   scene        hit_spec 19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6
#   cube-double  hit_spec 19:22:0.4,28:31:0.4
#
# Phase 2 carries the adopted rzK50 stack (freeze_b, nhat_head,
# reward_z_ratio=0.5 reward_z_family=kernel) exactly as s42 did. Validity
# penalties stay OFF on multi-env phase 2 — the mass anchor's in-batch N-hat
# clamps at 2/B at conjunctive hit rates (ledger 9.15 multi-env note).
#
# NO EVALS here: the measure-eval scripts are cube-single-only until spec 14 3.4
# lands. Training only; `run_multienv_measure.sh` scores s42 and s43 together.
set -u
cd /home/adityamohan/git/Factored-FB
PY=.venv/bin/python
OUT=/home/adityamohan/exp_spec10/torchfb
LOG=/dev/shm/spec10_logs; mkdir -p "$LOG"
DATA=/dev/shm/factored-fb/datasets

for d in scene-play-v0 cube-double-play-v0; do
  if [ ! -d "$DATA/$d/buffer" ]; then
    echo "[s43] FATAL: dataset missing at $DATA/$d (/dev/shm gets wiped)"; exit 1
  fi
done

# tag|domain|hit_spec|gpu
CHAINS=(
  "scn|scene|19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6|3"
  "cdbl|cube_double|19:22:0.4,28:31:0.4|4"
)
SEED=43

run_chain() { # tag domain hit_spec gpu
  local TAG=$1 DOM=$2 HS=$3 G=$4
  local P1=$OUT/p1_${TAG}_s$SEED
  local P2=$OUT/p2_${TAG}_s$SEED

  if [ -f "$P1/checkpoints/final.pt" ]; then
    echo "[s43] p1 $TAG already complete - skipping to phase 2"
  else
    CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure_flowbc domain=$DOM \
      data_path=$DATA num_train_steps=500000 save_every=100000 \
      measure_mode=first_passage hit_spec="'$HS'" \
      seed=$SEED hydra.run.dir=$P1 \
      > $LOG/tfb_p1_${TAG}_s$SEED.log 2>&1
    echo "[s43] p1 $TAG exit=$? $(date +%H:%M:%S)"
  fi
  [ -f "$P1/checkpoints/final.pt" ] || { echo "[s43] $TAG: no phase-1 ckpt, chain stops"; return 1; }

  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure_flowbc domain=$DOM \
    data_path=$DATA num_train_steps=500000 save_every=100000 \
    measure_mode=visitation freeze_b=true nhat_head=true \
    reward_z_ratio=0.5 reward_z_family=kernel hit_spec="'$HS'" \
    warm_start_ckpt=$P1/checkpoints/final.pt \
    seed=$SEED hydra.run.dir=$P2 \
    > $LOG/tfb_p2_${TAG}_s$SEED.log 2>&1
  echo "[s43] p2 $TAG exit=$? $(date +%H:%M:%S)"
}

for c in "${CHAINS[@]}"; do
  IFS='|' read -r tag dom hs gpu <<<"$c"
  echo "[s43] $(date +%H:%M:%S) launch chain $tag (domain=$dom) on GPU$gpu"
  run_chain "$tag" "$dom" "$hs" "$gpu" &
  sleep 20
done
wait
echo "[s43] MULTI-ENV SEED-43 CHAINS DONE $(date +%H:%M:%S)"
mkdir -p $OUT/_logs_backup
cp -f $LOG/tfb_p*_{scn,cdbl}_s$SEED.log $OUT/_logs_backup/ 2>/dev/null
cp -f "$0" $OUT/_logs_backup/ 2>/dev/null
