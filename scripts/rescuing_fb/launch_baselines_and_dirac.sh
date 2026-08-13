#!/bin/bash
# 2026-07-29 --- the two experiments the ICLR draft's own Limitations section concedes.
#
# EXP A (eval-only, minutes): cross-method arms through the IDENTICAL L2 harness.
#   Until now every L2 number compared the measure against its own oracle and its
#   own floor, so we could say it beats nothing, not that it beats an alternative.
#   Two arms already have checkpoints:
#     p1_s4{2,3}        first-passage = the REACHING VALUE (no revisit mass).
#                       Asks the paper's central unanswered question: does the
#                       visitation measure buy novel-reward transfer over a plain
#                       goal-reaching value? If p1 matches p2 here, phase 2 buys
#                       nothing on the axis it exists for.
#     hitdirac_s4{2,3}  FB's OWN loss (linear diag + ortho) at MATCHED ball support.
#                       Isolates loss form from support on the transfer axis.
#
# EXP B (training, ~2.5h x 4): the missing cell of the loss-form x support 2x2.
#                     FB loss        bounded loss
#     Dirac  I        0.42 (stock)   <- dirac arm (hit_identity=true)
#     ball            0.367          0.967 (phase 1)
#   Plus an "effector" arm: same bounded loss and ball radius, but the region is
#   built on the EFFECTOR slice (12:15) instead of the cube (19:22) -- an
#   abstraction that is not the evaluation variable. Caveat recorded up front:
#   the effector sits within 0.4 of the cube in 64.5% of dataset states, so this
#   is a CORRELATED alternative abstraction, not an independent one; a null here
#   is weaker evidence than a positive.
set -u
cd /home/adityamohan/git/Factored-FB
PY=.venv/bin/python
OUT=/home/adityamohan/exp_spec10/torchfb
LOG=/dev/shm/spec10_logs; mkdir -p "$LOG"
DATA=/dev/shm/factored-fb/datasets
[ -f "$DATA/cube-single-play-v0.npz" ] || { echo "[bd] FATAL: dataset missing"; exit 1; }

# ---------------------------------------------------------------- EXP A ----
expA() { # tag gpu
  local TAG=$1 G=$2
  [ -f "$OUT/$TAG/checkpoints/final.pt" ] || { echo "[A] $TAG: no ckpt, skip"; return 1; }
  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $OUT/$TAG --stage B --seed 0 \
    --out results/spec10_torchfb/L2_B_$TAG.json > $LOG/L2_B_$TAG.log 2>&1
  echo "[A] L2-B $TAG exit=$? $(date +%H:%M:%S)"
}

echo "[bd] === EXP A: cross-method baselines through the L2 harness ==="
expA p1_s42 0 &
expA p1_s43 1 &
expA hitdirac_s42 2 &
expA hitdirac_s43 3 &
wait
echo "[bd] EXP A done $(date +%H:%M:%S)"

# ---------------------------------------------------------------- EXP B ----
# Phase 1 (first_passage) throughout: the 2x2 is about the CONTROL claim, which
# is phase 1's. 500k steps, matched to p1_s4{2,3} in every other respect.
expB() { # tag seed gpu extra...
  local TAG=$1 S=$2 G=$3; shift 3
  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure_flowbc domain=cube_single \
    data_path=$DATA num_train_steps=500000 save_every=100000 \
    measure_mode=first_passage seed=$S "$@" \
    hydra.run.dir=$OUT/$TAG > $LOG/tfb_$TAG.log 2>&1
  echo "[B] $TAG exit=$? $(date +%H:%M:%S)"
}

echo "[bd] === EXP B: the missing 2x2 cell ==="
expB p1_dirac_s42    42 0 hit_identity=true &
expB p1_dirac_s43    43 1 hit_identity=true &
expB p1_effector_s42 42 2 "hit_slice=[12,15]" &
expB p1_effector_s43 43 3 "hit_slice=[12,15]" &
wait
echo "[bd] EXP B training done $(date +%H:%M:%S)"

# eval the new arms through the same harness (one GPU each, in parallel)
expA p1_dirac_s42    0 &
expA p1_dirac_s43    1 &
expA p1_effector_s42 2 &
expA p1_effector_s43 3 &
wait
echo "[bd] ALL DONE $(date +%H:%M:%S)"
cp -f $LOG/*dirac*.log $LOG/*effector*.log $LOG/L2_B_p1_*.log $LOG/L2_B_hitdirac_*.log $OUT/_logs_backup/ 2>/dev/null
cp -f "$0" $OUT/_logs_backup/ 2>/dev/null
