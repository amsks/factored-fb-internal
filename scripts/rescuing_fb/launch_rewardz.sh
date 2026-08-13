#!/bin/bash
# Reward-z campaign (2026-07-29) — spec 12 / PHASE2-LEDGER 9.10 next cell.
#
# WHY. L2 (zero-shot, eval-only, lambda=0 phase-2 ckpts) found that novel-reward
# transfer WORKS when the reward is spatially concentrated (gap_frac ball0.4
# 0.87/0.74, ball0.8 0.70/0.61, twozone 0.84/0.89 — composition works) and dies
# as reward bandwidth grows (smooth0.8 0.23/0.08, ball1.6 ~ floor). L2w then
# showed the 1/N-hat density correction fixes PREDICTION scale (0.014-0.03 ->
# 0.79-2.2) but rescues CONTROL nowhere and regresses it where it worked. That
# splits the defect off the inference rule and onto the ACTOR: its trained z law
# is 50% gaussian directions + 50% B(one goal), and a broad reward's inferred
# z = project(sum_i r_i B(s'_i)) is an average of many B's lying off both.
#
# WHAT. Add a third z component at TRAIN time, drawn exactly as the eval readout
# draws it (agents/fb/agent.py:_draw_reward_z). Phase 1 is UNCHANGED and reused
# from the existing chain, so the only difference from the control is phase 2's
# z law.
#
#   control  exp_spec10/torchfb/p2_s4{2,3}     reward_z_ratio=0.0 (already run)
#   rzK25    kernel  family, ratio 0.25        eval families' FORM, random params
#   rzK50    kernel  family, ratio 0.50        dose-response on the same arm
#   rzR25    rff     family, ratio 0.25        DIFFERENT form => L2's families
#                                              stay held out (non-circular claim)
#
# PRE-REGISTERED READ (write the verdict into PHASE2-LEDGER 9.11 either way):
#   PRIMARY   stage-B gap_frac on the broad families rises: smooth0.8 from
#             0.23/0.08 and ball1.6 from ~0 to >0.4, WITHOUT losing the
#             concentrated ones (ball0.4 >=0.7, twozone >=0.7).
#   GUARD 1   task success (eval.csv, zero-shot relabel protocol) stays within
#             one seed-sd of the control's 0.887 — a z law that buys broad-reward
#             steering by giving up control is not a win.
#   GUARD 2   L1 set-mass ratio stays ~1.1-1.2x at hw 0.8/1.6 and Spearman stays
#             >=0.7 — the measure must not decalibrate to fit the wider z set.
#   NULL      if rzK (matched form) also fails, the defect is NOT the actor's
#             z-manifold and the next suspect is F's capacity to be accurate at
#             averaged z (the bilinear rank bound), not the sampling law.
#   If rzK works and rzR does not, the claim is family-conditional: report it as
#   "trained on the eval family's form", not as general zero-shot transfer.
set -u
cd /home/adityamohan/git/Factored-FB
PY=.venv/bin/python
OUT=/home/adityamohan/exp_spec10/torchfb
LOG=/dev/shm/spec10_logs; mkdir -p "$LOG"
DATA=/dev/shm/factored-fb/datasets

if [ ! -f "$DATA/cube-single-play-v0.npz" ]; then
  echo "[rz] FATAL: dataset missing at $DATA (/dev/shm gets wiped)"; exit 1
fi

# tag|family|ratio|seed|gpu   (GPUs 5,6 left free for the other session)
JOBS=(
  "rzK25_s42|kernel|0.25|42|0"
  "rzK25_s43|kernel|0.25|43|1"
  "rzK50_s42|kernel|0.50|42|2"
  "rzK50_s43|kernel|0.50|43|3"
  "rzR25_s42|rff|0.25|42|4"
  "rzR25_s43|rff|0.25|43|7"
)

run_one() { # tag family ratio seed gpu
  local TAG=$1 FAM=$2 RATIO=$3 S=$4 G=$5
  local P1=$OUT/p1_s$S
  local P2=$OUT/p2_$TAG

  if [ ! -f "$P1/checkpoints/final.pt" ]; then
    echo "[rz] $TAG: phase-1 ckpt $P1 MISSING - skip"; return 1
  fi

  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure_flowbc domain=cube_single \
    data_path=$DATA num_train_steps=500000 save_every=100000 \
    measure_mode=visitation freeze_b=true nhat_head=true \
    reward_z_ratio=$RATIO reward_z_family=$FAM \
    warm_start_ckpt=$P1/checkpoints/final.pt seed=$S hydra.run.dir=$P2 \
    > $LOG/tfb_p2_$TAG.log 2>&1
  echo "[rz] p2 $TAG exit=$? $(date +%H:%M:%S)"
  [ -f "$P2/checkpoints/final.pt" ] || { echo "[rz] $TAG: no ckpt, no evals"; return 1; }

  # PRIMARY: zero-shot control on the 48 never-trained rewards (stage B), then
  # prediction (stage A), then the L1 calibration guard. Same flags as the
  # control's runs so the numbers are directly comparable to L2_B_p2_s4{2,3}.
  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $P2 --stage B --seed 0 --out results/spec10_torchfb/L2_B_$TAG.json \
    > $LOG/L2_B_$TAG.log 2>&1
  echo "[rz] L2-B $TAG exit=$? $(date +%H:%M:%S)"

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $P2 --stage A --seed 0 --out results/spec10_torchfb/L2_A_$TAG.json \
    > $LOG/L2_A_$TAG.log 2>&1
  echo "[rz] L2-A $TAG exit=$? $(date +%H:%M:%S)"

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_full_measure.py \
    --run $P2 --starts 32 --mc 3 --horizon 200 --boxes 12 --rho-samples 200000 \
    --seed 0 --out results/spec10_torchfb/L1fix_$TAG.json \
    > $LOG/L1fix_$TAG.log 2>&1
  echo "[rz] L1 $TAG exit=$? $(date +%H:%M:%S)"
}

for j in "${JOBS[@]}"; do
  IFS='|' read -r tag fam ratio seed gpu <<<"$j"
  echo "[rz] $(date +%H:%M:%S) launch $tag (family=$fam ratio=$ratio seed=$seed) on GPU$gpu"
  run_one "$tag" "$fam" "$ratio" "$seed" "$gpu" &
  sleep 20
done
wait
echo "[rz] REWARD-Z CAMPAIGN DONE $(date +%H:%M:%S)"
# /dev/shm gets wiped — back the logs up (ledger volatility note)
cp -f $LOG/*rz*.log $OUT/_logs_backup/ 2>/dev/null
cp -f "$0" $OUT/_logs_backup/ 2>/dev/null
