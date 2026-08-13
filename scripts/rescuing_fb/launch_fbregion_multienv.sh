#!/usr/bin/env bash
# 2026-08-05 --- the last two "---" cells of the ICLR draft's tab:main:
# "FB loss on regions" on cube-double and scene.
#
# This arm is stock FB's loss (quadratic offdiag + LINEAR diag + ortho) at
# MATCHED ball support. On the multi-object envs "matched" means the same
# CONJUNCTIVE region the measure arm trains on, which HitDiracAgent could not
# express until today (hit_spec support + tests, 2026-08-05). Running it with
# the default single cube slice would have compared a one-object region against
# a two/five-object region -- the exact support confound the loss-form x support
# 2x2 exists to remove.
#
# Matched to hitdirac_s4{2,3,4} on cube-single (0.284) in every other respect;
# the hit_spec values are the envs' own success criteria, the same strings the
# phase-1 measure runs used (exp_spec10/torchfb/p1_cdbl_*, p1_scn_*).
#
# NOTE the quoting: hit_spec must reach hydra WITH its single quotes, or the
# comma makes the value ambiguous ("To use it as a list, use key=[v1,v2]").
#
# Run:      tmux new -d -s fbregion 'scripts/rescuing_fb/launch_fbregion_multienv.sh 2>&1 | tee /dev/shm/fbregion/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_fbregion_multienv.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_missbase}
LOG=/dev/shm/fbregion; mkdir -p "$LOG" "$OUT"
DRY_RUN=${DRY_RUN:-0}
STEPS=${STEPS:-500000}
DATA=/dev/shm/factored-fb/datasets

CDBL_SPEC="19:22:0.4,28:31:0.4"
SCN_SPEC="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"

hd() { # tag gpu domain seed spec
  local TAG=$1 G=$2 DOM=$3 S=$4 SPEC=$5
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] GPU$G $TAG domain=$DOM seed=$S spec=$SPEC"; return 0; fi
  echo "[fr] start $TAG on GPU$G $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_hitdirac_flowbc domain="$DOM" \
    data_path=$DATA num_train_steps=$STEPS save_every=100000 eval_every=100000 \
    seed="$S" "hit_spec='$SPEC'" use_wandb=true wandb_entity=amsks \
    wandb_group="fr_$TAG" device=cuda hydra.run.dir=$OUT/$TAG > "$LOG/$TAG.log" 2>&1
  echo "[fr] done  $TAG exit=$? $(date +%H:%M:%S)"
}

# Shares GPUs 0-3 with the launch_missing_baselines.sh wave: those trainers hold
# ~424 MiB and ~25% util each on 23 GB L4s, so this is a CPU-side add, not a
# memory risk.
hd fbregion_cdbl_s42 0 cube_double 42 "$CDBL_SPEC" &
hd fbregion_cdbl_s43 1 cube_double 43 "$CDBL_SPEC" &
hd fbregion_scn_s42  2 scene       42 "$SCN_SPEC"  &
hd fbregion_scn_s43  3 scene       43 "$SCN_SPEC"  &
wait
echo "[fr] ALL DONE $(date +%H:%M:%S)"
