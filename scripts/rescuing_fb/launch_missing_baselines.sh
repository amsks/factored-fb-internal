#!/usr/bin/env bash
# 2026-08-05 --- fill the "---" (not run) cells of the ICLR draft's Table 1.
#
# edited_main.tex tab:main leaves seven baseline cells empty. This script runs the
# five that need no code change; the other two are BLOCKED and recorded below.
#
#   row                       cube-s   cube-d        scene
#   CRL                        0.311   [C1] launch   [C2] launch
#   Tuned FB                   0.42    [F1] launch   0.047 (done)
#   FB (reference reimpl.)     0.532   [F2] launch   [F3] launch
#   FB loss on regions         0.284   BLOCKED       BLOCKED
#
# BLOCKED -- "FB loss on regions" on the multi-object envs. The control is stock
# FB's loss at MATCHED ball support, so on cube-double/scene it must query the
# same CONJUNCTIVE region the measure arm uses (hit_spec '19:22:0.4,28:31:0.4'
# and the five-part scene spec). train.py threads `hit_spec` only into the
# fb_measure_flowbc branch (train.py:333); HitDiracAgent.__init__ takes a single
# `hit_slice` pair (agents/fb/hitdirac/agent.py:49) and has no conjunctive path.
# Running it with the default single cube slice would silently compare a
# one-object region against a two/five-object region -- exactly the support
# confound the 2x2 exists to remove. Needs a small agent change + a test first.
#
# Recipes are copied from the runs that produced the cube-single numbers, so the
# only variable is the environment:
#   Tuned FB              exp_spec10/torchfb (FB-State-Flow)  agent=fb_flowbc, ortho_coef from the domain yaml (1.0)
#   FB (reference reimpl) exp_spec10/torchfb/fb_stock_s0      agent=fb_flowbc ortho_coef=1000
#   CRL                   configs/gciql/cube_single_state_crl_flowbc.yaml
#
# Run:      tmux new -d -s missbase 'scripts/rescuing_fb/launch_missing_baselines.sh 2>&1 | tee /dev/shm/missbase/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_missing_baselines.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_missbase}
LOG=/dev/shm/missbase; mkdir -p "$LOG" "$OUT"
DRY_RUN=${DRY_RUN:-0}
STEPS=${STEPS:-500000}
SEEDS=${SEEDS:-"42 43"}
DATA=/dev/shm/factored-fb/datasets

for d in cube-double-play-v0 scene-play-v0; do
  [ -d "$DATA/$d/buffer" ] || { echo "[mb] FATAL: torch buffer missing: $DATA/$d/buffer"; exit 1; }
done

run() { # tag gpu cmd...
  local TAG=$1 G=$2; shift 2
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] GPU$G $TAG :: $*"; return 0; fi
  echo "[mb] start $TAG on GPU$G $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$G "$@" > "$LOG/$TAG.log" 2>&1
  echo "[mb] done  $TAG exit=$? $(date +%H:%M:%S)"
}

# ------------------------------------------------------------------ torch FB --
# wandb_entity is pinned to amsks: the team default is wrong for this project.
fb() { # tag gpu domain seed extra...
  local TAG=$1 G=$2 DOM=$3 S=$4; shift 4
  run "$TAG" "$G" $PY train.py agent=fb_flowbc domain="$DOM" \
    data_path=$DATA num_train_steps=$STEPS save_every=100000 eval_every=100000 \
    seed="$S" use_wandb=true wandb_entity=amsks wandb_group="mb_$TAG" \
    device=cuda "$@" hydra.run.dir=$OUT/$TAG
}

# --------------------------------------------------------------------- CRL ----
crl() { # tag gpu config seed
  local TAG=$1 G=$2 CFG=$3 S=$4
  run "$TAG" "$G" $PY run_gciql.py --config-name "$CFG" \
    seed="$S" train_steps=$STEPS eval_interval=100000 \
    hydra.run.dir=$OUT/$TAG
}

# Explicit GPU per job: an arithmetic counter does not survive the background
# forks, and silently stacking six trainers on GPU 0 wastes the whole wave.
fb  tunedfb_cdbl_s42 0  cube_double 42                  &   # F1
fb  fbref_cdbl_s42   1  cube_double 42 ortho_coef=1000  &   # F2
fb  fbref_scn_s42    2  scene       42 ortho_coef=1000  &   # F3
fb  tunedfb_cdbl_s43 3  cube_double 43                  &
fb  fbref_cdbl_s43   4  cube_double 43 ortho_coef=1000  &
fb  fbref_scn_s43    5  scene       43 ortho_coef=1000  &
# six torch jobs occupy GPUs 0-5; CRL takes 6 and 7, two seeds at a time.
i=0
for S in 0 1; do
  crl "crl_cdbl_s$S" 6 cube_double_state_crl_flowbc "$S" &   # C1
  crl "crl_scn_s$S"  7 scene_state_crl_flowbc       "$S" &   # C2
  wait                                                       # serialise seeds on 6/7
done
wait
echo "[mb] ALL DONE $(date +%H:%M:%S)"
