#!/usr/bin/env bash
# 2026-08-05 --- region-FB WITHOUT the flow actor, phase 1, three envs, 2 seeds.
#
# Every region-FB number the project has is a flow-actor number: the measure
# critic was welded to FBFlowBCAgent by inheritance until today's
# RegionMeasureMixin refactor. This arm is the same critic behind FBAgent's
# native TD3+BC actor -- ONE changed axis, bc_coeff matched at 3.0.
#
# What it is for. CRL+flow reaches 0.384 on scene (this box, 2026-08-05) while
# torch region-FB+flow floors there (p1_scn 0.00/0.10, p1_cdbl 0.04/0.00),
# even though the jax region-FB reports 0.484/0.549 on the same envs. So
# "region-FB with flow" currently has two contradictory answers on multi-object
# and no answer at all without flow. This arm splits objective from actor:
#
#   cube-single  flow works (0.96/0.86). If no-flow also works, the actor is
#                not load-bearing there.
#   cube-double  flow floors in torch. If no-flow floors too, the objective (or
#                the torch stack) is at fault, not the actor. If no-flow
#                RECOVERS, the flow actor is the bottleneck on conjunctive envs
#                -- which would be the answer to "why can't the two-phase
#                method leverage this expressive actor".
#   scene        same question where CRL+flow does its best work.
#
# Phase 2 is deliberately NOT launched: it warm-starts from these checkpoints,
# so it is chained only after phase 1 reads.
#
# Run:      tmux new -d -s noflow 'scripts/rescuing_fb/launch_noflow_region.sh 2>&1 | tee /dev/shm/noflow/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_noflow_region.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_noflow}
LOG=/dev/shm/noflow; mkdir -p "$LOG" "$OUT"
DRY_RUN=${DRY_RUN:-0}
STEPS=${STEPS:-500000}
DATA=/dev/shm/factored-fb/datasets

CDBL_SPEC="19:22:0.4,28:31:0.4"
SCN_SPEC="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"

nf() { # tag gpu domain seed [spec]
  local TAG=$1 G=$2 DOM=$3 S=$4 SPEC=${5:-}
  local -a EXTRA=()
  [ -n "$SPEC" ] && EXTRA+=("hit_spec='$SPEC'")
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] GPU$G $TAG domain=$DOM seed=$S ${EXTRA[*]:-<single-ball>}"; return 0; fi
  echo "[nf] start $TAG on GPU$G $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure domain="$DOM" \
    data_path=$DATA num_train_steps=$STEPS save_every=100000 eval_every=100000 \
    measure_mode=first_passage seed="$S" "${EXTRA[@]}" \
    use_wandb=true wandb_entity=amsks wandb_group="nf_$TAG" device=cuda \
    hydra.run.dir=$OUT/$TAG > "$LOG/$TAG.log" 2>&1
  echo "[nf] done  $TAG exit=$? $(date +%H:%M:%S)"
}

nf nf_cs_s42   0 cube_single 42                &
nf nf_cs_s43   1 cube_single 43                &
nf nf_cdbl_s42 2 cube_double 42 "$CDBL_SPEC"   &
nf nf_cdbl_s43 3 cube_double 43 "$CDBL_SPEC"   &
nf nf_scn_s42  4 scene       42 "$SCN_SPEC"    &
nf nf_scn_s43  5 scene       43 "$SCN_SPEC"    &
wait
echo "[nf] ALL DONE $(date +%H:%M:%S)"
