#!/usr/bin/env bash
# 2026-08-05 --- WAVE T: torch region-FB phase 1 with GROUNDED actor z's
# (train_goal_mode=traj: z = B(own trajectory future) — the zHER knob), 3 envs,
# 2 seeds. The reproducibility fix for the paper's method rows.
#
# Why. The claims audit (PAPER/FB-Manipulation-ICLR/CLAIMS-AUDIT.md) found the
# paper's multi-object method numbers come from the jax goal-conditioned stack
# while the method section + all measure/transfer machinery describe the torch
# z-conditioned agent — which floors on conjunctive envs (cd 0.013 vs jax
# 0.549, 41x). Mechanism hypothesis: with train_goal_mode='perm' the actor's
# z's are permuted-batch goals, reachable w.p. ~ region mass (0.033% on
# cube-double), so the actor never trains where it is evaluated. traj-z makes
# every actor z a future of the SAME trajectory — grounded by construction,
# mirroring the jax actor-goal distribution while keeping the z-conditioned
# FB-style agent the paper describes.
#
# Reads (pre-registered):
#   cube-single: must HOLD (perm gets 0.92/0.97). traj materially below 0.9
#                => traj costs where perm suffices; recipe becomes env-cond.
#   cdbl/scene:  RECOVERY toward jax (0.549/0.484) => the torch agent is the
#                reproducible method artifact; chain phase 2 + direct-z
#                zero-shot evals and update the paper's method rows.
#                Still floored => z-conditioning itself (not the z
#                distribution) is the obstruction in torch; the single-agent
#                story needs fork (B) instead. Compare the concurrent no-flow
#                wave (actor class) before concluding.
#
# NOTE reward_z_ratio (phase-2 transfer knob) currently asserts
# train_goal_mode=='perm' (agents/fb/agent.py:433) — phase 1 unaffected; the
# assert must be revisited before wave U enables rz on traj.
#
# Stacked on GPUs 0-5 alongside the noflow wave (light jobs, ~430MB/25% each;
# the missbase+fbregion double-stack ran fine at ~97% util).
#
# Run:      tmux new -d -s trajz 'scripts/rescuing_fb/launch_traj_region.sh 2>&1 | tee /dev/shm/trajz/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_traj_region.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_trajz}
LOG=/dev/shm/trajz; mkdir -p "$LOG" "$OUT"
DRY_RUN=${DRY_RUN:-0}
STEPS=${STEPS:-500000}
DATA=/dev/shm/factored-fb/datasets

CDBL_SPEC="19:22:0.4,28:31:0.4"
SCN_SPEC="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"

tz() { # tag gpu domain seed [spec]
  local TAG=$1 G=$2 DOM=$3 S=$4 SPEC=${5:-}
  local -a EXTRA=()
  [ -n "$SPEC" ] && EXTRA+=("hit_spec='$SPEC'")
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] GPU$G $TAG domain=$DOM seed=$S ${EXTRA[*]:-<single-ball>}"; return 0; fi
  echo "[tz] start $TAG on GPU$G $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$G $PY train.py agent=fb_measure_flowbc domain="$DOM" \
    data_path=$DATA num_train_steps=$STEPS save_every=100000 eval_every=100000 \
    measure_mode=first_passage train_goal_mode=traj seed="$S" "${EXTRA[@]}" \
    use_wandb=true wandb_entity=amsks wandb_group="tz_$TAG" device=cuda \
    hydra.run.dir=$OUT/$TAG > "$LOG/$TAG.log" 2>&1
  echo "[tz] done  $TAG exit=$? $(date +%H:%M:%S)"
}

tz tz_cs_s42   0 cube_single 42                &
tz tz_cs_s43   1 cube_single 43                &
tz tz_cdbl_s42 2 cube_double 42 "$CDBL_SPEC"   &
tz tz_cdbl_s43 3 cube_double 43 "$CDBL_SPEC"   &
tz tz_scn_s42  4 scene       42 "$SCN_SPEC"    &
tz tz_scn_s43  5 scene       43 "$SCN_SPEC"    &
wait
echo "[tz] ALL DONE $(date +%H:%M:%S)"
