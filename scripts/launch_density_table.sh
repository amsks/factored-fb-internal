#!/bin/bash
# Part B of docs/specs/2026-08-16-density-fb-bridge-port.md: fill the Density-FB column of
# PAPER/decoding-bilinear/table.tex with ONE frozen config.
#
#   DENSITY_ACTOR=ddpgbc_density DENSITY_OV='' DRYRUN=1 bash scripts/launch_density_table.sh mw ms
#   DENSITY_ACTOR=ddpgbc_density DENSITY_OV='' bash scripts/launch_density_table.sh mw ms
#   ... bash scripts/launch_density_table.sh ogb          # the OGBench OOD-goal split, opt-in
#
# THE UNIFORM-CONFIG RULE IS THE POINT (HPO spec; Part B rule 1). One density config wins or
# loses the whole table, so this launcher takes the frozen config ONCE, in two variables, and
# applies it to every cell. There is deliberately no per-cell override hook. What DOES vary
# per cell is protocol, not tuning: the data root, the goal-channel mask dims, the hit spec's
# object slice and success radius, and the step budget (MW 500k / ManiSkill 300k, the table
# convention).
#
# `DENSITY_ACTOR` and `DENSITY_OV` must be set explicitly — including to the empty string —
# so that "which config filled this column" is always an answered question. Every launch
# writes the pair to $SAVE_ROOT/FROZEN_CONFIG.txt; if that file already exists with a
# DIFFERENT pair, the launcher refuses, because mixing configs within the column is the one
# thing Part B rule 4 forbids outright.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

: "${DENSITY_ACTOR:?set DENSITY_ACTOR (e.g. ddpgbc_density or flowbc) — the frozen actor}"
if [ -z "${DENSITY_OV+x}" ]; then
  echo "set DENSITY_OV (may be empty: DENSITY_OV='') — the frozen extra overrides" >&2; exit 2
fi

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"
VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-density-table}"
MANIFEST="logs/density-table-$(date +%Y%m%d-%H%M%S).txt"

# ---- the frozen config, recorded once ------------------------------------------------
mkdir -p "$SAVE_ROOT"
FROZEN="actor=$DENSITY_ACTOR|ov=$DENSITY_OV"
FROZEN_FILE="$SAVE_ROOT/FROZEN_CONFIG.txt"
if [ -f "$FROZEN_FILE" ]; then
  prev=$(head -1 "$FROZEN_FILE")
  if [ "$prev" != "$FROZEN" ]; then
    echo "REFUSING: $SAVE_ROOT already holds cells filled with a DIFFERENT config." >&2
    echo "  on disk: $prev" >&2
    echo "  yours:   $FROZEN" >&2
    echo "Part B rule 4: never mix configs within the Density-FB column. Either use the" >&2
    echo "config on disk, or point SAVE_ROOT at a fresh directory and refill every cell." >&2
    exit 3
  fi
else
  [ "$DRYRUN" = "1" ] || { echo "$FROZEN"; date -u +"frozen %Y-%m-%dT%H:%M:%SZ"; } > "$FROZEN_FILE"
fi
echo "== frozen config: $FROZEN =="

# ---- protocol constants (NOT tuning) --------------------------------------------------
# Metaworld: repaired protocol (deconfound spec E3). p_randomgoal=0's mass must move to
# trajgoal or GCDataset's sum-to-1 assertion fires at startup.
MW_PROTO="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
declare -A ROOTS=( [pr]=$SCRATCH/fb-data/play-resample \
                   [prh]=$SCRATCH/fb-data/play-resample-hard \
                   [pnh1m]=$SCRATCH/fb-data/play-noise-hard-1m \
                   [ms]=$SCRATCH/fb-data/ms-play-resample )
# hit spec = the object slice and the env's own success radius.
declare -A HIT=( [mw-push]="4:7:0.05" [mw-pick-place]="4:7:0.07" \
                 [ms-pick-cube]="29:32:0.025" [ms-push-cube]="28:30:0.10" \
                 [cube-single]="19:22:0.4" )
# ManiSkill goal-channel mask. ms-pick-cube needs SIX dims, not three: its observation carries
# obj_to_goal_pos at 39:42, so the goal is exactly reconstructible as obs[29:32] + obs[39:42]
# and masking [26,27,28] alone removes nothing (deconfound spec §8.1).
declare -A MS_MASK=( [ms-push-cube]="[25,26,27]" [ms-pick-cube]="[26,27,28,39,40,41]" )

submit() {  # submit <name> <domain> <envname> <root> <steps> <venv> <tag> <proto overrides>
  local name=$1 dom=$2 envn=$3 root=$4 steps=$5 venv=$6 tag=$7; shift 7
  local sd="$SAVE_ROOT/$tag"
  local done_file="$sd/${envn}_density_fb_${DENSITY_ACTOR}_seed_${SEED}/eval.csv"
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0' | grep -q ",${steps}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  # tr, not xargs: xargs strips the escaped quotes measure_hit_spec depends on, and an
  # unquoted a:b:eps is read by OmegaConf as sexagesimal (4:7:0.07 -> 14820.07).
  local ov
  ov="$(echo "$* measure_density_hard=true measure_hit_spec=\"${HIT[$dom]}\" $DENSITY_OV" \
        | tr -s ' ' | sed 's/^ //;s/ $//')"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name  steps=$steps venv=$(basename "$venv")" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$ov'" | tee -a "$MANIFEST"
    echo "        SAVE_DIR=$sd  root=$root" | tee -a "$MANIFEST"
    return
  fi
  local id
  # Never --dependency-chain: SLURM purges completed jobs and dependents die.
  until id=$(CRITIC=density_fb ACTOR=$DENSITY_ACTOR DOMAIN=$dom SEED=$SEED OVERRIDE="$ov" \
             TRAIN_STEPS=$steps EVAL_INTERVAL=50000 SAVE_INTERVAL=100000 \
             EVAL_EPISODES=50 NUM_VIDEO=2 FB_DATA_ROOT=$root \
             CODE_DIR=$CODE_DIR VENV=$venv SAVE_DIR="$sd" \
             sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "submit refused, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name ov='$ov'" | tee -a "$MANIFEST"
}

arm_mw() {   # 6 cells x 3 seeds = 18 jobs, 500k
  for REG in pr prh pnh1m; do
    for DOM in mw-push mw-pick-place; do
      for SEED in $SEEDS; do
        submit "DT-$REG-${DOM#mw-}-s$SEED" "$DOM" "$DOM" "${ROOTS[$REG]}" 500000 \
               "$VENV_JAX" "$REG/$DOM" "$MW_PROTO"
      done
    done
  done
}

arm_ms() {   # 2 cells x 3 seeds = 6 jobs, 300k (table convention)
  # Both ManiSkill cells are degenerate controls, filled for completeness: ms-push-cube's
  # goal-blind BC saturates at 0.987 (ceiling-censored) and masked ms-pick-cube floors near
  # 0.02-0.04 for every arm. Do NOT tune on them.
  for DOM in ms-push-cube ms-pick-cube; do
    local d="${MS_MASK[$DOM]}"
    for SEED in $SEEDS; do
      submit "DT-ms-${DOM#ms-}-s$SEED" "$DOM" "$DOM" "${ROOTS[ms]}" 300000 \
             "$VENV_MS" "ms/$DOM" \
             "obs_mask_dims=$d goal_mask_dims=$d value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
    done
  done
}

arm_ogb() {  # the OGBench OOD-goal split: 1 cell x 3 seeds = 3 jobs, 500k. Opt-in, so the
             # default `mw ms` batch is exactly the spec's 24-job envelope.
  # OGBench path: no goal masking and no synth eval (the env draws its own eval goals), so
  # the Metaworld protocol string does not apply. Arm D's half-space filter is the cell.
  for SEED in $SEEDS; do
    submit "DT-ogb-ood-s$SEED" cube-single cube-single-play-v0 "" 500000 \
           "$VENV_JAX" "ogb/cube-single-ood" \
           "goal_filter_dim=19 goal_filter_thresh=0.0 goal_filter_side=below"
  done
}

[ "$#" -gt 0 ] || { echo "usage: $0 {mw|ms|ogb} [...]" >&2; exit 2; }
for a in "$@"; do
  case "$a" in
    mw)  echo "== MW: pr / prh / pnh1m x push / pick-place =="; arm_mw ;;
    ms)  echo "== ManiSkill: push-cube / pick-cube (degenerate controls) =="; arm_ms ;;
    ogb) echo "== OGBench cube-single OOD-goal split =="; arm_ogb ;;
    *)   echo "unknown arm $a (want mw, ms or ogb)" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
echo "== read cells with: bash scripts/density_screen_report.sh 'logs/DT-*.o*' =="
