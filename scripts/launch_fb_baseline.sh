#!/bin/bash
# FB baseline column for PAPER/decoding-bilinear/table.tex: vanilla Forward-Backward
# (impls/critics/fb.py, PR1 port of the bridge) at the STOCK bridge OGBench HPs — i.e.
# critic/fb.yaml defaults, no overrides beyond the per-suite eval protocol. This is a
# baseline, not a tuned arm: the uniform config is the yaml itself, so there is no freeze
# machinery and deliberately no tuning hook.
#
#   DRYRUN=1 bash scripts/launch_fb_baseline.sh mw ms
#   bash scripts/launch_fb_baseline.sh mw ms
#
# Protocol constants (identical to launch_density_table.sh, which is the reference):
# MW repaired protocol; ManiSkill 6-dim pick-cube mask; MW 500k / MS 300k steps.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"
VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-fb-baseline}"
MANIFEST="logs/fb-baseline-$(date +%Y%m%d-%H%M%S).txt"

MW_PROTO="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
declare -A ROOTS=( [pr]=$SCRATCH/fb-data/play-resample \
                   [prh]=$SCRATCH/fb-data/play-resample-hard \
                   [pnh1m]=$SCRATCH/fb-data/play-noise-hard-1m \
                   [ms]=$SCRATCH/fb-data/ms-play-resample )
declare -A MS_MASK=( [ms-push-cube]="[25,26,27]" [ms-pick-cube]="[26,27,28,39,40,41]" )

submit() {  # submit <name> <domain> <root> <steps> <venv> <tag> <proto overrides>
  local name=$1 dom=$2 root=$3 steps=$4 venv=$5 tag=$6; shift 6
  local sd="$SAVE_ROOT/$tag"
  local done_file="$sd/${dom}_fb_flowbc_seed_${SEED}/eval.csv"
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0\r' | grep -q ",${steps}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  local ov="$*"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name  steps=$steps venv=$(basename "$venv")" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$ov'  root=$root" | tee -a "$MANIFEST"
    return
  fi
  local id
  # Never --dependency-chain: SLURM purges completed jobs and dependents die.
  until id=$(CRITIC=fb ACTOR=flowbc DOMAIN=$dom SEED=$SEED OVERRIDE="$ov" \
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
        submit "FB-$REG-${DOM#mw-}-s$SEED" "$DOM" "${ROOTS[$REG]}" 500000 \
               "$VENV_JAX" "$REG/$DOM" "$MW_PROTO"
      done
    done
  done
}

arm_ms() {   # 2 cells x 3 seeds = 6 jobs, 300k (table convention; degenerate controls)
  for DOM in ms-push-cube ms-pick-cube; do
    local d="${MS_MASK[$DOM]}"
    for SEED in $SEEDS; do
      submit "FB-ms-${DOM#ms-}-s$SEED" "$DOM" "${ROOTS[ms]}" 300000 \
             "$VENV_MS" "ms/$DOM" \
             "obs_mask_dims=$d goal_mask_dims=$d value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: $0 {mw|ms} [...]" >&2; exit 2; }
for a in "$@"; do
  case "$a" in
    mw) echo "== MW: pr / prh / pnh1m x push / pick-place =="; arm_mw ;;
    ms) echo "== ManiSkill: push-cube / pick-cube =="; arm_ms ;;
    *)  echo "unknown arm $a (want mw or ms)" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
