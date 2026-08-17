#!/bin/bash
# Launcher for docs/specs/2026-08-10-deconfound-size-coverage.md — E1 (cube size ladder),
# E2 (ManiSkill at cube scale), E3 (the MW v3 definitive square).
#
#   bash scripts/launch_deconfound.sh e3          # data already exists; launch first
#   bash scripts/launch_deconfound.sh e1
#   bash scripts/launch_deconfound.sh e2          # only after the collection + gate pass
#   bash scripts/launch_deconfound.sh e3 e1       # several arms in one pass
#   DRYRUN=1 bash scripts/launch_deconfound.sh e1 # print the grid, submit nothing
#
# Idempotent: skips rows whose run dir already holds a final eval.csv and rows whose job name
# is already queued, and waits out the per-user submit cap. Safe to re-run after a partial
# drain.
#
# PRECONDITIONS
#   * scripts/e0_smoke.slurm has PASSED (synth eval, distance logging, truncation knob).
#   * e2 additionally requires scripts/collect_ms_play_resample.slurm to have finished AND its
#     structural gate to line up with cube.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"   # cube + Metaworld
VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"         # ManiSkill backend
MW_ROOT="${MW_ROOT:-$SCRATCH/fb-data/play-resample}"                 # v3, 2000 x 500
MS_ROOT="${MS_ROOT:-$SCRATCH/fb-data/ms-play-resample}"              # E2 collection
MANIFEST="logs/deconfound-$(date +%Y%m%d-%H%M%S).txt"

# p_randomgoal=0 in Arm C's spelling: GCDataset asserts the three value-goal probabilities sum
# to 1, so removing the random-goal mass moves it to trajgoal. The actor's is already 0.
NOGR="value_p_randomgoal=0.0 value_p_trajgoal=0.8"
# The three rows every square compares. BC is the floor (bcflow is a pure BC actor, so its
# critic never reaches the policy); the contrast under test is the last two.
ROWS=("gciql|bcflow||BC" "gciql|flowbc||GCIQL" "iql_bilinear|flowbc|value_latent_dim=50|biGCIQL")

submit() {  # submit <name> <expected env_name> <critic> <actor> <save_dir> <train_steps>
  local name=$1 envn=$2 cr=$3 ac=$4 sd=$5 steps=$6
  local done_file="$sd/${envn}_${cr}_${ac}_seed_${SEED}/eval.csv"
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0' | grep -q ",${steps}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return
  fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return
  fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name  critic=$cr actor=$ac steps=$steps sd=$sd" | tee -a "$MANIFEST"
    echo "         OVERRIDE='$OVERRIDE'  FB_DATA_ROOT='${FB_DATA_ROOT:-}'" | tee -a "$MANIFEST"
    return
  fi
  local id
  # NEVER gate this on --dependency of a short job: SLURM purges completed jobs and the
  # dependent submission dies with "Job dependency problem". Verify outputs, then submit.
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit or submit refused, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name  critic=$cr actor=$ac sd=$sd" | tee -a "$MANIFEST"
}

# ── E1: cube size ladder ────────────────────────────────────────────────────────────────
# The direct test of C1 in the only environment where the bilinear advantage exists. No new
# data: cube-single-play-v0 truncated to whole-trajectory subsets by dataset_num_transitions.
# The existing full-1M campaign rows are the top rung; `full` re-runs ONE sanity seed through
# the new knob to confirm it is a no-op at full size.
arm_e1() {
  local steps=500000
  for SIZE in ${E1_SIZES:-48000 100000 250000 full}; do
    local sizeov="dataset_num_transitions=$SIZE" tagsize=$SIZE seeds="$SEEDS"
    local rows=("${ROWS[@]}")
    if [ "$SIZE" = full ]; then
      # The anchor sanity: ONE run at the full row count, so the knob is exercised at a size
      # where it must be a no-op and the result must reproduce the existing full-1M biGCIQL
      # number. Passing the count (not omitting the knob) is the point — omitting it would
      # test nothing.
      sizeov="dataset_num_transitions=${E1_FULL_ROWS:-1001000}"
      seeds=0; rows=("iql_bilinear|flowbc|value_latent_dim=50|biGCIQL")
    fi
    for ROW in "${rows[@]}"; do
      IFS='|' read -r CR AC OV TAG <<< "$ROW"
      # Per-TAG SAVE_DIR subdir is MANDATORY: the run dir is {env}_{critic}_{actor}_seed and
      # encodes no overrides, so two sizes (or d50 vs default) would silently clobber.
      local SD="$SCRATCH/factored-fb-E1/$tagsize/$TAG"
      for SEED in $seeds; do
        OVERRIDE="$(echo "$OV $sizeov" | xargs)"
        CRITIC=$CR ACTOR=$AC DOMAIN=cube-single SEED=$SEED OVERRIDE="$OVERRIDE" \
        TRAIN_STEPS=$steps EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 \
        EVAL_EPISODES=50 NUM_VIDEO=2 \
        CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD" \
        submit "E1-$tagsize-$TAG-s$SEED" cube-single-play-v0 "$CR" "$AC" "$SD" $steps
      done
    done
  done
}

# ── E3: MW v3 definitive square ─────────────────────────────────────────────────────────
# The 1M dataset nobody has trained on, with C3 (obs mask) and C4 (synth eval goals) both
# repaired. `E3_ROOT=$SCRATCH/fb-data/play-resample-hard` runs the pre-planned headroom
# fallback if the first square comes out flat at a high BC floor.
arm_e3() {
  local steps=300000 root="${E3_ROOT:-$MW_ROOT}"
  local suffix=""; [ "$root" = "$MW_ROOT" ] || suffix="-hard"
  local MASK="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38]"
  for DOM in mw-push mw-pick-place; do
    for ROW in "${ROWS[@]}"; do
      IFS='|' read -r CR AC OV TAG <<< "$ROW"
      local SD="$SCRATCH/factored-fb-E3$suffix/$DOM/$TAG"
      for SEED in $SEEDS; do
        OVERRIDE="$(echo "$OV $MASK $NOGR eval_goal_mode=synth" | xargs)"
        CRITIC=$CR ACTOR=$AC DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
        FB_DATA_ROOT=$root \
        TRAIN_STEPS=$steps EVAL_INTERVAL=25000 SAVE_INTERVAL=50000 \
        EVAL_EPISODES=50 NUM_VIDEO=2 \
        CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD" \
        submit "E3$suffix-${DOM#mw-}-$TAG-s$SEED" "$DOM" "$CR" "$AC" "$SD" $steps
      done
    done
  done
}

# ── E2: ManiSkill at cube scale ─────────────────────────────────────────────────────────
# ms-pick-cube's mask is SIX dims, not three: its observation carries obj_to_goal_pos at
# 39:42, so the goal is exactly reconstructible as obs[29:32] + obs[39:42] and masking
# [26,27,28] alone removes nothing (spec §8.1 — this is what made Arm E's "masked" pick-cube
# rows unmasked runs). The unmasked BC-only control re-confirms the censoring baseline on the
# new data and is the reason to keep both mask spellings visible here.
declare -A MS_MASK=( [ms-push-cube]="[25,26,27]" [ms-pick-cube]="[26,27,28,39,40,41]" )
arm_e2() {
  local steps=300000
  for DOM in ms-push-cube ms-pick-cube; do
    local d="${MS_MASK[$DOM]}"
    for ROW in "${ROWS[@]}"; do
      IFS='|' read -r CR AC OV TAG <<< "$ROW"
      local SD="$SCRATCH/factored-fb-E2/$DOM/$TAG"
      for SEED in $SEEDS; do
        OVERRIDE="$(echo "$OV obs_mask_dims=$d goal_mask_dims=$d $NOGR eval_goal_mode=synth" | xargs)"
        CRITIC=$CR ACTOR=$AC DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
        FB_DATA_ROOT=$MS_ROOT \
        TRAIN_STEPS=$steps EVAL_INTERVAL=25000 SAVE_INTERVAL=50000 \
        EVAL_EPISODES=50 NUM_VIDEO=2 \
        CODE_DIR=$CODE_DIR VENV=$VENV_MS SAVE_DIR="$SD" \
        submit "E2-${DOM#ms-}-$TAG-s$SEED" "$DOM" "$CR" "$AC" "$SD" $steps
      done
    done
    # Unmasked BC-only control: the censoring baseline on the new data. If this matches the
    # masked GCIQL/biGCIQL rows, the goal channel is still doing the work and the square says
    # nothing about critics.
    local SD="$SCRATCH/factored-fb-E2/$DOM/BC-nomask"
    for SEED in $SEEDS; do
      OVERRIDE=""
      CRITIC=gciql ACTOR=bcflow DOMAIN=$DOM SEED=$SEED OVERRIDE="" \
      FB_DATA_ROOT=$MS_ROOT \
      TRAIN_STEPS=$steps EVAL_INTERVAL=25000 SAVE_INTERVAL=50000 \
      EVAL_EPISODES=50 NUM_VIDEO=2 \
      CODE_DIR=$CODE_DIR VENV=$VENV_MS SAVE_DIR="$SD" \
      submit "E2-${DOM#ms-}-BCnomask-s$SEED" "$DOM" gciql bcflow "$SD" $steps
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: $0 {e1|e2|e3} [...]" >&2; exit 2; }
for arm in "$@"; do
  case "$arm" in
    e1) echo "== E1: cube size ladder =="        ; arm_e1 ;;
    e2) echo "== E2: ManiSkill at cube scale ==" ; arm_e2 ;;
    e3) echo "== E3: MW v3 definitive square ==" ; arm_e3 ;;
    *)  echo "unknown arm $arm (want e1, e2 or e3)" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
