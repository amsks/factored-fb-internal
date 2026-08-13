#!/bin/bash
# Fills the non-MW missing cells of PAPER/decoding-bilinear/table.tex tab:main for the
# "biGCIQL d512" and "density-hard" columns (2026-08-13). The MW cells (play-resample v3,
# play-noise-hard-1m, play-resample-hard v3) are handled by scripts/launch_measure_tier1.sh
# (arms biql512, density-ext); this script does the rest:
#
#   OGBench, standard eval (no masks, no synth), 500k steps:
#     - cube-single OOD-goal split (Arm D's goal_filter_*): biGCIQL512 + densityhard
#     - cube-double / scene / antmaze-medium: densityhard only (biGCIQL512 already ran,
#       see $SCRATCH/factored-fb-armA/<dom>/biql-d512 — table already has these numbers)
#
#   ManiSkill ms-play-resample, masked+synth eval (deconfound E2 protocol), 300k steps:
#     - ms-pick-cube + ms-push-cube: biGCIQL512 AND densityhard
#
#   bash scripts/launch_table_fill_ogb_ms.sh ogb ms
#   DRYRUN=1 bash scripts/launch_table_fill_ogb_ms.sh ogb ms
#
# Idempotent (SKIP(done)/SKIP(queued)/cap-wait), same convention as launch_measure_tier1.sh /
# launch_deconfound.sh. Never share a SAVE_DIR between two configs: run-dir names encode
# domain+critic+actor+seed but NOT overrides, so d50 vs d512 vs densityhard all need their
# own TAG subdir.
#
# Hit-spec provenance (measure_hit_spec, density_fb critic):
#   cube-single(-ood)  "19:22:0.4"                                    -- env default (density_fb.yaml)
#   cube-double        "19:22:0.4,28:31:0.4"                          -- given (spec-21 campaign)
#   scene              "19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6" -- given
#   antmaze-medium     "0:2:0.5"                                      -- DERIVED here, see below
#   ms-pick-cube       "29:32:0.025"                                  -- DERIVED here, see below
#   ms-push-cube       "28:30:0.1"                                    -- DERIVED here, see below
#
# antmaze-medium derivation: ogbench's AntEnv.get_ob() = concat(qpos, qvel), so obs[0:2] ==
# qpos[:2] == get_xy(). MazeEnv.compute_success() (third_party/ogbench/ogbench/locomaze/
# maze.py) is norm(get_xy() - cur_goal_xy) <= self._goal_tol, and _goal_tol = 0.5 for the
# 'ant' loco type. Verified live (gymnasium.make('antmaze-medium-v0'), 500 random steps
# across 5 tasks): our formula norm(obs[0:2]-goal_xy)<=0.5 vs env's own info['success'] —
# 0 mismatches / 500. npz sanity: obs[:,0:2] range [-1.7,21.8]^2 matches the medium maze's
# ~23-unit footprint; mean fraction of in-episode steps within 0.5 of that episode's own
# final xy is 11.9%±22.9% (non-degenerate — not ~0, not ~1).
#
# ms-pick-cube/ms-push-cube derivation: env/unified.py's _GOAL_LAYOUT gives object/target
# obs slices (ms-pick-cube object=(29,32) target=(26,29); ms-push-cube object=(28,31)
# target=(25,28), but push's success rule is planar-only per ManiSkill3 source). Radii are
# ManiSkill3's own PickCubeEnv.goal_thresh=0.025 (panda cfg) and PushCubeEnv.goal_radius=0.1
# (mani_skill/envs/tasks/tabletop/{pick_cube,push_cube}.py). Validated against the packed
# ms-play-resample npz's recorded `success` column using ONLY the position formula:
#   pick-cube  norm(obs[29:32]-obs[26:29])<=0.025  vs recorded success: 93.6% agreement
#              (residual = ManiSkill's extra is_robot_static gate, not position-observable)
#   push-cube  norm(obs[28:30]-obs[25:27])<=0.1    vs recorded success: 96.8% agreement
#              (planar beats the full-3D block, 96.8% vs 96.0% — matches "planar-only" in
#              PushCubeEnv.evaluate(); residual = the on-table height gate)
# hit_matrix compares the SAME obs slice on both sides (state vs. candidate goal frame), so
# the block is the OBJECT slice, not the target slice — mirrors the MW/cube-single
# convention where the hit-spec block is the achieved-object dims, not the baked-in-goal dims.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"     # cube + Metaworld (deconfound.sh convention)
VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"           # ManiSkill backend
MS_ROOT="${MS_ROOT:-$SCRATCH/fb-data/ms-play-resample}"
MANIFEST="logs/table-fill-ogb-ms-$(date +%Y%m%d-%H%M%S).txt"

submit() {  # submit <name> <envname> <critic> <actor> <sd> <steps>
  local name=$1 envn=$2 cr=$3 ac=$4 sd=$5 steps=$6
  local done_file="$sd/${envn}_${cr}_${ac}_seed_${SEED}/eval.csv"
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0' | grep -q ",${steps}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name critic=$cr actor=$ac steps=$steps OVERRIDE='$OVERRIDE' root=${FB_DATA_ROOT:-} sd=$sd" | tee -a "$MANIFEST"; return; fi
  local id
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit or submit refused, waiting 300s ($name)"; sleep 300; done
  echo "$id $name  critic=$cr actor=$ac sd=$sd" | tee -a "$MANIFEST"
}

declare -A HIT=(
  [cube-single-ood]="19:22:0.4"
  [cube-double]="19:22:0.4,28:31:0.4"
  [scene]="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"
  [antmaze]="0:2:0.5"
  [ms-pick-cube]="29:32:0.025"
  [ms-push-cube]="28:30:0.1"
)
FILT="goal_filter_dim=19 goal_filter_thresh=0.0 goal_filter_side=below"   # Arm D's OOD-goal split

# ── OGBench: standard eval, no masks, no synth, 500k steps ────────────────────────────────
arm_ogb() {
  local steps=500000
  # cube-single-ood: BOTH cells (biGCIQL512 and densityhard) are missing.
  local ENVN
  ENVN=$(grep -oE 'env_name: \S+' env/domains/cube-single.yaml | cut -d' ' -f2)
  local SD="$SCRATCH/factored-fb-tier1/ogb/cube-single-ood"
  for SEED in $SEEDS; do
    OVERRIDE="value_latent_dim=512 $FILT"
    CRITIC=iql_bilinear ACTOR=flowbc DOMAIN=cube-single SEED=$SEED OVERRIDE="$OVERRIDE" \
    TRAIN_STEPS=$steps EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=2 \
    CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD/biGCIQL512" \
    submit "ogb-cube-single-ood-biGCIQL512-s$SEED" "$ENVN" iql_bilinear flowbc "$SD/biGCIQL512" $steps

    OVERRIDE="measure_density_hard=true measure_hit_spec=\"${HIT[cube-single-ood]}\" $FILT"
    CRITIC=density_fb ACTOR=flowbc DOMAIN=cube-single SEED=$SEED OVERRIDE="$OVERRIDE" \
    TRAIN_STEPS=$steps EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=2 \
    CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD/densityhard" \
    submit "ogb-cube-single-ood-densityhard-s$SEED" "$ENVN" density_fb flowbc "$SD/densityhard" $steps
  done

  # cube-double / scene / antmaze: densityhard only (d512 already ran under factored-fb-armA).
  for DOM in cube-double scene antmaze; do
    ENVN=$(grep -oE 'env_name: \S+' env/domains/$DOM.yaml | cut -d' ' -f2)
    SD="$SCRATCH/factored-fb-tier1/ogb/$DOM/densityhard"
    for SEED in $SEEDS; do
      OVERRIDE="measure_density_hard=true measure_hit_spec=\"${HIT[$DOM]}\""
      CRITIC=density_fb ACTOR=flowbc DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
      TRAIN_STEPS=$steps EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=2 \
      CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD" \
      submit "ogb-$DOM-densityhard-s$SEED" "$ENVN" density_fb flowbc "$SD" $steps
    done
  done
}

# ── ManiSkill: E2 protocol (masked+synth), 300k steps — biGCIQL512 AND densityhard ────────
declare -A MS_MASK=( [ms-push-cube]="[25,26,27]" [ms-pick-cube]="[26,27,28,39,40,41]" )
NOGR="value_p_randomgoal=0.0 value_p_trajgoal=0.8"
arm_ms() {
  local steps=300000
  for DOM in ms-push-cube ms-pick-cube; do
    local d="${MS_MASK[$DOM]}"

    local SD="$SCRATCH/factored-fb-tier1/ms/$DOM/biGCIQL512"
    for SEED in $SEEDS; do
      OVERRIDE="$(echo "value_latent_dim=512 obs_mask_dims=$d goal_mask_dims=$d $NOGR eval_goal_mode=synth" | xargs)"
      CRITIC=iql_bilinear ACTOR=flowbc DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
      FB_DATA_ROOT=$MS_ROOT \
      TRAIN_STEPS=$steps EVAL_INTERVAL=25000 SAVE_INTERVAL=50000 EVAL_EPISODES=50 NUM_VIDEO=2 \
      CODE_DIR=$CODE_DIR VENV=$VENV_MS SAVE_DIR="$SD" \
      submit "ms-${DOM#ms-}-biGCIQL512-s$SEED" "$DOM" iql_bilinear flowbc "$SD" $steps
    done

    local SD2="$SCRATCH/factored-fb-tier1/ms/$DOM/densityhard"
    for SEED in $SEEDS; do
      OVERRIDE="$(echo "measure_density_hard=true obs_mask_dims=$d goal_mask_dims=$d $NOGR eval_goal_mode=synth" | xargs) measure_hit_spec=\"${HIT[$DOM]}\""
      CRITIC=density_fb ACTOR=flowbc DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
      FB_DATA_ROOT=$MS_ROOT \
      TRAIN_STEPS=$steps EVAL_INTERVAL=25000 SAVE_INTERVAL=50000 EVAL_EPISODES=50 NUM_VIDEO=2 \
      CODE_DIR=$CODE_DIR VENV=$VENV_MS SAVE_DIR="$SD2" \
      submit "ms-${DOM#ms-}-densityhard-s$SEED" "$DOM" density_fb flowbc "$SD2" $steps
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: $0 {ogb|ms} [...]" >&2; exit 2; }
for arm in "$@"; do
  case "$arm" in
    ogb) echo "== OGBench table-fill (cube-single-ood, cube-double, scene, antmaze) =="; arm_ogb ;;
    ms)  echo "== ManiSkill table-fill (ms-pick-cube, ms-push-cube biGCIQL512) =="; arm_ms ;;
    *) echo "unknown arm $arm (want ogb or ms)" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
