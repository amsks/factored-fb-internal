#!/bin/bash
# Tier-1 MW measure matrix (docs/specs/2026-08-10-measure-objectives-on-mw-play.md §4)
# — launched per-arm as its P5 cube anchor gate passes.
#
#   bash scripts/launch_measure_tier1.sh baselines density   # after density gate (2026-08-11)
#   bash scripts/launch_measure_tier1.sh tdfb                # after the td_fb p1/p2 gates
#   bash scripts/launch_measure_tier1.sh biql512              # d512 fill (table's biGCIQL d512 col), pr+pnh1m+prh
#   bash scripts/launch_measure_tier1.sh density-ext           # density-hard fill on pnh1m+prh (pr's cells already ran)
#   DRYRUN=1 bash scripts/launch_measure_tier1.sh ...        # print, submit nothing
#
# Protocol = deconfound E3 (the only protocol where MW measures anything): obs+goal mask
# 36:39, p_randomgoal=0 (mass to trajgoal), per-episode synth eval goals, 500k steps
# (spec P6: the 300k historical baselines are NOT comparable cells). The measure arms'
# hit-spec IS the env success rule, raw meters: 4:7:0.05 push / 4:7:0.07 pick-place.
# Density runs the HARD immediate target (spec 21 verdict; our anchors: hard 0.916 vs
# soft 0.892, bridge 0.915/0.929).
#
# biql512/density-ext (2026-08-13, PAPER/decoding-bilinear/table.tex tab:main fill): pnh1m =
# play-noise-hard-1m (10000x100, 1M pack — NOT the old sub-1M "pnh" root, which stays
# quarantined and untouched by these two arms); prh = play-resample-hard v3 (2000x500, 1M).
# Both roots verified 1M via npz shape before wiring in.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"
MANIFEST="logs/measure-tier1-$(date +%Y%m%d-%H%M%S).txt"
STEPS=500000

E3P="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
declare -A ROOTS=( [pr]=$SCRATCH/fb-data/play-resample [pnh]=$SCRATCH/fb-data/play-noise-hard \
                   [pnh1m]=$SCRATCH/fb-data/play-noise-hard-1m [prh]=$SCRATCH/fb-data/play-resample-hard )
declare -A HIT=( [mw-push]="4:7:0.05" [mw-pick-place]="4:7:0.07" )

submit() {  # submit <name> <envname> <critic> <actor> <sd>
  local name=$1 envn=$2 cr=$3 ac=$4 sd=$5
  local done_file="$sd/${envn}_${cr}_${ac}_seed_${SEED}/eval.csv"
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0' | grep -q ",${STEPS}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name critic=$cr actor=$ac OVERRIDE='$OVERRIDE' root=$FB_DATA_ROOT" | tee -a "$MANIFEST"; return; fi
  local id
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit, waiting 300s ($name)"; sleep 300; done
  echo "$id $name  critic=$cr actor=$ac sd=$sd" | tee -a "$MANIFEST"
}

run_rows() {  # run_rows <regimes> <rows...> ; ROW = critic|actor|extra_override|TAG
  local regimes=$1; shift
  for REG in $regimes; do
    for DOM in mw-push mw-pick-place; do
      for ROW in "$@"; do
        IFS='|' read -r CR AC OV TAG <<< "$ROW"
        local_ov="$OV"
        # Escaped quotes are load-bearing: OmegaConf.from_dotlist yaml-parses values, and
        # YAML 1.1 reads bare a:b:c numbers as SEXAGESIMAL — 4:7:0.05 became 14820.05 and
        # crashed all 12 density cells of the first Tier-1 wave at build.
        case "$CR" in density_fb|td_fb) local_ov="$local_ov measure_hit_spec=\"${HIT[$DOM]}\"";; esac
        SD="$SCRATCH/factored-fb-tier1/$REG/$DOM/$TAG"
        for SEED in $SEEDS; do
          # tr, not xargs: xargs strips the escaped quotes measure_hit_spec depends on.
          OVERRIDE="$(echo "$E3P $local_ov" | tr -s ' ' | sed 's/^ //;s/ $//')"
          CRITIC=$CR ACTOR=$AC DOMAIN=$DOM SEED=$SEED OVERRIDE="$OVERRIDE" \
          TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 \
          EVAL_EPISODES=50 NUM_VIDEO=2 FB_DATA_ROOT=${ROOTS[$REG]} \
          CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$SD" \
          submit "T1-$REG-${DOM#mw-}-$TAG-s$SEED" "$DOM" "$CR" "$AC" "$SD"
        done
      done
    done
  done
}

for ARM in "$@"; do
  case "$ARM" in
    baselines) run_rows "pr pnh" "gciql|bcflow||BC" "gciql|flowbc||GCIQL" "iql_bilinear|flowbc|value_latent_dim=50|biGCIQL" ;;
    density)   run_rows "pr pnh" "density_fb|flowbc|measure_density_hard=true|densityhard" ;;
    tdfb)      run_rows "pr pnh" "td_fb|flowbc||tdfb-p1" ;;  # p2 cells need per-run warm_start_path; launch by hand after p1 cells land
    # table.tex tab:main fill (2026-08-13): biGCIQL d512 column needs pr+pnh1m+prh; the
    # density-hard column already has pr cells (via `density` above) and only needs pnh1m+prh.
    biql512)     run_rows "pr pnh1m prh" "iql_bilinear|flowbc|value_latent_dim=512|biGCIQL512" ;;
    density-ext) run_rows "pnh1m prh" "density_fb|flowbc|measure_density_hard=true|densityhard" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
