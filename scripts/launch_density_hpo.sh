#!/bin/bash
# Density-FB HPO (docs/specs/2026-08-14-density-fb-hpo.md), stage-gated.
#
#   DRYRUN=1 bash scripts/launch_density_hpo.sh s0     # print, submit nothing (ALWAYS smoke first)
#   bash scripts/launch_density_hpo.sh s0              # S0 fidelity repair, 9 jobs
#
# S0 tests H1 (`lr_actor=3.0e-4`) on the three cells the spec names: antmaze (P-A, the port
# gap vs the bridge's 0.676) and mw-pick-place on both play-resample v3 and
# play-noise-hard-1m (P-B, the worst cell at 0.133 and the near-tie at 0.460). H1 is the one
# axis where our stack is KNOWN to differ from the reference the bridge numbers came from,
# so per spec §2 it is a fidelity repair rather than tuning -- and it is the cheapest thing
# that could close either problem.
#
# Everything here is DEV-set only (spec §2 dev/holdout split): cube-single, antmaze and
# mw-pick-place-v3 are the only cells selection may look at. mw-pick-place-pnh1m is included
# in S0 because H1 is a fidelity repair applied everywhere, not a selected hyperparameter;
# if H1 graduates it goes into the frozen config for every cell including holdout.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs results

STAGE="${1:-}"
ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"
STEPS="${STEPS:-500000}"
MANIFEST="logs/density-hpo-${STAGE:-none}-$(date +%Y%m%d-%H%M%S).txt"

# E3 repaired protocol -- the only protocol under which Metaworld measures anything
# (obs+goal goal-dims masked, no random goals, per-episode synthesized eval goal).
E3P="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
declare -A ROOTS=( [pr]=$SCRATCH/fb-data/play-resample \
                   [pnh1m]=$SCRATCH/fb-data/play-noise-hard-1m \
                   [prh]=$SCRATCH/fb-data/play-resample-hard )
# measure_hit_spec per domain: task geometry, not a tuned knob (spec §S3).
declare -A HIT=( [mw-push]="4:7:0.05" [mw-pick-place]="4:7:0.07" [antmaze]="0:2:0.5" \
                 [cube-single]="19:22:0.4" )
declare -A ENVN=( [mw-push]=mw-push [mw-pick-place]=mw-pick-place \
                  [antmaze]=antmaze-medium-navigate-v0 [cube-single]=cube-single-play-v0 )

submit() {  # submit <name> <envname> <critic> <actor> <savedir>
  local name=$1 envn=$2 cr=$3 ac=$4 sd=$5
  local done_file="$sd/${envn}_${cr}_${ac}_seed_${SEED}/eval.csv"
  # eval.csv can carry null bytes from a duplicated pass; strip before matching.
  if [ -f "$done_file" ] && tail -1 "$done_file" | tr -d '\0' | grep -q ",${STEPS}$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name" | tee -a "$MANIFEST"
    echo "       critic=$cr actor=$ac dom=${DOMAIN} sd=$sd" | tee -a "$MANIFEST"
    echo "       root=${FB_DATA_ROOT:-<none>}" | tee -a "$MANIFEST"
    echo "       OVERRIDE='${OVERRIDE}'" | tee -a "$MANIFEST"
    return; fi
  local id
  # Never --dependency-chain: slurm purges completed jobs and the chain dies. Drip-feed
  # against the 20-job cap instead.
  until id=$(sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "cap hit, waiting 300s ($name)" ; sleep 300; done
  echo "$id $name  critic=$cr actor=$ac sd=$sd" | tee -a "$MANIFEST"
}

# run_cell <cellkey> <domain> <regime|-> <tag> <extra_override...>
run_cell() {
  local cell=$1 dom=$2 reg=$3 tag=$4; shift 4
  local extra="$*"
  local proto=""
  [ "$reg" = "-" ] || proto="$E3P"
  local sd
  if [ "$reg" = "-" ]; then sd="$SCRATCH/factored-fb-tier1/ogb/$dom/$tag"
  else sd="$SCRATCH/factored-fb-tier1/$reg/$dom/$tag"; fi
  for SEED in $SEEDS; do
    # Escaped quotes on measure_hit_spec are load-bearing: OmegaConf yaml-parses dotlist
    # values and YAML 1.1 reads bare a:b:c as SEXAGESIMAL (4:7:0.05 -> 14820.05). Use tr to
    # squeeze spaces -- xargs would strip the quotes and reintroduce the bug.
    OVERRIDE="$(echo "measure_density_hard=true measure_hit_spec=\"${HIT[$dom]}\" $proto $extra" \
                | tr -s ' ' | sed 's/^ //;s/ $//')"
    if [ "$reg" = "-" ]; then unset FB_DATA_ROOT || true
    else export FB_DATA_ROOT="${ROOTS[$reg]}"; fi
    DOMAIN=$dom CRITIC=density_fb ACTOR=flowbc SEED=$SEED OVERRIDE="$OVERRIDE" \
    TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 \
    EVAL_EPISODES=50 NUM_VIDEO=2 \
    CODE_DIR=$CODE_DIR VENV=$VENV_JAX SAVE_DIR="$sd" \
    submit "$cell-s$SEED" "${ENVN[$dom]}" density_fb flowbc "$sd"
  done
}

case "$STAGE" in
  s0)
    echo "== S0: H1 lr_actor=3.0e-4, 3 cells x ${SEEDS// /,} ==" | tee -a "$MANIFEST"
    run_cell "H0-antmaze-lra3e4"      antmaze       -     densityhard-lra3e4 lr_actor=3.0e-4
    run_cell "H0-pickplace-pr-lra3e4" mw-pick-place pr    densityhard-lra3e4 lr_actor=3.0e-4
    run_cell "H0-pickplace-p1m-lra3e4" mw-pick-place pnh1m densityhard-lra3e4 lr_actor=3.0e-4
    ;;
  *)
    echo "usage: [DRYRUN=1] bash $0 s0" >&2; exit 2;;
esac
echo "manifest: $MANIFEST"
