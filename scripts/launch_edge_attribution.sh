#!/bin/bash
# Arms for docs/specs/2026-08-17-gciql-edge-attribution.md.
#
#   DRYRUN=1 bash scripts/launch_edge_attribution.sh ktB ktC ktD
#   bash scripts/launch_edge_attribution.sh ktB ktC ktD
#   bash scripts/launch_edge_attribution.sh ktB-prh          # gated: only if ktB-sarsa >=0.9 stable
#
# Same protocol and SAVE_DIR root as scripts/launch_density_port_screens.sh, so
# scripts/density_screen_report.sh reads these arms alongside the collapse-spec ones.
# 1-seed screens on the DEV cell (pr mw-pick-place) unless the arm says otherwise.
#
# Budget: ktB 1 + ktC 2 + ktD 4 = 7 jobs, + ktB-prh 1 if it unlocks = 8. W4's forensics is a
# separate small job (scripts/reeval_kt1_qc03.slurm).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEED="${SEED:-0}"
DRYRUN="${DRYRUN:-0}"
ROOT_PR="${ROOT_PR:-$SCRATCH/fb-data/play-resample}"
ROOT_PRH="${ROOT_PRH:-$SCRATCH/fb-data/play-resample-hard}"
MANIFEST="logs/edge-attr-$(date +%Y%m%d-%H%M%S).txt"

E3P="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
# Escaped quotes are load-bearing: an unquoted a:b:eps is read by OmegaConf as sexagesimal
# (4:7:0.07 -> 14820.07). `tr`, never `xargs` — xargs strips them.
HIT_PICK='measure_hit_spec="4:7:0.07"'
HIT_PUSH='measure_hit_spec="4:7:0.05"'

submit() {  # submit <tag> <critic> <actor> <domain> <root> <overrides...>
  local tag=$1 critic=$2 actor=$3 dom=$4 root=$5; shift 5
  local name="EA-$tag-s$SEED"
  local sd="$SCRATCH/factored-fb-density-port/$tag"
  local run="$sd/${dom}_${critic}_${actor}_seed_${SEED}"
  if [ -f "$run/eval.csv" ] && tail -1 "$run/eval.csv" | tr -d '\0\r' | grep -q ",500000$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  local ov; ov="$(echo "$E3P $*" | tr -s ' ' | sed 's/^ //;s/ $//')"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name  critic=$critic actor=$actor dom=$dom" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$ov'" | tee -a "$MANIFEST"
    echo "        SAVE_DIR=$sd  root=$root" | tee -a "$MANIFEST"; return; fi
  local id
  until id=$(CRITIC=$critic ACTOR=$actor DOMAIN=$dom SEED=$SEED OVERRIDE="$ov" \
             TRAIN_STEPS=500000 EVAL_INTERVAL=50000 SAVE_INTERVAL=50000 \
             EVAL_EPISODES=50 NUM_VIDEO=2 FB_DATA_ROOT=$root \
             CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR="$sd" \
             sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "submit refused, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name ov='$ov'" | tee -a "$MANIFEST"
}

DENS="measure_density_hard=true"

# ---- KT-B (M1 sufficiency + the 0.99-reachability read) --------------------------------
# q_coeff STAYS 1.0. The point is not to mitigate the collapse again but to remove the actor
# from the critic's fixed point entirely and see where a full-strength ascent then lands: a
# stable high number says M1 was the whole story and the kernel readout is not the ceiling.
ktB() { submit ktB-sarsa density_fb flowbc mw-pick-place "$ROOT_PR" \
                "$DENS $HIT_PICK measure_density_bootstrap=sarsa"; }

# Gated follow-up: precision generalization, only worth a job if ktB-sarsa clears >=0.9 stable.
ktB_prh() { submit ktB-sarsa-prh density_fb flowbc mw-pick-place "$ROOT_PRH" \
                    "$DENS $HIT_PICK measure_density_bootstrap=sarsa"; }

# ---- KT-C (M2 residual, and M4) ---------------------------------------------------------
# atemp2 smooths ONLY what the actor ascends; the CE target keeps measure_density_temp, so a
# gain here is attributable to readout smoothness rather than to a different measure.
# nocenter is mechanism-side: the bridge asserts center=true on this path.
ktC() {
  submit ktC-atemp2   density_fb flowbc mw-pick-place "$ROOT_PR" \
         "$DENS $HIT_PICK actor.q_coeff=0.3 measure_density_actor_temp=2.0"
  submit ktC-nocenter density_fb flowbc mw-pick-place "$ROOT_PR" \
         "$DENS $HIT_PICK actor.q_coeff=0.3 actor_q_center=false"
}

# ---- KT-D (M3: is GCIQL's expectile pessimism what costs it mw-push?) --------------------
# GCIQL, both tasks: pick-place invariant => M3 dead for the edge; push moving => it explains
# GCIQL's push mediocrity. No density knobs and no hit spec — gciql has neither.
ktD() {
  for e in 0.7 0.95; do
    local t="e${e/./}"
    submit "ktD-$t-pick" gciql flowbc mw-pick-place "$ROOT_PR" "expectile=$e"
    submit "ktD-$t-push" gciql flowbc mw-push       "$ROOT_PR" "expectile=$e"
  done
}

# ---- confirm: the two 1-seed findings that change a claim, at 3 seeds -------------------
# Only these two. ktC-atemp2 is the graduating density arm (0.94/0.90, drop 0.04) and the
# spec's gate says graduation -> 3 seeds. ktD-e07-push matters because GCIQL at expectile 0.7
# reaches 0.840 on push, which erases density's 0.820 win — a claim that cannot go in the
# table on one seed. ktC-nocenter and the e095 arms are NOT confirmed: they failed or moved
# nothing, so seeding them buys nothing.
#   for s in 1 2; do SEED=$s bash scripts/launch_edge_attribution.sh confirm; done
confirm() {
  submit ktC-atemp2 density_fb flowbc mw-pick-place "$ROOT_PR" \
         "$DENS $HIT_PICK actor.q_coeff=0.3 measure_density_actor_temp=2.0"
  submit ktD-e07-push gciql flowbc mw-push "$ROOT_PR" "expectile=0.7"
}

[ "$#" -gt 0 ] || { echo "usage: $0 {ktB|ktC|ktD|ktB-prh|confirm} [...]" >&2; exit 2; }
for a in "$@"; do
  case "$a" in
    ktB)     echo "== KT-B: SARSA bootstrap (M1) =="; ktB ;;
    ktB-prh) echo "== KT-B on play-resample-hard (gated) =="; ktB_prh ;;
    ktC)     echo "== KT-C: actor readout temperature (M2) + normalizer (M4) =="; ktC ;;
    ktD)     echo "== KT-D: GCIQL expectile sweep (M3) =="; ktD ;;
    confirm) echo "== confirm ktC-atemp2 + ktD-e07-push at SEED=$SEED =="; confirm ;;
    *) echo "unknown arm $a" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
echo "== read with: bash scripts/density_screen_report.sh 'logs/EA-*.o*' =="
