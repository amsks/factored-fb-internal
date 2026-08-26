#!/bin/bash
# ExoRL at the PUBLISHED FB hyperparameters, to test whether density_fb and fb close the gap
# to Touati et al.'s reported numbers (walker walk 760+-19, cheetah run 247+-9, RND buffer).
#
#   DRYRUN=1 bash scripts/launch_exorl_paper.sh      # print, submit nothing
#            bash scripts/launch_exorl_paper.sh      # 12 jobs
#
# CONFIG PROVENANCE -- Touati et al. 2023 Appendix G.5 Table 1, walker/cheetah column:
#   replay buffer 5e6 (= our exorl_max_episodes 5000)   [already matched]
#   representation dim 50                                [already matched]
#   learning rate 1e-4, z mixing ratio 0.5               [already matched]
#   batch size 1024                    <- we ran 256
#   discount 0.98                      <- we ran 0.99 (their 0.99 is MAZE-only)
#   target momentum 0.99 => tau 0.01   <- we ran 0.005
#   policy smoothing stddev 0.2, truncation 0.3  <- we had NONE (const_std, mode())
#   gradient steps 1e6                 <- we ran 5e5
#   reward labels for task inference 1e4  <- we ran 5e4
# The smoothing knob is new code (impls/actors/ddpgbc.py::_smooth); it defaults OFF so every
# earlier cell stays bit-identical.
#
# Separate SAVE ROOT on purpose: these share the `ddpgbc` actor name with the 500k cells in
# $SCRATCH/factored-fb-exorl, and the report keys cells by (domain, critic, actor) -- same
# root would average two different configs together.
#   python scripts/exorl_report.py --root $SCRATCH/factored-fb-exorl-paper
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-exorl-paper}"
STEPS="${STEPS:-1000000}"
MANIFEST="logs/exorl-paper-$(date +%Y%m%d-%H%M%S).txt"
declare -A HIT=( [dmc-walker]="0:24:15.9" [dmc-cheetah]="0:17:7.5" )
PAPER="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 \
batch_size=1024 discount=0.98 lr_actor=1.0e-4"

for CRITIC in density_fb fb; do
  case "$CRITIC" in
    density_fb) ACTOR=ddpgbc_density ;;   # centered q normalizer; see exorl-ran-uncentered-actor
    *)          ACTOR=ddpgbc ;;
  esac
  for DOM in dmc-walker dmc-cheetah; do
    OV="$PAPER"
    case "$CRITIC" in
      fb)         OV="$OV f_target_tau=0.01 b_target_tau=0.01" ;;
      density_fb) OV="$OV tau=0.01 measure_density_z=true measure_hit_spec=\"${HIT[$DOM]}\"" ;;
    esac
    for SEED in $SEEDS; do
      NAME="XP-$CRITIC-${DOM#dmc-}-s$SEED"
      SD="$SAVE_ROOT/$DOM/$CRITIC"
      DONE_F="$SD/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}/eval.csv"
      # CRLF-safe done check (see scripts/launch_exorl.sh::cell_done for why grep is not).
      if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
      if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
        echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN $NAME actor=$ACTOR steps=$STEPS" | tee -a "$MANIFEST"
        echo "        OVERRIDE='$OV'" | tee -a "$MANIFEST"; continue; fi
      id=$(CRITIC=$CRITIC ACTOR=$ACTOR DOMAIN=$DOM SEED=$SEED OVERRIDE="$OV" \
           EXTRA_FLAGS="--infer_cond_episodes 10000 --eval_on_inferred_cond" \
           TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=250000 \
           EVAL_EPISODES=50 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
           sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
           | grep -oE '^[0-9]+' | tail -1)
      echo "${id:-SUBMIT-FAILED} $NAME actor=$ACTOR ov='$OV'" | tee -a "$MANIFEST"
    done
  done
done
echo "== manifest $MANIFEST =="
