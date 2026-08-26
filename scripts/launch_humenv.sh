#!/bin/bash
# Density-FB on the HumEnv SMPL humanoid, offline on Meta Motivo's released FB-CPR buffer.
#
#   DRYRUN=1 bash scripts/launch_humenv.sh        # print, submit nothing
#            bash scripts/launch_humenv.sh        # 3 seeds at d=128
#   DIMS="50 128 256" bash scripts/launch_humenv.sh   # the spec 23 §6.5 capacity sweep
#
# CONFIG comes from the arm scripts/humenv_smoke.slurm already validated end-to-end; every
# knob below is that arm, not a fresh guess:
#   measure_hit_spec="0:214:2.548"  the benchmark's OWN pose block (metrics.py slices
#       next_obs[..., :214]) at a radius calibrated to cube-single's hit rate. Sharing their
#       block is what lets this report the PUBLISHED proximity metric.
#   actor ddpgbc_density + alpha=0   FB-CPR's actor is `-Q_disc*reg - Q_fb`, no BC at all.
#       The faithful analogue is FB-CPR minus CPR = pure Q-ascent. ddpgbc_density (not stock
#       ddpgbc) keeps density_fb's CENTERED q normalizer -- stock throws that leg away.
#   measure_density_z=true          the density objective normalises the successor measure,
#       so FB's z = E_rho[r*B] identity does not hold; the centered dual replaces it.
#   eval_pool_m=1024                measured ESS(r) 616/1024 vs 155/256 at this eps; the
#       ExoRL M=256 fallback failure is exactly what a small pool cost us there.
#
# EVAL SCORES NOTHING HERE, by design: HumEnv returns reward 0.0 and emits no success key
# (env/humenv_env.py). `episode_return` will be a constant 0 -- do not read it. The metric is
# humenv.bench.GoalEvaluation, a POST-HOC pass over checkpoints, so what this launch has to
# produce is CHECKPOINTS. Hence one eval at the end and a dense save interval.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DIMS="${DIMS:-128}"          # spec 23 §6.5 pre-registers {50,128,256}; d=50 is UNVALIDATED at this scale
DRYRUN="${DRYRUN:-0}"
STEPS="${STEPS:-500000}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-humenv}"
MANIFEST="logs/humenv-$(date +%Y%m%d-%H%M%S).txt"
HIT='measure_hit_spec="0:214:2.548"'

for D in $DIMS; do
  for SEED in $SEEDS; do
    NAME="XH-density-d$D-s$SEED"
    SD="$SAVE_ROOT/d$D"
    DONE_F="$SD/hum-humanoid_density_fb_ddpgbc_density_seed_${SEED}/eval.csv"
    if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
      echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
    if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
      echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
    OV="$HIT actor.alpha=0.0 measure_density_z=true measure_density_z_eval_pool_m=1024 measure_density_dim=$D"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRYRUN $NAME steps=$STEPS" | tee -a "$MANIFEST"
      echo "        OVERRIDE='$OV'" | tee -a "$MANIFEST"; continue; fi
    id=$(CRITIC=density_fb ACTOR=ddpgbc_density DOMAIN=hum-humanoid SEED=$SEED OVERRIDE="$OV" \
         TRAIN_STEPS=$STEPS EVAL_INTERVAL=$STEPS SAVE_INTERVAL=100000 \
         EVAL_EPISODES=2 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
         sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
         | grep -oE '^[0-9]+' | tail -1)
    echo "${id:-SUBMIT-FAILED} $NAME d=$D ov='$OV'" | tee -a "$MANIFEST"
  done
done
echo "== manifest $MANIFEST =="
