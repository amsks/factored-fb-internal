#!/bin/bash
# HumEnv CPR arms: the motivo parity config (launch_humenv_motivo.sh, all its provenance
# notes apply) with the actor swapped ddpgbc_density -> ddpgbc_density_cpr, which adds the
# hindsight-z-conditioned BC term (impls/actors/ddpgbc_density_cpr.py).
#
#   DRYRUN=1 bash scripts/launch_humenv_cpr.sh     # print, submit nothing
#            bash scripts/launch_humenv_cpr.sh     # 2 arms x 3 seeds = 6 jobs
#
# WHY (2026-08-26). The parity arm copied FB-CPR's "no BC term in the actor" while removing
# the discriminator that justifies it, so its actor trained with no grounding at all and
# every conditioned policy scored ~0 on goal reaching under BOTH prompting routes. These
# arms restore the offline core of that anchoring and split the question:
#   cpr    q_coeff=1  cpr_alpha=1  -- critic-max + hindsight BC (the CPR-style arm)
#   cprbc  q_coeff=0  cpr_alpha=1  -- hindsight BC alone, no critic term (the floor: if even
#          direct per-z distillation of Motivo's own replay stays at 0, the defect is
#          upstream of the actor and no CPR variant will fix it)
# Predictions on record: cprbc lifts goal success off 0.00 (the buffer is FB-CPR's own
# replay, so its trajectories demonstrate reachable z's); cpr >= cprbc unless the critic
# term actively fights the anchor as it did at ExoRL-cheetah scale.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
STEPS="${STEPS:-1000000}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-humenv}"
MANIFEST="logs/humenv-cpr-$(date +%Y%m%d-%H%M%S).txt"

# Verbatim launch_humenv_motivo.sh OV (its HP map + NOT-REPRODUCED notes are the source of
# truth), plus the CPR knobs at the end.
OV='measure_hit_spec="0:214:2.548"'
OV="$OV discount=0.98 tau=0.01 batch_size=1024"
OV="$OV lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-5 lr_actor=1.0e-4"
OV="$OV value_hidden_dims=[1024,1024] measure_density_b_hidden_dims=[256]"
OV="$OV measure_density_dim=256 norm_obs=true"
OV="$OV measure_density_archi=motivo actor.motivo_archi=true measure_density_b_norm=true"
OV="$OV measure_density_z=true measure_density_z_rand_frac=0.8 measure_density_z_eval_pool_m=1024"
OV="$OV actor.hidden_dims=[1024,1024] actor.alpha=0.0 actor.const_std=true"
OV="$OV actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3"
OV="$OV actor.cpr_alpha=1.0"

for ARM in cpr cprbc; do
  QC=$([ "$ARM" = cpr ] && echo 1.0 || echo 0.0)
  SD="$SAVE_ROOT/motivo-$ARM"
  for SEED in $SEEDS; do
    NAME="XHC-$ARM-s$SEED"
    DONE_F="$SD/hum-humanoid_density_fb_ddpgbc_density_cpr_seed_${SEED}/eval.csv"
    if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
      echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
    if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
      echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
    if [ "$DRYRUN" = "1" ]; then
      echo "DRYRUN $NAME q_coeff=$QC steps=$STEPS" | tee -a "$MANIFEST"
      echo "        OVERRIDE='$OV actor.q_coeff=$QC'" | tee -a "$MANIFEST"; continue; fi
    id=$(CRITIC=density_fb ACTOR=ddpgbc_density_cpr DOMAIN=hum-humanoid SEED=$SEED \
         OVERRIDE="$OV actor.q_coeff=$QC" \
         TRAIN_STEPS=$STEPS EVAL_INTERVAL=$STEPS SAVE_INTERVAL=250000 \
         EVAL_EPISODES=2 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
         sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
         | grep -oE '^[0-9]+' | tail -1)
    echo "${id:-SUBMIT-FAILED} $NAME q_coeff=$QC ov='$OV actor.q_coeff=$QC'" | tee -a "$MANIFEST"
  done
done
echo "== manifest $MANIFEST =="
