#!/bin/bash
# ExoRL arch-shape configuration extended to 3M gradient steps, 3 domains, fb + density_fb.
#
#   DRYRUN=1 bash scripts/launch_exorl_arch3m.sh     # print, submit nothing
#            bash scripts/launch_exorl_arch3m.sh     # 18 jobs
#
# WHY. The 1M arch-shape arm (scripts/launch_exorl_arch.sh) closed at walker density_fb
# 756.1 +- 45.6 (reference 760 +- 19) and cheetah density_fb 111.7 +- 30.8 (reference
# 247 +- 9); the 2026-08-26 diagnosis found cheetah at a uniform ~0.6-0.7x of the paper's
# FB on ALL four velocity tasks (memory: cheetah-gap-uniform-speed-deficit). Neither curve
# was still climbing at 1M, so 3M is a test of that reading, not a tuned bet. Fresh runs in
# a fresh root -- exorl_report keys cells by (domain, critic, actor), and resuming into the
# 1M root would average two budgets into one number.
#
# Walker/cheetah cells are the arch-shape overrides VERBATIM (config.json-audited set:
# motivo Y-structure, B [256,256] + b_norm, all lr 1e-4, discount 0.98). Maze is new to
# this arm; its two reference points agree on z_dim 100 and our critic default 0.99 already
# equals the paper's maze discount, so no discount override is passed there.
# STATED DEVIATION: Touati Table 1 says a 10M buffer for maze; we keep MetaMotivo's
# load_n_episodes=5000 (5M) from env/domains/dmc-point-mass-maze.yaml -- the same protocol
# every prior arm used, and T3 maze already sits at the ~760 ceiling with it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-exorl-arch-3m}"
STEPS="${STEPS:-3000000}"
MANIFEST="logs/exorl-arch3m-$(date +%Y%m%d-%H%M%S).txt"
declare -A HIT=( [dmc-point-mass-maze]="0:2:0.03" [dmc-walker]="0:24:15.9" [dmc-cheetah]="0:17:7.5" )
declare -A ZDIM=( [dmc-point-mass-maze]="100" [dmc-walker]="100" [dmc-cheetah]="50" )
declare -A DISC=( [dmc-point-mass-maze]="0.99" [dmc-walker]="0.98" [dmc-cheetah]="0.98" )

OPT="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 \
batch_size=1024 lr_actor=1.0e-4"
ARCH_ACTOR="actor.hidden_dims=[1024]"
ARCH_B="measure_density_b_hidden_dims=[256,256]"
ARCH_SHAPE="measure_density_archi=motivo actor.motivo_archi=true"
ARCH_NORM="measure_density_b_norm=true lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-4"

for CRITIC in density_fb fb; do
  case "$CRITIC" in
    density_fb) ACTOR=ddpgbc_density ;;
    *)          ACTOR=ddpgbc ;;
  esac
  for DOM in dmc-point-mass-maze dmc-walker dmc-cheetah; do
    Z=${ZDIM[$DOM]}
    OV="$OPT discount=${DISC[$DOM]} $ARCH_ACTOR"
    case "$CRITIC" in
      fb) OV="$OV f_target_tau=0.01 b_target_tau=0.01 z_dim=$Z L_dim=$Z \
forward.hidden_dim=1024 forward.hidden_layers=1 \
left_encoder.hidden_dim=1024 left_encoder.hidden_layers=1 \
backward.hidden_dim=256 backward.hidden_layers=2" ;;
      density_fb) OV="$OV tau=0.01 measure_density_z=true measure_hit_spec=\"${HIT[$DOM]}\" \
measure_density_dim=$Z value_hidden_dims=[1024] $ARCH_B \
$ARCH_SHAPE $ARCH_NORM" ;;
    esac
    for SEED in $SEEDS; do
      NAME="X3-$CRITIC-${DOM#dmc-}-s$SEED"
      SD="$SAVE_ROOT/$DOM/$CRITIC"
      DONE_F="$SD/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}/eval.csv"
      if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
      if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
        echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN $NAME actor=$ACTOR z=$Z disc=${DISC[$DOM]}" | tee -a "$MANIFEST"
        echo "        OVERRIDE='$(echo $OV)'" | tee -a "$MANIFEST"; continue; fi
      id=$(CRITIC=$CRITIC ACTOR=$ACTOR DOMAIN=$DOM SEED=$SEED OVERRIDE="$OV" \
           EXTRA_FLAGS="--infer_cond_episodes 10000 --eval_on_inferred_cond" \
           TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=500000 \
           EVAL_EPISODES=50 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
           sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
           | grep -oE '^[0-9]+' | tail -1)
      echo "${id:-SUBMIT-FAILED} $NAME actor=$ACTOR z=$Z disc=${DISC[$DOM]} ov='$(echo $OV)'" | tee -a "$MANIFEST"
    done
  done
done
echo "== manifest $MANIFEST =="
