#!/bin/bash
# ExoRL at TOUATI PARITY: every knob "Does Zero-Shot RL Exist?" (arXiv 2209.14935) sets,
# per environment, verified against BOTH the paper appendix (Table 1, App. G) and the
# released code (facebookresearch/controllable_agent url_benchmark/agent/{fb_ddpg,
# fb_modules}.py). Where prose and code disagree, the CODE wins — it is what produced the
# published numbers:
#   * B is 2 hidden layers x 526 (code backward_hidden_dim=526), NOT the appendix's 3x256.
#   * The aux Q loss coefficient: appendix Listing 1 divides by z_dim; the repo default is
#     a 0.01 coefficient with q_loss=False. We enable it at 1/z_dim (the paper listing).
#
# PER-ENV TABLE (paper):            maze        walker      cheetah
#   z_dim / repr dim                100         50          50
#   discount                        0.99        0.98        0.98
#   buffer (transitions)            10M         5M          5M
#   extra hidden layer (F + pi)     yes         no          no
# Common: batch 1024, Adam lr 1e-4 (all towers), target momentum 0.99 (tau 0.01), policy
# smoothing 0.2/0.3 (bootstrap AND the actor-loss action, eq. 43), z-mix 0.5, ortho 1,
# B on the sqrt(d) sphere, F/pi as two preprocessing branches (1024 -> 512) + 1024-wide
# trunk, min over the two F heads in the TD target, the aux Q target, and the actor
# objective. Steps: paper is 1e6; STEPS=3000000 here by user directive (2026-08-27) — the
# 1M eval row of each run IS the paper-budget cell.
#
# WHAT THE CODE CHANGES ADDED FOR THIS ARM (all default-off, prior cells bit-identical):
#   impls/critics/fb.py     q_loss_coef (eq. 42), actor_pessimism_penalty (0.5 == min),
#                           left_encoder.identity (F reads raw obs — the reference has no
#                           encoder; every earlier fb cell ran obs through a learned
#                           z_dim-wide PhiMap, which parity removes)
#   impls/actors/ddpgbc*.py q_noise_std/clip (actor-loss action noise, eq. 43)
#   fb_pessimism_penalty=0.5 turns the existing mean-of-heads TD reduction into their min.
#
# DENSITY_FB gets the SAME per-env HPs translated (shapes, z dim, lr, discount, buffer,
# noise); its kernel z channel, hard min(q1,q2) (= their min already) and centered actor
# normalizer are the method itself and stay. Stated residuals for fb: our two F towers are
# fully independent ensembles where the reference shares preprocessors+trunk between heads;
# our ddpgbc divides the q term by |q|.mean() where the reference is unnormalized.
#
#   DRYRUN=1 bash scripts/launch_exorl_paperx.sh     # print, submit nothing
#            bash scripts/launch_exorl_paperx.sh     # 18 jobs
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-exorl-paperx}"
STEPS="${STEPS:-3000000}"
MANIFEST="logs/exorl-paperx-$(date +%Y%m%d-%H%M%S).txt"
declare -A HIT=(  [dmc-point-mass-maze]="0:2:0.03" [dmc-walker]="0:24:15.9" [dmc-cheetah]="0:17:7.5" )
declare -A ZDIM=( [dmc-point-mass-maze]="100" [dmc-walker]="50" [dmc-cheetah]="50" )
declare -A DISC=( [dmc-point-mass-maze]="0.99" [dmc-walker]="0.98" [dmc-cheetah]="0.98" )
declare -A EPS=(  [dmc-point-mass-maze]="10000" [dmc-walker]="5000" [dmc-cheetah]="5000" )
declare -A QLC=(  [dmc-point-mass-maze]="0.01" [dmc-walker]="0.02" [dmc-cheetah]="0.02" )   # 1/z_dim
declare -A AH=(   [dmc-point-mass-maze]="[1024,1024,1024]" [dmc-walker]="[1024,1024]" [dmc-cheetah]="[1024,1024]" )
declare -A FT=(   [dmc-point-mass-maze]="3" [dmc-walker]="2" [dmc-cheetah]="2" )            # F trunk layers

OPT="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.q_noise_std=0.2 \
actor.q_noise_clip=0.3 actor.alpha=0.0 actor.const_std=true actor.motivo_archi=true \
batch_size=1024 lr_actor=1.0e-4"

for CRITIC in density_fb fb; do
  case "$CRITIC" in
    density_fb) ACTOR=ddpgbc_density ;;
    *)          ACTOR=ddpgbc ;;
  esac
  for DOM in dmc-point-mass-maze dmc-walker dmc-cheetah; do
    Z=${ZDIM[$DOM]}
    OV="$OPT discount=${DISC[$DOM]} exorl_max_episodes=${EPS[$DOM]} actor.hidden_dims=${AH[$DOM]}"
    case "$CRITIC" in
      fb) OV="$OV f_target_tau=0.01 b_target_tau=0.01 z_dim=$Z L_dim=$Z \
forward.hidden_dim=1024 forward.hidden_layers=${FT[$DOM]} \
left_encoder.identity=true backward.hidden_dim=526 backward.hidden_layers=2 \
q_loss_coef=${QLC[$DOM]} fb_pessimism_penalty=0.5 actor_pessimism_penalty=0.5" ;;
      density_fb) OV="$OV tau=0.01 measure_density_z=true measure_hit_spec=\"${HIT[$DOM]}\" \
measure_density_dim=$Z value_hidden_dims=${AH[$DOM]} \
measure_density_b_hidden_dims=[526,526] measure_density_archi=motivo \
measure_density_b_norm=true lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-4" ;;
    esac
    for SEED in $SEEDS; do
      NAME="XP-$CRITIC-${DOM#dmc-}-s$SEED"
      SD="$SAVE_ROOT/$DOM/$CRITIC"
      DONE_F="$SD/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}/eval.csv"
      if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
      if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
        echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN $NAME actor=$ACTOR z=$Z disc=${DISC[$DOM]} eps=${EPS[$DOM]}" | tee -a "$MANIFEST"
        echo "        OVERRIDE='$(echo $OV)'" | tee -a "$MANIFEST"; continue; fi
      id=$(CRITIC=$CRITIC ACTOR=$ACTOR DOMAIN=$DOM SEED=$SEED OVERRIDE="$OV" \
           EXTRA_FLAGS="--infer_cond_episodes 10000 --eval_on_inferred_cond" \
           TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=500000 \
           EVAL_EPISODES=50 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
           sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
           | grep -oE '^[0-9]+' | tail -1)
      echo "${id:-SUBMIT-FAILED} $NAME actor=$ACTOR z=$Z ov='$(echo $OV)'" | tee -a "$MANIFEST"
    done
  done
done
echo "== manifest $MANIFEST =="
