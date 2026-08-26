#!/bin/bash
# ExoRL with the reference FB ARCHITECTURE as well as its optimisation hyperparameters.
#
#   DRYRUN=1 bash scripts/launch_exorl_arch.sh     # print, submit nothing
#            bash scripts/launch_exorl_arch.sh     # 12 jobs
#
# WHY THIS ARM EXISTS. scripts/launch_exorl_paper.sh matched Touati et al. Table 1's
# OPTIMISATION knobs (batch 1024, discount 0.98, tau 0.01, policy smoothing 0.2/0.3, 1e6
# steps) and left the network shapes alone -- calling it "paper-exact" was overstated. Our
# FB reproduces neither published number (walker walk 760+-19, cheetah run 247+-9), and on
# cheetah nothing we run beats the RND behaviour policy's own 36.5 return, so the defect is
# in the pipeline rather than in any one method's hyperparameters.
#
# ARCHITECTURE, from metamotivo `examples/fb_train_dmc.py` (the DMC/ExoRL reference):
#     archi.f.hidden_dim     1024   f.hidden_layers     1      <- we ran 512 x 2
#     archi.actor.hidden_dim 1024   actor.hidden_layers 1      <- we ran 512 x 3
#     archi.b.hidden_dim      256   b.hidden_layers     2      <- we ran 512 x 4
#     archi.z_dim  100 for walker/pointmass, 50 otherwise      <- we ran 50 everywhere
#     archi.b.norm true, norm_z true, batch 1024               <- already matched
#     model.norm_obs = FALSE  -- the reference explicitly DISABLES observation
#         normalisation on DMC, so our lack of a normaliser is correct and is NOT the bug
#         (metamotivo's own default is True; the DMC example turns it off).
#
# TWO DEVIATIONS, stated rather than buried:
#   * `left_encoder` has no counterpart in metamotivo's FB, whose forward map eats the
#     observation directly. Ours comes from the td_jepa lineage this critic was ported from
#     (docs/rng_equivalence_findings.md). It is sized with `f` here so the whole F path is
#     1024-wide/1-deep; that is a judgement call, not a match.
#   * z_dim 100 on walker follows metamotivo; Touati Table 1 says 50 with 100 only for maze.
#     The two references disagree, so walker changes TWO variables at once. Cheetah does not
#     -- both references say 50 there -- which makes cheetah the clean architecture test.
# RE-RUN AFTER THE 2026-08-25 SHAPE FIX MUST USE A NEW ROOT. The finished cells under
# $SCRATCH/factored-fb-exorl-arch were produced WITHOUT the b sizing and WITHOUT the Y
# structure; exorl_report keys cells by (domain, critic, actor), so the same root would
# average two different configs into one number. Launch it as:
#     SAVE_ROOT=$SCRATCH/factored-fb-exorl-arch-shape bash scripts/launch_exorl_arch.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-exorl-arch}"
STEPS="${STEPS:-1000000}"
MANIFEST="logs/exorl-arch-$(date +%Y%m%d-%H%M%S).txt"
declare -A HIT=( [dmc-walker]="0:24:15.9" [dmc-cheetah]="0:17:7.5" )
declare -A ZDIM=( [dmc-walker]="100" [dmc-cheetah]="50" )

OPT="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 \
batch_size=1024 discount=0.98 lr_actor=1.0e-4"
ARCH_ACTOR="actor.hidden_dims=[1024]"
# 2026-08-25 FIX: the header above documents b 256 x 2, but no run before this line ever set
# it -- measure_density_b_hidden_dims stayed null, which falls back to value_hidden_dims, so
# B rode [1024] and the arm was one shape short of what it claims. The four cells it produced
# (walker density_fb 123.0+-23.7, cheetah 54.0+-22.4) were measured WITHOUT this.
ARCH_B="measure_density_b_hidden_dims=[256,256]"
# 2026-08-25: the shapes are a Y, not an MLP -- two embedding branches narrowing to
# hidden_dim//2 into a trunk (metamotivo/nn_models.py). Width overrides alone never
# reproduced that. ARCH_SHAPE turns on the real structure for F, B and the policy net.
ARCH_SHAPE="measure_density_archi=motivo actor.motivo_archi=true"
# 2026-08-25, second pass. Two more knobs the header called "already matched" that no run
# ever set, both read straight off examples/fb_train_dmc.py:
#   line 66  archi.b.norm = True   -- B on the sqrt(d) sphere. It is True in BOTH references
#       (the DMC example and the humenv one); only norm_obs differs between them, and the
#       DMC example sets that False (line 58), which is why norm_obs stays off here.
#   lines 80-82  lr_f = lr_b = lr_actor = 1e-4 for walker/cheetah. Ours rode the critic
#       default 3e-4 on BOTH density branches -- lr_actor was the only rate the earlier arms
#       ever changed, because until impls/agents/__init__.py::_group_lr existed there was no
#       way to spell the other two.
ARCH_NORM="measure_density_b_norm=true lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-4"

for CRITIC in density_fb fb; do
  case "$CRITIC" in
    density_fb) ACTOR=ddpgbc_density ;;
    *)          ACTOR=ddpgbc ;;
  esac
  for DOM in dmc-walker dmc-cheetah; do
    Z=${ZDIM[$DOM]}
    OV="$OPT $ARCH_ACTOR"
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
      NAME="XA-$CRITIC-${DOM#dmc-}-s$SEED"
      SD="$SAVE_ROOT/$DOM/$CRITIC"
      DONE_F="$SD/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}/eval.csv"
      if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
      if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
        echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN $NAME actor=$ACTOR z=$Z" | tee -a "$MANIFEST"
        echo "        OVERRIDE='$(echo $OV)'" | tee -a "$MANIFEST"; continue; fi
      id=$(CRITIC=$CRITIC ACTOR=$ACTOR DOMAIN=$DOM SEED=$SEED OVERRIDE="$OV" \
           EXTRA_FLAGS="--infer_cond_episodes 10000 --eval_on_inferred_cond" \
           TRAIN_STEPS=$STEPS EVAL_INTERVAL=100000 SAVE_INTERVAL=250000 \
           EVAL_EPISODES=50 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
           sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
           | grep -oE '^[0-9]+' | tail -1)
      echo "${id:-SUBMIT-FAILED} $NAME actor=$ACTOR z=$Z ov='$(echo $OV)'" | tee -a "$MANIFEST"
    done
  done
done
echo "== manifest $MANIFEST =="
