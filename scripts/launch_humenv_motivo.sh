#!/bin/bash
# Density-FB on HumEnv at META MOTIVO PARITY: every hyperparameter FB-CPR's own humanoid
# recipe sets, with the OBJECTIVE (and only the objective) replaced by density_fb.
#
#   DRYRUN=1 bash scripts/launch_humenv_motivo.sh     # print, submit nothing
#            bash scripts/launch_humenv_motivo.sh     # 3 seeds
#
# SOURCE OF THE NUMBERS -- facebookresearch/metamotivo @ main, all verbatim:
#   examples/fbcpr_train_humenv.py  (the paper's S-1 recipe, lines ~426-475)
#   metamotivo/fb/agent.py::TrainConfig  (defaults it does not override)
#   the released checkpoint's own config.json (archi, mirrors the example)
#
# HP MAP -- Motivo knob -> our knob. Anything absent from this list is either identical
# already or listed under NOT HONOURED below.
#   batch_size 1024              -> batch_size 1024                (already the default)
#   discount 0.98                -> discount=0.98                  (our default is 0.99)
#   fb_target_tau 0.01           -> tau=0.01                       (our default is 0.005)
#   lr_f 1e-4 / lr_b 1e-5        -> lr_f / lr_b                    (density_fb rode a single
#       `lr`=3e-4 for BOTH branches until impls/agents/__init__.py::_group_lr; without that
#       change these two keys merge in SILENTLY INERT and the run trains at 3e-4)
#   lr_actor 1e-4                -> lr_actor=1.0e-4                (our default is 3e-4)
#   f: hidden_dim 1024, layers 2 -> value_hidden_dims=[1024,1024]
#   b: hidden_dim 256, layers 1  -> measure_density_b_hidden_dims=[256]
#   actor: 1024 x 2              -> actor.hidden_dims=[1024,1024]
#   z_dim 256                    -> measure_density_dim=256
#   norm_z true                  -> already true: the centered index is sqrt(d)-normalized
#   actor_std 0.2, stddev_clip 0.3 -> actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3
#       (their TruncatedNormal(mu, 0.2) clipped at 0.3 is the bootstrap-smoothing law; our
#       const_std actor takes the mode and adds exactly that noise at the bootstrap)
#   train_goal_ratio 0.2         -> measure_density_z_rand_frac=0.8
#       (fb/agent.py:127 -- train_goal_ratio is the fraction of z's taken from B(goal); the
#       remainder is a random sphere draw, which is what rand_frac counts. 0.2 -> 0.8.)
#   no BC term in FB-CPR's actor -> actor.alpha=0.0
#   weight_decay 0, clip_grad_norm 0, Adam(0.9, 0.999) -> what optax.adam already gives us
#   3M gradient updates          -> STEPS=1000000, a DELIBERATE SHORTFALL (decided 2026-08-25)
#       Their 30M env steps at update_agent_every = 10 x online_parallel_envs and
#       num_agent_updates = online_parallel_envs is one update per 10 env steps, i.e. ~3M
#       updates. We run 1M: double the 500k the d128 arm ran, a third of theirs. Say so
#       wherever a number from here is reported -- it is the one parity item traded for
#       walltime, not an oversight. Raise with STEPS=3000000 to close it.
#
#   f/actor embedding_layers 2   -> measure_density_archi=motivo, actor.motivo_archi=true
#       Their forward map and policy are a Y, not an MLP: embed(obs,cond) and embed(obs,act)
#       each narrow to hidden_dim//2 and concatenate back to hidden_dim, then a trunk.
#       `value_hidden_dims=[1024,1024]` alone does NOT express that -- it is two 1024 layers
#       over one concatenation, about half the depth. impls/actors/_motivo_nets.py and
#       density_fb.py::_MotivoF/_MotivoB carry the transcription.
#   archi.b.norm true            -> measure_density_b_norm=true
#       B's output on the sqrt(d) sphere. CONFLICTS with docs/density-fb-design.md, which
#       records scale normalization as separately harmful on this path. Run per the parity
#       directive; read the result knowing the conflict was declared up front.
#   norm_obs true                -> norm_obs=true
#       New knob (impls/critics/density_fb.yaml). Standardises the observation wherever a
#       network reads one -- F's observation leg, B's query input, the policy net's
#       observation leg -- from running per-dimension statistics EMA'd at Motivo's
#       BatchNorm1d momentum 0.01. THE HIT KERNEL STAYS IN RAW UNITS, so radius 2.548 and
#       humenv's own success ball over next_obs[:214] keep meaning the same thing.
#
# NOT REPRODUCED, and why -- read before calling this "parity":
#   * ortho_coef 100, reg_coeff, q_loss_coef 0.1, fb/actor pessimism penalties,
#     expert_asm_ratio, relabel_ratio, seq_length 8, update_z_every_step 150, the whole
#     discriminator block (lr 1e-5, grad penalty 10, 1024x3). These ARE the FB-CPR objective,
#     so "replace the objective" removes them by definition. Note two consequences:
#     orthonormalisation is measured HARMFUL on our path (cube-single 0.967 -> 0.225), and
#     actor_pessimism_penalty 0.5 is a soft ensemble penalty where our q_fn takes a hard
#     min(q1, q2) -- there is no knob for the soft form.
#   * ONLINE vs OFFLINE. Motivo's numbers come from 30M interactive steps with AMASS expert
#     regularisation on a growing replay. This runs offline on the frozen buffer. Same HPs,
#     different regime; the checkpoint stays an oracle, never a baseline.
#   * discount 0.98 COLLIDES with the "default-discount cells only" table rule (gamma=0.99).
#     A hum- cell from this launch is a 0.98 cell. Either the rule takes an explicit
#     hum-humanoid carve-out or this arm is mechanism-side, not table-side. Decide before
#     anything from here reaches a table.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
ALLOC="${ALLOC:-ASC26008}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
STEPS="${STEPS:-1000000}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-humenv}"
SD="${SAVE_DIR:-$SAVE_ROOT/motivo}"
MANIFEST="logs/humenv-motivo-$(date +%Y%m%d-%H%M%S).txt"

OV='measure_hit_spec="0:214:2.548"'
OV="$OV discount=0.98 tau=0.01 batch_size=1024"
OV="$OV lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-5 lr_actor=1.0e-4"
OV="$OV value_hidden_dims=[1024,1024] measure_density_b_hidden_dims=[256]"
OV="$OV measure_density_dim=256 norm_obs=true"
# The SHAPES, not just the widths (2026-08-25). f/actor are a Y -- two embedding branches
# narrowing to hidden_dim//2 into a trunk -- and archi.b.norm puts B on the sqrt(d) sphere.
# value_hidden_dims=[1024,1024] is read as width 1024 x 2 trunk layers under this, and
# measure_density_b_hidden_dims=[256] as width 256 x 1.
OV="$OV measure_density_archi=motivo actor.motivo_archi=true measure_density_b_norm=true"
OV="$OV measure_density_z=true measure_density_z_rand_frac=0.8 measure_density_z_eval_pool_m=1024"
OV="$OV actor.hidden_dims=[1024,1024] actor.alpha=0.0 actor.const_std=true"
OV="$OV actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3"

for SEED in $SEEDS; do
  NAME="XHM-motivo-s$SEED"
  DONE_F="$SD/hum-humanoid_density_fb_ddpgbc_density_seed_${SEED}/eval.csv"
  if [ -f "$DONE_F" ] && awk -F, -v w="$STEPS" 'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$DONE_F"; then
    echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi
  if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $NAME" | tee -a "$MANIFEST"; continue; fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $NAME steps=$STEPS" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$OV'" | tee -a "$MANIFEST"; continue; fi
  id=$(CRITIC=density_fb ACTOR=ddpgbc_density DOMAIN=hum-humanoid SEED=$SEED OVERRIDE="$OV" \
       TRAIN_STEPS=$STEPS EVAL_INTERVAL=$STEPS SAVE_INTERVAL=250000 \
       EVAL_EPISODES=2 NUM_VIDEO=0 CODE_DIR=$PWD VENV=$VENV SAVE_DIR="$SD" \
       sbatch -A "$ALLOC" -J "$NAME" --parsable scripts/vista_train.slurm 2>/dev/null \
       | grep -oE '^[0-9]+' | tail -1)
  echo "${id:-SUBMIT-FAILED} $NAME ov='$OV'" | tee -a "$MANIFEST"
done
echo "== manifest $MANIFEST =="
