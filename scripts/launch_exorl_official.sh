#!/bin/bash
# ExoRL OFFICIAL-PARITY campaign, phase 1 (the replication gate).
# docs/specs/2026-08-28-exorl-official-parity-validation.md §4.
#
# There is no slurm on this box, so this is the local twin of scripts/launch_exorl_*.sh:
# one run per GPU, each in its own tmux window, 8 GPUs, a manifest per launch.
#
#   DRYRUN=1 bash scripts/launch_exorl_official.sh        # print the matrix, start nothing
#            bash scripts/launch_exorl_official.sh        # launch
#   CRITICS="rldp_official" DOMAINS="dmc-walker" SEEDS="0 1" bash scripts/...
#
# Watch:  tmux attach -t exorl_official      (ctrl-b w to pick a window)
# Report: python scripts/exorl_report.py --root <SAVE_ROOT> --steps <STEPS>
#         (pass --steps EXPLICITLY -- it silently defaults to 500k)
#
# WHY THE MUJOCO OVERLAY IS SET HERE. env/exorl_env.py relabels rewards by re-simulating
# MuJoCo physics, which needs a mujoco that this venv's dm_control (1.0.41) can drive.
# /dev/shm/.venv-jax ships mujoco 3.3.0, which is too OLD (dm_control's struct indexer
# looks for `B_colind`), and the overlay probe only searches beside the repo, so it finds
# nothing. The sibling checkout's venv has 3.8.1, which works. Read-only: nothing is
# installed and no file in that tree is touched. Override FB_MUJOCO_SITE_PACKAGES to point
# somewhere else if that checkout moves.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"

PY="${PY:-/dev/shm/.venv-jax/bin/python}"
SESSION="${SESSION:-exorl_official}"
CRITICS="${CRITICS:-rldp_official}"
DOMAINS="${DOMAINS:-dmc-walker dmc-cheetah dmc-point-mass-maze}"
SEEDS="${SEEDS:-0 1 2 3}"
ACTOR="${ACTOR:-ddpgbc}"
STEPS="${STEPS:-5000000}"          # rldp: paper text 2M representation + 3M FB.
                                   # HILP is 1M state steps (spec §3) -- set per critic below.
PHASE1="${PHASE1:-2000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
# 250k, not 1M. A CUDA stream-capture fault has now taken out one cell in each of two
# runs, and with a 1M interval walker-s1 died at 980k with NO checkpoint -- 2.8 h of work
# lost 20k steps short of its first save. Checkpoints are ~150 MB, so a tighter interval
# costs disk and buys back hours.
SAVE_INTERVAL="${SAVE_INTERVAL:-250000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"   # in-training only; the GATE number is the paper-exact
                                       # final eval (50 eps) run by scripts/exorl_paper_eval.py
NGPU="${NGPU:-8}"
# In-training eval MUST infer z. Without these the eval runs on eval_cond0 = {"z": zeros},
# so Q = F.0 = 0 and the policy is goal-blind -- run 1 produced 5M steps of meaningless
# eval rows this way, and no warning fires because the warning lives inside the
# `infer_cond_episodes > 0` block that never runs. The count is a TRANSITION count despite
# the flag's name (it goes straight to train_ds.sample), so this is the paper's 50k.
INFER_EPISODES="${INFER_EPISODES:-50000}"
INFER_FLAGS="--infer_cond_episodes $INFER_EPISODES --eval_on_inferred_cond"
DRYRUN="${DRYRUN:-0}"
# NO DEFAULT that points at a completed campaign. `exp/exorl-official` is run 1's root and
# every cell there has a 5M eval row, so a relaunch that forgets SAVE_ROOT would read those
# as done and SKIP the whole matrix -- silently, with a plausible-looking "SKIP(done)" per
# cell. Requiring it explicitly turns a silent no-op into an error.
if [ -z "${SAVE_ROOT:-}" ]; then
  echo "SAVE_ROOT is required (e.g. SAVE_ROOT=\$PWD/exp/exorl-official-r2)." >&2
  echo "There is deliberately no default: the old one pointed at run 1, whose cells are" >&2
  echo "all complete, so a forgotten SAVE_ROOT would skip every cell as already done." >&2
  exit 2
fi
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$HOME/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$HOME/exorl_data/relabel_cache}"
export FB_MUJOCO_SITE_PACKAGES="${FB_MUJOCO_SITE_PACKAGES:-$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$SAVE_ROOT"
MANIFEST="logs/exorl-official-$(date +%Y%m%d-%H%M%S).txt"

# Per-domain overrides. `measure_hit_spec` is not read by rldp/rldp_official (it has no
# kernel), but the density arms need it and it is harmless here, so the same table serves
# both and cannot drift between them. Radii are the calibrated ones from env/domains/*.
declare -A HIT=( [dmc-walker]="0:24:15.9" [dmc-cheetah]="0:17:7.5" [dmc-point-mass-maze]="0:2:0.03" )
# Pointmass is the one domain the paper runs at 0.99 (goal-reaching, sparse); the two
# locomotion domains use 0.98. Recorded as an ambiguity resolution in the spec's §6 item 2.
declare -A DISC=( [dmc-walker]="0.98" [dmc-cheetah]="0.98" [dmc-point-mass-maze]="0.99" )

# THE ACTOR IS PART OF THE PARITY, and the seam's ddpgbc defaults are not the reference's.
# Official RLDP extracts with a PURE TD3 objective -- `actor_loss = -Q.mean()`
# (baselines/rldp rldp/agent.py::update_td3_actor, reached from update_actor) -- with:
#   * NO behavior-cloning term at all                    => actor.alpha=0.0
#   * the actor-loss action SAMPLED, not the mode:
#     dist.sample(clip=stddev_clip) at actor_std          => q_noise_std=0.2, clip=0.3
#     (model.py:57 actor_std=0.2; agent.py:28 stddev_clip=0.3)
#   * the same sampling on the bootstrap next action
#     (agent.py:312-313)                                  => boot_noise_std=0.2, clip=0.3
#   * NO q-magnitude normalizer on the value term         => actor.q_normalize=false
# The seam ships alpha=0.3, noiseless modes and the normalizer on, because those are right
# for the goal-conditioned manipulation cells this actor was written for. Left at the
# defaults, this campaign would compare a DDPG+BC extractor against a published TD3 one and
# call the difference a replication result.
# Pessimism (0.5, agent.py:27) rides the CRITIC's q_fn, which is what the actor calls, and
# is already set in critic/rldp_official.yaml.
# ARCHITECTURE, not just coefficients. Official's actor is nn_models.Actor: two
# simple_embedding branches (over [obs,z] and over obs alone, each h//2) concatenated to h,
# a ReLU trunk, and `mu = torch.tanh(policy(embedding))`. The seam's DEFAULT ddpgbc actor is
# a single MLP over concat([obs,cvec]) with tanh_squash=False and a HARD CLIP at +-1 -- and
# that clip is an absorbing fixed point: past +-1 the gradient through it is exactly zero.
# Measured consequence in runs 1 and 2: policy saturation 1.000 on both domains (dataset
# 0.26), and an action-blind critic. impls/actors/_motivo_nets.py::MotivoPolicy is already
# a faithful port of official's actor, tanh included; it was simply never switched on.
# hidden_dims is read as (width, n_trunk_layers), so [1024] is official's
# archi.actor.hidden_dim 1024 / hidden_layers 1.
# See docs/notes/2026-08-30-rldp-network-audit.md.
ACTOR_OV=( --override "actor.motivo_archi=true"
           --override "actor.hidden_dims=[1024]"
           --override "actor.alpha=0.0"
           --override "actor.q_noise_std=0.2"  --override "actor.q_noise_clip=0.3"
           --override "actor.boot_noise_std=0.2" --override "actor.boot_noise_clip=0.3"
           --override "actor.q_normalize=false" )

# WINDOW_ONLY=1 adds cells to an EXISTING session instead of refusing. Without it a
# single-cell restart is impossible: the guard below exits 1, and scraping the command out
# of a DRYRUN then yields an empty string that `tmux new-window` happily runs as a bare
# shell -- which looks like a successful restart and is not. That cost a silent no-op
# during run 1's cheetah-s3 recovery. A transient CUDA fault taking out one of eight cells
# is the normal case, not the exception, so recovering one cell has to be a supported path.
# DRYRUN is exempt: it never touches tmux, so refusing to PLAN a matrix because an
# unrelated one is still running just hides the plan. Planning the next wave while the
# current one trains is the normal way this script gets used.
if [ "${WINDOW_ONLY:-0}" != "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
  tmux has-session -t="$SESSION" 2>/dev/null && {
    echo "tmux session '$SESSION' already exists -- attach and finish or kill it first,"
    echo "or set WINDOW_ONLY=1 to add these cells to it (single-cell restart):"
    echo "  tmux attach -t $SESSION ; tmux kill-session -t $SESSION"; exit 1; }
fi

# GPU_START offsets the device counter. Restarting a subset of a running matrix needs it:
# the counter always begins at 0, so relaunching two dead cells would put them on GPUs 0
# and 1 -- on top of two healthy runs -- rather than on the devices the dead cells freed.
GPU=${GPU_START:-0}
LAUNCHED=0
for CRITIC in $CRITICS; do
  for DOM in $DOMAINS; do
    for SEED in $SEEDS; do
      NAME="${CRITIC}-${DOM#dmc-}-s${SEED}"
      RUN_DIR="$SAVE_ROOT/$DOM/$CRITIC/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}"
      # done check: eval.csv already carries a row at the final step. CRLF-safe (see
      # scripts/launch_exorl.sh::cell_done for why a bare grep is not).
      if [ -f "$RUN_DIR/eval.csv" ] && awk -F, -v w="${CSTEPS:-$STEPS}" \
           'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$RUN_DIR/eval.csv"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi

      # RESUME=1: continue an existing cell from its newest checkpoint instead of
      # restarting. start_step becomes restore_epoch+1 and the phase is derived from it
      # (impls/main.py), so a cell resumed past rldp_phase1_steps re-enters phase 2
      # directly and one resumed before it finishes phase 1 first -- no flag needed.
      RES=()
      if [ "${RESUME:-0}" = "1" ]; then
        EP=$(ls "$RUN_DIR"/params_*.pkl 2>/dev/null | sed -E 's/.*params_([0-9]+)\.pkl/\1/' \
             | sort -n | tail -1)
        if [ -n "$EP" ]; then
          RES=( --restore_path "$RUN_DIR" --restore_epoch "$EP" )
          echo "RESUME $NAME from step $EP" | tee -a "$MANIFEST"
        else
          echo "RESUME $NAME: no checkpoint, starting from scratch" | tee -a "$MANIFEST"
        fi
      fi
      # Per-critic budget and overrides. The two methods are not interchangeable: RLDP is
      # 5M two-phase steps, HILP is 1M with no phases and its own per-domain settings from
      # the official README's reproducing commands (hilp_discount 0.96 walker / 0.98
      # cheetah, q_loss False for both).
      case "$CRITIC" in
        hilp*)
          CSTEPS="${HILP_STEPS:-1000000}"
          declare -A HDISC=( [dmc-walker]="0.96" [dmc-cheetah]="0.98" \
                             [dmc-quadruped]="0.98" [dmc-point-mass-maze]="0.98" )
          declare -A HQL=( [dmc-walker]="false" [dmc-cheetah]="false" \
                           [dmc-quadruped]="true" [dmc-point-mass-maze]="true" )
          OV=( --override "discount=${DISC[$DOM]}"
               --override "hilp_discount=${HDISC[$DOM]}"
               --override "hilp_q_loss=${HQL[$DOM]}"
               "${ACTOR_OV[@]}" ) ;;
        *)
          CSTEPS="$STEPS"
          OV=( --override "discount=${DISC[$DOM]}"
               --override "rldp_phase1_steps=$PHASE1"
               --override "measure_hit_spec=\"${HIT[$DOM]}\""
               "${ACTOR_OV[@]}" ) ;;
      esac
      # ENV MUST BE INSIDE $CMD, not exported. A tmux window inherits the environment of
      # the tmux SERVER, which on this box is already running from an older session -- an
      # `export` in this script never reaches it, and the run dies in _mujoco_overlay()
      # with "no usable mujoco/dm_control pairing found". Setting them in the command
      # string means the shell tmux spawns sets them itself.
      # CUDA GRAPHS OFF. `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` has now killed a cell in
      # every wave -- 4 occurrences, ~25% of cells -- always inside the logging line's
      # device->host transfer. Run 2's walker-s1 was restarted with command buffers
      # disabled and ran to 5M without recurrence. Disabling them is an execution-
      # scheduling change; it does not alter numerics.
      ENVV="FB_MUJOCO_SITE_PACKAGES=$FB_MUJOCO_SITE_PACKAGES \
EXORL_DATA_ROOT=$EXORL_DATA_ROOT EXORL_CACHE_ROOT=$EXORL_CACHE_ROOT \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer="
      CMD="cd $REPO && $ENVV CUDA_VISIBLE_DEVICES=$((GPU % NGPU)) $PY -m impls.main \
--critic $CRITIC --actor $ACTOR --domain $DOM --seed $SEED \
--train_steps $CSTEPS --eval_interval $EVAL_INTERVAL --save_interval $SAVE_INTERVAL \
--eval_episodes $EVAL_EPISODES --num_video 0 $INFER_FLAGS \
--save_dir $SAVE_ROOT/$DOM/$CRITIC ${RES[*]} ${OV[*]} 2>&1 | tee -a $REPO/logs/$NAME.log"

      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN gpu=$((GPU % NGPU)) $NAME" | tee -a "$MANIFEST"
        echo "    $CMD" | tee -a "$MANIFEST"
      else
        if [ "$LAUNCHED" = "0" ] && [ "${WINDOW_ONLY:-0}" != "1" ]; then
          tmux new-session -d -s "$SESSION" -n "$NAME" "$CMD; echo '== $NAME EXITED rc='\$?; exec bash"
        else
          tmux new-window -t "$SESSION" -n "$NAME" "$CMD; echo '== $NAME EXITED rc='\$?; exec bash"
        fi
        echo "LAUNCHED gpu=$((GPU % NGPU)) $NAME" | tee -a "$MANIFEST"
      fi
      GPU=$((GPU + 1)); LAUNCHED=$((LAUNCHED + 1))
    done
  done
done

echo "== $LAUNCHED cells; manifest $MANIFEST ==" | tee -a "$MANIFEST"
[ "$LAUNCHED" -gt "$NGPU" ] && echo "NOTE: $LAUNCHED cells on $NGPU GPUs -- windows past the \
first $NGPU share a device. Run in waves (set SEEDS / DOMAINS) if that is not wanted." \
  | tee -a "$MANIFEST"
[ "$DRYRUN" = "1" ] || echo "watch: tmux attach -t $SESSION"
