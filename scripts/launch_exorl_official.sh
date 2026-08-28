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
STEPS="${STEPS:-5000000}"          # paper text: 2M representation + 3M FB
PHASE1="${PHASE1:-2000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"   # in-training only; the GATE number is the paper-exact
                                       # final eval (50 eps) run by scripts/exorl_paper_eval.py
NGPU="${NGPU:-8}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$REPO/exp/exorl-official}"
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
ACTOR_OV=( --override "actor.alpha=0.0"
           --override "actor.q_noise_std=0.2"  --override "actor.q_noise_clip=0.3"
           --override "actor.boot_noise_std=0.2" --override "actor.boot_noise_clip=0.3"
           --override "actor.q_normalize=false" )

tmux has-session -t="$SESSION" 2>/dev/null && {
  echo "tmux session '$SESSION' already exists -- attach and finish or kill it first:"
  echo "  tmux attach -t $SESSION ; tmux kill-session -t $SESSION"; exit 1; }

GPU=0
LAUNCHED=0
for CRITIC in $CRITICS; do
  for DOM in $DOMAINS; do
    for SEED in $SEEDS; do
      NAME="${CRITIC}-${DOM#dmc-}-s${SEED}"
      RUN_DIR="$SAVE_ROOT/$DOM/$CRITIC/${DOM}_${CRITIC}_${ACTOR}_seed_${SEED}"
      # done check: eval.csv already carries a row at the final step. CRLF-safe (see
      # scripts/launch_exorl.sh::cell_done for why a bare grep is not).
      if [ -f "$RUN_DIR/eval.csv" ] && awk -F, -v w="$STEPS" \
           'NR>1{v=$NF;gsub(/[\r[:space:]]/,"",v);if(v==w)f=1}END{exit !f}' "$RUN_DIR/eval.csv"; then
        echo "SKIP(done) $NAME" | tee -a "$MANIFEST"; continue; fi

      OV=( --override "discount=${DISC[$DOM]}"
           --override "rldp_phase1_steps=$PHASE1"
           --override "measure_hit_spec=\"${HIT[$DOM]}\""
           "${ACTOR_OV[@]}" )
      CMD="cd $REPO && CUDA_VISIBLE_DEVICES=$((GPU % NGPU)) $PY -m impls.main \
--critic $CRITIC --actor $ACTOR --domain $DOM --seed $SEED \
--train_steps $STEPS --eval_interval $EVAL_INTERVAL --save_interval $SAVE_INTERVAL \
--eval_episodes $EVAL_EPISODES --num_video 0 \
--save_dir $SAVE_ROOT/$DOM/$CRITIC ${OV[*]} 2>&1 | tee $REPO/logs/$NAME.log"

      if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN gpu=$((GPU % NGPU)) $NAME" | tee -a "$MANIFEST"
        echo "    $CMD" | tee -a "$MANIFEST"
      else
        if [ "$LAUNCHED" = "0" ]; then
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
