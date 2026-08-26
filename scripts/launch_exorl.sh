#!/bin/bash
# T3 (ExoRL, reward-specified) and T4 (online Density-FB) of
# docs/specs/2026-08-21-density-fb-agent-branch.md section 7.
#
#   DRYRUN=1 bash scripts/launch_exorl.sh t3 t4      # print, submit nothing
#            bash scripts/launch_exorl.sh t3 t4      # 36 + 9 jobs
#            bash scripts/launch_exorl.sh t3         # T3 only
#
# WHY ONE SBATCH PER CELL and not campaign/gen_impls.py: gen_impls applies one --override set
# to every cell it emits, but `measure_hit_spec` is per-domain AND only density_fb/td_fb read
# it, so a single T3 invocation cannot express this table. It also does not emit
# --eval_on_inferred_cond, without which --infer_cond_episodes computes an index that eval
# then discards (impls/main.py, that flag's own help text) -- i.e. T3's whole zero-shot claim
# would silently not happen. Same submit idiom as scripts/launch_density_table.sh.
#
# PROTOCOL NOTES, all decided rather than defaulted:
#   * hit specs: point_mass_maze 0:2:0.03 (the (x,y) block, MazeMultiGoal's target_size);
#     walker 0:24 and cheetah 0:17 are the FULL observation block, radius calibrated to
#     cube-single's own hit rate. See the header of each env/domains/dmc-*.yaml.
#   * 50000 inference samples: MetaMotivo's `num_inference_samples` for ExoRL/DMC
#     (examples/fb_train_dmc.py). At the 256 the critic yaml defaults to, point_mass_maze's
#     sparser targets land on the constant-reward fallback outright -- only 0.31% of RND rows
#     carry nonzero reach_bottom_right reward. `inferred_cond_cos_neutral` in eval.csv reads
#     1.0 whenever that fallback fired, so the runs report it rather than hiding it.
#   * datasets are 5000 episodes (5M transitions), also MetaMotivo's setting; set in the
#     domain yamls, not here.
#   * NUM_VIDEO=0: every dmc domain yaml sets `render: false`.
#   * T4 runs the GOAL-CONDITIONED arm (measure_density_z off), which is the online
#     configuration that has actually been smoked end-to-end against a real env. Online with
#     the reward index on is NOT launched here -- it needs its own smoke first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"
SAVE_ROOT="${SAVE_ROOT:-$SCRATCH/factored-fb-exorl}"
STEPS="${STEPS:-500000}"
MANIFEST="logs/exorl-$(date +%Y%m%d-%H%M%S).txt"
# MetaMotivo's num_inference_samples for ExoRL/DMC.
INFER_M="${INFER_M:-50000}"

DOMAINS="${DOMAINS:-dmc-point-mass-maze dmc-walker dmc-cheetah}"
# T3 methods. Overridable so an unproven port can be held back from the launch without
# editing the arm: psm and rldp have loss-level parity but had never been run on a dmc domain.
T3_CRITICS="${T3_CRITICS:-density_fb fb psm rldp}"
# ACTOR. Default `ddpgbc` with alpha=0, NOT the `flowbc` the first launch used, because
# flowbc carries bc_coeff=3.0 against a unit-normalised Q term and the data here is the RND
# EXPLORATION buffer -- i.e. it clones an explorer. Neither reference implementation does
# that on ExoRL: metamotivo/fb/agent.py is `actor_loss = -Q.mean()` with no BC, and
# td_jepa's scripts/train/proprio/launch_fb_dmc.py instantiates FBAgent with no bc_coeff
# (its flow_bc/ variants exist, but its OGBench launchers are what use them -- which is
# exactly the split we collapsed by holding one actor fixed across both benchmarks).
# Soft FB likewise uses an unregularised SAC-style Gaussian. Measured consequence of the old
# default: every offline maze cell scored at or BELOW a random policy (7-15 return vs 18.9).
ACTOR="${ACTOR:-ddpgbc}"
ACTOR_OV="${ACTOR_OV:-actor.alpha=0.0}"
# density_fb gets ddpgbc_density, NOT stock ddpgbc, and this is not interchangeable:
# critic/density_fb.yaml sets actor_q_center=true, so q_fn returns the CENTERED normalizer as
# its second value; ddpgbc.py:57 divides by |q|.mean() (the first) and throws that leg away,
# while ddpgbc_density.py:59 uses |qs|.mean(). The bridge measured that difference as
# antmaze 0.920/0.868/0.844 centered vs 0.772/0.580/0.656 uncentered. It was silently
# inactive for the whole first ddpgbc campaign because this defaulted to stock ddpgbc for
# every critic, so the per-critic choice is made HERE rather than left to the caller.
ACTOR_DENSITY="${ACTOR_DENSITY:-ddpgbc_density}"

actor_for () {   # -> the actor this critic must run with
  case "$1" in
    density_fb) echo "$ACTOR_DENSITY" ;;
    *)          echo "$ACTOR" ;;
  esac
}
declare -A HIT=( [dmc-point-mass-maze]="0:2:0.03" \
                 [dmc-walker]="0:24:15.9" \
                 [dmc-cheetah]="0:17:7.5" )

cell_done () {   # cell_done <eval.csv> -> 0 if this cell already reached $STEPS
  # Checks ANY row's step field, not just the last one, and strips the CR that eval.csv's
  # CRLF line endings leave on it. Two reasons this is not the obvious
  # `tail -1 | grep ",$STEPS$"`:
  #   * eval.csv is CRLF, so the last field is "500000\r" and a `$`-anchored grep is at the
  #     mercy of whether the tool strips it.
  #   * CsvLogger APPENDS, so a resumed or relaunched run leaves two runs in one file and the
  #     last row is the NEW run's early progress -- a finished cell then reads as unfinished
  #     and gets resubmitted onto its own directory.
  # Getting this wrong resubmits completed cells; it happened twice on 2026-08-24.
  [ -f "$1" ] || return 1
  awk -F, -v want="$STEPS" '
    NR > 1 { v = $NF; gsub(/[\r[:space:]]/, "", v); if (v == want) { found = 1 } }
    END { exit !found }' "$1"
}

submit () {  # submit <name> <critic> <domain> <seed> <tag> <override> <extra_flags> [actor]
  local name=$1 critic=$2 dom=$3 seed=$4 tag=$5 ov=$6 extra=$7
  local actor="${8:-$(actor_for "$critic")}"
  local sd="$SAVE_ROOT/$tag"
  local done_file="$sd/${dom}_${critic}_${actor}_seed_${seed}/eval.csv"
  if cell_done "$done_file"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return; fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return; fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name" | tee -a "$MANIFEST"
    echo "        critic=$critic actor=$actor domain=$dom seed=$seed steps=$STEPS" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$ov'" | tee -a "$MANIFEST"
    echo "        EXTRA_FLAGS='$extra'" | tee -a "$MANIFEST"
    echo "        SAVE_DIR=$sd" | tee -a "$MANIFEST"
    return
  fi
  local id
  # Never --dependency: SLURM purges completed jobs and dependents then die.
  #
  # The guards above are re-checked INSIDE this loop, and that is not belt-and-braces. The
  # `gh` queue caps submitted jobs per user (QOSMaxSubmitJobPerUserLimit), so a launcher can
  # park here for hours; by the time a slot frees, the cell it is holding may have been run
  # and FINISHED by another launcher, or be queued by one. Retrying the bare sbatch then
  # resubmits a completed cell onto its own SAVE_DIR, and impls.main restarts from scratch
  # and overwrites that run's eval.csv -- i.e. it destroys a finished result. That happened:
  # nine density_fb cells were re-submitted this way and had to be cancelled while PENDING.
  until id=$(CRITIC=$critic ACTOR=$actor DOMAIN=$dom SEED=$seed OVERRIDE="$ov" \
             EXTRA_FLAGS="$extra" TRAIN_STEPS=$STEPS EVAL_INTERVAL=50000 \
             SAVE_INTERVAL=100000 EVAL_EPISODES=50 NUM_VIDEO=0 \
             CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR="$sd" \
             sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "submit refused, waiting 300s ($name)"; sleep 300
    if cell_done "$done_file"; then
      echo "SKIP(done while waiting) $name" | tee -a "$MANIFEST"; return; fi
    if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
      echo "SKIP(queued while waiting) $name" | tee -a "$MANIFEST"; return; fi
  done
  echo "$id $name actor=$actor ov='$ov' extra='$extra'" | tee -a "$MANIFEST"
}

arm_t3 () {   # 4 methods x 3 domains x 3 seeds = 36
  # The four critics with a reward interface. density_fb additionally needs the index turned
  # on (measure_density_z defaults OFF so every published goal-conditioned cell is untouched)
  # and its per-domain hit spec; fb / psm / rldp read neither key.
  for CRITIC in $T3_CRITICS; do
    for DOM in $DOMAINS; do
      local ov="$ACTOR_OV"
      [ "$CRITIC" = "density_fb" ] && ov="$ov measure_density_z=true measure_hit_spec=\"${HIT[$DOM]}\""
      for SEED in $SEEDS; do
        submit "X3-${CRITIC}-${DOM#dmc-}-s$SEED" "$CRITIC" "$DOM" "$SEED" \
               "t3/$DOM/$CRITIC" "$ov" \
               "--infer_cond_episodes $INFER_M --eval_on_inferred_cond"
      done
    done
  done
}

arm_t4 () {   # density_fb online x 3 domains x 3 seeds = 9
  for DOM in $DOMAINS; do
    for SEED in $SEEDS; do
      submit "X4-online-${DOM#dmc-}-s$SEED" density_fb "$DOM" "$SEED" \
             "t4/$DOM" "$ACTOR_OV measure_hit_spec=\"${HIT[$DOM]}\"" "--online"
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: $0 {t3|t4} [...]" >&2; exit 2; }
for a in "$@"; do
  case "$a" in
    t3) echo "== T3: density_fb / fb / psm / rldp x 3 ExoRL domains, zero-shot from rewards =="; arm_t3 ;;
    t4) echo "== T4: online density_fb x 3 ExoRL domains =="; arm_t4 ;;
    *)  echo "unknown arm $a (want t3 or t4)" >&2; exit 2 ;;
  esac
done
echo "== launcher done; manifest $MANIFEST =="
