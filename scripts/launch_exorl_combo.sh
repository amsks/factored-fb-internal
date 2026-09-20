#!/usr/bin/env bash
# Round 2 after the single-lever sweep (scripts/launch_exorl_levers.sh; logs/exorl_score/levers_summary.log):
# sphere-only task vectors (measure_density_z_rand_frac=1.0) and the actor Q normaliser
# (actor.q_normalize=true) helped BOTH walker and cheetah; tower1024 / tau001 / headmean / disc099
# helped cheetah (+62..+106) but hurt walker. This launches, at the 1M recipe + ONE combined arm:
#   (A) sq = sphere + qnorm, walker AND cheetah, seeds 0-4                          10 jobs
#   (B) cheetah-only one-seed combos on top of sq: sq_tower1024, sq_tau001, sq_headmean,
#       sq_tower1024_tau001, sq_tower1024_headmean                                     5 jobs
#   (C) walker-only sq_tower1024 seed 0 (does the tower's walker loss survive under sq?)  1 job
# Base = the 1M-recipe overrides verbatim; arm overrides LAST so they win the OmegaConf merge.
#   usage: bash scripts/launch_exorl_combo.sh             # 16 jobs in tmux session exorl_combo, GPUs 0-7
#          bash scripts/launch_exorl_combo.sh smoke DIR   # every arm's walker cmd at 300 steps on GPU 0
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
# --- the 1M recipe, copied from scripts/launch_exorl_1mrecipe.sh --------------------------------
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
HIT_walker='0:24:15.9'
HIT_cheetah='0:17:7.5'
DOMAIN_walker="--domain dmc-walker --override discount=0.98 --override 'measure_hit_spec=\"$HIT_walker\"' --override lr_actor=1e-4"
DOMAIN_cheetah="--domain dmc-cheetah --override discount=0.98 --override 'measure_hit_spec=\"$HIT_cheetah\"' --override lr_actor=1e-4"
# --- the arms: name -> the combined overrides ------------------------------------------------------
SQ="--override measure_density_z_rand_frac=1.0 --override actor.q_normalize=true"
ARMS=(sq sq_tower1024 sq_tau001 sq_headmean sq_tower1024_tau001 sq_tower1024_headmean)
declare -A ARM=(
  [sq]="$SQ"
  [sq_tower1024]="$SQ --override 'value_hidden_dims=[1024]'"
  [sq_tau001]="$SQ --override tau=0.001"
  [sq_headmean]="$SQ --override measure_density_actor_head_reduce=mean"
  [sq_tower1024_tau001]="$SQ --override 'value_hidden_dims=[1024]' --override tau=0.001"
  [sq_tower1024_headmean]="$SQ --override 'value_hidden_dims=[1024]' --override measure_density_actor_head_reduce=mean"
)
# JOBS: "<arm> <domain> <seed>"
JOBS=()
for s in 0 1 2 3 4; do JOBS+=("sq walker $s" "sq cheetah $s"); done                       # (A) 10
for a in sq_tower1024 sq_tau001 sq_headmean sq_tower1024_tau001 sq_tower1024_headmean; do
  JOBS+=("$a cheetah 0"); done                                                              # (B) 5
JOBS+=("sq_tower1024 walker 0")                                                             # (C) 1
# cmd <arm> <domain> <seed> <save_dir> <extra flags> -> the full training command (arm overrides LAST)
cmd() {
  local arm=$1 dom=$2 seed=$3 save=$4 extra=$5 domflags
  domflags=$(eval echo "\"\$DOMAIN_$dom\"")
  echo "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $extra --save_dir $save $DFB $ACT $domflags --seed $seed ${ARM[$arm]}"
}

if [[ "${1:-}" == "smoke" ]]; then
  OUT="${2:?smoke needs an output dir}"
  mkdir -p "$OUT"
  SMOKE="--train_steps 300 --eval_interval 250000 --save_interval 250000 --eval_episodes 1 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
  fail=0
  for arm in "${ARMS[@]}"; do
    log=$OUT/smoke_${arm}_walker.log
    CUDA_VISIBLE_DEVICES=0 bash -c "$(cmd $arm walker 0 $OUT/exp/$arm "$SMOKE") > $log 2>&1" || true
    if grep -q Traceback "$log"; then echo "SMOKE FAIL $arm  ($log)"; grep -m1 -A3 Traceback "$log" | tail -3; fail=1; else echo "smoke ok   $arm"; fi
  done
  exit $fail
fi

mkdir -p logs/exorl_combo
SESSION=exorl_combo
tmux new-session -d -s "$SESSION" -n init "sleep 1"
i=0
for J in "${JOBS[@]}"; do
  read -r arm dom s <<< "$J"
  g=$(( i % 8 ))   # 16 jobs, 2 per GPU
  log=logs/exorl_combo/${arm}_${dom}_s${s}.log
  tmux new-window -t "$SESSION" -n "${arm}_${dom}_s${s}" \
    "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$g $(cmd $arm $dom $s exp/exorl_combo/$arm "$COMMON") > $log 2>&1; echo JOB_DONE rc=\$? >> $log; sleep 86400"
  i=$((i+1)); sleep 20
done
tmux kill-window -t "$SESSION:init" || true
echo "launched $i jobs in tmux session $SESSION"
