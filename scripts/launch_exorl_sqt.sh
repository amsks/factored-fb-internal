#!/usr/bin/env bash
# Round 3 (2026-09-19): the ONE arm that came out of the lever sweep + combos, `sqt` = the 1M recipe
# (scripts/launch_exorl_1mrecipe.sh) + sphere-only task vectors (measure_density_z_rand_frac=1.0)
# + the actor Q normaliser (actor.q_normalize=true) + F tower [1024] + tau 0.001 (combo scores:
# sq 5-seed walker 730 / cheetah 341; sq_tower1024_tau001 cheetah seed 0 = 472), on all FOUR ExoRL
# domains at the batch's per-domain hit spec / discount / actor lr, 5 seeds each. Cheetah seed 0
# already exists (exp/exorl_combo/sq_tower1024_tau001) and is symlinked into exp/exorl_sqt/dmc-cheetah/
# so the scorer sees 5 seeds => 19 jobs, up to 3 per GPU round-robin over GPUs 0-7.
#   usage: bash scripts/launch_exorl_sqt.sh             # 19 jobs in tmux session exorl_sqt
#          bash scripts/launch_exorl_sqt.sh smoke DIR   # one walker + one maze cmd at 300 steps on GPU 0
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
# --- the 1M recipe, copied from scripts/launch_exorl_1mrecipe.sh --------------------------------
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
# per-domain settings exactly as exp/exorl_dfb_1mrecipe ran them
DOMAIN_walker="--domain dmc-walker --override discount=0.98 --override 'measure_hit_spec=\"0:24:15.9\"' --override lr_actor=1e-4"
DOMAIN_cheetah="--domain dmc-cheetah --override discount=0.98 --override 'measure_hit_spec=\"0:17:7.5\"' --override lr_actor=1e-4"
DOMAIN_quadruped="--domain dmc-quadruped --override discount=0.98 --override 'measure_hit_spec=\"0:78:21.087\"' --override lr_actor=1e-4"
DOMAIN_maze="--domain dmc-point-mass-maze --override discount=0.99 --override 'measure_hit_spec=\"0:2:0.03\"' --override lr_actor=1e-6"
# --- the arm (LAST so it wins the OmegaConf dotlist merge) ------------------------------------------
SQT="--override measure_density_z_rand_frac=1.0 --override actor.q_normalize=true --override 'value_hidden_dims=[1024]' --override tau=0.001"
# JOBS: "<domain-short> <seed>"; cheetah seed 0 = exp/exorl_combo/sq_tower1024_tau001 (symlinked below)
JOBS=()
for s in 0 1 2 3 4; do JOBS+=("walker $s" "quadruped $s" "maze $s"); done
for s in 1 2 3 4; do JOBS+=("cheetah $s"); done
# cmd <domain-short> <seed> <save_dir> <extra flags> -> the full training command
cmd() {
  local dom=$1 seed=$2 save=$3 extra=$4 domflags
  domflags=$(eval echo "\"\$DOMAIN_$dom\"")
  echo "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $extra --save_dir $save $DFB $ACT $domflags --seed $seed $SQT"
}

if [[ "${1:-}" == "smoke" ]]; then
  OUT="${2:?smoke needs an output dir}"
  mkdir -p "$OUT"
  SMOKE="--train_steps 300 --eval_interval 250000 --save_interval 250000 --eval_episodes 1 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
  fail=0
  for dom in walker maze; do
    log=$OUT/smoke_${dom}_s0.log
    CUDA_VISIBLE_DEVICES=0 bash -c "$(cmd $dom 0 $OUT/exp/$dom "$SMOKE") > $log 2>&1" || true
    if grep -q Traceback "$log"; then echo "SMOKE FAIL $dom  ($log)"; grep -m1 -A3 Traceback "$log" | tail -3; fail=1; else echo "smoke ok   $dom"; fi
  done
  exit $fail
fi

mkdir -p logs/exorl_sqt exp/exorl_sqt/dmc-cheetah
# cheetah seed 0 is the finished combo run; link it so the scorer's 5-seed layout is complete
ln -sfn ../../exorl_combo/sq_tower1024_tau001/dmc-cheetah_density_fb_z_ddpgbc_seed_0 exp/exorl_sqt/dmc-cheetah/dmc-cheetah_density_fb_z_ddpgbc_seed_0
SESSION=exorl_sqt
tmux new-session -d -s "$SESSION" -n init "sleep 1"
i=0
for J in "${JOBS[@]}"; do
  read -r dom s <<< "$J"
  g=$(( i % 8 ))   # 19 jobs, up to 3 per GPU
  full=$(eval echo "\"\$DOMAIN_$dom\""); full=${full#--domain }; full=${full%% *}   # dmc-<name>
  log=logs/exorl_sqt/${dom}_s${s}.log
  tmux new-window -t "$SESSION" -n "${dom}_s${s}" \
    "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$g $(cmd $dom $s exp/exorl_sqt/$full "$COMMON") > $log 2>&1; echo JOB_DONE rc=\$? >> $log; sleep 86400"
  i=$((i+1)); sleep 20
done
tmux kill-window -t "$SESSION:init" || true
echo "launched $i jobs in tmux session $SESSION"
