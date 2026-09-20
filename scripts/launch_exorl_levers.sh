#!/usr/bin/env bash
# Single-lever sweep for Density-FB (density_fb_z x ddpgbc) on ExoRL walker + cheetah, 1 seed, 1M
# steps: which ONE change lifts the 1M recipe (scripts/launch_exorl_1mrecipe.sh; walker 725 /
# cheetah 301 at 5 seeds, seed 0 = 744 / 274) toward the FB reference (777 / 366)? Base = the
# 1M-recipe overrides verbatim (tau 0.005, value [512,512,512] motivo, B [256,256] motivo b_norm,
# lr 1e-4, lr_actor 1e-4, discount 0.98, sphere fraction 0.5, latent_centered actor reward, hard
# immediate term, actor motivo [1024], alpha 0, noise 0.2/0.3, q_normalize false) plus ONE
# override, placed LAST so it wins the OmegaConf dotlist merge (impls/utils/config_utils.py).
# Levers (docs/notes/2026-09-18-density-fb-exorl-impl-audit.md, 2026-09-19 lever audit):
#   qnorm     actor.q_normalize=true                     actor Q term / sg|Q|.mean (ddpgbc.py:97)
#   sphere    measure_density_z_rand_frac=1.0            sphere-only task vectors (density_fb.py _z_mix_masked)
#   tower1024 value_hidden_dims=[1024]                    F tower 1024x1 at tau 0.005
#   tau001    tau=0.001                                   target polyak rate at [512,512,512]
#   headmean  measure_density_actor_head_reduce=mean      actor reads mean of the 2 F heads, not min (q_fn)
#   temp2     measure_density_actor_temp=2.0              flatter actor softmax (W2)
#   disc099   discount=0.99
#   zdim100   measure_density_dim=100                     z / B dim 100
#   usage: bash scripts/launch_exorl_levers.sh             # 16 jobs in tmux session exorl_levers, GPUs 0-7
#          bash scripts/launch_exorl_levers.sh smoke DIR   # every lever's walker cmd at 300 steps on GPU 0
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
# --- the levers: name -> the ONE extra override ---------------------------------------------------
LEVERS=(qnorm sphere tower1024 tau001 headmean temp2 disc099 zdim100)
declare -A LEVER=(
  [qnorm]="--override actor.q_normalize=true"
  [sphere]="--override measure_density_z_rand_frac=1.0"
  [tower1024]="--override 'value_hidden_dims=[1024]'"
  [tau001]="--override tau=0.001"
  [headmean]="--override measure_density_actor_head_reduce=mean"
  [temp2]="--override measure_density_actor_temp=2.0"
  [disc099]="--override discount=0.99"
  [zdim100]="--override measure_density_dim=100"
)
# cmd <lever> <domain> <save_dir> <extra flags>  -> the full training command (lever override LAST)
cmd() {
  local lever=$1 dom=$2 save=$3 extra=$4 domflags
  domflags=$(eval echo "\"\$DOMAIN_$dom\"")
  echo "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $extra --save_dir $save $DFB $ACT $domflags --seed 0 ${LEVER[$lever]}"
}

if [[ "${1:-}" == "smoke" ]]; then
  OUT="${2:?smoke needs an output dir}"
  mkdir -p "$OUT"
  SMOKE="--train_steps 300 --eval_interval 250000 --save_interval 250000 --eval_episodes 1 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
  fail=0
  for lever in "${LEVERS[@]}"; do
    log=$OUT/smoke_${lever}_walker.log
    CUDA_VISIBLE_DEVICES=0 bash -c "$(cmd $lever walker $OUT/exp/$lever "$SMOKE") > $log 2>&1" || true
    if grep -q Traceback "$log"; then echo "SMOKE FAIL $lever  ($log)"; grep -m1 -A3 Traceback "$log" | tail -3; fail=1; else echo "smoke ok   $lever"; fi
  done
  exit $fail
fi

mkdir -p logs/exorl_levers
SESSION=exorl_levers
tmux new-session -d -s "$SESSION" -n init "sleep 1"
i=0
for lever in "${LEVERS[@]}"; do
  for dom in walker cheetah; do
    g=$(( i % 8 ))   # 16 jobs, 2 per GPU
    log=logs/exorl_levers/${lever}_${dom}.log
    tmux new-window -t "$SESSION" -n "${lever}_${dom}" \
      "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$g $(cmd $lever $dom exp/exorl_levers/$lever "$COMMON") > $log 2>&1; echo JOB_DONE rc=\$? >> $log; sleep 86400"
    i=$((i+1)); sleep 20
  done
done
tmux kill-window -t "$SESSION:init" || true
echo "launched $i jobs in tmux session $SESSION"
