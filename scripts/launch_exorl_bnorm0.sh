#!/usr/bin/env bash
# Probe D of docs/notes/2026-09-18-density-fb-exorl-impl-audit.md: Density-FB (density_fb_z)
# at EXACTLY the exp/walker_arch recipe (the 681-walker arm: tau 0.005, value tower
# [512,512,512] motivo, B [256,256] motivo, lr 1e-4, actor lr = the ddpgbc yaml default 3e-4
# -- matched to exp/walker_fbref -- discount 0.98, TD3 noise 0.2/0.3, alpha 0, q_normalize
# false, 50k-row reward inference at eval) with ONE change: measure_density_b_norm=false.
# The audit measured B's mean pairwise cosine climbing 0.38 (init) -> 0.87 (250k) -> 0.95 (1M)
# under b_norm=true (FB reference: 0.03); this asks whether the sphere pin is what turns that
# drift into a shrinking actor signal. Walker seeds 0-2 on GPUs 0-2, cheetah seeds 0-1 on
# GPUs 3-4, one tmux window per job in session `exorl_bnorm0`.
#   usage: bash scripts/launch_exorl_bnorm0.sh            # launch the 5 jobs in tmux
#          bash scripts/launch_exorl_bnorm0.sh smoke DIR  # 300-step smoke of walker s0 on GPU 0 into DIR
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
# NO lr_actor override: the ddpgbc yaml default 3e-4 is what exp/walker_arch and exp/walker_fbref ran.
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=false --override lr=1e-4 --override discount=0.98"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
HIT_walker='0:24:15.9'
HIT_cheetah='0:17:7.5'

if [[ "${1:-}" == "smoke" ]]; then
  OUT="${2:?smoke needs an output dir}"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=0 bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc --domain dmc-walker --save_dir $OUT/exp --train_steps 300 --eval_interval 250000 --save_interval 250000 --eval_episodes 1 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond $DFB --override 'measure_hit_spec=\"$HIT_walker\"' $ACT --seed 0 > $OUT/smoke_walker_s0.log 2>&1"
  echo "smoke rc=$?  log: $OUT/smoke_walker_s0.log"
  exit 0
fi

mkdir -p logs/exorl_bnorm0
SESSION=exorl_bnorm0
tmux new-session -d -s "$SESSION" -n init "sleep 1"
launch() {  # launch <domain-short> <seed> <gpu>
  local dom=$1 s=$2 g=$3 hit
  hit=$(eval echo "\$HIT_$dom")
  tmux new-window -t "$SESSION" -n "${dom}_s${s}" \
    "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$g $ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc --domain dmc-$dom --save_dir exp/exorl_dfb_bnorm0 $COMMON $DFB --override 'measure_hit_spec=\"$hit\"' $ACT --seed $s > logs/exorl_bnorm0/${dom}_s${s}.log 2>&1; echo JOB_DONE rc=\$? >> logs/exorl_bnorm0/${dom}_s${s}.log; sleep 86400"
  sleep 20
}
launch walker 0 0
launch walker 1 1
launch walker 2 2
launch cheetah 0 3
launch cheetah 1 4
tmux kill-window -t "$SESSION:init" || true
echo "launched 5 jobs in tmux session $SESSION"
