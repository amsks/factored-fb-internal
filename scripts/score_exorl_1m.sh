#!/usr/bin/env bash
# Score the 1M ExoRL runs: arch-shape Density-FB/FB on walker, arch-shape Density-FB on
# cheetah, and FB at the td_jepa/metamotivo reference configuration on walker.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/exorl_score
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
link () {  # root domain critic
  mkdir -p "exp/$1/$2/$3"
  for s in 0 1 2; do
    src="$PWD/exp/$1/$2_$3_ddpgbc_seed_${s}"
    [ -d "$src" ] && ln -sfn "$src" "exp/$1/$2/$3/$2_$3_ddpgbc_seed_${s}"
  done
}
score () {  # root domain critic gpu out
  link "$1" "$2" "$3"
  CUDA_VISIBLE_DEVICES=$4 /dev/shm/.venv-jax/bin/python scripts/exorl_paper_eval.py \
    --root "exp/$1" --critic "$3" --actor ddpgbc --domains "$2" --seeds 0 1 2 \
    --steps 1000000 --episodes 10 --inference-samples 10000 --task-set tdjepa \
    --out "$5" > "logs/exorl_score/$1_$2_$3.log" 2>&1 || echo "FAIL $1 $2 $3"
}
score walker_arch  dmc-walker  density_fb_z 0 exp/paper_eval_exorl_1m.jsonl &
score walker_arch  dmc-walker  fb           1 exp/paper_eval_exorl_1m.jsonl &
score exorl_arch   dmc-cheetah density_fb_z 2 exp/paper_eval_exorl_1m.jsonl &
score walker_fbref dmc-walker  fb           3 exp/paper_eval_exorl_1m_fbref.jsonl &
wait
echo EXORL_1M_SCORED
