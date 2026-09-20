#!/usr/bin/env bash
# Score one cheetah round-4 arm (scripts/launch_exorl_cheetah3.sh) under the paper protocol.
#   usage: bash scripts/score_exorl_cheetah3.sh <arm> "<seeds>" <gpu>     e.g.  tau0025 "0 1 2" 0
# -> exp/paper_eval_exorl_cheetah3_<arm>.jsonl ; compare with exp/paper_eval_exorl_sqt.jsonl (round 3, 431.0)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ARM=$1; SEEDS=$2; GPU=${3:-0}
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; R=exp/exorl_disc_score_cheetah3/$ARM; mkdir -p $R/dmc-cheetah/density_fb_z logs/exorl_score
for s in $SEEDS; do run=dmc-cheetah_density_fb_z_ddpgbc_seed_$s; ln -sfn ../../../../exorl_cheetah3/$ARM/$run $R/dmc-cheetah/density_fb_z/$run; done
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/exorl_paper_eval.py ${EXTRA:-} --root $R --critic density_fb_z --actor ddpgbc --domains dmc-cheetah --seeds $SEEDS --steps 1000000 --episodes 10 --inference-samples 50000 --task-set tdjepa --jobs 2 --out exp/paper_eval_exorl_cheetah3_$ARM.jsonl > logs/exorl_score/cheetah3_$ARM.log 2>&1
echo "SCORE_DONE arm=$ARM rc=$?" >> logs/exorl_score/cheetah3_$ARM.log
