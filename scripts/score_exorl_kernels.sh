#!/usr/bin/env bash
# Score one reward-relevant hit-kernel arm (scripts/launch_exorl_kernels.sh) under the paper protocol.
#   usage: bash scripts/score_exorl_kernels.sh <arm> "<seeds>" <gpu>     e.g.  quadruped_vu "0 1 2" 0
# arms: walker_vel walker_hv quadruped_vel quadruped_vu (domain + hit spec derived from the arm name).
# -> exp/paper_eval_exorl_kernels_<arm>.jsonl
# compare: walker vs exp/paper_eval_exorl_sqt.jsonl (round 3, 750.2), quadruped vs exp/paper_eval_exorl_1mrecipe.jsonl (655.9)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ARM=$1; SEEDS=$2; GPU=${3:-0}; DOM=${ARM%%_*}
case $ARM in
  walker_vel)    SPEC="15:24:15.428" ;;
  walker_hv)     SPEC="14:15:0.007,15:24:15.428" ;;
  quadruped_vel) SPEC="44:47:0.110" ;;
  quadruped_vu)  SPEC="44:47:0.110,47:48:0.001" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; R=exp/exorl_disc_score_kernels/$ARM; mkdir -p $R/dmc-$DOM/density_fb_z logs/exorl_score
for s in $SEEDS; do run=dmc-${DOM}_density_fb_z_ddpgbc_seed_$s; ln -sfn ../../../../exorl_kernels/$ARM/$run $R/dmc-$DOM/density_fb_z/$run; done
# The scorer forces the default hit spec unless overridden -> ScopeParamShapeError without this override.
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/exorl_paper_eval.py --override "measure_hit_spec=\"$SPEC\"" ${EXTRA:-} --root $R --critic density_fb_z --actor ddpgbc --domains dmc-$DOM --seeds $SEEDS --steps 1000000 --episodes 10 --inference-samples 50000 --task-set tdjepa --jobs 2 --out exp/paper_eval_exorl_kernels_$ARM.jsonl > logs/exorl_score/kernels_$ARM.log 2>&1
echo "SCORE_DONE arm=$ARM rc=$?" >> logs/exorl_score/kernels_$ARM.log
