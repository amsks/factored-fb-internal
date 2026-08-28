#!/bin/bash
# Eval-only diagnostics on the arch-shape cheetah density_fb 1M checkpoints (eps 7.5).
# T3 route throughout: --infer_cond_episodes M --eval_on_inferred_cond. No training.
#
# MODE=bigm  (default): seeds 0/1/2, task `run`, M in {10000, 500000}. M=10000 replicates
#            the training-time eval's reward-sample count (launch_exorl_arch.sh); 500000 is
#            50x. M is the row count entering z_from_reward_samples (impls/main.py:344 ->
#            density_fb.infer_cond); measure_density_z_eval_pool_m is a different route.
# MODE=tasks: seed 0 only, M=10000, exorl_task in run / run_backward / flip /
#            flip_backward. The three non-`run` tasks relabel the 5M-transition buffer on
#            first use (cached after).
# MODE=walk: cheetah walk (speed threshold 2, Touati controllable_agent task — the paper's
#            Table 2 reports cheetah walk 827±41 / walk-backward 793±66 on RND, not flip)
#            for seeds 0/1/2, then walk_backward seed 0 last so a timeout costs the least
#            informative cell. Discriminates gait capability from run-z ranking: walk ~800
#            means the policy can hold >=2 m/s when the reward saturates there.
#
# Overrides are the training run's own override list, verbatim from its config.json, so
# the rebuilt agent matches the checkpoint. Submit:
#   sbatch --export=ALL,MODE=bigm  scripts/exorl_cheetah_reeval_diag.slurm
#   sbatch --export=ALL,MODE=tasks scripts/exorl_cheetah_reeval_diag.slurm
#SBATCH -J exorl-cheetah-diag
#SBATCH -A ASC26008
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets"
export EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
BASE="$SCRATCH/factored-fb-exorl-arch-shape/dmc-cheetah/density_fb"
OUT="$SCRATCH/factored-fb-exorl-arch-shape/cheetah_reeval_diag.jsonl"
OVR=(--override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3
     --override actor.alpha=0.0 --override batch_size=1024 --override discount=0.98
     --override lr_actor=1.0e-4 --override 'actor.hidden_dims=[1024]' --override tau=0.01
     --override measure_density_z=true --override 'measure_hit_spec="0:17:7.5"'
     --override measure_density_dim=50 --override 'value_hidden_dims=[1024]'
     --override 'measure_density_b_hidden_dims=[256,256]'
     --override measure_density_archi=motivo --override actor.motivo_archi=true
     --override measure_density_b_norm=true --override lr=1.0e-4
     --override lr_f=1.0e-4 --override lr_b=1.0e-4)
MODE="${MODE:-bigm}"
fail=0
run_cell() {  # seed, M, task, tag
  local SEED="$1" M="$2" T="$3" TAG="$4"
  echo "==== $TAG (seed $SEED, M=$M, task=$T) ===="
  $PY -m scripts.reeval_checkpoint --critic density_fb --actor ddpgbc_density \
     --domain dmc-cheetah --seed "$SEED" \
     --restore_path "$BASE/dmc-cheetah_density_fb_ddpgbc_density_seed_$SEED" \
     --restore_epoch 1000000 --eval_episodes 50 \
     --infer_cond_episodes "$M" --eval_on_inferred_cond \
     "${OVR[@]}" --override "exorl_task=$T" \
     --tag "$TAG" --out_jsonl "$OUT" \
    && echo "PASS $TAG" || { echo "FAIL $TAG"; fail=$((fail+1)); }
}
if [ "$MODE" = bigm ]; then
  for SEED in 0 1 2; do
    for M in 10000 500000; do
      run_cell "$SEED" "$M" run "bigm-s$SEED-M$M"
    done
  done
elif [ "$MODE" = tasks ]; then
  for T in run run_backward flip flip_backward; do
    run_cell 0 10000 "$T" "task-$T"
  done
elif [ "$MODE" = walk ]; then
  for SEED in 0 1 2; do
    run_cell "$SEED" 10000 walk "walk-s$SEED"
  done
  run_cell 0 10000 walk_backward "walkback-s0"
else
  echo "unknown MODE=$MODE"; exit 2
fi
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
