#!/bin/bash
# Goal-eval the 6 humenv CPR checkpoints (launch_humenv_cpr.sh) at 1M, on the SAME
# protocol as every row already in $SCRATCH/factored-fb-humenv/goal_eval*.jsonl:
# 4 buffer goals x 3 episodes = 12 episodes, --seed 0, so the goal picks match the
# Motivo oracle's 0.750 row. Two prompts per checkpoint: sample (the training-consistent
# route, primary) and point (Motivo's B(goal) prompt — tells whether prompting works once
# the actor is anchored).
#   sbatch -A ASC26008 scripts/humenv_cpr_goal_eval.slurm
#SBATCH -J humenv-cpr-goaleval
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
ROOT="$SCRATCH/factored-fb-humenv"
OV=(--override 'measure_hit_spec="0:214:2.548"'
    --override discount=0.98 --override tau=0.01 --override batch_size=1024
    --override lr=1.0e-4 --override lr_f=1.0e-4 --override lr_b=1.0e-5 --override lr_actor=1.0e-4
    --override 'value_hidden_dims=[1024,1024]' --override 'measure_density_b_hidden_dims=[256]'
    --override measure_density_dim=256 --override norm_obs=true
    --override measure_density_archi=motivo --override actor.motivo_archi=true
    --override measure_density_b_norm=true
    --override measure_density_z=true --override measure_density_z_rand_frac=0.8
    --override measure_density_z_eval_pool_m=1024
    --override 'actor.hidden_dims=[1024,1024]' --override actor.alpha=0.0
    --override actor.const_std=true
    --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3
    --override actor.cpr_alpha=1.0)
fail=0
for ARM in cpr cprbc; do
  QC=$([ "$ARM" = cpr ] && echo 1.0 || echo 0.0)
  for SEED in 0 1 2; do
    RP="$ROOT/motivo-$ARM/hum-humanoid_density_fb_ddpgbc_density_cpr_seed_$SEED"
    for PROMPT in sample point; do
      echo "==== $ARM seed $SEED prompt $PROMPT ===="
      $PY scripts/humenv_goal_eval.py --agent jax --critic density_fb \
         --actor ddpgbc_density_cpr --domain hum-humanoid --seed 0 \
         --restore_path "$RP" --restore_epoch 1000000 \
         "${OV[@]}" --override actor.q_coeff=$QC \
         --goal_prompt "$PROMPT" --goals buffer --num_goals 4 --episodes_per_goal 3 \
         --out_jsonl "$ROOT/goal_eval_motivo_${ARM}_${PROMPT}.jsonl" \
        && echo "PASS $ARM-s$SEED-$PROMPT" \
        || { echo "FAIL $ARM-s$SEED-$PROMPT"; fail=$((fail+1)); }
    done
  done
done
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
