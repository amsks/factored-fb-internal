#!/bin/bash
# Does the NON-FLOW actor compose with every T3 critic on ExoRL?
# ddpgbc with alpha=0.0 is our nearest equivalent to the actor MetaMotivo and TD-JEPA
# actually use on ExoRL/DMC: `actor_loss = -Q.mean()`, a plain Gaussian MLP maximising Q
# with NO behaviour-cloning term (metamotivo/fb/agent.py; td_jepa scripts/train/proprio/
# launch_fb_dmc.py instantiates FBAgent with no bc_coeff at all). Our impls/actors/ddpgbc.py
# is `-q.mean()/|q|.mean() + alpha*(-log_prob)`, so alpha=0 leaves pure Q maximisation; the
# |q| normalisation is an adaptive gradient rescale upstream applies only when BC is on, and
# it cannot move the optimum.
#SBATCH -J exorl-nonflow
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets"
export EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
fail=0
for CRITIC in fb psm rldp density_fb; do
  OV="actor.alpha=0.0"
  [ "$CRITIC" = "density_fb" ] && OV="$OV measure_density_z=true measure_hit_spec=\"0:24:15.9\""
  echo "---- $CRITIC x ddpgbc(alpha=0) on dmc-walker ----"
  $PY -m impls.main --critic $CRITIC --actor ddpgbc --domain dmc-walker --seed 0 \
     --train_steps 300 --log_interval 150 --eval_interval 300 --save_interval 300 \
     --eval_episodes 3 --num_video 0 --save_dir "$SCRATCH/exorl-nonflow/$CRITIC" \
     --infer_cond_episodes 50000 --eval_on_inferred_cond \
     $(for kv in $OV; do echo --override "$kv"; done) \
     --override exorl_max_episodes=100 \
     > "logs/exorl-nonflow-$CRITIC.log" 2>&1 \
    && echo "PASS $CRITIC" || { echo "FAIL $CRITIC"; tail -25 "logs/exorl-nonflow-$CRITIC.log"; fail=$((fail+1)); }
done
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
