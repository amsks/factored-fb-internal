#!/bin/bash
# ExoRL + online bring-up smoke for density_fb (spec 2026-08-21 workstreams 5/6/7).
# Login nodes cannot run JAX (Eigen thread pool vs the per-user cap), so everything runs here.
#
#   sbatch -A ASC26008 scripts/exorl_smoke.slurm
#
#SBATCH -J exorl-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$SCRATCH/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$SCRATCH/exorl_data/relabel_cache}"
OUT="$SCRATCH/exorl-smoke"
# point_mass_maze is the ONLY dmc domain with a decided measure_hit_spec
# (env/domains/dmc-point-mass-maze.yaml: (x,y) block, target_size 0.03). walker/cheetah are
# NOT VERIFIED and are deliberately absent from this smoke.
HIT='measure_hit_spec="0:2:0.03"'
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

echo "---- 1. exorl env suite against the REAL buffers (2 tests were skipped without data) ----"
$PY -m pytest tests/impls/test_exorl_env.py -q -p no:cacheprovider > logs/exorl-smoke-tests.log 2>&1 \
  && echo "PASS exorl env suite" || { echo "FAIL -> logs/exorl-smoke-tests.log"; tail -30 logs/exorl-smoke-tests.log; fail=$((fail+1)); }

echo "---- 2. offline density_fb x flowbc, goal-conditioned arm (z OFF) ----"
$PY -m impls.main --critic density_fb --actor flowbc --domain dmc-point-mass-maze --seed 0 \
   --train_steps 300 --log_interval 100 --eval_interval 300 --save_interval 300 \
   --eval_episodes 3 --save_dir "$OUT/off-goal" \
   --override "$HIT" --override exorl_max_episodes=100 \
   > logs/exorl-smoke-off-goal.log 2>&1 \
  && echo "PASS offline z-off" || { echo "FAIL -> logs/exorl-smoke-off-goal.log"; tail -40 logs/exorl-smoke-off-goal.log; fail=$((fail+1)); }

echo "---- 3. offline density_fb, REWARD INDEX arm (z ON, eval through infer_cond) ----"
$PY -m impls.main --critic density_fb --actor flowbc --domain dmc-point-mass-maze --seed 0 \
   --train_steps 300 --log_interval 100 --eval_interval 300 --save_interval 300 \
   --eval_episodes 3 --save_dir "$OUT/off-z" \
   --infer_cond_episodes 256 --eval_on_inferred_cond \
   --override "$HIT" --override exorl_max_episodes=100 --override measure_density_z=true \
   > logs/exorl-smoke-off-z.log 2>&1 \
  && echo "PASS offline z-on" || { echo "FAIL -> logs/exorl-smoke-off-z.log"; tail -40 logs/exorl-smoke-off-z.log; fail=$((fail+1)); }

echo "---- 4. fb critic on the same domain (T3's reference reward-interface method) ----"
$PY -m impls.main --critic fb --actor flowbc --domain dmc-point-mass-maze --seed 0 \
   --train_steps 300 --log_interval 100 --eval_interval 300 --save_interval 300 \
   --eval_episodes 3 --save_dir "$OUT/off-fb" \
   --infer_cond_episodes 256 --eval_on_inferred_cond \
   --override exorl_max_episodes=100 \
   > logs/exorl-smoke-off-fb.log 2>&1 \
  && echo "PASS offline fb" || { echo "FAIL -> logs/exorl-smoke-off-fb.log"; tail -40 logs/exorl-smoke-off-fb.log; fail=$((fail+1)); }

echo "---- 5. ONLINE density_fb against the real env (never run before this) ----"
$PY -m impls.main --critic density_fb --actor flowbc --domain dmc-point-mass-maze --seed 0 \
   --online --online_start_steps 2000 --online_buffer_size 20000 \
   --online_env_steps_per_update 1.0 \
   --train_steps 300 --log_interval 100 --eval_interval 300 --save_interval 300 \
   --eval_episodes 3 --save_dir "$OUT/online" \
   --override "$HIT" --override exorl_max_episodes=20 \
   > logs/exorl-smoke-online.log 2>&1 \
  && echo "PASS online" || { echo "FAIL -> logs/exorl-smoke-online.log"; tail -60 logs/exorl-smoke-online.log; fail=$((fail+1)); }

echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail STAGE(S) FAILED" ) == $(date -u +%FT%TZ)"
exit $fail
