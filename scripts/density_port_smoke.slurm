#!/bin/bash
# Gate for docs/specs/2026-08-16-density-fb-bridge-port.md Part A (A1-A4).
# Login nodes cannot run JAX (Eigen pool vs the ~100 thread/user cap), so the unit tests and
# the DRYRUN-equivalent config checks run here.
#
#   sbatch -A ASC26008 scripts/density_port_smoke.slurm
#
#SBATCH -J density-port-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/play-resample}"
fail=0
echo "== host $(hostname) =="

echo "---- A3/A4 unit tests: tests/impls/test_density_fb.py ----"
$PY -m pytest tests/impls/test_density_fb.py -q > logs/density-port-tests.log 2>&1 \
  && echo "PASS density_fb tests" || { echo "FAIL -> logs/density-port-tests.log"; tail -40 logs/density-port-tests.log; fail=$((fail+1)); }

echo "---- regression: actors/critics untouched by the registry edit ----"
$PY -m pytest tests/impls -q -k "combo or registry or actor_static or pr3_gciql or iql_bilinear" \
  > logs/density-port-reg.log 2>&1 \
  && echo "PASS regression subset" || { echo "FAIL -> logs/density-port-reg.log"; tail -40 logs/density-port-reg.log; fail=$((fail+1)); }

echo "---- 200-step real run: density_fb x ddpgbc_density on the DEV cell ----"
$PY -m impls.main --critic density_fb --actor ddpgbc_density --domain mw-pick-place --seed 0 \
   --train_steps 200 --log_interval 200 --eval_interval 200 --save_interval 200 \
   --eval_episodes 4 --num_video 0 \
   --override 'measure_hit_spec="4:7:0.07"' \
   --override obs_mask_dims=[36,37,38] --override goal_mask_dims=[36,37,38] \
   --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8 \
   --override eval_goal_mode=synth \
   --save_dir "$SCRATCH/factored-fb-density-port-smoke/ddpgbc" \
   > logs/density-port-run-ddpgbc.log 2>&1 \
  && { echo "PASS ddpgbc_density run"; grep -h "^eval" logs/density-port-run-ddpgbc.log | tail -1; } \
  || { echo "FAIL -> logs/density-port-run-ddpgbc.log"; tail -30 logs/density-port-run-ddpgbc.log; fail=$((fail+1)); }

echo "---- 200-step real run: flow path + A4 instrumentation ----"
$PY -m impls.main --critic density_fb --actor flowbc --domain mw-pick-place --seed 0 \
   --train_steps 200 --log_interval 200 --eval_interval 200 --save_interval 200 \
   --eval_episodes 4 --num_video 0 \
   --override 'measure_hit_spec="4:7:0.07"' --override lr_actor=3.0e-4 \
   --override obs_mask_dims=[36,37,38] --override goal_mask_dims=[36,37,38] \
   --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8 \
   --override eval_goal_mode=synth \
   --save_dir "$SCRATCH/factored-fb-density-port-smoke/flow" \
   > logs/density-port-run-flow.log 2>&1 \
  && { echo "PASS flowbc run"; grep -o "action_gap_l2[^,]*" logs/density-port-run-flow.log | tail -1;
       grep -o "q_gap_data_minus_actor[^,]*" logs/density-port-run-flow.log | tail -1; } \
  || { echo "FAIL -> logs/density-port-run-flow.log"; tail -30 logs/density-port-run-flow.log; fail=$((fail+1)); }

echo "== density port smoke done: $fail failure(s) =="
exit $fail
