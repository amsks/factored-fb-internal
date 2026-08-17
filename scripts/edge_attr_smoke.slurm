#!/bin/bash
# Gate for docs/specs/2026-08-17-gciql-edge-attribution.md W1/W2 (+W3 config check).
#   sbatch -A ASC26008 scripts/edge_attr_smoke.slurm
#SBATCH -J edge-attr-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/play-resample}"
STEPS="${STEPS:-200}"; OUT="${OUT:-$SCRATCH/factored-fb-edge-attr-smoke}"
fail=0; echo "== host $(hostname) =="

echo "---- W1/W2 unit tests ----"
$PY -m pytest tests/impls/test_density_fb.py -q > logs/edge-attr-tests.log 2>&1 \
  && echo "PASS density_fb tests" || { echo "FAIL -> logs/edge-attr-tests.log"; tail -40 logs/edge-attr-tests.log; fail=$((fail+1)); }

echo "---- regression: GCDataset next_actions must not disturb other critics ----"
$PY -m pytest tests/impls -q -k "gcdataset or dataset_seam or pr3_gciql or iql_bilinear or td_fb or combo" \
  > logs/edge-attr-reg.log 2>&1 \
  && echo "PASS regression subset" || { echo "FAIL -> logs/edge-attr-reg.log"; tail -40 logs/edge-attr-reg.log; fail=$((fail+1)); }

E3P=(--override obs_mask_dims=[36,37,38] --override goal_mask_dims=[36,37,38]
     --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8
     --override eval_goal_mode=synth --override measure_density_hard=true
     --override 'measure_hit_spec="4:7:0.07"')
run() { # run <tag> <critic> <domain> <extra...>
  local tag=$1 cr=$2 dom=$3; shift 3
  local ov=(); for kv in "$@"; do ov+=(--override "$kv"); done
  echo "---- $tag ----"
  if $PY -m impls.main --critic "$cr" --actor flowbc --domain "$dom" --seed 0 \
      --train_steps $STEPS --log_interval $STEPS --eval_interval $STEPS \
      --save_interval $STEPS --eval_episodes 4 --num_video 0 \
      "${E3P[@]}" "${ov[@]}" --save_dir "$OUT/$tag" > "logs/edge-attr-$tag.log" 2>&1; then
    echo "PASS $tag"; grep -h "^eval" "logs/edge-attr-$tag.log" | tail -1
  else echo "FAIL $tag -> logs/edge-attr-$tag.log"; tail -30 "logs/edge-attr-$tag.log"; fail=$((fail+1)); fi
}
run ktB-sarsa   density_fb mw-pick-place measure_density_bootstrap=sarsa
run ktC-atemp2  density_fb mw-pick-place actor.q_coeff=0.3 measure_density_actor_temp=2.0
run ktC-nocenter density_fb mw-pick-place actor.q_coeff=0.3 actor_q_center=false
run default     density_fb mw-pick-place

echo "---- W3: gciql expectile on both tasks ----"
for d in mw-pick-place mw-push; do
  if $PY -m impls.main --critic gciql --actor flowbc --domain $d --seed 0 \
      --train_steps $STEPS --log_interval $STEPS --eval_interval $STEPS --save_interval $STEPS \
      --eval_episodes 4 --num_video 0 \
      --override obs_mask_dims=[36,37,38] --override goal_mask_dims=[36,37,38] \
      --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8 \
      --override eval_goal_mode=synth --override expectile=0.7 \
      --save_dir "$OUT/ktD-$d" > "logs/edge-attr-ktD-$d.log" 2>&1; then
    echo "PASS ktD-$d"
  else echo "FAIL ktD-$d -> logs/edge-attr-ktD-$d.log"; tail -30 "logs/edge-attr-ktD-$d.log"; fail=$((fail+1)); fi
done

echo "---- density_sarsa_subst logged? ----"
grep -oE "density_sarsa_subst[^,}]*" logs/edge-attr-default.log | tail -1 \
  && echo "PASS subst logged" || { echo "FAIL: density_sarsa_subst missing"; fail=$((fail+1)); }

echo "== edge-attr smoke done: $fail failure(s) =="
exit $fail
