#!/bin/bash
# Full-measure gate campaign (spec 10 L1+L2 on the SMB failures, 2026-07-26).
# 12 jobs over the ZX checkpoints:
#   Wing B density predictor:      visz_s0/s1 (the object), zx_s0/s1 (units control)
#   Wing A cmoment+measure-select: visz_s0/s1, tiers t3,t4,t5,t6
#   Wing A cmoment only (ablation): visz_s0/s1  -- isolates z-inference from z-select
#   Wing A moment+measure-select (ablation): visz_s0/s1 -- isolates z-select
# Usage: bash run_smb_fullmeasure.sh <lane 0..3>   (GPUs 0/2/3/4; 1/5/6 forbidden)
set -u
REPO=/home/adityamohan/git/Factored-FB
PY=/dev/shm/.venv-jax/bin/python
TASKS=$REPO/analysis/rescuing-fb/smb_cube_single_v1.json
OUT=$REPO/analysis/rescuing-fb/smb
LOGS=/dev/shm/smb_logs
mkdir -p "$OUT" "$LOGS"
GPUS=(0 2 3 4)
LANE=${1:?lane 0..3}
GPU=${GPUS[$LANE]}

VISZ0=/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Measure-BallVisit-ZX-d50/visz_s0
VISZ1=/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Measure-BallVisit-ZX-d50/visz_s1
ZX0=/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s0
ZX1=/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s1

# kind|name|dir|args
JOBS=(
  "B|fm_Bd_visz_s0|$VISZ0|--predictor density"
  "B|fm_Bd_visz_s1|$VISZ1|--predictor density"
  "B|fm_Bd_zx_s0|$ZX0|--predictor density"
  "B|fm_Bd_zx_s1|$ZX1|--predictor density"
  "A|fm_A_full_visz_s0|$VISZ0|--z-inference cmoment --z-select measure"
  "A|fm_A_full_visz_s1|$VISZ1|--z-inference cmoment --z-select measure"
  "A|fm_A_cmom_visz_s0|$VISZ0|--z-inference cmoment"
  "A|fm_A_cmom_visz_s1|$VISZ1|--z-inference cmoment"
  "A|fm_A_msel_visz_s0|$VISZ0|--z-inference moment --z-select measure"
  "A|fm_A_msel_visz_s1|$VISZ1|--z-inference moment --z-select measure"
)

cd "$REPO"
i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r kind name dir extra <<<"$j"
  if (( i % ${#GPUS[@]} == LANE )); then
    out="$OUT/smb_${name}.json"
    if [[ -s "$out" ]]; then
      echo "[fm lane $LANE] skip $name"
    else
      echo "[fm lane $LANE gpu $GPU] $name start $(date +%H:%M:%S)"
      if [[ $kind == B ]]; then
        CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
          $PY scripts/rescuing_fb/eval_smb_measure.py --run-dir "$dir" \
          --checkpoint-step 500000 --tasks-json "$TASKS" \
          --episodes 25 --horizon 200 $extra --out "$out" \
          >"$LOGS/${name}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
          $PY scripts/rescuing_fb/eval_smb.py --run-dir "$dir" \
          --checkpoint-step 500000 --tasks-json "$TASKS" \
          --tiers t3,t4,t5,t6 --episodes 25 --horizon 200 --gpi $extra \
          --out "$out" >"$LOGS/${name}.log" 2>&1
      fi
      echo "[fm lane $LANE] $name exit=$? $(date +%H:%M:%S)"
    fi
  fi
  i=$((i + 1))
done
echo "[fm lane $LANE] DONE"
