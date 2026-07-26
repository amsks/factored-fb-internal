#!/bin/bash
# SMB grid (spec 11 §4): Wing A (inferred-z), Wing A oracle-z, Wing B on every
# checkpoint row. 12 ckpts x 3 jobs = 36 jobs, round-robin over this track's GPUs
# (0/2/3/4/7 -- 1/5/6 belong to the other session, never touch). Sequential per
# lane. Usage: bash scripts/rescuing_fb/run_smb_grid.sh <lane-index>
set -u
REPO=/home/adityamohan/git/Factored-FB
PY=/dev/shm/.venv-jax/bin/python
TASKS=$REPO/analysis/rescuing-fb/smb_cube_single_v1.json
OUT=$REPO/analysis/rescuing-fb/smb
LOGS=/dev/shm/smb_logs
mkdir -p "$OUT" "$LOGS"
GPUS=(0 2 3 4 7)
LANE=${1:?lane index 0..4}
GPU=${GPUS[$LANE]}

# name|run_dir|z-route  (retrieval k=1 = the proven decode for goal-mode ckpts;
# moment = direct-z for the separated ZX form)
ROWS=(
  "anchor_s0|/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Bilinear-d50/an_cs_s0|retrieval"
  "anchor_s1|/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Bilinear-d50/an_cs_s1|retrieval"
  "anchor_s2|/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Bilinear-d50/an_cs_s2|retrieval"
  "fp_s0|/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s0|retrieval"
  "fp_s1|/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s1|retrieval"
  "fp_s2|/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s2|retrieval"
  "vis_s0|/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50/vis_s0|retrieval"
  "vis_s1|/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50/vis_s1|retrieval"
  "zx_s0|/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s0|moment"
  "zx_s1|/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s1|moment"
  "visz_s0|/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Measure-BallVisit-ZX-d50/visz_s0|moment"
  "visz_s1|/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Measure-BallVisit-ZX-d50/visz_s1|moment"
)

cd "$REPO"
i=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r name dir route <<<"$row"
  for kind in A Aoracle B; do
    if (( i % ${#GPUS[@]} == LANE )); then
      out="$OUT/smb_${kind}_${name}.json"
      log="$LOGS/${kind}_${name}.log"
      if [[ -s "$out" ]]; then
        echo "[lane $LANE] skip $kind $name (exists)"
      else
        echo "[lane $LANE gpu $GPU] $kind $name start $(date +%H:%M:%S)"
        if [[ $kind == B ]]; then
          CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
            $PY scripts/rescuing_fb/eval_smb_measure.py --run-dir "$dir" \
            --checkpoint-step 500000 --tasks-json "$TASKS" \
            --episodes 25 --horizon 200 --out "$out" >"$log" 2>&1
        else
          extra=""
          [[ $kind == Aoracle ]] && extra="--oracle-z"
          CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
            $PY scripts/rescuing_fb/eval_smb.py --run-dir "$dir" \
            --checkpoint-step 500000 --tasks-json "$TASKS" \
            --tiers t1,t2,t3,t4,t5,t6 --episodes 25 --horizon 200 \
            --gpi --z-inference "$route" --retrieval-k 1 $extra \
            --out "$out" >"$log" 2>&1
        fi
        echo "[lane $LANE] $kind $name exit=$? $(date +%H:%M:%S)"
      fi
    fi
    i=$((i + 1))
  done
done
echo "[lane $LANE] DONE"
