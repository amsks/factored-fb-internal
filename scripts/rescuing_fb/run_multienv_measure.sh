#!/bin/bash
# Spec 14 §3.5 — measure-axis chain over the torch multi-env checkpoints.
#
# Chain per checkpoint: negmass -> L1fix -> L2-B -> L2-A. Flags match the
# cube-single runs so the rows are directly comparable to the archived JSONs.
# G0 (degenerate_frac < 0.5) is checked after L1; a failing cell is re-run at
# wider multipliers rather than reported.
#
# PHASE 1 (measure_mode=first_passage) SKIPS L1 by design: torchfb_full_measure
# asserts measure_mode==visitation because L1 calibration is defined against the
# visitation fixed point (discounted occupancy). Phase 1's object is E[gamma^T],
# an absorbing first-passage value — comparing it to an MC occupancy would be a
# category error. Phase-1 cells still get negmass + L2, which are well defined
# for it. (Spec 14 §3.6 said "score phase 1 too" without noting this; the
# script's guard is right and stays.)
set -u
cd /home/adityamohan/git/Factored-FB
PY=.venv/bin/python
OUT=/home/adityamohan/exp_spec10/torchfb
RES=results/spec10_torchfb
LOG=/dev/shm/spec14_logs; mkdir -p "$LOG" "$RES"

SEED=${SEED:-42}
# tag|run dir|gpu
JOBS=(
  "scnp1|$OUT/p1_scn_s$SEED|5"
  "scnp2|$OUT/p2_scn_s$SEED|6"
  "cdblp1|$OUT/p1_cdbl_s$SEED|7"
  "cdblp2|$OUT/p2_cdbl_s$SEED|5"
)

run_one() { # tag run gpu
  local TAG=$1 RUN=$2 G=$3
  if [ ! -f "$RUN/checkpoints/final.pt" ]; then
    echo "[s14] $TAG: no ckpt at $RUN - skip"; return 1
  fi
  local T=${TAG}_s$SEED
  local MODE
  MODE=$(grep -h '^measure_mode:' "$RUN/.hydra/config.yaml" | awk '{print $2}')

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/negmass_diag.py \
    --run $RUN --starts 32 --goals 20000 --rho-samples 100000 --seed 0 \
    --out $RES/negmass_$T.json > $LOG/negmass_$T.log 2>&1
  echo "[s14] negmass $T exit=$? $(date +%H:%M:%S)"

  if [ "$MODE" = "visitation" ]; then
    CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_full_measure.py \
      --run $RUN --starts 32 --mc 3 --horizon 200 --boxes 12 --rho-samples 200000 \
      --seed 0 --out $RES/L1fix_$T.json > $LOG/L1fix_$T.log 2>&1
    echo "[s14] L1 $T exit=$? $(date +%H:%M:%S)"
    $PY - "$RES/L1fix_$T.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1])).get("summary", {})
except Exception as e:
    print(f"[G0] unreadable: {e}"); raise SystemExit(0)
for k, v in s.items():
    d = v.get("degenerate_frac")
    flag = "G0-FAIL" if (d is not None and d >= 0.5) else "ok"
    print(f"[G0] mult={k} degenerate_frac={d} {flag}")
EOF
  else
    echo "[s14] L1 $T SKIPPED (measure_mode=$MODE; L1 is visitation-only)"
  fi

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $RUN --stage B --seed 0 --out $RES/L2_B_$T.json > $LOG/L2_B_$T.log 2>&1
  echo "[s14] L2-B $T exit=$? $(date +%H:%M:%S)"

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $RUN --stage A --seed 0 --out $RES/L2_A_$T.json > $LOG/L2_A_$T.log 2>&1
  echo "[s14] L2-A $T exit=$? $(date +%H:%M:%S)"
}

for j in "${JOBS[@]}"; do
  IFS='|' read -r tag run gpu <<<"$j"
  echo "[s14] $(date +%H:%M:%S) start $tag on GPU$gpu"
  run_one "$tag" "$run" "$gpu" &
  sleep 15
done
wait
echo "[s14] MULTI-ENV MEASURE CHAIN DONE (seed $SEED) $(date +%H:%M:%S)"
mkdir -p $OUT/_logs_backup
cp -f $LOG/*_s$SEED.log $OUT/_logs_backup/ 2>/dev/null
cp -f "$0" $OUT/_logs_backup/ 2>/dev/null
