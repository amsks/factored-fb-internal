#!/bin/bash
# Spec 14 §5 — re-score the archived cube-single phase-2 checkpoint through the
# ported (conjunctive-geometry) code, then diff against the archived JSON.
# This is the hard gate: no multi-env cell is trusted until this passes.
set -u
cd /home/adityamohan/git/Factored-FB
OUT=/dev/shm/spec14_logs; mkdir -p $OUT

CUDA_VISIBLE_DEVICES=${GPU:-5} MUJOCO_GL=egl .venv/bin/python \
  scripts/rescuing_fb/torchfb_full_measure.py \
  --run /home/adityamohan/exp_spec10/torchfb/p2_rzK50_s42 \
  --starts 32 --mc 3 --horizon 200 --boxes 12 --rho-samples 200000 \
  --seed 0 --out $OUT/L1_REPLAY.json

.venv/bin/python scripts/rescuing_fb/check_l1_regression.py \
  results/spec10_torchfb/L1fix_rzK50_s42.json $OUT/L1_REPLAY.json
