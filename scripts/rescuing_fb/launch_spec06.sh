#!/usr/bin/env bash
# Spec 06 phenomenon-implementations launcher (PAPER/rescuing-fb/specs/06-*.md).
# 2026-07-20. Pre-registered; DRY_RUN=1 by default (prints the plan, launches nothing).
#
# ARMS (select with ARMS="...", space-separated; default "all"):
#   d512v0     item 2 : RGFB-Cube-Single-State-d512-v0                (3 seeds, 500k)
#   v1d50      item 4 : RGFB-Cube-Single-State-d50-v1-taskprior       (3 seeds, 500k)
#   v1d512     item 4 : RGFB-Cube-Single-State-d512-v1-taskprior      (only if item 2 won)
#   measure    items 5/6/7 : the 4 single-axis measure cells          (3 seeds each, 500k)
#              FirstPassage / Expectile-FP / Expectile / HindsightTP
#
# ORDERING & STOPPING RULE (spec 06): 1 -> 2 -> 3 -> 4 (each gates the next's config),
# then 5 -> 6 -> 7. Stop the method chain if zero-shot >=0.90 via any of items 2/3/4.
# Items 1/3/8 are CPU EVAL scripts (t6_gpi_eval.py / rgfb_rerank_eval.py /
# measure_readout_head.py) -- run those directly, they are not training campaigns.
#
# Env invariants: run in tmux; wandb entity amsks; XLA_PYTHON_CLIENT_PREALLOCATE=false;
# jax co-init OOMs -> stagger >=90s. GPU etiquette (updated 2026-07-21: the old "1/5/6 =
# other track" fence is STALE -- the sota track runs on 0/2/3/4/7 now): there is no fixed
# ownership; run `nvidia-smi --query-compute-apps` and place on the emptiest GPUs, max
# ~3 procs/GPU for these state cells, and never kill/move another session's pids.
#   tmux new -d -s s06 'DRY_RUN=0 ARMS="measure" scripts/rescuing_fb/launch_spec06.sh 2>&1 | tee /dev/shm/s06_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher (repo .venv, NOT .venv-jax);
                                    # it subprocesses ogbench into the jax venv itself.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec06}
STAGGER=${STAGGER:-100}
SEEDS=(${SEEDS:-0 1 2})
ARMS=${ARMS:-all}
GPUS=(${GPUS:-2 3 4})               # one seed per GPU for the RG-FB arms; measure packs on GPUS[0]
LOGDIR=/dev/shm/s06_logs
mkdir -p "$LOGDIR" "$OUTROOT"

want() { [ "$ARMS" = "all" ] || [[ " $ARMS " == *" $1 "* ]]; }

# tag ||| extra-env ||| hydra args
JOBS=()
if want d512v0; then
  for s in "${SEEDS[@]}"; do
    JOBS+=("d512v0_s${s}||| |||--config-name cube_single_state_rgfb_d512 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=d512v0_s${s}")
  done
fi
if want v1d50; then
  for s in "${SEEDS[@]}"; do
    JOBS+=("v1d50_s${s}||| |||--config-name cube_single_state_rgfb_v1_taskprior_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=v1d50_s${s}")
  done
fi
if want v1d512; then
  for s in "${SEEDS[@]}"; do
    JOBS+=("v1d512_s${s}||| |||--config-name cube_single_state_rgfb_v1_taskprior_d512 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=v1d512_s${s}")
  done
fi
if want noortho; then
  # 2026-07-21 scale-fix reruns: CellW0 + FP-only with ortho=0 (M-units TD carries (1-g)^2;
  # ortho lam=1 was effectively 1e3-1e4x => the 0.072 ortho-catastrophe fingerprint).
  for cfg in cube_single_state_measure_cellw0_noortho_d50 \
             cube_single_state_measure_firstpassage_noortho_d50; do
    short=$(echo "$cfg" | sed 's/cube_single_state_measure_//;s/_d50//')
    for s in "${SEEDS[@]}"; do
      JOBS+=("m_${short}_s${s}||| |||--config-name $cfg seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=m_${short}_s${s}")
    done
  done
fi
if want cellw0; then
  # CELL W w=0: theorem-backed win (== bilinear GC-IQL under M=1+(1-g)Q).
  for s in "${SEEDS[@]}"; do
    JOBS+=("m_cellw0_s${s}||| |||--config-name cube_single_state_measure_cellw0_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=m_cellw0_s${s}")
  done
fi
if want cellw03; then
  # CELL W w=0.3: + off-diagonal matrix supervision at a GC-IQL-like mix.
  for s in "${SEEDS[@]}"; do
    JOBS+=("m_cellw03_s${s}||| |||--config-name cube_single_state_measure_cellw03_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=m_cellw03_s${s}")
  done
fi
if want corner; then
  # FULL-CORNER measure (fp+expectile+hindsight = all three axes; the headline arm).
  for s in "${SEEDS[@]}"; do
    JOBS+=("m_corner_s${s}||| |||--config-name cube_single_state_measure_gciqlcorner_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=m_corner_s${s}")
  done
fi
if want measure; then
  for cfg in cube_single_state_measure_firstpassage_d50 \
             cube_single_state_measure_expectile_fp_d50 \
             cube_single_state_measure_expectile_d50 \
             cube_single_state_measure_hindsighttp_d50; do
    short=${cfg#cube_single_state_measure_}; short=${short%_d50}
    for s in "${SEEDS[@]}"; do
      JOBS+=("m_${short}_s${s}||| |||--config-name $cfg seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=m_${short}_s${s}")
    done
  done
fi

echo "[s06] ARMS='$ARMS' -> ${#JOBS[@]} runs @${STEPS} over GPUs ${GPUS[*]} -> $OUTROOT"
if [ "${#JOBS[@]}" = "0" ]; then echo "[s06] no arms selected (ARMS='$ARMS')"; exit 1; fi
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  echo "[s06] DRY_RUN=1 (set DRY_RUN=0 to launch). Nothing launched."
  exit 0
fi

i=0; first=1
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; rest=${j#*|||}; extra_env=${rest%%|||*}; args=${rest#*|||}
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  if [ "$first" = "0" ]; then
    echo "[stagger] sleeping ${STAGGER}s before $tag (jax co-init OOMs)"; sleep "$STAGGER"
  fi
  first=0
  echo "[launch] GPU$g  $tag  (env: ${extra_env:-none})"
  env $extra_env XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
done
echo "[s06] launched ${#JOBS[@]} runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
