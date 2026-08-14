#!/usr/bin/env bash
# spec 28 P0b -- three-encoding reward reconstruction over every checkpoint the
# spec names: z24 cube-double (3 seeds), the four z25 stage-1 arms (cube-single
# 500k), and the two g-indexed anchors. EVAL ONLY, no rollouts, CPU jax.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=results/spec28_p0b
mkdir -p "$OUT"
PY=/dev/shm/.venv-jax/bin/python
ENVV="JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl"

run () {  # name  run-dir
  local name="$1" dir="$2"
  if [[ -f "$OUT/$name.json" ]]; then echo "[p0b] skip $name (exists)"; return; fi
  echo "[p0b] === $name"
  env $ENVV "$PY" scripts/rescuing_fb/eval_spec28_p0b.py \
    --run-dir "$dir" --checkpoint-step 500000 --pool_size 16384 --seed 0 \
    --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  echo "[p0b] $name EXIT=$? $(grep -c . "$OUT/$name.log") log lines"
}

S24=$HOME/exp_spec24/factored-fb/GCIQL-CubeDouble-Flow-Measure-DensityZ-d50
S25=$HOME/exp_spec25_stage1/factored-fb
run z24_cd_z_s0 "$S24/z24_cd_z_s0"
run z24_cd_z_s1 "$S24/z24_cd_z_s1"
run z24_cd_z_s2 "$S24/z24_cd_z_s2"
run z25_cs_tilt  "$S25/GCIQL-CubeSingle-Flow-Measure-DensityZ-C1Tilt-d50/z25_cs_tilt"
run z25_cs_dec   "$S25/GCIQL-CubeSingle-Flow-Measure-DensityZ-C2Dec-d50/z25_cs_dec"
run z25_cs_actor "$S25/GCIQL-CubeSingle-Flow-Measure-DensityZ-C3Actor-d50/z25_cs_actor"
run z25_cs_full  "$S25/GCIQL-CubeSingle-Flow-Measure-DensityZ-Full-d50/z25_cs_full"
run g_cs_dens_s0 "$HOME/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0"
run g_cd_dens_s0 "$HOME/exp_spec18x/factored-fb/GCIQL-CubeDouble-Flow-Measure-Density-d50/d18x_cd_dens_s0"
echo "[p0b] ALL DONE"
touch "$OUT/.done"
