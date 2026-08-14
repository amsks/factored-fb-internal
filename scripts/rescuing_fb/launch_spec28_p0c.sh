#!/usr/bin/env bash
# spec 28 P0c -- the e2b assignment protocol re-scored with the ridge (dual)
# index, readout-only, coordinate-percentile metric. Every z checkpoint runs
# with its BIT-EXACT control gate against the archived e2b JSON; the g-indexed
# anchors run ungated as the reference arm (a g checkpoint has no archived e2b
# -- its own control gate is the family-means one in launch_spec28_gate.sh).
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=results/spec28_p0c
mkdir -p "$OUT"
PY=/dev/shm/.venv-jax/bin/python
ENVV="JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl"

run () {  # name  run-dir  [gate-json]
  local name="$1" dir="$2" gate="${3:-}"
  if [[ -f "$OUT/$name.json" ]]; then echo "[p0c] skip $name (exists)"; return; fi
  echo "[p0c] === $name  gate=${gate:-none}"
  env $ENVV "$PY" scripts/rescuing_fb/eval_spec28_p0c.py \
    --run-dir "$dir" --checkpoint-step 500000 --pool_size 16384 --seed 0 \
    ${gate:+--gate-against "$gate"} --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1
  echo "[p0c] $name EXIT=$?"
}

S24=$HOME/exp_spec24/factored-fb/GCIQL-CubeDouble-Flow-Measure-DensityZ-d50
S25=$HOME/exp_spec25_stage1/factored-fb
run z25_cs_actor "$S25/GCIQL-CubeSingle-Flow-Measure-DensityZ-C3Actor-d50/z25_cs_actor" \
    results/spec27_q2/e2b_z25_cs_actor.json
run z24_cd_z_s0 "$S24/z24_cd_z_s0" results/spec24_eval/cd_e2b_s0_v2.json
run z24_cd_z_s1 "$S24/z24_cd_z_s1" results/spec24_eval/cd_e2b_s1_v2.json
run z24_cd_z_s2 "$S24/z24_cd_z_s2" results/spec24_eval/cd_e2b_s2_v2.json
run g_cs_dens_s0 "$HOME/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0"
run g_cd_dens_s0 "$HOME/exp_spec18x/factored-fb/GCIQL-CubeDouble-Flow-Measure-Density-d50/d18x_cd_dens_s0"
echo "[p0c] ALL DONE"
touch "$OUT/.done"
