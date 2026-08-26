#!/bin/bash
# HumEnv (Meta Motivo SMPL humanoid) bring-up smoke — the gate before any long hum- launch.
# Login nodes cannot run JAX (Eigen thread pool vs the per-user cap), so everything runs here.
#
#   sbatch -A ASC26008 scripts/humenv_smoke.slurm
#
# Checks, in order of what they would kill:
#   1. the converted buffer loads and its shapes match the env's spaces
#   2. the env builds and steps under EGL on a compute node
#   3. density_fb x flowbc trains for a few hundred steps without NaN
#   4. density_ess_median lands in the healthy band at the CALIBRATED hit spec — the one
#      number that decides whether the 214-dim pose block is usable at all
#
#SBATCH -J humenv-smoke
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
OUT="$SCRATCH/humenv-smoke"
# The calibrated block+radius (scripts/calibrate_hit_spec.py; see env/domains/hum-humanoid.yaml).
HIT='measure_hit_spec="0:214:2.548"'
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

echo "---- 1. converted buffer + env shapes agree ----"
$PY - <<'EOF' && echo "PASS shapes" || { echo "FAIL shapes"; fail=$((fail+1)); }
import glob, os, numpy as np
d = os.environ.get("FB_HUMENV_ROOT", os.environ["SCRATCH"] + "/fb-data/humenv-S1")
eps = sorted(glob.glob(os.path.join(d, "buffer", "episode_*.npz")))
assert eps, f"no episodes under {d}/buffer"
z = np.load(eps[0])
print(f"  episodes={len(eps):,}  obs{z['observation'].shape}  act{z['action'].shape}  nobs{z['next_observation'].shape}")
assert z["observation"].shape[1] == 358 and z["action"].shape[1] == 69
assert z["observation"].shape[0] == z["action"].shape[0] == z["next_observation"].shape[0]
from env.humenv_env import make_humenv_env
env, horizon = make_humenv_env("hum-humanoid")
assert env.observation_space.shape == (358,) and env.action_space.shape == (69,), env.observation_space
o, _ = env.reset(); o2, r, te, tr, _ = env.step(env.action_space.sample())
print(f"  env ok: obs{np.asarray(o).shape} {np.asarray(o).dtype}  horizon={horizon}  reward={r} (0.0 by design)")
EOF

echo "---- 2. offline density_fb x ddpgbc_density, FAITHFUL arm (alpha=0, z index on) ----"
# THE FAITHFUL ARM, matching Meta Motivo's actor as closely as this stack can:
#   * `ddpgbc_density`, NOT stock `ddpgbc`. density_fb.yaml sets actor_q_center=true and
#     q_fn returns the centered normalizer as its SECOND value; stock ddpgbc's loss divides
#     by |q|.mean() and throws that leg away. The bridge measured the difference as antmaze
#     0.920/0.868/0.844 vs 0.772/0.580/0.656 (spec 22). scripts/launch_exorl.sh defaulted to
#     stock ddpgbc, so every ExoRL density_fb cell ran uncentered.
#   * actor.alpha=0.0 -- NO behaviour cloning. FB-CPR's actor loss is
#     `-Q_discriminator*reg_coeff*weight - Q_fb` with no BC term at all
#     (metamotivo/fb_cpr/agent.py:358). Its regulariser is a discriminator reward trained
#     against AMASS, which we cannot replicate, so the faithful actor is FB-CPR's minus CPR
#     = pure Q-ascent. The deviation is "no CPR", never "we added BC".
#   * measure_density_z=true -- the density objective NORMALISES the successor measure, so
#     FB's z = E_rho[r*B] identity does not hold; the centered dual (spec 30 P1) replaces it.
#   * pool 1024, not the default 256. The pool is drawn from ONE batch's next states so it is
#     capped at batch_size. Measured ESS(r) at the calibrated eps: 155/256 and 616/1024 --
#     informative at both, but degenerate (constant -> all-ones fallback) by eps=4.0. This is
#     the ExoRL M=256 failure mode, which cost us a relaunch there.
# eval_interval == train_steps: one eval only. A hum- eval scores nothing (env reward is 0
# by design — see env/humenv_env.py) and exists here purely to prove the path runs.
$PY -m impls.main --critic density_fb --actor ddpgbc_density --domain hum-humanoid --seed 0 \
   --train_steps 300 --log_interval 50 --eval_interval 300 --save_interval 300 \
   --eval_episodes 2 --num_video 0 --save_dir "$OUT/off-goal" \
   --override "$HIT" --override actor.alpha=0.0 --override measure_density_z=true \
   --override measure_density_z_eval_pool_m=1024 --override measure_density_dim=128 \
   > logs/humenv-smoke-off-goal.log 2>&1 \
  && echo "PASS offline" || { echo "FAIL -> logs/humenv-smoke-off-goal.log"; tail -40 logs/humenv-smoke-off-goal.log; fail=$((fail+1)); }

echo "---- 3. ESS / CE readout at the calibrated spec ----"
$PY - <<'EOF' || fail=$((fail+1))
import csv, glob, io, os
f = glob.glob(os.environ["SCRATCH"] + "/humenv-smoke/off-goal/**/train.csv", recursive=True)
if not f:
    print("  no train.csv — step 2 failed"); raise SystemExit(1)
txt = open(f[0], "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("density_ess_median")]
last = rows[-1]
# The actors disagree on the q column name: flowbc logs `q`, the ddpgbc family logs
# `q_mean` (plus q_scale / q_loss). Read whichever is present rather than assuming an actor.
ess, ce = float(last["density_ess_median"]), float(last["density_ce"])
qk = "q" if "q" in last else "q_mean"
gap = float(last.get("action_gap_l2", "nan"))
print(f"  step {last['step']}: ess_median={ess:.1f}  density_ce={ce:.3f}  {qk}={float(last[qk]):.5f}  action_gap_l2={gap:.2f}")
# Band from the cube/ExoRL runs; 300 steps is far too early to judge, so this only flags the
# two degenerate ends (uniform kernel, or collapsed to a single query).
if ess > 900:   print("  WARN kernel is near-UNIFORM at this radius — eps too large")
elif ess < 2:   print("  WARN kernel collapsed to ~1 effective query — eps too small")
else:           print("  ESS is in a workable range at 300 steps")
EOF

echo "== smoke failures: $fail =="
exit $fail
