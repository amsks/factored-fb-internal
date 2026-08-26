#!/bin/bash
# Pre-launch check for the Meta Motivo parity arm (scripts/launch_humenv_motivo.sh).
# Login nodes cannot run JAX here, so the unit tests run on the compute node too.
#
#   sbatch -A ASC26008 scripts/humenv_motivo_smoke.slurm
#
# What it checks, in the order that would kill the launch:
#   1. the density_fb unit suite still passes with norm_obs off  (nothing published moved)
#   2. 300 steps of the full parity config train without NaN
#   3. the running observation statistics actually moved off their init, and the standardized
#      observation has ~zero mean / ~unit variance on a fresh batch
#   4. lr_f / lr_b reached the two optimizers as 1e-4 / 1e-5 rather than merging in inert
#SBATCH -J humenv-motivo-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT="$SCRATCH/humenv-motivo-smoke"
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

echo "---- 1. density_fb unit suite (norm_obs off => every earlier cell unchanged) ----"
# JAX_PLATFORMS=cpu is REQUIRED, not tidiness: the byte-identity fixtures were recorded on
# CPU and every comparison in them fails on GPU in the last bits (tests/impls/
# test_boot_contract.py::_require_cpu says so and refuses to run).
JAX_PLATFORMS=cpu $PY -m pytest -q tests/impls/test_density_fb.py tests/impls/test_density_fb_actors.py \
     tests/impls/test_density_fb_reward_index.py tests/impls/test_boot_contract.py \
     > logs/humenv-motivo-smoke-pytest.log 2>&1 \
  && echo "PASS pytest" || { echo "FAIL -> logs/humenv-motivo-smoke-pytest.log"; tail -30 logs/humenv-motivo-smoke-pytest.log; fail=$((fail+1)); }

echo "---- 2. 300 steps at the full parity config ----"
OV_HIT='measure_hit_spec="0:214:2.548"'
$PY -m impls.main --critic density_fb --actor ddpgbc_density --domain hum-humanoid --seed 0 \
   --train_steps 300 --log_interval 50 --eval_interval 300 --save_interval 300 \
   --eval_episodes 2 --num_video 0 --save_dir "$OUT/parity" \
   --override "$OV_HIT" \
   --override discount=0.98 --override tau=0.01 --override batch_size=1024 \
   --override lr=1.0e-4 --override lr_f=1.0e-4 --override lr_b=1.0e-5 --override lr_actor=1.0e-4 \
   --override value_hidden_dims=[1024,1024] --override measure_density_b_hidden_dims=[256] \
   --override measure_density_dim=256 --override norm_obs=true \
   --override measure_density_z=true --override measure_density_z_rand_frac=0.8 \
   --override measure_density_z_eval_pool_m=1024 \
   --override actor.hidden_dims=[1024,1024] --override actor.alpha=0.0 \
   --override actor.const_std=true \
   --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 \
   > logs/humenv-motivo-smoke-train.log 2>&1 \
  && echo "PASS train" || { echo "FAIL -> logs/humenv-motivo-smoke-train.log"; tail -40 logs/humenv-motivo-smoke-train.log; fail=$((fail+1)); }

echo "---- 3. the running observation statistics moved, and standardizing works ----"
$PY - <<'PYEOF' || fail=$((fail+1))
import csv, glob, io, math, os
import numpy as np
root = os.environ["SCRATCH"] + "/humenv-motivo-smoke/parity"
ck = sorted(glob.glob(root + "/**/params_*.pkl", recursive=True))
if not ck:
    print("  no checkpoint — step 2 failed"); raise SystemExit(1)
import pickle
with open(ck[-1], "rb") as f:
    blob = pickle.load(f)
def find(o, key, depth=0):
    if depth > 6 or not isinstance(o, dict): return None
    if key in o: return o[key]
    for v in o.values():
        r = find(v, key, depth + 1)
        if r is not None: return r
    return None
st = find(blob, "density_obsnorm")
if st is None:
    print("  FAIL: no density_obsnorm in the checkpoint — the group was never registered")
    raise SystemExit(1)
mean = np.asarray(find(st, "mean")); var = np.asarray(find(st, "var"))
print(f"  mean: |.|max={np.abs(mean).max():.4f}  var: min={var.min():.4f} max={var.max():.4f}")
if np.allclose(mean, 0.0) and np.allclose(var, 1.0):
    print("  FAIL: statistics never moved off their init"); raise SystemExit(1)
d = os.environ.get("FB_HUMENV_ROOT", os.environ["SCRATCH"] + "/fb-data/humenv-S1")
z = np.load(sorted(glob.glob(os.path.join(d, "buffer", "episode_*.npz")))[0])
x = z["observation"][:256]
xs = (x - mean) / np.sqrt(var + 1e-5)
print(f"  standardized batch: mean={xs.mean():+.3f}  std={xs.std():.3f}  (EMA is 300 steps old)")
csvf = glob.glob(root + "/**/train.csv", recursive=True)
txt = open(csvf[0], "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("density_ess_median")]
last = rows[-1]
ess, ce = float(last["density_ess_median"]), float(last["density_ce"])
print(f"  step {last['step']}: ess_median={ess:.1f}  density_ce={ce:.3f}  q_mean={float(last['q_mean']):.5f}")
if any(math.isnan(float(v)) for v in (ess, ce)):
    print("  FAIL: NaN in the training log"); raise SystemExit(1)
PYEOF

echo "---- 4. lr_f / lr_b reached the two optimizers ----"
$PY - <<'PYEOF' || fail=$((fail+1))
from impls.agents import _group_lr
c = {"lr": 1e-4, "lr_f": 1e-4, "lr_b": 1e-5, "lr_actor": 1e-4}
got = (_group_lr("density_f", c), _group_lr("density_b", c))
print(f"  density_f lr={got[0]}  density_b lr={got[1]}")
assert got == (1e-4, 1e-5), got
old = {"lr": 3e-4, "lr_actor": 3e-4}
assert (_group_lr("density_f", old), _group_lr("density_b", old)) == (3e-4, 3e-4)
print("  and a config without the split still rides the single lr")
PYEOF

echo "== smoke failures: $fail =="
exit $fail
