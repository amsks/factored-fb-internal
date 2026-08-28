#!/bin/bash
# Pre-launch check for the humenv CPR arms (scripts/launch_humenv_cpr.sh):
#   1. unit suite still passes (the `_z_hindsight` cond key and `_z_mix_masked` refactor
#      change no published number; the byte-identity fixtures police that on CPU)
#   2. 300 steps of ddpgbc_density_cpr at the motivo parity config (q_coeff=1) train
#      without NaN, log a finite cpr_bc_loss, and hindsight_frac ~ 1 - z_rand_frac = 0.2
#   3. same at q_coeff=0 (the pure hindsight-BC floor arm)
#
#   sbatch -A ASC26008 scripts/humenv_cpr_smoke.slurm
#SBATCH -J humenv-cpr-smoke
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
OUT="$SCRATCH/humenv-cpr-smoke"
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

echo "---- 1. unit suite (hindsight-mask publication must not move any published number) ----"
JAX_PLATFORMS=cpu $PY -m pytest -q tests/impls/test_density_fb.py tests/impls/test_density_fb_actors.py \
     tests/impls/test_density_fb_reward_index.py tests/impls/test_boot_contract.py \
     tests/impls/test_combo_validation.py tests/impls/test_actor_static.py \
     > logs/humenv-cpr-smoke-pytest.log 2>&1 \
  && echo "PASS pytest" || { echo "FAIL -> logs/humenv-cpr-smoke-pytest.log"; tail -30 logs/humenv-cpr-smoke-pytest.log; fail=$((fail+1)); }

OV=(--override 'measure_hit_spec="0:214:2.548"'
    --override discount=0.98 --override tau=0.01 --override batch_size=1024
    --override lr=1.0e-4 --override lr_f=1.0e-4 --override lr_b=1.0e-5 --override lr_actor=1.0e-4
    --override 'value_hidden_dims=[1024,1024]' --override 'measure_density_b_hidden_dims=[256]'
    --override measure_density_dim=256 --override norm_obs=true
    --override measure_density_archi=motivo --override actor.motivo_archi=true
    --override measure_density_b_norm=true
    --override measure_density_z=true --override measure_density_z_rand_frac=0.8
    --override measure_density_z_eval_pool_m=1024
    --override 'actor.hidden_dims=[1024,1024]' --override actor.alpha=0.0
    --override actor.const_std=true
    --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3
    --override actor.cpr_alpha=1.0)

for QC in 1.0 0.0; do
  TAG=$([ "$QC" = "1.0" ] && echo cpr || echo cprbc)
  echo "---- 300 steps ddpgbc_density_cpr q_coeff=$QC ($TAG) ----"
  $PY -m impls.main --critic density_fb --actor ddpgbc_density_cpr --domain hum-humanoid --seed 0 \
     --train_steps 300 --log_interval 50 --eval_interval 300 --save_interval 300 \
     --eval_episodes 2 --num_video 0 --save_dir "$OUT/$TAG" \
     "${OV[@]}" --override actor.q_coeff=$QC \
     > "logs/humenv-cpr-smoke-$TAG.log" 2>&1 \
    && echo "PASS train $TAG" || { echo "FAIL -> logs/humenv-cpr-smoke-$TAG.log"; tail -40 "logs/humenv-cpr-smoke-$TAG.log"; fail=$((fail+1)); }
done

echo "---- cpr_bc_loss finite, hindsight_frac ~ 0.2, floor arm's q term actually off ----"
$PY - <<'PYEOF' || fail=$((fail+1))
import csv, glob, io, math, os
def last_row(tag):
    f = glob.glob(os.environ["SCRATCH"] + f"/humenv-cpr-smoke/{tag}/**/train.csv", recursive=True)
    txt = open(f[0], "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
    rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("cpr_bc_loss")]
    return rows[-1]
ok = True
for tag in ("cpr", "cprbc"):
    r = last_row(tag)
    cpr, hf = float(r["cpr_bc_loss"]), float(r["hindsight_frac"])
    print(f"  {tag}: step {r['step']} cpr_bc_loss={cpr:.4f} hindsight_frac={hf:.3f} "
          f"q_loss={float(r['q_loss']):.4f} actor_loss={float(r['actor_loss']):.4f}")
    if math.isnan(cpr) or not (0.10 <= hf <= 0.30):
        print(f"  FAIL: {tag} cpr term NaN or mask off target"); ok = False
r = last_row("cprbc")
if abs(float(r["actor_loss"]) - (float(r["bc_loss"]) + float(r["cpr_bc_loss"]))) > 1e-4:
    print("  FAIL: q_coeff=0 arm's actor_loss still contains the q term"); ok = False
raise SystemExit(0 if ok else 1)
PYEOF

echo "== smoke failures: $fail =="
exit $fail
