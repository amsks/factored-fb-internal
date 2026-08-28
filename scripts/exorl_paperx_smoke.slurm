#!/bin/bash
# Pre-launch check for the Touati-parity arm (scripts/launch_exorl_paperx.sh).
# Every changed code path is exercised: fb with aux Q loss + min heads + identity
# left_encoder + 526-wide B + motivo actor; density_fb at the translated shapes; both with
# actor-loss q noise. Override strings are taken from the launcher's own DRYRUN output, so
# what is smoked is byte-for-byte what launches. exorl_max_episodes=100 is appended LAST so
# it wins over the parity value — data volume is not a code path and the maze 10M load
# would eat the window.
#   sbatch -A ASC26008 scripts/exorl_paperx_smoke.slurm
#SBATCH -J exorl-paperx-smoke
#SBATCH -A ASC26008
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
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets" EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
OUT="$SCRATCH/exorl-paperx-smoke"
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

echo "---- 1. unit suite (new knobs default off => prior cells bit-identical) ----"
JAX_PLATFORMS=cpu $PY -m pytest -q tests/impls/test_density_fb.py tests/impls/test_density_fb_actors.py \
     tests/impls/test_fb_critic_static.py tests/impls/test_fb_math.py tests/impls/test_actor_static.py \
     tests/impls/test_combo_validation.py \
     > logs/paperx-smoke-pytest.log 2>&1 \
  && echo "PASS pytest" || { echo "FAIL -> logs/paperx-smoke-pytest.log"; tail -30 logs/paperx-smoke-pytest.log; fail=$((fail+1)); }

echo "---- 2. 200 steps of every seed-0 cell, overrides from the launcher's DRYRUN ----"
DRYRUN=1 SEEDS=0 bash scripts/launch_exorl_paperx.sh > "$OUT.dryrun" 2>&1 || true
grep -c OVERRIDE "$OUT.dryrun" | grep -q '^6$' || { echo "FAIL: expected 6 DRYRUN cells"; cat "$OUT.dryrun"; exit 1; }
paste -d'|' <(grep ^DRYRUN "$OUT.dryrun" | awk '{print $2}') \
            <(grep OVERRIDE "$OUT.dryrun" | sed "s/^ *OVERRIDE='//; s/'$//") \
| while IFS='|' read -r NAME OVSTR; do
  CRITIC=$(echo "$NAME" | cut -d- -f2)
  DOM=dmc-$(echo "$NAME" | sed 's/^XP-[a-z_]*-//; s/-s0$//')
  ACTOR=$([ "$CRITIC" = density_fb ] && echo ddpgbc_density || echo ddpgbc)
  echo "-- $CRITIC on $DOM --"
  eval "OVA=($(echo "$OVSTR" | sed "s/\([^ ]*\)/--override '\1'/g"))"
  $PY -m impls.main --critic "$CRITIC" --actor "$ACTOR" --domain "$DOM" --seed 0 \
     --train_steps 200 --log_interval 100 --eval_interval 200 --save_interval 200 \
     --eval_episodes 2 --num_video 0 --save_dir "$OUT/$CRITIC-$DOM" \
     --infer_cond_episodes 10000 --eval_on_inferred_cond \
     "${OVA[@]}" --override exorl_max_episodes=100 \
     > "logs/paperx-smoke-$CRITIC-${DOM#dmc-}.log" 2>&1 \
    && echo "PASS $CRITIC $DOM" \
    || { echo "FAIL -> logs/paperx-smoke-$CRITIC-${DOM#dmc-}.log"; tail -30 "logs/paperx-smoke-$CRITIC-${DOM#dmc-}.log"; echo FAILMARK >> "$OUT.fails"; }
done
[ -f "$OUT.fails" ] && fail=$((fail + $(grep -c FAILMARK "$OUT.fails"))) && rm -f "$OUT.fails"

echo "---- 3. fb's aux Q loss actually engaged and is finite ----"
$PY - <<'PYEOF' || fail=$((fail+1))
import csv, glob, io, math, os
ok = True
for dom in ("dmc-point-mass-maze", "dmc-walker", "dmc-cheetah"):
    f = glob.glob(os.environ["SCRATCH"] + f"/exorl-paperx-smoke/fb-{dom}/**/train.csv", recursive=True)
    if not f:
        print(f"  {dom}: no train.csv (train step failed)"); ok = False; continue
    txt = open(f[0], "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
    rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("fb_q_loss")]
    if not rows:
        print(f"  {dom}: fb_q_loss column ABSENT — q_loss_coef never engaged"); ok = False; continue
    v = float(rows[-1]["fb_q_loss"])
    print(f"  {dom}: fb_q_loss={v:.4f} implicit_reward={float(rows[-1]['fb_implicit_reward']):.4f}")
    if math.isnan(v): print(f"  {dom}: NaN"); ok = False
raise SystemExit(0 if ok else 1)
PYEOF

echo "== smoke failures: $fail =="
exit $fail
