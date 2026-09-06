#!/usr/bin/env bash
# Fill the goal-conditioned cube table to 3 seeds x 500k (PSM arms excluded per 2026-09-04).
#
#   bash scripts/launch_cube_table_seeds.sh            # all 8 GPUs
#   GPUS="0 1 2 3" bash scripts/launch_cube_table_seeds.sh
#   DRY=1 bash scripts/launch_cube_table_seeds.sh      # print the job list and exit
#
# IDEMPOTENT BY CONSTRUCTION. The job list is recomputed on every invocation by scanning
# exp/ for params_500000.pkl, and every job runs with --auto_resume. So: re-running after a
# crash, a walltime kill, or a GPU fault relaunches only what is still missing, and a job
# that died at 300k continues from its last checkpoint rather than restarting. Nothing that
# already has a 500k checkpoint is ever touched, which is what keeps the existing table
# cells byte-stable.
#
# OVERRIDES are not guessed: the rule below is read off the config.json of the runs already
# on disk. cube-single takes none; cube-double takes the two-block hit spec for every arm
# whose kernel slices object coordinates (density_* and *_oc), and none for the plain
# full-state arms (fb, fb_ref, hilp_cube, rldp). `rldp_density` on cube-double has never
# been run at all -- it takes the hit spec by analogy with `density_fb_state`, which is the
# same shape of arm (state-input backward tower, object-coordinate kernel).
#
# CUDA GRAPHS OFF (`--xla_gpu_enable_command_buffer=`): CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
# has taken a cell in ~25% of waves historically and killed cd-fb_oc at 465k.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/cubeseed_logs
SESSION="${SESSION:-cubeseed}"

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000        # 250k AND 500k on disk -> a 1M continuation can resume from either

read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"
cd "$REPO"

# ── the job list, recomputed from disk ──────────────────────────────────────────────────
JOBS=$("$PY" - <<'PY'
import re, pathlib, collections
ARMS = ["density_fb_state","density_fb","rldp_density","rldp_density_oc",
        "hilp_density_cube","hilp_density_oc","fb","fb_oc","rldp","rldp_oc",
        "hilp_cube","hilp_oc","fb_ref"]
ENVS = {"cube-single-play-v0":"cube_single", "cube-double-play-v0":"cube_double"}
CD_HIT = "measure_hit_spec=19:22:0.4,28:31:0.4"
have = collections.defaultdict(dict)
for d in pathlib.Path("exp").glob("cube-*play-v0_*_flowbc_seed_*"):
    m = re.match(r"(cube-\w+-play-v0)_(.+)_flowbc_seed_(\d+)$", d.name)
    if not m:
        continue
    eps = [int(p.stem.split("_")[1]) for p in d.glob("params_*.pkl")]
    if eps:
        have[(m.group(1), m.group(2))][int(m.group(3))] = max(eps)
for env, dom in ENVS.items():
    for critic in ARMS:
        done = {s for s, e in have.get((env, critic), {}).items() if e >= 500000}
        for seed in (0, 1, 2):
            if seed in done:
                continue
            hit = dom == "cube_double" and (critic.startswith(("density_", "rldp_density",
                                                               "hilp_density", "psm_density"))
                                            or critic.endswith("_oc"))
            print(f"{critic} {dom} {seed} {CD_HIT if hit else '-'}")
PY
)
N=$(printf '%s\n' "$JOBS" | grep -c . || true)
echo "$N jobs to run across ${#GPU_LIST[@]} GPUs:"
printf '%s\n' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing missing - table is already 3x500k"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0

tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }

# ── one worker per GPU, each draining its own slice sequentially ────────────────────────
NG=${#GPU_LIST[@]}
first=1
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"
  slice=$(printf '%s\n' "$JOBS" | awk -v w="$w" -v n="$NG" 'NF && (NR-1)%n==w')
  [ -n "$slice" ] || continue
  body="cd $REPO; "
  while read -r critic dom seed ov; do
    [ -n "${critic:-}" ] || continue
    ovflag=""; [ "$ov" = "-" ] || ovflag="--override $ov"
    tag="${dom}-${critic}-s${seed}"
    body+="echo '>>> $tag'; \
CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
$PY -m impls.main --critic $critic --actor flowbc --domain $dom --seed $seed \
  --auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY \
  --save_interval $SAVE_EVERY $ovflag 2>&1 | tee $LOGDIR/$tag.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s\n' "$slice" | grep -c .) jobs"
done
echo "launched session '$SESSION'"
