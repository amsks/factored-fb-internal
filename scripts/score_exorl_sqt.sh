#!/usr/bin/env bash
# Score the `sqt` arm (scripts/launch_exorl_sqt.sh; exp/exorl_sqt/<domain>/, 4 domains x 5 seeds) under
# the paper protocol (10 eps, 50k reward samples, params_1000000; tdjepa task set, paper for maze) and
# print per-domain n / mean / pop std / per-seed and the delta vs the 5-seed 1M-recipe batch mean
# computed from exp/paper_eval_exorl_1mrecipe.jsonl. Mirrors scripts/score_exorl_1mrecipe.sh.
# Usage: bash scripts/score_exorl_sqt.sh   (4 scorer jobs on GPUs 0-3; idempotent symlink layout)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; R=exp/exorl_disc_score_sqt; OUT=exp/paper_eval_exorl_sqt.jsonl; mkdir -p logs/exorl_score
g=0
for dom in dmc-walker dmc-cheetah dmc-quadruped dmc-point-mass-maze; do
  mkdir -p $R/$dom/density_fb_z
  for s in 0 1 2 3 4; do run=${dom}_density_fb_z_ddpgbc_seed_$s; ln -sfn ../../../exorl_sqt/$dom/$run $R/$dom/density_fb_z/$run; done
  TS=tdjepa; [ "$dom" = dmc-point-mass-maze ] && TS=paper
  CUDA_VISIBLE_DEVICES=$g $PY scripts/exorl_paper_eval.py --root $R --critic density_fb_z --actor ddpgbc --domains $dom --seeds 0 1 2 3 4 --steps 1000000 --episodes 10 --inference-samples 50000 --task-set $TS --jobs 2 --out $OUT > logs/exorl_score/sqt_$dom.log 2>&1 &
  g=$((g+1))
done
wait
python3 - <<'PY'
import json, collections, statistics as st
def load(path):
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))   # domain -> seed -> task -> return
    for line in open(path):
        r = json.loads(line); rows[r["domain"]][r["seed"]][r["exorl_task"]] = r["episode_return"]
    return rows
def seed_means(per_seed): return [st.mean(t.values()) for s, t in sorted(per_seed.items()) if t]
base = load("exp/paper_eval_exorl_1mrecipe.jsonl"); sqt = load("exp/paper_eval_exorl_sqt.jsonl")
print(f"{'domain':22s} {'n':>2s} {'mean':>7s} {'popstd':>7s} {'base':>7s} {'delta':>7s}  per-seed / per-task")
for d in ["dmc-walker", "dmc-cheetah", "dmc-quadruped", "dmc-point-mass-maze"]:
    m = seed_means(sqt.get(d, {})); bm = st.mean(seed_means(base[d]))
    if not m: print(f"{d:22s}  0 {'--':>7s} {'--':>7s} {bm:7.1f}"); continue
    tasks = sorted({t for s in sqt[d] for t in sqt[d][s]})
    pt = {t: round(st.mean(sqt[d][s][t] for s in sqt[d] if t in sqt[d][s])) for t in tasks}
    print(f"{d:22s} {len(m):2d} {st.mean(m):7.1f} {st.pstdev(m):7.1f} {bm:7.1f} {st.mean(m)-bm:+7.1f}  {[round(x) for x in m]} {pt}")
PY
