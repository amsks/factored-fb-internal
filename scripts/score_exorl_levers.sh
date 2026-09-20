#!/usr/bin/env bash
# Score the single-lever sweep (scripts/launch_exorl_levers.sh; exp/exorl_levers/<lever>/, walker +
# cheetah, seed 0) under the paper protocol (10 eps, 50k reward samples, params_1000000, tdjepa task
# set) and print the domain mean per lever next to the 1M-recipe seed-0 baseline computed from
# exp/paper_eval_exorl_1mrecipe.jsonl. Mirrors scripts/score_exorl_1mrecipe.sh.
# Usage: bash scripts/score_exorl_levers.sh [lever ...]   (default: all 8; 2 scorer procs per lever, GPUs 0-7 round-robin)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; OUT=exp/paper_eval_exorl_levers.jsonl; mkdir -p logs/exorl_score
LEVERS=("$@"); [ ${#LEVERS[@]} -eq 0 ] && LEVERS=(qnorm sphere tower1024 tau001 headmean temp2 disc099 zdim100)
g=0
for lever in "${LEVERS[@]}"; do
  R=exp/exorl_disc_score_levers_$lever
  for dom in dmc-walker dmc-cheetah; do
    mkdir -p $R/$dom/density_fb_z
    run=${dom}_density_fb_z_ddpgbc_seed_0
    ln -sfn ../../../exorl_levers/$lever/$run $R/$dom/density_fb_z/$run
  done
  CUDA_VISIBLE_DEVICES=$((g % 8)) $PY scripts/exorl_paper_eval.py --root $R --critic density_fb_z --actor ddpgbc --domains dmc-walker dmc-cheetah --seeds 0 --steps 1000000 --episodes 10 --inference-samples 50000 --task-set tdjepa --jobs 2 --out $OUT > logs/exorl_score/levers_$lever.log 2>&1 &
  g=$((g+1))
done
wait
python3 - <<'PY'
import json, re, collections, statistics as st
def load(path, key):
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))   # group -> domain -> task -> return
    for line in open(path):
        r = json.loads(line)
        k = key(r)
        if k is not None: rows[k][r["domain"]][r["exorl_task"]] = r["episode_return"]
    return rows
# baseline: the 1M recipe, seed 0 only (the sweep's seed), same tdjepa tasks
base = load("exp/paper_eval_exorl_1mrecipe.jsonl", lambda r: "1mrecipe_s0" if r["seed"] == 0 else None)["1mrecipe_s0"]
lev = load("exp/paper_eval_exorl_levers.jsonl",
           lambda r: (m.group(1) if (m := re.search(r"exorl_disc_score_levers_([^/]+)/", r["restore_path"])) else None))
doms = ["dmc-walker", "dmc-cheetah"]
def dm(t): return st.mean(t.values()) if t else float("nan")
print(f"{'lever':12s} " + " ".join(f"{d[4:]:>10s}" for d in doms) + "   (baseline seed 0: " +
      ", ".join(f"{d[4:]} {dm(base[d]):.0f} n_tasks={len(base[d])}" for d in doms) + ")")
for L, per in sorted(lev.items()):
    cells = []
    for d in doms:
        t = per.get(d, {})
        cells.append(f"{dm(t):7.0f}({dm(t)-dm(base[d]):+4.0f})" if t else f"{'--':>10s}")
    print(f"{L:12s} " + " ".join(cells) + "  " + " ".join(f"{d[4:]}:{ {k: round(v) for k, v in sorted(per.get(d, {}).items())} }" for d in doms))
PY
