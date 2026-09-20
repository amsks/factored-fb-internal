#!/usr/bin/env bash
# Score the round-2 combo arms (scripts/launch_exorl_combo.sh; exp/exorl_combo/<arm>/) under the paper
# protocol (10 eps, 50k reward samples, params_1000000, tdjepa task set) and print, per arm and domain,
# n seeds / mean / pop std / per-seed list and the delta vs the 5-seed 1M-recipe batch mean computed
# from exp/paper_eval_exorl_1mrecipe.jsonl. Mirrors scripts/score_exorl_levers.sh.
# Usage: bash scripts/score_exorl_combo.sh [arm ...]   (default: all 6; one scorer per arm, GPUs 0-7 round-robin)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; OUT=exp/paper_eval_exorl_combo.jsonl; mkdir -p logs/exorl_score
# arm -> "<domains>|<seeds>" exactly as launched
declare -A CELLS=(
  [sq]="dmc-walker dmc-cheetah|0 1 2 3 4"
  [sq_tower1024]="dmc-walker dmc-cheetah|0"
  [sq_tau001]="dmc-cheetah|0"
  [sq_headmean]="dmc-cheetah|0"
  [sq_tower1024_tau001]="dmc-cheetah|0"
  [sq_tower1024_headmean]="dmc-cheetah|0"
)
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(sq sq_tower1024 sq_tau001 sq_headmean sq_tower1024_tau001 sq_tower1024_headmean)
g=0
for arm in "${ARMS[@]}"; do
  doms=${CELLS[$arm]%%|*}; seeds=${CELLS[$arm]##*|}
  R=exp/exorl_disc_score_combo_$arm
  for dom in $doms; do
    mkdir -p $R/$dom/density_fb_z
    for s in $seeds; do run=${dom}_density_fb_z_ddpgbc_seed_$s; ln -sfn ../../../exorl_combo/$arm/$run $R/$dom/density_fb_z/$run; done
  done
  CUDA_VISIBLE_DEVICES=$((g % 8)) $PY scripts/exorl_paper_eval.py --root $R --critic density_fb_z --actor ddpgbc --domains $doms --seeds $seeds --steps 1000000 --episodes 10 --inference-samples 50000 --task-set tdjepa --jobs 2 --out $OUT > logs/exorl_score/combo_$arm.log 2>&1 &
  g=$((g+1))
done
wait
python3 - <<'PY'
import json, re, collections, statistics as st
def load(path, key):
    rows = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))  # group -> domain -> seed -> task -> return
    for line in open(path):
        r = json.loads(line); k = key(r)
        if k is not None: rows[k][r["domain"]][r["seed"]][r["exorl_task"]] = r["episode_return"]
    return rows
base = load("exp/paper_eval_exorl_1mrecipe.jsonl", lambda r: "1mrecipe")["1mrecipe"]
combo = load("exp/paper_eval_exorl_combo.jsonl",
             lambda r: (m.group(1) if (m := re.search(r"exorl_disc_score_combo_([^/]+)/", r["restore_path"])) else None))
doms = ["dmc-walker", "dmc-cheetah"]
def seed_means(per_seed): return [st.mean(t.values()) for s, t in sorted(per_seed.items()) if t]
bm = {d: st.mean(seed_means(base[d])) for d in doms}
print("baseline 1M recipe (5 seeds): " + ", ".join(f"{d[4:]} {bm[d]:.1f} (n={len(seed_means(base[d]))})" for d in doms))
print(f"{'arm':24s} {'domain':8s} {'n':>2s} {'mean':>7s} {'popstd':>7s} {'delta':>7s}  per-seed")
for arm, per in sorted(combo.items()):
    for d in doms:
        m = seed_means(per.get(d, {}))
        if not m: continue
        print(f"{arm:24s} {d[4:]:8s} {len(m):2d} {st.mean(m):7.1f} {st.pstdev(m):7.1f} {st.mean(m)-bm[d]:+7.1f}  {[round(x) for x in m]}")
PY
