# Density-FB

Code for the ICLR 2027 paper on Density-FB.

## Overview

Density-FB is a forward-backward critic for offline RL whose successor measure is a
distribution over the batch's next states, trained by cross-entropy to a TD target.
A task is given either as a goal state or as reward samples, from which a latent task
vector z is inferred. The repository also contains our FB, HILP and RLDP baselines and
the scripts behind every table and figure in the paper.

## Installation

The code is based on JAX and tested with Python 3.11.

```bash
uv venv --python 3.11 .venv && uv pip install -e '.[exorl,figures,dev]'
```

Datasets: OGBench downloads itself on first use; ExoRL, MetaWorld and ManiSkill are
described [below](#data).

## Usage

The main implementation of Density-FB is in [impls/critics/density_fb.py](impls/critics/density_fb.py),
and the baselines (FB, HILP and RLDP) are in the same directory. Every run pairs a critic
with an actor (`flowbc` or `ddpgbc`) and a domain:

```bash
# Density-FB on OGBench cube-single, task given as a goal state
python -m impls.main --critic density_fb --actor flowbc --domain cube_single --override 'goal_spec="19:22:0.4"'
# Density-FB with the task inferred from reward samples
python -m impls.main --critic density_fb_z --actor flowbc --domain cube_single --override 'goal_spec="19:22:0.4"' --override eval_relabel_size=10000
# Baselines
python -m impls.main --critic fb --actor flowbc --domain cube_double
python -m impls.main --critic hilp --actor flowbc --domain cube_double
python -m impls.main --critic rldp --actor flowbc --domain cube_double
```

`goal_spec` names the goal observation columns and the success radius
(`start:end:radius`, one block per object). Configs live in `impls/configs/` and
`env/domains/`; any value can be changed with `--override key=value`. Run configs saved
under older key or critic names still load.

## Reproducing the main results

The launchers reproduce the exact configurations of the reported runs (`DRY=1` prints
the command):

```bash
bash scripts/launch_ogbench.sh density_fb_z cube_double 0   # method, domain, seed
bash scripts/launch_exorl.sh walker 0                       # walker, quadruped, point-mass, cheetah
```

Logs and checkpoints of every reported run are on Hugging Face
(`amsks/factored-fb-checkpoints`, branch `iclr2027`). To rebuild the tables from them:

```bash
huggingface-cli download amsks/factored-fb-checkpoints --revision iclr2027 \
    --include 'exp/**/eval*.csv' 'exp/paper_eval_*.jsonl' --local-dir hf
python scripts/score_tables.py --exp hf/exp
```

<details>
<summary><b>All launchers, scoring rules and figure scripts</b></summary>

```bash
# OGBench methods: fb fb_500k hilp rldp density_fb density_fb_z density_fb_state
#                  density_fb_linear density_fb_z_linear density_fb_500k density_fb_z_500k
bash scripts/launch_ogbench.sh <method> <cube_single|cube_double|scene_play> <seed>
# ExoRL evaluation of trained runs
python -m scripts.exorl_paper_eval --root exp/exorl --domains dmc-walker --out exp/paper_eval_exorl_sqt.jsonl
# MetaWorld / ManiSkill
FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh   # FB, HILP, RLDP
FB_DATA_ROOT=... bash scripts/launch_mwms_gc.sh          # Density-FB
# Compare the scored cells with the paper's table, or rewrite it
python scripts/score_tables.py --exp hf/exp --tex experiments.tex [--write]
```

An OGBench cell is the task-mean success averaged over the evaluations at 800k, 900k and
1M steps, then mean and standard error over 5 seeds. With the task given as reward
samples, rollouts keep OGBench's single-task settings (no cube permutation at reset).
An ExoRL cell is the task-mean return at 1M steps. GCIQL, CRL and QRL (OGBench) and FB,
HILP and RLDP (ExoRL) are published numbers. The ExoRL evaluation files read by the
scorer are `paper_eval_exorl_sqt.jsonl` (walker), `paper_eval_exorl_kernels_quadruped_vu.jsonl`
(quadruped), `paper_eval_exorl_1mrecipe.jsonl` (point-mass) and
`paper_eval_exorl_cheetah3_velblock.jsonl` (cheetah). The paper's run directories use
the older critic names `fb_ref` and `fb_tdj` for FB.

```bash
python scripts/probes/cube_bottleneck/fig_lead_combined.py     # lead figure
bash scripts/probes/cube_bottleneck/failfig_pipeline.sh        # cube-single failure analysis
python scripts/didactic/plot_single_bottleneck_mdp.py          # nine-state MDP diagram
SB_PROCESS=p9 python analysis/single_bottleneck/paper_figs_61.py
python analysis/single_bottleneck/family9.py
python analysis/single_bottleneck/check_lemma.py
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests -q       # tests
```

</details>

## Data

**ExoRL** (RND exploration data):

```bash
git clone https://github.com/denisyarats/exorl && cd exorl
for d in walker quadruped cheetah point_mass_maze; do ./download.sh $d rnd; done
export EXORL_DATA_ROOT=$PWD/datasets
```

If the installed `mujoco` cannot compile `dm_control` models, set `FB_MUJOCO_SITE_PACKAGES`
to a site-packages with a matching version.

**MetaWorld and ManiSkill** are in the gated dataset `TommyShen/Factor-FB-Offline`
(request access, then `huggingface-cli login`):

```bash
huggingface-cli download TommyShen/Factor-FB-Offline --repo-type dataset \
    --include 'manifest.json' 'data/metaworld/*' 'data/maniskill/*' --local-dir Factor-FB-Offline
export FB_DATA_ROOT=$PWD/Factor-FB-Offline
```

## Acknowledgments

This codebase is built on [OGBench](https://github.com/seohongpark/ogbench)'s reference
implementations. The FB, HILP and RLDP baselines follow the
[TD-JEPA](https://github.com/facebookresearch/td_jepa) implementations and settings.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
