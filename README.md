# Density-FB (ICLR 2027)

This repository holds the code behind the ICLR 2027 Density-FB paper.
It is offline RL in JAX: a run pairs one critic (the value representation) with one actor (the policy) and trains both on a fixed dataset.
It also contains the launchers for every reported run, the scorer that turns evaluation logs into table cells, and the figure scripts.

Contents:
- Four critics: Density-FB (ours), FB, HILP and RLDP
- Two actors: a flow-matching actor (OGBench, MetaWorld, ManiSkill) and a DDPG-style actor (ExoRL)
- Environments: OGBench, ExoRL (dm_control), MetaWorld and ManiSkill
- Launchers, the table scorer, figure scripts and tests

## Methods

All four critics use a forward network F(s, a, task) and a backward network B that embeds the states the agent may reach.
The task is a goal state g or a latent task vector z of width 50.
The actor is trained against the frozen critic, and the next action a' in each TD target comes from the current actor.

- **Density-FB** (`impls/critics/density_fb.py`). The critic predicts a probability distribution over which of the batch's next states the agent will visit in the future: `w(s, a, g, u_j) = softmax_j( F(s, a, g)^T B(u_j) )`. Its TD target puts weight (1 - gamma) on the row's own next state plus gamma times the target network's distribution at (s', a'), and it is trained by cross-entropy. The actor maximizes the reward averaged under that distribution. For a goal, the reward is a "goal reached" kernel over the goal observation columns and success radius; for z, it is the linear reward z represents.
- **FB** (`impls/critics/fb.py`, config `fb`). F(s, a, z)^T B(s') regresses the discounted successor density, with TD target "1 at the row's own next state, plus gamma times the target F^T B at s''", and an orthonormality penalty on B. The actor maximizes F(s, a, z)^T z.
- **HILP** (`impls/critics/hilp.py`, config `hilp`). A goal-conditioned value V(s, g) = -||phi(s) - phi(g)|| is fit by TD to learn features phi. Successor features F(s, a, z) then regress the discounted sum of centered phi, with TD target phi(s') + gamma F_target(s', a', z). The actor maximizes F(s, a, z)^T z.
- **RLDP** (`impls/critics/rldp.py`, config `rldp`). F is trained as in FB with B held fixed in that loss. B is trained by a 5-step latent prediction loss along the dataset actions. The actor and z are as in FB.

There are two ways to give the agent a task at evaluation.
Goal-conditioned evaluation (`eval.csv`) gives the environment's goal state; reward-inference evaluation (`eval_relabel.csv` on OGBench, and all ExoRL scoring) gives only dataset transitions labelled with the task reward, from which z is inferred and the policy is rolled out on z.

## Installation

The code is tested with Python 3.11. Install with uv:

```bash
uv venv --python 3.11 .venv && uv pip install -e '.[exorl,figures,dev]'   # see pyproject.toml extras
```

If the installed `mujoco` is too old for `dm_control`'s reward re-simulation, point `FB_MUJOCO_SITE_PACKAGES` at a newer one (see `env/exorl_env.py`).

## Running

Every run goes through `impls/main.py` with a critic, an actor and a domain.
Critic names are `density_fb`, `density_fb_z`, `density_fb_state` (B reads the full state), `fb`, `hilp` and `rldp`.
Config values can be changed with `--override key=value`.

```bash
# Density-FB, goal-conditioned, on OGBench cube-single
python -m impls.main --critic density_fb --actor flowbc --domain cube_single --seed 0 \
    --override 'goal_spec="19:22:0.4"'
# Density-FB with reward inference
python -m impls.main --critic density_fb_z --actor flowbc --domain cube_single --seed 0 \
    --override 'goal_spec="19:22:0.4"' --override eval_relabel_size=10000
# FB, HILP and RLDP
python -m impls.main --critic fb   --actor flowbc --domain cube_double --seed 0
python -m impls.main --critic hilp --actor flowbc --domain cube_double --seed 0
python -m impls.main --critic rldp      --actor flowbc --domain cube_double --seed 0
```

`goal_spec` lists the goal observation columns and success radius as `start:end:radius`, comma separated per object.
A critic yaml may start with `base: <name>` to inherit another.
Configs and critic names saved by older runs (such as `measure_hit_spec`, or the critic `fb_tdj`) are still accepted and renamed on load (`impls/utils/config_utils.py`).

## Reproducing the paper

Each launcher reproduces the config recorded in the reported runs' `config.json`. `DRY=1` prints the command instead of running it.

```bash
bash scripts/launch_ogbench.sh density_fb cube_double 0     # methods: fb hilp rldp density_fb density_fb_z ...
bash scripts/launch_exorl.sh walker 0                       # walker quadruped point-mass cheetah
python -m scripts.exorl_paper_eval --root exp/exorl --domains dmc-walker \
    --out exp/paper_eval_exorl_sqt.jsonl                    # ExoRL table evaluation
FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh       # FB, HILP, RLDP on MetaWorld / ManiSkill
FB_DATA_ROOT=... bash scripts/launch_mwms_gc.sh              # Density-FB on MetaWorld / ManiSkill
```

The ExoRL evaluation files that the scorer reads are `paper_eval_exorl_sqt.jsonl` (walker), `paper_eval_exorl_kernels_quadruped_vu.jsonl` (quadruped), `paper_eval_exorl_1mrecipe.jsonl` (point-mass) and `paper_eval_exorl_cheetah3_velblock.jsonl` (cheetah).

Run directories, evaluation CSVs and checkpoints for every reported run are on the Hugging Face repo `amsks/factored-fb-checkpoints`, branch `iclr2027`. Its `paper_inventory.json` maps every table and figure to its runs. To rebuild the tables from those logs:

```bash
huggingface-cli download amsks/factored-fb-checkpoints --revision iclr2027 \
    --include 'exp/**/eval*.csv' 'exp/paper_eval_*.jsonl' --local-dir hf
python scripts/score_tables.py --exp hf/exp                           # every cell, per seed
python scripts/score_tables.py --exp hf/exp --tex experiments.tex     # diff against the paper
python scripts/score_tables.py --exp hf/exp --tex experiments.tex --write   # rewrite the table body
```

An OGBench cell is the task-mean success averaged over the evaluations at 800k, 900k and 1M steps, then mean and standard error over 5 seeds.
An ExoRL cell is the task-mean return at 1M steps (10 episodes, 50k reward samples), then mean and standard error over 5 seeds.
GCIQL, CRL and QRL (OGBench) and FB, HILP and RLDP (ExoRL) are published numbers read from the tex.
The MetaWorld and ManiSkill cells were produced on other machines and are not re-derived here.

Figures are written under `figures/`:

```bash
python scripts/probes/cube_bottleneck/fig_lead_combined.py      # lead figure (also fig_drop_strip.py, fig_carry_strip_v2.py)
bash scripts/probes/cube_bottleneck/failfig_pipeline.sh         # cube-single failure analysis (also fig_phase_strip.py, fig_alignment_main.py)
python scripts/didactic/plot_single_bottleneck_mdp.py           # nine-state MDP diagram
SB_PROCESS=p9 python analysis/single_bottleneck/paper_figs_61.py   # bottleneck panels, feature-dimension ablation (--recompute retrains on CPU)
python analysis/single_bottleneck/family9.py                    # policy family table (also fb_failure_visual9.py)
python analysis/single_bottleneck/check_lemma.py                # lemma check (CPU)
```

The lead figure uses FB and `density_fb_state` on cube-single, seed 0, at 500k steps. The failure analysis uses FB and `density_fb_z` on cube-single, seeds 0-2, at 1M steps. The paper's run directories carry the older critic names `fb_ref` and `fb_tdj`.

Run the tests with:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests -q
```

## Code layout

- `impls/main.py`: training loop and both evaluation modes (reward inference is the `eval_relabel_size` block)
- `impls/agents/`: pairs a critic with an actor and sets the update order
- `impls/critics/`: Density-FB, FB, HILP and RLDP; `goal_spec.py` parses the goal columns and success radii
- `impls/actors/`: flow actor (`flowbc.py`) and DDPG-style actor (`ddpgbc.py`, `_motivo_nets.py`)
- `impls/utils/`: networks, datasets and goal sampling, config loading, evaluation
- `impls/configs/`: critic, actor and OGBench domain yamls
- `env/`: OGBench (with reward relabelling), ExoRL, MetaWorld and ManiSkill; domain yamls in `env/domains/`
- `scripts/`: launchers, the scorer, ExoRL evaluation and figure scripts
- `analysis/`: small tabular MDPs for the theory figures
- `tests/`: unit tests

## Data

**OGBench** (cube-single, cube-double, scene play data). The `ogbench` package downloads each dataset to `~/.ogbench/data/` on first use.

**ExoRL** (walker, quadruped, cheetah, point-mass maze; RND exploration data). Each domain lands in `<root>/<domain>/rnd/buffer/episode_*.npz`, the layout `env/exorl_env.py` reads.

```bash
git clone https://github.com/denisyarats/exorl && cd exorl
for d in walker quadruped cheetah point_mass_maze; do ./download.sh $d rnd; done
export EXORL_DATA_ROOT=$PWD/datasets                 # default: ~/exorl_data/exorl/datasets
export EXORL_CACHE_ROOT=~/exorl_data/relabel_cache   # where relabelled task rewards are cached
```

**MetaWorld and ManiSkill** (one `.npz` per task). They are in the gated Hugging Face dataset `TommyShen/Factor-FB-Offline`. Request access on its page, then log in with `huggingface-cli login`. The launchers read `$FB_DATA_ROOT/data/<metaworld|maniskill>/<task>.npz`.

```bash
huggingface-cli download TommyShen/Factor-FB-Offline --repo-type dataset \
    --include 'manifest.json' 'data/metaworld/*' 'data/maniskill/*' --local-dir Factor-FB-Offline
export FB_DATA_ROOT=$PWD/Factor-FB-Offline
```

The paper's MetaWorld and ManiSkill runs were made on another machine, and which version of these files they used is not recorded here.

**Other datasets used during the project** (not needed to reproduce the paper). The two play datasets have grown since these revisions; the revisions pin the versions used for training.

```bash
huggingface-cli download TommyShen/Factor-FB-Offline-tdmpc2 --local-dir Factor-FB-Offline-tdmpc2   # model repo
huggingface-cli download TommyShen/factored-fb-play --repo-type dataset --revision 2af2725 --local-dir factored-fb-play
huggingface-cli download amsks/factoredfb-metaworld-play --repo-type dataset --revision 137ecb8 --local-dir factoredfb-metaworld-play
```
