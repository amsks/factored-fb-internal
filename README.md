# Density-FB (ICLR 2027)

This branch holds only the code behind the ICLR 2027 Density-FB paper: the four critics it
compares, the two actors they are trained with, the environments, the scripts that train
every reported run, the scorer that turns their evaluation logs into the table cells, and
the figure scripts.

Everything is offline RL in JAX. A run composes one **critic** (the value representation)
with one **actor** (the policy) and trains both on a fixed dataset (`impls/main.py`).

- [Methods](#methods): what each critic learns and what its actor maximizes
- [Two ways to specify a task](#two-ways-to-specify-a-task-at-evaluation)
- [Code map](#code-map)
- [Reproducing the paper](#reproducing-the-paper)

## Methods

All four critics are forward-backward style: a forward network F reads the state, the
action and the task; a backward network B embeds the states the agent may reach. The task
is either a goal state g or a latent task vector z of width d = 50. In every method the
actor is trained against the critic with the critic frozen, and the TD target's next
action a' is drawn from the current actor at s'.

### Density-FB (ours)

Critic: `impls/critics/density_fb.py`.

**What the critic predicts.** For state s, action a and task g, it predicts a probability
distribution over where the agent will be in the future. The candidates are the M = 1024
next states u_1..u_M of the current batch:

    w(s, a, g, u_j) = softmax_j( F(s, a, g)^T B(u_j) )

There are two F heads and one B network. B reads only the task-relevant observation
columns, each divided by its success radius (for example the cube position divided by
0.04 m).

**TD target.** The critic is trained by cross-entropy toward

    t_j = (1 - gamma) * 1[u_j is this row's own next state]
          + gamma * w_target(s', a', g, u_j)

where w_target uses slowly updated target copies of F and B (rate 0.005). This is the
discounted state-occupancy recursion written over the batch's next states. There is no
orthonormality loss or other regularizer on F or B.

**Actor.** The actor maximizes the reward averaged under the predicted distribution:

    Q(s, a, g) = sum_j w(s, a, g, u_j) * r(u_j)

It takes the smaller of the two heads. The reward depends on the task interface:

- **Goal-conditioned** (`density_fb`, the table's Density-FB column). r(u_j) is a Gaussian
  "has the goal been reached" kernel around g at the environment's success radius,
  normalized over the candidates. It is exact at every candidate, so tasks that require
  several objects to be placed at once are scored correctly.
- **Reward inference** (`density_fb_z`: the ExoRL column and the in-text OGBench
  numbers). The task is a latent z, and r(u) = (B_slow(u) - mu)^T z / sqrt(d) is the linear
  reward z represents. B_slow is a slow copy of B (rate 5e-4), and mu is the running mean of
  its features.

**Inferring z from reward samples.** z is found by ridge regression of the rewards r_j on
the centered slow features at the sampled states u_j:

    m = mean_j (B_slow(u_j) - mu)(r_j - mean r),    c = (C + lam I)^-1 m,
    z = sqrt(d) c / ||c||

Here C is the running covariance of B_slow. This z does not change if a constant is added
to the reward or the reward is rescaled by a positive factor.

**Training z.** Each row's z is regressed from the goal-reaching reward of its hindsight
goal (a future state of the same trajectory). With probability 0.5 it is instead a
uniformly random direction; ExoRL walker and cheetah use probability 1. The TD target
bootstraps at the same task as the row, so a' is drawn for the row's own z.

**Ablation.** Replacing the averaged reward by FB's linear value F(s, a, z)^T z, with
z = B(g) for the goal-conditioned model, gives the actor-readout ablation. The critic is
unchanged.

### FB (Forward-Backward, Touati & Ollivier)

Critic: `impls/critics/fb.py`. Config `fb_tdj`, the TD-JEPA reference settings.

**Critic.** The product F(s, a, z)^T B(s') regresses the discounted successor density from
(s, a) to s' under the policy for z. It is fit on the batch matrix of all (row, next-state)
pairs toward the TD target

    1[s' = this row's next state] + gamma * F_target(s', a', z)^T B_target(s'')

The indicator term is the diagonal of that matrix and the bootstrapped term fills the
off-diagonal entries. An orthonormality penalty (coefficient 1000) keeps B's covariance
near the identity.

**Actor.** Maximizes Q(s, a, z) = F(s, a, z)^T z, averaged over the two F heads.

**z.** In training, half of the rows use B at a shuffled batch next-state and the rest a
random direction, all scaled to norm sqrt(d). From reward samples, z = sum_i r_i B(s_i),
rescaled to norm sqrt(d). For a goal, z = B(g).

### HILP (Hilbert representations, Park et al.)

Critic: `impls/critics/hilp.py`. Config `hilp_tdjc`, TD-JEPA's OGBench configuration.

Training has two stages, both computed from the same pre-step parameters.

1. **Representation.** A goal-conditioned value V(s, g) = -||phi(s) - phi(g)|| is fit by
   TD. The reward is -1 until the goal is reached and 0 at the goal, where the target
   does not bootstrap. Goals are geometric-future states of the same trajectory, or random
   states (probability 0.375). The regression is plain squared error.
2. **Successor features.** F(s, a, z) regresses the discounted sum of the centered features
   phi(s') - mean(phi). The TD target is phi(s') + gamma * F_target(s', a', z).

**Actor.** Maximizes F(s, a, z)^T z, averaged over heads.

**z.** In training, random directions, half replaced by whitened feature directions. From
reward samples, z is the least-squares fit of r to phi(s')^T z, normalized. For a goal,
z = normalize(phi(g) - phi(s)), recomputed at every step.

### RLDP (FB with a self-predictive B)

Critic: `impls/critics/rldp.py`.

**Critic.** F is trained exactly as in FB, but B is stop-gradient inside that TD loss, so
the TD error trains F only. B is trained instead by a 5-step latent prediction loss: a
predictor rolls B(s_t) forward along the dataset actions and is matched at each step to
the target B of the state actually reached. The orthonormality penalty (coefficient 1) is
applied at B(s_t).

**Actor and z.** Same as FB, except that z from reward samples is the mean (not the sum)
of r_i B(s_i); the normalization makes the two identical.

### Actors

- **Flow actor** (`impls/actors/flowbc.py`), used on OGBench, MetaWorld and ManiSkill. A
  flow-matching network models the dataset's actions. A one-step policy maps noise to an
  action. It maximizes Q / mean|Q| plus 3 times the squared distance to the flow's action
  for the same noise, which keeps it on the data.
- **DDPG-style actor** (`impls/actors/ddpgbc.py`), used on ExoRL. It is a deterministic
  Meta Motivo policy network that maximizes Q, optionally divided by mean|Q|. Clipped
  Gaussian noise (std 0.2, clip 0.3) is added to the action in the actor loss and to a'
  in the TD target. There is no behaviour-cloning term.

## Two ways to specify a task at evaluation

The two tables do not always use the same interface, so every number says which one it
used.

- **Goal-conditioned** (`eval.csv`). The environment's goal state is given to the agent.
  Density-FB's goal-conditioned model reads it directly. For the FB-family and HILP it is
  turned into z, and for Density-FB's reward-inference model it is regressed into z on a
  pool of states frozen at the first update. The OGBench Density-FB column comes from this
  file.
- **Reward inference** (`eval_relabel.csv` on OGBench; ExoRL scoring). No goal is given.
  Dataset transitions are relabelled with the task's reward. On OGBench the reward is
  the number of goal conditions met at the next state, minus the number required, plus 1,
  so it is 1 when the task is solved (10k transitions). On ExoRL it is the dm_control task
  reward re-simulated from the stored physics state (50k transitions).
  z is inferred from those samples and the policy is rolled on z. The FB, HILP and RLDP
  OGBench cells, the in-text Density-FB reward-inference numbers and every ExoRL cell use
  this route.

## Code map

| What | Where |
|---|---|
| Training loop, both evaluation routes | `impls/main.py` (reward-inference eval: the `eval_relabel_size` block) |
| Composing critic and actor, update order (critic stages, actor, targets, statistics) | `impls/agents/__init__.py` |
| Density-FB distribution, TD target, cross-entropy | `impls/critics/density_fb.py` `stage_loss` (TD target at lines 417-420) |
| Density-FB actor value (averaged reward, min over heads) | `density_fb.py` `q_fn` |
| Goal-reaching kernel reward | `density_fb.py` `_delta_phi` |
| z from reward samples (ridge regression) | `density_fb.py` `_z_centered_moment`, `_z_centered_coef`, `_z_index` |
| Training z (hindsight goals, random directions) | `density_fb.py` `sample_cond`, `_z_mix` |
| Slow B copy, running mean and covariance, frozen evaluation pool | `density_fb.py` `update_stats` |
| FB successor loss, orthonormality, z | `impls/critics/fb.py` `stage_loss`, `sample_cond`, `infer_cond` |
| HILP value loss and successor features | `impls/critics/hilp.py` `hilp_phi_loss`, `stage_loss` |
| RLDP multi-step prediction loss | `impls/critics/rldp.py` `_sp_loss` |
| Flow actor / DDPG-style actor | `impls/actors/flowbc.py` / `impls/actors/ddpgbc.py` (+ `_motivo_nets.py`) |
| Network definitions | `impls/utils/networks.py` (FB, HILP, RLDP, flow actor); Density-FB builds its own in `density_fb.py` |
| Goal sampling (hindsight / random goals) | `impls/utils/datasets.py` `GCDataset`; RLDP windows `WindowedDataset`; HILP goals `HILPFutureDataset` |
| Hit spec: task columns and success radii | `impls/critics/_measure_common.py` `parse_hit_spec` |
| OGBench envs and reward relabelling | `env/ogbench.py` `relabel_infer_batch` |
| ExoRL datasets and reward re-simulation | `env/exorl_env.py` `load_domain_data`, `relabel_rewards` |
| MetaWorld / ManiSkill | `env/unified.py`, `env/metaworld_env.py`, `env/maniskill_env.py` |
| Configs | `impls/configs/critic/*.yaml`, `impls/configs/actor/*.yaml`, `impls/configs/domain/`, `env/domains/` |

Critic names on the command line: `density_fb`, `density_fb_z`, `density_fb_state` (the
backward network reads the full state), `fb_tdj` and `fb_ref` (identical FB configs; both
names appear in run directories), `hilp_tdjc` and `rldp`. A yaml may start with
`base: <name>` to inherit another.

## Reproducing the paper

### Setup

```bash
uv venv --python 3.11 .venv && uv pip install -e '.[exorl,figures,dev]'   # see pyproject.toml extras
```

If the installed `mujoco` is too old for `dm_control`'s reward re-simulation, point
`FB_MUJOCO_SITE_PACKAGES` at a newer one (see `env/exorl_env.py`).

### Datasets

**OGBench** (cube-single, cube-double, scene play data). The `ogbench` package downloads
each dataset to `~/.ogbench/data/` on first use; nothing to do by hand.

**ExoRL** (walker, quadruped, cheetah, point-mass maze; RND exploration data). Use the
download script of the ExoRL repository; each domain lands in
`<root>/<domain>/rnd/buffer/episode_*.npz`, the layout `env/exorl_env.py` reads.

```bash
git clone https://github.com/denisyarats/exorl && cd exorl
for d in walker quadruped cheetah point_mass_maze; do ./download.sh $d rnd; done
export EXORL_DATA_ROOT=$PWD/datasets              # default: ~/exorl_data/exorl/datasets
export EXORL_CACHE_ROOT=~/exorl_data/relabel_cache   # where relabelled task rewards are cached
```

**MetaWorld and ManiSkill** (packed offline datasets, one `.npz` per task). They are in the
gated Hugging Face dataset `TommyShen/Factor-FB-Offline`: request access on its page, then
log in (`huggingface-cli login`). The launchers read `$FB_DATA_ROOT/data/<metaworld|maniskill>/<task>.npz`.

```bash
huggingface-cli download TommyShen/Factor-FB-Offline --repo-type dataset \
    --include 'manifest.json' 'data/metaworld/*' 'data/maniskill/*' --local-dir Factor-FB-Offline
export FB_DATA_ROOT=$PWD/Factor-FB-Offline
```

The runs behind the paper's MetaWorld and ManiSkill cells were made on another machine, and
which version of these files they used is not recorded here.

**Other datasets used during the project** (not needed to reproduce the paper):

```bash
huggingface-cli download TommyShen/Factor-FB-Offline-tdmpc2 --local-dir Factor-FB-Offline-tdmpc2   # model repo
huggingface-cli download TommyShen/factored-fb-play --repo-type dataset --revision 2af2725 --local-dir factored-fb-play
huggingface-cli download amsks/factoredfb-metaworld-play --repo-type dataset --revision 137ecb8 --local-dir factoredfb-metaworld-play
```

The two play datasets have grown since these revisions (to about 188 GB and 756 MB); the
revisions pin the versions used on the training box.

Run directories, evaluation CSVs and checkpoints for every reported run are on the Hugging
Face repo `amsks/factored-fb-checkpoints`, branch `iclr2027`. `paper_inventory.json` there
maps every table and figure to its runs.

### Tables from the evaluation logs

```bash
huggingface-cli download amsks/factored-fb-checkpoints --revision iclr2027 \
    --include 'exp/**/eval*.csv' 'exp/paper_eval_*.jsonl' --local-dir hf
python scripts/score_tables.py --exp hf/exp                           # every cell, per seed
python scripts/score_tables.py --exp hf/exp --tex experiments.tex     # diff against the paper
python scripts/score_tables.py --exp hf/exp --tex experiments.tex --write
```

The last command rewrites the benchmark table body: cells, benchmark means and the Δ
column.

- **OGBench cell:** per run, the task-mean success averaged over the last three
  evaluations (800k, 900k, 1M steps); then mean ± SE (sample sd / √n) over 5 seeds.
- **ExoRL cell:** per seed, the task-mean return at 1M steps (10 episodes, 50k reward
  samples); then mean ± SE (population sd / √n) over 5 seeds.
- **Published numbers:** GCIQL, CRL and QRL (OGBench) and FB, HILP and RLDP (ExoRL) are
  published numbers read from the tex.
- **Not re-derived here:** the MetaWorld and ManiSkill cells were produced on other
  machines; their runs are not in this repository or on the HF branch.

### Training

```bash
bash scripts/launch_ogbench.sh density_fb cube_double 0     # methods: fb hilp rldp density_fb density_fb_z ...
bash scripts/launch_exorl.sh walker 0                       # walker quadruped point-mass cheetah
python -m scripts.exorl_paper_eval --root exp/exorl --domains dmc-walker \
    --out exp/paper_eval_exorl_sqt.jsonl                    # ExoRL table evaluation
bash scripts/launch_mwms_baselines.sh / launch_mwms_gc.sh   # MetaWorld / ManiSkill
```

`DRY=1` prints a launcher's command. Each launcher reproduces the config recorded in the
reported runs' `config.json`: `launch_ogbench.sh` and `launch_exorl.sh` were checked
against 26 of them. The ExoRL jsonl names that `score_tables.py` reads are:

- walker: `paper_eval_exorl_sqt.jsonl`
- quadruped: `paper_eval_exorl_kernels_quadruped_vu.jsonl`
- point-mass: `paper_eval_exorl_1mrecipe.jsonl`
- cheetah: `paper_eval_exorl_cheetah3_velblock.jsonl`

### Figures

| Figure | Script | Inputs |
|---|---|---|
| Lead figure (cube occupancy, FB vs Density-FB) | `scripts/probes/cube_bottleneck/fig_lead_combined.py` (+ `fig_drop_strip.py`, `fig_carry_strip_v2.py`) | `fb_ref` and `density_fb_state` cube-single seed 0, `params_500000.pkl` |
| Cube-single failure analysis | `scripts/probes/cube_bottleneck/failfig_pipeline.sh`, `fig_phase_strip.py`, `fig_alignment_main.py` | `fb_tdj` and `density_fb_z` cube-single seeds 0-2 at 1M |
| Nine-state MDP diagram | `scripts/didactic/plot_single_bottleneck_mdp.py` | none |
| Bottleneck panels, feature-dimension ablation | `SB_PROCESS=p9 python analysis/single_bottleneck/paper_figs_61.py` | `analysis/single_bottleneck/paper_figs_61_p9_cache.npz` (`--recompute` retrains, CPU) |
| Policy family table and graphs | `analysis/single_bottleneck/family9.py`, `fb_failure_visual9.py` | same cache |
| Lemma check | `analysis/single_bottleneck/check_lemma.py` | none (CPU) |

Figures are written under `figures/`.

### Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests -q
```
