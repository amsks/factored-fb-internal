# ExoRL pipeline audit (dataset / relabelling / task inference / evaluation) vs the official protocols

Date 2026-09-17. Read-only audit of `/home/adityamohan/git/Factored-FB-seam` against
(a) Touati–Rapin–Ollivier official code `/home/adityamohan/git/baselines/RLU/controllable_agent/url_benchmark`,
(b) TD-JEPA `/home/adityamohan/git/baselines/td_jepa` (metamotivo),
(c) RLDP `/home/adityamohan/git/baselines/rldp`, (d) HILP `/home/adityamohan/git/baselines/HILP/hilp_zsrl`.
All numbers below were measured on this box (small env runs from the worktree); nothing in the repo was edited.

## Headline

The environment / data / relabelling / evaluation pipeline is faithful to the official protocol on
every point that could depress returns. Transition alignment, reward alignment, observation layout,
episode length, action repeat, observation scaling, the reward column used for task inference, the
deterministic policy at eval, and the checkpoint scored all match the references. The deviations that
exist (listed in section 6) are second-order: a 5% validation hold-out (4750 instead of 5000 training
episodes), a different number of reward-labelled samples for the task vector (50k where TD-JEPA uses
10k and the official FB code 5120 -- more samples, not fewer), fixed per-episode reset seeds, and the
point-mass-maze reward being the ExoRL/FB/HILP one rather than TD-JEPA's 6x-wider one. None of them
explains a low cheetah or quadruped return. The low cheetah-run number (60 at 2M steps) is a learner
result, not a pipeline artefact: the dataset itself averages 50 per episode on that task and a random
policy 3.7, and the same checkpoints and scorer give walker stand/walk/spin 715/641/976.

## 1. Dataset

### What is on disk
`/home/adityamohan/exorl_data/exorl/datasets/{walker,cheetah,quadruped,point_mass_maze}/rnd/buffer/`
holds 10 000 files per domain, `episode_000000_1000.npz` .. `episode_009999_1000.npz`. Each file
(measured on `episode_000000_1000.npz`):

| domain | observation | action | reward | discount | physics |
|---|---|---|---|---|---|
| walker | (1001, 24) f32 | (1001, 6) | (1001, 1) | (1001, 1) | (1001, 18) f64 |
| cheetah | (1001, 17) f32 | (1001, 6) | (1001, 1) | (1001, 1) | (1001, 18) f64 |
| quadruped | (1001, 78) f32 | (1001, 12) | (1001, 1) | (1001, 1) | (1001, 57) f64 |
| point_mass_maze | (1001, 4) f32 | (1001, 2) | (1001, 4) | (1001, 1) | (1001, 4) f64 |

`action[0]` is all zeros and `reward[0]` is 0 in every domain (the dummy first row); discount is 1
throughout. Observation dims 24/17/78/4 are the state ("proprio") observations the references use
(RLDP paper Table 4: 24/6, 17/6, 78/12, 4/2). The stored `reward` column is the reward of the task the
RND collector was run under (walker: `stand`, cheetah: `run`, quadruped: `walk` -- see section 2;
point_mass_maze: numerically zero, ~1e-40).

### How many episodes we load vs the references
- Ours: `exorl_max_episodes: 5000` in every dmc domain yaml (`env/domains/dmc-walker.yaml:37`,
  `env/domains/dmc-cheetah.yaml:41`), applied as the first 5000 files sorted by episode index
  (`env/exorl_env.py:368-372`). Then `env/unified.py:334` holds out the last 5% of rows as a validation
  set (`n_val = int(n * cfg.get("val_frac", 0.05))`), so training and task inference see 4 750 000
  transitions (4750 episodes), not 5 000 000. The run configs carry no `val_frac` key
  (`exp/exorl_dfb/dmc-walker/density_fb_z/dmc-walker_density_fb_z_ddpgbc_seed_0/config.json`), so the
  default 0.05 applied.
- Official FB code: `replay_buffer_episodes: int = 5000` (`url_benchmark/pretrain.py:83`), loaded as
  `sorted(replay_dir.glob('*.npz'))[:5000]` (`in_memory_replay_buffer.py:195-210`) = 5M transitions.
- TD-JEPA: `load_n_episodes: int = 5_000` (`metamotivo/data_loading/dmc.py:16`), first 5000 of an
  unsorted glob (`data_loading/base.py:131-132`) = 5M transitions.
- RLDP: `load_n_episodes: int = 5_000` (`examples/rldp_train_dmc.py:139`).
- HILP: README commands use `--num_episodes=5000` for walker/cheetah/quadruped.

So the compendium's line "10k episodes per domain on the cluster vs 5000 here"
(`docs/2026-08-25-exorl-humenv-compendium.md:17`) describes what is on disk, not what any reference
trains on: every reference trains on 5000 episodes. Our only difference is the 5% hold-out.

### Transition construction and alignment
Ours (`env/exorl_env.py:375-398`): per episode emit `obs[:-1]`, `act[1:]`, `obs[1:]`, i.e. the T=1000
real transitions `(o_t, a_{t+1-row}, o_{t+1})`, dropping the dummy action row; reward =
`relabel_rewards(physics)[1:]` (`env/exorl_env.py:483`), i.e. the reward of the NEXT state's physics.
References do the same thing: official FB `sample()` returns `obs[step-1], action[step], next_obs[step],
physics[step], reward[step]` with `step >= 1` (`in_memory_replay_buffer.py:154-171`); TD-JEPA
`load_transitions` uses `observation[:-1]`, `action[1:]`, `next.observation[1:]`, `next.physics[1:]`
(`metamotivo/data_loading/base.py:47-52`).

Verified numerically (episode 3 of each domain, `relabel_rewards(physics)` vs the stored column):
walker `stand`: corr(stored[t], relabel[t]) = 1.0000, max |diff| = 0.0000 (vs 0.9859 for a one-step
shift); cheetah `run`: 0.9998 / max diff 0.0012; quadruped `walk`: 1.0000 / 0.0000. So `physics[t]`,
`observation[t]` and `reward[t]` all describe the same post-action state, and our `[1:]` slice puts the
reward on the correct transition. Reconstructing the observation from `physics[t]` reproduces the stored
`observation[t]` to 0.0 for walker and cheetah (and differs from `obs[t-1]`/`obs[t+1]` by ~10); for
quadruped the 27 accelerometer/force-torque sensor dims differ (dims 48-50, 54-77) because those need
the forward-dynamics pass the collector ran, while the 51 position/velocity dims match exactly. We never
reconstruct observations from physics -- we use the stored ones -- so this is informational.

### Episode length, action repeat, observation scaling
- Episode length: 1000 control steps + dummy row in the data; the eval env truncates after exactly 1000
  steps (measured: `walker` reaches `trunc` at step 1000; `env/exorl_env.py:80`). References: 1000
  (walker 25 s / 0.025, cheetah 10 s / 0.01, quadruped 20 s / 0.02; `url_benchmark/custom_dmc_tasks/walker.py:44-45`,
  TD-JEPA `dmc_tasks/walker.py:26-27`).
- Action repeat: 1 everywhere. Ours has no repeat wrapper (`ExoRLWrapper.step`, `env/exorl_env.py:322-328`);
  official `action_repeat: int = 1` for states (`pretrain.py:64`); TD-JEPA has no repeat code at all.
  Measured `n_sub_steps`: walker 10 (0.025 s control step), cheetah 1 (0.01 s), quadruped 4 (0.02 s),
  identical to the collectors' settings since the stored obs match.
- Observation normalization: none in our loader (`env/exorl_env.py:397`, plain float32 cast;
  `impls/utils/datasets.py:67-72` `Dataset.create` stores arrays as-is), none in the references
  (official: no obs stats anywhere; TD-JEPA `IdentityNormalizerConfig`, `launch_fb_dmc.py:27-29`).
  `density_fb.yaml:87` has `norm_obs: false` and the run config records none.
- Observation layout at eval: `_flat_obs` concatenates the dm_control OrderedDict in emission order
  (`env/exorl_env.py:239-249`), which is what `flat_observation=True` does in the references
  (`url_benchmark/dmc.py:358-368`); verified above by exact match against stored observations.
- Actions are clipped to [-1, 1] in the loader (`env/exorl_env.py:487`) and at eval
  (`env/exorl_env.py:323`); references use `action_scale.Wrapper(-1, +1)`.

## 2. Reward relabelling

Ours: `env/exorl_env.py:406-426` -- for every stored physics row, `with env.physics.reset_context():
env.physics.set_state(physics[i])` then `env.task.get_reward(env.physics)`, using the env built for
the target reward task (`_load_dmc`, `env/exorl_env.py:222-236`, suite task first then vendored ExoRL
custom task, the same resolution order as `url_benchmark/dmc.py:357`). Cached per
(domain, algo, task, episode set) under `~/exorl_data/relabel_cache/` (`env/exorl_env.py:429-456`).

Official FB code does exactly this (`in_memory_replay_buffer.py:40-55` at load time and
`goals.py:233-237` at sample time: `reset_context` + `set_state` + `task.get_reward`), with the reward
attached to the next state (section 1). TD-JEPA does it lazily at eval time on `next.physics` with the
addition of `set_control(action)` + `mj_forward` (`metamotivo/evaluations/dmc.py:96-102`); RLDP
likewise sets control (`examples/rldp_train_dmc.py:315-334`). Setting the control only matters where
the reward reads `physics.control()`: none of the walker/cheetah/quadruped rewards do, so the three
locomotion domains are bit-comparable; point_mass_maze has a `small_control` factor
`(control_reward + 4)/5`, which is 1.0 under our zero control and <= 1.0 under theirs (a <= 20% scale
on maze returns in OUR favour, not against).

Task definitions (vendored from exorl @ a3fb07a, `env/exorl_dmc_tasks/PROVENANCE.md`):
- walker: `_STAND_HEIGHT 1.2, _WALK_SPEED 1, _RUN_SPEED 8, _SPIN_SPEED 5` (`env/exorl_dmc_tasks/walker.py:41-46`);
  `flip` = tolerance(angmomentum, (5, inf), margin 5, linear) x stand term (`walker.py:210-221`) --
  identical to `url_benchmark/custom_dmc_tasks/walker.py:163-182`. `spin` = tolerance(angmomentum,
  (10, inf), margin 10, linear) returned alone (`walker.py:201-207`) -- identical to TD-JEPA
  `dmc_tasks/walker.py:181-185`. stand/walk/run come from the dm_control suite in all three code bases.
- cheetah: `_RUN_SPEED 10, _WALK_SPEED 2, _SPIN_SPEED 5`; `walk`/`walk_backward` =
  tolerance(+-speed, (2, inf), margin 2, linear); `run_backward` = tolerance(-speed, (10, inf), margin
  10) (`env/exorl_dmc_tasks/cheetah.py:32-34, 176-190`) -- identical to
  `url_benchmark/custom_dmc_tasks/cheetah.py:39-45, 187-202`.
- quadruped: `_RUN_SPEED 5, _WALK_SPEED 0.5, _JUMP_HEIGHT 1.0`; `stand` = upright term; `jump` =
  upright x tolerance(com_height, (1, inf), margin 1, value_at_margin .5)
  (`env/exorl_dmc_tasks/quadruped.py:40-43, 587-595`) -- identical to
  `url_benchmark/custom_dmc_tasks/quadruped.py:44-55, 444-492`. walk/run from the suite.
- point_mass_maze: near_target margin = `target_size` (0.015) (`env/exorl_dmc_tasks/point_mass_maze.py:183-190`),
  identical to `url_benchmark/custom_dmc_tasks/point_mass_maze.py:195-202` and HILP
  (`hilp_zsrl/url_benchmark/custom_dmc_tasks/point_mass_maze.py:194-196`). TD-JEPA's maze reward uses
  margin = `6 * target_size` (`metamotivo/envs/dmc_tasks/pointmass.py:232-236`), a 6x wider reward
  shell, so TD-JEPA maze numbers are NOT comparable to ours or to the FB/HILP papers.

The eval env is built by the same `_load_dmc(domain, reward_task)` call the relabeller uses
(`env/exorl_env.py:343-349` and `env/unified.py:325-326`, both fed by `exorl_task`), so training
reward and eval reward are the same function by construction.

## 3. Zero-shot task inference at eval

Code path. Training-time eval (`impls/main.py:478-509`) and the scorer
(`scripts/reeval_checkpoint.py:158-168`) are the same two calls: `rb = train_ds.sample(N)`; `rows =
infer_cond_rows(critic)`; `agent.infer_eval_cond(rb[rows], rb["rewards"])`; then `InferredCondAgent`
so the env goal is never forwarded. `scripts/exorl_paper_eval.py:153-173` drives `reeval_checkpoint`
with `--override exorl_task=<task>`, which both relabels the dataset and builds the env for that task.

Which reward column. On ExoRL the base dataset's `rewards` IS the per-task relabelled reward
(`env/exorl_env.py:483`). `keep_dataset_rewards: true` (`env/domains/dmc-walker.yaml:38`) makes
`GCDataset`/`HGCDataset` write that column back over the goal-reaching reward they synthesise
(`impls/utils/datasets.py:363-397`); `FBDataset` (fb), `WindowedDataset` (rldp) and
`HILPFutureDataset` (hilp) pass the dataset column through untouched
(`impls/utils/datasets.py:163-164, 628-629, 761`). The task is selected per scored cell:
`reeval_checkpoint` drops the run's recorded `exorl_task` from the replayed config
(`scripts/reeval_checkpoint.py:113-115`) and the driver's override supplies `stand`/`run`/`spin`/...
(`scripts/exorl_paper_eval.py:162`). The relabel cache confirms it: 5M-row caches exist for walker
stand/walk/run/flip/spin, cheetah run/run_backward/walk/walk_backward, quadruped stand/walk/run/jump.
So the "solves for a constant" bug of `docs/notes/2026-09-15-transfer-table-and-exorl.md` (OGBench play
data, where the dataset column is the goal-reaching reward) does NOT apply here; the yaml's
`exorl_task: walk` is only the training-time default and is overridden for every scored task.
Witness columns agree: every scored row has `inferred_cond_cos_neutral` far from 1
(e.g. -0.03 .. 0.21 in `exp/paper_eval_exorl_dfb_2M.jsonl`), i.e. the constant-reward fallback never fired.

Which states. All reward-interface critics declare `INFER_COND_MARGINAL = "next_observations"`
(`impls/critics/fb.py:77`, `rldp.py:169`, `hilp.py:213`, `density_fb.py:527`), so B (or phi) is evaluated
on the next state whose reward was relabelled -- the references' convention (official FB
`pretrain.py:169` `batch.next_obs`; TD-JEPA `evaluations/dmc.py:109-112` `batch["next"]["observation"]`).

Estimator. fb: `z = r^T B` then sqrt(d)-projection (`impls/critics/fb.py:201-205`) = official
(`fb_ddpg.py:201-222`, mean then normalize; scale-free). hilp: least squares on phi (`impls/critics/hilp.py:437-446`)
= HILP `sf.py::infer_meta_from_obs_and_rewards`. rldp: as fb.

Sample counts. Ours: `--infer_cond_episodes` (a transition count) = 50 000 for the 2M density_fb_z
campaign, both in-run (`logs/exorl_dfb/gen_phase1.sh:45`) and in the scorer (`INFERENCE_SAMPLES` has
no `density_fb_z` entry so it falls back to 50 000, `scripts/exorl_paper_eval.py:276`, the
"--inference-samples defaulted to 50000" line); 10 000 for the 1M fbref/arch scoring
(`scripts/score_exorl_1m.sh:21`). References: official FB 5120 (`fb_ddpg.py:55`), TD-JEPA 10 000
(`launch_fb_dmc.py:181-182`), RLDP 50 000 (`rldp_train_dmc.py:151`), HILP 10 000. Samples are drawn
uniformly with replacement from the training rows (`Dataset.get_random_idxs`, `impls/utils/datasets.py:87-101`,
oversampling off); references draw uniformly too. More samples only lowers the estimator's noise.

Observed inference noise: for walker walk seed 0 the in-run eval at 2M scored 687
(`exp/exorl_dfb/.../seed_0/eval.csv`, same 10 reset seeds) and the scorer 614, the only difference
being the 50k-row draw and the RNG state. So a +-50 spread on walker from the inference draw alone is
real and should be expected when comparing single cells.

## 4. Evaluation

- Episodes: 10 per (task, seed) in every quoted row (`eval_episodes: 10` in both jsonl files;
  `scripts/score_exorl_1m.sh:21`, `logs/exorl_dfb/gen_phase1.sh:45`). References: official FB 10
  (`pretrain.py:70,72` and `finalize()` 10 x 1 episode); TD-JEPA 10 (`launch_fb_dmc.py:181`); RLDP code
  default 10 (`rldp_train_dmc.py:150`; the paper text says 50); HILP final eval 50.
- Actions: deterministic, `dist.mode()` clipped (`impls/actors/ddpgbc.py:132-134`, called with
  `temperature=0` from `env/unified.py:502-504`). References: `dist.mean` (`fb_ddpg.py:258-280`,
  TD-JEPA `agents/fb/model.py:128-132`).
- Horizon 1000, undiscounted return summed over `info["reward"]` (`env/unified.py:505-506`,
  `env/exorl_env.py:326`) = the benchmark metric.
- Reset seeding: ours reseeds the task RNG with the episode index 0..9 (`env/unified.py:484`,
  `env/exorl_env.py:293-320`), so every checkpoint and every seed sees the same 10 initial states.
  References construct the env with one seed and take consecutive resets (official `pretrain.py:238-239`,
  TD-JEPA env seed 0 for all training seeds). Same information content; ours removes reset variance
  across checkpoints but means a re-eval cannot draw fresh states without `eval_seed_offset`.
- Checkpoint scored: the final one (`--restore_epoch 2000000` / `1000000`), no best-of and no last-N
  averaging (`scripts/exorl_paper_eval.py:145-173`). References: official FB scores the final
  checkpoint after 1M steps in `finalize()`; TD-JEPA has no selection code (3M steps, evals every 250k
  logged to CSV); RLDP scores the last logged eval of a single overwritten checkpoint after 6M steps (3M encoder-only + 3M FB, `examples/rldp_train_dmc.py:138,142`; the paper text says 2M + 3M), domain score = unweighted mean over tasks (`rldp_train_dmc.py:311`); HILP scores the final model after 1M steps with 50 episodes per task (`train_offline.py:73-76, 420-455`).
- Aggregation: our driver reports per (critic, domain, task) mean +- std over seeds keyed by seed with
  last-row-wins on re-scores (`scripts/exorl_paper_eval.py:209-246`). Domain scores in the references
  are the plain mean over the four tasks.
- Training budget is NOT part of this audit but is the biggest protocol gap for the FB/TD-JEPA
  comparison: our cells are 1M (fbref) and 2M (density_fb_z) gradient steps against TD-JEPA's 3M
  (`train.py:84`, `launch_fb_dmc.py:16`) and RLDP's 5M.

## 5. Floors

Random-policy floors (`exp/random_floors.json`, 10 episodes, uniform actions in [-1,1]; quadruped
measured now with 5 episodes) and the relabelled dataset's own mean episode return (5000 episodes,
from the relabel caches):

| domain/task | random policy | dataset mean (std, max) | our best cell (2M density_fb_z, 5 seeds) |
|---|---|---|---|
| walker stand | 126 | 385 (48, 548) | 715 |
| walker walk | 34 | 189 (46, 408) | 641 |
| walker run | 23 | 87 (15, 176) | 204 |
| walker flip | 50 | 238 (45, 395) | -- |
| walker spin | -- | 471 (90, 830) | 976 |
| cheetah run | 3.7 | 50 (29, 202) | 60 |
| cheetah run_backward | 20 | 47 (29, 203) | 96 |
| cheetah walk | 18 | 231 (118, 816) | 303 |
| cheetah walk_backward | 101 | 218 (120, 777) | 469 |
| quadruped stand | 601 | 647 (181, 996) | 795 |
| quadruped walk | 307 | 309 (97, 593) | 467 |
| quadruped run | 301 | 319 (90, 501) | 404 |
| quadruped jump | 439 | 565 (163, 850) | 609 |

Readings: cheetah run 60 is above the data floor (50) but barely, and far below published FB (247 in
Touati et al.); walker cells are 2-3x the data mean; quadruped is a domain where a random policy already
scores 300-600, so the 2M cells are only 1.1-1.5x random. The 1M FB-at-reference-config cells
(`exp/paper_eval_exorl_1m_fbref.jsonl`: stand 920, walk 804, run 400, spin 985) show the same pipeline
producing published-level walker numbers, which is the strongest evidence the pipeline is not the
bottleneck.

## Reference protocol summary (for the record)

| | official FB (url_benchmark) | TD-JEPA | RLDP | HILP | ours |
|---|---|---|---|---|---|
| episodes loaded | 5000 sorted (`pretrain.py:83`) | 5000 unsorted (`dmc.py:16`) | 5000 unsorted (`rldp_train_dmc.py:139`) | 5000 sorted (`train_offline.py:87`) | first 5000 sorted, minus 5% hold-out |
| reward relabel | set_state on next physics (`in_memory_replay_buffer.py:40-55`) | set_state + set_control + mj_forward on next physics (`evaluations/dmc.py:96-102`) | same as TD-JEPA (`rldp_train_dmc.py:321-328`) | set_state on next physics (`in_memory_replay_buffer.py:39-41`) | set_state on next physics (`env/exorl_env.py:423-425`) |
| obs normalization | none | none (Identity) | none (`norm_obs=False`) | none (phi centering only) | none |
| action repeat | 1 | 1 | 1 | 1 | 1 |
| inference samples | 5120 (`fb_ddpg.py:55`) | 10 000 | 50 000 | 10 000 | 50 000 (2M campaign) / 10 000 (1M scoring) |
| map input | next_obs | next_obs | next_obs | next_obs | next_obs |
| eval episodes | 10 | 10 | 10 (paper text 50) | 50 final | 10 |
| actions | mean | mean | mean | mean | mode |
| gradient steps | 1M | 3M | 3M + 3M | 1M | 1M / 2M |
| checkpoint | final | logged per eval, no selection in code | last logged eval | final | final |
| walker 4th task | flip | spin | flip | flip | both registered; scored set chosen by `--task-set` |

## 6. Deviations from the official protocol, ranked by expected effect on the reported return

1. Training budget (context, not this pipeline): 1M/2M gradient steps vs TD-JEPA 3M / RLDP 5M
   (`launch_fb_dmc.py:16`, `rldp_train_dmc.py:138,142` = 3M encoder + 3M FB). Largest expected effect; a real gap, not a bug.
2. 5% validation hold-out: training and task inference use 4750 of the 5000 episodes
   (`env/unified.py:334`). References use all 5000. Effect: small (5% less data); direction: could only
   lower returns slightly.
3. Number of reward-labelled samples for the task vector: 50 000 (density_fb_z campaign) vs 10 000
   (TD-JEPA/HILP) and 5120 (official FB); 10 000 in the 1M scoring. Effect: only reduces estimator noise;
   cannot depress returns. Note the +-50-point walker spread between two 50k draws on the same checkpoint.
4. Fixed reset seeds 0..9 per episode vs consecutive resets in the references (`env/unified.py:484`).
   Effect: none in expectation; changes only the variance and the comparability of re-evals.
5. point_mass_maze reward: ours = ExoRL/official FB/HILP (margin = target_size, zero control) vs
   TD-JEPA (margin = 6 x target_size, control set). Effect: maze rows are not comparable to TD-JEPA's
   table; no effect on walker/cheetah/quadruped.
6. Relabelling sets the physics state but not the control (`env/exorl_env.py:423-425`), like the
   official FB code and unlike TD-JEPA/RLDP. Effect: zero on the three locomotion domains (rewards
   never read `physics.control()`); <= 20% upward on maze.
7. Eval episodes 10 vs RLDP paper text 50 / HILP final 50. Effect: variance only.
8. `exorl_report.py` (`scripts/exorl_report.py:275-277`) does not know `dmc-quadruped` or the
   `density_fb_z`/`hilp`/`rldp_*` critic names, so its run-dir parser would mislabel those runs; the
   quoted numbers come from `exorl_paper_eval.py`, which is unaffected.

Outright bugs found: none in the data, relabelling, inference or evaluation path. Two cosmetic
issues: the "LEGACY-ARCH ... rldp_official_nets=false" line only fires for RLDP runs whose
config.json predates that key and pins them to the architecture they were trained with
(`scripts/exorl_paper_eval.py:100-131`), and `--infer_cond_episodes` is a transition count despite
its name (`scripts/exorl_paper_eval.py:21-25`).
