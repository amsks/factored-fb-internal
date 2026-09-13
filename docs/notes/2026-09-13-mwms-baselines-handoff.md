# MetaWorld / ManiSkill baselines (HILP, RLDP, FB) — collaborator handoff (2026-09-13)

Goal: fill the `HILP` and `RLDP` columns (and re-run `FB` for reference) of the MetaWorld/ManiSkill
blocks of `PAPER/ICLR/content/experiments.tex` `tab:manipulation` — 8 tasks: mw-push, mw-bin-picking,
mw-box-close, ms-push-cube, ms-pick-cube, ms-stack-cube, ms-poke-cube, ms-lift-peg — under the
protocol that produced the existing FB / Density-FB cells: 1M gradient steps, 3 seeds, eval every 100k
with 50 episodes, success read at the last step. Everything below lives on branch `density-fb`
(worktree `Factored-FB-seam`), uncommitted at the time of writing (see "files" at the end).

## 0. Read this first — three things that are NOT settled

1. **The dataset the table used is not on the HF repo.** The paper appendix
   (`PAPER/ICLR/content/appendix/data-gen-mw-ms.tex`) describes MetaWorld packs of N=2010 x T=500
   (scripted MetaWorld oracles + grasp primitive, 1,005,000 transitions per task) and ManiSkill packs of
   N=8000 x T=125 (newt TD-MPC2 checkpoints, 1,000,000 transitions per task). The HF repo
   `TommyShen/Factor-FB-Offline` (sha `9f36471`, last modified 2026-07-22; local copy
   `/home/adityamohan/git/datasets/Factor-FB-Offline`, manifest name `factoredfb-testbed-demos`)
   holds a DIFFERENT collection: MetaWorld N=1000 x T=100 and ManiSkill N=1000 x T=25, both from
   newt TD-MPC2 experts, and it has NO `mw-push` at all (its MetaWorld tasks are assembly,
   basketball, bin-picking, box-close, button-press). So the local HF copy is not the table's
   "expert-fixed" collection. Before launching, obtain the collector's packs (ask whoever produced the
   Density-FB column; the appendix says the MetaWorld packs additionally carry `goal_switch`, which is
   a quick fingerprint: `np.load(p).files`) and put them at
   `$FB_DATA_ROOT/data/metaworld/<task>.npz` and `$FB_DATA_ROOT/data/maniskill/<task>.npz`.
   Running on the HF packs is possible (the pipeline was smoked on them) but yields a different table.
2. **The Density-FB reward-sample-index column used the `ddpgbc_density` actor**, the goal column
   `flowbc`. The three baselines here all run `flowbc` (bc_coeff 3.0, as on OGBench manipulation),
   which matches the goal column and the OGBench baseline protocol; the reward-inference route is
   therefore actor-matched to FB/HILP/RLDP's OGBench cells, not to Density-FB's MW/MS index column.
3. **Budget.** `scripts/launch_density_table.sh` (the earlier MW/MS Density-FB launcher) ran MW at
   500k and ManiSkill at 300k with eval every 50k; the table text says 1M. This launcher defaults to
   1M / eval every 100k so it matches the table; set `TRAIN_STEPS=300000 EVAL_INTERVAL=50000` to
   match the old launcher instead. Whichever you pick, state it next to the numbers.

## 1. Data

- HF repo: `TommyShen/Factor-FB-Offline` (`hf download TommyShen/Factor-FB-Offline --repo-type dataset
  --local-dir <root>`), layout `data/<family>/<task>.npz` + `<task>.meta.json` + `manifest.json`.
  Each npz packs `observation, action, next_observation, achieved_goal, reward, terminal, timeout,
  success` as `[N, T, ...]`.
- `FB_DATA_ROOT=<root>` is the only path the domain yamls read (`env/domains/mw-*.yaml`,
  `ms-*.yaml`: `data_path: ${oc.env:FB_DATA_ROOT}/data/<family>/<task>.npz`). Two older yamls
  (`mw-bin-picking`, `mw-box-close`) read `${oc.env:FB_HF_EXPORT}`; the launcher exports both
  variables to the same root, so one variable is enough when you go through it.
- Loader: `env/unified.py::load_packed` flattens to `[N*T, ...]`, clips actions to [-1, 1], marks the
  LAST row of every episode in `terminals` (the trajectory boundary every sampler respects), keeps
  the per-step `success` flag (new; used only by the reward-inference eval), and holds out the last
  5% of rows as validation.
- Known data property (HF packs; check yours): the last row of every episode stores the NEXT
  episode's reset frame as `next_observation` (its `success` is 0 and the object-target distance
  jumps back to the reset value). That is 1/T of the transitions (1% MW, 4% MS at T=25) with a
  wrong next state. No loader in this repo drops them; Density-FB trained on the same rows.

## 2. Environment / venv

Working venv on the AWS box: `/dev/shm/.venv-jax/bin/python` (Python 3.11.15). Versions that ran the
smoke: jax 0.10.0, jaxlib 0.10.0, flax 0.12.7, optax 0.2.8, numpy 2.4.6, metaworld 2.0.0,
mani_skill 3.0.1, sapien 3.0.3, mujoco 3.3.0, gymnasium 1.3.0, torch 2.13.0+cu130 (not used by
training), omegaconf 2.3.1, wandb 0.27.0, huggingface_hub 1.24.0. One venv serves both families
(no separate ManiSkill venv is needed; ManiSkill runs headless with `sim_backend="auto"` and
rendering off — `render: false` in the yamls, `--num_video 0`). Env vars every run sets:
`XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline
XLA_FLAGS=--xla_gpu_enable_command_buffer=`. On Vista, `scripts/vista_train.slurm` also ships a
Vulkan ICD for SAPIEN (`VK_ICD_FILENAMES`) and caps `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`.

## 3. What runs: critics, actor, eval routes

| column | `--critic` | config | note |
|---|---|---|---|
| FB | `fb_tdj` | `impls/configs/critic/fb_tdj.yaml` | FB with ortho_coef 1000 (the td_jepa parity arm) |
| HILP | `hilp_tdjc` | `impls/configs/critic/hilp_tdjc.yaml` | TD-JEPA's published OGBench HILP cell; dataset `HILPFutureDataset` |
| RLDP | `rldp` + `--override ortho_coef=1.0` | `impls/configs/critic/rldp.yaml` | RLDP's own default coefficient; dataset `WindowedDataset` (h=5) |

Actor for all: `flowbc` (`impls/configs/actor/flowbc.yaml`, bc_coeff 3.0). Two eval routes per
eval point, as on OGBench:

- **Goal-conditioned (Route A) -> `eval.csv`.** The env goal frame is embedded by each critic's
  `map_eval_cond` (FB/RLDP: z = B(goal); HILP: z = normalize(phi(g) - phi(s)) per step).
- **Reward inference (Route B) -> `eval_relabel.csv`**, via `eval_relabel_size=10000
  eval_relabel_shift=1.0 eval_relabel_episodes=10`: 10k random training transitions are labelled
  with the task reward of their next state, the critic infers z from (next_obs, reward)
  (FB/RLDP: reward-weighted mean of B; HILP: least squares on phi), and the policy is rolled on that
  z with the env goal withheld. Each row carries `route_b_valid` (1 = inference was live) and the
  `inferred_cond_*` witness columns. This route did not exist for the unified adapter before
  2026-09-13 (it was OGBench-only); see section 6.

Per-task protocol overrides (applied identically to every critic; `PROTO` in the launcher):

| task | overrides |
|---|---|
| mw-push, mw-bin-picking, mw-box-close | `obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth` |
| ms-push-cube | `obs_mask_dims=[25,26,27] goal_mask_dims=[25,26,27] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth` |
| ms-pick-cube | `obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth` |
| ms-stack-cube, ms-poke-cube, ms-lift-peg | none (no goal position in the observation; `eval_goal_mode=fixed`, i.e. the goal frame is the highest-reward next_obs in the data) |

What they mean: the mask blanks the target-position dims from every observation and goal frame so
the policy has to get the task from the conditioning; `eval_goal_mode=synth` builds the eval goal
frame from each episode's own reset (`env/unified.py::synth_goal_frame`); the `value_p_*` pair only
matters for goal-sampling dataset classes (kept for parity with the Density-FB launches). The three
tasks without a goal-in-obs layout are new yamls (`env/domains/ms-{stack-cube,poke-cube,lift-peg}.yaml`);
the Density-FB column's settings for them are not recorded in this repo — copy them from those runs'
`config.json` (`config.obs_mask_dims`, `goal_mask_dims`, `eval_goal_mode`) into `PROTO` if they differ.
Hit specs (`measure_hit_spec`) are Density-only; the three baselines never read them
(`backward_input: state`).

## 4. Launching

`scripts/launch_mwms_baselines.sh` — dry run by default, prints every command line:

```bash
FB_DATA_ROOT=/path/to/root bash scripts/launch_mwms_baselines.sh                     # 72 lines
MODE=tmux GPUS="0 1 2 3" FB_DATA_ROOT=... bash scripts/launch_mwms_baselines.sh      # local: one tmux window per GPU, runs sequential per window
MODE=slurm ALLOC=<alloc> FB_DATA_ROOT=... VENV=$SCRATCH/.venv-jax-fb bash scripts/launch_mwms_baselines.sh   # one sbatch --array
MODE=slurm DRYRUN=1 ...                                                               # print the job lines + the sbatch line
```

Knobs: `CRITICS`, `TASKS`, `SEEDS` (default `0 1 2`), `TRAIN_STEPS` (1000000), `EVAL_INTERVAL`
(100000), `SAVE_INTERVAL` (250000), `EVAL_EPISODES` (50), `RELABEL_SIZE/SHIFT/EPISODES`
(10000/1.0/10), `EXTRA_OV` (extra `k=v` for every run), `SESSION` (tmux, default `mwms_baselines`),
`MAX_PARALLEL` (slurm array throttle, 20), `PY` (default `/dev/shm/.venv-jax/bin/python`).
Runs whose `params_<TRAIN_STEPS>.pkl` exists are skipped; partial runs resume (`--auto_resume`).
The slurm path writes `logs/mwms_baselines/jobs-<stamp>.txt` (one run per line in
`scripts/vista_train.slurm`'s env-var contract) and `scripts/mwms_array.slurm` reads its line and
`exec`s `vista_train.slurm`, so the cluster environment is the one every earlier campaign used.

Wall clock: on an L4 the smoke ran at roughly 300 steps + compile in ~1-2 min per run; expect the
MetaWorld eval (50 episodes x 100 steps x 2 sim steps, MuJoCo-bound) and ManiSkill eval (50 x 25) to
dominate a 1M run at ~10 eval points; budget ~4-6 h per run on a GH200, 72 runs.

## 5. Outputs and scoring

Run dir: `exp/mwms_baselines/<critic>/<task>_<critic>_flowbc_seed_<s>/`
- `config.json` — the resolved config (masks, overrides, critic, seed): the provenance of the row.
- `train.csv` — losses every `--log_interval` steps.
- `eval.csv` — goal-conditioned route, one row per eval step: `success` (any step of the episode
  hit the env's success flag), `success_final` (flag on the episode's last step — the table's
  "success at the final step" reading, new 2026-09-13), `goal_dist_min/final`, `task_id=-1`.
- `eval_relabel.csv` — reward-inference route: the same metrics plus `route_b_valid`,
  `invalid_reason`, `inferred_cond_norm`, `inferred_cond_goals_dropped` (> 0 proves the env goal was
  withheld). Only rows with `route_b_valid == 1` are results.
- `eval_per_episode.jsonl`-style sidecars written by `_drain_per_episode` (per-episode success and
  goal distances) for re-scoring at other radii; `params_<step>.pkl` checkpoints.
- `logs/mwms_baselines/<task>/<critic>/s<seed>.log` (tmux) or `logs/mwms.o<job>_<task>` (slurm).

Score = mean over tasks of the per-run mean of `success` over the LAST THREE eval points, then
mean +- sd over seeds, separately for `eval.csv` and `eval_relabel.csv`
(`scripts/make_results_table.py::score_run` does exactly the last-three mean; pass `--csv-name` is
not supported there, so for Route B use the snippet):

```python
import csv, glob, statistics as st
from collections import defaultdict
def score(path, col="success"):
    pts = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if col in r and r[col] not in ("", "nan") and r.get("route_b_valid", "1.0") != "0.0":
            pts[int(float(r["step"]))].append(float(r[col]))
    return st.mean(st.mean(pts[s]) for s in sorted(pts)[-3:]) if pts else None
for csvname in ("eval.csv", "eval_relabel.csv"):
    for f in sorted(glob.glob(f"exp/mwms_baselines/*/*/{csvname}")):
        print(csvname, f.split("/")[-2], score(f), score(f, "success_final"))
```

Report both `success` (any-step) and `success_final`; say which one a table cell is.

## 6. What to send back

Per run: `eval.csv`, `eval_relabel.csv`, `config.json` (and `train.csv` if small) — e.g.
`tar czf mwms_baselines_<site>.tgz exp/mwms_baselines/*/*/{eval.csv,eval_relabel.csv,config.json}`.
With hub access: `python scripts/mailbox.py post --from <site> --file note.md` and
`python hf_dataset/hf_upload_checkpoints.py` / `scripts/mailbox.py` conventions on
`amsks/factored-fb-checkpoints` (status json per run under `status/`). There is no
`publish_runs.py` in this tree; the mailbox script is the supported channel.

## 7. Pipeline changes made for this (all uncommitted on `density-fb`, 2026-09-13)

- `env/unified.py`: `relabel_infer_batch` (Route B for MetaWorld/ManiSkill/Box2D packs; reward =
  stored per-step `success` mapped to the OGBench convention `(success - 1) + shift`, so 0/1 at
  shift 1.0; `--override eval_relabel_reward=dense` uses the stored dense env reward instead; rows
  are blanked with the training mask union); `evaluate` accepts `any_step_success` (parity with
  the OGBench evaluator, which `impls/main.py` passes) and reports `success_final`; `load_packed`
  keeps the `success` column. `env/base.py`: `column_backed_dataset` (unwraps sampler wrappers).
- `impls/main.py`: the Route B loop no longer assumes `env.unwrapped.task_infos` (OGBench-only);
  single-task adapters run one relabel task.
- `impls/utils/datasets.py::HILPFutureDataset`: applies `obs_mask_dims | goal_mask_dims` to
  observations, next observations and the hindsight future goal. Before this, HILP under the
  MW/MS mask protocol trained with the target visible and was evaluated with it blanked
  (`WindowedDataset` and `FBDataset` already masked). Any earlier HILP number on a masked unified
  domain is not comparable.
- `env/domains/ms-{stack-cube,poke-cube,lift-peg}.yaml`: new.
- `tests/impls/test_unified_episode_boundaries.py`: on the packed layout, HILP's geometric future
  and RLDP's h-step window never cross an episode end; the HILP mask; the unified relabel batch.
- `scripts/launch_mwms_baselines.sh`, `scripts/mwms_array.slurm`: the launcher.

## 8. Smoke (2026-09-13, GPU 7, 300 steps, 2 eval episodes, `--log_interval 100`)

Data: mw-push from the play-resample pack (`factoredfb-metaworld-play`, 200 x 500) because the HF
expert repo has no mw-push; ms-push-cube from the HF pack. One venv (`/dev/shm/.venv-jax`) served
both; ManiSkill needed no extra setup (headless, no rendering). All four ran clean (no traceback):

| run | train.csv | eval.csv (steps 1, 300) | eval_relabel.csv (`route_b_valid`) |
|---|---|---|---|
| mw-push / hilp_tdjc | 3 rows, all finite | success 0.0, 0.0 | 2 rows, valid 1, 1; inferred z norm 7.07 |
| mw-push / rldp (ortho 1.0) | 3 rows, all finite | success 0.0, 0.0 | 2 rows, valid 1, 1 |
| ms-push-cube / hilp_tdjc | 3 rows, all finite | success 0.0, 0.0 | 2 rows, valid 1, 1 |
| ms-push-cube / rldp (ortho 1.0) | 3 rows, all finite | success 0.0, 0.5 | 2 rows, valid 1, 1; success 0.0, 0.5 |

(2 episodes at 300 steps: the numbers only show the pipeline is live.) `inferred_cond_goals_dropped`
is 0 on the unified evaluator by construction -- it never builds a goal frame for an agent that
rolls an inferred conditioning -- so on these domains the proof that Route B was live is
`route_b_valid=1` + `inferred_cond_norm`, not the dropped-goal count. Smoke outputs were deleted;
the related test set (125 tests incl. the new boundary/mask/relabel file) passed on CPU.

## 9. Open questions

- Dataset identity (section 0.1): which packs produced the table, and where are they hosted?
- Actor mismatch with the Density-FB index column (section 0.2): should Density-FB's MW/MS
  reward-sample column be re-run with `flowbc`, or the baselines with `ddpgbc`-style actors?
- Budget (section 0.3): 300k/500k (old launcher) vs 1M (table text).
- "Success at the final step": any-step `success` vs last-step `success_final` — the old Density-FB
  MW/MS numbers came from `eval.csv::success` (any-step), the paper text says final step.
- Protocol for ms-stack-cube / ms-poke-cube / ms-lift-peg (section 3): no masks, fixed goal — confirm
  against the Density-FB runs' `config.json`.
- The last-row-of-episode next_observation issue (section 1) applies to every method equally but is
  worth a one-line fix (`valids` mask) if the collaborator's packs share it.
