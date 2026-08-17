# Deconfounding the bilinear advantage: dataset size, goal coverage, and the cube question

**Date:** 2026-08-10 · **Status:** SPEC — ready to execute
**Runs on:** Vista (SLURM). Venvs: `$SCRATCH/.venv-jax-fb` (MW/cube/box2d), `$SCRATCH/.venv-ms` (ManiSkill training), `.venv-gen` + newt paths (MS collection).

## 1. What this spec settles

Two claims are on the table:

- **RQ1:** bilinear GCIQL (`iql_bilinear`, Q = ⟨φ(s,a),ψ(g)⟩/√d) beats monolithic GCIQL and
  BC on *play* data generally.
- **RQ2:** the reason it wins — currently the surviving mechanism is off-distribution
  ranking coherence / matrix completion (P2 2×2, BoN matrix), with boundary conditions from
  Arms A/D.

Both are currently supported almost entirely by **one cell**: OGBench cube-single-play.
Every attempted replication has run on data that differs from cube not just in structure
(fixed in the play-resample collectors) but in **size** and **goal coverage**:

| dataset | transitions | trajs × len | trained on it? | verdict so far |
|---|---|---|---|---|
| cube-single-play-v0 | 1,001,000 | ~1000 × 1000 | yes (the headline) | biql 93 vs gciql 73 |
| MS upstream play-resample (TommyShen) | **48,000** | 120 × 400 | yes (Arm E) | null (all 0.13–0.25, 20 eval eps) |
| MW play-resample v1 | ~100,000 | 200 × 500 | yes (self-censored square) | uninformative (goal-in-obs shortcut; v1 data defects) |
| MW play-resample **v3** | 1,000,000 | 2000 × 500 | **no runs yet** | — |
| MW play-noise-hard | ~small | — | yes | partial replication (push +0.27 biql vs +0.05 gciql) |

So "bilinear doesn't replicate off cube" is confounded with "every off-cube play dataset was
7–21× smaller and/or had a broken goal channel." Conversely "bilinear wins on cube" has never
been tested at MS/MW-scale data. This spec kills the confound from both directions and fixes
the last known eval defect on the way.

## 2. Confound inventory

- **C1 — size.** Cube 1.0M vs MS 48k vs MW-v1 100k. Untested in either direction.
- **C2 — goal coverage.** Arm D already showed causally, on cube, that biql *needs*
  goal-marginal coverage (OOD half-space: biql 0.37 < gciql 0.49). The goal-coverage figure
  (`PAPER/decoding-bilinear/figs/goal_coverage.png`, commit bc64b56) was built from MW **v1**
  and the 48k MS set — it needs regenerating, and eval-goal in-supportness needs to be a
  number, not a picture.
- **C3 — goal observability.** Solved knob-wise: `obs_mask_dims` (NOT `goal_mask_dims` alone —
  that blanks only the emitted goal and leaves the shortcut open). Must be ON for every MW/MS
  run in this spec. MW dims `[36,37,38]`; MS `ms-push-cube [25,26,27]`, **`ms-pick-cube
  [26,27,28,39,40,41]`** — see the correction in §8.1: the three dims quoted in
  `scripts/launch_upstream_resample.sh` are NOT sufficient for pick-cube.
- **C4 — eval conditioning.** STILL BROKEN for trained-policy eval:
  `env/unified.py` caches ONE goal frame (`next_observations[argmax(rewards)]`, line ~186)
  and conditions every episode on it, while MW/MS redraw the target per reset. Per-episode
  goal synthesis exists only in `eval/bon_probe.py --goal_mode synth` (where it lifted every
  MW selection cell by +0.12..+0.43). Must be ported to `evaluate()` (E0 below).
- **C5 — discriminability.** MW's goal box is small relative to its success radius:
  P(relabeled goal already satisfied at 1–2× eval-horizon lookahead) = 0.249 (push) / 0.155
  (pick-place) vs cube 0.013 (`scripts/compare_play_resample_to_ogbench.py`). Temporal
  distance carries less signal, so MW is a weaker critic discriminator *even with everything
  else fixed*. Property of the env, not the data. Handled by matched-radius re-scoring (E5).

## 3. Shared infrastructure (E0) — do this first

1. **`eval_goal_mode=synth` in `env/unified.py::evaluate`.** New config knob,
   `fixed` (default, current behavior) | `synth`. In synth mode, after `seeded_reset`:
   read the RAW obs (before `_mask_obs`), build the goal frame per episode exactly as
   `eval/bon_probe.py` does for MW (goal = copy of raw obs with hand `[0:3]`(?) and object
   `[4:7]` slots ← target `raw_obs[36:39]` — copy bon_probe's verified construction, don't
   re-derive), then apply the same `obs_mask_dims`/`goal_mask_dims` masking as training.
   For MS: same pattern; the target sits in the raw obs at the mask dims, so the synth frame
   is constructible from the reset obs alone — but the **object-position slots in the MS obs
   layout must be verified first** (check `env/maniskill_env.py` / ManiSkill 3.0.1 obs docs;
   confirm by regressing recorded `achieved_goal` onto obs columns in the packed npz).
2. **Per-episode distance logging.** In `evaluate()`, record the min-over-episode and final
   `‖object − target‖` per episode (object slot per family, target from the env, not the
   masked obs) and write them into the eval output dict. Costs nothing, enables E5 without
   re-running anything.
3. **Cube dataset truncation knob.** `dataset_num_transitions` (or `dataset_num_traj`) in the
   OGBench adapter (`env/ogbench.py`): after `ogbench.make_env_and_datasets`, keep a random
   subset of WHOLE trajectories (split on `terminals`, `np.random.RandomState(0)`, same
   subset for every run at a given size) totalling ≥ the requested transition count. Train
   split only — leave `-val` untouched. Sanity: knob at full size must be a bit-for-bit
   no-op.
4. **Do not use `output_scale`** in any of these runs — the knob crashes at init
   (`self.param` outside `@compact`, `impls/utils/networks.py:462`, all 6 aA2 scale runs
   died). It's orthogonal to this spec; leave it to the cube-double thread.

Smoke-test all of E0 with 200-step runs on a SLURM node before launching anything (login
nodes cannot run JAX — 100-thread ulimit).

## 4. Experiments

### E1 — Cube size ladder (does the cube advantage survive at MS/MW scale?)

The direct test of C1 in the only environment where the advantage exists. No new data.

- **Data:** cube-single-play-v0 truncated to **48k** (= MS scale), **100k** (= MW-v1 scale),
  **250k**, plus the existing full-1M results as anchor (re-run 1 sanity seed at full size
  through the new knob to confirm no-op).
- **Matrix:** {bcflow (BC floor), gciql+flowbc, iql_bilinear-d50+flowbc} × {48k, 100k, 250k}
  × 3 seeds × 500k grad steps = **27 runs** (+1 sanity). Standard cube eval (5 task_infos,
  the OGBench adapter path — C3/C4 don't apply to cube).
- **Pre-registered readings:**
  - *Gap (biql − gciql) persists at 48k* → size cannot explain the Arm E null; the residual
    suspects are C4/C5 → E2 with synth eval arbitrates.
  - *Gap collapses at ≤100k but holds at 250k+* → Arm E and the v1 square were data-starved
    and say nothing about the bilinear head; the claim becomes "bilinear wins on play data at
    scale," and E2/E3 at 1M decide whether it's cube-specific.
  - Secondary: BC-floor and CRL-free scaling curves come for free; note whether gciql or
    biql degrades faster (matrix-completion story predicts biql is *more* robust to fewer
    (s,g) samples at fixed marginal coverage — a falsifiable side prediction worth stating
    in the writeup either way).

### E2 — ManiSkill play-resample at cube scale (the "MS is pretty small" fix)

- **Collector:** new `data_collection/collect_ms_play_resample.py`, built from
  `collect_ms_play.py` (newt TD-MPC2 oracle, obs padding, Vulkan ICD env) + every v3 lesson
  from `collect_mw_play_resample.py` / `hf_dataset/README_play.md`:
  - timer-only target re-draw (never on success), target moved through the env's own goal
    mechanism, stored goal channel clipped exactly as `step()` would;
  - trajectory length **125 steps = 5× the 25-step eval horizon** (matching cube's 5.0 and MW
    v3's 5.0 — NOT the upstream 400/25 = 16×); re-draw timer tuned so segment_len/horizon
    lands ≈ 0.42 (start at 10–11 steps, verify);
  - **no epsilon-random actions**; Gaussian action noise σ ≤ 0.1;
  - reject-and-retry on escaped/unhealthy trajectories (cube knocked off workspace = the MS
    analog of the MW absorbing-escape defect); record `rejected_trajectories` in meta;
  - no zero-velocity synthetic frames — splice only the goal dims on re-draw;
  - record `goal_switch`.
- **Scale:** ≥ 8000 trajs × 125 = **1,000,000 transitions** per task, `seed=0`, tasks
  `ms-push-cube`, `ms-pick-cube` → `$SCRATCH/fb-data/ms-play-resample/`.
- **Verification gate before any training:** extend
  `scripts/compare_play_resample_to_ogbench.py` with an `analyse_ms` (same columns:
  relocations/traj, segment/H, rest fraction, entropy ratio, stale transitions, escape count,
  P(goal satisfied at dt) table, success-rule agreement against the goal dims). The table
  must line up with cube the way the MW v3 column does. Do not launch the square until it
  does.
- **Matrix:** {bcflow, gciql+flowbc, iql_bilinear-d50+flowbc} × {ms-push-cube, ms-pick-cube}
  × 3 seeds × 300k, `obs_mask_dims` + `goal_mask_dims` per task (dims above),
  `p_randomgoal=0`, `eval_goal_mode=synth`, **50 eval episodes** (Arm E's 20 gave ±0.1
  spread) = **18 runs**. Plus an unmasked BC-only control (2 tasks × 3 seeds = 6 runs) to
  re-confirm the censoring baseline on the new data.
- **Pre-registered readings:** this is the Arm E rerun with all of C1 (size), C2 (synth eval
  goals are in-support by construction — the env draws eval targets from the same goal space
  collection sampled), C3 (mask) and C4 (synth) simultaneously repaired. A null here is,
  finally, informative.

### E3 — MW v3 definitive square (the 1M dataset nobody has trained on)

- **Data:** `$SCRATCH/fb-data/play-resample` (v3, 2000×500, published to HF) and, as a
  second regime, `play-resample-hard`. v1 archives must not be touched.
- **Matrix:** {bcflow, gciql+flowbc, iql_bilinear-d50+flowbc} × {mw-push, mw-pick-place}
  × 3 seeds × 300k, Arm C recipe + new eval:
  `obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] p_randomgoal=0 eval_goal_mode=synth`,
  50 eval episodes = **18 runs** (+18 on `play-resample-hard` if the first square is flat —
  the hard regime is what opened headroom before).
- **Pre-registered readings:**
  - biql > gciql > BC with positive guidance → cross-env replication on matched-size,
    goal-diverse, channel-repaired data; RQ1 generalizes.
  - All flat at a high BC floor → headroom censoring again (near-oracle actions) → the hard
    regime square is the fallback and was pre-planned, not post-hoc.
  - Separation appears but only after E5 radius tightening → C5 was the mask, see E5.

### E4 — Coverage measurement refresh (analysis only, no training)

- Regenerate the goal-coverage figure from **v3** MW data, the **new** E2 MS data, and cube;
  archive the v1-based `goal_coverage.png`.
- Add a scalar in-supportness metric per (dataset, eval protocol): for each eval-episode
  goal, distance to the nearest training achieved-goal ÷ that env's success radius; report
  median and P95. Under `eval_goal_mode=synth` this should be ≈ in-support everywhere; the
  point is to *show* C2 is controlled, with Arm D as the causal anchor for what happens when
  it isn't. Extend the compare script's dt table to all three envs in the same units.

### E5 — Matched-discriminability re-scoring (analysis on E2/E3 logs)

- From the per-episode distance logs (E0.2), recompute success at a sweep of radii and find
  the radius r′ per MW/MS task at which P(relabeled goal satisfied at 1–2×H) matches cube's
  0.013 (compute from the compare script's displacement distributions).
- Report every E2/E3 cell at both the native radius and r′. Pre-registered: if biql–gciql
  separation appears only at r′, C5 (env discriminability) was hiding a real effect and the
  paper should report matched-radius numbers; if cells stay flat at r′ too, parity on MW/MS
  is genuine.

## 5. Decision matrix

| E1: gap at 48k cube | E2: gap on 1M MS (masked+synth) | conclusion |
|---|---|---|
| persists | appears | Size (+ broken channel) explains every prior null. RQ1: bilinear wins on play data generally once the goal channel is honest. Strongest possible paper claim. |
| persists | still null (also at r′, E5) | Size and coverage exonerated; advantage is cube-geometry-conditional. Merge with Arm A boundary story (γ-band, goal-latent dim): the claim becomes conditional and the paper says so. |
| collapses | appears | Advantage is a data-scale phenomenon everywhere; prior nulls were starvation artifacts. Claim survives with an explicit scale requirement. |
| collapses | still null | Advantage needs big data AND cube-like structure. Report the boundary honestly; RQ2's completion mechanism remains supported by the probe/BoN evidence but its precondition set grows. |

E3 gives an independent MW verdict under the same matrix logic (with the hard-regime square
as the headroom fallback). Any deciding cell with |Δ| < 2sd at 3 seeds: extend to 5 seeds
before interpreting.

## 6. Operational gotchas (all previously hit — do not rediscover)

- **Per-TAG SAVE_DIR subdirs are mandatory.** Run dirs do not encode overrides; two configs
  writing the same dir silently clobber (cost us the first cube-double/scene biql rows).
- **Never gate sbatch on `--dependency` of a short job** — SLURM purges completed jobs and
  the dependent submission dies ("Job dependency problem"). Verify outputs, then submit.
- **`make_config` takes a dotlist LIST** — passing a string char-iterates and silently drops
  overrides. `rollout_probe` uses `action="append"`; copy that pattern.
- **eval.csv files can be overwritten/corrupted by duplicate passes.** Job `.o` logs are
  ground truth; take the highest job ID per (tag, seed) within a generation.
- **Login nodes cannot run JAX** (100-thread ulimit). Everything through SLURM
  (`scripts/vista_train.slurm`, `ACCOUNT=ASC26008`).
- **Collectors invoked by file path need repo root on sys.path** (v3 collector has the
  incantation; copy it).
- **Never `return` a value from an absl `main`.** `app.run(main)` does `sys.exit(main(argv))`,
  so a returned list becomes the exit status: Python prints it to stderr and exits 1. Job
  903390 reported "12 failure(s)" while all 12 re-evals had in fact succeeded and written
  their records (`scripts/reeval_checkpoint.py`, fixed).
- MS collection needs `VK_ICD_FILENAMES=$SCRATCH/vulkan/nvidia_icd.json` and the newt/tdmpc2
  paths hardcoded at the top of `collect_ms_play.py`.
- The eval-RNG restore (`env/unified.py::_global_np_seed`) is in place, so NEW runs have
  honest seed error bars — but do not compare seed-sd against pre-fix runs.
- Old play-resample training results (`$SCRATCH/factored-fb-exp-play-resample*`) are v1-data
  runs; never mix them into v3 tables.

## 7. Order of execution and cost

1. **E0** (eval synth + distance logging + cube truncation knob) — half a day, smoke-tested.
   Start the **E2 collector** in parallel; it is the longest pole.
2. **E3** launch (data already exists): 18 runs.
3. **E1** launch: 28 runs.
4. **E2**: collect → verification gate (compare script) → 24 runs.
5. **E4/E5**: analysis passes over the artifacts; no GPU time beyond re-eval.

≈ 70 training runs × ~3 h (300–500k steps, prior campaigns' walltime) + ~1 GPU-day of MS
collection. All 3-seed cells first; 5-seed extensions only where the decision matrix demands.

## 8. Addenda found while executing

### 8.1 — Arm E's `ms-pick-cube` mask was not a mask (correction to C3)

Verifying the ManiSkill obs layout for E0.1 turned up a leak that changes what Arm E means.
ManiSkill 3's `PickCube-v1` state observation is

    qpos9 | qvel9 | is_grasped1 | tcp_pose7 | goal_pos3 | obj_pose7 | tcp_to_obj3 | obj_to_goal3
    0:9     9:18    18            19:26       26:29       29:36       36:39         39:42

so it carries **two derived difference channels**. On the packed upstream data,
`obs[39:42] == obs[26:29] − obs[29:32]` on **100.00%** of the 48,000 frames (max abs error
0.0). Masking `[26,27,28]` therefore removes nothing: the goal is recovered exactly as
`obs[29:32] + obs[39:42]`. Regressing each goal dim on all remaining obs dims confirms it —
R² = 1.000000, residual rms 0.000000 on all three dims.

Consequences:

- The "masked" Arm E `ms-pick-cube` rows are **unmasked runs**. They are not evidence about
  a repaired goal channel and must not be cited as such; the `ms-push-cube` rows are fine
  (same regression gives R² = 0.02 / 0.16 / 0.05, no leak).
- Every `ms-pick-cube` run in this spec uses `obs_mask_dims=goal_mask_dims=[26,27,28,39,40,41]`.
- `ms-push-cube`'s 35-dim layout (`qpos9 | qvel9 | tcp_pose7 | goal_pos3 | obj_pose7`) has no
  derived channels, so `[25,26,27]` stands.
- Metaworld was re-audited the same way on the v3 data: R² = 0.88 / 0.77 (push) and
  0.92 / 0.84 / 0.91 (pick-place) — high, but that is the *object* predicting the goal because
  the play policy parks it there (dwell_frac 0.86), not an algebraic channel. `[36,37,38]`
  stands.

### 8.2 — E0 as built

- `eval_goal_mode: fixed|synth` and the per-episode distance log live in
  `env/unified.py`; the synth construction is `synth_goal_frame()`, shared with
  `eval/bon_probe.py` so the probe and trained-policy eval cannot drift.
  `_GOAL_LAYOUT` is the single place the per-family obs slots are written down.
- Per-episode distances are **not** in `eval.csv` (CsvLogger freezes its header on the first
  row, and wandb takes scalars). `impls/main.py` drains them to
  `<run_dir>/eval_episodes.jsonl`, one append-only record per (step, scope), carrying
  `goal_dist_min`, `goal_dist_final` and `success` per episode. That file is E5's input.
  `eval.csv` gains the two means as `goal_dist_min` / `goal_dist_final`.
- `dataset_num_transitions` lives in `env/ogbench.py::truncate_whole_trajectories`. It counts
  ROWS (OGBench's compact layout stores T+1 rows per length-T trajectory, and 1,001,000 is
  the row count the §1 table quotes), keeps whole trajectories, and draws them from one fixed
  `RandomState(0)` permutation truncated at the requested count — so the rungs **nest**
  (48k ⊂ 100k ⊂ 250k) and a size effect cannot be confounded with a change of which
  trajectories are present.
- Gate: `sbatch -A ASC26008 scripts/e0_smoke.slurm` (unit tests in `tests/env/` + the
  real-cube no-op/ladder check + 200-step runs on MW, cube and both MS tasks).
- `p_randomgoal=0` is spelled `value_p_randomgoal=0.0 value_p_trajgoal=0.8` (Arm C's
  spelling). GCDataset asserts the three value-goal probabilities sum to 1, so dropping the
  random-goal mass without moving it to trajgoal aborts at startup.

### 8.3 — E2's `ms-push-cube` arm cannot be cube-matched (structure, not size)

PushCube's own reset always places the goal at `cube_xy + [0.1 + goal_radius, 0]`, so its
TD-MPC2 oracle only ever learned to push in **+x** and cannot pull. Each relocation must
therefore advance the cube ~0.2 m down a reachable band about 0.4 m long, and chaining
cube-scale relocation counts marches the cube out of the workspace. Measured on the upstream
400-step set:

| | ms-push-cube | ms-pick-cube | cube-single |
|---|---|---|---|
| dwell, first 100 steps → last 100 | 0.070 → **0.006** | 0.300 → 0.290 | — |
| median object–goal distance | **0.306** (radius 0.10) | 0.021 (radius 0.025) | — |
| object at rest | **0.900** | 0.028 | 0.532 |
| P(goal satisfied at 1–2 × H) | **0.823** | 0.023 | 0.013 |
| segments whose goal was reached | **0.313** | 0.962 | — |
| segments with the goal BEHIND the cube | **0.63** (reached on 0.176) | — | — |

So the upstream push-cube half of Arm E is largely dead data, independently of size. Decision
taken (2026-08-10): collect push-cube anyway, at the **one relocation per trajectory** its
geometry admits (50 steps = 2.0 × H, segment 25 = 1.0 × H), and report it as a
**structure-mismatched control**, not a cube-matched replication. `relocations_per_traj` and
`segment_over_horizon` in the collection meta state the mismatch. `ms-pick-cube` is the
cube-matched cell: 125 steps = 5.00 × H, segment 11 = 0.44 × H, 8000 trajectories.

### 8.4 — What exists, and how to fire it

| step | command | state |
|---|---|---|
| E0 gate | `sbatch -A ASC26008 scripts/e0_smoke.slurm` | **PASSED**, job 902006, 0 failures |
| E2 collection | `sbatch -A ASC26008 scripts/collect_ms_play_resample.slurm` | **DONE**, 1M/task at `$SCRATCH/fb-data/ms-play-resample`; gate ran at the tail |
| E3 | `bash scripts/launch_deconfound.sh e3` (18 runs) | **DONE**, 18/18 at 300k |
| E1 | `bash scripts/launch_deconfound.sh e1` (28 runs) | **DONE**, 28/28 at 500k |
| E2 | `bash scripts/launch_deconfound.sh e2` (24 runs) | **DONE**, 24/24 at 300k |
| E4 | `sbatch -A ASC26008 scripts/goal_coverage.slurm` | **DONE**, job 904072, all 5 panels, 0 failures; v1 figure archived to `figs/goal_coverage-v1.png` |
| E5 | `python scripts/rescore_radius.py --data <npz> --task <task> --runs '<glob>'` | **DONE** for all four tasks, `results/deconfound/e5-*.json` |

`DRYRUN=1` on the launcher prints the grid and submits nothing. The launcher is idempotent
(skips finished run dirs and queued job names) and waits out the submit cap.

## 9. Results

### 9.1 — E1: the cube advantage is a data-scale phenomenon, and it dies at MS scale

Final eval, mean ± sd over 3 seeds at 500k steps.

| train rows | BC | GCIQL | biGCIQL | biGCIQL − GCIQL |
|---|---|---|---|---|
| 48,000 (MS-upstream scale) | 0.052 ± .024 | 0.315 ± .052 | 0.343 ± .027 | **+0.028** |
| 100,000 (MW-v1 scale) | 0.055 ± .008 | 0.508 ± .028 | 0.701 ± .012 | +0.193 |
| 250,000 | 0.072 ± .021 | 0.704 ± .062 | 0.904 ± .016 | +0.200 |
| 1,001,000 (anchor, 1 seed) | — | — | 0.908 | — |

The full-size anchor reproduces the historical ~0.907 headline, so `dataset_num_transitions`
is a no-op at full size **at the training level**, not only bit-for-bit on the arrays.

Reading: the gap **collapses at 48k** (+0.028, inside the seed spread) and is already intact
at 100k. So Arm E's ManiSkill null was data-starved and says nothing about the bilinear head;
the MW v1 square at ~100k was **not** starved and its null stands. The spec's secondary
prediction — that the matrix-completion story makes biGCIQL *more* robust to fewer (s, g)
samples — is **falsified**: biGCIQL degrades faster than GCIQL below 100k
(0.904 → 0.343 = −62%, against 0.704 → 0.315 = −55%).

### 9.2 — E3: on repaired MW data the bilinear advantage does not replicate; it reverses

| | BC | GCIQL | biGCIQL |
|---|---|---|---|
| mw-push | 0.567 ± .050 | **0.727 ± .050** | 0.433 ± .083 |
| mw-pick-place | 0.293 ± .031 | **0.947 ± .061** | 0.893 ± .046 |

With size (1M), goal channel (`obs_mask_dims`) and eval conditioning (`eval_goal_mode=synth`)
all repaired, GCIQL beats biGCIQL on both tasks, and on mw-push biGCIQL falls **below the BC
floor**. The hard-regime fallback square is not needed: BC is at 0.293/0.567, so there is
ample headroom and the square is not censored.

### 9.3 — E2: both ManiSkill cells are censored, and no radius rescues them

| | BC (masked) | BC (unmasked) | GCIQL | biGCIQL |
|---|---|---|---|---|
| ms-pick-cube | 0.020 | 0.967 | 0.040 | 0.040 |
| ms-push-cube | 0.987 | 0.980 | 0.600 | 0.767 |

- **pick-cube is a precision floor.** `goal_dist_min` falls 0.186 → 0.079 by 75k and then
  plateaus at ~0.094 m against a **0.025 m** radius; unmasked BC reaches 0.011–0.014 m. So the
  masked policies learn direction but not precision. It is not a goal-support problem: eval
  targets sit **0.11 radii (P95 0.28, max 0.63)** from the nearest training achieved goal. It
  is not a synth-goal-frame artefact either — re-evaluating all 12 checkpoints against the
  corrected `synth_goal_frame` (tcp keeps its grasp offset rather than collapsing onto the
  target) reproduced the same 0.02–0.04.
- **push-cube is ceiling-censored.** Masked BC 0.987 ≈ unmasked BC 0.980: the mask changes
  nothing, because the structure-mismatched 1-relocation design (§8.3) leaves the goal always
  +x ahead, so an unconditional push policy solves it. Both critics score *below* that floor.

Radius sweep over the E0.2 per-episode logs (fraction of episodes reaching within k × the
native radius), which is what rules out a hidden effect:

| task | cell | 0.5× | 1× | 2× | 3× | 4× | 6× |
|---|---|---|---|---|---|---|---|
| ms-pick-cube | BC-nomask | 0.83 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 |
| | BC | 0.01 | 0.04 | 0.25 | 0.42 | 0.57 | 0.84 |
| | GCIQL | 0.02 | 0.06 | 0.17 | 0.26 | 0.41 | 0.65 |
| | biGCIQL | 0.03 | 0.07 | 0.27 | 0.43 | 0.57 | 0.76 |
| ms-push-cube | BC | 0.16 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| | GCIQL | 0.09 | 0.60 | 0.99 | 1.00 | 1.00 | 1.00 |
| | biGCIQL | 0.21 | 0.77 | 1.00 | 1.00 | 1.00 | 1.00 |
| mw-push | BC | 0.31 | 0.57 | 0.92 | 1.00 | 1.00 | 1.00 |
| | GCIQL | 0.47 | 0.73 | 0.99 | 0.99 | 1.00 | 1.00 |
| | biGCIQL | 0.25 | 0.43 | 0.82 | 0.93 | 0.98 | 1.00 |
| mw-pick-place | BC | 0.06 | 0.29 | 0.79 | 0.98 | 1.00 | 1.00 |
| | GCIQL | 0.77 | 0.95 | 0.95 | 0.99 | 1.00 | 1.00 |
| | biGCIQL | 0.31 | 0.89 | 0.99 | 1.00 | 1.00 | 1.00 |

On pick-cube, biGCIQL tracks masked BC at **every** tolerance (0.27/0.43/0.57 against
0.25/0.42/0.57) and GCIQL sits below it: goal conditioning contributes nothing there, at any
radius. **In all four off-cube cells, at every tolerance, the bilinear critic never beats the
monolithic one.**

### 9.4 — E4: C2 is controlled everywhere, and C4 was worth 1–5 success radii

`results/deconfound/goal-coverage.json`, figure regenerated at
`PAPER/decoding-bilinear/figs/goal_coverage.png` (the v1-based version is archived as
`goal_coverage-v1.png`, not overwritten). Distances are in units of each env's success radius.

| dataset | eval goal → nearest training achieved goal (med / P95) | `fixed` conditioning error (med / P95) |
|---|---|---|
| cube-single | 0.03 / 0.06 | — (OGBench draws its own eval goals) |
| ms-pick-cube | 0.09 / 0.32 | **5.23 / 7.12** |
| ms-push-cube | 0.01 / 0.02 | 1.46 / 2.26 |
| mw-pick-place | 0.03 / 0.37 | 1.20 / 2.17 |
| mw-push | 0.03 / 0.24 | 1.09 / 2.28 |

Under `eval_goal_mode=synth` every eval goal is deep inside the training goal marginal — P95 is
at worst 0.37 of a success radius — so **C2 is controlled in all four squares** and cannot
explain any of §9.1–9.3. Arm D remains the causal anchor for what happens when it is not.

The second column is C4 as a scalar, and it is why E0.1 was worth doing: the old `fixed`
protocol conditioned every episode on a cached dataset frame **1.1–5.2 success radii** away
from the target the env was actually scoring. On ms-pick-cube that is ~0.13 m against a 0.025 m
radius — the critic was being asked for a goal the episode never had.

It also independently confirms §9.3's diagnosis: ms-pick-cube's masked floor is **not** a goal
support problem (0.09 radii median, P95 0.32, well inside one radius).

### 9.5 — E5: C5 was not hiding anything, and the new MS data does not need it

| task | native radius | P(satisfied at 1–2 × H) | r′ | r′ / native |
|---|---|---|---|---|
| mw-push | 0.050 | 0.257 | 0.0083 | 0.17 |
| mw-pick-place | 0.070 | 0.147 | 0.0270 | 0.39 |
| ms-pick-cube | 0.025 | **0.007** | 0.0305 | 1.22 |
| ms-push-cube | 0.100 | **0.009** | 0.1043 | 1.04 |

MW needs a much tighter radius to match cube's 0.013, and at r′ the ordering is unchanged and
GCIQL's lead **widens** (mw-pick-place GCIQL 0.53 vs biGCIQL 0.18 mean over seeds). Both new
1M ManiSkill collections are already *more* discriminating than cube (r′ ≥ native), which is
the collector's structural matching working — so C5 is not a confound on the E2 data and
cannot be the reason those cells are flat.

### 9.6 — Verdict against the §5 decision matrix

**E1 collapses at 48k; E3 is not merely null but reversed; E2 cannot arbitrate** (both cells
censored, and the radius sweep shows no hidden effect). That is the matrix's bottom-right row,
with E3 strengthening it: the bilinear advantage needs **big data AND cube-like structure**,
and off cube the monolithic critic is the better one. RQ1 as stated ("bilinear beats
monolithic on play data generally") is **refuted**. RQ2's completion mechanism keeps its
probe/BoN support on cube, but its precondition set now includes dataset scale ≥ ~100k and
cube-like goal geometry, and the falsified sample-efficiency side-prediction (§9.1) should be
reported.

Open: E2's two cells are censored from opposite ends and neither measures a critic. Rescuing
them means changing the task, not the analysis — a looser pick-cube tolerance would be a
different benchmark, and an un-censored push-cube needs a goal the oracle can reach in more
than one direction. Reporting both as censored, with §9.3's sweep as the evidence, is the
honest option and costs no further compute.
