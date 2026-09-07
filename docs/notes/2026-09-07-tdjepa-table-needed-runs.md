# Runs still needed for the TD-JEPA-protocol OGBench proprio table (2026-09-07)

**Decision 2026-09-07: hard target only; soft-target arms are out of scope.** Every Density-FB-family
launch below carries `--override measure_density_hard=true` (z runs also
`--override measure_density_z_actor_reward=latent_centered`) so `config.json` self-evidences the cell.
A 500k checkpoint may be resumed to 1M only if its `config.json` records `measure_density_hard: true`
(the TD target must not change mid-run); a checkpoint recording `false` is not resumed — it is
replaced by a fresh hard run from step 0 into `exp/tdjtab/`. Audit 2026-09-07: none of the local
`exp/` density checkpoints is soft (see §1), so no fresh replacement runs were added.

Stack: `/home/adityamohan/git/Factored-FB-seam` @ `density-fb`. Entry point is
`python -m impls.main` (`impls/main.py`); run dir = `<save_dir>/<env_name>_<critic>_<actor>_seed_<n>`
(`impls/utils/log_utils.py::get_exp_name`). `--domain X` resolves `env/domains/X.yaml` first,
then `impls/configs/domain/X.yaml` (`env/config.py::_load_domain`). Only `cube_single` /
`cube_double` exist under `impls/configs/domain/`; `env/domains/` has `antmaze.yaml`
(= antmaze-medium-navigate, `eval_aggregate: true`) and `scene.yaml` (scene-play, aggregate).

Protocol flags (as in `scripts/launch_hilp_tdjc.sh` / `launch_partb_tier1.sh`):
`--actor flowbc --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --auto_resume
--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10`
(Route B = `eval_relabel.csv`, Route A = `eval.csv`, both written by every run). Env:
`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl`,
python `/dev/shm/.venv-jax/bin/python`.

Column → critic config (all verified in `impls/configs/critic/`): FB = `fb_tdj.yaml` (ortho 1000, lr_b 1e-4);
HILP = `hilp_tdjc.yaml` (group_lr phi 1e-4, p_rand 0.375); RLDP = `rldp.yaml` (ortho 0.01, lr_b 1e-4;
`rldp_tdj.yaml` exists but is the DMC-shaped variant: batch 1024, L_dim 256, γ 0.98 — do NOT use for OGBench);
Density-FB goal = `density_fb.yaml` (obj-coord B) / `density_fb_state.yaml` (full-state B);
Density-FB z = `density_fb_z.yaml` (default now `measure_density_z_actor_reward: latent_centered`,
`measure_density_hard: true` since da74bfd 2026-09-06 18:36; state variant `density_fb_z_state.yaml`);
HILP-density = `hilp_density_tdjc.yaml`; RLDP-density = `rldp_density.yaml`.

Hard-target defaults (verified 2026-09-07): `measure_density_hard: true` in `density_fb.yaml:21`,
`density_fb_state.yaml:22`, `density_fb_z.yaml:17`, `density_fb_z_state.yaml:18` (and `density_fb_pixels.yaml:45`);
code fallback if the key is absent is also `True` (`impls/critics/density_fb.py:1316`,
`config.get("measure_density_hard", True)`); the switch is consumed at `density_fb.py:911` (`if self.hard:` →
one-hot at the row's own successor, else the eps-kernel). `hilp_density_tdjc.yaml` and `rldp_density.yaml`
have NO soft/hard key and their critics have no switch: the immediate CE target is hard-coded as
`k = jnp.eye(B)` (`impls/critics/hilp_density.py:117`, `impls/critics/rldp_density.py:138`), i.e. they are
hard-only. Passing `measure_density_hard=true` to them is accepted (OmegaConf dotlist merge,
`env/config.py:44`; only `_INERT_ALIASES` is rejected) and inert — kept for uniform `config.json` evidence.
bc_coeff lives in the actor yaml (`impls/configs/actor/flowbc.yaml: actor.bc_coeff: 3.0`); antmaze needs
`--override actor.bc_coeff=0.3`.

## 1. Status matrix (env × method)

Legend: D1M-R = done 1M with Route-B relabel eval; D1M-Rr = done 1M, Route B via eval-only rescore
(`scripts/relabel_rescore.py`, no in-run `eval_relabel.csv`); D500k = 500k goal-cond only; CONT = contaminated;
MISSING = nothing. Evidence dirs are under `exp/` of the seam repo unless noted.

| env | FB (fb_tdj) | HILP (hilp_tdjc) | RLDP (rldp) | DFB goal (density_fb/_state) | DFB z (density_fb_z fixed) | HILP-dens (hilp_density_tdjc) | RLDP-dens (rldp_density) |
|---|---|---|---|---|---|---|---|
| cube-single-play-v0 | **CONT** — `cube-single-play-v0_fb_tdj_flowbc_seed_{0,1,2}` (config.json records lr_b 1e-5 for s0,s1 and 1e-4 for s2; `eval_relabel.csv` 61 rows vs 58 in s0/s2 = two arms appended into one dir) | D1M-R `cube-single-play-v0_hilp_tdjc_flowbc_seed_{0,1,2}` (.83) | D1M-Rr `cube-single-play-v0_rldp_flowbc_seed_{0,1,2}` params_1000000 + `analysis/relabel_rescore_cs1m.jsonl` | D500k `..._density_fb_flowbc_seed_{0,1,2}`, `..._density_fb_state_flowbc_seed_{0,1,2}` (params_500000; config.json `measure_density_hard: true` all 6 — resumable) | fixed: 500k ×3 local `exp/zfix/lc_hard/` (config.json `measure_density_hard: true` + `latent_centered`, relabel csv present); **1M ×3 on cluster only** (per audit; verify hard+latent_centered in config.json before quoting). ⚠ local `exp/partb-tier1/cube-single-play-v0_density_fb_z_flowbc_seed_{0,1,2}` (1M, relabel) are hard (`measure_density_hard: true`) but the PRE-FIX 'latent' variant: config.json has no `measure_density_z_actor_reward` (default 'latent', density_fb.py:1134) and ckpts are dated 09-06 17:58 < fix commit 18:36 | D1M-R `..._hilp_density_tdjc_flowbc_seed_{0,1,2}` (params_1000000; hard-only critic, no key in config.json) | D1M-Rr `..._rldp_density_flowbc_seed_{0,1,2}` (params_1000000; hard-only critic) + `relabel_rescore_cs1m.jsonl` |
| cube-double-play-v0 | MISSING (queued in `scripts/launch_partb_tier1.sh`, never run) | D1M-R `cube-double-play-v0_hilp_tdjc_flowbc_seed_{0,1,2}` | D500k `cube-double-play-v0_rldp_flowbc_seed_{0,1,2}` (params_500000, no relabel) | D500k `..._density_fb_flowbc_seed_{0,1,2}`, `..._density_fb_state_...` (config.json `measure_density_hard: true` all 6 — resumable) | fixed: cluster 1 seed only (verify hard+latent_centered). ⚠ local `exp/partb-tier1/cube-double-play-v0_density_fb_z_flowbc_seed_{0,1,2}` (1M) are hard but PRE-FIX (same evidence as above) | D1M-R `cube-double-play-v0_hilp_density_tdjc_flowbc_seed_{0,1,2}` (params_1000000; hard-only critic) | D500k `cube-double-play-v0_rldp_density_flowbc_seed_{0,1,2}` (hard-only critic — resumable) |
| scene-play-v0 | MISSING | MISSING | MISSING | D500k bridge stack only (main worktree, `scripts/results_map.py` scene groups; not the seam) | MISSING | MISSING | MISSING |
| puzzle-3x3-play-v0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| antmaze-medium-navigate-v0 | MISSING | MISSING | MISSING | D500k bridge stack only (spec22, .669±.240, 3 seeds; not the seam) | MISSING | MISSING | MISSING |
| antmaze-large-navigate-v0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| antmaze-medium-stitch-v0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| antmaze-large-stitch-v0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| antmaze-medium-explore-v0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

Hard/soft audit of every local density dir (`config.json`, 2026-09-07): density_fb ×6, density_fb_state ×6,
partb-tier1 density_fb_z ×6, zfix/lc_hard ×3 all record `measure_density_hard: true`; hilp_density_tdjc ×6 and
rldp_density ×6 have no key (hard-only critics). No soft checkpoint exists locally (`exp/zfix/` holds only
`lc_hard`, `lc_hard_state`; the launcher's `lc_soft`/`l_soft` arms never produced dirs here), so no DONE
cell is disqualified and no resume is blocked by the decision.

Route-B data check: `grep -l . exp/*/eval_relabel.csv` — only hilp_tdjc / hilp_density_tdjc / fb_tdj (contaminated)
/ partb-tier1 (pre-fix) / zfix (500k) carry one. `inferred_cond_goals_dropped` > 0 in all inspected final rows
(e.g. partb-tier1 cs s0 = 3733, cd s2 = 10000).

## 2. Runs still needed (tier 1 = chosen cell × seeds 0,1,2)

Common prefix (put in a launcher; identical to `launch_partb_tier1.sh`'s body):

    RUN="CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
      /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000"
    RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
    CD_HIT="--override measure_hit_spec=19:22:0.4,28:31:0.4"     # cube_double, density variants only
    NAV="--override actor.bc_coeff=0.3"                            # antmaze only (td_jepa sweep_antmaze bc_coeff 0.3)
    HARD="--override measure_density_hard=true"                    # EVERY density-variant run (decision 2026-09-07)
    ZFIX="--override measure_density_z_actor_reward=latent_centered $HARD"   # density_fb_z(_state) runs

    # per-env hit specs for the density variants (derived + verified 2026-09-07, see §5 item 3):
    SCENE_HIT='--override '"'"'measure_hit_spec="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"'"'"''
    PUZ_HIT='--override '"'"'measure_hit_spec="19:21:0.5,23:25:0.5,27:29:0.5,31:33:0.5,35:37:0.5,39:41:0.5,43:45:0.5,47:49:0.5,51:53:0.5"'"'"''
    ANT_HIT='--override '"'"'measure_hit_spec="0:2:0.5"'"'"''   # single block MUST be quoted (sexagesimal landmine, _measure_common.py:44)

**Materialized 2026-09-07:** every tier-1 command below is written out, fully expanded, one per line, in
`scripts/tdjtab_manifest.sh` (167 lines, id `# <domain>/<critic>/s<seed> :: <cmd>`; the 12 resumes carry a
trailing `## RESUME ...` note and no `--save_dir`). `scripts/launch_tdjtab.sh '<id regex>' '<gpu list>'`
launches matching lines round-robin in tmux session `tdjtab` (logs `logs/tdjtab/<domain>/<critic>/s<seed>.log`,
skips runs whose `params_1000000.pkl` exists; `DRY=1` prints the assignment). Nothing launched yet.

Domain yamls **created 2026-09-07** in `impls/configs/domain/` (structure of `cube_single.yaml`: the single
`env_name` key plus a comment; per-task eval over the 5 task_infos, NOT `eval_aggregate`): `scene_play.yaml`
(scene-play-v0), `puzzle_3x3.yaml` (puzzle-3x3-play-v0), `antmaze_medium_navigate.yaml`, `antmaze_large_navigate.yaml`,
`antmaze_medium_stitch.yaml`, `antmaze_large_stitch.yaml`, `antmaze_medium_explore.yaml`. Verified: each composes
through `env.config.make_config`, resolves to an npz present in `~/.ogbench/data/`, and none is shadowed by
`env/domains/*.yaml` (those are `scene.yaml` / `antmaze.yaml`, different names).

Save-dir convention: `--save_dir exp/tdjtab/` for every new run so nothing lands on top of the contaminated
`exp/cube-single-play-v0_fb_tdj_*` dirs or the pre-fix `exp/partb-tier1/`. `--auto_resume` reads only its own
dir, so a fresh `save_dir` is a clean start.

### FB (`fb_tdj`) — 24 runs
    for s in 0 1 2; do $RUN --critic fb_tdj --domain cube_single --seed $s --save_dir exp/tdjtab $RELABEL; done   # RE-RUN (contaminated)
    for s in 0 1 2; do $RUN --critic fb_tdj --domain cube_double --seed $s --save_dir exp/tdjtab $RELABEL; done
    for s in 0 1 2; do $RUN --critic fb_tdj --domain scene_play  --seed $s --save_dir exp/tdjtab; done            # RELABEL blocked (see §5)
    for s in 0 1 2; do $RUN --critic fb_tdj --domain puzzle_3x3  --seed $s --save_dir exp/tdjtab; done            # RELABEL blocked
    for d in antmaze_medium_navigate antmaze_large_navigate antmaze_medium_stitch antmaze_large_stitch antmaze_medium_explore; do
      for s in 0 1 2; do $RUN --critic fb_tdj --domain $d --seed $s --save_dir exp/tdjtab $NAV; done; done       # RELABEL blocked
(`fb_tdj.yaml` is env-agnostic; ortho 1000 is td_jepa's sweep value for antmaze too.)

### HILP (`hilp_tdjc`) — 21 runs
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (antmaze with $NAV). Same loops as FB with --critic hilp_tdjc.
`hilp_tdjc.yaml` carries `measure_hit_spec: "19:22:0.4"` but `backward_input: state`, so the spec is inert
(hilp.py:476-490 reads it only for `backward_input=hit_spec`) — no config change needed for non-cube envs.
Optional: a `hilp_tdjc_nav.yaml` is NOT needed; bc_coeff is the only nav delta and it is an actor override.

### RLDP (`rldp`) — 24 runs
    cube_double ×3: $RUN --critic rldp --domain cube_double --seed $s $RELABEL   # NO --save_dir: resumes exp/cube-double-play-v0_rldp_flowbc_seed_$s from params_500000 → 1M (+500k each)
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (with $NAV) into exp/tdjtab.
`rldp.yaml` ortho 0.01 = the chosen cell for both manip (sweep {0.01,0.1}) and antmaze (sweep {0.001,0.01}); no
per-env config needed. Resume caveat: the existing cube_double dirs were trained WITHOUT the relabel override;
adding it on resume only adds Route-B rows from 600k on — final-step gate still holds.

### Density-FB goal-conditioned (`density_fb`, obj-coord B = paper method) — 24 runs (+24 optional `density_fb_state`)
    for s in 0 1 2; do $RUN --critic density_fb --domain cube_single --seed $s $HARD; done                    # NO --save_dir: resumes exp/cube-single-play-v0_density_fb_flowbc_seed_$s from params_500000 (hard) → 1M
    for s in 0 1 2; do $RUN --critic density_fb --domain cube_double --seed $s $HARD $CD_HIT; done            # resumes exp/cube-double-play-v0_density_fb_flowbc_seed_$s (hard) → 1M
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (with $NAV, always $HARD, the per-env $*_HIT, --save_dir exp/tdjtab) — hit specs derived, unblocked (§5 item 3)
    optional density_fb_state: same loops with --critic density_fb_state $HARD (cube resumes are hard, verified)
Route B does not exist for this variant (no `infer_cond` path) — never pass $RELABEL; it is the Route-A column only.
The six cube resumes are allowed only because their `config.json` records `measure_density_hard: true` (checked
2026-09-07); `$HARD` on resume is a no-op that re-states the cell.

### Density-FB z (`density_fb_z`, latent_centered + hard) — 23 runs
    cube_single: 0 new (cluster 1M ×3) — but PULL the cluster ckpts and confirm config.json has
      `measure_density_hard: true` + `measure_density_z_actor_reward: latent_centered`; do not use exp/partb-tier1 (pre-fix).
    for s in 1 2; do $RUN --critic density_fb_z --domain cube_double --seed $s --save_dir exp/tdjtab $ZFIX $CD_HIT $RELABEL; done
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (with $NAV, always $ZFIX, the per-env $*_HIT, --save_dir exp/tdjtab) — hit spec unblocked; Route B still needs the relabel reward (§5 item 1)
The yaml default is already the fixed cell (`git show da74bfd`); `$ZFIX` restates it so config.json is self-evidencing.
Never resume `exp/zfix/lc_hard/` (500k) into the table: it was trained at `--eval_interval 50000` under a different
save_dir; the table's cube-single z cell is the cluster 1M set.

### HILP-density (`hilp_density_tdjc`) — 21 runs
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (with $NAV, always $HARD, --save_dir exp/tdjtab):
      $RUN --critic hilp_density_tdjc --domain $d --seed $s --save_dir exp/tdjtab $HARD [$NAV]
    Hit spec passed for uniform config.json evidence, but NOTE (verified 2026-09-07): `hilp_density.py:186-189`
    parses `measure_hit_spec` only when `goal_kernel=true`, and `hilp_density_tdjc.yaml` has no `goal_kernel` key
    (default False) — so on this critic the spec is INERT, as is `$HARD` (hard-only critic, `hilp_density.py:117`).
    Route B still needs the relabel reward (§5 item 1).

### RLDP-density (`rldp_density`) — 24 runs
    for s in 0 1 2; do $RUN --critic rldp_density --domain cube_double --seed $s $HARD $CD_HIT $RELABEL; done   # NO --save_dir: resumes exp/cube-double-play-v0_rldp_density_flowbc_seed_$s from params_500000 → 1M
    scene_play ×3, puzzle_3x3 ×3, 5 antmaze ×3 (with $NAV, always $HARD, --save_dir exp/tdjtab):
      $RUN --critic rldp_density --domain $d --seed $s --save_dir exp/tdjtab $HARD [$NAV]
    Same as HILP-density: `rldp_density.py:231-237` reads the spec only under `goal_kernel=true` (absent from
    `rldp_density.yaml`, default False) — inert here, passed for evidence. Critic is hard-only (`rldp_density.py:138`);
    the cube-double resumes are therefore hard by construction and allowed.

Decoupling recommendation: for scene/puzzle/antmaze, train Route-A-only NOW (omit $RELABEL — passing it
crashes at the first eval on the `goal_xyzs` assert, env/ogbench.py:201), keep `--save_interval 250000`, and
score Route B afterwards with `scripts/relabel_rescore.py --env <id> --critics ... --epoch 1000000 --exp-root exp/tdjtab`
once `relabel_infer_batch` supports the env. Route B is final-step only by the spec's gate 2, so nothing is lost.

## 3. Tier 2 — the other sweep cells (per env, 3 seeds each)

td_jepa's reported number is best-of-sweep over 10 seeds (`~/git/baselines/td_jepa/scripts/train/proprio/launch_*_ogbench.py`).
Extra cells beyond the chosen one:

| method | extra cells | override | runs/env | runs/9 envs |
|---|---|---|---|---|
| FB | ortho 100 / lr_b 1e-4; ortho 1000 / lr_b 1e-5; ortho 100 / lr_b 1e-5 | `ortho_coef=100`, `lr_b=1.0e-5` | 9 | 81 |
| HILP | lr 1e-5 / p 0.375; lr 1e-4 / p 0.5; lr 1e-5 / p 0.5 | `group_lr.hilp_phi1=1.0e-5 group_lr.hilp_phi2=1.0e-5`, `hilp_p_randomgoal=0.5` | 9 | 81 |
| RLDP manip (cube×2, scene, puzzle) | ortho 0.1 / lr_b 1e-4; 0.01 / 1e-5; 0.1 / 1e-5 | `ortho_coef=0.1`, `lr_b=1.0e-5` | 9 | 36 |
| RLDP antmaze (×5) | ortho 0.001 / lr_b 1e-4; 0.01 / 1e-5; 0.001 / 1e-5 | `ortho_coef=0.001`, `lr_b=1.0e-5` | 9 | 45 |
Total tier 2 = 243 runs. Each cell must get its own `--save_dir` (e.g. `exp/tdjtab-o100-lrb5/`) — the fb_tdj
contamination came from two lr_b arms sharing the default `exp/`. The `lr_b=1e-5` cube-single FB cell already
exists but is entangled in the contaminated dirs (s0, s1 record lr_b 1e-5) — treat as lost.

## 4. Counts, wall time, datasets

Tier-1 runs needed (hard target only; recount 2026-09-07 — unchanged, because no local density checkpoint is soft):

| method | new 1M runs | 500k→1M resumes (hard ckpts) | total |
|---|---|---|---|
| FB (fb_tdj) | 27 (cube_single 3 + cube_double 3 + 7 non-cube × 3) | 0 | 27 |
| HILP (hilp_tdjc) | 21 | 0 | 21 |
| RLDP (rldp) | 21 | 3 | 24 |
| DFB goal (density_fb) | 21 (7 non-cube × 3) | 6 (cube ×2 envs ×3 seeds, all `measure_density_hard: true`) | 27 |
| DFB z (density_fb_z, latent_centered+hard) | 23 | 0 | 23 |
| HILP-dens (hilp_density_tdjc, hard-only) | 21 | 0 | 21 |
| RLDP-dens (rldp_density, hard-only) | 21 | 3 | 24 |
| **tier 1** | **155** | **12** | **167** |

(+24 optional `density_fb_state`, 6 of them hard resumes, not counted.) Tier 2 unchanged at 243 (§3).
**Count correction 2026-09-07:** the earlier total of 161 undercounted FB and density_fb by 3 each (both rows
forgot one of the 7 non-cube envs / cube_double); the §2 enumeration — and `scripts/tdjtab_manifest.sh`, which
is generated from it — has 167 lines: 27 + 21 + 24 + 27 + 23 + 21 + 24.
Delta vs the pre-decision list: 0 runs added, 0 removed — every resume candidate was already hard, and no
soft arm was ever in the list. Launchable today without new code: cube-single fb_tdj ×3, cube-double
fb_tdj/rldp/rldp_density/density_fb/density_fb_z(2) = 17 runs (11 of them density-family, all with `$HARD`/`$ZFIX`);
Route-A-only for FB/HILP/RLDP on scene/puzzle/antmaze = 63 runs (need only the 7 domain yamls);
the remaining 81 density-variant runs on scene/puzzle/antmaze need §5 items first.

Wall time, measured from `params_*.pkl` mtimes, NVIDIA L4 24 GB, one run per GPU (this box; the spec's "~10 GPU-h on
A100-class" is conservative):
- fb_tdj cube-single 1M: 3.6 h (config 03:13 → params_1000000 06:49, 2026-09-03)
- hilp_tdjc cube-single 1M: 3.4 h; cube-double 1M: 7.3 h (23:59 → 07:17, 09-05)
- rldp_density cube-single 1M: 3.5 h; rldp cube-double 500k: 6.5 h → ≈13 h/1M
- density_fb_state cube-single 500k: 1.8 h (≈3.6 h/1M); density_fb cube-double 500k: 3.9 h (≈8 h/1M);
  density_fb_z (batch 1024) cube-single 500k: 3.0 h (≈6 h/1M)
- scene/puzzle/antmaze: no measurement. Training cost ≈ cube-single (obs 40/55/29 dims), but eval is 3.75–5× longer
  per episode (scene 750, puzzle 500, antmaze 1000 steps vs cube 200) × (20 Route-A + 10 Route-B eps × 5 tasks × 10
  evals) — budget 5–8 h/run, ~10 h for antmaze-large.
Rough total tier 1 (hard only): 155 fresh × ~5.5 h ≈ 850 GPU-h + 12 resumes × ~4 h (half of ≈8 h/1M cube-double
for density_fb/rldp_density/rldp) ≈ 50 GPU-h ⇒ **≈ 900 GPU-h ≈ 4.7 days on 8 GPUs** (previous estimate ~900 GPU-h
treated every run as a full 1M).

Datasets (all present, no download needed): `~/.ogbench/data/` (OGBench's default dir) has all 9 table envs + val
splits (antmaze-{medium,large}-{navigate,stitch,explore}, cube-single/double/triple/quadruple, puzzle-3x3/4x4/4x5/4x6,
scene-play, visual-cube-single); `/dev/shm/ogbench_data/data/` mirrors the same 34 npz (volatile — wiped on reboot).
`Factored-FB/datasets/` (main worktree) has cube-single only. All npz carry `qpos` (antmaze 15-wide, scene 25, puzzle 23),
and scene/puzzle also `button_states` — loaded when `eval_relabel_size` is set (`add_info=True`).

## 5. Blockers

1. **Route-B relabel reward is cube-only.** `env/ogbench.py::relabel_infer_batch` asserts `goal_xyzs` in
   `task_infos` (line 201). Needed (transcribe from `~/git/baselines/td_jepa/metamotivo/envs/ogbench.py:30-58` and
   `get_relabel_fn` :115-127, thresholds included):
   - antmaze: `ant_reward_fn`: `‖qpos[:, :2] − goal_xy‖ ≤ 0.5` − 1; goal = `task_infos[t]['goal_xy']`
     (`ogbench/locomaze/maze.py:357`; td_jepa reads `env.get_oracle_rep()` after `reset(task_id)`, same value).
   - scene: `scene_reward_fn`: cube `qpos[14:17]` within 0.04 of goal xyz, buttons `qpos[-2]`, `qpos[-1]` equal,
     drawer `qpos[-4]`, window `qpos[-3]` within 0.04 of `task_infos[t]['goal']` (dict: cube xyz, 2 buttons, drawer, window),
     minus 5. The spec's Tier-2 note says "matched to OGBench's scene success rule + a test" — td_jepa's version is the
     protocol to match.
   - puzzle-3x3: `puzzle_reward_fn`: `(qpos[:, -9:] == goal_button_states).sum() − 9`; goal = `task_infos[t]['goal_button_states']`.
   - Shift 1.0 turns each into a 0/1 all-satisfied indicator for a single-object env only; for scene/puzzle the shifted
     reward is (satisfied − k + 1) ∈ {−4..1}/{−8..1} — that is td_jepa's protocol verbatim, keep it and say so in captions.
   - Add a test per reward next to the cube one (`tests/impls/`), then the 63 Route-A-only runs can be rescored.
2. ~~**Domain configs missing** for all 7 non-cube envs (§2 list).~~ **RESOLVED 2026-09-07** — the 7 yamls exist in
   `impls/configs/domain/` (see §2); each carries the derived hit spec in its comment header.
3. ~~**`measure_hit_spec` for density variants on non-cube envs**~~ **RESOLVED 2026-09-07** (derived from the
   installed ogbench source, `ogbench/` in this repo, and verified by instantiating each env). The kernel is a
   conjunction of observation-column blocks `a:b:eps` (`parse_hit_spec`, `impls/critics/_measure_common.py:26`:
   any number of blocks, per-block radius, blocks need not be contiguous or ordered; each block is an L2 ball on
   its slice). The spec is consumed by `density_fb` / `density_fb_z` (B input = `scaled_hit_cols`, and the eval
   goal kernel; `density_fb.py:1128`); it is INERT on `hilp_density_tdjc` and `rldp_density` (read only under
   `goal_kernel=true`, which neither yaml sets) and on `density_fb_state` / `density_fb_z_state` (B eats the
   full state; `u_raw` still uses it for the kernel reward, `density_fb_state.yaml:155-164`). Every yaml defaults to
   the cube block `19:22:0.4`, which is silently wrong elsewhere (obs width passes the assert on all three
   families), so the override is mandatory on scene/puzzle/antmaze.

   | env | obs width | override (quoted string) | derivation |
   |---|---|---|---|
   | antmaze-*-v0 (all 5) | 29 = qpos 15 + qvel 14 (`ant.py:97-101`) | `measure_hit_spec="0:2:0.5"` | ant xy = `qpos[:2]` = obs 0:2 in raw metres (`ant.py:116`); success = `‖xy − goal_xy‖ ≤ _goal_tol`, 0.5 for ant (`maze.py:86,487`) |
   | scene-play-v0 | 40 = 19 proprio + cube 9 (pos 19:22, quat 22:26, cos/sin 26:28) + 2 × button [one-hot 28:30 / 32:34, pos 30 / 34, vel 31 / 35] + drawer pos 36 vel 37 + window pos 38 vel 39 (`scene_env.py:630-681`) | `measure_hit_spec="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"` | success rule `scene_env.py:485-501`: cube `‖Δ‖ ≤ 0.04` m × `xyz_scaler` 10 → 0.4; button state equality → one-hot mismatch has norm √2, radius 0.5 = exact; drawer `|Δ| ≤ 0.04` × `drawer_scaler` 18 → 0.72; window `|Δ| ≤ 0.04` × `window_scaler` 15 → 0.6 |
   | puzzle-3x3-play-v0 | 55 = 19 proprio + 9 × [one-hot 2, pos, vel]; button i one-hot at `19+4i:21+4i` (`puzzle_env.py:653-684`) | `measure_hit_spec="19:21:0.5,23:25:0.5,27:29:0.5,31:33:0.5,35:37:0.5,39:41:0.5,43:45:0.5,47:49:0.5,51:53:0.5"` | goal = 9 binary button states (`compute_oracle_observation`); **design choice:** nine one-hot blocks each with radius 0.5 (a mismatch has norm √2), so the conjunction is exactly "all 9 buttons match" = the env's success rule. NOT one block over 19:55 — that would drag the continuous button pos/vel columns into the goal |

   Evidence (scratch script `verify_hitspec.py`, 2026-09-07): for each env, obs and `info['goal']` widths match
   (29/40/55); antmaze `obs[0:2] == get_xy()` and moves under `set_xy`; scene `obs[19:22] == (block_0_pos − c)·10`,
   `obs[28:30]`/`obs[32:34]` equal `eye(2)[button_state]`, `obs[36] == drawer_pos·18`, `obs[38] == window_pos·15`;
   puzzle all 9 one-hot slices equal `eye(2)[_cur_button_states[i]]`. Over all 5 tasks per env the spec is
   satisfied goal-vs-goal (5/5) and NOT reset-vs-goal (0/5) — scene per-block shows the expected partial matches
   (e.g. task1_open: cube+buttons already satisfied, drawer/window not). On the full 1M-row datasets every one-hot
   block sums to 1 in 100% of rows; scene drawer col ranges [−2.95, 0.07], window [−0.04, 3.04] (obs units).
   Calibration note: antmaze eps is in raw metres, unscaled (no `xyz_scaler` on locomaze obs).
   Quoting: single-block specs (antmaze) MUST be passed as `--override 'measure_hit_spec="0:2:0.5"'` — unquoted,
   OmegaConf yields 120.5 and the parser refuses (`_measure_common.py:44`); the manifest quotes all of them.
   `hilp_tdjc`/`rldp`/`fb_tdj` are unaffected (spec inert unless `backward_input=hit_spec`).
4. **`density_fb_z` cube 1M fixed-variant checkpoints live only on the cluster**; local `exp/partb-tier1/` is pre-fix.
   Pull them (`scripts/hf_upload_checkpoints.py` / `amsks/factored-fb-checkpoints`) before quoting; the cube-double
   seeds 1,2 in §2 assume the cluster's seed 0 is the fixed cell — verify its config.json has
   `measure_density_z_actor_reward: latent_centered` AND `measure_density_hard: true` first; a cluster ckpt
   recording `measure_density_hard: false` is out of scope and its seed is re-run fresh into `exp/tdjtab/`.
5. **fb_tdj cube-single contamination**: the three dirs mix lr_b 1e-4 and 1e-5 rows; `exp/cube-single-play-v0_fb_tdj_*`
   must not be resumed or rescored — re-run into `exp/tdjtab/`.
6. Eval-time asserts: `eval_relabel_size` set on a non-cube env crashes at step 100k (blocker 1). Launch those
   without `$RELABEL` until 1 lands.
7. Minor: `relabel_rescore.py` takes `--env` and `--exp-root` (verified), so the post-hoc Route-B path works for
   `exp/tdjtab/` once blocker 1 is fixed; `rldp_tdj.yaml`/`rldp_density_tdj.yaml` are DMC-shaped — do not use.

## 6. Smoke matrix (2026-09-07, 300 steps, eval+save at 300, `--eval_episodes 1`, 1 seed, tmux `tdjtab-smoke`)

Each cell: the manifest's flags for that (env, critic) minus `--train_steps/--eval_interval/--save_interval`, plus
`--eval_episodes 1` (the absl flag that cuts eval to 1 episode per task; 5 tasks each). Every cell exits 0, writes
`params_300.pkl`, and logs two eval rounds (step 1 and step 300) over all 5 tasks. Wall time is dominated by
eval (antmaze 1000-step, scene 750-step, puzzle 500-step episodes), not training.

| env | fb_tdj | hilp_tdjc | rldp | density_fb (hard + per-env hit spec) |
|---|---|---|---|---|
| antmaze-medium-navigate (bc_coeff 0.3) | PASS 429 s | PASS 286 s | PASS 441 s | PASS 281 s |
| scene-play | PASS 346 s | PASS 246 s | PASS 357 s | PASS 426 s |
| puzzle-3x3-play | PASS 249 s | PASS 179 s | PASS 253 s | PASS 435 s |

One non-result: the first antmaze/density_fb attempt died at 6 s with the sexagesimal assert because the smoke
script's `eval` stripped the inner quotes (`measure_hit_spec=0:2:0.5` → 120.5, `_measure_common.py:44`); rerun
with the manifest's quoting (`--override 'measure_hit_spec="0:2:0.5"'`) passes, and that quoting was checked to
reach python as the string `0:2:0.5`. `config.json` of each density smoke records the per-env spec and
`measure_density_hard: true`; antmaze smokes record `actor.bc_coeff: 0.3`.
