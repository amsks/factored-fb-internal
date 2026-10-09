---
name: spec08-e8-fourenv-launched-20260725
description: "2026-07-25 spec08 E8 LAUNCHED — two-phase curriculum on the 4 envs with published vanilla-FB numbers (cube-single/double, scene, puzzle-3x3); 15 wave-A jobs tmux s08e8a; TD-JEPA FB HPs recovered from the official codebase; eval-protocol mismatch is the open comparability risk"
metadata: 
  node_type: memory
  type: project
  originSessionId: 80d7f590-a9cd-4185-87c0-59b0cdd44447
  modified: 2026-07-25T06:17:28.333Z
---

2026-07-25 (bridge track, follow-on to [[spec08-e7-multienv-launched-20260724]]). Scope was
first planned as all 9 OGBench state manipulation envs, then **narrowed by the user to the
four envs that have a published vanilla-FB number**, state-based only.

**E7 wave-1 result that shaped this:** scene env-faithful 5-way region **0.468/0.480** at
500k vs its GC-IQL bilinear-d50 anchor **0.292** — reach-first BEATS the anchor on scene.
Its denser cube-only sparsity control scored only 0.224/0.232, i.e. the *sparser*
env-faithful region won by 2x. So sparsity controls were dropped for E8; the matched anchor
does that job instead.

**TD-JEPA FB hyperparameters — recovered from the OFFICIAL CODEBASE, not the paper prose**
(`/home/adityamohan/git/baselines/td_jepa/scripts/train/proprio/launch_fb_ogbench.py`,
function `sweep_cube()` — which is *exactly* these four envs):
1M steps, **batch 256**, z_dim=L_dim=**50**, discount 0.99, **bc_coeff 3.0**,
ortho_coef swept {100,1000}, lr_b swept {1e-4,1e-5}, lr_f/lr_actor 1e-4,
f/b_target_tau 0.005, F 512x2, B 512x4 (norm), left_encoder 512x4 (norm), norm_z=True,
actor = noise_conditioned 512x2 + actor_vf 512x4, train_goal_ratio 0.5, 10 seeds.
d=50, gamma=0.99 and bc_coeff=3.0 MATCH our recipe; steps (1M vs our 500k) and batch
(256 vs our 1024) do not.

**⚠ THE COMPARABILITY TRAP — do not put FB in the same table without fixing this.**
TD-JEPA's OGBench numbers come from `OGBenchRewardEvaluation`
(`metamotivo/evaluations/ogbench.py`): **zero-shot reward eval**, z inferred from 10,000
relabelled samples, **10 episodes/task**, success = *any* step succeeded. Our 0.967 / 0.468
are **native goal-conditioned** eval, 50 episodes/task, goal handed to the policy. These are
different protocols and the GC one is easier. Our comparable numbers are the note's
zero-shot rows (retrieval-1 dirac 0.89/0.96/0.97; direct-z 0.90/0.92) — and that harness is
still cube-single-only (needs the `env_geometry.py` registry; `eval_zeroshot_rewards.py`
et al. hardcode `CUBE_OBS=slice(19,22)`/`EPS_OBS=0.4`, and truncate multi-cube envs SILENTLY).

**Reference numbers.** FB (TD-JEPA, state): cube-single 49.6±3.8, cube-double 2.6±0.4,
scene 12.8±1.6, puzzle-3x3 4.8±0.7. Stock GCIQL (OGBench paper Table 2, 1M, 8 seeds):
68±6 / 40±5 / 51±4 / **95±1 (puzzle-3x3 — UNVERIFIED, automated table read, check the PDF)**.
Consistency check that passed: OGBench GCIQL scene 51 vs our stock-MLP measurement 0.516.

**Region density** (`scripts/rescuing_fb/measure_region_density.py`, off-diag positives/row
at B=1024; reproduces wave-1 exactly): cube-single 13.90, scene 2.85, puzzle-3x3 2.04,
cube-double 0.46. These four are the DENSEST in the suite — the sparsity risk that dominated
the 9-env plan (cube-triple 0.03, quadruple/4x5/4x6 0.01) is largely absent here.

**LAUNCHED 05:38, tmux `s08e8a`, `launch_spec08_e8_waveA.sh`, out `/home/adityamohan/exp_spec08`,
logs `/dev/shm/spec08_logs/e8a/`.** 15 jobs (phase-1 top-ups p1_cd_s2/p1_sc_s2 + puzzle-3x3
p1_p3_s0/1/2; anchors an_cs_s0/1/2 @500k, an_cd_s2, an_sc_s0/1/2, an_p3_s0/1/2), MAX=14.
Wave B (phase 2, 10 cells) = `launch_spec08_e8_waveB.sh`, GATED on phase-1 ckpts, safe to
re-run repeatedly. All new configs smoke-tested 300/300; scene phase-2 warm-start splice
verified (`[warm_start] spliced ... fresh-init []`).

**Verified, not assumed:** `scripts/rescuing_fb/verify_hit_specs.py` checks every region
slice against the env's privileged state + conjunction width vs `_compute_successes`.
Configs for the other 5 envs are written + smoke-tested and left in place; re-add their job
lines to extend.

**STATUS at handoff (2026-07-25 ~06:00):** 9/15 wave-A jobs dispatched, 15 training procs
live (incl. the 4 E7 cube-double jobs: cd_fp_s0/s1 ~100k, cd_anch_s0/s1 ~50k, all slow at
~18h under contention). Remaining 6 (an_sc_s0/1/2, an_p3_s0/1/2) queue behind MAX=14 and
release automatically as E7 finishes — the dispatcher stays alive in tmux `s08e8a`, do not
kill it. Eval dominates wall-clock on long-horizon envs (scene 750, puzzle-3x3 500,
cube-double 500 steps/episode; one eval pass 9-19 min x 11 evals/run).

**NEXT ACTIONS, in order**
1. When phase-1 cells land: run `launch_spec08_e8_waveB.sh` (idempotent, re-run freely).
2. Verify OGBench puzzle-3x3 stock-GCIQL 95±1 against the PDF — automated read, implausible.
3. Decide the eval-protocol question below; if the FB column is wanted, build
   `scripts/rescuing_fb/env_geometry.py` and port the zero-shot harness off cube-single.
4. After collection: refresh CSVs ([[results-csv-pipeline]]) + `DECISIONS.md` entry.

**⚠ EVERYTHING IS UNCOMMITTED** (~115 dirty/untracked paths; branch `gciql-fb-bridge` is
shared with the other track). New this session: `scripts/rescuing_fb/{verify_hit_specs.py,
measure_region_density.py,gen_spec08_e8_configs.py,smoke_spec08_e8.sh,
launch_spec08_e8_wave{A,B}.sh}`, 20 configs in `configs/gciql/`, 18 run_groups in
`scripts/results_map.py`, `PAPER/rescuing-fb/specs/09-e8-full-manipulation-suite.md`.

**Disk:** freed 457 G by deleting `git/offline-rl-aditya/AOSSControllerPythonAgent/train/`
(10,799 stale Apr-2026 workload .jsonl, untracked, unreferenced). `/` now 55% used, 471 G
free. Note `~/.ogbench` is a SYMLINK to `/dev/shm/ogbench_data/data` — all 9 datasets are
downloaded but live in tmpfs and die on reboot.

**How to apply:** before building any FB-vs-ours table, settle the eval-protocol question —
native GC and zero-shot reward eval are NOT interchangeable, and that is the single most
likely way this campaign produces a misleading headline.
