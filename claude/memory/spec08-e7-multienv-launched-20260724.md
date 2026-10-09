---
name: spec08-e7-multienv-launched-20260724
description: 2026-07-24 spec08 E7 LAUNCHED — two-phase curriculum ported to cube-double + scene via a new measure_hit_spec knob; 10 jobs in tmux s08e7; conjunctive regions are 27x/4.5x sparser so each cell ships a sparsity control
metadata: 
  node_type: memory
  type: project
  originSessionId: 95a83213-2dc4-47dc-8385-75884e058973
  modified: 2026-07-24T23:46:25.853Z
---

2026-07-24 (bridge track, follow-on to [[spec08-launched-20260724]]). User asked which
experiments produced `PAPER/rescuing-fb/two-phase-successor-measure-note.tex` and to start the
two-phase curriculum on **cube-double** and **scene**. That is spec 08 **E7**, previously
deferred on one blocker.

**Provenance of the note's numbers** (answer to the first question): phase-1 0.967 =
`cube_single_state_measure_firstpassage_noortho_d50.yaml` / group
`GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50` in `exp_spec06`; phase-2 0.952/0.928 =
`launch_phase2_occupancy.sh` `vis_s0/s1` / `...-BallVisit-FrozenPsi-d50`; density-TD divergence
= the `occfz` arm; (z,x) 0.904/0.960 = `cube_single_state_measure_fp_zx_d50.yaml` /
`...-FP-ZX-d50`; zero-shot grid = `eval_zeroshot_rewards.py` on those ckpts.

**The blocker is gone.** ε=0.4 / obs[19:22] was hardcoded in `gciql_flow.py`. Now
`config.measure_hit_spec` — a STRING of comma-separated `start:stop:eps` obs blocks ANDed
together (`_parse_hit_spec` + `GCIQLFlowAgent._hit_matrix`). Default `'19:22:0.4'` is
**bit-identical**, so all cube-single results/checkpoints stand. Only `measure_loss` was
generalized; `task_prior`/`task_prior_actor` still hardcode cube-single and the assert-lattice
now refuses to compose them with a non-default spec. Tests
`tests/test_gciql_measure_hit_spec.py` 6/6 + the 23 existing measure/phase-2 tests green.

**Regions mirror each env's own success criterion** (verified in ogbench source):
cube-double `'19:22:0.4,28:31:0.4'` (success = all cubes, `cube_env.py:728`);
scene `'19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'` (cube ∧ 2 button one-hots ∧
drawer ∧ window, `scene_env.py:552`; one-hot equality ⇔ L2≤0.5; drawer/window = 0.04 m at
scalers 18/15). Obs dims: cube-double 37, scene 40; action dim 5 for all three.

**⚠ THE KEY RISK — measured, not guessed** (real datasets, B=1024 = the training batch):
off-diagonal positives per row are cube-single **14.0**, cube-double 2-cube **0.51** (27×
sparser), scene 5-way **3.07** (4.5× sparser). So every conjunctive cell ships a **sparsity
control** with a cube-single-density region (cube-double 1-cube 18.6; scene cube-only 61.9).
Without it, a floor is confounded between "reach-first doesn't transfer" and "the reward is
27× sparser". The controls' regions ignore factors the native eval scores — read their
TRAINING traces, not their overall success.

**⚠ ANCHOR CALIBRATION:** the note's 0.90 bar is cube-single-only. Scene GC-IQL bilinear-d50 =
**0.292** (flow), stock-MLP 0.516, FB 0.047. No cube-double anchor existed, so this wave
trains one.

**LAUNCHED 23:31, tmux `s08e7`, `scripts/rescuing_fb/launch_spec08_e7_phase1.sh`,
out `/home/adityamohan/exp_spec08`, logs `/dev/shm/spec08_logs/`,** 10 jobs (2 seeds each,
500k, ~9 h), one per GPU on the idle 8×L4 box: `cd_fp_s{0,1}`, `cd_fp1c_s{0,1}`,
`sc_fp_s{0,1}`, `sc_fpc_s{0,1}`, `cd_anch_s{0,1}`. All 10 verified running with the correct
hit_spec on the child cmdline; both new configs smoke-tested end-to-end first.

**PHASE 2 ready, NOT launched** (needs wave-1 ckpts): `cube_double_state_measure_ballvisit_d50.yaml`,
`scene_state_measure_ballvisit_d50.yaml` — exact cube-single recipe (`measure_fp_absorbing=false`,
`measure_stopgrad_psi=true`, `actor_q_center=true`) + `warm_start_ckpt`/`warm_start_epoch=500000`.

**NOT ported — the zero-shot reward-family harness.** `eval_zeroshot_rewards.py`,
`t6_gpi_eval.py`, `t6_generality_probe.py`, `renewal_head.py`, `measure_readout_head.py` each
redeclare `CUBE_OBS=slice(19,22)` / `CUBE_QPOS=slice(14,17)` / `EPS_OBS=0.4` / cube-single
`EVAL_TASKS` / cube-single table `REGIONS`. Two are **silent wrong-answer** bugs, not crashes:
`cur_task_info["goal_xyzs"].reshape(-1)[:3]` truncates cube-double to cube 0, and scene has no
`goal_xyzs` key. The **native GC eval used by wave 1 is fine** (env-parameterized via
`GCIQL_FB_ENV` in `sitecustomize.py`). Fix when needed: one `env_geometry.py` registry.

6 new run_groups added to `scripts/results_map.py`; CSV refresh due after collection
([[results-csv-pipeline]]). Code **UNCOMMITTED** (branch shared with the other track).

**How to apply:** this is the experiment that decides whether the note claims a *phenomenon*
or a *cube-single result*. Read conjunction-vs-control per env FIRST — that attribution is the
whole point of the wave.
