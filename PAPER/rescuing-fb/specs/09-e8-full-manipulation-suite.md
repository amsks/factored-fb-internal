# Spec 09 — E8: the two-phase curriculum on the full OGBench manipulation suite

Follow-on to spec 08 **E7** ("Scope: second environment"). E7 asked whether the reach-first
curriculum transfers off cube-single and answered *yes on scene*: the env-faithful 5-way
region reached **0.468/0.480** at 500k against a GC-IQL bilinear-d50 anchor of **0.292**
(FB 0.047). E8 settles the scope question properly — **all nine** OGBench state-based
manipulation environments, **3 seeds**, **500k** steps, phase 1 + phase 2 + a matched anchor
in every cell.

The claim under test is the note's lead caveat: *"one environment so far (`cube-single`,
state observations)."* If phase 1 tracks its anchor across the suite, the note claims a
phenomenon; if it tracks the anchor only where the anchor is high, the note claims a
cube-single result with a scene corroboration.

## What is held fixed

Everything except the goal region. The recipe is the cube-single winner
(`cube_single_state_measure_firstpassage_noortho_d50`, 0.967): bilinear critic
φ(s,a)ᵀψ(g)/√d with **d = 50**, twin heads with min-backup, γ = 0.99, **no orthonormality**
(`measure_ortho_coef=0`), `measure_scale_norm`, `measure_only` on the critic legs,
`bc_coeff=3.0`, flow-BC + one-step-distillation actor, batch 1024, 500k steps.

`value_latent_dim` stays at 50 on every env deliberately: it makes the environment the only
moving part. The known risk is that rank 50 genuinely under-fits the largest goal spaces
(puzzle-4x6 has 2²⁴ button configurations), in which case a floor there is ambiguous between
"the method fails" and "the rank is too small". The anchor shares the same d = 50 and the
same actor, so it absorbs exactly that ambiguity — see *Why every env gets an anchor* below.

## The one per-env hyperparameter: `measure_hit_spec`

Each region mirrors that environment's own success criterion, read from the ogbench source
and **verified against privileged state** by `scripts/rescuing_fb/verify_hit_specs.py`
(which recomputes the observation transform from `compute_ob_info()` and asserts each slice
carries the object it is assigned, that the conjunction width equals the number of objects
the env conjoins in `_compute_successes`, and that the radius reproduces the env threshold).

Observation layout is `[19 proprio] + per-object blocks`; a cube block is 9 dims
(pos 3, quat 4, cos/sin yaw 2), a button block is 4 dims (one-hot 2, pos 1, vel 1).

| env | obs | blocks | region | radius rationale |
|---|---|---|---|---|
| cube-single | 28 | 1 | `19:22:0.4` | 0.04 m env threshold × xyz_scaler 10 |
| cube-double | 37 | 2 | `19:22:0.4,28:31:0.4` | success = **all** cubes |
| cube-triple | 46 | 3 | `+37:40:0.4` | " |
| cube-quadruple | 55 | 4 | `+46:49:0.4` | " |
| scene | 40 | 5 | `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6` | cube ∧ 2 buttons ∧ drawer ∧ window |
| puzzle-3x3 | 55 | 9 | `19+4i : 21+4i : 0.5`, i<9 | one-hot equality ⇔ L2 ≤ 0.5 (‖eᵢ−eⱼ‖ ∈ {0, √2}) |
| puzzle-4x4 | 83 | 16 | " i<16 | " |
| puzzle-4x5 | 99 | 20 | " i<20 | " |
| puzzle-4x6 | 115 | 24 | " i<24 | " |

## The pre-registered risk: region density

`scripts/rescuing_fb/measure_region_density.py` measures off-diagonal positives per row at
the training batch size B = 1024 (the diagonal fires by construction, so the off-diagonal is
the only signal teaching many-to-one region structure). It reproduces E7's wave-1 numbers
(cube-single 13.90 vs 14.0, cube-double 0.46 vs 0.51, scene 2.85 vs 3.07):

| env | +/row | vs cube-single |
|---|---|---|
| cube-single | 13.90 | 1× |
| scene | 2.85 | 4.9× sparser |
| puzzle-3x3 | 2.04 | 6.8× |
| cube-double | 0.46 | 30× |
| puzzle-4x4 | 0.06 | 248× |
| cube-triple | 0.03 | 487× |
| cube-quadruple | 0.01 | 1452× |
| puzzle-4x5 | 0.01 | 962× |
| puzzle-4x6 | 0.01 | 1547× |

**The bottom five have essentially no off-diagonal signal** — roughly 10 positives per
1M-entry batch, so only the 1024 grounded diagonal entries anchor each step. E7's lesson is
that sparsity alone does not predict failure (scene's *sparser* conjunction beat its denser
cube-only control 0.47 vs 0.23, 2×), but that lesson was learned within one order of
magnitude; these envs sit three orders down. E8 does **not** ship sparsity controls, on E7's
evidence; the anchors do that job instead, and more cheaply.

## Why every env gets an anchor

Three reasons, in order of importance:

1. **The bar is env-specific.** The note's 0.90 came from cube-single. Scene's is 0.292.
   Comparing a new env's phase-1 number to 0.90 would be meaningless.
2. **It makes a floor interpretable.** The anchor trains on the same data, with the same
   actor class, the same d = 50, and the same goal-reaching event. If both anchor and
   measure cell floor, the environment is hard for goal-conditioned TD generally — not for
   the measure objective specifically. This is the control that replaces E7's sparsity arms.
3. **It absorbs the rank question.** A d = 50 anchor that reaches high success on
   puzzle-4x6 would prove rank 50 suffices there, isolating any measure-side floor to the
   objective. A d = 50 anchor that also floors leaves the rank question open — and says so
   honestly rather than silently.

The cube-single anchor is re-run at 500k (its pinned config is a 1M-step campaign cell) so
every anchor in the table is measured at the same budget.

## Cells

Per env, 3 seeds each:

- **phase 1** — `<env>_state_measure_firstpassage_noortho_d50`: absorbing first-passage TD,
  fixed point E[γ^(T−1)] ∈ [0,1].
- **phase 2** — `<env>_state_measure_ballvisit_d50`: non-absorbing backup (revisits count),
  `measure_stopgrad_psi` (ψ frozen from phase 1), `actor_q_center`, warm-started from the
  matching phase-1 checkpoint at 500k.
- **anchor** — `<env>_state_gciql_flow_bilinear_d50`.

Reused rather than relaunched: cube-single phase 1 (3 seeds, 0.967, `exp_spec06`),
cube-single phase 2 seeds 0/1 (0.952/0.928, `exp_spec07`), scene phase 1 seeds 0/1
(0.468/0.480), cube-double phase 1 and anchor seeds 0/1 (in flight, tmux `s08e7`).

Wave A (phase 1 + anchors) is 45 jobs; wave B (phase 2) is 25. Wave B is gated on wave-A
checkpoints and refuses to launch a cell whose `params_500000.pkl` is absent.

## Reads

- **Primary.** Per env, phase-1 final vs its anchor. Reach-first tracking the anchor across
  ≥ 2 envs beyond cube-single ⇒ phenomenon, not artifact.
- **Secondary.** Phase 2 holds the phase-1 level, and the critic scale plateaus near the
  region's Monte-Carlo visitation mass instead of inflating. On cube-single this was 24 vs a
  true 24.7; sparse regions have less revisit mass, so expect *smaller* plateaus, not larger.
- **Sparsity.** Plot phase-1-minus-anchor against region density. If the gap is flat in
  density, sparsity is not the mechanism and E7's finding extends three orders of magnitude.
  If it opens up below ~0.1 positives/row, we have located the breaking point of the
  hindsight-matrix estimator — itself a publishable boundary.
- **Negative cells are informative** and are kept: "where does reach-first break?" is the
  question E7 was written to ask.

## Not in scope

The zero-shot reward-family harness is still cube-single-only
(`eval_zeroshot_rewards.py`, `t6_gpi_eval.py`, `t6_generality_probe.py`, `renewal_head.py`,
`measure_readout_head.py` each redeclare `CUBE_OBS=slice(19,22)` / `EPS_OBS=0.4` /
cube-single `EVAL_TASKS`). Two are *silent wrong-answer* bugs on multi-object envs, not
crashes. The **native GC eval used by this campaign is unaffected** — it is env-parameterized
through `GCIQL_FB_ENV` in `sitecustomize.py`. Porting the grid needs one
`scripts/rescuing_fb/env_geometry.py` registry keyed by `env_name`. Pixels remain deferred.

## Artifacts

| what | where |
|---|---|
| region verification | `scripts/rescuing_fb/verify_hit_specs.py` |
| density pre-flight | `scripts/rescuing_fb/measure_region_density.py` |
| config generator | `scripts/rescuing_fb/gen_spec08_e8_configs.py` |
| smoke test | `scripts/rescuing_fb/smoke_spec08_e8.sh` |
| wave A launcher | `scripts/rescuing_fb/launch_spec08_e8_waveA.sh` |
| wave B launcher | `scripts/rescuing_fb/launch_spec08_e8_waveB.sh` |
| outputs | `/home/adityamohan/exp_spec08`, logs `/dev/shm/spec08_logs/e8a`, `e8b` |

After collection: add every `run_group` to `scripts/results_map.py`, refresh with
`.venv/bin/python scripts/compile_results_csvs.py --refresh`, and add a `DECISIONS.md`
entry — the standing rule for this repo.
