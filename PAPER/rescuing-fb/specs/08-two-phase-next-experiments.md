# Spec 08 — Next experiments for the two-phase successor-measure curriculum

Companion to `PAPER/rescuing-fb/two-phase-successor-measure-note.tex` (esp. the new
§4 *From density-ratio TD to finite-region regression*, §7 *What object is actually
learned*, §8 *Toward dense rewards*). Each experiment closes a gap the note itself
flags. Ordered by leverage: **E1** is the headline; **E2–E4** are the honest open
items; **E5–E6** validate load-bearing assumptions; **E7** is scope.

Status legend per experiment: **[config-only]** = existing knobs, **[+knob]** = one
small addition to `tools/wandb_mode_shim/gciql_flow.py`, **[+eval]** = addition to an
eval/analysis script.

---

## Shared conventions (verified against the repo)

- **Agent knobs** live in `tools/wandb_mode_shim/gciql_flow.py`. Relevant defaults:
  `measure_first_passage=False`, `measure_ortho_coef` (0.0 in the winning configs),
  `measure_fp_absorbing=True` (line ~1393), `measure_stopgrad_psi=False` (freeze ψ),
  `measure_z_random_frac=0.0` (line ~1246), `fb_forward_z=False`, `actor_cond`
  ∈ {`goal`,`z`,infer}, `actor_z_from_critic=False`, `actor_q_center=False`.
  Warm-start: `warm_start_ckpt`+`warm_start_epoch` splices *every* shared module and
  leaves the optimizer fully trainable (phase-1→2 path); `freeze_critic_ckpt`+
  `freeze_critic_epoch` splices value/critic and masks the optimizer to actor-only (the
  template E2 mirrors).
- **Kernel radius ε = 0.4 obs-units (4 cm) is HARDCODED** in three places in
  `gciql_flow.py` (`_measure_terms`, `_sample_tasks`, `_tpactor_critic_mix_loss`), not a
  config knob (cube xyz = `obs[19:22]`, 1 unit = 10 cm). E7 (other envs) must edit those
  literals or promote ε to a knob first.
- **Training launch** pattern (from `scripts/rescuing_fb/launch_phase2_occupancy.sh`):
  ```
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py --config-name <cfg> seed=<s> \
      output_root=<OUT> exp_name=<tag> \
      agent_overrides.K=V +agent_overrides.NEWKEY=V \
      run_group=<GROUP>
  ```
  `$PY` = the launcher's python (repo `.venv` for `run_gciql.py`, which works — see the
  occfz runs). CPU analysis/renewal scripts use `/dev/shm/.venv-jax/bin/python`
  (the e3 gotcha; repo `.venv` renewal driver died). **wandb entity = `amsks`**, never
  the team default. Always in **tmux**. `+agent_overrides.` prefix for keys not already
  in the config; bare `agent_overrides.` to override existing ones.
- **Warm-start** (phase 2 / curriculum): `+agent_overrides.warm_start_ckpt=<run_dir>`
  `+agent_overrides.warm_start_epoch=500000`.
- **Zero-shot eval**: `scripts/rescuing_fb/eval_zeroshot_rewards.py`
  ```
  $PY eval_zeroshot_rewards.py --run-dir <RUNDIR> --checkpoint-step 500000 \
    --families dirac,mixture,region,dense,linear,reachavoid \
    --tasks 1,2,3,4,5 --episodes 25 --horizon 200 [--gpi] \
    [--z-inference {moment,retrieval}] [--retrieval-k K] \
    [--z-weighting {plain,renewal}] [--renewal-head <npz>]
  ```
  Native GC eval + GPI gate: `scripts/rescuing_fb/t6_gpi_eval.py --run-dir <RUNDIR>
  --actor-cond z --stage full`.
- **Existing checkpoints to build on** (per memory / prior campaigns):
  - Phase-1 FP-NoOrtho 0.967: `exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s{0,1,2}`
  - Phase-2 ball-visitation (diagonal) 0.952/0.928: group `...-BallVisit-FrozenPsi-d50`
  - Separated (z,x) phase-1 0.904/0.960: group `GCIQL-State-Flow-Measure-FP-ZX-d50`
    (`fb_forward_z=true`, `measure_z_random_frac=0.5`, `actor_cond=z`,
    `actor_z_from_critic=true`, cold start).
- **Compute**: 500k steps ≈ ~9 h/seed on this box; GPUs 0/2/3/4/7 are this track's
  (1/5/6 = other session, never touch). 2–3 seeds/cell.

---

## E1 — Phase-2 ball-visitation on the *separated* (z,x) form  **[config-only]**  ★ headline

**Question.** Phase 2 (visitation) has only ever run on the *diagonal* goal-tied form
(0.952/0.928). Does the non-absorbing visitation TD stay stable on the **separated**
form, and — the test that actually matters (§8) — does **direct-z extraction from
non-goal-reducible rewards** improve once genuine revisit mass is present?

**Cell.** Warm-start from a ZX phase-1 checkpoint; identical phase-2 recipe as the
diagonal ball-visit, but on the z-conditioned forward map.
- `fb_forward_z=true`, `actor_cond=z`, `actor_z_from_critic=true`,
  `measure_z_random_frac=0.5`  (carry the ZX training form)
- `measure_first_passage=true`, `measure_ortho_coef=0.0`
- `measure_fp_absorbing=false`  ← the phase-1→2 switch (revisits counted)
- `measure_stopgrad_psi=true`   ← freeze B (=ψ)
- `actor_q_center=true`         ← dilution defense (M carries the shared R baseline)

**Launch** (per ZX seed; `ZXRUN` = the FP-ZX-d50 run dir for that seed):
```
$PY run_gciql.py --config-name cube_single_state_measure_firstpassage_noortho_d50 seed=0 \
  output_root=/home/adityamohan/exp_spec08 exp_name=visz_s0 \
  agent_overrides.measure_fp_absorbing=false \
  +agent_overrides.fb_forward_z=true +agent_overrides.actor_cond=z \
  +agent_overrides.actor_z_from_critic=true +agent_overrides.measure_z_random_frac=0.5 \
  +agent_overrides.measure_stopgrad_psi=true +agent_overrides.actor_q_center=true \
  +agent_overrides.warm_start_ckpt=$ZXRUN +agent_overrides.warm_start_epoch=500000 \
  run_group=GCIQL-State-Flow-Measure-BallVisit-ZX-d50
```
(2 seeds, matching the ZX seeds.)

**Eval.** Direct-z, no goal decode, on the *non-goal-reducible* families:
```
$PY eval_zeroshot_rewards.py --run-dir $VISZRUN --families region,linear,mixture --gpi \
  --z-inference moment --tasks 1,2,3,4,5 --episodes 25 --horizon 200
```
Also `--z-weighting renewal --renewal-head <npz>` (occupancy-correct integrand).

**Pre-registered read.**
- Training stability: critic scale plateaus ≈ the MC visitation mass (~24), no
  inflation, both seeds to 500k. (Diagonal form did this; ZX may reintroduce the
  policy–measure loop — see E2.)
- Extraction: **region ≥ 0.80** (clears the retrieval ceiling of 0.73–0.80) and/or
  **linear beats the random-z control with an edge** = the first evidence the *measure*
  (not the decode) is doing work. If region/linear are unchanged vs. phase-1 ZX, phase 2
  adds no arbitrary-reward capability and the note's claim stays goal-reducible.

**Depends on.** ZX phase-1 checkpoints (already trained). No new code.

---

## E2 — The frozen-B / frozen-actor visitation ablation  **[+knob]**

**Question.** Is phase-2 fragility caused by the **policy–measure feedback loop**
(§note "circularity"; the tex §4/§7)? Complete the 2×3 that B recommends. Two cells
already exist; one is missing.

| Backward B | Actor | Status | Object |
|---|---|---|---|
| frozen | frozen | **MISSING (this E2)** | pure fixed-policy visitation eval |
| frozen | updated | done — ball-visit 0.952/0.928 | |
| updated | updated | done — occtr, collapses | |

The frozen/frozen cell is an ordinary γ-contraction (fixed π): if it succeeds and
frozen/updated is only marginally better, the loop is inert here; if frozen/frozen is
clean but frozen/updated wobbles, the loop is the fragility source.

**Code needed.** No freeze-actor knob exists (`grep freeze_actor|stopgrad_actor` →
empty). Add `freeze_actor: bool=False` to the agent, mirroring the existing
`freeze_critic_ckpt` mechanism (`create` lines ~2658-2681: `optax.multi_transform`
+ `set_to_zero` masks the optimizer to a submodule). Here mask the **actor** module to
`set_to_zero` so π stays at the warm-started phase-1 policy while φ trains. Default-off
must be bit-identical; add a test alongside `tests/test_gciql_phase2_occupancy.py`.

**Cell.** The done ball-visit recipe (diagonal is fine here) `+ freeze_actor=true`,
warm-started from the matching FP-NoOrtho seed. 2 seeds.

**Eval.** Native GC (`t6_gpi_eval.py`) + critic-scale trace; compare final and curve
shape against the frozen/updated ball-visit numbers.

**Read.** frozen/frozen ≥ 0.90 and ≤ frozen/updated ⇒ loop inert, phase-2 is safe
fixed-policy evaluation (supports removing the caveat in tex §caveats). frozen/frozen
≫ frozen/updated ⇒ actor drift during phase 2 is real; report it.

**Depends on.** FP-NoOrtho checkpoints. Small knob + test.

---

## E3 — Dense-inference forensic: ridge vs. moment readout  **[+eval]**

**Question.** Resolve the ZX dense seed-split (0.90 vs 0.13). The note (§8, §sep)
attributes it to **covariance anisotropy**: the moment readout equals least-squares only
under isotropic features, and a dense reward's `z_r` averages hundreds of directions.

**Code needed.** `eval_zeroshot_rewards.py` has `--z-inference {moment,retrieval}` but
no ridge. Add `--z-inference ridge`: `z_r = (Σ ψψ^T/m + λI)^{-1} (Σ r ψ/m)` with
`--ridge-lambda` (sweep λ ∈ {1e-3,1e-2,1e-1}·tr(Σ)/d). This is the whitened LS solve
already written informally in the tex §"Task inference".

**Cell.** No training. Run on both ZX phase-1 seeds (and, once E1 lands, the ZX phase-2
visitation seeds).
```
$PY eval_zeroshot_rewards.py --run-dir $ZXRUN --families dense \
  --z-inference ridge --ridge-lambda 0.01 --tasks 1,2,3,4,5 --episodes 25 --horizon 200
```

**Read.** Ridge closes the split (both seeds → ~0.90) ⇒ diagnosis confirmed, moment
readout is the culprit, adopt ridge as the default dense-inference route. Split persists
⇒ the misaim is elsewhere (actor, not inference) — re-open.

**Depends on.** ZX checkpoints. Eval-only code.

---

## E4 — Kernel normalization for ∫ r dM (region + linear)  **[+eval]**

**Question.** The tex §7 flags that the learned object is a **kernel-smoothed mass**,
not a density ratio: integrating over overlapping ε-balls double-counts and inherits the
dataset density over g, so `Q_r=∫r dM` is not yet justified. This is the gate on the
paper claiming *arbitrary-reward* zero-shot rather than *goal-policy composition*.

**Code needed.** In the z-inference path (`renewal_head.infer_z` / eval harness), add a
kernel/importance normalization: weight each reward sample by an estimate of the local
overlap / inverse dataset density (`1/ρ̂(g_j)` via a kNN density estimate on cube
positions), i.e. compute `z_r ∝ Σ_j r_j w_j B(g_j)` with `w_j` the normalization. Expose
`--kernel-normalize {none,knn}` `--knn-k`.

**Cell.** No training. Target **region** and **linear** (the families that genuinely
query mass, not a single goal), on E1's ZX visitation checkpoints.
```
$PY eval_zeroshot_rewards.py --run-dir $VISZRUN --families region,linear \
  --kernel-normalize knn --knn-k 32 --z-weighting renewal --renewal-head <npz> --gpi
```

**Read.** Region/linear improve under normalization vs. E1's un-normalized numbers ⇒
the mass really is being integrated and the correction matters — the strongest form of
the arbitrary-reward claim. No change ⇒ either the object is already ≈density-calibrated
(good, cite it) or the families are extraction-bound, not integration-bound (diagnose
which via the linear-family achieved-reward-vs-control metric).

**Depends on.** E1 (needs genuine visitation mass to integrate). Eval-only code.

---

## E5 — Action-independence diagnostics of the recurrence factor R  **[+eval]**

**Question.** The whole curriculum rests on `M = Q·R` with **R action-independent**
(tex Eq. factor). Renewal fits validated the *magnitude* claim (Spearman 0.60–0.63
uniformly across horizon buckets; R carries essentially all of occupancy). The two
diagnostics the note lists as "not yet run" test the *action-independence* directly.

**Code needed.** Small analysis script (extend `scripts/rescuing_fb/renewal_head.py` or
a new `r_action_independence.py`) computing, on the FP + ball-visit checkpoints over the
dataset:
1. **Conditional variance of M/Q given g** — how much post-arrival occupancy varies
   across arrivals at the same cube position (grasp/velocity/config differences).
2. **Agreement rate of argmax_a Q(s,a,g) with argmax_a M(s,a,g)** — if R were exactly
   action-independent the argmaxes coincide.

**Cell.** No training; CPU (`/dev/shm/.venv-jax/bin/python`). Run on FP-NoOrtho s0–s2.

**Read.** Low conditional variance + high argmax agreement (>~0.9) ⇒ decomposition is
sound on this data, the curriculum is well-founded, state it as a *measured* fact.
High variance / low agreement ⇒ R carries real control information (secure placement vs.
pass-through) that the split discards — a genuine caveat to add to §3, and a motivation
for an interaction-conditioned R (ties to the CLAUDE.md under-resolution thesis).

**Depends on.** Existing checkpoints. Analysis-only.

---

## E6 — Policy-latent schedule during phase 1  **[config-only + small knob for the anneal]**

**Question.** How should z be sampled in the separated form (tex §"Choosing policy
latents", §8)? Goal-derived z has clear semantics; random z is a surrogate. Does mixing
random directions help or hurt *extraction* (not just training stability)?

**Cells.** Three ZX phase-1 retrains, differing only in the row-z sampling:
- `measure_z_random_frac=0.0`  (goal-only)          — **[config-only]**
- `measure_z_random_frac=0.5`  (constant mix; = current ZX baseline)
- `p_random: 0.0→0.5` annealed over training         — **[+knob]** (the constant knob
  exists; the schedule needs a linear-anneal wrapper, mirroring the existing
  `anneal_*` configs).

**Launch.** As the ZX phase-1 command (group `...-FP-ZX-zfrac{0,0.5,anneal}-d50`),
2 seeds each, cold start.

**Eval.** Direct-z on dirac/mixture/region/linear; the discriminating metric is
**region + linear** extraction (goal-reducible families are insensitive to z coverage).

**Read.** If `zfrac=0` matches `0.5` on region/linear, random-z reaching is unnecessary
(simpler story). If the mix or anneal wins on region/linear, random-z coverage is what
lets inferred `z_r` land on-manifold — cite it as required for arbitrary rewards.

**Depends on.** Parallels E1 (same form). Feeds E1's extraction quality.

---

## E7 — Scope: second environment and/or pixels  **[config-heavy]**

**Question.** The note's lead caveat is "one environment, state observations." Does the
reach-first curriculum transfer?

**Cells.** Re-run the phase-1 (FP-NoOrtho) → phase-2 (ball-visit) → ZX pipeline on:
- `scene` and/or `cube-double` (state) — new env configs, same knobs.
- `cube-single` **pixels** — DrQ/pixel path (verify the pixel encoder + goal-ball on
  cube position is still defined; may need the `measure` knobs wired through the pixel
  agent).

**Eval.** Native GC + the zero-shot grid, same harness.

**Read.** Phase-1 reaches its GC-IQL anchor and phase-2 stays stable on ≥1 new env ⇒
the diagnosis generalizes; the paper claims a *phenomenon*, not a *cube-single* result.
Any env where phase-1 fails to reach anchor is itself informative (where does reach-first
break?).

**Depends on.** Env/pixel configs; largest effort. Do after E1–E4 settle the method on
cube-single.

### E7 as launched (2026-07-24, state obs; pixels still deferred)

**The blocker was removed, not worked around.** The goal region is now a config knob:
`measure_hit_spec`, a STRING of comma-separated `start:stop:eps` observation blocks ANDed
together (`_parse_hit_spec` + `GCIQLFlowAgent._hit_matrix` in `gciql_flow.py`). Default
`'19:22:0.4'` is bit-identical to the literal it replaced — every cube-single checkpoint and
result stands. Only the `measure_loss` region was generalized; the `task_prior` /
`task_prior_actor` mixture arms still hardcode cube-single, so the assert-lattice now refuses
to compose them with a non-default spec rather than train a mis-specified reward.

Regions (verified against ogbench source; each mirrors that env's own success criterion):

| env | `measure_hit_spec` | why |
|---|---|---|
| cube-single | `19:22:0.4` | cube xyz within 4cm (xyz_scaler 10) |
| cube-double | `19:22:0.4,28:31:0.4` | `cube_env.py:728` success = **all** cubes within 4cm |
| scene | `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6` | `scene_env.py:552` success = cube ∧ button0 ∧ button1 ∧ drawer ∧ window. One-hot equality ⇔ L2 ≤ 0.5 (‖e_i−e_j‖ ∈ {0,√2}); drawer/window are 0.04 m at scalers 18/15. Velocity and plunger-position dims are excluded — they are not part of success. |

**Pre-flight sparsity (measured on the real datasets at the training batch size B=1024).**
This is the load-bearing risk, and it is why the wave carries controls:

| cell | off-diag positives / row | vs cube-single |
|---|---|---|
| cube-single (the 0.967 cell) | 14.0 | 1× |
| cube-double, 2-cube conjunction | **0.51** | 27× sparser |
| cube-double, cube-0 only (control) | 18.6 | 1.3× denser |
| scene, 5-way conjunction | **3.07** | 4.5× sparser |
| scene, cube-only (control) | 61.9 | 4.4× denser |

Without the controls a floor on the conjunctive cells is uninterpretable — it would confound
"reach-first does not transfer" with "the reward is 27× sparser here". The controls' regions
ignore factors the native eval scores, so read their **training traces and partial credit**,
not their overall success.

**Anchor calibration.** The note's "0.90 bar" is cube-single-specific. Matched-architecture
GC-IQL bilinear-d50: scene **0.292** (flow) / 0.260 (ddpgbc), stock-MLP 0.516, FB 0.047
(`results/summary_agents.csv`). No cube-double number existed at all, so this wave trains it.

**WAVE 1 LAUNCHED** — tmux `s08e7`, `scripts/rescuing_fb/launch_spec08_e7_phase1.sh`,
out `/home/adityamohan/exp_spec08`, logs `/dev/shm/spec08_logs/`, 10 jobs × 500k × 2 seeds,
one per GPU on the (idle) 8×L4 box:

| tag | config | run_group |
|---|---|---|
| `cd_fp_s{0,1}` | `cube_double_state_measure_firstpassage_noortho_d50` | `GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50` |
| `cd_fp1c_s{0,1}` | `cube_double_state_measure_fp_noortho_1cube_d50` | `...-Measure-FP-NoOrtho-1Cube-d50` |
| `sc_fp_s{0,1}` | `scene_state_measure_firstpassage_noortho_d50` | `GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50` |
| `sc_fpc_s{0,1}` | `scene_state_measure_fp_noortho_cubeonly_d50` | `...-Measure-FP-NoOrtho-CubeOnly-d50` |
| `cd_anch_s{0,1}` | `cube_double_state_gciql_flow_bilinear_d50` | `GCIQL-CubeDouble-Flow-Bilinear-d50` |

**WAVE 2 (phase 2), ready but not launched** — needs wave 1's checkpoints for the warm start.
Configs `cube_double_state_measure_ballvisit_d50.yaml` and `scene_state_measure_ballvisit_d50.yaml`
carry the exact cube-single phase-2 recipe (`measure_fp_absorbing=false`,
`measure_stopgrad_psi=true`, `actor_q_center=true`, same region), and expect
`+agent_overrides.warm_start_ckpt=<phase-1 run dir> +agent_overrides.warm_start_epoch=500000`.

**Tests.** `tests/test_gciql_measure_hit_spec.py` (6: parser, default bit-identity +
checkpoint compat, conjunction semantics, multi-block training, phase-2 composition with
frozen ψ, assert-lattice) — green, plus the 23 existing measure/phase-2 tests re-run green.
Both new env configs smoke-tested end-to-end (300 steps, checkpoint written, spec confirmed
intact through the Hydra→absl CLI round trip).

**Not ported (follow-on, gates the zero-shot grid only).** The zero-shot reward-family harness
is still cube-single-only: `eval_zeroshot_rewards.py` / `t6_gpi_eval.py` /
`t6_generality_probe.py` / `renewal_head.py` / `measure_readout_head.py` each redeclare
`CUBE_OBS=slice(19,22)`, `CUBE_QPOS=slice(14,17)`, `EPS_OBS=0.4`, `EVAL_TASKS` on
`cube-single-play-singletask-task{i}-v0`, and the cube-single table `REGIONS`. Two are
*silent wrong-answer* bugs rather than crashes: `cur_task_info["goal_xyzs"].reshape(-1)[:3]`
truncates cube-double to cube 0, and scene has no `goal_xyzs` key at all (it uses
`block_xyzs`/`button_states`/`drawer_pos`/`window_pos`). The **native GC eval used by wave 1
is unaffected** — it is env-parameterized through `GCIQL_FB_ENV` in `sitecustomize.py`.
Recommended fix when the grid is wanted on these envs: one `scripts/rescuing_fb/env_geometry.py`
registry keyed by `env_name`, imported by all five scripts.

---

## Status (2026-07-24)

- **E1 — LAUNCHED** (tmux `s08e1`, `launch_spec08_e1.sh`): `visz_s0`/`visz_s1` warm-started
  from the FP-ZX phase-1 ckpts, config `cube_single_state_measure_ballvisit_zx_d50.yaml`,
  group `GCIQL-State-Flow-Measure-BallVisit-ZX-d50`, out `/home/adityamohan/exp_spec08`.
  Warm-start param-tree asserts passed; step-0 eval 0.90/0.936 = ZX anchor (splice good).
  500k running on GPUs 0/2. **Payoff eval pending checkpoints** (direct-z region/linear).
- **E2 — knob IMPLEMENTED + LAUNCHED.** `freeze_actor` added to `gciql_flow.py` (create():
  `optax.multi_transform` masking `modules_actor`+`modules_actor_vf` to `set_to_zero`;
  default-off byte-identical). Tests: `tests/test_gciql_phase2_occupancy.py` 7/7,
  `test_gciql_measure_axes.py` 6/6. Run: tmux `s08e2`, `launch_spec08_e2.sh`,
  `faz_s0`/`faz_s1` warm-started from FP-NoOrtho, config
  `cube_single_state_measure_ballvisit_frozenactor_d50.yaml`, group
  `...-BallVisit-FrozenActor-d50`. Log confirms `[freeze_actor] masked ...`; step-0 eval
  0.964 = FP anchor (actor frozen at phase-1 policy). 500k running on GPUs 3/4.
- **E3 — ridge inference IMPLEMENTED + forensic RUNNING.** `ridge_z` (GLS/whitened moment,
  trace-relative λ) + `--z-inference ridge --ridge-lambda` added to
  `eval_zeroshot_rewards.py`; tests `tests/test_eval_zeroshot_rewards.py` 21/21 (mirrors the
  proven `evals/z_inference.infer_l1` properties). Forensic (tmux `s08e3`,
  `run_spec08_e3_ridge.sh`): dense family, moment + ridge λ∈{1e-3,1e-2,1e-1}, ZX phase-1
  s0/s1, on the existing ckpts → `analysis/rescuing-fb/e3_dense_*.json`. Resolves the
  0.90/0.13 split.
- **E5 — script DONE + RUN.** `scripts/rescuing_fb/r_action_independence.py` (loads FP-Q +
  BallVisit-M critics, shared frozen ψ; ICC of R=M/Q given g + argmax_a Q vs M agreement +
  per-(s,g) rank corr). Result (s0/s1): argmax agreement **0.30/0.30**, R ICC **0.24/0.29**,
  within-goal CV **1.3/1.8** → **strict pointwise action-independence of R is moderately
  refuted** (consistent with §3's hedge; caveat: random-action Q-floor inflates M/Q noise —
  the goal-averaged magnitude claim, renewal Spearman 0.60, is a separate smoother estimand).
  Outputs `analysis/rescuing-fb/e5_action_indep_s{0,1}.json`.
- **E6 — knob IMPLEMENTED + arms LAUNCHED.** ZX phase-1 config
  `cube_single_state_measure_fp_zx_d50.yaml` + `measure_z_random_frac_anneal` knob
  (`gciql_flow.py`: ramp the bernoulli random-z prob 0→frac over `anneal_ramp`, branch static,
  knob-off bit-identical; tests 2/2). Arms (`launch_spec08_e6.sh`): `zfrac0` goal-only
  RUNNING (tmux `s08e6zf0`, cold start); `anneal` queued at the MAX=6 gate (tmux `s08e6an`);
  the `zfrac=0.5` arm reuses the existing `zx_s0/s1`.
- **E4 — pending.** Kernel normalization; E1's phase-2 checkpoints have since landed, so it
  is unblocked.
- **E7 — `measure_hit_spec` knob IMPLEMENTED + WAVE 1 LAUNCHED (2026-07-24 23:31).** The
  hardcoded ε=0.4 / obs[19:22] region is now a config knob (default bit-identical); phase 1
  runs on cube-double and scene with env-faithful conjunctive regions, each paired with a
  sparsity control, plus the missing cube-double GC-IQL anchor. 10 jobs, tmux `s08e7`,
  ~9 h. Phase-2 configs written and waiting on those checkpoints. Full detail in the E7
  section above; the zero-shot reward-family harness is NOT yet ported (native GC eval is).

## Suggested ordering / batching

1. **Launch E1 + E6 together** (same ZX form, both are retrains) — the headline plus its
   coverage ablation, ~9 h.
2. **In parallel, E5** (CPU analysis, no GPU) and the **E2 knob + test** (code).
3. When E1 checkpoints land: **E3** (ridge) and **E4** (kernel norm) are eval-only, fast.
4. **E2 frozen/frozen** run once the knob is green.
5. **E7** last, once the method is settled on cube-single.

Each cell: add its `run_group` to `scripts/results_map.py` and refresh the CSVs
(`.venv/bin/python scripts/compile_results_csvs.py --refresh`) after collection, and add
a `DECISIONS.md` entry — the standing rule for this repo.
