# Cube bottleneck-mechanism probe — does the estimator's deficit live at the control bottleneck? (2026-09-04)

Status: SPEC (analysis-only; no training). Branch `density-fb`, worktree
`~/git/Factored-FB-seam`. Written for an executing agent starting fresh.
All paths are repo-root-relative to the worktree named in brackets:
`[seam]` = `~/git/Factored-FB-seam` (jax, Density-FB + the 2026-09-02 cube
checkpoints), `[torch]` = `~/git/Factored-FB` (the phase diagnostic, state
restore, counterfactual probes). Both are worktrees of one repo; the branch
decides what exists, not the directory.

Read first: `PAPER/ICLR/content/introduction.tex:10-27` (the hypothesis in the
paper's words), `[torch] PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md` Part 2 and
Part 7 (mechanism story, and why it is "not yet established"),
`[torch] PAPER/fb-cube-failure/fb_cube_failure.tex` §Finding 2–3.

---

## 0. Purpose

The paper says (`introduction.tex:10`): *"We hypothesize that this loss of
precision arises at control bottlenecks: states where a small number of precise
actions determine access to a much broader region of future behavior."* The
word is "hypothesize" on purpose. What cube evidence exists:

| established on cube | source |
|---|---|
| FB's failure is located **after** the grasp (drop / misplace); state FB grasps about as often as GCIQL | phase diagnostic, `[torch] PAPER/fb-cube-failure/findings_so_far.md` §4c |
| FB's `B` under-resolves the interaction variable (held vs free AUC .75 vs PSM .94) and the terminal set (R² .39, near-miss acc .67) | task-distribution probe; fb_cube_failure Finding 3 |
| Giving `B` the object coordinates does **not** rescue FB (.11 → .34 cube-single) while Density-FB at **full-state** input reaches .91; the estimator carries the larger share (.300 → .736 on cube-double) | `PAPER/ICLR/table_privcoord.tex`; `experiments.tex:115-124` |
| The only direct test of capacity misallocation under dwell (fork-and-dwell, tabular) was **null twice**; FB's bilinear critic was flat | DENSITY-FB-EXPLAINED Part 7 |

So "the representation is too coarse" is refuted as the whole story, the
estimator is where the win lives, and **nothing on cube has asked what the two
estimators do differently at the bottleneck.** Two analysis-only tests close
that gap, on the checkpoints we already have. If they come back clean, the
ExoRL passive-object experiment becomes a generalization check; if not, the
introduction's mechanism paragraph must be rewritten before a reviewer does it.

**Non-goals.** No training. No new estimator. No ExoRL. No pixels (the state
restore path is state-only — `[torch] evals/phase_probe.py:208-232` rejects
pixel envs for S1/S2 and nothing here changes that).

---

## 1. Arms and checkpoints (all present, all 500k, `[seam] exp/`)

Directory pattern `exp/<domain>_<critic>_flowbc_seed_<n>/params_500000.pkl`;
`config.json` carries `{seed, critic, actor, domain, override, config}`; the
critic name resolves to `impls/configs/critic/<name>.yaml`.

| role | critic | B input | cube-single seeds | cube-double seeds |
|---|---|---|---|---|
| **primary A: plain FB, Touati-parity** | `fb_ref` | full state | 0, 1, 2 | 0 |
| **primary B: Density-FB, no privileged coords** | `density_fb_state` | full state | 0, 1, 2 | 0, 1, 2 |
| secondary: Density-FB as shipped | `density_fb` | `φ` = cube xyz (`measure_hit_spec 19:22:0.4`; cd adds `28:31:0.4`) | 0, 1, 2 | 0, 1, 2 |
| secondary: plain FB + privileged coords | `fb_oc` | `φ` | 0 | 0 (**250k only** — do not mix budgets; drop) |
| reference: repo FB (ships `ortho_coef 1.0`) | `fb` | full state | 0 | 0 |

**The primary pair is `fb_ref` vs `density_fb_state`**: both towers see the
full observation, so the only difference is the estimator. That is the
comparison the mechanism claim is about. `density_fb` is reported alongside
because it is the paper's method; `fb_oc` is the "coarseness" control and is
already known to land at .34.

Final-step checkpoints only. Never quote a mid-run eval (standing rule).

**Gate G1 — loader parity.** Before either test, re-evaluate one cell per arm
family through the probe's own loader (`impls/utils/flax_utils.py:174
restore_agent`, construction pattern `scripts/probe_eval_interface.py:138`) on
the standard 5-task goal-conditioned protocol, 20 episodes, and match
`table_privcoord.tex` within the eval noise floor (±0.05 absolute on
cube-single). Cells: `fb_ref` cs seed 0, `density_fb_state` cs seed 0. If G1
fails, nothing downstream is read.

---

## 2. Test 1 — held-start evaluation

**Question.** Remove the bottleneck at test time by starting the episode with
the cube already secured in the gripper. If plain FB's deficit is *at* the
bottleneck, FB should recover toward Density-FB from a held start. If the
deficit is in post-grasp control (transport / retention), it should not.

**Prior art, and why it is not enough.** The torch phase probe already has a
pre-grasped scenario **S2** (`[torch] evals/phase_probe.py:196-206`: cube
teleported to the effector at the *reset* pose, gripper `ctrl=255`, 5 settle
steps). Recorded results, all "S2 does not exceed S0":

| method (torch ckpts) | S0 success | S2 success | source |
|---|---|---|---|
| PSM | 28.6 ± 4.9 | 20.7 ± 3.9 | `[torch] analysis/profiles/psm_phase/aggregate/overall.md` |
| RLDP | 26.2 ± 6.9 | 20.3 ± 3.9 | `.../rldp_phase/aggregate/overall.md` |
| GCIQL bilinear | 95.0 ± 5.4 | 79.5 ± 18.9 | `[torch] analysis/phase_probe_gciql_bilinear/aggregate/overall.md` |

Two problems: (i) S2 is an **off-distribution clamp** — arm at the home pose
holding a cube it never grasped, which the readout itself flags; even GCIQL
loses 15 points, so a drop under S2 says nothing about a method. (ii) It has
never been run on Density-FB, nor on the seam FB checkpoints. Test 1 replaces
S2 with an in-distribution held start and runs both estimators.

### 2.1 Start conditions

- **H0 — standard reset.** `env.reset(options=dict(task_id=t))`. This is the
  table protocol and must reproduce G1.
- **H1 — in-distribution held start.** Restore a *recorded* held state from the
  play dataset, then let the policy take over toward the task's own goal.
  Selection criterion is the one `[torch] scripts/value/counterfactual_value_probe.py:74-85
  sample_grasps` uses — `obs[18] > 0.5` (contact) and `obs[21]/10 > 0.05`
  (cube lifted; obs is ×10 scaled, see `[torch] scripts/atlas/oracle.py:11`) —
  applied to episode files of the **same dataset the arms trained on**.
  Additionally require the held run to persist ≥ 5 consecutive steps
  (`k_steps`, `[torch] evals/phase_probe.py:36`) so the state is not a
  transient. Bank: **64 states per env**, `numpy.random.default_rng(0)`,
  written to `analysis/cube_bottleneck/<env>/held_bank.npz` with
  `(obs, qpos, qvel, episode, step)` and reused by **every** arm, seed and task,
  so paired comparisons are legitimate.
- **H2 — S2 clamp (continuity only).** Port `apply_scenario` S2 verbatim, run at
  reduced episode count (10 per task), reported in a separate column so the new
  numbers can be placed against the old PSM/RLDP/GCIQL rows. Not a deciding
  input.

### 2.2 Restore procedure and its gate

Two independent existing recipes disagree on the gripper sign — `phase_probe.py`
closes with `ctrl = 255`, `counterfactual_value_probe.py:165` "establishes the
grasp" with `ctrl = 0.0`. **Do not guess.** Confirmed today on this box:
`env.action_space = Box(-1, 1, (5,))`, `u._gripper_actuator_ids = [6]`, and
`u._target_effector_pose` exists. The executor must:

1. `env.reset(options=dict(task_id=t))` (valid goal mocap for task `t`);
2. `u.set_state(qpos, qvel)` from the bank row, `mujoco.mj_forward`;
3. **reset the hidden IK integrator** — `u._target_effector_pose` lives outside
   `mjData` (`[torch] scripts/atlas/prefix_replay.py:1-8` documents this as the
   reason snapshot/restore was abandoned there); set it from the restored
   effector pose, or prove empirically it does not matter (step 5);
4. hold the gripper-close command for `N_SETTLE = 5` internal steps with no
   policy action, using whichever ctrl value actually closes (determine once
   with a 2-value test, record the answer in the results doc);
5. **Gate G2 — restore fidelity.** After steps 1–4 and one further step with the
   zero action, the phase label must be `held` (`transport_mask`,
   `[torch] evals/_profile_core.py:27-32`: lift > 0.03 above table and grip >
   0.5) and the cube must have moved < 0.04 m from the recorded `qpos`
   (`[torch] scripts/atlas/gates.py:25-55 dataset_state_restore_drift` uses
   the same bound). A bank row failing G2 is dropped **before** any arm sees
   it; report the drop count. If > 25 % of rows drop, the restore is not
   in-distribution and Test 1 is not read.

### 2.3 Protocol

- Goals: the env's 5 standard tasks, `z = map_eval_cond(goal)` exactly as the
  table (`impls/critics/fb.py:89-97`, `impls/critics/density_fb.py:797-839`).
  For H1 the held cube is wherever the dataset had it; the goal is the task's.
- Episodes: **50 per (arm, seed, task, start)**, bank rows cycled in fixed
  order, same rollout seeds across arms. Episode budget = the env default.
- Both worktrees' phase machinery is torch-free numpy: vendor
  `classify_phases` + `PhaseThresholds` (`[torch] evals/phase_probe.py:32-102`)
  and `transport_mask` into `[seam] scripts/probes/cube_bottleneck/phases.py`
  with a test that the vendored copy is byte-identical to the source.

### 2.4 Metrics (per seed, never pooled across seeds)

- `success` per (arm, seed, task, start), with 95 % bootstrap CI over episodes.
- Terminal decomposition from H1: `maintain_fail` (cube back on table at end),
  `transport_fail` (still held, not at goal), split further into
  `dropped / near_goal / far_goal` (`[torch] scripts/probes/phase_probe_transport_buckets.py:27-33`).
- `hold_retention_50` = fraction of the first 50 steps in which the held mask is
  true (H1 only). `time_to_first_drop` (steps; censored at episode end).
- `Δ_held = success(H1) − success(H0)` per (arm, seed, task) and its CI.

### 2.5 Pre-registered readings (cube-single is deciding; cube-double reported)

Deciding arms: `fb_ref` and `density_fb_state`, 3 seeds each.

- **R-HELD1 — the bottleneck is load-bearing for FB.** `Δ_held(fb_ref) ≥ +0.30`
  on ≥ 2 of 3 seeds (task mean), **and** `success_H1(fb_ref)` within 0.10 of
  `success_H1(density_fb_state)` on those seeds. → Once past the grasp, plain FB
  controls the cube about as well as Density-FB; the estimator's deficit is
  concentrated at the acquisition step. The introduction's "critical action"
  wording is licensed as shown on cube.
- **R-HELD2 — the deficit lives after the bottleneck.** `Δ_held(fb_ref) ≤ +0.10`
  on ≥ 2 of 3 seeds **and** `hold_retention_50(fb_ref) < 0.6` or
  `maintain_fail + dropped ≥ 50 %` of FB's H1 failures, while
  `success_H1(density_fb_state) ≥ success_H0(density_fb_state) − 0.05`. →
  Handing FB the secured cube does not help; it drops or misplaces it. The
  mechanism paragraph must be rewritten around *post-bottleneck* successor
  structure (diffuse held futures under-resolved), dropping "a small number of
  precise actions determine access". Consistent with the 2026-05 phase
  diagnostic and the S2 rows above.
- **R-HELD3 — mixed.** FB gains but stays far below Density-FB from held
  (`+0.10 < Δ_held < +0.30`, gap to Density-FB > 0.10). → Both channels
  contribute; report the split, and Test 2 decides which one the estimator is
  responsible for.
- **Floor clause.** If `density_fb_state` itself loses > 0.15 from H0 to H1 on
  ≥ 2 seeds, the held start is off-distribution for reasons G2 did not catch
  (IK integrator, contact impulses); report the restore diagnostics and read
  nothing. If G2 drops > 25 % of the bank, same.

**Prediction, stated before the numbers: R-HELD2.** Every prior probe puts FB's
loss after acquisition, and S2 helped none of PSM/RLDP/GCIQL. Being wrong here
would be a useful surprise.

---

## 3. Test 2 — phase-conditioned measure probe at the bottleneck

**Question.** At grasp-adjacent states, do the two estimators put successor
mass on *held* futures for the actions that actually secure the cube, and on
*free* futures for the ones that do not? This is the measure-level version of
the fork-and-dwell question, on real data with the real critics.

### 3.1 State bank (the decision points)

From the same play dataset, select **P2 states**: gripper contact channel
`obs[18] > 0.5`, cube on the table (physical `qpos[14:17][2]` < 0.03 above
table), cube-effector xy distance < 0.06 (`[torch] scripts/atlas/banks.py:16-23`
labels this phase 2). **K = 64 per env**, rng 0, stored with `(obs, qpos, qvel)`.
Controls: **P3** (already held, from the Test 1 bank) and **P1** (contact = 0,
distance > 0.10), 32 each.

### 3.2 Counterfactual actions and their ground-truth outcome

At each bank state, form an action set of size **12**: the dataset's own action
at that row; 6 samples from the BC flow actor (`flowbc`, the arms' own actor
class, any arm — it is not the quantity under test); the dataset action with
the gripper channel (index 4 of the 5-dim action) forced to its closed extreme;
the same forced open; and 3 uniform-random actions. For each `(s, a)`:
restore `s` (§2.2 recipe, G2 applies), apply `a` once, then continue under the
BC actor for **H = 25 steps**, **8 independent continuations**; record
`p_secure(s,a)` = fraction of continuations whose phase label at step 25 is
`held`.

**Gate G4 — the bank is at the bottleneck.** Within a P2 state, the std of
`p_secure` across the 12 actions must exceed 0.20 for at least 40 of 64
states; otherwise actions do not matter at these states and they are not
decision points. Re-select with a tighter distance bound before proceeding.
At P1 the same std should be < 0.10 (nothing secures in 25 steps from far);
at P3 it should be small too (already held). These two are the null controls.

### 3.3 Estimator readouts at each `(s, a)`

Pool `U` = **1024 dataset next-states** (fixed, rng 0) with phase labels, so
`held(U)` is known. Goal `g` = each of the 5 task goals; `z` / index as in §2.3.

- **Plain FB** (`impls/critics/fb.py`): `M(s,a,z; u) = F(s,a,z)·B(u)` for all
  `u ∈ U` (the einsum at `:143`, evaluated online, min over the two heads to
  match the training target). Define
  `held_mass_FB = Σ_{u∈held} max(M,0) / Σ_u max(M,0)` and
  `row_mass_FB = mean_u M(s,a,z;u)` (the quantity a mass bound would cap).
  Also `Q_FB(s,a,z) = F·z` (`q_fn`, `:171-181`).
- **Density-FB** (`impls/critics/density_fb.py`): `w(s,a,g; u_j) = softmax_j`
  of the logits at `:863-866` over the same pool `U` (use the critic's own
  query-row builder `_b_query :611` / `_kernel_rows :536` with `U` supplied as
  the pool). `held_mass_D = Σ_{j∈held} w_j` — a probability by construction.
  `Q_D` from `q_fn :927-971`.

Both readouts must run through the **same** pool and the same `z`/goal;
assert `|U| = 1024` and the label vector identical across arms.

### 3.4 Metrics (per state, then per seed)

- `auc_heldmass(s)`: within-state AUC of `held_mass` against
  `1[p_secure > 0.5]` over the 12 actions (undefined if G4 fails for that
  state; report coverage). Seed-level = median over states. Same for `Q`.
- `spearman_heldmass(s)`: rank correlation of `held_mass` with `p_secure`.
- `calib_slope`: OLS slope of `held_mass` on `p_secure` pooled over states.
  Density-FB's is interpretable as a probability calibration; FB's is not, so
  report FB's on the normalized `held_mass_FB` only.
- `row_mass_ratio_FB` = median `row_mass_FB` at P3 / median at P2, per seed.
  The true total occupancy is `1/(1−γ)` at every state; this is the direct
  observable for the "inflate scale instead of reallocate" route
  (DENSITY-FB-EXPLAINED §2.3). Density-FB's analogue is 1 identically.
- Null controls: the same AUCs at P1 must sit in `[0.4, 0.6]` for both arms.

### 3.5 Pre-registered readings (cube-single deciding, 3 seeds per arm)

- **R-MEAS1 — misallocation at the bottleneck, shown.** At P2,
  `auc_heldmass(fb_ref) < 0.60` on ≥ 2 of 3 seeds while
  `auc_heldmass(density_fb_state) ≥ 0.75` on ≥ 2 of 3, both arms' P1 controls
  inside `[0.4, 0.6]`, and `auc_Q(fb_ref) < 0.60` too. → Plain FB's measure
  does not distinguish securing from non-securing actions where it matters;
  the normalized estimator does. The hypothesis moves from "we hypothesize"
  to "we show" on cube, with a figure: `held_mass` vs `p_secure` scatter per
  arm.
- **R-MEAS2 — the measure sees it, the readout loses it.**
  `auc_heldmass(fb_ref) ≥ 0.75` but `auc_Q(fb_ref) < 0.60`. → `F·B` resolves
  the bottleneck; projecting onto `z` and reading `F·z` throws it away. This
  points at the z-conditioning obstruction found on the bridge track
  (`stack-divergence-audit-20260805`), not at the estimator. Report as such;
  the paper's mechanism paragraph must then be about the readout.
- **R-MEAS3 — both resolve it.** Both arms ≥ 0.75 at P2. → The estimators do
  not differ at the bottleneck. Combined with R-HELD2 this says the difference
  is downstream (retention / transport), and the introduction's bottleneck
  framing is not supported on cube. Write that plainly.
- **R-MEAS4 — the scale route is real (independent of AUC).**
  `row_mass_ratio_FB > 3` on ≥ 2 of 3 seeds. → FB's predicted total mass
  differs by phase while the truth is constant: direct evidence for the
  unconstrained-scale candidate factor. Report alongside whichever of
  R-MEAS1–3 fired; it is not exclusive with them.
- **Floor clause.** G4 fails after one re-selection → the bank is not at a
  decision point and no P2 number is read. P1 controls outside `[0.4, 0.6]`
  for an arm → that arm's AUCs carry a state-dependent bias; report but do not
  read R-MEAS1–3 for it.

**Prediction, stated before the numbers.** R-MEAS2 or R-MEAS3, with R-MEAS4
plausible. The fork-and-dwell null and the flat bilinear critic argue against
R-MEAS1; the bridge track's z-obstruction result argues for R-MEAS2. If
R-MEAS1 fires, that is the paper's missing mechanism figure.

---

## 4. Implementation notes

- **Where.** New package `[seam] scripts/probes/cube_bottleneck/`:
  `phases.py` (vendored, tested byte-identical), `restore.py` (§2.2 recipe +
  G2), `banks.py` (§2.1, §3.1), `held_start_eval.py` (Test 1),
  `measure_probe.py` (Test 2), `readings.py` (applies §2.5/§3.5 mechanically
  and prints which fired). Outputs under `analysis/cube_bottleneck/` (untracked)
  with CSVs `held_start.csv`, `measure_probe.csv`, `banks_stats.csv`.
- **Loader.** `restore_agent` + config as in `scripts/probe_eval_interface.py`.
  `os.environ.setdefault("MUJOCO_GL", "egl")` before any mujoco import.
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Probes are small; `JAX_PLATFORMS=cpu`
  is acceptable and avoids fighting the GPUs, but check that the pixel-free
  path does not silently pick a different code branch under CPU.
- **Datasets.** The arms' own training datasets (`cube-single-play-v0`,
  `cube-double-play-v0`); confirm the dataset path from each run's
  `config.json` and bank from that file set, not from a re-download.
- **Cost.** Test 1 ≈ 2 arms × 3 seeds × 5 tasks × 2 starts × 50 eps ≈ 3k
  episodes per env, plus secondary arms; a few CPU-hours per env with the
  four BLAS thread pins. Test 2 rollouts ≈ 64 × 12 × 8 × 25 ≈ 150k env steps
  per env; critic evaluation is negligible.
- **Run in tmux**, one window per env; never a bare `&`.
- **Do not touch** the seam's dirty files (`impls/critics/{fb,density_fb}.py`,
  `impls/utils/datasets.py`, `PAPER/decoding-bilinear/table.tex`) — they belong
  to the other active track.
- **Per-seed reporting always.** Seed 43-style isolated collapses have shown up
  with mid-pack training signals before; a collapsed seed is a row, not an
  outlier.

## 5. Deliverables

1. `analysis/cube_bottleneck/RESULTS.md`: G1–G4 gate table; per-seed tables for
   §2.4 and §3.4; **every** R-HELD*/R-MEAS* called explicitly, including "does
   not fire" with the conjunct that blocked it; the gripper-ctrl convention
   and the IK-integrator handling recorded as facts.
2. Figures: (a) `success` H0 vs H1 per arm with per-seed points; (b)
   `held_mass` vs `p_secure` scatter at P2, one panel per arm, P1 control as a
   faint inset; (c) `row_mass_FB` by phase.
3. A one-paragraph proposed replacement for `introduction.tex:10-27`
   conditional on the reading, left for the user to accept — not applied.
4. A new entry in `[torch] PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md` Part 7
   summarizing which candidate factor survived.

## 6. Open questions for the user (do not resolve unilaterally)

- **Q1.** Primary pair confirmed as `fb_ref` vs `density_fb_state` (matched
  full-state input)? The alternative, `fb` vs `density_fb`, is the paper's
  headline pair but confounds estimator with `B` input and ortho coefficient.
- **Q2.** Cube-double: run both tests (seeds exist for both primary arms:
  `fb_ref` seed 0 only, `density_fb_state` 0–2) or cube-single only for the
  deciding numbers? Spec treats cube-single as deciding and cube-double as
  reported.
- **Q3.** Is the H2 (old S2 clamp) continuity column worth its 10 episodes per
  task, or drop it?
