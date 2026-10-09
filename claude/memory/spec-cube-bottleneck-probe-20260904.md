---
name: spec-cube-bottleneck-probe-20260904
description: "2026-09-04 EXECUTED: cube bottleneck probe REFUTES the paper's control-bottleneck story. R-HELD2 fired 3/3 (held start does not rescue FB, .21->.26 vs Density-FB .99; 49-56% of its failures drop the cube). Test 2 REVERSED the expected ordering — plain FB discriminates securing actions BETTER (AUC .55/.78/.75, Q .95) than Density-FB (.44/.55/.64, Q .50); best discriminator fb@ortho1.0 (.91) is the worst controller (.096). R-MEAS4 negative kills the unconstrained-scale candidate (row mass 87-105 vs truth 1/(1-g)=100, <8% phase variation)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fde0be1b-5c3c-4bd3-a1a9-6dea26223182
  modified: 2026-09-04T04:27:41.678Z
---

Spec `~/git/Factored-FB-seam/docs/specs/2026-09-04-cube-bottleneck-mechanism-probe.md`.
Code `[seam] scripts/probes/cube_bottleneck/` + `tests/probes/` (29 tests, UNCOMMITTED).
Results `[seam] analysis/cube_bottleneck/{cube-single,cube-double}-play-v0/RESULTS.md`.
User answers: primary pair `fb_ref` vs `density_fb_state`; cube-single deciding; H2 kept.

**Outcome — the intro's bottleneck mechanism is refuted on cube, not merely unestablished.**
- **R-HELD2 fired 3/3 seeds** (the pre-registered prediction). In-distribution held start moves
  `fb_ref` .21→.26 (Δ +.020/+.100/+.032) while `density_fb_state` sits at .99. 49-56% of FB's
  held-start failures end with the cube back on the table; hold-retention 70-75% vs 81-86%.
  Density-FB itself GAINS from a held start (.90→.99), so the floor clause never fires.
- **NO R-MEAS reading fires — the ordering is reversed.** All three presuppose FB is the weaker
  discriminator at P2. Measured: FB `auc_heldmass` .55/.78/.75 and `auc_Q` .95/.95/.91;
  Density-FB .44/.55/.64 and .55/.50/.50. Bottleneck discrimination is ANTI-correlated with
  control across 4 arms. **Do not claim Density-FB wins by resolving bottleneck actions.**
- **R-MEAS4 negative — drop "unconstrained scale" from the paper's candidate factors.** FB's
  mean row mass 87-105 against the true 1/(1-γ)=100, <±8% across P1/P2/P3 (gate wanted >3×).
  FB's measure is well calibrated in total mass and phase-invariant in it.
- Gates: G1 within .024 on all 10 arms (cube-single) and all 8 (cube-double); G2 kept 64/64 both
  envs; G4 44/64. **cube-double is REPORTED not read**: both deciding arms at floor (.012/.04-.09)
  AND a held start helps nobody there, `density_fb` at .6 included (it needs BOTH cubes placed).

**Implementation facts that will bite the next restore-based probe:**
- `counterfactual_value_probe.py:165`'s `ctrl=0.0` "establish the grasp" is INVERTED — that is the
  OPEN extreme. It passes the geometric `held` test while contact collapses to .01 (latent drop).
  **255 / action channel +1 closes** (`ctrl = 255*target_gripper_opening`, driver joint rises as
  the Robotiq closes). G2 must check contact at hand-over, not just lift+grip geometry.
- **`obs[18]` needs rebuilding after a qpos/qvel restore.** It is `||cfrc_ext||/50`, which depends
  on `data.ctrl` — state no dataset records. mj_forward alone → 0.00; +`mj_rnePostConstraint` →
  0.16; **two** settle steps with the gripper commanded closed → .964 vs a recorded .963.
  N_SETTLE=2 (not the spec's 5): drift grows linearly and G2 starts failing at 20.
- **There is NO cross-step IK integrator** on this ogbench build: `manipspace_env.set_control`
  recomputes effector pose/yaw/opening from `_data` every call and consumes
  `_target_effector_pose` inside that same call. `scripts/atlas/prefix_replay.py`'s docstring is
  wrong for this path — snapshot/restore IS available, every obs channel but contact
  reconstructs to <0.01. 98.7% of restored channels sit inside the dataset's held-state 1-99 band.
- **Batch the rollouts.** Sequential per-episode/per-continuation actor calls made both tests
  untenable (plain FB fails ⇒ every episode runs to the step limit, ~12h/arm). Lockstep-stepping
  a pool of envs with ONE batched `sample_actions` (the `scripts/atlas/oracle.py` trick) measured
  131-703× fewer actor-seconds; one arm went 12h → 2m44s. See [[jax-venv-location]].
- Spec-internal tension resolved and documented: §3.4 wants P1 AUCs in [.4,.6] while §3.2/G4
  expect NO label variation at P1. Both cannot hold — P1 coverage is 0.00, so every P1 AUC is
  UNDEFINED, not .5. Treated as the null holding by construction, verified via G4's std check.
- `tabulate` is missing from the torch `.venv`; run `readings.py` under `/dev/shm/.venv-jax`.

Deliverables 3-4 written: `[seam] analysis/cube_bottleneck/PROPOSED-INTRO-PARAGRAPH.md`
(replacement for `introduction.tex:10-27`, deliberately **NOT applied**) and a new §7.1 in
`[torch] PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md`. Related: [[density-fb-privileged-coords-20260902]],
[[primary-paper-dir-iclr]], [[fb-state-run-name-map]].

**RESULTS (same day, executor ran it; `analysis/cube_bottleneck/cube-single-play-v0/RESULTS.md`).**
R-HELD2 FIRED 3/3: held start gives fb_ref Δ +.02/+.10/+.03 (≈.25) vs density_fb_state ≈.99;
FB drops mid-transport (median step 13-18, not a restore transient; not state-specific;
gripper cmd at t=0 = dataset = Density-FB). Scale exonerated (row mass 87-105 vs 100).
FB's Q ranks the grasp action at AUC .95, Density-FB's at .50 — **NOT a mechanism**: Test 2's
label was rolled under fb_ref's actor (GT_CRITIC=fb_ref); under Density-FB's own actor the
action-spread halves (.29→.15). Policy-conditioned measure ⇒ worse policy ⇒ more action
dependence. Executor's PROPOSED-INTRO-PARAGRAPH over-reads this; do not use its
"anti-correlated" claim. Bottleneck story dropped; mechanism = post-acquisition goal-blindness.
Synthesis: `Factored-FB-seam/docs/notes/2026-09-04-density-fb-central-perspective.md`
(three separable differences: (a) object-outcome target, (b) direct goal vs z ρ-moment,
(c) CE on simplex; retrofits transplant only (c) ⇒ fail at full state, work with coords).

**CORRECTION after user pushback (same day, note §7):** my (a)/(b) attributions were wrong.
`density_fb_state` runs HARD (one-hot full-state successor target, `measure_density_hard: true`
default) ⇒ object kernel never load-bearing; centered-z Density-FB .82 ⇒ goal-conditioning not
the separator; fb_oc .34 ⇒ object coords in B not the separator. ONE separator survives every
matched control: the ESTIMATOR — normalized categorical CE (or a BOUNDED target: first-passage
no-ortho .967 vs occupancy no-ortho floored, 2026-07-22 2x2) vs the unbounded occupancy ratio
under object DWELL. "Bottleneck" only in the sense "task variable dwells"; not control, not
corridor/coverage. Owed: centered-z eval on plain-FB cube ckpts; rldp_density + centered index;
loss ladder on cube-double (LSIF / bounded / first-passage / CE).
