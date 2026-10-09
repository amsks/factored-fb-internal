---
name: p2-multienv-decay-mechanism-20260730
description: "2026-07-30 WHY phase 2 costs success on cube-double/scene: visitation fixed point collapses to the hindsight dwell CONSTANT (q_abs 43.5 vs dwell-implied 43.0 on cd, 32.1 vs 31.0 on scene) because off-diag grounding at eps=0.4 is 28x sparser than cube-single; actor_q_center renormalizes the flat Q's noise to full strength -> fast actor drift in first 50-100k. Ledger §9.16; arms = density-matched eps (cd x2.4, sc x1.9), freeze_actor, hindsight testpoints"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9c41d5e3-1a15-4159-9d1e-89b191d6e43b
  modified: 2026-08-03T04:01:38.350Z
---

2026-07-30 (bridge track). User asked why multi-env phase 1 beats anchors but
phase 2 regresses (−0.19 cd / −0.14 sc), and whether HPs must change across
envs. Full writeup + tables: **`PAPER/rescuing-fb/PHASE2-LEDGER.md` §9.16**
(appended after §9.15c). Probe script kept in session scratchpad
(`p2_ranking_probe.py`); JSON alongside it.

**Mechanism (three findings, all measured):**
1. Damage is EARLY ACTOR DRIFT: step-1 eval = phase-1 parent exactly (warm
   start clean); decay happens 0→100k then plateaus ~0.35–0.40; cube-single
   never decays.
2. The visitation fixed point on sparse-region envs collapses to the hindsight
   dwell constant `(1−γ^dwell)/(1−γ)`: dwell ~56 steps on cs AND cd (37 sc) ⇒
   implied mass cd 43.0 / sc 31.0; measured p2 `q_abs_mean` cd **43.5** / sc
   **32.1** (match to 3%), all-positive, ~no batch variation. Cube-single sits
   at 23.4 ≪ its 43 ceiling with 11–18% signed variation. Cause: off-diag
   grounding rate at the training ε — cs **1.30%**, sc 0.29%, p3 0.20%, cd
   **0.046%** (28× sparser). The 28×-sparser region ends with the LARGEST Q =
   pure bootstrap misallocation taken to the constant-function limit. Phase 1
   is protected by the absorbing bound (M≤1, diag pinned at 1).
3. `actor_q_center` divides the q-term by sg[mean|q−mean q|], so a flat Q
   AMPLIFIES its residual noise to unit scale against bc_coeff=3 ⇒ drift.

**Gotchas learned:** `1−q_mean/q_abs_mean` in train.csv is a NEGATIVE-MASS
fraction, not a variation measure (equal to 0 whenever Q>0 everywhere).
p1↔p2 action-argmax agreement on random dataset actions is ~0.2–0.3 on ALL
envs incl. healthy cube-single — it does not discriminate (consistent with
§4/E5); the discriminator is the mass structure.

**STATUS 2026-07-31 04:40 — arms LAUNCHED (tmux `p2fix`, §9.16b pre-reg,
`scripts/rescuing_fb/launch_p2fix_arms.sh`, 12 cells). Partial collection:**
- **eps arm (`p2e_*`) DONE = NULL/negative**: cd 0.340/0.380 ≈ control 0.358,
  sc 0.288/0.284 < control 0.342; q_abs stays at the dwell constant (cd 42.0,
  sc 31.3). Density-matched ε does NOT rescue ⇒ §9.16's "ε is the
  non-transferring HP" claim is FALSIFIED as stated.
- **hindsight arm (`p2h_*`) = the WIN**: sc s1 final **0.904** (phase 1 0.484,
  control 0.342, anchor 0.281), monotone 0.52→0.90, no decay; cd s0/s1 0.58–0.61
  at 50–100k vs control's 0.368 by 100k. q_abs **59–60**, well ABOVE the dwell
  constant — a different mass regime, not the predicted sub-dwell signature.
  sc s0 + both cd still training (100–150k of 500k).
- **freeze arm (`p2f_*`) = sanity PASSES**: sc s1 0.504 ≈ phase 1, flat; q_abs
  28.6 ≈ dwell 31 under a FROZEN actor ⇒ the measure collapse is not
  actor-feedback-driven (§9.16 finding 2 survives; finding 3 is only the
  success-side amplifier).

**§9.16d WRITTEN 2026-08-03 — wave B (d128) COLLECTED, campaign FULLY CLOSED.**
All 4 stage-2 cells ran 500k. PRIMARY fails (cd .480/.260 ≈ control; sc
.312/.336 = control, same decay shape; d128 doesn't help phase 1 either).
SECONDARY largely confirmed: q_abs LEAVES the dwell constant (cd 27 vs 43, sc
22 vs 31) with signed variation returning ⇒ NULL clause did not fire, d50 rank
WAS the capacity constraint behind the mass pinning. **⇒ mass collapse and
success regression are DOUBLY dissociated** (hindsight fixes success without
the mass; d128 fixes the mass without success). Standing recipe: hindsight
test-points on cd/sc, batch on cube-single, d50 for success-axis work. Open:
WHY hindsight works; cube-single batch+hindsight mixing arm; zero-shot harness
still blocked on env_geometry.py.

**§9.16e+f WRITTEN 2026-08-03 — TWO MORE MECHANISMS REFUTED (both NULLs fired,
pre-registered).** e: goal-support mismatch DEAD — band (traj-future-goal)
action rankings in batch arms are as seed-reproducible as hindsight's (.48 vs
.42–.51; needed <.15). f: converged-Q exploitation DEAD — no sign flip under
cross referees (all advs ≈+0.01), and batch/hindsight policies are the SAME
distance from the certified dataset action (‖π−a0‖≈.27) despite .35-vs-.90
success. Post-hoc: EVERY static instrument on the dataset distribution is
arm-identical (certified-action rank .67–.69 all arms; band/far separation
present in batch too). ⇒ the failure expresses ONLY on the policy's own
ROLLOUT distribution and/or the 0→100k transient. Probes + raw grids:
session scratchpad `band_probe.py`/`exploit_probe.py` (f238489d…). Next
discriminators pre-listed in ledger §9.16f: (1) rollout-distribution probe
(re-run instruments on states each policy actually visits, split by phase),
(2) dense-ckpt transient (save_interval=10k first 100k), (3) q_center=False
interventional arm.

**§9.16g+h WRITTEN 2026-08-03 — rollout-distribution and gradient-field probes
ALSO NULL (six pre-registered NULLs total). ⇒ MECHANISM PINNED TO THE 0→100k
TRAINING TRANSIENT** (phase-1 absorbing → ball-visit objective migration while
the actor co-adapts; converged ckpts carry no readable scar). g strict reads
failed (no POST-acquisition collapse; pref flip absent) but found a replicated
POST-HOC marker: batch far-field Q = seed noise on APPROACH states (seed_agree
.05–.17 vs hind .21–.49, 4/4 pools) — a marker only, NOT causal
(actor_p_trajgoal=1.0, eval never consults Q). h: band ∂Q/∂a cross-seed cosine
arm-equal (.32–.39); only magnitude differs (batch ~10 vs hind ~6-7, q_center
normalizes it away). **DENSE-TRANSIENT WAVE LAUNCHED (tmux `p2dense`, GPUs 0–4,
§9.16i pre-registered):** 5 cells ×150k, save/eval every 10k (cd/sc ×
batch/hs + cs control), same seeds/parents as wave A; analysis instruments =
actor divergence ‖π_t−π_0‖ + ranking churn on the band grid + q_abs migration;
NULL ⇒ interventional `actor_freeze_steps` arm (needs new knob). Launcher:
scratchpad launch_p2dense.sh; probes: band/exploit/rollout/grad_probe.py.

**§9.16i RESULT + §9.16j P2 (2026-08-03, evening): BOTH observational stories
now dead; freeze does NOT fix it.** i: transient instruments do not separate
arms — actor jump ~0.1 by 10k in EVERY cell (healthy cs has the LARGEST), Q
churn INVERTED (cs 0.31 deepest churn, healthiest); NULL fired. j: sanity
passes (parent success through the 50k freeze) but P1 FAILS — after unfreeze
both envs decay to the control plateau within 50k ⇒ THE TRANSIENT IS NOT THE
CAUSE; the (d50, batch-column) critic's actor-training signal is damaging AT
its settled fixed point despite every static equivalence (e–h). Success is
the ONLY observable that separates arms across 7 probe families. §9.16k
(d128+freeze, tmux `p2frzd128`, RUNNING on GPUs 0–3) is now the CRITICAL
two-factor cell: holds ⇒ recipe = d128+freeze (both §9.16d and §9.16j were
single-factor failures); decays ⇒ batch columns dead for actor training here,
column MIXING is the last ρ-semantics option.

**§9.16i+j LAUNCHED 2026-08-03.** Dense-transient wave (tmux `p2dense`, GPUs
0–3, §9.16i): partial curves show the damage is a 0→10k SHOCK at the phase
switch hitting BOTH arms (hindsight cd dips DEEPER, .252 vs .384) — the arm
difference is RECOVERY, not dip depth (hindsight climbs back, batch stays
pinned; cs control flat .94–.99). q_abs migrates 2.7→43 over ~60k. R1 as
pre-registered will likely FAIL as written — the finding is recovery-vs-trap.
**NEW KNOB SHIPPED (TDD green, UNCOMMITTED): `actor_freeze_steps=N`** in
gciql_flow.py — actor grads gated to zero pre-adam for first N steps, critic
trains from step 0, default-0 bit-identical, refuses freeze_actor/
freeze_critic_ckpt; 3 tests in test_gciql_phase2_occupancy.py, 20-test suite
green. **§9.16j arm RUNNING (tmux `p2frz`, GPUs 4–7): p2frz_{cd,sc}_{s0,s1} =
batch columns + actor_freeze_steps=50000, 500k**, same parents as wave A;
P1 = no decay after unfreeze ⇒ batch+delayed-actor becomes the recipe with
ρ-referenced semantics preserved; P2 = decays after 50k ⇒ column-mixing arm
next. §9.16i ckpt analysis still OWED when dense wave lands.

**CLOSED 2026-08-02 — §9.16c WRITTEN (ledger, wave A complete).** Hindsight
replicates at 3 seeds: scene .916/.904/.960 (mean .927 vs phase-1 .484),
cd .500/.544/.548 (.531 vs .549), 0→100k decay REVERSED on both. But the
**pre-registered mechanism is REFUTED**: §9.16b predicted the working arm drives
q_abs BELOW the dwell constant (cd <35, sc <27) with neg-mass ABOVE 5%; observed
q_abs rises far ABOVE it everywhere (cd →~55, sc →~59, p3 →~100, cs →~74) and
neg-mass COLLAPSES to ≤0.3%. **The arm works; §9.16's collapse story does not
explain why — do not cite it as the reason.** Scale-out, both negative:
puzzle-3x3 NULL (.204/.188 off a .200 phase-1 floor ⇒ hindsight PROTECTS
structure, never ADDS it), and the cube-single control PAYS ~0.17–0.24
(.904/.980 → .784/.744) ⇒ never make hindsight an unconditional default.
**Caution for [[validmeasure-campaign-20260729]]:** puzzle-3x3 reaches neg-frac
0.0% at q_abs ≈100 while success stays degenerate at .200 ⇒ low neg_frac + large
Q is reachable by a fully degenerate solution and is NOT sufficient for a valid
measure; that gate needs a companion.

**How to apply (arms, none launched — pre-register in ledger before running):**
- The HP that fails to transfer is `measure_hit_spec` ε (density), NOT
  lr/batch. Density-matched phase-2 ε to hit ~1.3% grounding: cube-double
  ×2.4 (≈0.95/cube), scene ×1.9. Pre-registered read: cd q_abs drops off 43,
  signed variation returns, 0→100k decay shrinks.
- `freeze_actor=true` (E2, cs 0.972) erases the regression by construction and
  is the right multi-env default given §4 (success never certified the measure).
- `measure_testpoint_dist='hindsight'` = sparsity-independent grounding
  (incompatible with nhat_head/multiscale asserts).
- §9.15b mass anchor would fix the inflation but in-batch N̂ clamps at 2/B at
  0.05% hit rate — port to jax only with one of the above.
- p2 is not iso-budget (p1+500k); score multi-env p2 on measure axes once
  `env_geometry.py` unblocks the zero-shot harness.

Related: [[phase2-ledger-20260728]], [[spec08-e8-fourenv-launched-20260725]],
[[spec08-e7-multienv-launched-20260724]], [[validmeasure-campaign-20260729]].
