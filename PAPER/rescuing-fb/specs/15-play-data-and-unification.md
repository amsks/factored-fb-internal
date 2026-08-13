# Spec 15 — Play data across datasets, and a less hand-crafted successor-measure algorithm

Status: DRAFT (survey + empirics sections pending agent reports, 2026-08-05).
Question owner: Aditya. Companion datasets: OGBench cube/scene play;
TommyShen/Factor-FB-Offline (play-noise, play-resample); tl-play-data code.

## 0. The question

Our current method is a two-stage pipeline: (1) TD on a first-passage
(reaching) objective, (2) warm-started TD on a visitation objective. This
works but reads as hand-crafted. Is there a single principle under it, and
does that principle transfer beyond OGBench cube-single?

## 1. The claim to test: play data is a semi-Markov jump process in object space

**Precise statement of the phenomenon (sharpened per external review
2026-08-05):** reset-light manipulation play concentrates occupancy at
*stable object configurations*, because the robot keeps acting — retracting,
switching subtasks, manipulating another object — while the goal-relevant
object coordinate c(s) stays unchanged. The full state s=(r,c) rarely
repeats (r keeps moving); what repeats is occupancy of a region in the
object-centric projection c(s). Terminology: call this **persistent
occupancy / post-arrival dwell**, NOT "revisitation" — genuine
leave-and-return re-entry is a separate, smaller channel (§4) and the two
must not be conflated.

Look at what the *object coordinates* (not the arm) do in a play dataset:

- Long stretches where the object position is CONSTANT (it sits where it was
  placed) — call these **holds**.
- Short stretches where it moves under the gripper — **transports**.
- Occasional physics noise (slips, nudges) — small perturbations of holds.

So the object trajectory is piecewise-constant with jumps: a semi-Markov
process. Two consequences:

1. Any region around an object position accumulates occupancy mass almost
   entirely during holds. Mass(region) ≈ P(a transport ends inside region) ×
   E[discounted hold duration]. This is renewal theory: occupancy =
   hitting × dwell.
2. The controllable part of the problem is ONLY "which transport happens"
   (arrival). The dwell factor is a property of the DATA-COLLECTION process
   (how long the scripted policy leaves objects alone), not of the agent's
   actions.

**Prediction across datasets:** any play-style dataset (OGBench resampling
scripted policy, play-noise, CALVIN-style human play) will show (a) long
in-region runs whose length distribution is set by the collection process,
(b) occupancy mass ≈ arrival-probability × dwell-constant to good accuracy,
(c) dwell approximately action-independent given arrival. Datasets differ in
the dwell CONSTANT, not in the factorized STRUCTURE. [Empirics: §4 pending.]

## 2. Why this makes our two-phase design principled rather than hand-crafted

Phase 1 (first-passage TD) learns exactly the arrival factor H(s,a,g) =
discounted probability of ever entering the region. It is a bounded [0,1]
quantity, its TD operator is a contraction with an absorbing anchor
(diagonal pinned at 1 inside the region), and — critically — it is the ONLY
factor the actor needs, because dwell is action-independent.

Phase 2 (visitation TD) tries to learn H × dwell *by bootstrapping*, i.e. it
re-derives by TD a product whose second factor needs no TD at all. That is
where every pathology of the last month came from: the dwell constant is
~40× the arrival signal, so on sparse regions the TD fixed point collapses
to the constant (§9.16), the mass anchor destroys ordering (§9.15c), and the
actor gets a flat critic.

So the "hand-crafted" two-phase split is actually the renewal factorization —
we just implemented the second factor with the wrong estimator (TD) instead
of the right one (counting).

## 3. The candidate elegant algorithm (working name: renewal successor measure, RSM)

Train ONCE, three components, no phases:

1. **Arrival critic** H(s,a,g) = φ(s,a)ᵀψ(g), trained by first-passage TD
   (identical to our phase 1 / to a goal-conditioned 0-1-reward value
   function). Bilinear so the reward readout below stays a dot product.
   Goal-conditioned, NOT z-conditioned — per the 2026-08-05 wave T + noflow
   result, z-conditioning of the critic/actor is what floors multi-object;
   goals + hindsight-grounded training points are what works.
2. **Dwell head** D(g) = expected discounted in-region mass GIVEN entry,
   trained by SUPERVISED regression on the dataset: for each entry event
   into a region around g, the target is Σ_t γ^t 1[in region] over the
   observed run. No bootstrapping, no sparsity problem (every placement
   event is a training example), and it can be conditioned on whatever it
   empirically depends on (region size, object identity, local clutter).
3. **Actor** π(a|s,g), BC-regularized, trained against H only (the factor it
   can influence), goals from trajectory futures (hindsight) — this is the
   arm that already works everywhere including cube-double/scene.

**Zero-shot readout** for a reward r over dataset states:
z_r = E_ρ[ψ(g) · D(g) · r(g)]  ⇒  Q_r(s,a) = φ(s,a)ᵀ z_r.
The task vector exists ONLY at inference. Test-time policy: decode z_r to a
goal (nearest-neighbour over dataset ψ, as in the jax stack) or GPI over a
candidate goal set. The 1/N̂ scale factor from the §9 audit is exactly the
normalization D absorbs.

Properties:
- Every trained quantity is grounded by events that actually occur in the
  data (transitions for H's diagonal, placement events for D). Nothing
  bootstraps an unbounded quantity. Phase 2, the mass anchor, the nonneg
  hinge, actor_freeze — all disappear.
- Degrades gracefully: on a dataset with no dwell structure (e.g. constant
  motion), D → small constant and RSM reduces to first-passage GCRL.
- Honest failure mode to pre-register: rewards that depend on STAYING
  (e.g. "hold the cube in the air") need action-dependent dwell; RSM v1
  assumes D independent of the arriving action. Test: compare D fitted with
  vs without action features (§5 E3).

## 3b. REFRAME (user direction, 2026-08-05 late): the successor measure stays
the learned object; dwell structure is the APPROXIMATION SCHEME

Objection to §3 as written: H is a goal-conditioned value, so the method
reads as abandoning the successor measure. Corrected goal: learn
M(s,a,g) = expected discounted time in ball(g) — the successor measure
itself — but replace the standard one-step estimator with one that
exploits the measured dwell structure. Two schemes:

**Scheme A — factored estimator (≡ §3, restated).** Parameterize the
measure as M(s,a,g) = φ(s,a)ᵀψ(g)·D(g). The effective backward map is
ψ̃(g) = ψ(g)D(g); the FB API (z_r = E[ψ̃(g)r(g)], Q = φᵀz_r) is unchanged.
The renewal factorization is an inductive bias INSIDE the measure
estimator, with the ill-conditioned factor fit by regression.

**Scheme B — dwell-compressed (semi-Markov) TD; the measure-native
answer.** ONE network M(s,a,g) = φ(s,a)ᵀψ(g), trained by TD whose
operator uses the dwell structure:
1. Preprocess: segment each trajectory by object-motion events (object
   starts/stops moving; trivial in state space). During a rest segment
   the in-ball indicator is constant.
2. TD update from (s_t,a_t) toward ball(g): let τ = the next event
   boundary after t. Target =
   Σ_{k=t}^{τ-1} γ^{k−t}·1[c(s_k)∈ball(g)]  +  γ^{τ−t}·M̄(s_τ,a_τ,g).
   The first term is MEASURED from the data (during a dwell it is the
   analytic run mass, during transport it is ~0-2 grazing steps); only
   the second term bootstraps.
3. Properties: (a) the dwell constant enters as an observed immediate
   term — TD never has to reconstruct it by bootstrap iteration, which
   is precisely where §9.16's collapse happened; (b) bootstrap error is
   suppressed by γ^Δ exactly where segments are long; (c) revisits are
   handled natively by the bootstrap chain (no D-definition question);
   (d) one-step TD is the degenerate case τ=t+1; phase-1 first-passage
   is the other limit (τ = first entry, tail dropped). This is n-step
   TD with n set by the data's event structure — "TD in event time."
4. Readout identical to FB: z_r = E_ρ[ψ(g)r(g)] (no separate D), and
   the actor can train on M directly or on its first-passage part.

Preference order: Scheme B is the answer to "approximate the successor
measure USING the dwell structure"; Scheme A is its cheap ablation (and
E4's eval-only version). Gate to add: on cube-double phase-2 conditions,
Scheme B's q_abs must NOT pin to the dwell constant (the §9.16
signature), and signed cross-state variation must survive.

**3c. CONSTRAINT (user, 2026-08-06): NO data preprocessing.** Event
segmentation rejected — same category of hack as chopping 1001-step play
into per-pick-place expert demos for BC: it edits the data to fit the
method. The structure must live in the ESTIMATOR (architecture +
objective + return estimator over raw trajectory windows), data untouched.
Scheme B's event-time operator is replaced by:
- **Structured measure head (the core):** M(s,a,g) = D(g)·H(s,a,g), one
  network pair trained END-TO-END by measure TD on raw transitions.
  D = exp(g-only head) ≥ 0 (per-goal magnitude); H = σ-bounded bilinear
  φ(s,a)ᵀψ(g) ∈ [0,1] (state-action resolution). The dueling-style
  split gives the action-independent dwell magnitude a dedicated,
  low-capacity home so the TD fixed point cannot spend φᵀψ on it; the
  §9.16 constant-collapse lands in D(g) BY CONSTRUCTION and leaves H's
  ranking intact. (Identifiability: H≤1 pins D(g) = sup over (s,a).)
- **Per-transition first-passage anchor on H (aux loss):** absorb-on-entry
  TD (1[c(s')∈ball(g)] computable per transition — no segmentation) ties
  H to arrival semantics; weight small, default-off ablation arm.
- **Return estimator matched to the slow timescale:** k-step / λ-returns
  over raw trajectory windows (the sampler already reads windows for
  hindsight goals — not preprocessing) with k ~ dwell scale, so most
  dwell mass sits in the OBSERVED part of the target and bootstrap error
  is γ^k-suppressed. Fixed k, no event detection.
Readout: ψ̃(g)=ψ(g)D(g)H-free... (z_r = E_ρ[ψ(g)D(g)·? r(g)] — resolve
the exact readout algebra when speccing the run; the measure evaluated
at (s,a,g) is M=D·H regardless).

## 4. Empirical validation (E1+E2 COLLECTED 2026-08-05)

**E1/E2 result: the renewal factorization holds on every dataset tested, and
dwell dominance GROWS with clutter.** 50 centers/dataset, eps = 4 cm
physical (= our 0.4 obs-unit convention), γ=0.99. Script + JSON: session
scratchpad `dwell/dwell_stats.{py,json}`.

| dataset | P(arrive)/ep | run len mean/med/p90 | longest-run frac | revisit prob | mass model fit (log-Pearson) | actual/model ratio |
|---|---|---|---|---|---|---|
| cube-single-play | 0.38 | 22 / 4 / 57 | 0.86 | 0.36 | 0.998 | 0.98 |
| cube-double-play | 0.21 | 60 / 5 / 209 | 0.91 | 0.28 | 0.998 | 0.89 |
| cube-triple-play | 0.14 | 120 / 53 / 325 | 0.95 | 0.22 | 0.999 | 0.82 |
| cube-quadruple-play | 0.10 | 140 / 49 / 413 | 0.95 | 0.22 | 0.993 | 0.80 |
| scene-play (cube dims) | 0.46 | 86 / 57 / 210 | 0.87 | 0.43 | 0.997 | 0.93 |
| ManiSkill pick-cube (expert) | 0.08 | 4.7 / 2 / 18 | 0.95 | 0.15 | 0.991 | 1.19 |
| Metaworld bin-picking (expert) | 0.90 | 66 / 74 / 77 | 0.9993 | 0.04 | 0.9995 | 1.0003 |

Mass model per center: P(arrive) × E[γ^t_first | arrive] × (1−γ^L̄)/(1−γ).

**E2 UPDATE (2026-08-05, late): the INDEPENDENT play-resample data lands
the same result.** Tommy's v2 Metaworld play-resample (collected TODAY,
scripted Sawyer oracle + goal re-draw on success-after-dwell or timer,
500-step trajectories, local `~/git/datasets/factoredfb-metaworld-play`;
battery `scripts/rescuing_fb/dwell_mw_play.py`, JSON
`results/dwell_mw_play.json`; ε = each task's own success radius):

| regime/task | P(arrive) | run mean/med/p90 | first-sojourn mass frac | revisit prob | fit (log-Pearson) | ratio |
|---|---|---|---|---|---|---|
| play-resample/mw-push | 0.55 | 38/24/78 | 0.74 | 0.51 | 0.986 | 1.03 |
| play-resample/mw-pick-place | 0.71 | 27/13/49 | 0.74 | 0.64 | 0.913 | 1.24 |
| hard/mw-push | 0.41 | 51/18/148 | 0.80 | 0.45 | 0.994 | 1.05 |
| hard/mw-pick-place | 0.46 | 55/18/167 | 0.80 | 0.46 | 0.988 | 1.01 |

Readings: (a) the arrival×dwell model fits an independently collected,
different-embodiment, different-collector play dataset at 0.91–0.99 —
external transfer of the phenomenon is now MEASURED, not predicted;
(b) re-entry is a BIGGER channel here (revisit 0.45–0.64; first sojourn
carries 74–80% of mass vs OGBench's 86–95%) ⇒ D must be defined as
per-EPISODE in-region mass given arrival (revisits folded in), not
first-sojourn-only — exactly the review's M_dwell/M_reentry split, now
measured; (c) the hard regime lengthens runs (noise knocks the object
around less than it interrupts the oracle) without breaking the fit;
(d) caveat: Metaworld wrapper steps ≠ OGBench steps — seconds-normalized
rerun still owed before the paper table. Also note Tommy's v2 changelog
independently measured cube-single dwell (median 46 of ~87 steps) —
consistent with our §4 numbers, a second pair of eyes on the phenomenon.

Readings:
- Cross-region mass variation is explained ≥0.99 by arrival × dwell on ALL
  datasets, including non-play expert data (the degenerate one-approach
  limit). The factorization is a property of manipulation data broadly, not
  of OGBench's collector.
- Sojourns lengthen 22→60→120→140 steps single→quadruple while P(arrive)
  falls 0.38→0.10: exactly the regime where visitation TD drowns (constant
  ×4 larger, signal ×4 rarer). Matches §6's mechanism (per-cube re-pick
  probability drops with object count).
- Revisits are a real but secondary channel (0.2–0.4 per exit; scene 0.43),
  so RSM's D(g) should be "discounted mass per ENTRY" with revisits folded
  into H's renewal chain or ignored at ~10% error.
- The "~56-step dwell" figure from §9.16 is the p90/resting mode; run
  lengths are bimodal (short grazing passes + long rests). D(g) as a
  regression naturally absorbs this; a single global constant would not —
  another reason phase-2's implicit constant is the wrong estimator.
- ⚠ Right-censoring at episode ends biases the model ratio to 0.8–0.9 on
  triple/quad; any fitted D should censor-correct or cap at episode end.
- ⚠ **Cross-dataset comparability (external review catch): step counts are
  not comparable across control frequencies.** Metaworld logs at 40 Hz,
  ManiSkill at 10 Hz, OGBench at 20 Hz; γ=0.99 has a ~69-step half-life =
  1.7 s / 6.9 s / 3.5 s respectively. Before any cross-dataset table goes
  in the paper: report dwell in SECONDS alongside steps, and either use
  γ(Δt)=exp(−Δt/τ) with one physical τ or downsample to a common rate.
  The §4 table's cross-family rows are currently step-based — fix on rerun.
- Additional statistics to add to `dwell_stats.py` (per external review,
  all cheap): (a) explicit M_dwell vs M_reentry mass split (first sojourn
  vs post-exit returns) + re-entry count N_reentry, so "dwell dominates"
  is a measured mass fraction, not inferred from run lengths; (b)
  robot–object decoupling P(‖Δc‖<δ ∧ ‖Δr‖>η) — the cleanest single
  statistic for "robot keeps moving while the object rests"; (c) number of
  simultaneously active objects per step (prediction: almost always 0 or
  1); (d) stationarity at several strides k, not just consecutive frames.
- Note on ε: OGBench's own success criterion is object-to-target distance
  ≤ 0.04 m (`cube_env.py:709-715`) — our ε=4 cm equals the benchmark's
  task tolerance, so the region scale is externally pinned, not tuned.
- E3 (action-independence of dwell): regress observed dwell mass on entry
  (s,a) features; R² gain over region-only features should be small on all
  play datasets. NOT LAUNCHED.
- E4 (RSM on cube-single): components already exist — phase-1 ckpt = H,
  dwell head = new ~100-line supervised script, readout = L2 harness with
  weight D(g). Compare zero-shot vs the two-phase L2 numbers. NOT LAUNCHED.
- E5 (RSM on cube-double/scene): H = the jax hindsight arm; the test is
  whether D-weighted readout beats NN-decode-only zero-shot (spec14 §7
  anchor: scene dirac 0.872, region-mean 0.45). NOT LAUNCHED.
- E6 (external validity — CALVIN, per review): rerun the dwell battery on
  CALVIN's state arrays only (24-D `scene_obs`: 3 block 6-DoF poses,
  drawer, slider, button, switch, lights; robot state separate) — human
  teleop collector + different scene, exact object states, no policy
  training needed. Strongest independent test; also fixes the current
  weakness that all play-style rows in §4 come from one collector family.
  Optional secondary: Minari Franka-Kitchen mixed (articulated mechanisms,
  621 eps / 156k steps). NOT LAUNCHED.
- E7 (causal test, per review — the strongest paper experiment): intervene
  on dwell multiplicity in cube-single and retrain FB vs region-TD on:
  (a) first-hit-truncated data (drop transitions after first arrival at
  the collector's current target), (b) stationary-run deduplicated (keep
  1-in-k repeated object states), (c) inverse-run-length-weighted
  sampling, (d) temporally downsampled. Pre-registered read: shrinking
  dwell multiplicity stabilizes the FB critic (ms_abs_mean stops
  diverging), improves its action ranking, and narrows the FB-vs-region
  gap ⇒ dwell is the CAUSE of the linear-diag divergence, not a
  correlate. Directly upgrades the §9.15/torch-measure story from
  correlation to intervention. NOT LAUNCHED — needs a dataset-transform
  utility + 4-6 training cells, ~1 GPU-day each.

## 5. Survey: what play data is and why people use it (DONE — full annotated version: `PAPER/rescuing-fb/play-data-survey.md`)

- **Definition (Lynch et al. 2019, the origin):** "an unbounded sequence of
  states and actions corresponding to voluntary, repeated, non-stereotyped
  object interaction" — teleoperated, undirected, non-episodic, label-free,
  multimodal, suboptimal. Benchmarks in this tradition: Playground/3D
  Playroom (Lynch), relay-kitchen (Gupta 2019), CALVIN (24h VR play),
  TACO play (9h real Franka).
- **OGBench's place:** its paper explicitly says "'play'-style datasets
  (Lynch et al., 2019) collected by non-Markovian expert policies with
  temporally correlated noise" — a scripted, reproducible PROXY for human
  play that keeps the surface statistics (random atomic manipulations,
  temporal correlation, no per-task episodes). Widely adopted 2025-26
  (Dual Goal Representations, Q-Chunking line, Horizon Reduction's 1B-step
  play datasets), so "does the phenomenon hold on play data" is a question
  the community has standing to care about.
- **Why play data (the field's arguments):** cheap (no staging/labels/
  resets); high coverage (~4× interaction space per collection hour, Lynch);
  task-agnostic (no task list at collection time); natively hindsight-
  relabelable; multimodal; suboptimality doubles as robustness fuel.
  Ghugare et al. (ICLR 2024) add the TD-side argument: play streams pass
  through goals without heading to them, so extracting behavior REQUIRES
  stitching/dynamic programming — exactly the regime our methods target.
- **"Playworld":** two referents — the Lynch-line sim is actually called
  Playground/3D Playroom; "PlayWorld" proper is Yin et al. 2026
  (arXiv:2603.09030), autonomous robot self-play for video world models.
- **THE GAP (for our paper):** the published vocabulary for play-data
  structure stops at coverage (TQ×SACo), stitching demand, and noise
  correlation. **No published work reports dwell/revisit/occupancy
  statistics of play datasets or ties them to GCRL performance.** §4's
  measurements + the renewal reading are unclaimed territory and directly
  justify studying manipulation-from-play with occupancy-based methods.

## 6. How OGBench actually collects the play data (verified at code level, 2026-08-05)

Source: `third_party/ogbench/data_gen_scripts/generate_manipspace.py` +
oracle classes (byte-identical to the installed v1.2.1 package).

- **Scripted expert, two flavors.** `dataset_type='play'` = open-loop
  PlanOracle: precomputes keyframe plans (pick→lift→place→retract→random
  arm pose), interpolates, adds *temporally smoothed* Gaussian noise to the
  plan poses (`scale [0.05,0.05,0.05,0.3,1.0]×0.1`, `gaussian_filter1d
  σ=0.5`), tracks with a P-controller. `dataset_type='noisy'` = closed-loop
  Markov state machine with i.i.d. per-step Gaussian action noise
  (`xi~U(0,0.1)` per episode) + 10–20% fully random actions. So OGBench
  "play" noise is on the PLAN (smooth), not on the actions.
- **No idling, immediate resampling.** The instant a subtask's plan ends, a
  new random subtask is sampled (uniform over "top" blocks; scene mixes in
  button/drawer/window). One cube pick-and-place ≈ 85–91 control steps;
  episodes are 1001 steps ⇒ ~11 subtasks/episode; full randomized reset
  between episodes.
- **Dwell mechanism.** Between subtasks that target it, a cube is perfectly
  static (the arm flies off to a random pose). Expected dwell:
  cube-single ≈ 1 subtask (the only cube is re-picked every time);
  cube-double ≈ Geometric(1/2) subtasks ≈ ~176 steps; scene longer still
  (cube tasks are only ~1/3 of subtasks, and cubes in the drawer are frozen
  until re-exposed). **Dwell grows with env complexity — largest exactly
  where phase-2 visitation TD collapses.** The dwell "constant" is also
  clearly region-dependent (which object, drawer state) ⇒ D(g) must be
  learned per region, which the supervised head does for free.

## 6b-RESOLVED (2026-08-05, evening): `tl-play-data` branch fetched — checklist scored from code

Branch `origin/tl-play-data` (8 commits, post-07-22). Collectors:
`data_collection/collect_play.py`, `collect_play_resample.py` (+`_ms.py`);
doc `data_collection/PLAY_DATA.md`. Data NOT yet downloadable: the doc's
HF id `TommyShen/factored-fb-play` 404s even authenticated — files live at
`/raid/tongle/rl/fb-data/collected_play{,_resample}/` on Tommy's box.

**What the two datasets actually are:**
- **play ("play-noise"), 9 tasks / 1.35M transitions:** trained TD-MPC2
  rolled out exploratorily — stochastic prior (`eval_mode=False`),
  planning off, epsilon-random + Gaussian action noise, fixed-horizon
  episodes run to the end (post-success wandering kept), ONE task goal per
  episode, no success filtering. Diverse STATES, not diverse goals.
- **play-resample, 6 tasks / 896k transitions / 20k chained goals:** one
  LONG continuous episode per env (Box2D 1000 steps, ManiSkill 400):
  reach goal → hold ~8 steps → resample a valid new goal (min-distance,
  wall-checked; ManiSkill moves the SAPIEN goal actor) → continue, NO
  reset; terminal=0 throughout; a goal-switch marker array is recorded; a
  "redirection gate" excludes checkpoints that ignore the obs goal.

**Checklist verdict (play-resample vs OGBench manipulation):**
1. Long horizon, world persists, no reset between targets — **PASS**.
2. New target sampled at completion — **PASS** (trigger = success+hold or
   stuck-timeout; OGBench resamples at plan END regardless of success, so
   failed attempts end differently — minor).
3. Temporally coherent per-target execution — **PARTIAL**: controller is a
   reactive Markovian learned policy, not a precomputed plan. On the
   §6b controller axis this is a goal-chaining cousin of OGBench-NOISY
   rather than OGBench-play. Still goal-directed behavior, not noise.
4. Noise — different: policy stochasticity / per-action, vs OGBench-play's
   temporally smoothed plan noise.
5. No success filtering — **PASS**.

**Bottom line: play-resample reproduces the structure that matters for our
argument** (persistent world, within-trajectory goal diversity, resample-
on-completion, unfiltered) **with a reactive learned controller in place of
OGBench's scripted plan oracle.** Bonus: it records goal-switch
boundaries — strictly better than OGBench's release, per §6b schema
feedback. play-noise is the control condition (perturbation without goal
diversity), exactly the review's framing.

**Pre-registered dwell prediction for when the data lands:** the renewal
factorization should HOLD on play-resample but with a SMALL dwell factor —
hold≈8 steps + approach time, single object re-manipulated immediately
(no multi-object rest amplification), and the Box2D tasks put c(s) = the
controlled body itself (the review's "transfer is weaker" boundary case).
That makes play-resample a dose-response test of the mechanism: FB's
dwell-driven pathology should be mild there and grow toward OGBench
multi-object. Battery: `scripts/rescuing_fb/dwell_stats.py`; object dims:
ms-pick-cube obs[29:32] (verified), ms-push-cube TBD, Box2D = body xy.

## 6b. TommyShen datasets vs OGBench collection (INVESTIGATED 2026-08-05 — premise correction, superseded by 6b-RESOLVED above)

**As of 2026-08-05, neither HF repo contains any play-noise or play-resample
data.** Verified via the full 282-file listing, all 7 dataset-repo commits
(last 2026-07-22), and HF-wide search.

What they DO contain:
- `datasets/TommyShen/Factor-FB-Offline` (~1.6 GB): **expert TD-MPC2
  demonstrations** — 18 tasks × 1000 episodes over Box2D (8 tasks, 200
  steps), ManiSkill (5 tasks, 25 steps), Metaworld (5 tasks, 100 steps) in a
  unified `[N,T,·]` npz schema, + Metaworld MT50 (5M transitions), + 228 BC
  "teacher" checkpoints (flow-matching `dp` and MLP `l2bc` heads; the clean
  set includes a cube-single teacher trained on OGBench cube-single-play).
- `TommyShen/Factor-FB-Offline-tdmpc2`: the 8 Box2D TD-MPC2 collector
  checkpoints.
- The accessible collector (`origin/tl-jax-unify:data_collection/
  collect_expert.py`): trained TD-MPC2 planner, one fixed task per episode,
  fixed horizon, noise = epsilon-UNIFORM random actions only (eps ∈ {0,
  0.1, 1.0}), Box2D success-filtered. No subtask resampling anywhere.
- `tl-play-data` code: **unreachable** — presumably a post-07-22 branch of
  the private LUH-AI repo; this box's GitHub credentials are dead. Every
  claim about how the play variants are collected is currently inference.

**Answer to "is play-resample close to OGBench?": cannot be verified yet —
no data and no code are published.** What can be said:
- The described play-noise ("Gaussian noise on policy execution") matches
  OGBench's **`noisy`** type (Markov oracle, per-episode xi~U(0,0.1),
  per-step Gaussian ×[1,1,1,3,10], +10% random actions), NOT OGBench "play".
  OGBench "play"'s noise is temporally smoothed noise on the PLAN.
- ⚠ **Correction (external review, consistent with §6): subtask resampling
  does NOT distinguish OGBench play from OGBench noisy — BOTH resample via
  `set_new_target` on completion, with no reset.** The distinguishing axes
  are controller class (non-Markovian precomputed plan vs Markovian
  reactive state machine) and noise structure (temporally smoothed
  plan-noise vs i.i.d. Gaussian + random actions). Consequence: a
  play-resample built on a REACTIVE controller (e.g. the published dp/l2bc
  teacher heads — the likeliest rollout policies given those actor-only
  checkpoints were uploaded) is closer to a low-noise OGBench-NOISY than
  to OGBench-play; matching "play" specifically needs temporally coherent
  per-subtask execution (plan / option / action-chunk with internal
  phase). If it instead reuses the vendored `data_gen_scripts/` plan
  oracle, it is genuinely OGBench-play.
- Checklist to close this when the code/data appear: (1) long horizon, no
  reset between targets (world state persists); (2) new target sampled on
  completion of the previous one; (3) temporally coherent execution per
  target (else it's noisy-style, see above); (4) noise = plan-smoothed vs
  per-action Gaussian vs eps-uniform; (5) NO success filtering (play's
  value includes failures/retries — the Box2D success-only split is the
  anti-pattern). Then rerun §4's dwell battery on both variants
  (`scripts/rescuing_fb/dwell_stats.py`).
- **Reframe worth keeping (external review): the two variants are an
  experiment, not just data.** play-noise tests whether local action
  perturbation alone reproduces play structure; play-resample tests
  whether persistent task-switching does. Pre-registered expectation:
  resampling, not noise, creates the post-arrival occupancy structure.
  Clean version if affordable: factorial over goal-switching × reset ×
  noise + a first-hit-truncated arm (6 cells).
- **Schema feedback for Tommy:** (a) populate `achieved_goal` with true
  object state c(s) — currently an all-zeros placeholder; (b) store the
  collector's current goal SEPARATELY from the observation — Metaworld/
  ManiSkill obs embed the goal, so target resampling makes obs jump with
  no physical change, poisoning any stationarity/occupancy statistic
  computed on full obs; (c) record subtask boundaries and an
  `environment_reset` flag — OGBench's own release drops target-switch
  boundaries, so keeping them makes these datasets MORE analyzable than
  OGBench.

## 7. Alternatives considered in the brainstorm (kept for the record)

- **Event-time / semi-Markov re-timing**: define the MDP over "object
  configuration change" events; in event time dwell vanishes and occupancy =
  chained first-passages. Cleaner theory, but requires an event detector and
  changes the discount semantics against every baseline. Deferred.
- **γ-model / geometric-horizon generative model** of future object state:
  handles dwell implicitly but gives densities, not set masses, and brings
  generative-model training cost. Deferred.
- **Distributional occupancy (learn the run-length distribution per region
  by TD)**: strictly more information than D(g), same sparsity problem as
  phase 2. Rejected.
- **Keep z-conditioned FB and fix it** (column mixing, mass anchors):
  the entire 07/28–08/05 record argues against; four+ pre-registered
  mechanisms refuted, wave T/noflow closed it. Rejected for the method
  artifact.

## 8. Bottom line

1. **Setting validated.** Play data is a 2019-to-present tradition (Lynch →
   CALVIN/TACO → OGBench → PlayWorld) with standard published
   justifications (coverage, cheapness, hindsight-native, stitching-
   demanding); OGBench's cube/scene play datasets are explicitly positioned
   in that tradition and widely adopted 2025-26. Citable chain in §5 +
   `play-data-survey.md`.
2. **The phenomenon is real, transferable, and unclaimed.** Occupancy mass
   = arrival × dwell at r≥0.99 on 7 datasets across 3 embodiment families;
   dwell dominance strengthens exactly where our methods fail; nobody has
   published these statistics for play data.
3. **The elegant algorithm falls out.** RSM (§3): first-passage
   goal-conditioned critic (already works everywhere) + supervised dwell
   head + renewal readout. Two-phase TD → one-phase TD + one regression.
   The task vector exists only at inference.
4. **Next experiments in order:** E4 (RSM readout on existing cube-single
   ckpts — eval-only, no training), E3 (dwell action-independence — pure
   data analysis), E5 (RSM on the jax hindsight cd/scene arms vs spec14
   zero-shot anchors). All three run on artifacts that already exist.
5. **TommyShen play variants:** not yet published (data or code); §6b has
   the verification checklist for when they land.
